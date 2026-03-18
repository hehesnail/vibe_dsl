/*!
 * \file target/blackhole_module.cc
 * \brief Blackhole module implementation with TT-Metal integration
 */

#include "blackhole_module.h"

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"

// TT-Metal headers
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>

namespace tvm {
namespace runtime {

using namespace tt::tt_metal;

// Forward declarations
class BlackholeModuleNode;

/*!
 * \brief Wrapper for Blackhole kernel execution
 */
class BlackholeWrappedFunc {
 public:
  void Init(BlackholeModuleNode* m, ObjectPtr<Object> sptr,
            const std::string& func_name,
            const ExecutableSpec& info) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    info_ = info;
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const;

 private:
  BlackholeModuleNode* m_;
  ObjectPtr<Object> sptr_;
  std::string func_name_;
  ExecutableSpec info_;
};

/*!
 * \brief Blackhole module node implementation with TT-Metal
 */
class BlackholeModuleNode : public ffi::ModuleObj {
 public:
  BlackholeModuleNode(
      std::unordered_map<std::string, ExecutableSpec> fmap,
      std::string kernel_dir)
      : fmap_(std::move(fmap)),
        kernel_dir_(std::move(kernel_dir)),
        mesh_device_(nullptr),
        device_initialized_(false) {}

  ~BlackholeModuleNode() {
    // Clean up TT-Metal resources
    if (mesh_device_) {
      delete static_cast<std::shared_ptr<distributed::MeshDevice>*>(mesh_device_);
    }
  }

  const char* kind() const final { return "blackhole"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);

    auto it = fmap_.find(name);
    if (it == fmap_.end()) {
      return ffi::Function();
    }

    const ExecutableSpec& info = it->second;
    BlackholeWrappedFunc f;
    f.Init(this, sptr_to_self, name, info);

    std::vector<FunctionInfo::ArgExtraTags> arg_extra_tags;
    return PackFuncVoidAddr(f, info.tvm_arg_types, arg_extra_tags);
  }

  void WriteToFile(const ffi::String& file_name,
                   const ffi::String& format) const final {
    LOG(WARNING) << "BlackholeModule WriteToFile not yet implemented";
  }

  ffi::Bytes SaveToBytes() const final {
    LOG(WARNING) << "BlackholeModule SaveToBytes not yet implemented";
    return ffi::Bytes("");
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    auto it = fmap_.find("default");
    if (it != fmap_.end()) {
      const auto& spec = it->second;
      return ffi::String(spec.kernels.empty() ? std::string() : spec.kernels.front().source_code);
    }
    if (!fmap_.empty()) {
      const auto& spec = fmap_.begin()->second;
      return ffi::String(spec.kernels.empty() ? std::string() : spec.kernels.front().source_code);
    }
    return ffi::String("");
  }

  /*!\brief Initialize TT-Metal device */
  void EnsureDeviceInitialized() {
    if (device_initialized_) return;

    LOG(INFO) << "Initializing Blackhole TT-Metal device...";

    try {
      // Create mesh device (single device for now)
      auto* device_ptr = new std::shared_ptr<distributed::MeshDevice>(
          distributed::MeshDevice::create_unit_mesh(0));
      mesh_device_ = device_ptr;

      LOG(INFO) << "Blackhole device initialized successfully";
      device_initialized_ = true;
    } catch (const std::exception& e) {
      LOG(FATAL) << "Failed to initialize Blackhole device: " << e.what();
    }
  }

  /*!\brief Get or compile a program for the given function */
  CompiledProgram& GetOrCompileProgram(const std::string& func_name) {
    auto it = program_cache_.find(func_name);
    if (it != program_cache_.end()) {
      return it->second;
    }

    CompiledProgram prog;
    prog.program = nullptr;
    prog.reader_kernel = nullptr;
    prog.compute_kernel = nullptr;
    prog.writer_kernel = nullptr;
    prog.is_compiled = false;

    auto fit = fmap_.find(func_name);
    if (fit == fmap_.end()) {
      LOG(FATAL) << "Function not found: " << func_name;
      return program_cache_[func_name] = std::move(prog);
    }

    const ExecutableSpec& info = fit->second;

    try {
      // Create program
      Program* program = new Program(CreateProgram());
      prog.program = program;

      // Use single core (0, 0) for now
      constexpr CoreCoord core = {0, 0};

      // Save kernel code to file
      std::string kernel_path = kernel_dir_ + "/" + func_name + "_kernel.cpp";
      {
        std::ofstream ofs(kernel_path);
        if (!ofs) {
          LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
        }
        if (info.kernels.empty()) {
          LOG(FATAL) << "ExecutableSpec has no kernels for function: " << func_name;
        }
        ofs << info.kernels.front().source_code;
      }

      // Create kernel based on type
      bool has_reader = false;
      bool has_compute = false;
      bool has_writer = false;
      for (const auto& kernel : info.kernels) {
        has_reader = has_reader || kernel.kind == "reader";
        has_compute = has_compute || kernel.kind == "compute";
        has_writer = has_writer || kernel.kind == "writer" || kernel.kind == "fused_dataflow";
      }
      if (has_reader || (!has_reader && !has_compute && !has_writer)) {
        // DataMovement kernel (BRISC)
        KernelHandle kernel = CreateKernel(
            *program,
            kernel_path,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                              .noc = NOC::RISCV_0_default});
        prog.reader_kernel = new KernelHandle(kernel);
      }

      if (has_compute) {
        // Compute kernel (TRISC)
        KernelHandle kernel = CreateKernel(
            *program,
            kernel_path + "_compute",  // Separate file for compute kernel
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                         .fp32_dest_acc_en = false,
                         .math_approx_mode = false});
        prog.compute_kernel = new KernelHandle(kernel);
      }

      if (has_writer) {
        // DataMovement kernel (NCRISC) for writer
        KernelHandle kernel = CreateKernel(
            *program,
            kernel_path + "_writer",  // Separate file for writer kernel
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                              .noc = NOC::RISCV_1_default});
        prog.writer_kernel = new KernelHandle(kernel);
      }

      prog.is_compiled = true;
      LOG(INFO) << "Compiled program for " << func_name;

    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to compile program: " << e.what();
    }

    program_cache_[func_name] = std::move(prog);
    return program_cache_[func_name];
  }

  /*!\brief Execute a function with given arguments */
  void Execute(const std::string& func_name,
               const std::vector<DLTensor*>& inputs,
               const std::vector<uint32_t>& scalar_args,
               const std::vector<DLTensor*>& outputs);

 private:
  std::unordered_map<std::string, ExecutableSpec> fmap_;
  std::string kernel_dir_;
  void* mesh_device_;
  bool device_initialized_;
  std::unordered_map<std::string, CompiledProgram> program_cache_;

  friend class BlackholeWrappedFunc;
};

void BlackholeModuleNode::Execute(const std::string& func_name,
                                   const std::vector<DLTensor*>& inputs,
                                   const std::vector<uint32_t>& scalar_args,
                                   const std::vector<DLTensor*>& outputs) {
  EnsureDeviceInitialized();

  auto* device_ptr = static_cast<std::shared_ptr<distributed::MeshDevice>*>(mesh_device_);
  if (!device_ptr || !*device_ptr) {
    LOG(FATAL) << "Device not initialized";
  }
  auto& mesh_device = *device_ptr;
  distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

  CompiledProgram& prog = GetOrCompileProgram(func_name);
  if (!prog.is_compiled) {
    LOG(FATAL) << "Program not compiled for " << func_name;
  }

  Program* program = static_cast<Program*>(prog.program);
  constexpr CoreCoord core = {0, 0};

  // Calculate total sizes
  size_t total_input_size = 0;
  for (auto* tensor : inputs) {
    total_input_size += GetDataSize(*tensor);
  }

  size_t total_output_size = 0;
  for (auto* tensor : outputs) {
    total_output_size += GetDataSize(*tensor);
  }

  // Create DRAM buffers
  auto fit = fmap_.find(func_name);
  const ExecutableSpec& info = fit->second;

  // Page size from CB config or default to 2048 (32x32 FP16 tile)
  uint32_t page_size = 2048;
  if (!info.cb_configs.empty()) {
    page_size = info.cb_configs[0].page_size_bytes;
  }

  distributed::DeviceLocalBufferConfig dram_config{
      .page_size = page_size,
      .buffer_type = BufferType::DRAM};

  // Create input buffer
  distributed::ReplicatedBufferConfig input_buffer_config{.size = total_input_size};
  auto input_buffer = distributed::MeshBuffer::create(
      input_buffer_config, dram_config, mesh_device.get());

  // Create output buffer
  distributed::ReplicatedBufferConfig output_buffer_config{.size = total_output_size};
  auto output_buffer = distributed::MeshBuffer::create(
      output_buffer_config, dram_config, mesh_device.get());

  // Write input data
  size_t offset = 0;
  for (auto* tensor : inputs) {
    size_t size = GetDataSize(*tensor);
    EnqueueWriteMeshBuffer(cq, input_buffer,
                          std::vector<uint8_t>(static_cast<uint8_t*>(tensor->data),
                                               static_cast<uint8_t*>(tensor->data) + size),
                          /*blocking=*/true);
    offset += size;
  }

  // Set runtime arguments
  std::vector<uint32_t> runtime_args = scalar_args;
  runtime_args.push_back(static_cast<uint32_t>(input_buffer->address()));
  runtime_args.push_back(static_cast<uint32_t>(output_buffer->address()));

  if (prog.reader_kernel) {
    SetRuntimeArgs(*program, *static_cast<KernelHandle*>(prog.reader_kernel), core, runtime_args);
  }

  if (prog.writer_kernel) {
    SetRuntimeArgs(*program, *static_cast<KernelHandle*>(prog.writer_kernel), core,
                  {static_cast<uint32_t>(output_buffer->address())});
  }

  // Execute program
  distributed::MeshWorkload workload;
  distributed::MeshCoordinateRange device_range(mesh_device->shape());
  workload.add_program(device_range, std::move(*program));
  distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

  // Read back results
  std::vector<uint8_t> output_data;
  distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

  // Copy to output tensors
  offset = 0;
  for (auto* tensor : outputs) {
    size_t size = GetDataSize(*tensor);
    std::memcpy(tensor->data, output_data.data() + offset, size);
    offset += size;
  }

  LOG(INFO) << "Execution completed for " << func_name;
}

void BlackholeWrappedFunc::operator()(ffi::PackedArgs args, ffi::Any* rv,
                                       void** void_args) const {
  // Collect arguments
  std::vector<DLTensor*> inputs;
  std::vector<DLTensor*> outputs;
  std::vector<uint32_t> scalars;

  for (size_t i = 0; i < info_.tvm_arg_types.size(); ++i) {
    if (info_.tvm_is_buffer_arg[i]) {
      DLTensor* tensor = static_cast<DLTensor*>(void_args[i]);
      if (i < info_.tvm_arg_types.size() - 1) {
        inputs.push_back(tensor);
      } else {
        outputs.push_back(tensor);
      }
    } else {
      // Extract scalar from packed args
      if (info_.tvm_arg_types[i].code == kDLInt) {
        scalars.push_back(static_cast<uint32_t>(args[i].operator int64_t()));
      } else if (info_.tvm_arg_types[i].code == kDLUInt) {
        scalars.push_back(static_cast<uint32_t>(args[i].operator uint64_t()));
      } else if (info_.tvm_arg_types[i].code == kDLFloat) {
        float f = args[i].operator double();
        scalars.push_back(*reinterpret_cast<uint32_t*>(&f));
      }
    }
  }

  // Execute
  m_->Execute(func_name_, inputs, scalars, outputs);
}

// Create function
ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir) {
  auto n = ffi::make_object<BlackholeModuleNode>(std::move(fmap), std::move(kernel_dir));
  return ffi::Module(std::move(n));
}

// Load module from bytes (deserialization)
ffi::Module BlackholeModuleLoadFromBytes(const ffi::Bytes& bytes) {
  LOG(FATAL) << "BlackholeModule LoadFromBytes not yet implemented";
  __builtin_unreachable();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.module.loadbinary_blackhole", BlackholeModuleLoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm
