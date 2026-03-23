/*!
 * \file target/blackhole_module.cc
 * \brief Unified Blackhole module implementation
 *
 * This file provides both execution paths for Blackhole kernels:
 * - Direct TT-Metal API path (default when compiled with TILELANG_BLACKHOLE_DIRECT)
 * - External runner process path (fallback, or when TT-Metal not linked)
 *
 * At runtime, set TILELANG_BH_USE_RUNNER=1 to force the external runner path.
 */

#include "blackhole_module.h"

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>

#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"

#ifdef TILELANG_BLACKHOLE_DIRECT
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#endif

namespace tvm {
namespace runtime {

// Forward declaration
class BlackholeWrappedFunc;

// ============================================================================
// JSON serialization helpers (for external runner path)
// ============================================================================

std::string EscapeJson(const std::string& value) {
  std::ostringstream os;
  for (char ch : value) {
    switch (ch) {
      case '\"':
        os << "\\\"";
        break;
      case '\\':
        os << "\\\\";
        break;
      case '\b':
        os << "\\b";
        break;
      case '\f':
        os << "\\f";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(static_cast<unsigned char>(ch));
        } else {
          os << ch;
        }
        break;
    }
  }
  return os.str();
}

std::string QuoteJson(const std::string& value) {
  return "\"" + EscapeJson(value) + "\"";
}

std::string SerializeKernelArgSpec(const KernelArgSpec& arg) {
  std::ostringstream os;
  os << "{"
     << "\"name\":" << QuoteJson(arg.name) << ","
     << "\"kind\":" << QuoteJson(arg.kind) << ","
     << "\"dtype\":" << QuoteJson(arg.dtype)
     << "}";
  return os.str();
}

std::string SerializePhysicalCore(const PhysicalCore& core) {
  std::ostringstream os;
  os << "{"
     << "\"core_x\":" << core.core_x << ","
     << "\"core_y\":" << core.core_y
     << "}";
  return os.str();
}

std::string SerializeWorkPacket(const WorkPacket& packet) {
  std::ostringstream os;
  os << "{"
     << "\"core_x\":" << packet.core_x << ","
     << "\"core_y\":" << packet.core_y << ","
     << "\"work_offset\":" << packet.work_offset << ","
     << "\"work_count\":" << packet.work_count
     << "}";
  return os.str();
}

std::string SerializeExecutableSpec(const ExecutableSpec& spec,
                                    const std::vector<uint32_t>& scalar_args,
                                    size_t input_size_bytes,
                                    size_t output_size_bytes,
                                    const std::vector<std::string>& kernel_paths) {
  std::ostringstream os;
  os << "{";
  os << "\"entry_name\":" << QuoteJson(spec.entry_name) << ",";
  os << "\"input_size_bytes\":" << input_size_bytes << ",";
  os << "\"output_size_bytes\":" << output_size_bytes << ",";

  os << "\"scalar_args\":[";
  for (size_t i = 0; i < scalar_args.size(); ++i) {
    if (i != 0) os << ",";
    os << scalar_args[i];
  }
  os << "],";

  os << "\"core_plan\":{"
     << "\"logical_grid_x\":" << spec.core_plan.logical_grid_x << ","
     << "\"logical_grid_y\":" << spec.core_plan.logical_grid_y << ","
     << "\"linearization\":" << QuoteJson(spec.core_plan.linearization) << ",";

  os << "\"physical_cores\":[";
  for (size_t i = 0; i < spec.core_plan.physical_cores.size(); ++i) {
    if (i != 0) os << ",";
    os << SerializePhysicalCore(spec.core_plan.physical_cores[i]);
  }
  os << "],";

  os << "\"work_packets\":[";
  for (size_t i = 0; i < spec.core_plan.work_packets.size(); ++i) {
    if (i != 0) os << ",";
    os << SerializeWorkPacket(spec.core_plan.work_packets[i]);
  }
  os << "]"
     << "},";

  os << "\"cb_configs\":[";
  for (size_t i = 0; i < spec.cb_configs.size(); ++i) {
    if (i != 0) os << ",";
    const auto& cb = spec.cb_configs[i];
    os << "{"
       << "\"cb_id\":" << cb.cb_id << ","
       << "\"name\":" << QuoteJson(cb.name) << ","
       << "\"role\":" << QuoteJson(cb.role) << ","
       << "\"num_pages\":" << cb.num_pages << ","
       << "\"page_size_bytes\":" << cb.page_size_bytes << ","
       << "\"data_format\":" << QuoteJson(cb.data_format)
       << "}";
  }
  os << "],";

  os << "\"kernels\":[";
  for (size_t i = 0; i < spec.kernels.size(); ++i) {
    if (i != 0) os << ",";
    const auto& kernel = spec.kernels[i];
    os << "{"
       << "\"name\":" << QuoteJson(kernel.name) << ","
       << "\"kind\":" << QuoteJson(kernel.kind) << ","
       << "\"core_type\":" << QuoteJson(kernel.core_type) << ","
       << "\"kernel_path\":" << QuoteJson(kernel_paths.at(i)) << ",";

    os << "\"compile_time_args\":[";
    for (size_t j = 0; j < kernel.compile_time_args.size(); ++j) {
      if (j != 0) os << ",";
      os << kernel.compile_time_args[j];
    }
    os << "],";

    os << "\"runtime_args\":[";
    for (size_t j = 0; j < kernel.runtime_args.size(); ++j) {
      if (j != 0) os << ",";
      os << SerializeKernelArgSpec(kernel.runtime_args[j]);
    }
    os << "]";
    os << "}";
  }
  os << "]";
  os << "}";
  return os.str();
}

// ============================================================================
// BlackholeWrappedFunc declaration
// ============================================================================

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

// ============================================================================
// External runner path utilities
// ============================================================================

std::string GetRunnerPath() {
  const char* env_path = std::getenv("TILELANG_BLACKHOLE_RUNNER");
  if (env_path) {
    return std::string(env_path);
  }

  std::vector<std::string> search_paths = {
    "./tilelang_blackhole_runner",
    "/usr/local/bin/tilelang_blackhole_runner",
    "/opt/tt-metal/tilelang_blackhole_runner"
  };

  const char* runner_build_dir = std::getenv("TILELANG_BLACKHOLE_RUNNER_BUILD_DIR");
  if (runner_build_dir) {
    search_paths.push_back(std::string(runner_build_dir) +
      "/tilelang_blackhole_runner");
  }

  const char* tilelang_home = std::getenv("TILELANG_HOME");
  if (tilelang_home) {
    search_paths.push_back(std::string(tilelang_home) +
      "/build-blackhole-runner/tilelang_blackhole_runner");
    search_paths.push_back(std::string(tilelang_home) +
      "/build_blackhole_runner/tilelang_blackhole_runner");
    search_paths.push_back(std::string(tilelang_home) +
      "/tools/blackhole_runner/build/tilelang_blackhole_runner");
  }

  for (const auto& path : search_paths) {
    if (std::filesystem::exists(path)) {
      return path;
    }
  }

  LOG(WARNING) << "tilelang_blackhole_runner not found. "
               << "Set TILELANG_BLACKHOLE_RUNNER environment variable.";
  return "";
}

static std::string MakeUniqueDirectKernelDir() {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  std::filesystem::path dir = std::filesystem::temp_directory_path() /
                              ("tilelang_bh_direct_" + std::to_string(getpid()) + "_" +
                               std::to_string(id));
  std::filesystem::create_directories(dir);
  return dir.string();
}

// Argument extraction helpers
uint32_t ExtractScalar(const ffi::AnyView& arg, DLDataType dtype) {
  if (dtype.code == kDLInt) {
    auto opt_i32 = arg.try_cast<int32_t>();
    if (opt_i32.has_value()) {
      return static_cast<uint32_t>(opt_i32.value());
    }
    auto opt_i64 = arg.try_cast<int64_t>();
    if (opt_i64.has_value()) {
      return static_cast<uint32_t>(opt_i64.value());
    }
  }
  if (dtype.code == kDLUInt) {
    auto opt_u32 = arg.try_cast<uint32_t>();
    if (opt_u32.has_value()) {
      return opt_u32.value();
    }
    auto opt_u64 = arg.try_cast<uint64_t>();
    if (opt_u64.has_value()) {
      return static_cast<uint32_t>(opt_u64.value());
    }
  }
  if (dtype.code == kDLFloat) {
    float f = 0.0f;
    auto opt_f = arg.try_cast<float>();
    if (opt_f.has_value()) {
      f = opt_f.value();
    } else {
      auto opt_d = arg.try_cast<double>();
      if (opt_d.has_value()) {
        f = static_cast<float>(opt_d.value());
      }
    }
    return *reinterpret_cast<uint32_t*>(&f);
  }
  LOG(FATAL) << "Cannot extract scalar of type code " << dtype.code;
  return 0;
}

DLTensor* ExtractTensorArg(const ffi::AnyView& arg, void* void_arg) {
  auto opt_tensor = arg.try_cast<DLTensor*>();
  if (opt_tensor.has_value()) {
    return opt_tensor.value();
  }
  if (void_arg != nullptr) {
    DLTensor* tensor = *reinterpret_cast<DLTensor**>(void_arg);
    if (tensor != nullptr) {
      return tensor;
    }
  }
  LOG(FATAL) << "Cannot extract DLTensor* from packed argument";
  return nullptr;
}

// ============================================================================
// Direct TT-Metal path helpers (only when linked against TT-Metal)
// ============================================================================

#ifdef TILELANG_BLACKHOLE_DIRECT

using namespace tt::tt_metal;

static tt::DataFormat ParseDataFormat(const std::string& value) {
  if (value == "Float16" || value == "Float16_b") return tt::DataFormat::Float16_b;
  if (value == "Float32") return tt::DataFormat::Float32;
  if (value == "UInt16") return tt::DataFormat::UInt16;
  if (value == "UInt32") return tt::DataFormat::UInt32;
  LOG(FATAL) << "Unsupported data format: " << value;
  return tt::DataFormat::Float16_b;
}

static uint32_t ChoosePageSize(const ExecutableSpec& spec, const std::string& role) {
  for (const auto& cb : spec.cb_configs) {
    if (cb.role == role) return cb.page_size_bytes;
  }
  if (!spec.cb_configs.empty()) return spec.cb_configs.front().page_size_bytes;
  return 2048;
}

static void CreateCircularBuffersFromSpec(
    Program& program, const CoreCoord& core, const ExecutableSpec& spec) {
  for (const auto& cb : spec.cb_configs) {
    uint32_t total_size = cb.num_pages * cb.page_size_bytes;
    CircularBufferConfig cb_config(
        total_size,
        {{static_cast<uint8_t>(cb.cb_id), ParseDataFormat(cb.data_format)}});
    cb_config.set_page_size(static_cast<uint8_t>(cb.cb_id), cb.page_size_bytes);
    CreateCircularBuffer(program, core, cb_config);
  }
}

static KernelHandle CreateKernelFromSpec(
    Program& program, const CoreCoord& core,
    const KernelSpec& kernel, const std::string& kernel_path) {
  if (kernel.core_type == "trisc") {
    return CreateKernel(
        program,
        kernel_path,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = kernel.compile_time_args});
  }

  DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
  NOC noc = NOC::RISCV_0_default;
  if (kernel.core_type == "ncrisc") {
    processor = DataMovementProcessor::RISCV_1;
    noc = NOC::RISCV_1_default;
  }

  return CreateKernel(
      program,
      kernel_path,
      core,
      DataMovementConfig{
          .processor = processor,
          .noc = noc,
          .compile_args = kernel.compile_time_args});
}

static bool KernelNeedsScratchL1(const KernelSpec& kernel) {
  for (const auto& arg : kernel.runtime_args) {
    if (arg.kind == "scratch_l1_buffer_addr32") return true;
  }
  return false;
}

static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    size_t total_input_size,
    const distributed::MeshBuffer& input_buffer,
    const distributed::MeshBuffer& output_buffer,
    const distributed::MeshBuffer* scratch_l1_buffer,
    const std::vector<uint32_t>& scalar_args) {
  std::vector<uint32_t> args;
  size_t scalar_index = 0;
  const uint32_t tile_size = ChoosePageSize(spec, "input");
  const uint64_t src_addr = input_buffer.address();
  const uint64_t dst_addr = output_buffer.address();

  for (const auto& arg_spec : kernel.runtime_args) {
    if (arg_spec.kind == "input_buffer_addr") {
      args.push_back(static_cast<uint32_t>(src_addr & 0xFFFFFFFF));
      args.push_back(static_cast<uint32_t>(src_addr >> 32));
    } else if (arg_spec.kind == "input_buffer_addr32") {
      args.push_back(static_cast<uint32_t>(src_addr));
    } else if (arg_spec.kind == "output_buffer_addr") {
      args.push_back(static_cast<uint32_t>(dst_addr & 0xFFFFFFFF));
      args.push_back(static_cast<uint32_t>(dst_addr >> 32));
    } else if (arg_spec.kind == "output_buffer_addr32") {
      args.push_back(static_cast<uint32_t>(dst_addr));
    } else if (arg_spec.kind == "tile_count") {
      // tile_count = total_input_size / tile_size (matching runner.cpp)
      args.push_back(tile_size == 0 ? 0 : static_cast<uint32_t>(total_input_size / tile_size));
    } else if (arg_spec.kind == "current_work_linear_id") {
      args.push_back(current_work_linear_id);
    } else if (arg_spec.kind == "scratch_l1_buffer_addr32") {
      ICHECK(scratch_l1_buffer != nullptr)
          << "Spec requested scratch L1 buffer but none was allocated";
      args.push_back(static_cast<uint32_t>(scratch_l1_buffer->address()));
    } else if (arg_spec.kind == "scalar_u32") {
      ICHECK(scalar_index < scalar_args.size())
          << "Spec requested more scalar args than provided";
      args.push_back(scalar_args[scalar_index++]);
    } else {
      LOG(FATAL) << "Unsupported runtime arg kind: " << arg_spec.kind;
    }
  }

  return args;
}

#endif  // TILELANG_BLACKHOLE_DIRECT

// ============================================================================
// BlackholeModuleNode implementation
// ============================================================================

BlackholeModuleNode::BlackholeModuleNode(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir)
    : fmap_(std::move(fmap)),
      kernel_dir_(std::move(kernel_dir)),
      mesh_device_(nullptr),
      mesh_command_queue_(nullptr),
      device_initialized_(false) {
}

BlackholeModuleNode::~BlackholeModuleNode() {
#ifdef TILELANG_BLACKHOLE_DIRECT
  if (mesh_device_) {
    delete static_cast<std::shared_ptr<tt::tt_metal::distributed::MeshDevice>*>(mesh_device_);
    mesh_device_ = nullptr;
  }
#endif
}

ffi::Optional<ffi::Function> BlackholeModuleNode::GetFunction(const ffi::String& name) {
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

void BlackholeModuleNode::WriteToFile(const ffi::String& file_name,
                                      const ffi::String& format) const {
  LOG(WARNING) << "BlackholeModule WriteToFile not yet implemented";
}

ffi::Bytes BlackholeModuleNode::SaveToBytes() const {
  LOG(WARNING) << "BlackholeModule SaveToBytes not yet implemented";
  return ffi::Bytes("");
}

ffi::String BlackholeModuleNode::InspectSource(const ffi::String& format) const {
  LOG(INFO) << "BlackholeModuleNode::InspectSource called";
  auto it = fmap_.find("default");
  if (it != fmap_.end()) {
    const auto& spec = it->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Found 'default' function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  if (!fmap_.empty()) {
    const auto& spec = fmap_.begin()->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Using first function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  LOG(WARNING) << "No functions found in fmap_";
  return ffi::String("");
}

// ============================================================================
// Device initialization
// ============================================================================

void BlackholeModuleNode::EnsureDeviceInitialized() {
#ifdef TILELANG_BLACKHOLE_DIRECT
  if (device_initialized_) return;

  LOG(INFO) << "Initializing Blackhole TT-Metal device...";

  try {
    auto* device_ptr = new std::shared_ptr<tt::tt_metal::distributed::MeshDevice>(
        tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0));
    mesh_device_ = device_ptr;

    LOG(INFO) << "Blackhole device initialized successfully";
    device_initialized_ = true;
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Blackhole device: " << e.what();
  }
#else
  // External process mode - device init happens in runner
  device_initialized_ = true;
#endif
}

// ============================================================================
// Program cache
// ============================================================================

CompiledProgram& BlackholeModuleNode::GetOrCompileProgram(const std::string& func_name) {
  auto it = program_cache_.find(func_name);
  if (it != program_cache_.end()) {
    return it->second;
  }

  // Create placeholder - actual program creation happens per-execution
  // because TT-Metal requires fresh Program per work item
  CompiledProgram prog;
  prog.program = nullptr;
  prog.reader_kernel = nullptr;
  prog.compute_kernel = nullptr;
  prog.writer_kernel = nullptr;
  prog.is_compiled = true;

  program_cache_[func_name] = std::move(prog);
  return program_cache_[func_name];
}

// ============================================================================
// External runner execution path
// ============================================================================

void BlackholeModuleNode::ExecuteExternal(
    const std::string& func_name,
    const std::vector<DLTensor*>& inputs,
    const std::vector<uint32_t>& scalar_args,
    const std::vector<DLTensor*>& outputs) {
  std::string runner_path = GetRunnerPath();
  if (runner_path.empty()) {
    LOG(FATAL) << "External runner not found. "
               << "Please build and install tilelang_blackhole_runner, "
               << "or set TILELANG_BLACKHOLE_RUNNER environment variable.";
  }

  auto fit = fmap_.find(func_name);
  if (fit == fmap_.end()) {
    LOG(FATAL) << "Function not found: " << func_name;
  }
  const ExecutableSpec& info = fit->second;
  if (info.kernels.empty()) {
    LOG(FATAL) << "ExecutableSpec has no kernels for function: " << func_name;
  }

  // Calculate sizes
  size_t total_input_size = 0;
  for (auto* tensor : inputs) {
    total_input_size += GetDataSize(*tensor);
  }

  size_t total_output_size = 0;
  for (auto* tensor : outputs) {
    total_output_size += GetDataSize(*tensor);
  }

  // Create temporary directory for I/O data
  std::string tmp_dir = "/tmp/tilelang_blackhole_" + std::to_string(getpid());
  std::filesystem::create_directories(tmp_dir);

  std::string input_path = tmp_dir + "/input.bin";
  std::string output_path = tmp_dir + "/output.bin";
  std::string spec_path = tmp_dir + "/spec.json";

  // Write input data to file
  {
    std::ofstream input_file(input_path, std::ios::binary);
    if (!input_file) {
      LOG(FATAL) << "Failed to create input file: " << input_path;
    }
    for (auto* tensor : inputs) {
      size_t size = GetDataSize(*tensor);
      input_file.write(static_cast<char*>(tensor->data), size);
    }
  }

  std::vector<std::string> kernel_paths;
  kernel_paths.reserve(info.kernels.size());
  for (size_t i = 0; i < info.kernels.size(); ++i) {
    const auto& kernel = info.kernels[i];
    std::string kernel_path = tmp_dir + "/" + func_name + "_" + std::to_string(i) + "_" +
                              kernel.kind + ".cpp";
    std::ofstream ofs(kernel_path);
    if (!ofs) {
      LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
    }
    ofs << kernel.source_code;
    kernel_paths.push_back(kernel_path);
  }

  {
    std::ofstream spec_file(spec_path);
    if (!spec_file) {
      LOG(FATAL) << "Failed to create spec file: " << spec_path;
    }
    spec_file << SerializeExecutableSpec(
        info, scalar_args, total_input_size, total_output_size, kernel_paths);
  }

  // Build command line
  std::vector<std::string> cmd_args = {
    runner_path,
    spec_path,
    input_path,
    output_path
  };

  // Execute external runner
  LOG(INFO) << "Executing external runner: " << runner_path;
  LOG(INFO) << "  Spec: " << spec_path;
  LOG(INFO) << "  Input: " << input_path << " (" << total_input_size << " bytes)";
  LOG(INFO) << "  Output: " << output_path << " (" << total_output_size << " bytes)";

  pid_t pid = fork();
  if (pid < 0) {
    LOG(FATAL) << "Fork failed: " << strerror(errno);
  }

  if (pid == 0) {
    // Child process
    std::vector<char*> argv;
    for (auto& arg : cmd_args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    execv(runner_path.c_str(), argv.data());
    std::cerr << "Failed to execute runner: " << strerror(errno) << std::endl;
    _exit(1);
  } else {
    // Parent process
    int status;
    if (waitpid(pid, &status, 0) < 0) {
      LOG(FATAL) << "Waitpid failed: " << strerror(errno);
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      LOG(FATAL) << "External runner failed with status: "
                 << (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
  }

  // Read output data from file
  {
    std::ifstream output_file(output_path, std::ios::binary);
    if (!output_file) {
      LOG(FATAL) << "Failed to open output file: " << output_path;
    }

    for (auto* tensor : outputs) {
      size_t size = GetDataSize(*tensor);
      if (!output_file.read(static_cast<char*>(tensor->data), size)) {
        LOG(FATAL) << "Failed to read output data";
      }
    }
  }

  // Cleanup temporary files
  std::filesystem::remove_all(tmp_dir);

  LOG(INFO) << "External runner execution completed for " << func_name;
}

// ============================================================================
// Direct TT-Metal execution path
// ============================================================================

void BlackholeModuleNode::ExecuteDirect(
    const std::string& func_name,
    const std::vector<DLTensor*>& inputs,
    const std::vector<uint32_t>& scalar_args,
    const std::vector<DLTensor*>& outputs) {
#ifdef TILELANG_BLACKHOLE_DIRECT
  using namespace tt::tt_metal;

  // Keep direct execution hermetic per call. Reusing a persistent MeshDevice across
  // multiple Python direct-call tests can leave simulator/device state behind and
  // cause cross-test contamination that does not exist in the runner process model.
  LOG(INFO) << "Initializing Blackhole TT-Metal device...";
  std::shared_ptr<distributed::MeshDevice> mesh_device;
  try {
    mesh_device = distributed::MeshDevice::create_unit_mesh(0);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Blackhole device: " << e.what();
  }
  ICHECK(mesh_device != nullptr);
  LOG(INFO) << "Blackhole device initialized successfully";

  distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

  auto fit = fmap_.find(func_name);
  if (fit == fmap_.end()) {
    LOG(FATAL) << "Function not found: " << func_name;
  }
  const ExecutableSpec& spec = fit->second;
  if (spec.kernels.empty()) {
    LOG(FATAL) << "ExecutableSpec has no kernels for function: " << func_name;
  }

  // Calculate total sizes
  size_t total_input_size = 0;
  for (auto* tensor : inputs) {
    total_input_size += GetDataSize(*tensor);
  }

  size_t total_output_size = 0;
  for (auto* tensor : outputs) {
    total_output_size += GetDataSize(*tensor);
  }

  // Use role-aware page size for DRAM buffers (matching runner.cpp)
  uint32_t input_page_size = ChoosePageSize(spec, "input");
  uint32_t output_page_size = ChoosePageSize(spec, "output");

  // Create DRAM buffers
  distributed::DeviceLocalBufferConfig input_dram_config{
      .page_size = input_page_size,
      .buffer_type = BufferType::DRAM};
  distributed::DeviceLocalBufferConfig output_dram_config{
      .page_size = output_page_size,
      .buffer_type = BufferType::DRAM};

  distributed::ReplicatedBufferConfig input_buffer_config{.size = total_input_size};
  auto input_buffer = distributed::MeshBuffer::create(
      input_buffer_config, input_dram_config, mesh_device.get());

  distributed::ReplicatedBufferConfig output_buffer_config{.size = total_output_size};
  auto output_buffer = distributed::MeshBuffer::create(
      output_buffer_config, output_dram_config, mesh_device.get());

  // Create scratch L1 buffer if any kernel needs it
  std::shared_ptr<distributed::MeshBuffer> scratch_l1_buffer;
  bool needs_scratch_l1 = false;
  for (const auto& kernel_spec : spec.kernels) {
    if (KernelNeedsScratchL1(kernel_spec)) {
      needs_scratch_l1 = true;
      break;
    }
  }
  if (needs_scratch_l1) {
    uint32_t scratch_size = input_page_size;
    for (const auto& cb : spec.cb_configs) {
      scratch_size = std::max(scratch_size, cb.num_pages * cb.page_size_bytes);
    }
    distributed::DeviceLocalBufferConfig scratch_l1_config{
        .page_size = scratch_size,
        .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig scratch_l1_buffer_config{.size = scratch_size};
    scratch_l1_buffer = distributed::MeshBuffer::create(
        scratch_l1_buffer_config, scratch_l1_config, mesh_device.get());
  }

  // Write input data to device
  std::vector<uint8_t> input_data;
  input_data.reserve(total_input_size);
  for (auto* tensor : inputs) {
    size_t size = GetDataSize(*tensor);
    auto* p = static_cast<uint8_t*>(tensor->data);
    input_data.insert(input_data.end(), p, p + size);
  }
  EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/true);

  // Build work IDs from work_packets (matching runner.cpp)
  std::vector<uint32_t> work_ids;
  for (const auto& packet : spec.core_plan.work_packets) {
    for (uint32_t i = 0; i < packet.work_count; ++i) {
      work_ids.push_back(packet.work_offset + i);
    }
  }
  if (work_ids.empty()) {
    work_ids.push_back(0);
  }

  // Write kernel source files to temp directory
  std::string tmp_dir = MakeUniqueDirectKernelDir();

  std::vector<std::string> kernel_paths;
  kernel_paths.reserve(spec.kernels.size());
  for (size_t i = 0; i < spec.kernels.size(); ++i) {
    const auto& kernel = spec.kernels[i];
    std::string kernel_path = tmp_dir + "/" + func_name + "_" + std::to_string(i) + "_" +
                              kernel.kind + ".cpp";
    std::ofstream ofs(kernel_path);
    if (!ofs) {
      LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
    }
    ofs << kernel.source_code;
    kernel_paths.push_back(kernel_path);
  }

  // Execute each work item (matching runner.cpp work-packet iteration)
  LOG(INFO) << "Direct path: executing " << work_ids.size()
            << " logical work items for " << func_name;

  constexpr CoreCoord core = {0, 0};

  for (uint32_t work_id : work_ids) {
    Program program = CreateProgram();

    // Create circular buffers for this program
    CreateCircularBuffersFromSpec(program, core, spec);

    // Create and configure each kernel
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
      const auto& kernel_spec = spec.kernels[ki];
      KernelHandle kernel = CreateKernelFromSpec(
          program, core, kernel_spec, kernel_paths[ki]);

      auto runtime_args = BuildRuntimeArgsFromSpec(
          kernel_spec, spec, work_id, total_input_size,
          *input_buffer, *output_buffer,
          scratch_l1_buffer ? scratch_l1_buffer.get() : nullptr,
          scalar_args);

      SetRuntimeArgs(program, kernel, core, runtime_args);
    }

    // Execute program on device
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
  }

  // Read back results
  std::vector<uint8_t> output_data;
  distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);

  // Copy to output tensors
  size_t offset = 0;
  for (auto* tensor : outputs) {
    size_t size = GetDataSize(*tensor);
    ICHECK(offset + size <= output_data.size())
        << "Output data size mismatch: need " << (offset + size)
        << " but got " << output_data.size();
    std::memcpy(tensor->data, output_data.data() + offset, size);
    offset += size;
  }

  // Cleanup kernel temp files
  std::filesystem::remove_all(tmp_dir);

  LOG(INFO) << "Direct path execution completed for " << func_name;

#else
  LOG(FATAL) << "Direct TT-Metal path not available. "
             << "Rebuild with TILELANG_BLACKHOLE_DIRECT=ON or use external runner.";
#endif  // TILELANG_BLACKHOLE_DIRECT
}

// ============================================================================
// Execution dispatch
// ============================================================================

/*!
 * \brief Check if direct execution mode should be used.
 *
 * Default behavior depends on compile-time flag:
 * - With TILELANG_BLACKHOLE_DIRECT: direct path is default
 * - Without: external runner is the only option
 *
 * Set TILELANG_BH_USE_RUNNER=1 to force external runner path.
 */
static bool ShouldUseDirectPath() {
#ifdef TILELANG_BLACKHOLE_DIRECT
  const char* env = std::getenv("TILELANG_BH_USE_RUNNER");
  if (env && std::string(env) == "1") {
    return false;
  }
  return true;
#else
  return false;
#endif
}

void BlackholeWrappedFunc::operator()(ffi::PackedArgs args, ffi::Any* rv,
                                       void** void_args) const {
  // Collect arguments
  std::vector<DLTensor*> inputs;
  std::vector<DLTensor*> outputs;
  std::vector<uint32_t> scalars;

  for (size_t i = 0; i < info_.tvm_arg_types.size(); ++i) {
    if (info_.tvm_is_buffer_arg[i]) {
      DLTensor* tensor = ExtractTensorArg(args[i], void_args != nullptr ? void_args[i] : nullptr);
      if (i < info_.tvm_arg_types.size() - 1) {
        inputs.push_back(tensor);
      } else {
        outputs.push_back(tensor);
      }
    } else {
      ffi::AnyView arg = args[i];
      uint32_t val = ExtractScalar(arg, info_.tvm_arg_types[i]);
      scalars.push_back(val);
    }
  }

  // Dispatch to direct or external path
  if (ShouldUseDirectPath()) {
    m_->ExecuteDirect(func_name_, inputs, scalars, outputs);
  } else {
    m_->ExecuteExternal(func_name_, inputs, scalars, outputs);
  }
}

// ============================================================================
// Module creation and registration
// ============================================================================

ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir) {
  auto n = ffi::make_object<BlackholeModuleNode>(std::move(fmap), std::move(kernel_dir));
  return ffi::Module(std::move(n));
}

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
