/*!
 * \file target/blackhole_module.cc
 * \brief Blackhole module implementation using external process execution
 *
 * This implementation uses an external runner process to execute kernels,
 * avoiding direct TT-Metal linking in TileLang.
 */

#include "blackhole_module.h"

#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>

#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"

namespace tvm {
namespace runtime {

// Forward declaration
class BlackholeWrappedFunc;

/*!
 * \brief Wrapper for Blackhole kernel execution
 */
class BlackholeWrappedFunc {
 public:
  void Init(BlackholeModuleNode* m, ObjectPtr<Object> sptr,
            const std::string& func_name,
            const BlackholeFunctionInfo& info) {
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
  BlackholeFunctionInfo info_;
};

// Get path to external runner executable
std::string GetRunnerPath() {
  // Check environment variable first
  const char* env_path = std::getenv("TILELANG_BLACKHOLE_RUNNER");
  if (env_path) {
    return std::string(env_path);
  }

  // Check standard locations
  std::vector<std::string> search_paths = {
    "./tilelang_blackhole_runner",
    "/usr/local/bin/tilelang_blackhole_runner",
    "/opt/tt-metal/tilelang_blackhole_runner"
  };

  // Check TT_METAL_HOME
  const char* tt_metal_home = std::getenv("TT_METAL_HOME");
  if (tt_metal_home) {
    search_paths.push_back(std::string(tt_metal_home) +
      "/build_Release/programming_examples/tilelang_blackhole_runner/tilelang_blackhole_runner");
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

BlackholeModuleNode::BlackholeModuleNode(
    std::unordered_map<std::string, BlackholeFunctionInfo> fmap,
    std::string kernel_dir)
    : fmap_(std::move(fmap)),
      kernel_dir_(std::move(kernel_dir)),
      mesh_device_(nullptr),
      mesh_command_queue_(nullptr),
      device_initialized_(false) {
}

BlackholeModuleNode::~BlackholeModuleNode() = default;

ffi::Optional<ffi::Function> BlackholeModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);

  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return ffi::Function();
  }

  const BlackholeFunctionInfo& info = it->second;
  BlackholeWrappedFunc f;
  f.Init(this, sptr_to_self, name, info);

  std::vector<FunctionInfo::ArgExtraTags> arg_extra_tags;
  return PackFuncVoidAddr(f, info.arg_types, arg_extra_tags);
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
  auto it = fmap_.find("default");
  if (it != fmap_.end()) {
    return ffi::String(it->second.kernel_code);
  }
  if (!fmap_.empty()) {
    return ffi::String(fmap_.begin()->second.kernel_code);
  }
  return ffi::String("");
}

void BlackholeModuleNode::EnsureDeviceInitialized() {
  // Not used in external process mode
  device_initialized_ = true;
}

CompiledProgram& BlackholeModuleNode::GetOrCompileProgram(const std::string& func_name) {
  auto it = program_cache_.find(func_name);
  if (it != program_cache_.end()) {
    return it->second;
  }

  // Create placeholder - actual compilation happens in external runner
  CompiledProgram prog;
  prog.program = nullptr;
  prog.reader_kernel = nullptr;
  prog.compute_kernel = nullptr;
  prog.writer_kernel = nullptr;
  prog.is_compiled = true;

  program_cache_[func_name] = std::move(prog);
  return program_cache_[func_name];
}

/*!
 * \brief Execute a function with given arguments using external process
 */
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
  const BlackholeFunctionInfo& info = fit->second;

  // Save kernel code to file
  std::string kernel_path = kernel_dir_ + "/" + func_name + "_kernel.cpp";
  {
    std::ofstream ofs(kernel_path);
    if (!ofs) {
      LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
    }
    ofs << info.kernel_code;
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

  // Build command line
  std::vector<std::string> cmd_args = {
    runner_path,
    kernel_path,
    input_path,
    output_path,
    std::to_string(total_input_size),
    std::to_string(total_output_size)
  };

  // Execute external runner
  LOG(INFO) << "Executing external runner: " << runner_path;
  LOG(INFO) << "  Kernel: " << kernel_path;
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
    // If we get here, execv failed
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

  LOG(INFO) << "Execution completed for " << func_name;
}

// Helper to extract scalar value from AnyView
uint32_t ExtractScalar(const ffi::AnyView& arg, DLDataType dtype) {
  // Try integer types
  if (dtype.code == kDLInt) {
    // Try different int sizes using cast
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

void BlackholeWrappedFunc::operator()(ffi::PackedArgs args, ffi::Any* rv,
                                       void** void_args) const {
  // Collect arguments
  std::vector<DLTensor*> inputs;
  std::vector<DLTensor*> outputs;
  std::vector<uint32_t> scalars;

  for (size_t i = 0; i < info_.arg_types.size(); ++i) {
    if (info_.is_buffer_arg[i]) {
      DLTensor* tensor = static_cast<DLTensor*>(void_args[i]);
      if (i < info_.arg_types.size() - 1) {
        inputs.push_back(tensor);
      } else {
        outputs.push_back(tensor);
      }
    } else {
      // Extract scalar from packed args
      ffi::AnyView arg = args[i];
      uint32_t val = ExtractScalar(arg, info_.arg_types[i]);
      scalars.push_back(val);
    }
  }

  // Execute via external process
  m_->ExecuteExternal(func_name_, inputs, scalars, outputs);
}

// Create function
ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, BlackholeFunctionInfo> fmap,
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
