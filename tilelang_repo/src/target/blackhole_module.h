/*!
 * \file target/blackhole_module.h
 * \brief Execution handling of Tenstorrent Blackhole kernels
 */
#ifndef TVM_TL_TARGET_BLACKHOLE_MODULE_H_
#define TVM_TL_TARGET_BLACKHOLE_MODULE_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief Maximum number of CBs supported on Blackhole */
static constexpr const uint32_t kBlackholeMaxCBs = 64;

/*!
 * \brief Runtime-ready circular buffer configuration.
 */
struct CBConfig {
  uint32_t cb_id;
  std::string name;
  std::string role;
  uint32_t num_pages;
  uint32_t page_size_bytes;
  std::string data_format;
};

/*!
 * \brief Host scheduling plan derived from Blackhole passes.
 */
struct CorePlan {
  uint32_t grid_x = 1;
  uint32_t grid_y = 1;
  uint32_t cores_needed = 1;
  uint32_t work_per_core = 1;
  uint32_t core_grid_x = 1;
  uint32_t core_grid_y = 1;
};

/*!
 * \brief Runtime argument schema for an emitted TT-Metal kernel.
 */
struct KernelArgSpec {
  std::string name;
  std::string kind;
  std::string dtype;
};

/*!
 * \brief Per-kernel source and argument metadata.
 */
struct KernelSpec {
  std::string name;
  std::string kind;
  std::string core_type;
  std::string source_code;
  std::vector<uint32_t> compile_time_args;
  std::vector<KernelArgSpec> runtime_args;
};

/*!
 * \brief Stage 0 executable description for a lowered PrimFunc.
 */
struct ExecutableSpec {
  std::string entry_name;
  std::string target_mode;
  std::vector<CBConfig> cb_configs;
  CorePlan core_plan;
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelSpec> kernels;

  // TVM runtime invocation metadata retained during Stage 0.
  std::vector<DLDataType> tvm_arg_types;
  std::vector<bool> tvm_is_buffer_arg;
};

/*!
 * \brief Compiled program with kernels
 */
struct CompiledProgram {
  // Program and kernels (forward declarations to avoid TT-Metal headers here)
  void* program;           // tt::tt_metal::Program*
  void* reader_kernel;     // tt::tt_metal::KernelHandle (or 0 if none)
  void* compute_kernel;    // tt::tt_metal::KernelHandle (or 0 if none)
  void* writer_kernel;     // tt::tt_metal::KernelHandle (or 0 if none)
  bool is_compiled;        // Whether this program has been JIT compiled
};

/*!
 * \brief Blackhole module for executing TT-Metal kernels
 *
 * This module manages the lifecycle of TT-Metal device, programs, and kernels.
 * It provides a TVM-compatible interface for executing kernels on Blackhole hardware
 * or TT-Sim simulator.
 */
class BlackholeModuleNode : public ffi::ModuleObj {
 public:
  /*! \brief Constructor */
  BlackholeModuleNode(std::unordered_map<std::string, ExecutableSpec> fmap,
                      std::string kernel_dir);

  /*! \brief Destructor - clean up TT-Metal resources */
  ~BlackholeModuleNode();

  /*! \brief Return module kind */
  const char* kind() const final { return "blackhole"; }

  /*! \brief Get module properties */
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  /*! \brief Get function by name */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  /*! \brief Save to file (serialization) */
  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final;

  /*! \brief Save to bytes (serialization) */
  ffi::Bytes SaveToBytes() const final;

  /*! \brief Inspect source code */
  ffi::String InspectSource(const ffi::String& format) const final;

  /*! \brief Initialize TT-Metal device (lazy initialization) */
  void EnsureDeviceInitialized();

  /*! \brief Get or compile a program */
  CompiledProgram& GetOrCompileProgram(const std::string& func_name);

  /*! \brief Execute function using external runner process */
  void ExecuteExternal(const std::string& func_name,
                       const std::vector<DLTensor*>& inputs,
                       const std::vector<uint32_t>& scalar_args,
                       const std::vector<DLTensor*>& outputs);

 private:
  // Function information map
  std::unordered_map<std::string, ExecutableSpec> fmap_;
  // Directory for kernel files
  std::string kernel_dir_;

  // TT-Metal resources (lazy initialization)
  void* mesh_device_;           // std::shared_ptr<tt::tt_metal::distributed::MeshDevice>*
  void* mesh_command_queue_;    // tt::tt_metal::distributed::MeshCommandQueue*
  bool device_initialized_;

  // Compiled program cache
  std::unordered_map<std::string, CompiledProgram> program_cache_;
};

/*!
 * \brief Create a Blackhole module
 * \param fmap Map of function name to function info
 * \param kernel_dir Directory for kernel files
 * \return The created module
 */
ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_TL_TARGET_BLACKHOLE_MODULE_H_
