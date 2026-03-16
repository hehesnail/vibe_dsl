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
 * \brief CB configuration for a kernel
 */
struct CBConfig {
  uint32_t cb_id;
  uint32_t num_pages;
  uint32_t page_size;
  std::string data_format;
};

/*!
 * \brief Function information for Blackhole kernels
 */
struct BlackholeFunctionInfo {
  std::string kernel_code;           // Generated kernel source code
  std::vector<DLDataType> arg_types;   // Argument types
  std::vector<bool> is_buffer_arg;   // Whether each arg is a buffer (vs scalar)
  std::vector<CBConfig> cb_configs;  // CB configurations
  std::string kernel_path;           // Path to saved kernel file
  bool has_reader = false;           // Whether this function has reader kernel
  bool has_compute = false;          // Whether this function has compute kernel
  bool has_writer = false;           // Whether this function has writer kernel
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
  BlackholeModuleNode(std::unordered_map<std::string, BlackholeFunctionInfo> fmap,
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
  std::unordered_map<std::string, BlackholeFunctionInfo> fmap_;
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
    std::unordered_map<std::string, BlackholeFunctionInfo> fmap,
    std::string kernel_dir);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_TL_TARGET_BLACKHOLE_MODULE_H_
