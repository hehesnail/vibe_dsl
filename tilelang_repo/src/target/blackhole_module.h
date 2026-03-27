/*!
 * \file target/blackhole_module.h
 * \brief Execution handling of Tenstorrent Blackhole kernels
 */
#ifndef TVM_TL_TARGET_BLACKHOLE_MODULE_H_
#define TVM_TL_TARGET_BLACKHOLE_MODULE_H_

#include <dmlc/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/runtime/data_type.h>

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

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("cb_id", static_cast<int64_t>(cb_id));
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("role", role);
    writer->WriteObjectKeyValue("num_pages", static_cast<int64_t>(num_pages));
    writer->WriteObjectKeyValue("page_size", static_cast<int64_t>(page_size_bytes));
    writer->WriteObjectKeyValue("data_format", data_format);
    writer->EndObject();
  }
};

/*!
 * \brief Host scheduling plan derived from Blackhole passes.
 */
struct PhysicalCore {
  uint32_t core_x = 0;
  uint32_t core_y = 0;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("core_x", static_cast<int64_t>(core_x));
    writer->WriteObjectKeyValue("core_y", static_cast<int64_t>(core_y));
    writer->EndObject();
  }
};

/*!
 * \brief Work packet assigned to a physical core.
 */
struct WorkPacket {
  uint32_t core_x = 0;
  uint32_t core_y = 0;
  uint32_t work_offset = 0;
  uint32_t work_count = 1;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("core_x", static_cast<int64_t>(core_x));
    writer->WriteObjectKeyValue("core_y", static_cast<int64_t>(core_y));
    writer->WriteObjectKeyValue("work_offset", static_cast<int64_t>(work_offset));
    writer->WriteObjectKeyValue("work_count", static_cast<int64_t>(work_count));
    writer->EndObject();
  }
};

/*!
 * \brief Host scheduling plan derived from Blackhole passes.
 */
struct CorePlan {
  uint32_t logical_grid_x = 1;
  uint32_t logical_grid_y = 1;
  std::string linearization = "row_major";
  std::vector<PhysicalCore> physical_cores;
  std::vector<WorkPacket> work_packets;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("logical_grid_x", static_cast<int64_t>(logical_grid_x));
    writer->WriteObjectKeyValue("logical_grid_y", static_cast<int64_t>(logical_grid_y));
    writer->WriteObjectKeyValue("linearization", linearization);
    writer->WriteObjectKeyValue("physical_cores", physical_cores);
    writer->WriteObjectKeyValue("work_packets", work_packets);
    writer->EndObject();
  }
};

/*!
 * \brief Runtime argument schema for an emitted TT-Metal kernel.
 */
struct KernelArgSpec {
  std::string name;
  std::string kind;
  std::string dtype;
  std::string buffer;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("dtype", dtype);
    if (!buffer.empty()) {
      writer->WriteObjectKeyValue("buffer", buffer);
    }
    writer->EndObject();
  }
};

struct CompileTimeArgSpec {
  std::string name;
  std::string kind;
  std::string dtype;
  uint32_t offset = 0;
  uint32_t count = 0;
  std::string buffer;
  std::string segment_role;
  std::vector<uint32_t> values;
  std::string layout;
  std::string memory_space;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("dtype", dtype);
    writer->WriteObjectKeyValue("offset", static_cast<int64_t>(offset));
    writer->WriteObjectKeyValue("count", static_cast<int64_t>(count));
    if (!buffer.empty()) {
      writer->WriteObjectKeyValue("buffer", buffer);
    }
    if (!segment_role.empty()) {
      writer->WriteObjectKeyValue("segment_role", segment_role);
    }
    if (!values.empty()) {
      std::vector<int64_t> encoded_values;
      encoded_values.reserve(values.size());
      for (uint32_t value : values) {
        encoded_values.push_back(static_cast<int64_t>(value));
      }
      writer->WriteObjectKeyValue("values", encoded_values);
    }
    if (!layout.empty()) {
      writer->WriteObjectKeyValue("layout", layout);
    }
    if (!memory_space.empty()) {
      writer->WriteObjectKeyValue("memory_space", memory_space);
    }
    writer->EndObject();
  }
};

struct KernelLaunchSpec {
  std::string core_type;
  std::string processor;
  std::string noc;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("core_type", core_type);
    writer->WriteObjectKeyValue("processor", processor);
    writer->WriteObjectKeyValue("noc", noc);
    writer->EndObject();
  }
};

struct KernelComputeConfigSpec {
  std::string math_fidelity;
  bool fp32_dest_acc_en = false;
  bool math_approx_mode = false;
  std::vector<std::string> unpack_to_dest_mode;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("math_fidelity", math_fidelity);
    writer->WriteObjectKeyValue("fp32_dest_acc_en", fp32_dest_acc_en);
    writer->WriteObjectKeyValue("math_approx_mode", math_approx_mode);
    if (!unpack_to_dest_mode.empty()) {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", unpack_to_dest_mode);
    } else {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", std::vector<std::string>{});
    }
    writer->EndObject();
  }
};

struct AccessorSpec {
  std::string buffer;
  uint32_t slot = 0;
  uint32_t compile_time_arg_offset = 0;
  uint32_t compile_time_arg_count = 0;
  uint32_t common_runtime_arg_offset = 0;
  uint32_t common_runtime_arg_count = 0;
  uint32_t args_config_bits = 0;
  std::string layout;
  std::string memory_space;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("buffer", buffer);
    writer->WriteObjectKeyValue("slot", static_cast<int64_t>(slot));
    writer->WriteObjectKeyValue("compile_time_arg_offset",
                                static_cast<int64_t>(compile_time_arg_offset));
    writer->WriteObjectKeyValue("compile_time_arg_count",
                                static_cast<int64_t>(compile_time_arg_count));
    writer->WriteObjectKeyValue("common_runtime_arg_offset",
                                static_cast<int64_t>(common_runtime_arg_offset));
    writer->WriteObjectKeyValue("common_runtime_arg_count",
                                static_cast<int64_t>(common_runtime_arg_count));
    writer->WriteObjectKeyValue("args_config_bits", static_cast<int64_t>(args_config_bits));
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("memory_space", memory_space);
    writer->EndObject();
  }
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
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs;
  bool has_launch_spec = false;
  KernelLaunchSpec launch_spec;
  bool has_compute_config = false;
  KernelComputeConfigSpec compute_config;
  std::vector<AccessorSpec> accessors;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("core_type", core_type);
    writer->WriteObjectKeyValue("source_code", source_code);
    if (!compile_time_args.empty()) {
      std::vector<int64_t> encoded_compile_time_args;
      encoded_compile_time_args.reserve(compile_time_args.size());
      for (uint32_t value : compile_time_args) {
        encoded_compile_time_args.push_back(static_cast<int64_t>(value));
      }
      writer->WriteObjectKeyValue("compile_time_args", encoded_compile_time_args);
    }
    if (!runtime_args.empty()) {
      writer->WriteObjectKeyValue("runtime_args", runtime_args);
    }
    if (!common_runtime_args.empty()) {
      writer->WriteObjectKeyValue("common_runtime_args", common_runtime_args);
    }
    if (!compile_time_arg_specs.empty()) {
      writer->WriteObjectKeyValue("compile_time_arg_specs", compile_time_arg_specs);
    }
    if (has_launch_spec) {
      writer->WriteObjectKeyValue("launch_spec", launch_spec);
    }
    if (has_compute_config) {
      writer->WriteObjectKeyValue("compute_config", compute_config);
    }
    if (!accessors.empty()) {
      writer->WriteObjectKeyValue("accessors", accessors);
    }
    writer->EndObject();
  }
};

struct GemmContractSpec {
  bool enabled = false;
  std::string a_buffer;
  std::string b_buffer;
  std::string c_buffer;
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  bool transpose_A = false;
  bool transpose_B = false;
  std::string a_tensor_dtype;
  std::string b_tensor_dtype;
  std::string c_tensor_dtype;
  std::string a_cb_dtype;
  std::string b_cb_dtype;
  std::string c_cb_dtype;
  std::string accumulator_dtype;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("enabled", enabled);
    writer->WriteObjectKeyValue("a_buffer", a_buffer);
    writer->WriteObjectKeyValue("b_buffer", b_buffer);
    writer->WriteObjectKeyValue("c_buffer", c_buffer);
    writer->WriteObjectKeyValue("M", static_cast<int64_t>(M));
    writer->WriteObjectKeyValue("N", static_cast<int64_t>(N));
    writer->WriteObjectKeyValue("K", static_cast<int64_t>(K));
    writer->WriteObjectKeyValue("transpose_A", transpose_A);
    writer->WriteObjectKeyValue("transpose_B", transpose_B);
    writer->WriteObjectKeyValue("a_tensor_dtype", a_tensor_dtype);
    writer->WriteObjectKeyValue("b_tensor_dtype", b_tensor_dtype);
    writer->WriteObjectKeyValue("c_tensor_dtype", c_tensor_dtype);
    writer->WriteObjectKeyValue("a_cb_dtype", a_cb_dtype);
    writer->WriteObjectKeyValue("b_cb_dtype", b_cb_dtype);
    writer->WriteObjectKeyValue("c_cb_dtype", c_cb_dtype);
    writer->WriteObjectKeyValue("accumulator_dtype", accumulator_dtype);
    writer->EndObject();
  }
};

struct ComputeContractSpec {
  bool enabled = false;
  std::string kind;
  std::string a_buffer;
  std::string b_buffer;
  std::string c_buffer;
  uint32_t M = 0;
  uint32_t N = 0;
  uint32_t K = 0;
  uint32_t Mt = 0;
  uint32_t Nt = 0;
  uint32_t Kt = 0;
  uint32_t block_m_tiles = 0;
  uint32_t block_n_tiles = 0;
  uint32_t block_k_tiles = 0;
  uint32_t subblock_m_tiles = 0;
  uint32_t subblock_n_tiles = 0;
  bool transpose_A = false;
  bool transpose_B = false;
  std::string a_tensor_dtype;
  std::string b_tensor_dtype;
  std::string c_tensor_dtype;
  std::string a_cb_dtype;
  std::string b_cb_dtype;
  std::string c_cb_dtype;
  std::string accumulator_dtype;
  std::string math_fidelity;
  bool fp32_dest_acc_en = false;
  bool math_approx_mode = false;
  std::vector<std::string> unpack_to_dest_mode;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("enabled", enabled);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("a_buffer", a_buffer);
    writer->WriteObjectKeyValue("b_buffer", b_buffer);
    writer->WriteObjectKeyValue("c_buffer", c_buffer);
    writer->WriteObjectKeyValue("M", static_cast<int64_t>(M));
    writer->WriteObjectKeyValue("N", static_cast<int64_t>(N));
    writer->WriteObjectKeyValue("K", static_cast<int64_t>(K));
    writer->WriteObjectKeyValue("Mt", static_cast<int64_t>(Mt));
    writer->WriteObjectKeyValue("Nt", static_cast<int64_t>(Nt));
    writer->WriteObjectKeyValue("Kt", static_cast<int64_t>(Kt));
    writer->WriteObjectKeyValue("block_m_tiles", static_cast<int64_t>(block_m_tiles));
    writer->WriteObjectKeyValue("block_n_tiles", static_cast<int64_t>(block_n_tiles));
    writer->WriteObjectKeyValue("block_k_tiles", static_cast<int64_t>(block_k_tiles));
    writer->WriteObjectKeyValue("subblock_m_tiles", static_cast<int64_t>(subblock_m_tiles));
    writer->WriteObjectKeyValue("subblock_n_tiles", static_cast<int64_t>(subblock_n_tiles));
    writer->WriteObjectKeyValue("transpose_A", transpose_A);
    writer->WriteObjectKeyValue("transpose_B", transpose_B);
    writer->WriteObjectKeyValue("a_tensor_dtype", a_tensor_dtype);
    writer->WriteObjectKeyValue("b_tensor_dtype", b_tensor_dtype);
    writer->WriteObjectKeyValue("c_tensor_dtype", c_tensor_dtype);
    writer->WriteObjectKeyValue("a_cb_dtype", a_cb_dtype);
    writer->WriteObjectKeyValue("b_cb_dtype", b_cb_dtype);
    writer->WriteObjectKeyValue("c_cb_dtype", c_cb_dtype);
    writer->WriteObjectKeyValue("accumulator_dtype", accumulator_dtype);
    writer->WriteObjectKeyValue("math_fidelity", math_fidelity);
    writer->WriteObjectKeyValue("fp32_dest_acc_en", fp32_dest_acc_en);
    writer->WriteObjectKeyValue("math_approx_mode", math_approx_mode);
    if (!unpack_to_dest_mode.empty()) {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", unpack_to_dest_mode);
    } else {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", std::vector<std::string>{});
    }
    writer->EndObject();
  }
};

/*!
 * \brief Stage 0 executable description for a lowered PrimFunc.
 */
struct ExecutableSpec {
  std::string entry_name;
  std::vector<CBConfig> cb_configs;
  CorePlan core_plan;
  std::string default_kernel_kind = "fused_dataflow";
  std::string default_kernel_core_type = "brisc";
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelSpec> kernels;
  GemmContractSpec gemm_contract;
  ComputeContractSpec compute_contract;

  // TVM runtime invocation metadata retained during Stage 0.
  std::vector<std::string> tvm_arg_names;
  std::vector<DLDataType> tvm_arg_types;
  std::vector<bool> tvm_is_buffer_arg;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("entry_name", entry_name);
    if (!cb_configs.empty()) {
      writer->WriteObjectKeyValue("cb_configs", cb_configs);
    }
    writer->WriteObjectKeyValue("core_plan", core_plan);
    writer->WriteObjectKeyValue("default_kernel_kind", default_kernel_kind);
    writer->WriteObjectKeyValue("default_kernel_core_type", default_kernel_core_type);
    if (!runtime_args.empty()) {
      writer->WriteObjectKeyValue("runtime_args", runtime_args);
    }
    if (!kernels.empty()) {
      writer->WriteObjectKeyValue("kernels", kernels);
    }
    writer->WriteObjectKeyValue("gemm_contract", gemm_contract);
    writer->WriteObjectKeyValue("compute_contract", compute_contract);
    if (!tvm_arg_names.empty()) {
      writer->WriteObjectKeyValue("tvm_arg_names", tvm_arg_names);
    }
    if (!tvm_arg_types.empty()) {
      std::vector<std::string> arg_types;
      arg_types.reserve(tvm_arg_types.size());
      for (const auto& dtype : tvm_arg_types) {
        arg_types.push_back(::tvm::runtime::DLDataTypeToString(dtype));
      }
      writer->WriteObjectKeyValue("tvm_arg_types", arg_types);
    }
    if (!tvm_is_buffer_arg.empty()) {
      std::vector<int64_t> is_buffer_arg;
      is_buffer_arg.reserve(tvm_is_buffer_arg.size());
      for (bool is_buffer : tvm_is_buffer_arg) {
        is_buffer_arg.push_back(is_buffer ? 1 : 0);
      }
      writer->WriteObjectKeyValue("tvm_is_buffer_arg", is_buffer_arg);
    }
    writer->EndObject();
  }
};

/*!
 * \brief Runtime tensor binding for direct Blackhole execution.
 */
struct RuntimeTensorBinding {
  std::string name;
  DLTensor* tensor = nullptr;
  bool is_output = false;
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

  /*! \brief Destructor */
  ~BlackholeModuleNode() = default;

  /*! \brief Return module kind */
  const char* kind() const final { return "blackhole"; }

  /*! \brief Get module properties */
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  /*! \brief Get function by name */
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  /*! \brief Get function metadata by name */
  ffi::Optional<ffi::String> GetFunctionMetadata(const ffi::String& name) final;

  /*! \brief Save to file (serialization) */
  void WriteToFile(const ffi::String& file_name, const ffi::String& format) const final;

  /*! \brief Save to bytes (serialization) */
  ffi::Bytes SaveToBytes() const final;

  /*! \brief Inspect source code */
  ffi::String InspectSource(const ffi::String& format) const final;

  /*! \brief Execute function using direct TT-Metal API (requires TILELANG_BLACKHOLE_DIRECT) */
  void ExecuteDirect(const std::string& func_name,
                     const std::vector<RuntimeTensorBinding>& buffer_args,
                     const std::vector<uint32_t>& scalar_args,
                     const std::vector<std::string>& output_names);

 private:
  // Function information map
  std::unordered_map<std::string, ExecutableSpec> fmap_;
  // Directory for kernel files
  std::string kernel_dir_;
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
