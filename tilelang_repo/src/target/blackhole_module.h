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

#include "../transform/common/blackhole_runtime_arg_schema.h"

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
  uint32_t initial_reserve_pages = 0;
  std::string flow_class = "state";
  uint32_t publish_pages_per_event = 0;
  uint32_t consume_pages_per_event = 0;
  std::string data_format;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("cb_id", static_cast<int64_t>(cb_id));
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("role", role);
    writer->WriteObjectKeyValue("num_pages", static_cast<int64_t>(num_pages));
    writer->WriteObjectKeyValue("page_size", static_cast<int64_t>(page_size_bytes));
    writer->WriteObjectKeyValue("initial_reserve_pages",
                                static_cast<int64_t>(initial_reserve_pages));
    writer->WriteObjectKeyValue("flow_class", flow_class);
    writer->WriteObjectKeyValue("publish_pages_per_event",
                                static_cast<int64_t>(publish_pages_per_event));
    writer->WriteObjectKeyValue("consume_pages_per_event",
                                static_cast<int64_t>(consume_pages_per_event));
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

struct CoreRangeSpec {
  PhysicalCore start;
  PhysicalCore end;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("start", start);
    writer->WriteObjectKeyValue("end", end);
    writer->EndObject();
  }
};

struct SemaphoreSpec {
  uint32_t id = 0;
  uint32_t initial_value = 0;
  std::string core_type = "worker";
  std::vector<CoreRangeSpec> core_ranges;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("id", static_cast<int64_t>(id));
    writer->WriteObjectKeyValue("initial_value", static_cast<int64_t>(initial_value));
    writer->WriteObjectKeyValue("core_type", core_type);
    writer->WriteObjectKeyValue("core_ranges", core_ranges);
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
  std::string identity;
  uint32_t core_x = 0;
  uint32_t core_y = 0;
  bool has_core_coord = false;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("dtype", dtype);
    if (!buffer.empty()) {
      writer->WriteObjectKeyValue("buffer", buffer);
    }
    writer->WriteObjectKeyValue("identity", identity);
    if (has_core_coord) {
      writer->WriteObjectKeyValue("core_x", static_cast<int64_t>(core_x));
      writer->WriteObjectKeyValue("core_y", static_cast<int64_t>(core_y));
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
  uint32_t args_config_bits = 0;
  uint32_t transport_page_size_bytes = 0;
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
    if (args_config_bits != 0) {
      writer->WriteObjectKeyValue("args_config_bits", static_cast<int64_t>(args_config_bits));
    }
    if (transport_page_size_bytes != 0) {
      writer->WriteObjectKeyValue("transport_page_size",
                                  static_cast<int64_t>(transport_page_size_bytes));
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

struct PerWorkArgSpec {
  std::string arg_kind;
  std::string arg_identity;
  std::string buffer;
  std::string value_kind;
  uint32_t constant_value = 0;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue(
        tl::blackhole_runtime_arg_schema::kArgKind, arg_kind);
    if (!arg_identity.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kArgIdentity, arg_identity);
    }
    if (!buffer.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kBuffer, buffer);
    }
    writer->WriteObjectKeyValue(
        tl::blackhole_runtime_arg_schema::kValueKind, value_kind);
    if (value_kind == tl::blackhole_runtime_arg_schema::kValueConstant) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kConstantValue,
          static_cast<int64_t>(constant_value));
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

struct KernelDefineSpec {
  std::string name;
  std::string value;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("value", value);
    writer->EndObject();
  }
};

struct NamedCompileArgSpec {
  std::string name;
  uint32_t value = 0;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("value", static_cast<int64_t>(value));
    writer->EndObject();
  }
};

struct KernelComputeConfigSpec {
  std::string math_fidelity;
  bool fp32_dest_acc_en = false;
  bool dst_full_sync_en = false;
  bool math_approx_mode = false;
  std::vector<std::string> unpack_to_dest_mode;
  bool bfp8_pack_precise = false;
  bool clear_accum = false;
  uint32_t k_pack = 1;
  int32_t wg_wait = 0;
  int32_t policy_type = 0;
  std::string policy_name;
  std::vector<KernelDefineSpec> defines;
  std::vector<NamedCompileArgSpec> named_compile_args;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("math_fidelity", math_fidelity);
    writer->WriteObjectKeyValue("fp32_dest_acc_en", fp32_dest_acc_en);
    writer->WriteObjectKeyValue("dst_full_sync_en", dst_full_sync_en);
    writer->WriteObjectKeyValue("math_approx_mode", math_approx_mode);
    writer->WriteObjectKeyValue("bfp8_pack_precise", bfp8_pack_precise);
    writer->WriteObjectKeyValue("clear_accum", clear_accum);
    writer->WriteObjectKeyValue("k_pack", static_cast<int64_t>(k_pack));
    writer->WriteObjectKeyValue("wg_wait", static_cast<int64_t>(wg_wait));
    writer->WriteObjectKeyValue("policy_type", static_cast<int64_t>(policy_type));
    writer->WriteObjectKeyValue("policy_name", policy_name);
    if (!defines.empty()) {
      writer->WriteObjectKeyValue("defines", defines);
    } else {
      writer->WriteObjectKeyValue("defines", std::vector<KernelDefineSpec>{});
    }
    if (!named_compile_args.empty()) {
      writer->WriteObjectKeyValue("named_compile_args", named_compile_args);
    } else {
      writer->WriteObjectKeyValue("named_compile_args", std::vector<NamedCompileArgSpec>{});
    }
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
  uint32_t transport_page_size_bytes = 0;
  std::string layout;
  std::string memory_space;
  std::vector<int64_t> host_axis_order;
  bool transpose_2d = false;

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
    if (transport_page_size_bytes != 0) {
      writer->WriteObjectKeyValue("transport_page_size",
                                  static_cast<int64_t>(transport_page_size_bytes));
    }
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("memory_space", memory_space);
    if (!host_axis_order.empty()) {
      writer->WriteObjectKeyValue("host_axis_order", host_axis_order);
    }
    if (transpose_2d) {
      writer->WriteObjectKeyValue("transpose_2d", transpose_2d);
    }
    writer->EndObject();
  }
};

struct SemaphoreBindingSpec {
  std::string name;
  uint32_t semaphore_id = 0;
  std::string arg_kind;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("semaphore_id", static_cast<int64_t>(semaphore_id));
    writer->WriteObjectKeyValue("arg_kind", arg_kind);
    writer->EndObject();
  }
};

struct RemoteCoreDescriptorSpec {
  std::string identity;
  uint32_t core_x = 0;
  uint32_t core_y = 0;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("identity", identity);
    writer->WriteObjectKeyValue("core_x", static_cast<int64_t>(core_x));
    writer->WriteObjectKeyValue("core_y", static_cast<int64_t>(core_y));
    writer->EndObject();
  }
};

struct BufferMaterializationSpec {
  std::string buffer;
  std::string materialization_kind = "replicated";
  std::string layout;
  std::string memory_space;
  uint32_t transport_page_size_bytes = 0;
  std::vector<int64_t> host_axis_order;
  bool transpose_2d = false;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("buffer", buffer);
    writer->WriteObjectKeyValue("materialization_kind", materialization_kind);
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("memory_space", memory_space);
    writer->WriteObjectKeyValue("transport_page_size",
                                static_cast<int64_t>(transport_page_size_bytes));
    if (!host_axis_order.empty()) {
      writer->WriteObjectKeyValue("host_axis_order", host_axis_order);
    }
    if (transpose_2d) {
      writer->WriteObjectKeyValue("transpose_2d", transpose_2d);
    }
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
  std::vector<PerWorkArgSpec> per_work_arg_specs;
  bool has_launch_spec = false;
  KernelLaunchSpec launch_spec;
  bool has_compute_config = false;
  KernelComputeConfigSpec compute_config;
  std::vector<AccessorSpec> accessors;
  std::vector<SemaphoreBindingSpec> semaphore_bindings;
  std::vector<RemoteCoreDescriptorSpec> remote_core_descriptors;

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
    if (!per_work_arg_specs.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs, per_work_arg_specs);
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
    if (!semaphore_bindings.empty()) {
      writer->WriteObjectKeyValue("semaphore_bindings", semaphore_bindings);
    }
    if (!remote_core_descriptors.empty()) {
      writer->WriteObjectKeyValue("remote_core_descriptors", remote_core_descriptors);
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
  struct EpilogueOpSpec {
    struct FragmentMaterializationContractSpec {
      std::string kind;
      std::string target_buffer;
      std::string scope;
      std::string materialization_kind;
      std::string bridge_kind;
      std::string value_role;
      std::string merge_kind;
      std::string execution_protocol;
      std::string result_live_form;
      std::string source_buffer;
      int logical_row_width = 0;
      int logical_element_count = 0;

      bool defined() const { return !kind.empty(); }

      void Save(dmlc::JSONWriter* writer) const {
        writer->BeginObject();
        writer->WriteObjectKeyValue("kind", kind);
        if (!target_buffer.empty()) writer->WriteObjectKeyValue("target_buffer", target_buffer);
        if (!scope.empty()) writer->WriteObjectKeyValue("scope", scope);
        if (!materialization_kind.empty()) {
          writer->WriteObjectKeyValue("materialization_kind", materialization_kind);
        }
        if (!bridge_kind.empty()) writer->WriteObjectKeyValue("bridge_kind", bridge_kind);
        if (!value_role.empty()) writer->WriteObjectKeyValue("value_role", value_role);
        if (!merge_kind.empty()) writer->WriteObjectKeyValue("merge_kind", merge_kind);
        if (!execution_protocol.empty()) {
          writer->WriteObjectKeyValue("execution_protocol", execution_protocol);
        }
        if (!result_live_form.empty()) {
          writer->WriteObjectKeyValue("result_live_form", result_live_form);
        }
        if (!source_buffer.empty()) {
          writer->WriteObjectKeyValue("source_buffer", source_buffer);
        }
        if (logical_row_width > 0) {
          writer->WriteObjectKeyValue("logical_row_width", logical_row_width);
        }
        if (logical_element_count > 0) {
          writer->WriteObjectKeyValue("logical_element_count", logical_element_count);
        }
        writer->EndObject();
      }
    };

    std::string kind;
    std::string dst_buffer;
    std::string src_buffer;
    std::string scalar_buffer;
    std::string lhs_buffer;
    std::string rhs_buffer;
    std::string add_buffer;
    std::string reduce_kind;
    std::string num_elements_expr;
    std::string row_width_expr;
    std::string dst_offset_expr;
    std::string src_offset_expr;
    std::string dst_scale_expr;
    std::string scalar_scale_expr;
    std::string lhs_scale_expr;
    std::string rhs_scale_expr;
    bool grouped = false;
    bool clear = false;
    bool publish_cb = false;
    FragmentMaterializationContractSpec fragment_materialization_contract;

    void Save(dmlc::JSONWriter* writer) const {
      writer->BeginObject();
      writer->WriteObjectKeyValue("kind", kind);
      if (!dst_buffer.empty()) writer->WriteObjectKeyValue("dst_buffer", dst_buffer);
      if (!src_buffer.empty()) writer->WriteObjectKeyValue("src_buffer", src_buffer);
      if (!scalar_buffer.empty()) writer->WriteObjectKeyValue("scalar_buffer", scalar_buffer);
      if (!lhs_buffer.empty()) writer->WriteObjectKeyValue("lhs_buffer", lhs_buffer);
      if (!rhs_buffer.empty()) writer->WriteObjectKeyValue("rhs_buffer", rhs_buffer);
      if (!add_buffer.empty()) writer->WriteObjectKeyValue("add_buffer", add_buffer);
      if (!reduce_kind.empty()) writer->WriteObjectKeyValue("reduce_kind", reduce_kind);
      if (!num_elements_expr.empty()) {
        writer->WriteObjectKeyValue("num_elements_expr", num_elements_expr);
      }
      if (!row_width_expr.empty()) writer->WriteObjectKeyValue("row_width_expr", row_width_expr);
      if (!dst_offset_expr.empty()) writer->WriteObjectKeyValue("dst_offset_expr", dst_offset_expr);
      if (!src_offset_expr.empty()) writer->WriteObjectKeyValue("src_offset_expr", src_offset_expr);
      if (!dst_scale_expr.empty()) writer->WriteObjectKeyValue("dst_scale_expr", dst_scale_expr);
      if (!scalar_scale_expr.empty()) {
        writer->WriteObjectKeyValue("scalar_scale_expr", scalar_scale_expr);
      }
      if (!lhs_scale_expr.empty()) writer->WriteObjectKeyValue("lhs_scale_expr", lhs_scale_expr);
      if (!rhs_scale_expr.empty()) writer->WriteObjectKeyValue("rhs_scale_expr", rhs_scale_expr);
      if (grouped) writer->WriteObjectKeyValue("grouped", grouped);
      if (clear) writer->WriteObjectKeyValue("clear", clear);
      if (publish_cb) writer->WriteObjectKeyValue("publish_cb", publish_cb);
      if (fragment_materialization_contract.defined()) {
        writer->WriteObjectKeyValue("fragment_materialization_contract",
                                    fragment_materialization_contract);
      }
      writer->EndObject();
    }
  };

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
  bool dst_full_sync_en = false;
  bool math_approx_mode = false;
  std::vector<std::string> unpack_to_dest_mode;
  bool bfp8_pack_precise = false;
  bool clear_accum = false;
  uint32_t k_pack = 1;
  int32_t wg_wait = 0;
  int32_t policy_type = 0;
  std::string policy_name;
  std::vector<KernelDefineSpec> defines;
  std::vector<NamedCompileArgSpec> named_compile_args;
  bool has_mbarrier = false;
  std::string mbarrier_buffer;
  std::string mbarrier_scope;
  std::vector<std::string> mbarrier_index_exprs;
  std::vector<EpilogueOpSpec> epilogue_ops;

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
    writer->WriteObjectKeyValue("dst_full_sync_en", dst_full_sync_en);
    writer->WriteObjectKeyValue("math_approx_mode", math_approx_mode);
    writer->WriteObjectKeyValue("bfp8_pack_precise", bfp8_pack_precise);
    writer->WriteObjectKeyValue("clear_accum", clear_accum);
    writer->WriteObjectKeyValue("k_pack", static_cast<int64_t>(k_pack));
    writer->WriteObjectKeyValue("wg_wait", static_cast<int64_t>(wg_wait));
    writer->WriteObjectKeyValue("policy_type", static_cast<int64_t>(policy_type));
    writer->WriteObjectKeyValue("policy_name", policy_name);
    if (!defines.empty()) {
      writer->WriteObjectKeyValue("defines", defines);
    } else {
      writer->WriteObjectKeyValue("defines", std::vector<KernelDefineSpec>{});
    }
    if (!named_compile_args.empty()) {
      writer->WriteObjectKeyValue("named_compile_args", named_compile_args);
    } else {
      writer->WriteObjectKeyValue("named_compile_args", std::vector<NamedCompileArgSpec>{});
    }
    writer->WriteObjectKeyValue("has_mbarrier", has_mbarrier);
    writer->WriteObjectKeyValue("mbarrier_buffer", mbarrier_buffer);
    writer->WriteObjectKeyValue("mbarrier_scope", mbarrier_scope);
    if (!mbarrier_index_exprs.empty()) {
      writer->WriteObjectKeyValue("mbarrier_index_exprs", mbarrier_index_exprs);
    } else {
      writer->WriteObjectKeyValue("mbarrier_index_exprs", std::vector<std::string>{});
    }
    if (!unpack_to_dest_mode.empty()) {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", unpack_to_dest_mode);
    } else {
      writer->WriteObjectKeyValue("unpack_to_dest_mode", std::vector<std::string>{});
    }
    if (!epilogue_ops.empty()) {
      writer->WriteObjectKeyValue("epilogue_ops", epilogue_ops);
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
  std::vector<SemaphoreSpec> semaphores;
  std::vector<BufferMaterializationSpec> buffer_materializations;
  std::string default_kernel_kind = "fused_dataflow";
  std::string default_kernel_core_type = "brisc";
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<PerWorkArgSpec> per_work_arg_specs;
  std::vector<KernelSpec> kernels;
  GemmContractSpec gemm_contract;
  ComputeContractSpec compute_contract;
  std::vector<GemmContractSpec> multi_gemm_contracts;
  std::vector<ComputeContractSpec> multi_compute_contracts;
  std::vector<ComputeContractSpec::EpilogueOpSpec> compute_epilogue_ops;
  std::vector<std::string> direct_runtime_unsupported_reasons;

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
    if (!semaphores.empty()) {
      writer->WriteObjectKeyValue("semaphores", semaphores);
    }
    if (!buffer_materializations.empty()) {
      writer->WriteObjectKeyValue("buffer_materializations", buffer_materializations);
    }
    writer->WriteObjectKeyValue("default_kernel_kind", default_kernel_kind);
    writer->WriteObjectKeyValue("default_kernel_core_type", default_kernel_core_type);
    if (!runtime_args.empty()) {
      writer->WriteObjectKeyValue("runtime_args", runtime_args);
    }
    if (!common_runtime_args.empty()) {
      writer->WriteObjectKeyValue("common_runtime_args", common_runtime_args);
    }
    if (!per_work_arg_specs.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs, per_work_arg_specs);
    }
    if (!kernels.empty()) {
      writer->WriteObjectKeyValue("kernels", kernels);
    }
    writer->WriteObjectKeyValue("gemm_contract", gemm_contract);
    writer->WriteObjectKeyValue("compute_contract", compute_contract);
    if (!multi_gemm_contracts.empty()) {
      writer->WriteObjectKeyValue("multi_gemm_contracts", multi_gemm_contracts);
    }
    if (!multi_compute_contracts.empty()) {
      writer->WriteObjectKeyValue("multi_compute_contracts", multi_compute_contracts);
    }
    if (!compute_epilogue_ops.empty()) {
      writer->WriteObjectKeyValue("compute_epilogue_ops", compute_epilogue_ops);
    }
    if (!direct_runtime_unsupported_reasons.empty()) {
      writer->WriteObjectKeyValue("direct_runtime_unsupported_reasons",
                                  direct_runtime_unsupported_reasons);
    }
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
