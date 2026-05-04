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
  std::vector<std::string> requirement_names;
  std::vector<int64_t> requirement_indices;

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
    if (!requirement_names.empty()) {
      writer->WriteObjectKeyValue("requirement_names", requirement_names);
    }
    if (!requirement_indices.empty()) {
      writer->WriteObjectKeyValue("requirement_indices", requirement_indices);
    }
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
  uint32_t logical_grid_z = 1;
  std::string linearization = "row_major";
  std::vector<PhysicalCore> physical_cores;
  std::vector<WorkPacket> work_packets;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("logical_grid_x", static_cast<int64_t>(logical_grid_x));
    writer->WriteObjectKeyValue("logical_grid_y", static_cast<int64_t>(logical_grid_y));
    writer->WriteObjectKeyValue("logical_grid_z", static_cast<int64_t>(logical_grid_z));
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
  std::vector<int64_t> host_axis_order;
  bool transpose_2d = false;

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
    if (!host_axis_order.empty()) {
      writer->WriteObjectKeyValue("host_axis_order", host_axis_order);
    }
    if (transpose_2d) {
      writer->WriteObjectKeyValue("transpose_2d", transpose_2d);
    }
    writer->EndObject();
  }
};

struct PerWorkArgSpec {
  std::string arg_kind;
  std::string arg_identity;
  std::string buffer;
  std::string descriptor_kind;
  std::string value_source;
  uint32_t constant_value = 0;
  std::string access_region;
  int64_t access_region_index = -1;
  std::string index_buffer;
  int64_t index_value_scale = 1;

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
    if (!descriptor_kind.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kDescriptorKind, descriptor_kind);
    }
    if (!value_source.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kValueSource, value_source);
    }
    if (value_source == tl::blackhole_runtime_arg_schema::kValueSourceConstant) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kConstantValue,
          static_cast<int64_t>(constant_value));
    }
    if (value_source == tl::blackhole_runtime_arg_schema::kValueSourceIndexTable) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kIndexBuffer, index_buffer);
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kIndexValueScale,
          static_cast<int64_t>(index_value_scale));
    }
    if (!access_region.empty()) {
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kAccessRegion, access_region);
      writer->WriteObjectKeyValue(
          tl::blackhole_runtime_arg_schema::kAccessRegionIndex,
          access_region_index);
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

struct ComputeOperandBindingSpec {
  std::string role;
  std::string buffer;
  std::string host_buffer;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("role", role);
    writer->WriteObjectKeyValue("buffer", buffer);
    if (!host_buffer.empty()) {
      writer->WriteObjectKeyValue("host_buffer", host_buffer);
    }
    writer->EndObject();
  }
};

struct KernelComputeOpSpec {
  bool enabled = false;
  std::string kind;
  std::string operation_name;
  std::string a_buffer;
  std::string b_buffer;
  std::string c_buffer;
  std::vector<ComputeOperandBindingSpec> operand_bindings;
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
  bool has_mbarrier = false;
  std::string mbarrier_buffer;
  std::string mbarrier_scope;
  std::vector<std::string> mbarrier_index_exprs;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("enabled", enabled);
    writer->WriteObjectKeyValue("kind", kind);
    writer->WriteObjectKeyValue("operation_name", operation_name);
    writer->WriteObjectKeyValue("a_buffer", a_buffer);
    writer->WriteObjectKeyValue("b_buffer", b_buffer);
    writer->WriteObjectKeyValue("c_buffer", c_buffer);
    writer->WriteObjectKeyValue("operand_bindings", operand_bindings);
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
    writer->WriteObjectKeyValue("has_mbarrier", has_mbarrier);
    writer->WriteObjectKeyValue("mbarrier_buffer", mbarrier_buffer);
    writer->WriteObjectKeyValue("mbarrier_scope", mbarrier_scope);
    if (!mbarrier_index_exprs.empty()) {
      writer->WriteObjectKeyValue("mbarrier_index_exprs", mbarrier_index_exprs);
    } else {
      writer->WriteObjectKeyValue("mbarrier_index_exprs", std::vector<std::string>{});
    }
    writer->EndObject();
  }
};

struct AccessorSpec {
  std::string buffer;
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
  std::string live_form_kind;
  std::string execution_topology_kind;
  uint32_t physical_local_extent = 0;
  uint32_t logical_element_count = 0;
  std::string producer_kernel;
  std::string materialization_protocol;
  std::string publication_protocol;

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
    if (!live_form_kind.empty()) {
      writer->WriteObjectKeyValue("live_form_kind", live_form_kind);
    }
    if (!execution_topology_kind.empty()) {
      writer->WriteObjectKeyValue("execution_topology_kind", execution_topology_kind);
    }
    if (physical_local_extent > 0) {
      writer->WriteObjectKeyValue("physical_local_extent",
                                  static_cast<int64_t>(physical_local_extent));
    }
    if (logical_element_count > 0) {
      writer->WriteObjectKeyValue("logical_element_count",
                                  static_cast<int64_t>(logical_element_count));
    }
    if (!producer_kernel.empty()) {
      writer->WriteObjectKeyValue("producer_kernel", producer_kernel);
    }
    if (!materialization_protocol.empty()) {
      writer->WriteObjectKeyValue("materialization_protocol", materialization_protocol);
    }
    if (!publication_protocol.empty()) {
      writer->WriteObjectKeyValue("publication_protocol", publication_protocol);
    }
    writer->EndObject();
  }
};

struct BufferDistributionSpec {
  std::string name;
  std::string buffer;
  std::string mesh_plan;
  int64_t mesh_plan_index = -1;
  std::string distribution_kind;
  std::string layout;
  std::string memory_space;
  uint32_t page_size_bytes = 0;
  std::vector<int64_t> shard_shape;
  std::vector<int64_t> shard_grid_shape;
  std::string sharding_strategy;
  std::string shard_orientation;
  std::string source_buffer;
  std::string source_region_kind;
  std::vector<int64_t> source_region_shape;
  std::string logical_index_mapping;
  std::string core_local_address_mapping;
  std::string host_visibility;
  std::string attached_core_group;
  int64_t attached_core_group_index = -1;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("buffer", buffer);
    writer->WriteObjectKeyValue("mesh_plan", mesh_plan);
    writer->WriteObjectKeyValue("mesh_plan_index", mesh_plan_index);
    writer->WriteObjectKeyValue("distribution_kind", distribution_kind);
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("memory_space", memory_space);
    writer->WriteObjectKeyValue("page_size_bytes",
                                static_cast<int64_t>(page_size_bytes));
    if (!shard_shape.empty()) {
      writer->WriteObjectKeyValue("shard_shape", shard_shape);
    }
    if (!shard_grid_shape.empty()) {
      writer->WriteObjectKeyValue("shard_grid_shape", shard_grid_shape);
    }
    writer->WriteObjectKeyValue("sharding_strategy", sharding_strategy);
    writer->WriteObjectKeyValue("shard_orientation", shard_orientation);
    if (!source_buffer.empty()) {
      writer->WriteObjectKeyValue("source_buffer", source_buffer);
    }
    writer->WriteObjectKeyValue("source_region_kind", source_region_kind);
    if (!source_region_shape.empty()) {
      writer->WriteObjectKeyValue("source_region_shape", source_region_shape);
    }
    writer->WriteObjectKeyValue("logical_index_mapping", logical_index_mapping);
    writer->WriteObjectKeyValue("core_local_address_mapping",
                                core_local_address_mapping);
    writer->WriteObjectKeyValue("host_visibility", host_visibility);
    if (!attached_core_group.empty()) {
      writer->WriteObjectKeyValue("attached_core_group", attached_core_group);
      writer->WriteObjectKeyValue("attached_core_group_index",
                                  attached_core_group_index);
    }
    writer->EndObject();
  }
};

struct TensorMemoryConfigSpec {
  std::string name;
  std::string subject;
  std::string value_identity;
  std::vector<int64_t> logical_shape;
  std::string dtype;
  std::string memory_layout;
  std::string buffer_type;
  std::string grid_ref;
  std::vector<int64_t> shard_grid_shape;
  std::vector<int64_t> shard_shape;
  std::string shard_orientation;
  std::string shard_distribution_strategy;
  std::vector<int64_t> page_shape;
  std::string origin;
  std::string source_buffer;
  std::string buffer_distribution_plan;
  int64_t buffer_distribution_plan_index = -1;
  bool has_runtime_accessor = false;
  bool requires_materialization = false;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("subject", subject);
    if (!value_identity.empty()) {
      writer->WriteObjectKeyValue("value_identity", value_identity);
    }
    if (!logical_shape.empty()) {
      writer->WriteObjectKeyValue("logical_shape", logical_shape);
    }
    if (!dtype.empty()) {
      writer->WriteObjectKeyValue("dtype", dtype);
    }
    writer->WriteObjectKeyValue("memory_layout", memory_layout);
    writer->WriteObjectKeyValue("buffer_type", buffer_type);
    if (!grid_ref.empty()) {
      writer->WriteObjectKeyValue("grid_ref", grid_ref);
    }
    if (!shard_grid_shape.empty()) {
      writer->WriteObjectKeyValue("shard_grid_shape", shard_grid_shape);
    }
    if (!shard_shape.empty()) {
      writer->WriteObjectKeyValue("shard_shape", shard_shape);
    }
    writer->WriteObjectKeyValue("shard_orientation", shard_orientation);
    writer->WriteObjectKeyValue("shard_distribution_strategy",
                                shard_distribution_strategy);
    if (!page_shape.empty()) {
      writer->WriteObjectKeyValue("page_shape", page_shape);
    }
    writer->WriteObjectKeyValue("origin", origin);
    if (!source_buffer.empty()) {
      writer->WriteObjectKeyValue("source_buffer", source_buffer);
    }
    writer->WriteObjectKeyValue("buffer_distribution_plan",
                                buffer_distribution_plan);
    writer->WriteObjectKeyValue("buffer_distribution_plan_index",
                                buffer_distribution_plan_index);
    writer->WriteObjectKeyValue("has_runtime_accessor", has_runtime_accessor);
    writer->WriteObjectKeyValue("requires_materialization",
                                requires_materialization);
    writer->EndObject();
  }
};

struct ReshardPlanSpec {
  std::string name;
  std::string source_value;
  std::string target_value;
  std::string source_memory_config_plan;
  int64_t source_memory_config_plan_index = -1;
  std::string target_memory_config_plan;
  int64_t target_memory_config_plan_index = -1;
  std::string conversion_kind;
  std::string source_region_kind;
  std::vector<int64_t> source_region_shape;
  std::string materialization_plan;
  int64_t materialization_plan_index = -1;
  std::string materialization_protocol;
  std::vector<int64_t> required_cb_plan_indices;
  std::vector<int64_t> required_sync_plan_indices;
  std::string scheduling_kind;
  std::string inserted_by;
  std::string admission_status;
  std::string unsupported_reason;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("source_value", source_value);
    writer->WriteObjectKeyValue("target_value", target_value);
    writer->WriteObjectKeyValue("source_memory_config_plan",
                                source_memory_config_plan);
    writer->WriteObjectKeyValue("source_memory_config_plan_index",
                                source_memory_config_plan_index);
    writer->WriteObjectKeyValue("target_memory_config_plan",
                                target_memory_config_plan);
    writer->WriteObjectKeyValue("target_memory_config_plan_index",
                                target_memory_config_plan_index);
    writer->WriteObjectKeyValue("conversion_kind", conversion_kind);
    writer->WriteObjectKeyValue("source_region_kind", source_region_kind);
    if (!source_region_shape.empty()) {
      writer->WriteObjectKeyValue("source_region_shape", source_region_shape);
    }
    if (!materialization_plan.empty()) {
      writer->WriteObjectKeyValue("materialization_plan", materialization_plan);
    }
    writer->WriteObjectKeyValue("materialization_plan_index",
                                materialization_plan_index);
    writer->WriteObjectKeyValue("materialization_protocol",
                                materialization_protocol);
    writer->WriteObjectKeyValue("required_cb_plan_indices",
                                required_cb_plan_indices);
    if (!required_sync_plan_indices.empty()) {
      writer->WriteObjectKeyValue("required_sync_plan_indices",
                                  required_sync_plan_indices);
    }
    writer->WriteObjectKeyValue("scheduling_kind", scheduling_kind);
    writer->WriteObjectKeyValue("inserted_by", inserted_by);
    writer->WriteObjectKeyValue("admission_status", admission_status);
    writer->WriteObjectKeyValue("unsupported_reason", unsupported_reason);
    writer->EndObject();
  }
};

struct LiveFormPlanSpec {
  std::string name;
  std::string logical_value;
  std::string spatial_live_value;
  int64_t spatial_live_value_index = -1;
  std::string producer_kernel;
  std::string physical_form;
  std::string execution_topology;
  uint32_t physical_local_extent = 0;
  uint32_t logical_element_count = 0;
  std::string ownership_kind;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("logical_value", logical_value);
    writer->WriteObjectKeyValue("spatial_live_value", spatial_live_value);
    writer->WriteObjectKeyValue("spatial_live_value_index", spatial_live_value_index);
    writer->WriteObjectKeyValue("producer_kernel", producer_kernel);
    writer->WriteObjectKeyValue("physical_form", physical_form);
    writer->WriteObjectKeyValue("execution_topology", execution_topology);
    writer->WriteObjectKeyValue("physical_local_extent",
                                static_cast<int64_t>(physical_local_extent));
    writer->WriteObjectKeyValue("logical_element_count",
                                static_cast<int64_t>(logical_element_count));
    writer->WriteObjectKeyValue("ownership_kind", ownership_kind);
    writer->EndObject();
  }
};

struct MaterializationPlanSpec {
  std::string name;
  std::string source_live_form;
  std::string materialization_boundary;
  int64_t materialization_boundary_index = -1;
  std::string target_buffer;
  std::string host_buffer;
  std::string target_kernel;
  std::string bridge_kind;
  std::string materialization_kind;
  std::string materialization_protocol;
  std::string publication_protocol;
  std::vector<int64_t> required_cb_plan_indices;
  std::vector<int64_t> required_sync_plan_indices;
  std::string produced_live_form;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("source_live_form", source_live_form);
    writer->WriteObjectKeyValue("materialization_boundary", materialization_boundary);
    writer->WriteObjectKeyValue("materialization_boundary_index", materialization_boundary_index);
    writer->WriteObjectKeyValue("target_buffer", target_buffer);
    writer->WriteObjectKeyValue("host_buffer", host_buffer);
    writer->WriteObjectKeyValue("target_kernel", target_kernel);
    if (!bridge_kind.empty()) {
      writer->WriteObjectKeyValue("bridge_kind", bridge_kind);
    }
    if (!materialization_kind.empty()) {
      writer->WriteObjectKeyValue("materialization_kind", materialization_kind);
    }
    writer->WriteObjectKeyValue("materialization_protocol", materialization_protocol);
    writer->WriteObjectKeyValue("publication_protocol", publication_protocol);
    writer->WriteObjectKeyValue("required_cb_plan_indices", required_cb_plan_indices);
    if (!required_sync_plan_indices.empty()) {
      writer->WriteObjectKeyValue("required_sync_plan_indices", required_sync_plan_indices);
    }
    writer->WriteObjectKeyValue("produced_live_form", produced_live_form);
    writer->EndObject();
  }
};

struct ConsumerBindingPlanSpec {
  std::string name;
  std::string consumer_kernel;
  std::string consumer_op_kind;
  std::string source_live_form;
  std::string live_value_edge;
  int64_t live_value_edge_index = -1;
  bool accepts_distributed_slice = false;
  bool requires_full_logical_tile = false;
  int64_t abi_plan_index = -1;
  std::string target_buffer;
  std::string materialization_plan;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("consumer_kernel", consumer_kernel);
    writer->WriteObjectKeyValue("consumer_op_kind", consumer_op_kind);
    writer->WriteObjectKeyValue("source_live_form", source_live_form);
    if (!target_buffer.empty()) {
      writer->WriteObjectKeyValue("target_buffer", target_buffer);
    }
    if (!materialization_plan.empty()) {
      writer->WriteObjectKeyValue("materialization_plan", materialization_plan);
    }
    writer->WriteObjectKeyValue("live_value_edge", live_value_edge);
    writer->WriteObjectKeyValue("live_value_edge_index", live_value_edge_index);
    writer->WriteObjectKeyValue("accepts_distributed_slice", accepts_distributed_slice);
    writer->WriteObjectKeyValue("requires_full_logical_tile", requires_full_logical_tile);
    if (abi_plan_index >= 0) {
      writer->WriteObjectKeyValue("abi_plan_index", abi_plan_index);
    }
    writer->EndObject();
  }
};

struct ExactCBVirtualValueSpec {
  std::string name;
  std::string logical_value;
  std::string live_form;
  int64_t live_form_index = -1;
  std::string producer_kernel;
  std::string producer_event;
  std::string event_lifetime_kind;
  std::string loop_role;
  uint32_t num_pages = 0;
  uint32_t page_size_bytes = 0;
  std::string data_format;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("logical_value", logical_value);
    writer->WriteObjectKeyValue("live_form", live_form);
    writer->WriteObjectKeyValue("live_form_index", live_form_index);
    writer->WriteObjectKeyValue("producer_kernel", producer_kernel);
    writer->WriteObjectKeyValue("producer_event", producer_event);
    writer->WriteObjectKeyValue("event_lifetime_kind", event_lifetime_kind);
    writer->WriteObjectKeyValue("loop_role", loop_role);
    writer->WriteObjectKeyValue("num_pages", static_cast<int64_t>(num_pages));
    writer->WriteObjectKeyValue("page_size_bytes",
                                static_cast<int64_t>(page_size_bytes));
    writer->WriteObjectKeyValue("data_format", data_format);
    writer->EndObject();
  }
};

struct ExactCBUseEventSpec {
  std::string name;
  std::string virtual_value;
  int64_t virtual_value_index = -1;
  std::string consumer_kernel;
  std::string consumer_event;
  std::string operand_role;
  int64_t program_point = -1;
  bool requires_full_logical_tile = false;
  std::string borrow_kind;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("virtual_value", virtual_value);
    writer->WriteObjectKeyValue("virtual_value_index", virtual_value_index);
    writer->WriteObjectKeyValue("consumer_kernel", consumer_kernel);
    writer->WriteObjectKeyValue("consumer_event", consumer_event);
    writer->WriteObjectKeyValue("operand_role", operand_role);
    writer->WriteObjectKeyValue("program_point", program_point);
    writer->WriteObjectKeyValue("requires_full_logical_tile",
                                requires_full_logical_tile);
    writer->WriteObjectKeyValue("borrow_kind", borrow_kind);
    writer->EndObject();
  }
};

struct ExactCBLiveIntervalSpec {
  std::string name;
  std::string virtual_value;
  int64_t virtual_value_index = -1;
  int64_t begin_point = -1;
  int64_t end_point = -1;
  bool live_in = false;
  bool live_out = false;
  bool loop_carried = false;
  std::string interference_class;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("virtual_value", virtual_value);
    writer->WriteObjectKeyValue("virtual_value_index", virtual_value_index);
    writer->WriteObjectKeyValue("begin_point", begin_point);
    writer->WriteObjectKeyValue("end_point", end_point);
    writer->WriteObjectKeyValue("live_in", live_in);
    writer->WriteObjectKeyValue("live_out", live_out);
    writer->WriteObjectKeyValue("loop_carried", loop_carried);
    writer->WriteObjectKeyValue("interference_class", interference_class);
    writer->EndObject();
  }
};

struct ExactCBAllocationSpec {
  std::string name;
  std::string virtual_value;
  int64_t virtual_value_index = -1;
  std::string cb_plan;
  int64_t cb_plan_index = -1;
  uint32_t physical_cb_id = 0;
  uint32_t page_count = 0;
  int64_t release_program_point = -1;
  std::string release_reason;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("virtual_value", virtual_value);
    writer->WriteObjectKeyValue("virtual_value_index", virtual_value_index);
    writer->WriteObjectKeyValue("cb_plan", cb_plan);
    writer->WriteObjectKeyValue("cb_plan_index", cb_plan_index);
    writer->WriteObjectKeyValue("physical_cb_id",
                                static_cast<int64_t>(physical_cb_id));
    writer->WriteObjectKeyValue("page_count", static_cast<int64_t>(page_count));
    writer->WriteObjectKeyValue("release_program_point",
                                release_program_point);
    writer->WriteObjectKeyValue("release_reason", release_reason);
    writer->EndObject();
  }
};

struct ExactCBReleaseEventSpec {
  std::string name;
  std::string allocation;
  int64_t allocation_index = -1;
  std::string cb_plan;
  int64_t cb_plan_index = -1;
  int64_t program_point = -1;
  uint32_t page_count = 0;
  std::string reason;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("allocation", allocation);
    writer->WriteObjectKeyValue("allocation_index", allocation_index);
    writer->WriteObjectKeyValue("cb_plan", cb_plan);
    writer->WriteObjectKeyValue("cb_plan_index", cb_plan_index);
    writer->WriteObjectKeyValue("program_point", program_point);
    writer->WriteObjectKeyValue("page_count", static_cast<int64_t>(page_count));
    writer->WriteObjectKeyValue("reason", reason);
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
  std::vector<KernelComputeOpSpec> compute_ops;
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
    if (!compute_ops.empty()) {
      writer->WriteObjectKeyValue("compute_ops", compute_ops);
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

/*!
 * \brief Stage 0 executable description for a lowered PrimFunc.
 */
struct ExecutableSpec {
  std::string entry_name;
  std::vector<CBConfig> cb_configs;
  CorePlan core_plan;
  std::vector<SemaphoreSpec> semaphores;
  std::vector<BufferDistributionSpec> buffer_distribution_plans;
  std::vector<TensorMemoryConfigSpec> tensor_memory_config_plans;
  std::vector<ReshardPlanSpec> reshard_plans;
  std::vector<BufferMaterializationSpec> buffer_materializations;
  std::vector<KernelArgSpec> runtime_args;
  std::vector<KernelArgSpec> common_runtime_args;
  std::vector<PerWorkArgSpec> per_work_arg_specs;
  std::vector<KernelSpec> kernels;
  std::vector<LiveFormPlanSpec> live_form_plans;
  std::vector<MaterializationPlanSpec> materialization_plans;
  std::vector<ConsumerBindingPlanSpec> consumer_binding_plans;
  std::vector<ExactCBVirtualValueSpec> exact_cb_virtual_values;
  std::vector<ExactCBUseEventSpec> exact_cb_use_events;
  std::vector<ExactCBLiveIntervalSpec> exact_cb_live_intervals;
  std::vector<ExactCBAllocationSpec> exact_cb_allocations;
  std::vector<ExactCBReleaseEventSpec> exact_cb_release_events;
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
    if (!buffer_distribution_plans.empty()) {
      writer->WriteObjectKeyValue("buffer_distribution_plans",
                                  buffer_distribution_plans);
    }
    if (!tensor_memory_config_plans.empty()) {
      writer->WriteObjectKeyValue("tensor_memory_config_plans",
                                  tensor_memory_config_plans);
    }
    if (!reshard_plans.empty()) {
      writer->WriteObjectKeyValue("reshard_plans", reshard_plans);
    }
    if (!buffer_materializations.empty()) {
      writer->WriteObjectKeyValue("buffer_materializations", buffer_materializations);
    }
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
    if (!live_form_plans.empty()) {
      writer->WriteObjectKeyValue("live_form_plans", live_form_plans);
    }
    if (!materialization_plans.empty()) {
      writer->WriteObjectKeyValue("materialization_plans", materialization_plans);
    }
    if (!consumer_binding_plans.empty()) {
      writer->WriteObjectKeyValue("consumer_binding_plans", consumer_binding_plans);
    }
    if (!exact_cb_virtual_values.empty()) {
      writer->WriteObjectKeyValue("exact_cb_virtual_values", exact_cb_virtual_values);
    }
    if (!exact_cb_use_events.empty()) {
      writer->WriteObjectKeyValue("exact_cb_use_events", exact_cb_use_events);
    }
    if (!exact_cb_live_intervals.empty()) {
      writer->WriteObjectKeyValue("exact_cb_live_intervals", exact_cb_live_intervals);
    }
    if (!exact_cb_allocations.empty()) {
      writer->WriteObjectKeyValue("exact_cb_allocations", exact_cb_allocations);
    }
    if (!exact_cb_release_events.empty()) {
      writer->WriteObjectKeyValue("exact_cb_release_events", exact_cb_release_events);
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
    // TVM export_library uses this bit to allow opaque imported runtime modules
    // in the host module tree. Actual Blackhole byte/file serialization remains
    // fail-closed until a real ExecutableSpec loader exists.
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
