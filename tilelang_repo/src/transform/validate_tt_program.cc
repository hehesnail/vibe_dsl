/*!
 * \file validate_tt_program.cc
 * \brief Validate TTProgram invariants for Phase C cutover.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/blackhole_tile_compute_covering.h"
#include "common/blackhole_tile_compute_legalizer.h"
#include "common/blackhole_utils.h"
#include "common/companion_base.h"
#include "common/spatial_plan.h"
#include "common/tt_hardware_model.h"
#include "common/tt_target_program.h"

namespace tvm {
namespace tl {

namespace {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

Map<String, Any> AsMap(const Any &any) {
  return any.as<Map<String, Any>>().value_or(Map<String, Any>());
}

int64_t GetIntOrDefault(const Map<String, Any> &map, const char *key,
                        int64_t default_value = -1) {
  if (auto value = map.Get(String(key))) {
    return Downcast<Integer>(value.value())->value;
  }
  return default_value;
}

int64_t RequireInt(const Map<String, Any> &map, const char *key,
                   const std::string &context) {
  auto value = map.Get(String(key));
  ICHECK(value.has_value()) << context << " requires " << key;
  return Downcast<Integer>(value.value())->value;
}

int64_t AlignUp(int64_t value, int64_t alignment) {
  if (alignment <= 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

std::string CoreCoordKey(int64_t x, int64_t y) {
  return std::to_string(x) + "," + std::to_string(y);
}

std::optional<Target> FindBlackholeTarget(const IRModule &mod) {
  for (const auto &[gvar, base_func] : mod->functions) {
    auto func = base_func.as<tir::PrimFunc>();
    if (!func || !IsBlackholePrimFunc(func.value())) {
      continue;
    }
    auto maybe_target = func.value()->GetAttr<Target>(tvm::attr::kTarget);
    if (maybe_target) {
      return maybe_target.value();
    }
  }
  return std::nullopt;
}

std::optional<TTHardwareModel> GetValidationHardwareModel(const IRModule &mod) {
  if (auto maybe_hardware_model = GetModuleTTHardwareModel(mod)) {
    return maybe_hardware_model.value();
  }
  if (auto maybe_target = FindBlackholeTarget(mod)) {
    return BuildBlackholeTTHardwareModel(maybe_target.value());
  }
  return std::nullopt;
}

void ValidatePositiveIntegerArray(const Array<Integer> &values,
                                  const std::string &context) {
  ICHECK(!values.empty()) << context << " requires non-empty shape";
  for (const Integer &value : values) {
    ICHECK_GT(value->value, 0) << context << " requires positive dimensions";
  }
}

int64_t IntegerArrayProduct(const Array<Integer> &values) {
  int64_t product = 1;
  for (const Integer &value : values) {
    product *= value->value;
  }
  return product;
}

int64_t CoreGroupWorkPacketCount(const TTCoreGroup &core_group) {
  int64_t work_count = 0;
  for (const Any &item : core_group->work_packets) {
    Map<String, Any> packet = AsMap(item);
    work_count += GetIntOrDefault(packet, "work_count", 0);
  }
  return work_count;
}

void ValidateMeshPlan(const TTMeshPlan &mesh_plan) {
  ICHECK(!mesh_plan->name.empty()) << "TTMeshPlan requires name";
  ICHECK(!mesh_plan->mesh_kind.empty()) << "TTMeshPlan requires mesh_kind";
  ValidatePositiveIntegerArray(mesh_plan->mesh_shape, "TTMeshPlan mesh_shape");
  ICHECK_EQ(mesh_plan->device_range_start.size(), mesh_plan->mesh_shape.size())
      << "TTMeshPlan device_range_start rank must match mesh_shape";
  ICHECK_EQ(mesh_plan->device_range_shape.size(), mesh_plan->mesh_shape.size())
      << "TTMeshPlan device_range_shape rank must match mesh_shape";
  for (int i = 0; i < mesh_plan->mesh_shape.size(); ++i) {
    ICHECK_GE(mesh_plan->device_range_start[i]->value, 0)
        << "TTMeshPlan device_range_start requires non-negative coordinates";
    ICHECK_GT(mesh_plan->device_range_shape[i]->value, 0)
        << "TTMeshPlan device_range_shape requires positive dimensions";
    ICHECK_LE(mesh_plan->device_range_start[i]->value +
                  mesh_plan->device_range_shape[i]->value,
              mesh_plan->mesh_shape[i]->value)
        << "TTMeshPlan device range must fit in mesh_shape";
  }
}

void ValidateBufferDistributionPlan(
    const TTBufferDistributionPlan &plan,
    const std::unordered_map<std::string, int64_t> &mesh_index_by_name,
    const std::unordered_map<std::string, int64_t> &core_group_index_by_name,
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  ICHECK(!plan->name.empty()) << "TTBufferDistributionPlan requires name";
  ICHECK(!plan->buffer.empty()) << "TTBufferDistributionPlan requires buffer";
  ICHECK(!plan->mesh_plan.empty())
      << "TTBufferDistributionPlan requires mesh_plan";
  ICHECK_GE(plan->mesh_plan_index, 0)
      << "TTBufferDistributionPlan requires mesh_plan_index";
  auto mesh_it =
      mesh_index_by_name.find(static_cast<std::string>(plan->mesh_plan));
  ICHECK(mesh_it != mesh_index_by_name.end())
      << "TTBufferDistributionPlan references unknown mesh_plan "
      << plan->mesh_plan;
  ICHECK_EQ(plan->mesh_plan_index, mesh_it->second)
      << "TTBufferDistributionPlan mesh_plan_index must match mesh_plan";
  ICHECK(!plan->distribution_kind.empty())
      << "TTBufferDistributionPlan requires distribution_kind";
  const std::string distribution_kind = plan->distribution_kind;
  const std::string sharding_strategy = plan->sharding_strategy;
  const std::string shard_orientation = plan->shard_orientation;
  const std::string source_region_kind = plan->source_region_kind;
  const std::string logical_index_mapping = plan->logical_index_mapping;
  const std::string core_local_address_mapping =
      plan->core_local_address_mapping;
  ICHECK(distribution_kind == "replicated" || distribution_kind == "sharded" ||
         distribution_kind == "interleaved")
      << "TTBufferDistributionPlan distribution_kind must be replicated, "
         "sharded, or interleaved";
  ICHECK(!plan->layout.empty()) << "TTBufferDistributionPlan requires layout";
  ICHECK(!plan->memory_space.empty())
      << "TTBufferDistributionPlan requires memory_space";
  const std::string memory_space = plan->memory_space;
  ICHECK(memory_space == "DRAM" || memory_space == "L1")
      << "TTBufferDistributionPlan memory_space must be DRAM or L1";
  ICHECK_GE(plan->page_size_bytes, 0)
      << "TTBufferDistributionPlan requires non-negative page_size_bytes";
  if (distribution_kind == "interleaved") {
    ICHECK_GT(plan->page_size_bytes, 0)
        << "TTBufferDistributionPlan interleaved placement requires "
           "page_size_bytes";
    ICHECK(plan->shard_shape.empty()) << "TTBufferDistributionPlan interleaved "
                                         "placement cannot carry shard_shape";
    ICHECK(plan->shard_grid_shape.empty())
        << "TTBufferDistributionPlan interleaved placement cannot carry "
           "shard_grid_shape";
  }
  if (distribution_kind == "sharded") {
    ICHECK(memory_space == "L1")
        << "TTBufferDistributionPlan sharded placement is only admitted for L1";
    ValidatePositiveIntegerArray(plan->shard_shape,
                                 "TTBufferDistributionPlan shard_shape");
    ValidatePositiveIntegerArray(plan->shard_grid_shape,
                                 "TTBufferDistributionPlan shard_grid_shape");
    ICHECK(!sharding_strategy.empty() && sharding_strategy != "none")
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "sharding_strategy";
    ICHECK(sharding_strategy == "height" || sharding_strategy == "width" ||
           sharding_strategy == "block")
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "sharding_strategy to be height, width, or block";
    ICHECK(shard_orientation == "row_major" || shard_orientation == "col_major")
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "shard_orientation to be row_major or col_major";
    ICHECK_GT(plan->page_size_bytes, 0)
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "page_size_bytes";
    ICHECK(!plan->attached_core_group.empty())
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "attached_core_group";
    ICHECK_GE(plan->attached_core_group_index, 0)
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "attached_core_group_index";
    ICHECK(!logical_index_mapping.empty() && logical_index_mapping != "none")
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "logical_index_mapping";
    ICHECK(!core_local_address_mapping.empty() &&
           core_local_address_mapping != "none")
        << "TTBufferDistributionPlan sharded L1 placement requires "
           "core_local_address_mapping";
    if (!plan->source_buffer.empty() || source_region_kind != "none" ||
        !plan->source_region_shape.empty()) {
      ICHECK(!plan->source_buffer.empty())
          << "TTBufferDistributionPlan sharded source region requires "
             "source_buffer";
      ICHECK(!source_region_kind.empty() && source_region_kind != "none")
          << "TTBufferDistributionPlan sharded source region requires "
             "source_region_kind";
      ValidatePositiveIntegerArray(
          plan->source_region_shape,
          "TTBufferDistributionPlan source_region_shape");
    }
  }
  ICHECK(!plan->shard_orientation.empty())
      << "TTBufferDistributionPlan requires shard_orientation";
  ICHECK(!plan->host_visibility.empty())
      << "TTBufferDistributionPlan requires host_visibility";
  if (!plan->attached_core_group.empty()) {
    const std::string attached_core_group =
        static_cast<std::string>(plan->attached_core_group);
    auto core_group_it = core_group_index_by_name.find(attached_core_group);
    ICHECK(core_group_it != core_group_index_by_name.end())
        << "TTBufferDistributionPlan attached_core_group references unknown "
           "core group "
        << plan->attached_core_group;
    ICHECK_EQ(plan->attached_core_group_index, core_group_it->second)
        << "TTBufferDistributionPlan attached_core_group_index must match "
           "attached_core_group";
  } else {
    ICHECK_EQ(plan->attached_core_group_index, -1)
        << "TTBufferDistributionPlan attached_core_group_index requires "
           "attached_core_group";
  }
  if (maybe_hardware_model) {
    const TTHardwareModel &hardware_model = maybe_hardware_model.value();
    if (memory_space == "DRAM") {
      ICHECK_GT(hardware_model->dram_view_count, 0)
          << "TTBufferDistributionPlan DRAM placement requires positive "
             "TTHardwareModel "
             "dram_view_count";
      if (plan->page_size_bytes > 0) {
        ICHECK_GT(hardware_model->dram_view_size, 0)
            << "TTBufferDistributionPlan DRAM placement requires positive "
               "TTHardwareModel "
               "dram_view_size";
        ICHECK_LE(plan->page_size_bytes, hardware_model->dram_view_size)
            << "TTBufferDistributionPlan DRAM view page_size_bytes exceeds "
               "hardware DRAM view: "
            << plan->page_size_bytes << " > " << hardware_model->dram_view_size;
      }
    }
    if (memory_space == "L1" && plan->page_size_bytes > 0) {
      ICHECK_GT(hardware_model->worker_l1_size, 0)
          << "TTBufferDistributionPlan L1 placement requires positive "
             "TTHardwareModel "
             "worker_l1_size";
      ICHECK_GT(hardware_model->l1_allocation_alignment_bytes, 0)
          << "TTBufferDistributionPlan L1 placement requires positive "
             "TTHardwareModel "
             "l1_allocation_alignment_bytes";
      const int64_t aligned_page_size = AlignUp(
          plan->page_size_bytes, hardware_model->l1_allocation_alignment_bytes);
      ICHECK_LE(aligned_page_size, hardware_model->worker_l1_size)
          << "TTBufferDistributionPlan L1 aligned page_size_bytes exceeds "
             "worker L1 budget: "
          << aligned_page_size << " > " << hardware_model->worker_l1_size;
    }
  }
  if (!plan->logical_shape.empty()) {
    ICHECK(!plan->local_shape.empty())
        << "TTBufferDistributionPlan logical_shape requires local_shape";
    ICHECK(plan->thread_extent.defined())
        << "TTBufferDistributionPlan logical_shape requires thread_extent";
    ICHECK(plan->replicate_extent.defined())
        << "TTBufferDistributionPlan logical_shape requires replicate_extent";
    ICHECK(!plan->inverse_logical_index_exprs.empty())
        << "TTBufferDistributionPlan logical_shape requires inverse layout "
           "expressions";
    ICHECK(!plan->inverse_logical_index_vars.empty())
        << "TTBufferDistributionPlan logical_shape requires inverse layout "
           "variables";
  }
}

void ValidateCoreGroup(
    const TTCoreGroup &core_group,
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  ICHECK_GT(core_group->logical_grid_x, 0)
      << "TTCoreGroup requires positive logical_grid_x";
  ICHECK_GT(core_group->logical_grid_y, 0)
      << "TTCoreGroup requires positive logical_grid_y";
  ICHECK_GT(core_group->logical_grid_z, 0)
      << "TTCoreGroup requires positive logical_grid_z";
  ICHECK(!core_group->physical_cores.empty())
      << "TTCoreGroup requires physical_cores";
  ICHECK(!core_group->work_packets.empty())
      << "TTCoreGroup requires work_packets";
  int64_t hardware_grid_x = 0;
  int64_t hardware_grid_y = 0;
  int64_t functional_worker_count = 0;
  if (maybe_hardware_model) {
    const TTHardwareModel &hardware_model = maybe_hardware_model.value();
    hardware_grid_x = hardware_model->logical_worker_grid_x;
    hardware_grid_y = hardware_model->logical_worker_grid_y;
    functional_worker_count = hardware_model->functional_worker_count;
    ICHECK_GT(hardware_grid_x, 0) << "TTCoreGroup validation requires positive "
                                     "TTHardwareModel logical_worker_grid_x";
    ICHECK_GT(hardware_grid_y, 0) << "TTCoreGroup validation requires positive "
                                     "TTHardwareModel logical_worker_grid_y";
    ICHECK_GT(functional_worker_count, 0)
        << "TTCoreGroup validation requires positive TTHardwareModel "
           "functional_worker_count";
    ICHECK_LE(static_cast<int64_t>(core_group->physical_cores.size()),
              functional_worker_count)
        << "TTCoreGroup physical_cores exceed hardware functional worker count";
  }

  std::unordered_set<std::string> physical_core_coords;
  for (const Any &item : core_group->physical_cores) {
    Map<String, Any> core = AsMap(item);
    ICHECK(!core.empty()) << "TTCoreGroup physical_core must be a map";
    const int64_t core_x =
        RequireInt(core, "core_x", "TTCoreGroup physical_core");
    const int64_t core_y =
        RequireInt(core, "core_y", "TTCoreGroup physical_core");
    ICHECK_GE(core_x, 0)
        << "TTCoreGroup physical_core requires non-negative core_x";
    ICHECK_GE(core_y, 0)
        << "TTCoreGroup physical_core requires non-negative core_y";
    if (maybe_hardware_model) {
      ICHECK_LT(core_x, hardware_grid_x)
          << "TTCoreGroup physical_core outside hardware logical worker grid";
      ICHECK_LT(core_y, hardware_grid_y)
          << "TTCoreGroup physical_core outside hardware logical worker grid";
    }
    ICHECK(physical_core_coords.insert(CoreCoordKey(core_x, core_y)).second)
        << "TTCoreGroup duplicate physical_core coordinate";
  }

  for (const Any &item : core_group->work_packets) {
    Map<String, Any> packet = AsMap(item);
    ICHECK(!packet.empty()) << "TTCoreGroup work_packet must be a map";
    const int64_t core_x =
        RequireInt(packet, "core_x", "TTCoreGroup work_packet");
    const int64_t core_y =
        RequireInt(packet, "core_y", "TTCoreGroup work_packet");
    ICHECK(physical_core_coords.count(CoreCoordKey(core_x, core_y)))
        << "TTCoreGroup work_packet references core outside physical_cores";
    ICHECK_GE(GetIntOrDefault(packet, "work_offset", -1), 0)
        << "TTCoreGroup work_packet requires non-negative work_offset";
    ICHECK_GT(GetIntOrDefault(packet, "work_count", 0), 0)
        << "TTCoreGroup work_packet requires positive work_count";
  }
}

std::string ExpectedTensorMemoryLayout(const TTBufferDistributionPlan &distribution) {
  const std::string kind = distribution->distribution_kind;
  if (kind == "interleaved" || kind == "replicated") {
    return "INTERLEAVED";
  }
  if (kind == "sharded") {
    const std::string strategy = distribution->sharding_strategy;
    if (strategy == "height") {
      return "HEIGHT_SHARDED";
    }
    if (strategy == "width") {
      return "WIDTH_SHARDED";
    }
    if (strategy == "block") {
      return "BLOCK_SHARDED";
    }
  }
  return "UNSUPPORTED";
}

void ValidateTensorMemoryConfigPlan(
    const TTTensorMemoryConfigPlan &plan,
    const std::unordered_map<std::string, int64_t> &distribution_index_by_name,
    const std::unordered_map<std::string, TTBufferDistributionPlan>
        &distribution_by_buffer) {
  ICHECK(!plan->name.empty()) << "TTTensorMemoryConfigPlan requires name";
  ICHECK(!plan->subject.empty())
      << "TTTensorMemoryConfigPlan " << plan->name << " requires subject";
  ICHECK(!plan->memory_layout.empty())
      << "TTTensorMemoryConfigPlan " << plan->name << " requires memory_layout";
  ICHECK(!plan->buffer_type.empty())
      << "TTTensorMemoryConfigPlan " << plan->name << " requires buffer_type";
  const std::string memory_layout = plan->memory_layout;
  ICHECK(memory_layout == "INTERLEAVED" || memory_layout == "HEIGHT_SHARDED" ||
         memory_layout == "WIDTH_SHARDED" || memory_layout == "BLOCK_SHARDED" ||
         memory_layout == "ND_SHARDED")
      << "TTTensorMemoryConfigPlan " << plan->name
      << " has invalid memory_layout " << plan->memory_layout;
  const std::string buffer_type = plan->buffer_type;
  ICHECK(buffer_type == "DRAM" || buffer_type == "L1")
      << "TTTensorMemoryConfigPlan " << plan->name
      << " has invalid buffer_type " << plan->buffer_type;
  ICHECK(!plan->origin.empty())
      << "TTTensorMemoryConfigPlan " << plan->name << " requires origin";
  auto distribution_it = distribution_by_buffer.find(str(plan->subject));
  ICHECK(distribution_it != distribution_by_buffer.end())
      << "TTTensorMemoryConfigPlan " << plan->name
      << " subject requires matching TTBufferDistributionPlan";
  const TTBufferDistributionPlan &distribution = distribution_it->second;
  ICHECK_EQ(str(plan->memory_layout), ExpectedTensorMemoryLayout(distribution))
      << "TTTensorMemoryConfigPlan " << plan->name
      << " memory_layout must match TTBufferDistributionPlan";
  ICHECK_EQ(str(plan->buffer_type), str(distribution->memory_space))
      << "TTTensorMemoryConfigPlan " << plan->name
      << " buffer_type must match TTBufferDistributionPlan memory_space";
  ICHECK_EQ(str(plan->buffer_distribution_plan), str(distribution->name))
      << "TTTensorMemoryConfigPlan " << plan->name
      << " must reference its TTBufferDistributionPlan";
  auto distribution_index_it =
      distribution_index_by_name.find(str(plan->buffer_distribution_plan));
  ICHECK(distribution_index_it != distribution_index_by_name.end())
      << "TTTensorMemoryConfigPlan " << plan->name
      << " references unknown TTBufferDistributionPlan";
  ICHECK_EQ(plan->buffer_distribution_plan_index, distribution_index_it->second)
      << "TTTensorMemoryConfigPlan " << plan->name
      << " buffer_distribution_plan_index mismatch";
  if (memory_layout != "INTERLEAVED") {
    ValidatePositiveIntegerArray(plan->shard_shape,
                                 "TTTensorMemoryConfigPlan shard_shape");
    ValidatePositiveIntegerArray(plan->shard_grid_shape,
                                 "TTTensorMemoryConfigPlan shard_grid_shape");
    ICHECK_EQ(plan->shard_shape.size(), distribution->shard_shape.size())
        << "TTTensorMemoryConfigPlan shard_shape must match distribution";
    ICHECK_EQ(plan->shard_grid_shape.size(), distribution->shard_grid_shape.size())
        << "TTTensorMemoryConfigPlan shard_grid_shape must match distribution";
    ICHECK(str(plan->shard_orientation) == "row_major" ||
           str(plan->shard_orientation) == "col_major")
        << "TTTensorMemoryConfigPlan " << plan->name
        << " has invalid shard_orientation";
  }
}

void ValidateBlockPlan(const TTBlockPlan &block_plan) {
  ICHECK(!block_plan->name.empty()) << "TTBlockPlan requires name";
  ICHECK(!block_plan->placement_kind.empty())
      << "TTBlockPlan requires placement_kind";
  ICHECK(!block_plan->task_indices.empty())
      << "TTBlockPlan requires task_indices";
}

void ValidateKernelPlan(const TTKernelPlan &kernel_plan, int64_t abi_plan_count,
                        int64_t block_plan_count) {
  ICHECK(!kernel_plan->name.empty()) << "TTKernelPlan requires name";
  ICHECK(!kernel_plan->kind.empty()) << "TTKernelPlan requires kind";
  ICHECK(!kernel_plan->core_type.empty()) << "TTKernelPlan requires core_type";
  ICHECK_GE(kernel_plan->abi_plan_index, 0)
      << "TTKernelPlan requires abi_plan_index";
  ICHECK_LT(kernel_plan->abi_plan_index, abi_plan_count)
      << "TTKernelPlan abi_plan_index out of bounds";
  ICHECK_GE(kernel_plan->block_plan_index, 0)
      << "TTKernelPlan requires block_plan_index";
  ICHECK_LT(kernel_plan->block_plan_index, block_plan_count)
      << "TTKernelPlan block_plan_index out of bounds";
}

void ValidateComputeOperandBindingPlan(
    const TTComputeOperandBindingPlan &binding) {
  ICHECK(!binding->role.empty()) << "TTComputeOperandBindingPlan requires role";
  ICHECK(!binding->buffer.empty())
      << "TTComputeOperandBindingPlan requires buffer";
  const std::string role = binding->role;
  ICHECK(role == "a" || role == "b" || role == "c" || role == "input" ||
         role == "lhs" || role == "rhs" || role == "output" || role == "scaler")
      << "TTComputeOperandBindingPlan unsupported role " << binding->role;
  if (!binding->transform_kind.empty()) {
    const std::string transform_kind = binding->transform_kind;
    ICHECK(transform_kind == "identity" || transform_kind == "transpose" ||
           transform_kind == "broadcast" || transform_kind == "cast")
        << "TTComputeOperandBindingPlan unsupported transform_kind "
        << binding->transform_kind;
  }
}

BlackholeTileComputeCoveringDecision
RequireSelectedBlackholeTileComputeCoveringForPlan(
    const TTComputeOpPlan &plan,
    const std::vector<std::string> &operand_roles) {
  const std::string operation_name = plan->operation_name;
  const BlackholeTileComputeCoveringDecision covering =
      SelectBlackholeTileComputeCovering(operation_name);
  ICHECK(covering.selected) << "TileCompute covering rejected operation "
                            << operation_name << ": " << covering.reject_reason;
  ICHECK_EQ(static_cast<std::string>(plan->kind), covering.result_kind)
      << "TileCompute covering selected result kind " << covering.result_kind
      << " for " << operation_name << ", but TTComputeOpPlan recorded "
      << plan->kind;
  RequireLegalBlackholeTileComputeSelection(
      covering.result_kind, covering.operation_name, operand_roles);
  return covering;
}

std::optional<BlackholeTileComputeCoveringDecision>
FindSelectedBlackholeTileComputeCoveringForSourceEmitter(
    const std::string &source_emitter) {
  for (const BlackholeTileComputePattern &pattern :
       GetBlackholeTileComputePatterns()) {
    if (!pattern.source_emitter ||
        source_emitter != ToString(*pattern.source_emitter)) {
      continue;
    }
    BlackholeTileComputeCoveringDecision covering =
        SelectBlackholeTileComputeCovering(ToString(pattern.operation));
    ICHECK(covering.selected)
        << "TileCompute covering rejected source emitter " << source_emitter
        << " operation " << ToString(pattern.operation) << ": "
        << covering.reject_reason;
    return covering;
  }
  return std::nullopt;
}

void ValidateComputeOpPlan(
    const TTComputeOpPlan &plan, int64_t kernel_plan_count,
    const std::unordered_set<std::string> &kernel_names) {
  ICHECK(!plan->name.empty()) << "TTComputeOpPlan requires name";
  ICHECK(!plan->kernel_name.empty()) << "TTComputeOpPlan requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(plan->kernel_name)))
      << "TTComputeOpPlan references unknown kernel " << plan->kernel_name;
  ICHECK_GE(plan->kernel_plan_index, 0)
      << "TTComputeOpPlan requires kernel_plan_index";
  ICHECK_LT(plan->kernel_plan_index, kernel_plan_count)
      << "TTComputeOpPlan kernel_plan_index out of bounds";
  ICHECK(!plan->kind.empty()) << "TTComputeOpPlan requires kind";
  ICHECK(!plan->operation_name.empty())
      << "TTComputeOpPlan requires operation_name";
  const std::string kind = plan->kind;
  ICHECK(kind == "gemm" || kind == "binary" || kind == "unary" ||
         kind == "reduce" || kind == "sfpu" || kind == "pack" ||
         kind == "copy" || kind == "fill")
      << "TTComputeOpPlan unsupported kind " << plan->kind;
  ICHECK(!plan->operand_bindings.empty())
      << "TTComputeOpPlan requires operand_bindings";
  std::unordered_set<std::string> roles;
  std::vector<std::string> operand_roles;
  for (const TTComputeOperandBindingPlan &binding : plan->operand_bindings) {
    ValidateComputeOperandBindingPlan(binding);
    operand_roles.push_back(static_cast<std::string>(binding->role));
    ICHECK(roles.insert(static_cast<std::string>(binding->role)).second)
        << "TTComputeOpPlan duplicate operand role " << binding->role;
  }
  if (kind == "gemm") {
    for (const char *role : {"a", "b", "c"}) {
      ICHECK(roles.count(role))
          << "TTComputeOpPlan GEMM requires operand role " << role;
    }
    for (const TTComputeOperandBindingPlan &binding : plan->operand_bindings) {
      ICHECK(!binding->host_buffer.empty())
          << "TTComputeOpPlan GEMM operand role " << binding->role
          << " requires host_buffer";
    }
    ICHECK_EQ(plan->problem_shape_axes.size(), 3)
        << "TTComputeOpPlan GEMM requires M/N/K problem_shape_axes";
    ICHECK_EQ(plan->problem_shape.size(), 3)
        << "TTComputeOpPlan GEMM requires M/N/K problem_shape";
    ICHECK_EQ(plan->tile_shape.size(), 3)
        << "TTComputeOpPlan GEMM requires tile_shape";
    ICHECK_EQ(plan->block_shape.size(), 3)
        << "TTComputeOpPlan GEMM requires block_shape";
    ICHECK_EQ(plan->subblock_shape.size(), 2)
        << "TTComputeOpPlan GEMM requires subblock_shape";
    ValidatePositiveIntegerArray(plan->problem_shape,
                                 "TTComputeOpPlan GEMM problem_shape");
    ValidatePositiveIntegerArray(plan->tile_shape,
                                 "TTComputeOpPlan GEMM tile_shape");
    ValidatePositiveIntegerArray(plan->block_shape,
                                 "TTComputeOpPlan GEMM block_shape");
    ValidatePositiveIntegerArray(plan->subblock_shape,
                                 "TTComputeOpPlan GEMM subblock_shape");
    ICHECK(!plan->accumulator_dtype.empty())
        << "TTComputeOpPlan GEMM requires accumulator_dtype";
  }
  RequireSelectedBlackholeTileComputeCoveringForPlan(plan, operand_roles);
  if (plan->tile_compute_dag_node_id >= 0) {
    ICHECK(!plan->tile_compute_source_emitter.empty())
        << "DAG-driven TTComputeOpPlan requires tile_compute_source_emitter";
    const std::string source_emitter = plan->tile_compute_source_emitter;
    const std::optional<BlackholeTileComputeCoveringDecision> source_covering =
        FindSelectedBlackholeTileComputeCoveringForSourceEmitter(
            source_emitter);
    ICHECK(source_covering) << "DAG-driven TTComputeOpPlan references unknown "
                               "tile_compute_source_emitter "
                            << source_emitter;
    ICHECK_EQ(plan->tile_compute_materialization_policy,
              source_covering->materialization_policy)
        << "DAG-driven TTComputeOpPlan materialization policy must match DAG "
           "source "
           "covering";
    ICHECK_GE(plan->tile_compute_fanout_use_count, 0)
        << "DAG-driven TTComputeOpPlan requires non-negative fanout use count";
    if (plan->tile_compute_fanout_use_count > 1) {
      const std::string fanout_policy = plan->tile_compute_fanout_policy;
      ICHECK(fanout_policy == "share_live_value" ||
             fanout_policy == "materialize_before_cross_event_use")
          << "DAG-driven TTComputeOpPlan unsupported fanout policy "
          << plan->tile_compute_fanout_policy;
    }
  }
}

bool ArrayContainsString(const Array<String> &values, const String &needle) {
  for (const String &value : values) {
    if (value == needle) {
      return true;
    }
  }
  return false;
}

void ValidateOpShardingContract(
    const TTOpShardingContract &contract,
    const Array<TTComputeOpPlan> &compute_op_plans,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans) {
  ICHECK(!contract->name.empty()) << "TTOpShardingContract requires name";
  ICHECK(!contract->compute_op_plan.empty())
      << "TTOpShardingContract requires compute_op_plan";
  ICHECK_GE(contract->compute_op_plan_index, 0)
      << "TTOpShardingContract requires compute_op_plan_index";
  ICHECK_LT(contract->compute_op_plan_index,
            static_cast<int64_t>(compute_op_plans.size()))
      << "TTOpShardingContract compute_op_plan_index out of bounds";
  const TTComputeOpPlan &compute_op =
      compute_op_plans[static_cast<size_t>(contract->compute_op_plan_index)];
  ICHECK_EQ(contract->compute_op_plan, compute_op->name)
      << "TTOpShardingContract compute_op_plan must match indexed "
         "TTComputeOpPlan";
  ICHECK_EQ(contract->operation_name, compute_op->operation_name)
      << "TTOpShardingContract operation_name must match TTComputeOpPlan";
  ICHECK_EQ(contract->op_kind, compute_op->kind)
      << "TTOpShardingContract op_kind must match TTComputeOpPlan";
  ICHECK(!contract->operand_role.empty())
      << "TTOpShardingContract requires operand_role";
  ICHECK(!contract->operand_buffer.empty())
      << "TTOpShardingContract requires operand_buffer";
  bool found_operand = false;
  for (const TTComputeOperandBindingPlan &binding :
       compute_op->operand_bindings) {
    if (binding->role != contract->operand_role) {
      continue;
    }
    found_operand = true;
    ICHECK_EQ(contract->operand_buffer, binding->buffer)
        << "TTOpShardingContract operand_buffer must match compute operand";
    ICHECK_EQ(contract->operand_host_buffer, binding->host_buffer)
        << "TTOpShardingContract operand_host_buffer must match compute operand";
    break;
  }
  ICHECK(found_operand) << "TTOpShardingContract references unknown operand role "
                        << contract->operand_role;

  ICHECK(!contract->memory_config_plan.empty())
      << "TTOpShardingContract requires memory_config_plan";
  ICHECK_GE(contract->memory_config_plan_index, 0)
      << "TTOpShardingContract requires memory_config_plan_index";
  ICHECK_LT(contract->memory_config_plan_index,
            static_cast<int64_t>(tensor_memory_config_plans.size()))
      << "TTOpShardingContract memory_config_plan_index out of bounds";
  const TTTensorMemoryConfigPlan &memory_config =
      tensor_memory_config_plans[static_cast<size_t>(
          contract->memory_config_plan_index)];
  ICHECK_EQ(contract->memory_config_plan, memory_config->name)
      << "TTOpShardingContract memory_config_plan must match indexed "
         "TTTensorMemoryConfigPlan";
  ICHECK_EQ(contract->operand_buffer, memory_config->subject)
      << "TTOpShardingContract memory config subject must match operand_buffer";
  ICHECK(!contract->accepted_memory_layouts.empty())
      << "TTOpShardingContract requires accepted_memory_layouts";
  ICHECK(ArrayContainsString(contract->accepted_memory_layouts,
                             memory_config->memory_layout))
      << "placement conflict: consumer " << contract->compute_op_plan
      << " operand " << contract->operand_role << " selected_memory_layout "
      << memory_config->memory_layout << " is not accepted by op contract";
  ICHECK(!contract->accepted_buffer_types.empty())
      << "TTOpShardingContract requires accepted_buffer_types";
  ICHECK(ArrayContainsString(contract->accepted_buffer_types,
                             memory_config->buffer_type))
      << "placement conflict: consumer " << contract->compute_op_plan
      << " operand " << contract->operand_role << " selected_buffer_type "
      << memory_config->buffer_type << " is not accepted by op contract";
  const String selected_strategy =
      memory_config->shard_distribution_strategy.empty()
          ? String("none")
          : memory_config->shard_distribution_strategy;
  ICHECK(!contract->accepted_sharding_strategies.empty())
      << "TTOpShardingContract requires accepted_sharding_strategies";
  ICHECK(ArrayContainsString(contract->accepted_sharding_strategies,
                             selected_strategy))
      << "placement conflict: consumer " << contract->compute_op_plan
      << " operand " << contract->operand_role << " selected_strategy "
      << selected_strategy << " is not accepted by op contract";
  const std::string orientation = contract->required_shard_orientation;
  ICHECK(orientation == "row_major" || orientation == "col_major")
      << "TTOpShardingContract required_shard_orientation must be row_major or "
         "col_major";
  ICHECK_EQ(contract->required_shard_orientation,
            memory_config->shard_orientation)
      << "TTOpShardingContract required_shard_orientation must match selected "
         "memory config";
  const std::string output_policy = contract->output_policy;
  ICHECK(output_policy == "not_output" ||
         output_policy == "produces_operand_placement" ||
         output_policy == "inherit_input" || output_policy == "caller_specified" ||
         output_policy == "op_selected" ||
         output_policy == "interleaved_default")
      << "TTOpShardingContract unsupported output_policy "
      << contract->output_policy;
  if (output_policy == "not_output") {
    ICHECK(!contract->can_produce_output_placement)
        << "TTOpShardingContract not_output cannot produce output placement";
  } else {
    ICHECK(contract->can_produce_output_placement)
        << "TTOpShardingContract output policy requires "
           "can_produce_output_placement";
  }
}

void ValidatePlacementResolutionPlan(
    const TTPlacementResolutionPlan &resolution,
    const Array<TTOpShardingContract> &op_sharding_contracts,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans) {
  ICHECK(!resolution->name.empty())
      << "TTPlacementResolutionPlan requires name";
  ICHECK(!resolution->op_sharding_contract.empty())
      << "TTPlacementResolutionPlan requires op_sharding_contract";
  ICHECK_GE(resolution->op_sharding_contract_index, 0)
      << "TTPlacementResolutionPlan requires op_sharding_contract_index";
  ICHECK_LT(resolution->op_sharding_contract_index,
            static_cast<int64_t>(op_sharding_contracts.size()))
      << "TTPlacementResolutionPlan op_sharding_contract_index out of bounds";
  const TTOpShardingContract &contract =
      op_sharding_contracts[static_cast<size_t>(
          resolution->op_sharding_contract_index)];
  ICHECK_EQ(resolution->op_sharding_contract, contract->name)
      << "TTPlacementResolutionPlan op_sharding_contract must match indexed "
         "TTOpShardingContract";
  ICHECK_EQ(resolution->consumer_op_plan, contract->compute_op_plan)
      << "TTPlacementResolutionPlan consumer_op_plan must match contract";
  ICHECK_EQ(resolution->consumer_op_plan_index,
            contract->compute_op_plan_index)
      << "TTPlacementResolutionPlan consumer_op_plan_index must match contract";
  ICHECK_EQ(resolution->consumer_operand_role, contract->operand_role)
      << "TTPlacementResolutionPlan consumer_operand_role must match contract";
  ICHECK_GE(resolution->selected_memory_config_plan_index, 0)
      << "TTPlacementResolutionPlan requires selected_memory_config_plan_index";
  ICHECK_LT(resolution->selected_memory_config_plan_index,
            static_cast<int64_t>(tensor_memory_config_plans.size()))
      << "TTPlacementResolutionPlan selected_memory_config_plan_index out of "
         "bounds";
  const TTTensorMemoryConfigPlan &memory_config =
      tensor_memory_config_plans[static_cast<size_t>(
          resolution->selected_memory_config_plan_index)];
  ICHECK_EQ(resolution->selected_memory_config_plan, memory_config->name)
      << "TTPlacementResolutionPlan selected_memory_config_plan must match "
         "indexed TTTensorMemoryConfigPlan";
  ICHECK_EQ(resolution->selected_memory_config_plan,
            contract->memory_config_plan)
      << "TTPlacementResolutionPlan selected memory config must match "
         "TTOpShardingContract";
  ICHECK_EQ(resolution->selected_memory_layout, memory_config->memory_layout)
      << "TTPlacementResolutionPlan selected_memory_layout must match selected "
         "memory config";
  ICHECK_EQ(resolution->selected_buffer_type, memory_config->buffer_type)
      << "TTPlacementResolutionPlan selected_buffer_type must match selected "
         "memory config";
  ICHECK(resolution->resolution_kind == "selected_existing")
      << "TTPlacementResolutionPlan unsupported resolution_kind "
      << resolution->resolution_kind;
  ICHECK(!resolution->conversion_required)
      << "TTPlacementResolutionPlan conversion_required requires TTReshardPlan";
  ICHECK(resolution->conversion_plan.empty())
      << "TTPlacementResolutionPlan conversion_plan requires "
         "conversion_required";
  ICHECK(resolution->conflict_reason.empty())
      << "TTPlacementResolutionPlan conflict_reason must fail validation before "
         "source emission";
}

void ValidateReshardPlan(
    const TTReshardPlan &plan,
    const Array<TTTensorMemoryConfigPlan> &tensor_memory_config_plans,
    const Array<TTMaterializationPlan> &materialization_plans) {
  ICHECK(!plan->name.empty()) << "TTReshardPlan requires name";
  ICHECK(!plan->source_value.empty()) << "TTReshardPlan requires source_value";
  ICHECK(!plan->target_value.empty()) << "TTReshardPlan requires target_value";
  ICHECK_NE(plan->source_value, plan->target_value)
      << "TTReshardPlan source_value and target_value must differ";
  ICHECK_GE(plan->source_memory_config_plan_index, 0)
      << "TTReshardPlan requires source_memory_config_plan_index";
  ICHECK_LT(plan->source_memory_config_plan_index,
            static_cast<int64_t>(tensor_memory_config_plans.size()))
      << "TTReshardPlan source_memory_config_plan_index out of bounds";
  ICHECK_GE(plan->target_memory_config_plan_index, 0)
      << "TTReshardPlan requires target_memory_config_plan_index";
  ICHECK_LT(plan->target_memory_config_plan_index,
            static_cast<int64_t>(tensor_memory_config_plans.size()))
      << "TTReshardPlan target_memory_config_plan_index out of bounds";
  const TTTensorMemoryConfigPlan &source_config =
      tensor_memory_config_plans[static_cast<size_t>(
          plan->source_memory_config_plan_index)];
  const TTTensorMemoryConfigPlan &target_config =
      tensor_memory_config_plans[static_cast<size_t>(
          plan->target_memory_config_plan_index)];
  ICHECK_EQ(plan->source_memory_config_plan, source_config->name)
      << "TTReshardPlan source_memory_config_plan must match indexed "
         "TTTensorMemoryConfigPlan";
  ICHECK_EQ(plan->target_memory_config_plan, target_config->name)
      << "TTReshardPlan target_memory_config_plan must match indexed "
         "TTTensorMemoryConfigPlan";
  ICHECK_EQ(plan->source_value, source_config->subject)
      << "TTReshardPlan source_value must match source memory config subject";
  ICHECK_EQ(plan->target_value, target_config->subject)
      << "TTReshardPlan target_value must match target memory config subject";
  const std::string conversion_kind = plan->conversion_kind;
  ICHECK(conversion_kind == "interleaved_to_sharded" ||
         conversion_kind == "sharded_to_interleaved" ||
         conversion_kind == "reshard" || conversion_kind == "unsupported")
      << "TTReshardPlan unsupported conversion_kind " << plan->conversion_kind;
  if (conversion_kind == "interleaved_to_sharded") {
    ICHECK_EQ(source_config->memory_layout, "INTERLEAVED")
        << "TTReshardPlan interleaved_to_sharded requires interleaved source";
    ICHECK_NE(target_config->memory_layout, "INTERLEAVED")
        << "TTReshardPlan interleaved_to_sharded requires sharded target";
    ICHECK(!plan->materialization_protocol.empty())
        << "TTReshardPlan interleaved_to_sharded requires "
           "materialization_protocol";
    ICHECK(!plan->source_region_kind.empty() &&
           plan->source_region_kind != "none")
        << "TTReshardPlan interleaved_to_sharded requires source_region_kind";
    ICHECK(!plan->source_region_shape.empty())
        << "TTReshardPlan interleaved_to_sharded requires source_region_shape";
  }
  if (plan->materialization_plan_index >= 0) {
    ICHECK_LT(plan->materialization_plan_index,
              static_cast<int64_t>(materialization_plans.size()))
        << "TTReshardPlan materialization_plan_index out of bounds";
    const TTMaterializationPlan &materialization =
        materialization_plans[static_cast<size_t>(
            plan->materialization_plan_index)];
    ICHECK_EQ(plan->materialization_plan, materialization->name)
        << "TTReshardPlan materialization_plan must match indexed "
           "TTMaterializationPlan";
    ICHECK_EQ(plan->target_value, materialization->target_buffer)
        << "TTReshardPlan materialization target_buffer must match target_value";
  }
  ICHECK(plan->scheduling_kind == "runtime" ||
         plan->scheduling_kind == "load_time" ||
         plan->scheduling_kind == "compile_time")
      << "TTReshardPlan unsupported scheduling_kind " << plan->scheduling_kind;
  ICHECK(plan->inserted_by == "planner" || plan->inserted_by == "user")
      << "TTReshardPlan unsupported inserted_by " << plan->inserted_by;
  ICHECK(plan->admission_status == "admitted" ||
         plan->admission_status == "unsupported")
      << "TTReshardPlan unsupported admission_status "
      << plan->admission_status;
  if (plan->admission_status == "admitted") {
    ICHECK(plan->unsupported_reason.empty())
        << "TTReshardPlan admitted conversion cannot carry unsupported_reason";
  } else {
    ICHECK(!plan->unsupported_reason.empty())
        << "TTReshardPlan unsupported conversion requires unsupported_reason";
  }
}

void ValidateTileComputeFanoutDemand(
    const TTTileComputeFanoutDemand &demand,
    const std::unordered_set<std::string> &kernel_names) {
  ICHECK(!demand->name.empty()) << "TTTileComputeFanoutDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTTileComputeFanoutDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTTileComputeFanoutDemand references unknown kernel "
      << demand->kernel_name;
  ICHECK_GE(demand->producer_node, 0)
      << "TTTileComputeFanoutDemand requires producer_node";
  ICHECK(!demand->producer_operation.empty())
      << "TTTileComputeFanoutDemand requires producer_operation";
  ICHECK_GT(demand->use_count, 1)
      << "TTTileComputeFanoutDemand requires fanout use_count > 1";
  ICHECK_EQ(demand->consumer_nodes.size(),
            static_cast<size_t>(demand->use_count))
      << "TTTileComputeFanoutDemand consumer_nodes must match use_count";
  const std::string policy = demand->policy;
  ICHECK(policy == "share_live_value" ||
         policy == "materialize_before_cross_event_use")
      << "TTTileComputeFanoutDemand unsupported policy " << demand->policy;
  ICHECK(!demand->evidence.empty())
      << "TTTileComputeFanoutDemand requires evidence";
}

void ValidateTileComputeMaterializationDemand(
    const TTTileComputeMaterializationDemand &demand,
    const std::unordered_set<std::string> &kernel_names) {
  ICHECK(!demand->name.empty())
      << "TTTileComputeMaterializationDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTTileComputeMaterializationDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTTileComputeMaterializationDemand references unknown kernel "
      << demand->kernel_name;
  ICHECK_GE(demand->node_id, 0)
      << "TTTileComputeMaterializationDemand requires node_id";
  ICHECK(!demand->operation_name.empty())
      << "TTTileComputeMaterializationDemand requires operation_name";
  ICHECK(!demand->pattern_name.empty())
      << "TTTileComputeMaterializationDemand requires pattern_name";
  ICHECK(!demand->policy.empty())
      << "TTTileComputeMaterializationDemand requires policy";
  ICHECK(demand->policy != "none")
      << "TTTileComputeMaterializationDemand cannot record policy=none";
  ICHECK(!demand->evidence.empty())
      << "TTTileComputeMaterializationDemand requires evidence";
}

void ValidateResourceDemand(const TTResourceDemand &demand,
                            const std::unordered_set<std::string> &kernel_names,
                            int64_t core_group_count) {
  ICHECK(!demand->name.empty()) << "TTResourceDemand requires name";
  ICHECK(!demand->kernel_name.empty())
      << "TTResourceDemand requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(demand->kernel_name)))
      << "TTResourceDemand references unknown kernel " << demand->kernel_name;
  ICHECK(!demand->core_group.empty()) << "TTResourceDemand requires core_group";
  ICHECK_GE(demand->core_group_index, 0)
      << "TTResourceDemand requires core_group_index";
  ICHECK_LT(demand->core_group_index, core_group_count)
      << "TTResourceDemand core_group_index out of bounds";
  ICHECK_GE(demand->cb_requirement_count, 0)
      << "TTResourceDemand requires non-negative cb_requirement_count";
  ICHECK_GE(demand->cb_l1_bytes, 0)
      << "TTResourceDemand requires non-negative cb_l1_bytes";
  ICHECK_GE(demand->semaphore_count, 0)
      << "TTResourceDemand requires non-negative semaphore_count";
  ICHECK_GE(demand->communication_edge_count, 0)
      << "TTResourceDemand requires non-negative communication_edge_count";
  ICHECK(!demand->tile_compute_fanout_demands.empty() ||
         !demand->tile_compute_materialization_demands.empty() ||
         !demand->tile_compute_unsupported_reasons.empty() ||
         demand->cb_requirement_count > 0 || demand->semaphore_count > 0 ||
         demand->communication_edge_count > 0)
      << "TTResourceDemand requires tile-compute or explicit resource demand "
         "evidence";
  for (const TTTileComputeFanoutDemand &fanout :
       demand->tile_compute_fanout_demands) {
    ValidateTileComputeFanoutDemand(fanout, kernel_names);
  }
  for (const TTTileComputeMaterializationDemand &materialization :
       demand->tile_compute_materialization_demands) {
    ValidateTileComputeMaterializationDemand(materialization, kernel_names);
  }
  for (const String &reason : demand->tile_compute_unsupported_reasons) {
    ICHECK(!reason.empty())
        << "TTResourceDemand tile_compute_unsupported_reasons cannot be empty";
  }
}

void ValidateResourcePressureReport(
    const TTResourcePressureReport &report,
    const std::unordered_set<std::string> &kernel_names,
    int64_t core_group_count,
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  ICHECK(!report->name.empty()) << "TTResourcePressureReport requires name";
  ICHECK(!report->kernel_name.empty())
      << "TTResourcePressureReport requires kernel_name";
  ICHECK(kernel_names.count(static_cast<std::string>(report->kernel_name)))
      << "TTResourcePressureReport references unknown kernel "
      << report->kernel_name;
  ICHECK(!report->core_group.empty())
      << "TTResourcePressureReport requires core_group";
  ICHECK_GE(report->core_group_index, 0)
      << "TTResourcePressureReport requires core_group_index";
  ICHECK_LT(report->core_group_index, core_group_count)
      << "TTResourcePressureReport core_group_index out of bounds";
  for (const TTTileComputeMaterializationDemand &materialization :
       report->required_materializations) {
    ValidateTileComputeMaterializationDemand(materialization, kernel_names);
  }
  ICHECK_GE(report->per_core_cb_id_pressure, 0)
      << "TTResourcePressureReport requires non-negative "
         "per_core_cb_id_pressure";
  ICHECK_GE(report->per_core_cb_l1_bytes, 0)
      << "TTResourcePressureReport requires non-negative per_core_cb_l1_bytes";
  ICHECK_GE(report->per_core_l1_buffer_bytes, 0)
      << "TTResourcePressureReport requires non-negative "
         "per_core_l1_buffer_bytes";
  ICHECK_GE(report->max_simultaneous_l1_bytes, 0)
      << "TTResourcePressureReport requires non-negative "
         "max_simultaneous_l1_bytes";
  ICHECK_GT(report->cb_id_limit, 0)
      << "TTResourcePressureReport requires positive cb_id_limit";
  ICHECK_GT(report->worker_l1_budget_bytes, 0)
      << "TTResourcePressureReport requires positive worker_l1_budget_bytes";
  ICHECK_GT(report->l1_alignment_bytes, 0)
      << "TTResourcePressureReport requires positive l1_alignment_bytes";
  ICHECK_GE(report->per_core_cb_l1_aligned_bytes, report->per_core_cb_l1_bytes)
      << "TTResourcePressureReport aligned CB L1 bytes must cover raw CB L1 "
         "bytes";
  ICHECK_EQ(report->l1_alignment_waste_bytes,
            report->per_core_cb_l1_aligned_bytes - report->per_core_cb_l1_bytes)
      << "TTResourcePressureReport l1_alignment_waste_bytes must equal aligned "
         "- raw CB bytes";
  ICHECK_LE(report->per_core_cb_id_pressure, report->cb_id_limit)
      << "ResourcePressureReport CB id pressure exceeds hardware limit: "
         "required "
      << report->per_core_cb_id_pressure << ", limit " << report->cb_id_limit;
  ICHECK_LE(report->max_simultaneous_l1_bytes, report->worker_l1_budget_bytes)
      << "ResourcePressureReport L1 pressure exceeds worker budget: required "
      << report->max_simultaneous_l1_bytes << ", budget "
      << report->worker_l1_budget_bytes;
  ICHECK_EQ(report->max_simultaneous_l1_bytes,
            report->per_core_cb_l1_aligned_bytes +
                report->per_core_l1_buffer_bytes)
      << "TTResourcePressureReport max_simultaneous_l1_bytes must equal "
         "aligned CB bytes "
         "plus L1 buffer bytes";
  if (maybe_hardware_model) {
    const TTHardwareModel &hardware_model = maybe_hardware_model.value();
    ICHECK_EQ(report->cb_id_limit, hardware_model->max_cb_count)
        << "TTResourcePressureReport cb_id_limit must match TTHardwareModel";
    ICHECK_EQ(report->worker_l1_budget_bytes, hardware_model->worker_l1_size)
        << "TTResourcePressureReport worker_l1_budget_bytes must match "
           "TTHardwareModel";
    ICHECK_EQ(report->l1_alignment_bytes,
              hardware_model->l1_allocation_alignment_bytes)
        << "TTResourcePressureReport l1_alignment_bytes must match "
           "TTHardwareModel";
  }
  ICHECK(report->tile_compute_unsupported_reasons.empty())
      << "ResourcePressureReport unsupported tile compute: "
      << report->tile_compute_unsupported_reasons[0];
  ICHECK(report->unsupported_reasons.empty())
      << "ResourcePressureReport unsupported: "
      << report->unsupported_reasons[0];
  ICHECK(!report->core_grid_requirement.empty())
      << "TTResourcePressureReport requires core_grid_requirement";
  ICHECK(!report->dram_view_requirement.empty())
      << "TTResourcePressureReport requires dram_view_requirement";
}

void ValidateSyncPlan(const TTSyncPlan &sync_plan) {
  ICHECK(!sync_plan->name.empty()) << "TTSyncPlan requires name";
  ICHECK(!sync_plan->kind.empty()) << "TTSyncPlan requires kind";
  ICHECK_GE(sync_plan->source_task_index, 0)
      << "TTSyncPlan requires source_task_index";
  ICHECK_GE(sync_plan->target_task_index, 0)
      << "TTSyncPlan requires target_task_index";
  ICHECK(!sync_plan->ordering_kind.empty())
      << "TTSyncPlan requires ordering_kind";
  ICHECK(!sync_plan->completion_kind.empty())
      << "TTSyncPlan requires completion_kind";
}

void ValidateCBPlan(const TTCBPlan &cb_plan) {
  ICHECK(!cb_plan->name.empty()) << "TTCBPlan requires name";
  ICHECK(!cb_plan->resource_class.empty())
      << "TTCBPlan requires resource_class";
  ICHECK_GT(cb_plan->num_pages, 0) << "TTCBPlan requires positive num_pages";
  ICHECK_GT(cb_plan->page_size_bytes, 0)
      << "TTCBPlan requires positive page_size_bytes";
  ICHECK(!cb_plan->data_format.empty()) << "TTCBPlan requires data_format";
  ICHECK_GE(cb_plan->initial_reserve_pages, 0)
      << "TTCBPlan requires non-negative initial_reserve_pages";
  ICHECK(!cb_plan->flow_class.empty()) << "TTCBPlan requires flow_class";
  const std::string flow_class = cb_plan->flow_class;
  ICHECK(flow_class == "state" || flow_class == "stream" ||
         flow_class == "republish")
      << "TTCBPlan flow_class must be one of state/stream/republish";
  ICHECK_GE(cb_plan->publish_pages_per_event, 0)
      << "TTCBPlan requires non-negative publish_pages_per_event";
  ICHECK_GE(cb_plan->consume_pages_per_event, 0)
      << "TTCBPlan requires non-negative consume_pages_per_event";
  if (flow_class == "republish") {
    ICHECK_GT(cb_plan->publish_pages_per_event, 0)
        << "republish TTCBPlan requires positive publish_pages_per_event";
    ICHECK_GT(cb_plan->consume_pages_per_event, 0)
        << "republish TTCBPlan requires positive consume_pages_per_event";
    ICHECK_LE(cb_plan->publish_pages_per_event, cb_plan->num_pages)
        << "republish TTCBPlan publish_pages_per_event must fit in num_pages";
    ICHECK_LE(cb_plan->consume_pages_per_event, cb_plan->num_pages)
        << "republish TTCBPlan consume_pages_per_event must fit in num_pages";
  }
  ICHECK_GE(cb_plan->lifetime_begin, 0)
      << "TTCBPlan requires non-negative lifetime_begin";
  ICHECK_GE(cb_plan->lifetime_end, cb_plan->lifetime_begin)
      << "TTCBPlan requires lifetime_end >= lifetime_begin";
}

void ValidateAccessor(const TTAccessorSpec &accessor) {
  ICHECK(!accessor->buffer.empty()) << "TTABIPlan accessor requires buffer";
  ICHECK_GT(accessor->compile_time_arg_count, 0)
      << "TTABIPlan accessor requires compile_time_arg_count";
  ICHECK(!accessor->layout.empty()) << "TTABIPlan accessor requires layout";
  ICHECK(!accessor->memory_space.empty())
      << "TTABIPlan accessor requires memory_space";
}

void ValidateShardedAccessorWorkMapping(
    const TTAccessorSpec &accessor,
    const std::unordered_map<std::string, TTBufferDistributionPlan>
        &distribution_by_buffer,
    const std::unordered_map<std::string, TTCoreGroup> &core_group_by_name) {
  const std::string accessor_layout = accessor->layout;
  const std::string accessor_memory_space = accessor->memory_space;
  if (accessor_layout != "sharded" ||
      (accessor_memory_space != "l1" && accessor_memory_space != "L1")) {
    return;
  }
  const std::string buffer = accessor->buffer;
  auto distribution_it = distribution_by_buffer.find(buffer);
  ICHECK(distribution_it != distribution_by_buffer.end())
      << "TTABIPlan sharded accessor buffer requires TTBufferDistributionPlan "
      << buffer;
  const TTBufferDistributionPlan &distribution = distribution_it->second;
  ICHECK(distribution->distribution_kind == "sharded")
      << "TTABIPlan sharded accessor requires sharded TTBufferDistributionPlan "
      << buffer;
  auto core_group_it =
      core_group_by_name.find(str(distribution->attached_core_group));
  ICHECK(core_group_it != core_group_by_name.end())
      << "TTABIPlan sharded accessor attached_core_group references unknown "
         "core group "
      << distribution->attached_core_group;
  const int64_t shard_count =
      IntegerArrayProduct(distribution->shard_grid_shape);
  const int64_t attached_work_count =
      CoreGroupWorkPacketCount(core_group_it->second);
  ICHECK_LE(shard_count, attached_work_count)
      << "TTABIPlan sharded accessor shard_grid_shape requires "
         "attached_core_group work_packets to cover every shard; "
         "retile/work-coarsening plan required: "
      << shard_count << " shards > " << attached_work_count
      << " attached work packets";
}

void ValidateCompileTimeArgSpec(const TTCompileTimeArgSpec &spec) {
  ICHECK(!spec->kind.empty())
      << "TTABIPlan compile_time_arg_spec requires kind";
  ICHECK(!spec->dtype.empty())
      << "TTABIPlan compile_time_arg_spec requires dtype";
  ICHECK_GE(spec->offset, 0)
      << "TTABIPlan compile_time_arg_spec requires offset";
  ICHECK_GE(spec->count, 0) << "TTABIPlan compile_time_arg_spec requires count";
}

void ValidateKernelLeafFields(const TTKernel &kernel) {
  ICHECK(kernel->launch_spec.defined()) << "TTKernel requires launch_spec";
  ICHECK(!kernel->launch_spec->core_type.empty())
      << "TTKernel launch_spec requires core_type";

  if (kernel->kind == "compute" || kernel->core_type == "trisc") {
    ICHECK(kernel->compute_config.defined() &&
           !kernel->compute_config->math_fidelity.empty())
        << "TTKernel compute kernels require compute_config";
    ICHECK_GT(kernel->compute_config->k_pack, 0)
        << "TTKernel compute_config requires positive k_pack";
  }
}

void ValidateLiveFormPlans(const TTProgram &program,
                           std::unordered_set<std::string> *live_form_names) {
  for (const TTLiveFormPlan &plan : program->live_form_plans) {
    ICHECK(!plan->name.empty()) << "TTLiveFormPlan requires name";
    ICHECK(!plan->logical_value.empty())
        << "TTLiveFormPlan requires logical_value";
    ICHECK(!plan->spatial_live_value.empty())
        << "TTLiveFormPlan requires spatial_live_value";
    ICHECK_GE(plan->spatial_live_value_index, 0)
        << "TTLiveFormPlan requires spatial_live_value_index";
    ICHECK(!plan->producer_kernel.empty())
        << "TTLiveFormPlan requires producer_kernel";
    ICHECK(!plan->physical_form.empty())
        << "TTLiveFormPlan requires physical_form";
    ICHECK(!plan->execution_topology.empty())
        << "TTLiveFormPlan requires execution_topology";
    ICHECK_GT(plan->physical_local_extent, 0)
        << "TTLiveFormPlan requires positive physical_local_extent";
    ICHECK_GT(plan->logical_element_count, 0)
        << "TTLiveFormPlan requires positive logical_element_count";
    ICHECK(live_form_names->insert(plan->name).second)
        << "duplicate TTLiveFormPlan name " << plan->name;
  }
}

void ValidateMaterializationPlans(
    const TTProgram &program,
    const std::unordered_set<std::string> &live_form_names,
    int64_t cb_plan_count) {
  for (const TTMaterializationPlan &plan : program->materialization_plans) {
    ICHECK(!plan->name.empty()) << "TTMaterializationPlan requires name";
    ICHECK(!plan->source_live_form.empty())
        << "TTMaterializationPlan requires source_live_form";
    ICHECK(live_form_names.count(plan->source_live_form))
        << "TTMaterializationPlan references unknown source_live_form "
        << plan->source_live_form;
    ICHECK(!plan->materialization_boundary.empty())
        << "TTMaterializationPlan requires materialization_boundary";
    ICHECK_GE(plan->materialization_boundary_index, 0)
        << "TTMaterializationPlan requires materialization_boundary_index";
    ICHECK(!plan->target_buffer.empty())
        << "TTMaterializationPlan requires target_buffer";
    ICHECK(!plan->target_kernel.empty())
        << "TTMaterializationPlan requires target_kernel";
    ICHECK(!plan->materialization_protocol.empty())
        << "TTMaterializationPlan requires materialization_protocol";
    ICHECK(!plan->publication_protocol.empty())
        << "TTMaterializationPlan requires publication_protocol";
    ICHECK(!plan->produced_live_form.empty())
        << "TTMaterializationPlan requires produced_live_form";
    ICHECK(live_form_names.count(plan->produced_live_form))
        << "TTMaterializationPlan references unknown produced_live_form "
        << plan->produced_live_form;
    if (plan->materialization_protocol ==
        buffer_materialization::kCBRepublish) {
      ICHECK(!plan->required_cb_plan_indices.empty())
          << "TTMaterializationPlan cb_republish requires "
             "required_cb_plan_indices";
      ICHECK(plan->publication_protocol ==
                 buffer_materialization::kMailboxWritePtr ||
             plan->publication_protocol ==
                 buffer_materialization::kPackThreadDirectStore ||
             plan->publication_protocol == buffer_materialization::kPackTile ||
             plan->publication_protocol ==
                 buffer_materialization::kTilizeCastFragmentSlice)
          << "TTMaterializationPlan cb_republish has unsupported "
             "publication_protocol "
          << plan->publication_protocol;
      if (plan->publication_protocol ==
              buffer_materialization::kPackThreadDirectStore ||
          plan->publication_protocol == buffer_materialization::kPackTile) {
        ICHECK(!plan->host_buffer.empty())
            << "TTMaterializationPlan requires host_buffer";
      }
    }
    for (const Integer &index : plan->required_cb_plan_indices) {
      ICHECK_GE(index->value, 0)
          << "TTMaterializationPlan requires non-negative CB plan index";
      ICHECK_LT(index->value, cb_plan_count)
          << "TTMaterializationPlan required_cb_plan_indices out of bounds";
    }
  }
}

void ValidateConsumerBindingPlans(
    const TTProgram &program,
    const std::unordered_set<std::string> &live_form_names,
    int64_t abi_plan_count) {
  for (const TTConsumerBindingPlan &plan : program->consumer_binding_plans) {
    ICHECK(!plan->name.empty()) << "TTConsumerBindingPlan requires name";
    ICHECK(!plan->consumer_kernel.empty())
        << "TTConsumerBindingPlan requires consumer_kernel";
    ICHECK(!plan->consumer_op_kind.empty())
        << "TTConsumerBindingPlan requires consumer_op_kind";
    ICHECK(!plan->source_live_form.empty())
        << "TTConsumerBindingPlan requires source_live_form";
    ICHECK(live_form_names.count(plan->source_live_form))
        << "TTConsumerBindingPlan references unknown source_live_form "
        << plan->source_live_form;
    ICHECK(!plan->live_value_edge.empty())
        << "TTConsumerBindingPlan requires live_value_edge";
    ICHECK_GE(plan->live_value_edge_index, 0)
        << "TTConsumerBindingPlan requires live_value_edge_index";
    if (plan->abi_plan_index >= 0) {
      ICHECK_LT(plan->abi_plan_index, abi_plan_count)
          << "TTConsumerBindingPlan abi_plan_index out of bounds";
    }
    ICHECK(plan->accepts_distributed_slice || plan->requires_full_logical_tile)
        << "TTConsumerBindingPlan must declare whether the consumer accepts a "
           "distributed slice "
           "or requires a full logical tile";
  }
}

void ValidateSpatialLiveReferences(const TTProgram &program,
                                   const SpatialPlan &spatial_plan) {
  std::unordered_map<std::string, std::string> live_value_name_by_form;
  for (const TTLiveFormPlan &plan : program->live_form_plans) {
    ICHECK_LT(plan->spatial_live_value_index,
              static_cast<int64_t>(spatial_plan->live_values.size()))
        << "TTLiveFormPlan spatial_live_value_index out of bounds";
    const LiveValue &live_value =
        spatial_plan
            ->live_values[static_cast<size_t>(plan->spatial_live_value_index)];
    ICHECK_EQ(plan->spatial_live_value, live_value->name)
        << "TTLiveFormPlan spatial_live_value must match SpatialPlan "
           "live_values index";
    ICHECK_EQ(plan->logical_value, live_value->subject)
        << "TTLiveFormPlan logical_value must match SpatialPlan LiveValue "
           "subject";
    ICHECK_GE(live_value->version_index, 0)
        << "TTLiveFormPlan requires versioned SpatialPlan LiveValue";
    ICHECK(!live_value->definition_kind.empty())
        << "TTLiveFormPlan requires SpatialPlan LiveValue definition_kind";
    live_value_name_by_form[static_cast<std::string>(plan->name)] =
        static_cast<std::string>(plan->spatial_live_value);
  }

  for (const TTMaterializationPlan &plan : program->materialization_plans) {
    ICHECK_LT(
        plan->materialization_boundary_index,
        static_cast<int64_t>(spatial_plan->materialization_boundaries.size()))
        << "TTMaterializationPlan materialization_boundary_index out of bounds";
    const MaterializationBoundary &boundary =
        spatial_plan->materialization_boundaries[static_cast<size_t>(
            plan->materialization_boundary_index)];
    ICHECK_EQ(plan->materialization_boundary, boundary->name)
        << "TTMaterializationPlan materialization_boundary must match "
           "SpatialPlan index";
    ICHECK(!boundary->event_lifetime_kind.empty())
        << "TTMaterializationPlan requires SpatialPlan MaterializationBoundary "
           "lifetime";
    ICHECK_GE(boundary->min_publish_pages, 1)
        << "TTMaterializationPlan requires bounded publish pages";
    auto source_it = live_value_name_by_form.find(
        static_cast<std::string>(plan->source_live_form));
    ICHECK(source_it != live_value_name_by_form.end())
        << "TTMaterializationPlan source_live_form missing matching "
           "TTLiveFormPlan";
    ICHECK_EQ(source_it->second,
              static_cast<std::string>(boundary->source_live_value))
        << "TTMaterializationPlan source_live_form must refer to boundary "
           "source_live_value";
    ICHECK_LT(boundary->target_live_value_index,
              static_cast<int64_t>(spatial_plan->live_values.size()))
        << "MaterializationBoundary target_live_value_index out of bounds";
    const LiveValue &target_live_value =
        spatial_plan->live_values[static_cast<size_t>(
            boundary->target_live_value_index)];
    ICHECK_EQ(boundary->target_live_value, target_live_value->name)
        << "MaterializationBoundary target_live_value must match SpatialPlan "
           "index";
    ICHECK_EQ(plan->target_buffer, target_live_value->subject)
        << "TTMaterializationPlan target_buffer must refer to boundary "
           "target_live_value";
  }

  for (const TTConsumerBindingPlan &plan : program->consumer_binding_plans) {
    ICHECK_LT(plan->live_value_edge_index,
              static_cast<int64_t>(spatial_plan->live_value_edges.size()))
        << "TTConsumerBindingPlan live_value_edge_index out of bounds";
    const LiveValueEdge &live_edge =
        spatial_plan->live_value_edges[static_cast<size_t>(
            plan->live_value_edge_index)];
    ICHECK_EQ(plan->live_value_edge, live_edge->name)
        << "TTConsumerBindingPlan live_value_edge must match SpatialPlan index";
    ICHECK(!live_edge->use_kind.empty())
        << "TTConsumerBindingPlan requires SpatialPlan LiveValueEdge use_kind";
    ICHECK_GE(live_edge->source_version_index, 0)
        << "TTConsumerBindingPlan requires SpatialPlan source version";
    auto source_it = live_value_name_by_form.find(
        static_cast<std::string>(plan->source_live_form));
    ICHECK(source_it != live_value_name_by_form.end())
        << "TTConsumerBindingPlan source_live_form missing matching "
           "TTLiveFormPlan";
    ICHECK_EQ(source_it->second,
              static_cast<std::string>(live_edge->source_live_value))
        << "TTConsumerBindingPlan source_live_form must refer to edge "
           "source_live_value";
  }
}

bool HasExactCBSlices(const TTProgram &program) {
  return !program->exact_cb_virtual_values.empty() ||
         !program->exact_cb_use_events.empty() ||
         !program->exact_cb_live_intervals.empty() ||
         !program->exact_cb_allocations.empty() ||
         !program->exact_cb_release_events.empty();
}

void ValidateExactCBLifecycleRecords(
    const TTProgram &program,
    const std::unordered_set<std::string> &kernel_names) {
  if (!HasExactCBSlices(program)) {
    return;
  }

  ICHECK(!program->exact_cb_virtual_values.empty())
      << "TTProgram exact-CB lifecycle requires virtual values";

  std::unordered_map<std::string, int64_t> live_form_index_by_name;
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->live_form_plans.size());
       ++index) {
    live_form_index_by_name.emplace(str(program->live_form_plans[index]->name),
                                    index);
  }

  std::unordered_map<std::string, int64_t> cb_plan_index_by_name;
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->cb_plans.size()); ++index) {
    cb_plan_index_by_name.emplace(str(program->cb_plans[index]->name), index);
  }

  std::unordered_map<std::string, int64_t> virtual_value_index_by_name;
  std::unordered_map<int64_t, int64_t> last_use_by_virtual_value_index;
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->exact_cb_virtual_values.size());
       ++index) {
    const TTExactCBVirtualValue &value =
        program->exact_cb_virtual_values[index];
    ICHECK(!value->name.empty()) << "TTExactCBVirtualValue requires name";
    ICHECK(virtual_value_index_by_name.emplace(str(value->name), index).second)
        << "duplicate TTExactCBVirtualValue name " << value->name;
    ICHECK(!value->logical_value.empty())
        << "TTExactCBVirtualValue requires logical_value";
    ICHECK(!value->live_form.empty())
        << "TTExactCBVirtualValue requires live_form";
    ICHECK_GE(value->live_form_index, 0)
        << "TTExactCBVirtualValue requires live_form_index";
    ICHECK_LT(value->live_form_index,
              static_cast<int64_t>(program->live_form_plans.size()))
        << "TTExactCBVirtualValue live_form_index out of bounds";
    const TTLiveFormPlan &live_form =
        program->live_form_plans[static_cast<size_t>(value->live_form_index)];
    ICHECK_EQ(value->live_form, live_form->name)
        << "TTExactCBVirtualValue live_form must match indexed TTLiveFormPlan";
    ICHECK_EQ(value->logical_value, live_form->logical_value)
        << "TTExactCBVirtualValue logical_value must match live_form";
    ICHECK(live_form_index_by_name.count(str(value->live_form)))
        << "TTExactCBVirtualValue references unknown live_form "
        << value->live_form;
    ICHECK(!value->producer_kernel.empty())
        << "TTExactCBVirtualValue requires producer_kernel";
    ICHECK(kernel_names.count(str(value->producer_kernel)))
        << "TTExactCBVirtualValue references unknown producer_kernel "
        << value->producer_kernel;
    ICHECK(!value->producer_event.empty())
        << "TTExactCBVirtualValue requires producer_event";
    ICHECK(!value->event_lifetime_kind.empty())
        << "TTExactCBVirtualValue requires event_lifetime_kind";
    ICHECK(!value->loop_role.empty())
        << "TTExactCBVirtualValue requires loop_role";
    ICHECK_GT(value->num_pages, 0)
        << "TTExactCBVirtualValue requires positive num_pages";
    ICHECK_GT(value->page_size_bytes, 0)
        << "TTExactCBVirtualValue requires positive page_size_bytes";
    ICHECK(!value->data_format.empty())
        << "TTExactCBVirtualValue requires data_format";
  }

  auto check_virtual_reference = [&](const ffi::String &name,
                                     int64_t index,
                                     const char *context) {
    ICHECK(!name.empty()) << context << " requires virtual_value";
    ICHECK_GE(index, 0) << context << " requires virtual_value_index";
    ICHECK_LT(index,
              static_cast<int64_t>(program->exact_cb_virtual_values.size()))
        << context << " virtual_value_index out of bounds";
    const TTExactCBVirtualValue &value =
        program->exact_cb_virtual_values[static_cast<size_t>(index)];
    ICHECK_EQ(name, value->name)
        << context
        << " virtual_value must match indexed TTExactCBVirtualValue";
  };

  for (const TTExactCBUseEvent &event : program->exact_cb_use_events) {
    ICHECK(!event->name.empty()) << "TTExactCBUseEvent requires name";
    check_virtual_reference(event->virtual_value, event->virtual_value_index,
                            "TTExactCBUseEvent");
    ICHECK(!event->consumer_kernel.empty())
        << "TTExactCBUseEvent requires consumer_kernel";
    ICHECK(kernel_names.count(str(event->consumer_kernel)))
        << "TTExactCBUseEvent references unknown consumer_kernel "
        << event->consumer_kernel;
    ICHECK(!event->consumer_event.empty())
        << "TTExactCBUseEvent requires consumer_event";
    ICHECK(!event->operand_role.empty())
        << "TTExactCBUseEvent requires operand_role";
    ICHECK_GE(event->program_point, 0)
        << "TTExactCBUseEvent requires program_point";
    ICHECK(!event->borrow_kind.empty())
        << "TTExactCBUseEvent requires borrow_kind";
    int64_t &last_use = last_use_by_virtual_value_index[event->virtual_value_index];
    last_use = std::max(last_use, event->program_point);
  }

  for (const TTExactCBLiveInterval &interval :
       program->exact_cb_live_intervals) {
    ICHECK(!interval->name.empty()) << "TTExactCBLiveInterval requires name";
    check_virtual_reference(interval->virtual_value,
                            interval->virtual_value_index,
                            "TTExactCBLiveInterval");
    ICHECK_GE(interval->begin_point, 0)
        << "TTExactCBLiveInterval requires begin_point";
    ICHECK_GE(interval->end_point, interval->begin_point)
        << "TTExactCBLiveInterval requires end_point >= begin_point";
    ICHECK(!interval->interference_class.empty())
        << "TTExactCBLiveInterval requires interference_class";
    auto use_it =
        last_use_by_virtual_value_index.find(interval->virtual_value_index);
    if (use_it != last_use_by_virtual_value_index.end()) {
      ICHECK_GE(interval->end_point, use_it->second)
          << "TTExactCBLiveInterval end_point must cover last use";
    }
  }

  std::unordered_map<std::string, int64_t> allocation_index_by_name;
  std::unordered_map<int64_t, int64_t> allocation_virtual_value_index;
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->exact_cb_allocations.size());
       ++index) {
    const TTExactCBAllocation &allocation =
        program->exact_cb_allocations[index];
    ICHECK(!allocation->name.empty()) << "TTExactCBAllocation requires name";
    ICHECK(allocation_index_by_name.emplace(str(allocation->name), index)
               .second)
        << "duplicate TTExactCBAllocation name " << allocation->name;
    check_virtual_reference(allocation->virtual_value,
                            allocation->virtual_value_index,
                            "TTExactCBAllocation");
    ICHECK(!allocation->cb_plan.empty())
        << "TTExactCBAllocation requires cb_plan";
    ICHECK_GE(allocation->cb_plan_index, 0)
        << "TTExactCBAllocation requires cb_plan_index";
    ICHECK_LT(allocation->cb_plan_index,
              static_cast<int64_t>(program->cb_plans.size()))
        << "TTExactCBAllocation cb_plan_index out of bounds";
    const TTCBPlan &cb =
        program->cb_plans[static_cast<size_t>(allocation->cb_plan_index)];
    ICHECK_EQ(allocation->cb_plan, cb->name)
        << "TTExactCBAllocation cb_plan must match indexed TTCBPlan";
    ICHECK(cb_plan_index_by_name.count(str(allocation->cb_plan)))
        << "TTExactCBAllocation references unknown cb_plan "
        << allocation->cb_plan;
    ICHECK_EQ(allocation->physical_cb_id, cb->cb_id)
        << "TTExactCBAllocation physical_cb_id must match TTCBPlan cb_id";
    ICHECK_GT(allocation->page_count, 0)
        << "TTExactCBAllocation requires positive page_count";
    ICHECK_LE(allocation->page_count, cb->num_pages)
        << "TTExactCBAllocation page_count must fit TTCBPlan num_pages";
    ICHECK_GE(allocation->release_program_point, 0)
        << "TTExactCBAllocation requires release_program_point";
    ICHECK(!allocation->release_reason.empty())
        << "TTExactCBAllocation requires release_reason";
    auto use_it =
        last_use_by_virtual_value_index.find(allocation->virtual_value_index);
    if (use_it != last_use_by_virtual_value_index.end()) {
      ICHECK_GE(allocation->release_program_point, use_it->second)
          << "TTExactCBAllocation must not release before last use";
    }
    allocation_virtual_value_index.emplace(index,
                                           allocation->virtual_value_index);
  }

  for (const TTExactCBReleaseEvent &event :
       program->exact_cb_release_events) {
    ICHECK(!event->name.empty()) << "TTExactCBReleaseEvent requires name";
    ICHECK(!event->allocation.empty())
        << "TTExactCBReleaseEvent requires allocation";
    ICHECK_GE(event->allocation_index, 0)
        << "TTExactCBReleaseEvent requires allocation_index";
    ICHECK_LT(event->allocation_index,
              static_cast<int64_t>(program->exact_cb_allocations.size()))
        << "TTExactCBReleaseEvent allocation_index out of bounds";
    const TTExactCBAllocation &allocation =
        program->exact_cb_allocations[static_cast<size_t>(
            event->allocation_index)];
    ICHECK_EQ(event->allocation, allocation->name)
        << "TTExactCBReleaseEvent allocation must match indexed allocation";
    ICHECK(!event->cb_plan.empty()) << "TTExactCBReleaseEvent requires cb_plan";
    ICHECK_GE(event->cb_plan_index, 0)
        << "TTExactCBReleaseEvent requires cb_plan_index";
    ICHECK_EQ(event->cb_plan_index, allocation->cb_plan_index)
        << "TTExactCBReleaseEvent cb_plan_index must match allocation";
    ICHECK_EQ(event->cb_plan, allocation->cb_plan)
        << "TTExactCBReleaseEvent cb_plan must match allocation";
    ICHECK_GE(event->program_point, 0)
        << "TTExactCBReleaseEvent requires program_point";
    ICHECK_GT(event->page_count, 0)
        << "TTExactCBReleaseEvent requires positive page_count";
    ICHECK_LE(event->page_count, allocation->page_count)
        << "TTExactCBReleaseEvent page_count must fit allocation";
    ICHECK(!event->reason.empty()) << "TTExactCBReleaseEvent requires reason";
    const int64_t virtual_value_index =
        allocation_virtual_value_index.at(event->allocation_index);
    auto use_it = last_use_by_virtual_value_index.find(virtual_value_index);
    if (use_it != last_use_by_virtual_value_index.end()) {
      ICHECK_GE(event->program_point, use_it->second)
          << "TTExactCBReleaseEvent must not release before last use";
    }
    ICHECK_EQ(event->program_point, allocation->release_program_point)
        << "TTExactCBReleaseEvent program_point must match allocation release";
    ICHECK_EQ(event->reason, allocation->release_reason)
        << "TTExactCBReleaseEvent reason must match allocation";
  }
}

void CheckTTProgram(
    const TTProgram &program, const SpatialPlan &spatial_plan,
    const std::optional<TTHardwareModel> &maybe_hardware_model) {
  ICHECK(!program->entry_name.empty()) << "TTProgram requires entry_name";
  ICHECK(!program->mesh_plans.empty())
      << "TTProgram requires at least one TTMeshPlan";
  ICHECK(!program->buffer_distribution_plans.empty())
      << "TTProgram requires at least one TTBufferDistributionPlan";
  ICHECK(!program->tensor_memory_config_plans.empty())
      << "TTProgram requires at least one TTTensorMemoryConfigPlan";
  ICHECK(!program->block_plans.empty())
      << "TTProgram requires at least one TTBlockPlan";
  ICHECK(!program->kernel_plans.empty())
      << "TTProgram requires at least one TTKernelPlan";
  ICHECK(!program->kernels.empty())
      << "TTProgram requires at least one TTKernel";
  ICHECK(!program->core_groups.empty())
      << "TTProgram requires at least one TTCoreGroup";
  ICHECK(!program->abi_plans.empty())
      << "TTProgram requires at least one TTABIPlan";
  ICHECK(!program->execution_plans.empty())
      << "TTProgram requires at least one TTExecutionPlan";
  ICHECK_EQ(program->block_plans.size(), program->core_groups.size())
      << "TTProgram requires aligned TTBlockPlan and TTCoreGroup owner truth";
  ICHECK_EQ(program->kernel_plans.size(), program->kernels.size())
      << "TTProgram requires aligned TTKernelPlan and TTKernel owner truth";
  ICHECK_EQ(program->sync_plans.size(), program->compute_sync_plans.size())
      << "TTProgram requires aligned TTSyncPlan and TTComputeSyncPlan owner "
         "truth";

  std::unordered_map<std::string, int64_t> mesh_index_by_name;
  for (int64_t mesh_index = 0;
       mesh_index < static_cast<int64_t>(program->mesh_plans.size());
       ++mesh_index) {
    const TTMeshPlan &mesh_plan = program->mesh_plans[mesh_index];
    ValidateMeshPlan(mesh_plan);
    ICHECK(mesh_index_by_name
               .emplace(static_cast<std::string>(mesh_plan->name), mesh_index)
               .second)
        << "duplicate TTMeshPlan name " << mesh_plan->name;
  }

  std::unordered_set<std::string> spatial_layout_subjects;
  for (const LayoutSpec &layout : spatial_plan->layout_specs) {
    spatial_layout_subjects.insert(static_cast<std::string>(layout->subject));
  }
  std::unordered_map<std::string, int64_t> core_group_index_by_name;
  std::unordered_map<std::string, TTCoreGroup> core_group_by_name;
  for (int64_t core_group_index = 0;
       core_group_index < static_cast<int64_t>(program->core_groups.size());
       ++core_group_index) {
    const TTCoreGroup &core_group = program->core_groups[core_group_index];
    const std::string core_group_name =
        static_cast<std::string>(core_group->name);
    ICHECK(core_group_index_by_name
               .emplace(core_group_name, core_group_index)
               .second)
        << "duplicate TTCoreGroup name " << core_group->name;
    core_group_by_name.emplace(core_group_name, core_group);
  }
  std::unordered_set<std::string> distributed_buffers;
  std::unordered_set<std::string> required_reshard_edges;
  std::unordered_map<std::string, TTBufferDistributionPlan> distribution_by_buffer;
  std::unordered_map<std::string, int64_t> distribution_index_by_name;
  for (const TTBufferDistributionPlan &distribution :
       program->buffer_distribution_plans) {
    ValidateBufferDistributionPlan(distribution, mesh_index_by_name,
                                   core_group_index_by_name,
                                   maybe_hardware_model);
    ICHECK(distributed_buffers
               .insert(static_cast<std::string>(distribution->buffer))
               .second)
        << "duplicate TTBufferDistributionPlan buffer " << distribution->buffer;
    distribution_by_buffer.emplace(str(distribution->buffer), distribution);
    if (!distribution->source_buffer.empty()) {
      required_reshard_edges.insert(str(distribution->source_buffer) + "|" +
                                    str(distribution->buffer));
    }
    distribution_index_by_name.emplace(str(distribution->name),
                                       static_cast<int64_t>(distribution_index_by_name.size()));
    ICHECK(spatial_layout_subjects.count(
        static_cast<std::string>(distribution->buffer)))
        << "TTBufferDistributionPlan buffer must match SpatialPlan LayoutSpec "
           "subject "
        << distribution->buffer;
  }

  std::unordered_set<std::string> memory_config_subjects;
  std::unordered_set<std::string> memory_config_names;
  for (const TTTensorMemoryConfigPlan &memory_config :
       program->tensor_memory_config_plans) {
    ValidateTensorMemoryConfigPlan(memory_config, distribution_index_by_name,
                                   distribution_by_buffer);
    ICHECK(memory_config_names.insert(str(memory_config->name)).second)
        << "duplicate TTTensorMemoryConfigPlan name " << memory_config->name;
    ICHECK(memory_config_subjects.insert(str(memory_config->subject)).second)
        << "duplicate TTTensorMemoryConfigPlan subject "
        << memory_config->subject;
  }
  for (const std::string &buffer : distributed_buffers) {
    ICHECK(memory_config_subjects.count(buffer) != 0U)
        << "TTBufferDistributionPlan buffer " << buffer
        << " requires TTTensorMemoryConfigPlan";
  }

  for (const TTBlockPlan &block_plan : program->block_plans) {
    ValidateBlockPlan(block_plan);
  }
  for (const TTSyncPlan &sync_plan : program->sync_plans) {
    ValidateSyncPlan(sync_plan);
  }

  std::unordered_set<std::string> kernel_names;
  for (const TTKernel &kernel : program->kernels) {
    ICHECK(!kernel->name.empty()) << "TTKernel requires name";
    ICHECK(!kernel->kind.empty()) << "TTKernel requires kind";
    ICHECK(!kernel->core_type.empty()) << "TTKernel requires core_type";
    ICHECK_GE(kernel->abi_plan_index, 0) << "TTKernel requires abi_plan_index";
    ICHECK_LT(kernel->abi_plan_index,
              static_cast<int64_t>(program->abi_plans.size()))
        << "TTKernel abi_plan_index out of bounds";
    ICHECK(kernel_names.insert(kernel->name).second)
        << "duplicate TTKernel name " << kernel->name;
    ValidateKernelLeafFields(kernel);
  }
  for (const TTKernelPlan &kernel_plan : program->kernel_plans) {
    ValidateKernelPlan(kernel_plan,
                       static_cast<int64_t>(program->abi_plans.size()),
                       static_cast<int64_t>(program->block_plans.size()));
    ICHECK(kernel_names.count(kernel_plan->name))
        << "TTKernelPlan missing matching TTKernel owner truth: "
        << kernel_plan->name;
  }
  std::unordered_set<std::string> compute_op_names;
  std::unordered_set<std::string> expected_op_contract_keys;
  std::unordered_set<std::string> tensor_memory_config_subjects;
  for (const TTTensorMemoryConfigPlan &memory_config :
       program->tensor_memory_config_plans) {
    if (!memory_config->subject.empty()) {
      tensor_memory_config_subjects.insert(str(memory_config->subject));
    }
  }
  for (int64_t compute_op_index = 0;
       compute_op_index < static_cast<int64_t>(program->compute_op_plans.size());
       ++compute_op_index) {
    const TTComputeOpPlan &compute_op_plan =
        program->compute_op_plans[static_cast<size_t>(compute_op_index)];
    ValidateComputeOpPlan(compute_op_plan,
                          static_cast<int64_t>(program->kernel_plans.size()),
                          kernel_names);
    ICHECK(
        compute_op_names.insert(static_cast<std::string>(compute_op_plan->name))
            .second)
        << "duplicate TTComputeOpPlan name " << compute_op_plan->name;
    for (const TTComputeOperandBindingPlan &binding :
         compute_op_plan->operand_bindings) {
      if (!tensor_memory_config_subjects.count(str(binding->buffer))) {
        continue;
      }
      expected_op_contract_keys.insert(str(compute_op_plan->name) + "|" +
                                       str(binding->role));
    }
  }
  std::unordered_set<std::string> op_contract_names;
  std::unordered_set<std::string> op_contract_keys;
  for (const TTOpShardingContract &contract : program->op_sharding_contracts) {
    ValidateOpShardingContract(contract, program->compute_op_plans,
                               program->tensor_memory_config_plans);
    ICHECK(op_contract_names.insert(str(contract->name)).second)
        << "duplicate TTOpShardingContract name " << contract->name;
    op_contract_keys.insert(str(contract->compute_op_plan) + "|" +
                            str(contract->operand_role));
  }
  for (const std::string &key : expected_op_contract_keys) {
    ICHECK(op_contract_keys.count(key))
        << "TTComputeOpPlan operand requires TTOpShardingContract " << key;
  }
  std::unordered_set<std::string> placement_resolution_names;
  std::unordered_set<int64_t> resolved_contract_indices;
  for (const TTPlacementResolutionPlan &resolution :
       program->placement_resolution_plans) {
    ValidatePlacementResolutionPlan(resolution, program->op_sharding_contracts,
                                    program->tensor_memory_config_plans);
    ICHECK(placement_resolution_names.insert(str(resolution->name)).second)
        << "duplicate TTPlacementResolutionPlan name " << resolution->name;
    ICHECK(resolved_contract_indices
               .insert(resolution->op_sharding_contract_index)
               .second)
        << "duplicate TTPlacementResolutionPlan for op contract "
        << resolution->op_sharding_contract;
  }
  for (int64_t index = 0;
       index < static_cast<int64_t>(program->op_sharding_contracts.size());
       ++index) {
    ICHECK(resolved_contract_indices.count(index))
        << "TTOpShardingContract requires TTPlacementResolutionPlan index "
        << index;
  }

  std::unordered_map<std::string, const TTResourcePressureReportNode *>
      resource_report_by_kernel;
  std::unordered_set<std::string> resource_demand_kernels;
  for (const TTResourcePressureReport &report :
       program->resource_pressure_reports) {
    ValidateResourcePressureReport(
        report, kernel_names, static_cast<int64_t>(program->core_groups.size()),
        maybe_hardware_model);
    const std::string kernel_name = report->kernel_name;
    ICHECK(resource_report_by_kernel.emplace(kernel_name, report.get()).second)
        << "duplicate TTResourcePressureReport for kernel "
        << report->kernel_name;
  }
  for (const TTResourceDemand &demand : program->resource_demands) {
    ValidateResourceDemand(demand, kernel_names,
                           static_cast<int64_t>(program->core_groups.size()));
    ICHECK(resource_demand_kernels
               .insert(static_cast<std::string>(demand->kernel_name))
               .second)
        << "duplicate TTResourceDemand for kernel " << demand->kernel_name;
    auto report_it = resource_report_by_kernel.find(
        static_cast<std::string>(demand->kernel_name));
    ICHECK(report_it != resource_report_by_kernel.end())
        << "TTResourceDemand requires matching ResourcePressureReport for "
           "kernel "
        << demand->kernel_name;
    ICHECK_GE(report_it->second->required_materializations.size(),
              demand->tile_compute_materialization_demands.size())
        << "ResourcePressureReport required_materializations must cover "
           "TTResourceDemand tile_compute_materialization_demands";
  }
  for (const auto &entry : resource_report_by_kernel) {
    ICHECK(resource_demand_kernels.count(entry.first))
        << "TTResourcePressureReport requires matching TTResourceDemand for "
           "kernel "
        << entry.first;
  }

  for (const TTCoreGroup &core_group : program->core_groups) {
    ValidateCoreGroup(core_group, maybe_hardware_model);
  }

  std::unordered_set<int64_t> cb_ids;
  for (const TTCBPlan &cb : program->cb_plans) {
    ValidateCBPlan(cb);
    ICHECK_GE(cb->cb_id, 0) << "TTCBPlan requires non-negative cb_id";
    ICHECK(cb_ids.insert(cb->cb_id).second)
        << "duplicate TTCBPlan cb_id " << cb->cb_id;
  }

  std::unordered_set<std::string> live_form_names;
  ValidateLiveFormPlans(program, &live_form_names);
  ValidateMaterializationPlans(program, live_form_names,
                               static_cast<int64_t>(program->cb_plans.size()));
  std::unordered_set<std::string> reshard_names;
  std::unordered_set<std::string> reshard_edges;
  for (const TTReshardPlan &reshard : program->reshard_plans) {
    ValidateReshardPlan(reshard, program->tensor_memory_config_plans,
                        program->materialization_plans);
    ICHECK(reshard_names.insert(str(reshard->name)).second)
        << "duplicate TTReshardPlan name " << reshard->name;
    ICHECK(reshard_edges
               .insert(str(reshard->source_value) + "|" +
                       str(reshard->target_value))
               .second)
        << "duplicate TTReshardPlan edge " << reshard->source_value << " -> "
        << reshard->target_value;
  }
  for (const std::string &edge : required_reshard_edges) {
    ICHECK(reshard_edges.count(edge))
        << "TTBufferDistributionPlan source_buffer edge requires TTReshardPlan "
        << edge;
  }

  std::unordered_set<std::string> abi_kernel_names;
  for (const TTABIPlan &abi : program->abi_plans) {
    ICHECK(!abi->kernel_name.empty()) << "TTABIPlan requires kernel_name";
    for (const TTAccessorSpec &accessor : abi->accessors) {
      ValidateAccessor(accessor);
      ICHECK(
          distributed_buffers.count(static_cast<std::string>(accessor->buffer)))
          << "TTABIPlan accessor buffer requires TTBufferDistributionPlan";
      ValidateShardedAccessorWorkMapping(accessor, distribution_by_buffer,
                                         core_group_by_name);
    }
    for (const TTCompileTimeArgSpec &spec : abi->compile_time_arg_specs) {
      ValidateCompileTimeArgSpec(spec);
    }
    abi_kernel_names.insert(abi->kernel_name);
  }
  for (const TTKernel &kernel : program->kernels) {
    ICHECK(abi_kernel_names.count(kernel->name))
        << "TTKernel missing matching TTABIPlan: " << kernel->name;
  }
  ValidateConsumerBindingPlans(program, live_form_names,
                               static_cast<int64_t>(program->abi_plans.size()));
  ValidateSpatialLiveReferences(program, spatial_plan);
  ValidateExactCBLifecycleRecords(program, kernel_names);

  for (const TTTransportPlan &transport : program->transport_plans) {
    ICHECK(!transport->kind.empty()) << "TTTransportPlan requires kind";
    ICHECK(!transport->value_kind.empty())
        << "TTTransportPlan requires value_kind";
    ICHECK(!transport->delivery_kind.empty())
        << "TTTransportPlan requires delivery_kind";
    ICHECK_GE(transport->source_task_index, 0)
        << "TTTransportPlan requires source_task_index";
    ICHECK_GE(transport->target_task_index, 0)
        << "TTTransportPlan requires target_task_index";
  }

  for (const TTSemaphorePlan &semaphore : program->semaphore_plans) {
    ICHECK_GE(semaphore->semaphore_id, 0)
        << "TTSemaphorePlan requires non-negative semaphore_id";
    ICHECK(!semaphore->kind.empty()) << "TTSemaphorePlan requires kind";
    ICHECK(!semaphore->core_type.empty())
        << "TTSemaphorePlan requires core_type";
  }

  for (const TTDstLayoutPlan &layout : program->dst_layout_plans) {
    ICHECK(!layout->buffer.empty()) << "TTDstLayoutPlan requires buffer";
    ICHECK(!layout->layout.empty()) << "TTDstLayoutPlan requires layout";
    ICHECK(!layout->memory_space.empty())
        << "TTDstLayoutPlan requires memory_space";
  }

  for (const TTExecutionPlan &execution : program->execution_plans) {
    ICHECK(!execution->kernel_names.empty())
        << "TTExecutionPlan requires kernel_names";
    ICHECK(!execution->phase_indices.empty())
        << "TTExecutionPlan requires phase_indices";
    for (const tvm::ffi::String &kernel_name : execution->kernel_names) {
      ICHECK(kernel_names.count(kernel_name))
          << "TTExecutionPlan references unknown kernel " << kernel_name;
    }
  }
}

} // namespace

tvm::transform::Pass ValidateTTProgram() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    const std::optional<TTHardwareModel> maybe_hardware_model =
        GetValidationHardwareModel(mod);
    for (const auto &[gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }
      auto maybe_program = func.value()->GetAttr<TTProgram>(attr::kTLTTProgram);
      if (!maybe_program) {
        continue;
      }
      auto maybe_spatial_plan =
          func.value()->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
      ICHECK(maybe_spatial_plan) << "ValidateTTProgram requires "
                                    "tl.spatial_plan for live-form validation";
      CheckTTProgram(maybe_program.value(), maybe_spatial_plan.value(),
                     maybe_hardware_model);
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "tl.transform.ValidateTTProgram", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateTTProgram", ValidateTTProgram);
}

} // namespace tl
} // namespace tvm
