/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_blackhole_state.cc
 * \brief Spatial live-form, materialization, and buffer-flow state for Blackhole lowering.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_utils.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::AttrStmt;
using tir::Buffer;
using tir::Call;
using tir::Evaluate;
using tir::PrimFunc;
using tir::SeqStmtNode;
using tir::Stmt;
using tir::StringImm;
using tir::VarNode;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_untilize_cb_front_tile_fragment;
using tvm::Bool;
using tvm::DataType;
using tvm::Integer;
using tvm::IntImm;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;

namespace {

Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

std::string GetStorageScope(const Buffer& buffer) {
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

int64_t StaticIntValueOrDefault(const PrimExpr& expr, int64_t default_value = 0) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return default_value;
}

std::optional<std::vector<int64_t>> ExtractStaticShape(const Array<PrimExpr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const PrimExpr& dim : shape) {
    const auto* imm = dim.as<IntImmNode>();
    if (!imm) {
      return std::nullopt;
    }
    dims.push_back(imm->value);
  }
  return dims;
}

int64_t ComputeStaticElementCount(const std::vector<int64_t>& shape) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    total_elements *= dim;
  }
  return total_elements;
}

bool IsUnsupportedResidualLocalScope(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

static std::unordered_map<std::string, Map<String, Any>> BuildLogicalTileLayoutSpecMap(
    const SpatialPlan& spatial_plan) {
  std::unordered_map<std::string, Map<String, Any>> specs_by_buffer;
  for (const LayoutSpec& layout : spatial_plan->layout_specs) {
    if (layout->logical_shape.empty()) {
      continue;
    }
    const std::string buffer_name = static_cast<std::string>(layout->subject);
    if (buffer_name.empty()) {
      continue;
    }
    Map<String, Any> spec;
    spec.Set(String(schema_key::kBuffer), layout->subject);
    spec.Set(String(schema_key::kScope), layout->scope);
    spec.Set(String(schema_key::kShape), layout->logical_shape);
    spec.Set(String(schema_key::kLocalShape), layout->local_shape);
    spec.Set(String(schema_key::kThreadExtent), layout->thread_extent);
    spec.Set(String(schema_key::kReplicateExtent), layout->replicate_extent);
    spec.Set(String(schema_key::kInverseLogicalIndexVars),
             layout->inverse_logical_index_vars);
    spec.Set(String(schema_key::kInverseLogicalIndexExprs),
             layout->inverse_logical_index_exprs);
    specs_by_buffer.emplace(buffer_name, std::move(spec));
  }
  return specs_by_buffer;
}

static int64_t ProductIntegerArrayField(const Map<String, Any>& map, const char* key,
                                        int64_t default_value = 0) {
  auto it = map.find(String(key));
  if (it == map.end()) {
    return default_value;
  }
  int64_t product = 1;
  for (const Integer& dim : Downcast<Array<Integer>>((*it).second)) {
    if (dim->value <= 0) {
      return default_value;
    }
    product *= dim->value;
  }
  return product;
}

}  // namespace

void PlanTTKernelABI::LoadLogicalTileLayoutSpecs(const SpatialPlan& spatial_plan) {
  logical_tile_layout_specs_by_buffer_ = BuildLogicalTileLayoutSpecMap(spatial_plan);
}

void PlanTTKernelABI::LoadSpatialLiveValueBoundaries(const SpatialPlan& plan) {
  spatial_live_value_by_subject_.clear();
  spatial_materialization_boundary_by_source_target_.clear();
  std::unordered_map<std::string, std::string> subject_by_live_value;
  std::unordered_map<std::string, int64_t> version_by_subject;

  for (int64_t i = 0; i < static_cast<int64_t>(plan->live_values.size()); ++i) {
    const LiveValue& live_value = plan->live_values[i];
    const std::string subject = static_cast<std::string>(live_value->subject);
    if (subject.empty()) {
      continue;
    }
    auto version_it = version_by_subject.find(subject);
    if (version_it == version_by_subject.end() ||
        live_value->version_index >= version_it->second) {
      version_by_subject[subject] = live_value->version_index;
      spatial_live_value_by_subject_[subject] =
          SpatialLiveValueRef{static_cast<std::string>(live_value->name), i};
    }
    subject_by_live_value.emplace(static_cast<std::string>(live_value->name), subject);
  }

  for (int64_t i = 0; i < static_cast<int64_t>(plan->materialization_boundaries.size()); ++i) {
    const MaterializationBoundary& boundary = plan->materialization_boundaries[i];
    const std::string source_live_value = static_cast<std::string>(boundary->source_live_value);
    const std::string target_live_value = static_cast<std::string>(boundary->target_live_value);
    auto source_subject_it = subject_by_live_value.find(source_live_value);
    auto target_subject_it = subject_by_live_value.find(target_live_value);
    if (source_subject_it == subject_by_live_value.end() ||
        target_subject_it == subject_by_live_value.end()) {
      continue;
    }
    const std::string key = source_subject_it->second + "->" + target_subject_it->second;
    if (spatial_materialization_boundary_by_source_target_.find(key) !=
        spatial_materialization_boundary_by_source_target_.end()) {
      continue;
    }
    spatial_materialization_boundary_by_source_target_[key] =
        SpatialMaterializationBoundaryRef{static_cast<std::string>(boundary->name),
                                          i,
                                          source_live_value,
                                          boundary->source_live_value_index,
                                          target_live_value,
                                          boundary->target_live_value_index,
                                          static_cast<std::string>(boundary->live_value_edge),
                                          boundary->live_value_edge_index};
  }
}

Stmt PlanTTKernelABI::MaybeWrapComputeSegment(const Stmt& stmt) const {
  if (!requires_compute_segment_ || !current_segment_kind_.empty()) {
    return stmt;
  }
  if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
    if (attr->attr_key == "blackhole.segment_kind") {
      return stmt;
    }
  }
  return AttrStmt(StringImm("blackhole.segment_kind"), "blackhole.segment_kind",
                  StringImm("compute"), stmt);
}

const Map<String, Any>* PlanTTKernelABI::FindLogicalTileLayoutSpec(const Buffer& buffer) const {
  const std::string buffer_name = BufferIdentityName(buffer);
  auto it = logical_tile_layout_specs_by_buffer_.find(buffer_name);
  if (it == logical_tile_layout_specs_by_buffer_.end()) {
    return nullptr;
  }
  return &it->second;
}

const PlanTTKernelABI::SpatialLiveValueRef* PlanTTKernelABI::FindSpatialLiveValueRef(
    const std::string& subject) const {
  auto it = spatial_live_value_by_subject_.find(subject);
  if (it == spatial_live_value_by_subject_.end()) {
    return nullptr;
  }
  return &it->second;
}

const PlanTTKernelABI::SpatialMaterializationBoundaryRef*
PlanTTKernelABI::FindSpatialMaterializationBoundaryRef(const std::string& source_subject,
                                                       const std::string& target_subject) const {
  const std::string key = source_subject + "->" + target_subject;
  auto it = spatial_materialization_boundary_by_source_target_.find(key);
  if (it == spatial_materialization_boundary_by_source_target_.end()) {
    return nullptr;
  }
  return &it->second;
}

const BlackholeBufferMaterializationFact* PlanTTKernelABI::FindBufferMaterializationFact(
    const Buffer& buffer) const {
  const std::string buffer_name = BufferIdentityName(buffer);
  auto it = buffer_materialization_facts_by_target_buffer_.find(buffer_name);
  if (it == buffer_materialization_facts_by_target_buffer_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool PlanTTKernelABI::BufferUsesTiledCBLiveForm(const Buffer& buffer) const {
  auto fact_uses_tiled_cb = [](const BlackholeBufferMaterializationFact& fact) {
    return fact.result_live_form == buffer_live_form::kTiledCB;
  };

  if (const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(buffer);
      fact != nullptr && fact_uses_tiled_cb(*fact)) {
    return true;
  }

  const std::string buffer_name = BufferIdentityName(buffer);
  if (buffer_name.empty()) {
    return false;
  }
  for (const auto& [_, fact] : buffer_materialization_facts_by_target_buffer_) {
    if (fact.source_buffer != buffer_name) {
      continue;
    }
    if (fact_uses_tiled_cb(fact)) {
      return true;
    }
  }
  return false;
}

void PlanTTKernelABI::ValidatePublishedBufferSourceEdge(const Buffer& src,
                                                        const Buffer& dst) const {
  const std::string src_name = BufferIdentityName(src);
  const std::string dst_name = BufferIdentityName(dst);
  auto live_form_it = buffer_live_form_cb_by_buffer_identity_.find(src_name);
  if (live_form_it == buffer_live_form_cb_by_buffer_identity_.end()) {
    return;
  }
  const BlackholeBufferMaterializationFact* dst_fact = FindBufferMaterializationFact(dst);
  ICHECK(dst_fact != nullptr)
      << "PlanTTKernelABI requires buffer materialization fact for consumer "
      << dst_name << " when source " << src_name << " is carried via explicit live-form CB";
  ICHECK(!dst_fact->source_buffer.empty())
      << "PlanTTKernelABI requires explicit source_buffer in buffer materialization fact "
         "for consumer "
      << dst_name << " when source " << src_name << " is carried via explicit live-form CB";
  ICHECK_EQ(dst_fact->source_buffer, src_name)
      << "PlanTTKernelABI requires buffer materialization fact source_buffer to match "
         "consumer source "
         << src_name << " for " << dst_name;
}

void PlanTTKernelABI::AppendPublishedBufferSourceMaterialization(
    const Buffer& src, int current_order_index, std::vector<Stmt>* prefix,
    std::vector<Stmt>* suffix) {
  ICHECK(prefix != nullptr);
  ICHECK(suffix != nullptr);
  const std::string src_name = BufferIdentityName(src);
  auto live_form_it = buffer_live_form_cb_by_buffer_identity_.find(src_name);
  if (live_form_it == buffer_live_form_cb_by_buffer_identity_.end()) {
    return;
  }
  ICHECK(BufferUsesTiledCBLiveForm(src))
      << "PlanTTKernelABI requires explicit tiled_cb result_live_form for source " << src_name;
  const int cb_id = live_form_it->second;
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const CBRequirement& requirement = cb_requirements_.at(cb_id);
  const int num_tiles = std::max(
      1, requirement.consume_pages_per_event > 0 ? requirement.consume_pages_per_event
                                                 : requirement.num_pages);
  ICHECK_GT(requirement.page_size, 0)
      << "PlanTTKernelABI requires a positive page_size for live-form source " << src_name;
  ICHECK_GT(src->dtype.bytes(), 0)
      << "PlanTTKernelABI requires a valid dtype for live-form source " << src_name;
  const int tile_elements = requirement.page_size / src->dtype.bytes();
  ICHECK_GT(tile_elements, 0)
      << "PlanTTKernelABI requires positive tile element count for live-form source "
      << src_name;
  const Buffer physical_src = ResolvePhysicalComputeBuffer(src);
  prefix->push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(num_tiles)}));
  for (int tile = 0; tile < num_tiles; ++tile) {
    prefix->push_back(MakeBlackholeCall(blackhole_untilize_cb_front_tile_fragment(),
                                        {physical_src->data, IntImm32(cb_id), IntImm32(tile),
                                         IntImm32(tile * tile_elements)}));
  }

  const FutureBufferUses future_uses =
      ClassifyFutureLiveCBReadsBeforeNextWrite(src, current_order_index);
  if (!future_uses.has_compute_consume && !future_uses.has_transport_consume &&
      !future_uses.has_reference) {
    suffix->push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(num_tiles)}));
    ClearTiledCBLiveFormAliases(src);
  }
}

void PlanTTKernelABI::RecordFragmentCastMaterializationPlans(
    const FragmentCastMatch& match, const BlackholeBufferMaterializationFact& fact,
    int cb_requirement_index,
    const PrimExpr& num_elements_expr, const std::string& publication_protocol) {
  const std::string source_name =
      !fact.source_buffer.empty() ? fact.source_buffer : BufferIdentityName(match.src);
  const std::string target_name = BufferIdentityName(match.dst);
  if (source_name.empty() || target_name.empty()) {
    return;
  }
  const std::string kernel_name =
      !current_segment_kind_.empty()
          ? current_segment_kind_
          : (requires_compute_segment_ ? std::string("compute") : std::string("main"));
  int64_t logical_element_count =
      fact.logical_element_count > 0
          ? fact.logical_element_count
          : StaticIntValueOrDefault(num_elements_expr, GetLogicalBufferElementCount(match.dst));
  auto bridge_logical_extent = [&](const Buffer& buffer) {
    const Map<String, Any>* spec = FindLogicalTileLayoutSpec(buffer);
    if (spec != nullptr) {
      return ProductIntegerArrayField(*spec, schema_key::kShape, int64_t{0});
    }
    return int64_t{0};
  };
  logical_element_count =
      std::max(logical_element_count,
               std::max(bridge_logical_extent(match.src), bridge_logical_extent(match.dst)));
  auto bridge_local_extent = [&](const Buffer& buffer) {
    const Map<String, Any>* spec = FindLogicalTileLayoutSpec(buffer);
    if (spec != nullptr) {
      const int64_t local_extent =
          ProductIntegerArrayField(*spec, schema_key::kLocalShape, int64_t{0});
      if (local_extent > 0) {
        return local_extent;
      }
    }
    if (auto static_shape = ExtractStaticShape(buffer->shape)) {
      return ComputeStaticElementCount(static_shape.value());
    }
    return int64_t{0};
  };
  const int64_t source_local_extent = bridge_local_extent(match.src);
  const int64_t target_local_extent = bridge_local_extent(match.dst);
  const SpatialLiveValueRef* source_live_value_ref = FindSpatialLiveValueRef(source_name);
  const SpatialLiveValueRef* target_live_value_ref = FindSpatialLiveValueRef(target_name);
  const SpatialMaterializationBoundaryRef* source_boundary_ref =
      FindSpatialMaterializationBoundaryRef(source_name, target_name);
  ICHECK(source_live_value_ref != nullptr)
      << "PlanTTKernelABI requires SpatialPlan LiveValue for materialization source "
      << source_name;
  ICHECK(target_live_value_ref != nullptr)
      << "PlanTTKernelABI requires SpatialPlan LiveValue for materialization target "
      << target_name;
  ICHECK(source_boundary_ref != nullptr)
      << "PlanTTKernelABI requires SpatialPlan MaterializationBoundary for materialization "
         "source/target "
      << source_name << " -> " << target_name;
  SpatialLiveValueRef boundary_source_live_value_ref{source_boundary_ref->source_live_value,
                                                     source_boundary_ref->source_live_value_index};
  SpatialLiveValueRef boundary_target_live_value_ref{source_boundary_ref->target_live_value,
                                                     source_boundary_ref->target_live_value_index};

  auto has_live_form = [&](const std::string& name) {
    for (const TTLiveFormPlan& plan : tt_live_form_plans_) {
      if (static_cast<std::string>(plan->name) == name) {
        return true;
      }
    }
    return false;
  };
  auto push_live_form = [&](const std::string& logical_value, const std::string& physical_form,
                            int64_t physical_local_extent, const char* ownership_kind,
                            const SpatialLiveValueRef& spatial_live_value) {
    const std::string name = "live_form_" + logical_value;
    if (has_live_form(name)) {
      return;
    }
    tt_live_form_plans_.push_back(TTLiveFormPlan(
        String(name), String(logical_value), String(spatial_live_value.name),
        spatial_live_value.index, String(kernel_name), String(physical_form),
        String("thread_distributed"), physical_local_extent, logical_element_count,
        String(ownership_kind)));
  };

  push_live_form(source_name, "thread_distributed_slice", source_local_extent,
                 "producer_thread_lane", boundary_source_live_value_ref);
  push_live_form(target_name, "cb_materialized_tile", target_local_extent,
                 "materialized_cb_pages", boundary_target_live_value_ref);

  const std::string source_live_form = "live_form_" + source_name;
  const std::string produced_live_form = "live_form_" + target_name;
  const std::string materialization_name = "materialize_" + source_name + "_to_" + target_name;
  bool has_materialization = false;
  for (const TTMaterializationPlan& plan : tt_materialization_plans_) {
    if (static_cast<std::string>(plan->name) == materialization_name) {
      has_materialization = true;
      break;
    }
  }
  if (!has_materialization) {
    Array<Integer> required_cb_indices{Integer(cb_requirement_index)};
    Array<Integer> required_sync_indices;
    tt_materialization_plans_.push_back(TTMaterializationPlan(
        String(materialization_name), String(source_live_form), String(source_boundary_ref->name),
        source_boundary_ref->index, String(target_name), String(), String(kernel_name),
        String(fact.bridge_kind),
        String(fact.materialization_kind),
        String(buffer_materialization::kCBRepublish), String(publication_protocol),
        required_cb_indices, required_sync_indices,
        String(produced_live_form)));
  }

  const std::string binding_name = "consume_" + source_name + "_as_cast_fragment_slice";
  bool has_binding = false;
  for (const TTConsumerBindingPlan& plan : tt_consumer_binding_plans_) {
    if (static_cast<std::string>(plan->name) == binding_name) {
      has_binding = true;
      break;
    }
  }
  if (!has_binding) {
    tt_consumer_binding_plans_.push_back(TTConsumerBindingPlan(
        String(binding_name), String(kernel_name), String("cast_fragment_slice"),
        String(source_live_form), String(source_boundary_ref->live_value_edge),
        source_boundary_ref->live_value_edge_index, /*accepts_distributed_slice=*/true,
        /*requires_full_logical_tile=*/false, /*abi_plan_index=*/-1, String(target_name),
        String(materialization_name)));
  }
}

void PlanTTKernelABI::FinalizeConsumerBindingABIIndices() {
  if (tt_consumer_binding_plans_.empty() || tt_abi_plans_.empty()) {
    return;
  }
  std::unordered_map<std::string, int64_t> abi_index_by_kernel;
  for (int64_t i = 0; i < static_cast<int64_t>(tt_abi_plans_.size()); ++i) {
    abi_index_by_kernel[static_cast<std::string>(tt_abi_plans_[i]->kernel_name)] = i;
  }
  Array<TTConsumerBindingPlan> finalized;
  for (const TTConsumerBindingPlan& plan : tt_consumer_binding_plans_) {
    int64_t abi_plan_index = plan->abi_plan_index;
    if (abi_plan_index < 0) {
      auto it = abi_index_by_kernel.find(static_cast<std::string>(plan->consumer_kernel));
      if (it != abi_index_by_kernel.end()) {
        abi_plan_index = it->second;
      }
    }
    finalized.push_back(TTConsumerBindingPlan(
        plan->name, plan->consumer_kernel, plan->consumer_op_kind, plan->source_live_form,
        plan->live_value_edge, plan->live_value_edge_index, plan->accepts_distributed_slice,
        plan->requires_full_logical_tile, abi_plan_index, plan->target_buffer,
        plan->materialization_plan));
  }
  tt_consumer_binding_plans_ = finalized;
}

void PlanTTKernelABI::FinalizeMaterializationPlanHostBuffers() {
  if (tt_materialization_plans_.empty()) {
    return;
  }

  std::unordered_set<std::string> accessor_buffers;
  for (const AccessorDescriptor& accessor : accessor_descriptors_) {
    if (!accessor.buffer_name.empty()) {
      accessor_buffers.insert(accessor.buffer_name);
    }
  }

  Array<TTMaterializationPlan> finalized;
  for (const TTMaterializationPlan& plan : tt_materialization_plans_) {
    const std::string target_buffer = static_cast<std::string>(plan->target_buffer);
    std::string host_buffer = static_cast<std::string>(plan->host_buffer);

    auto mapped_host = host_buffer_by_compute_operand_buffer_.find(target_buffer);
    if (mapped_host != host_buffer_by_compute_operand_buffer_.end() && !mapped_host->second.empty()) {
      host_buffer = mapped_host->second;
    } else if (host_buffer.empty() && accessor_buffers.count(target_buffer)) {
      host_buffer = target_buffer;
    }

    finalized.push_back(TTMaterializationPlan(
        plan->name, plan->source_live_form, plan->materialization_boundary,
        plan->materialization_boundary_index, plan->target_buffer, String(host_buffer),
        plan->target_kernel, plan->bridge_kind, plan->materialization_kind,
        plan->materialization_protocol, plan->publication_protocol,
        plan->required_cb_plan_indices, plan->required_sync_plan_indices,
        plan->produced_live_form));
  }
  tt_materialization_plans_ = finalized;
}

void PlanTTKernelABI::LoadPhysicalComputeBufferBindings(const PrimFunc& func) {
  compute_physical_buffers_by_data_.clear();
  compute_physical_buffers_by_identity_.clear();

  std::unordered_map<const VarNode*, std::vector<Buffer>> buffers_by_data;
  std::unordered_map<std::string, std::vector<Buffer>> buffers_by_identity;

  auto remember = [&](const Buffer& buffer) {
    if (!buffer.defined() || !IsUnsupportedResidualLocalScope(buffer)) {
      return;
    }
    if (const auto* data = BufferDataIdentity(buffer)) {
      auto& group = buffers_by_data[data];
      if (std::find(group.begin(), group.end(), buffer) == group.end()) {
        group.push_back(buffer);
      }
      return;
    }
    const std::string identity = BufferIdentityName(buffer);
    if (identity.empty()) {
      return;
    }
    auto& group = buffers_by_identity[identity];
    if (std::find(group.begin(), group.end(), buffer) == group.end()) {
      group.push_back(buffer);
    }
  };

  for (const auto& [_, buffer] : func->buffer_map) {
    remember(buffer);
  }
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (const auto* block = node.as<tir::BlockNode>()) {
      for (const Buffer& buffer : block->alloc_buffers) {
        remember(buffer);
      }
      return;
    }
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      remember(store->buffer);
      return;
    }
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      remember(load->buffer);
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call) {
      return;
    }
    for (const PrimExpr& arg : call->args) {
      if (IsBufferLikeExpr(arg)) {
        remember(NormalizeToBufferRegion(arg)->buffer);
      }
    }
  });

  auto preferred_scope_rank = [](const Buffer& buffer) {
    const std::string scope = buffer.scope();
    if (scope == "blackhole.acc") {
      return 3;
    }
    if (scope == "local.fragment") {
      return 2;
    }
    if (scope == "local") {
      return 1;
    }
    return 0;
  };
  auto choose_preferred_buffer = [&](const std::vector<Buffer>& group) -> Optional<Buffer> {
    Optional<Buffer> preferred;
    int preferred_rank = -1;
    for (const Buffer& candidate : group) {
      const int rank = preferred_scope_rank(candidate);
      if (!preferred || rank > preferred_rank) {
        preferred = candidate;
        preferred_rank = rank;
      }
    }
    return preferred;
  };

  for (const auto& [data, group] : buffers_by_data) {
    Optional<Buffer> preferred = choose_preferred_buffer(group);
    if (!preferred) {
      continue;
    }
    compute_physical_buffers_by_data_[data] = preferred.value();
    for (const Buffer& buffer : group) {
      const std::string identity = BufferIdentityName(buffer);
      if (!identity.empty()) {
        compute_physical_buffers_by_identity_[identity] = preferred.value();
      }
    }
  }
  for (const auto& [identity, group] : buffers_by_identity) {
    if (compute_physical_buffers_by_identity_.count(identity)) {
      continue;
    }
    Optional<Buffer> preferred = choose_preferred_buffer(group);
    if (preferred) {
      compute_physical_buffers_by_identity_[identity] = preferred.value();
    }
  }
}

Buffer PlanTTKernelABI::ResolvePhysicalComputeBuffer(const Buffer& buffer) const {
  if (!buffer.defined()) {
    return buffer;
  }
  if (buffer.scope() == "blackhole.acc") {
    return buffer;
  }
  if (const auto* data = BufferDataIdentity(buffer)) {
    auto by_data = compute_physical_buffers_by_data_.find(data);
    if (by_data != compute_physical_buffers_by_data_.end()) {
      return by_data->second;
    }
  }
  const std::string identity = BufferIdentityName(buffer);
  auto by_identity = compute_physical_buffers_by_identity_.find(identity);
  if (by_identity != compute_physical_buffers_by_identity_.end()) {
    return by_identity->second;
  }
  return buffer;
}

void PlanTTKernelABI::RecordTiledCBLiveFormAliases(const Buffer& buffer, int cb_id) {
  if (!buffer.defined() || cb_id < 0) {
    return;
  }
  const int order_index = current_lowering_order_index_;
  auto clear_exact_buffer = [&](const std::string& identity) {
    if (identity.empty()) {
      return false;
    }
    auto exact_order_it = exact_output_live_form_order_by_buffer_identity_.find(identity);
    if (order_index >= 0 &&
        exact_order_it != exact_output_live_form_order_by_buffer_identity_.end() &&
        exact_order_it->second > order_index) {
      return false;
    }
    exact_output_live_form_cb_by_buffer_identity_.erase(identity);
    if (order_index >= 0) {
      exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
    }
    return true;
  };
  auto record_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (!identity.empty()) {
      auto order_it = buffer_live_form_order_by_buffer_identity_.find(identity);
      if (order_index >= 0 && order_it != buffer_live_form_order_by_buffer_identity_.end() &&
          order_it->second > order_index) {
        return;
      }
      clear_exact_buffer(identity);
      buffer_live_form_cb_by_buffer_identity_[identity] = cb_id;
      buffer_live_form_order_by_buffer_identity_[identity] = order_index;
    }
  };
  record_buffer(buffer);
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined()) {
    record_buffer(physical);
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        clear_exact_buffer(identity);
        buffer_live_form_cb_by_buffer_identity_[identity] = cb_id;
        if (order_index >= 0) {
          buffer_live_form_order_by_buffer_identity_[identity] = order_index;
        }
      }
    }
  }
}

void PlanTTKernelABI::ClearTiledCBLiveFormIdentity(const std::string& identity) {
  if (identity.empty()) {
    return;
  }
  const int order_index = current_lowering_order_index_;
  auto order_it = buffer_live_form_order_by_buffer_identity_.find(identity);
  if (order_index >= 0 && order_it != buffer_live_form_order_by_buffer_identity_.end() &&
      order_it->second > order_index) {
    return;
  }
  buffer_live_form_cb_by_buffer_identity_.erase(identity);
  if (order_index >= 0) {
    buffer_live_form_order_by_buffer_identity_[identity] = order_index;
  }
  exact_output_live_form_cb_by_buffer_identity_.erase(identity);
  if (order_index >= 0) {
    exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
  }
}

void PlanTTKernelABI::ClearTiledCBLiveFormAliases(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    ClearTiledCBLiveFormIdentity(identity);
  }
}

void PlanTTKernelABI::InvalidateLastFragmentFillValue(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  auto erase_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (!identity.empty()) {
      last_fragment_fill_value_by_buffer_identity_.erase(identity);
    }
    if (const VarNode* data = BufferDataIdentity(candidate)) {
      last_fragment_fill_value_by_data_.erase(data);
    }
  };
  erase_buffer(buffer);
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined() && !physical.same_as(buffer)) {
    erase_buffer(physical);
  }
  if (physical.defined()) {
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        last_fragment_fill_value_by_buffer_identity_.erase(identity);
      }
    }
  }
}

void PlanTTKernelABI::ClearSelectedSourceLiveProducer(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  auto erase_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (!identity.empty()) {
      selected_source_live_producer_buffers_.erase(identity);
    }
  };
  erase_buffer(buffer);
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined() && !physical.same_as(buffer)) {
    erase_buffer(physical);
  }
  if (physical.defined()) {
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        selected_source_live_producer_buffers_.erase(identity);
      }
    }
  }
}

void PlanTTKernelABI::RecordSelectedSourceLiveProducer(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  ClearSelectedSourceLiveProducer(buffer);
  auto record_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (!identity.empty()) {
      selected_source_live_producer_buffers_.insert(identity);
    }
  };
  record_buffer(buffer);
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined() && !physical.same_as(buffer)) {
    record_buffer(physical);
  }
  if (physical.defined()) {
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        selected_source_live_producer_buffers_.insert(identity);
      }
    }
  }
}

bool PlanTTKernelABI::HasSelectedSourceLiveProducer(const Buffer& buffer) const {
  if (!buffer.defined()) {
    return false;
  }
  auto has_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    return !identity.empty() && selected_source_live_producer_buffers_.count(identity) != 0U;
  };
  if (has_buffer(buffer)) {
    return true;
  }
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined() && !physical.same_as(buffer) && has_buffer(physical)) {
    return true;
  }
  if (physical.defined()) {
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical) &&
          selected_source_live_producer_buffers_.count(identity) != 0U) {
        return true;
      }
    }
  }
  return false;
}

void PlanTTKernelABI::LoadBufferFlowFacts(
    const BlackholeLoweringSupportFacts& lowering_support_facts) {
  buffer_flow_facts_.clear();
  for (const BlackholeBufferFlowFact& fact : lowering_support_facts.buffer_flow_facts) {
    if (fact.buffer.empty()) {
      continue;
    }
    buffer_flow_facts_.emplace(fact.buffer, fact);
  }
}

std::vector<std::string> PlanTTKernelABI::CollectBufferFlowIdentities(
    const Buffer& buffer) const {
  std::vector<std::string> identities;
  auto add_identity = [&](const std::string& identity) {
    if (!identity.empty() &&
        std::find(identities.begin(), identities.end(), identity) == identities.end()) {
      identities.push_back(identity);
    }
  };
  if (!buffer.defined()) {
    return identities;
  }
  add_identity(BufferIdentityName(buffer));
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined()) {
    add_identity(BufferIdentityName(physical));
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        add_identity(identity);
      }
    }
  }
  return identities;
}

bool PlanTTKernelABI::HasInterveningBufferWrite(const Buffer& buffer,
                                                int live_order_index,
                                                int current_order_index) const {
  if (current_order_index < 0 || live_order_index < 0) {
    return false;
  }
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    auto it = buffer_flow_facts_.find(identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
          event.order_index > live_order_index &&
          event.order_index < current_order_index) {
        return true;
      }
    }
  }
  return false;
}

int PlanTTKernelABI::ResolveCurrentBufferTransferOrder(
    const Buffer& src, const Buffer& dst, int lower_bound_order_index) const {
  if (lower_bound_order_index < 0) {
    return lower_bound_order_index;
  }
  auto collect_orders = [&](const Buffer& buffer, bool want_write) {
    std::vector<int> orders;
    for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
      auto it = buffer_flow_facts_.find(identity);
      if (it == buffer_flow_facts_.end()) {
        continue;
      }
      for (const BlackholeBufferFlowEvent& event : it->second.events) {
        if (event.order_index < lower_bound_order_index) {
          continue;
        }
        const bool is_write = event.kind == BlackholeBufferFlowEventKind::kWrite;
        if (want_write != is_write) {
          continue;
        }
        if (std::find(orders.begin(), orders.end(), event.order_index) == orders.end()) {
          orders.push_back(event.order_index);
        }
      }
    }
    std::sort(orders.begin(), orders.end());
    return orders;
  };
  const std::vector<int> src_read_orders = collect_orders(src, /*want_write=*/false);
  const std::vector<int> dst_write_orders = collect_orders(dst, /*want_write=*/true);
  for (int src_order : src_read_orders) {
    if (std::find(dst_write_orders.begin(), dst_write_orders.end(), src_order) !=
        dst_write_orders.end()) {
      return src_order;
    }
  }
  return lower_bound_order_index;
}

PlanTTKernelABI::FutureBufferUses PlanTTKernelABI::ClassifyFutureBufferUses(
    const Buffer& buffer, int current_order_index) const {
  FutureBufferUses uses;
  const std::vector<std::string> identities = CollectBufferFlowIdentities(buffer);
  int next_write_order_index = -1;
  for (const std::string& buffer_identity : identities) {
    auto it = buffer_flow_facts_.find(buffer_identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.order_index <= current_order_index) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kWrite) {
        if (next_write_order_index < 0 || event.order_index < next_write_order_index) {
          next_write_order_index = event.order_index;
        }
      }
    }
  }
  for (const std::string& buffer_identity : identities) {
    auto it = buffer_flow_facts_.find(buffer_identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.order_index <= current_order_index) {
        continue;
      }
      if (next_write_order_index >= 0 && event.order_index > next_write_order_index) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kWrite) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kComputeConsume) {
        uses.has_compute_consume = true;
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kTransportConsume) {
        uses.has_transport_consume = true;
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kReference) {
        uses.has_reference = true;
      }
    }
  }
  return uses;
}

PlanTTKernelABI::FutureBufferUses
PlanTTKernelABI::ClassifyFutureLiveCBReadsBeforeNextWrite(
    const Buffer& buffer, int current_order_index) const {
  FutureBufferUses uses;
  const std::vector<std::string> identities = CollectBufferFlowIdentities(buffer);
  int next_write_order_index = -1;
  for (const std::string& buffer_identity : identities) {
    auto it = buffer_flow_facts_.find(buffer_identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.order_index <= current_order_index) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
          (next_write_order_index < 0 || event.order_index < next_write_order_index)) {
        next_write_order_index = event.order_index;
      }
    }
  }
  for (const std::string& buffer_identity : identities) {
    auto it = buffer_flow_facts_.find(buffer_identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.order_index <= current_order_index) {
        continue;
      }
      if (next_write_order_index >= 0 && event.order_index >= next_write_order_index) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kWrite) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kComputeConsume) {
        uses.has_compute_consume = true;
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kTransportConsume) {
        uses.has_transport_consume = true;
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kReference) {
        uses.has_reference = true;
      }
    }
  }
  return uses;
}

bool PlanTTKernelABI::ShouldRetainComputeInputBuffer(const Buffer& buffer,
                                                       int current_order_index) const {
  return ClassifyFutureBufferUses(buffer, current_order_index).has_compute_consume;
}

bool PlanTTKernelABI::ShouldReacquireComputeInputBuffer(const Buffer& buffer,
                                                          int current_order_index) const {
  if (GetStorageScope(buffer) != "blackhole.acc") {
    return false;
  }
  if (FindBufferMaterializationFact(buffer) != nullptr || BufferUsesTiledCBLiveForm(buffer)) {
    return false;
  }
  for (const std::string& buffer_identity : CollectBufferFlowIdentities(buffer)) {
    auto it = buffer_flow_facts_.find(buffer_identity);
    if (it == buffer_flow_facts_.end()) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& event : it->second.events) {
      if (event.order_index <= current_order_index) {
        continue;
      }
      if (event.kind == BlackholeBufferFlowEventKind::kWrite) {
        return true;
      }
      break;
    }
  }
  return false;
}

bool PlanTTKernelABI::ShouldPublishBufferResult(const Buffer& buffer,
                                                  int current_order_index) const {
  if (FindBufferMaterializationFact(buffer) != nullptr) {
    return true;
  }
  const FutureBufferUses uses = ClassifyFutureBufferUses(buffer, current_order_index);
  return uses.has_compute_consume || uses.has_transport_consume;
}

}  // namespace tl
}  // namespace tvm
