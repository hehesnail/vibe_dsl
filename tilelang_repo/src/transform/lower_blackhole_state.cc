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
#include "common/tt_live_form_solver.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/node/structural_equal.h>

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
using tir::SeqStmt;
using tir::SeqStmtNode;
using tir::Stmt;
using tir::StringImm;
using tir::Var;
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

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

bool IsBlackholeOpName(const tir::CallNode* op, const char* op_name) {
  if (op == nullptr || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  return Downcast<Op>(op->op)->name == op_name;
}

bool SamePrimExprArray(const ffi::Array<PrimExpr>& lhs,
                       const ffi::Array<PrimExpr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  StructuralEqual equal;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!equal(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
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
  spatial_materialization_boundaries_.clear();
  spatial_materialization_boundary_position_by_index_.clear();
  spatial_live_value_by_subject_.clear();
  spatial_lifetime_kind_by_subject_.clear();

  for (int64_t i = 0; i < static_cast<int64_t>(plan->live_values.size()); ++i) {
    const LiveValue& value = plan->live_values[i];
    const std::string subject = static_cast<std::string>(value->subject);
    if (!subject.empty()) {
      spatial_live_value_by_subject_.emplace(
          subject, SpatialLiveValueRef{static_cast<std::string>(value->name), i});
    }
  }

  auto lifetime_rank = [](const std::string& kind) {
    if (kind == "loop_carried") {
      return 3;
    }
    if (kind == "multi_event") {
      return 2;
    }
    if (kind == "single_event") {
      return 1;
    }
    return 0;
  };
  auto record_subject_lifetime = [&](const String& subject,
                                     const String& lifetime_kind) {
    const std::string subject_name = static_cast<std::string>(subject);
    const std::string lifetime = static_cast<std::string>(lifetime_kind);
    if (subject_name.empty() || lifetime.empty()) {
      return;
    }
    auto existing = spatial_lifetime_kind_by_subject_.find(subject_name);
    if (existing == spatial_lifetime_kind_by_subject_.end() ||
        lifetime_rank(lifetime) > lifetime_rank(existing->second)) {
      spatial_lifetime_kind_by_subject_[subject_name] = lifetime;
    }
  };

  for (int64_t i = 0; i < static_cast<int64_t>(plan->materialization_boundaries.size()); ++i) {
    const MaterializationBoundary& boundary = plan->materialization_boundaries[i];
    const std::string source_live_value = static_cast<std::string>(boundary->source_live_value);
    const std::string target_live_value = static_cast<std::string>(boundary->target_live_value);
    ICHECK_GE(boundary->source_live_value_index, 0)
        << "PlanTTKernelABI requires MaterializationBoundary source live-value index for "
        << boundary->name;
    ICHECK_GE(boundary->target_live_value_index, 0)
        << "PlanTTKernelABI requires MaterializationBoundary target live-value index for "
        << boundary->name;
    ICHECK_LT(boundary->source_live_value_index, static_cast<int64_t>(plan->live_values.size()))
        << "PlanTTKernelABI requires MaterializationBoundary source live-value index in bounds";
    ICHECK_LT(boundary->target_live_value_index, static_cast<int64_t>(plan->live_values.size()))
        << "PlanTTKernelABI requires MaterializationBoundary target live-value index in bounds";
    const LiveValue& source = plan->live_values[boundary->source_live_value_index];
    const LiveValue& target = plan->live_values[boundary->target_live_value_index];
    spatial_materialization_boundary_position_by_index_[i] =
        spatial_materialization_boundaries_.size();
    spatial_materialization_boundaries_.push_back(
        SpatialMaterializationBoundaryRef{static_cast<std::string>(boundary->name),
                                          i,
                                          source_live_value,
                                          boundary->source_live_value_index,
                                          static_cast<std::string>(source->subject),
                                          target_live_value,
                                          boundary->target_live_value_index,
                                          static_cast<std::string>(target->subject),
                                          static_cast<std::string>(boundary->live_value_edge),
                                          boundary->live_value_edge_index,
                                          static_cast<std::string>(boundary->logical_coverage),
                                          static_cast<std::string>(boundary->event_lifetime_kind),
                                          boundary->min_publish_pages,
                                          boundary->max_consume_pages});
    record_subject_lifetime(source->subject, boundary->event_lifetime_kind);
    record_subject_lifetime(target->subject, boundary->event_lifetime_kind);
  }
}

void PlanTTKernelABI::LoadSpatialAccessRegions(const SpatialPlan& plan) {
  spatial_access_regions_.clear();
  spatial_access_region_positions_by_subject_access_.clear();
  for (int64_t i = 0; i < static_cast<int64_t>(plan->access_regions.size()); ++i) {
    const AccessRegion& region = plan->access_regions[i];
    const std::string subject = static_cast<std::string>(region->subject);
    const std::string access_kind = static_cast<std::string>(region->access_kind);
    if (subject.empty() || access_kind.empty()) {
      continue;
    }
    const std::string key = subject + "|" + access_kind;
    spatial_access_region_positions_by_subject_access_[key].push_back(
        spatial_access_regions_.size());
    spatial_access_regions_.push_back(SpatialAccessRegionRef{
        static_cast<std::string>(region->name), i, subject, access_kind,
        region->index_exprs});
  }
}

const PlanTTKernelABI::SpatialAccessRegionRef*
PlanTTKernelABI::FindSpatialAccessRegionRef(
    const std::string& subject,
    const std::string& access_kind) const {
  const std::string key = subject + "|" + access_kind;
  auto it = spatial_access_region_positions_by_subject_access_.find(key);
  if (it == spatial_access_region_positions_by_subject_access_.end() ||
      it->second.empty()) {
    return nullptr;
  }
  ICHECK_LT(it->second.front(), spatial_access_regions_.size());
  return &spatial_access_regions_[it->second.front()];
}

const PlanTTKernelABI::SpatialAccessRegionRef*
PlanTTKernelABI::FindSpatialAccessRegionRef(
    const std::string& subject,
    const std::string& access_kind,
    const ffi::Array<PrimExpr>& index_exprs) const {
  const std::string key = subject + "|" + access_kind;
  auto it = spatial_access_region_positions_by_subject_access_.find(key);
  if (it == spatial_access_region_positions_by_subject_access_.end() ||
      it->second.empty()) {
    return nullptr;
  }
  if (!index_exprs.empty()) {
    for (size_t position : it->second) {
      ICHECK_LT(position, spatial_access_regions_.size());
      const SpatialAccessRegionRef& region = spatial_access_regions_[position];
      if (SamePrimExprArray(region.index_exprs, index_exprs)) {
        return &region;
      }
    }
    return nullptr;
  }
  ICHECK_LT(it->second.front(), spatial_access_regions_.size());
  return &spatial_access_regions_[it->second.front()];
}

std::optional<PlanTTKernelABI::SpatialLiveValueRef>
PlanTTKernelABI::FindSpatialLiveValueRef(const std::string& subject) const {
  auto it = spatial_live_value_by_subject_.find(subject);
  if (it == spatial_live_value_by_subject_.end()) {
    return std::nullopt;
  }
  return it->second;
}

namespace {

std::string SanitizeExactCBNameComponent(std::string value) {
  for (char& ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '_';
    if (!ok) {
      ch = '_';
    }
  }
  return value.empty() ? std::string("value") : value;
}

}  // namespace

int64_t PlanTTKernelABI::EnsureExactCBLiveFormPlan(
    const std::string& logical_value,
    const ExactTiledCBValue& value) {
  if (logical_value.empty()) {
    return -1;
  }
  auto cached = tt_exact_cb_live_form_index_by_logical_value_.find(logical_value);
  if (cached != tt_exact_cb_live_form_index_by_logical_value_.end()) {
    return cached->second;
  }
  for (int64_t i = 0; i < static_cast<int64_t>(tt_live_form_plans_.size()); ++i) {
    if (static_cast<std::string>(tt_live_form_plans_[i]->logical_value) == logical_value) {
      tt_exact_cb_live_form_index_by_logical_value_[logical_value] = i;
      return i;
    }
  }
  auto spatial_live_value = FindSpatialLiveValueRef(logical_value);
  if (!spatial_live_value) {
    return -1;
  }
  const int64_t live_form_index = static_cast<int64_t>(tt_live_form_plans_.size());
  const std::string name = "live_form_exact_cb_" + SanitizeExactCBNameComponent(logical_value);
  const std::string kernel_name =
      !current_segment_kind_.empty()
          ? current_segment_kind_
          : (requires_compute_segment_ ? std::string("compute") : std::string("main"));
  const int64_t physical_extent =
      value.num_elements > 0 ? value.num_elements : int64_t{32 * 32};
  tt_live_form_plans_.push_back(TTLiveFormPlan(
      String(name), String(logical_value), String(spatial_live_value->name),
      spatial_live_value->index, String(kernel_name),
      String("cb_materialized_tile"), String("thread_distributed"),
      physical_extent, physical_extent, String("materialized_cb_pages_multi_event")));
  tt_exact_cb_live_form_index_by_logical_value_[logical_value] = live_form_index;
  return live_form_index;
}

int64_t PlanTTKernelABI::EnsureExactCBVirtualValue(
    const std::string& logical_value,
    const ExactTiledCBValue& value,
    int current_order_index) {
  if (logical_value.empty() || value.cb_id < 0 ||
      value.cb_id >= static_cast<int>(cb_requirements_.size())) {
    return -1;
  }
  const std::string key =
      logical_value + "|" + std::to_string(value.cb_id) + "|" + value.live_identity;
  auto extend_interval_to_current_use = [&](int64_t virtual_index) {
    if (current_order_index < 0) {
      return;
    }
    const int64_t consumer_point = std::max<int64_t>(0, current_order_index);
    for (size_t i = 0; i < tt_exact_cb_live_intervals_.size(); ++i) {
      const TTExactCBLiveInterval& interval = tt_exact_cb_live_intervals_[i];
      if (interval->virtual_value_index != virtual_index ||
          interval->end_point >= consumer_point) {
        continue;
      }
      tt_exact_cb_live_intervals_.Set(
          i, TTExactCBLiveInterval(
                 interval->name, interval->virtual_value,
                 interval->virtual_value_index, interval->begin_point,
                 consumer_point, interval->live_in, interval->live_out,
                 interval->loop_carried, interval->interference_class));
      return;
    }
  };
  auto cached = tt_exact_cb_virtual_index_by_key_.find(key);
  if (cached != tt_exact_cb_virtual_index_by_key_.end()) {
    extend_interval_to_current_use(cached->second);
    return cached->second;
  }
  const int64_t live_form_index = EnsureExactCBLiveFormPlan(logical_value, value);
  if (live_form_index < 0) {
    return -1;
  }
  const TTLiveFormPlan& live_form =
      tt_live_form_plans_[static_cast<size_t>(live_form_index)];
  const CBRequirement& req = cb_requirements_.at(value.cb_id);
  const int resolved_producer_order = ResolveBorrowedExactInputProducerOrder(value);
  const int producer_order =
      resolved_producer_order >= 0 ? resolved_producer_order : req.lifetime_begin;
  const std::string lifetime_kind =
      spatial_lifetime_kind_by_subject_.count(logical_value)
          ? spatial_lifetime_kind_by_subject_.at(logical_value)
          : std::string("multi_event");
  const std::string loop_role = lifetime_kind == "loop_carried" ? "loop_carried" : "none";
  const int64_t virtual_index = static_cast<int64_t>(tt_exact_cb_virtual_values_.size());
  const std::string name =
      "exact_cb_value_" + SanitizeExactCBNameComponent(logical_value) + "_" +
      std::to_string(value.cb_id) + "_" + std::to_string(virtual_index);
  tt_exact_cb_virtual_values_.push_back(TTExactCBVirtualValue(
      String(name), String(logical_value), live_form->name, live_form_index,
      live_form->producer_kernel, String("program_point_" + std::to_string(
                                    current_order_index >= 0 ? producer_order : 0)),
      String(lifetime_kind), String(loop_role),
      std::max<int64_t>(1, value.num_tiles > 0 ? value.num_tiles : req.num_pages),
      req.page_size, String(req.data_format)));
  const int64_t consumer_point =
      std::max<int64_t>(0, current_order_index >= 0 ? current_order_index
                                                    : req.lifetime_end);
  const int64_t begin_point = std::max<int64_t>(
      0, std::min<int64_t>(producer_order, consumer_point));
  const int64_t end_point = std::max<int64_t>(begin_point, consumer_point);
  tt_exact_cb_live_intervals_.push_back(TTExactCBLiveInterval(
      String("exact_cb_interval_" + SanitizeExactCBNameComponent(logical_value) +
             "_" + std::to_string(value.cb_id) + "_" + std::to_string(virtual_index)),
      String(name), virtual_index, begin_point, end_point,
      lifetime_kind == "loop_carried", lifetime_kind == "loop_carried",
      lifetime_kind == "loop_carried", String("intermediate_exact_cb")));
  tt_exact_cb_virtual_index_by_key_[key] = virtual_index;
  return virtual_index;
}

int64_t PlanTTKernelABI::EnsureExactCBAllocation(
    int64_t virtual_value_index,
    const ExactTiledCBValue& value,
    int release_program_point,
    const std::string& release_reason) {
  if (virtual_value_index < 0 || value.cb_id < 0 ||
      value.cb_id >= static_cast<int>(cb_requirements_.size())) {
    return -1;
  }
  const TTExactCBVirtualValue& virtual_value =
      tt_exact_cb_virtual_values_[static_cast<size_t>(virtual_value_index)];
  const std::string key = std::to_string(virtual_value_index) + "|" +
                          std::to_string(value.cb_id);
  auto cached = tt_exact_cb_allocation_index_by_key_.find(key);
  if (cached != tt_exact_cb_allocation_index_by_key_.end()) {
    return cached->second;
  }
  const CBRequirement& req = cb_requirements_.at(value.cb_id);
  const int64_t allocation_index = static_cast<int64_t>(tt_exact_cb_allocations_.size());
  tt_exact_cb_allocations_.push_back(TTExactCBAllocation(
      String("exact_cb_alloc_" + std::to_string(allocation_index)),
      virtual_value->name, virtual_value_index, String(req.name), value.cb_id,
      value.cb_id, std::max<int64_t>(1, value.num_tiles),
      std::max<int>(release_program_point, 0), String(release_reason)));
  tt_exact_cb_allocation_index_by_key_[key] = allocation_index;
  return allocation_index;
}

Stmt PlanTTKernelABI::MaybeWrapComputeSegment(const Stmt& stmt) const {
  if (current_segment_kind_ == "compute") {
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

const PlanTTKernelABI::SpatialMaterializationBoundaryRef*
PlanTTKernelABI::FindSpatialMaterializationBoundaryRef(
    int64_t materialization_boundary_index) const {
  auto it =
      spatial_materialization_boundary_position_by_index_.find(materialization_boundary_index);
  if (it == spatial_materialization_boundary_position_by_index_.end()) {
    return nullptr;
  }
  ICHECK_LT(it->second, spatial_materialization_boundaries_.size());
  return &spatial_materialization_boundaries_[it->second];
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
  const FutureBufferUses any_future_uses =
      ClassifyFutureBufferUses(src, current_order_index);
  if (!future_uses.has_compute_consume && !future_uses.has_transport_consume &&
      !future_uses.has_reference && !any_future_uses.has_transport_consume) {
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
  ICHECK_GE(fact.spatial_materialization_boundary_index, 0)
      << "PlanTTKernelABI requires materialization fact for " << target_name
      << " to carry SpatialPlan MaterializationBoundary index";
  const SpatialMaterializationBoundaryRef* source_boundary_ref =
      FindSpatialMaterializationBoundaryRef(fact.spatial_materialization_boundary_index);
  ICHECK(source_boundary_ref != nullptr)
      << "PlanTTKernelABI requires SpatialPlan MaterializationBoundary for materialization "
      << fact.spatial_materialization_boundary_index;
  ICHECK_EQ(source_boundary_ref->source_subject, source_name)
      << "PlanTTKernelABI requires materialization fact source_buffer to match "
         "SpatialPlan boundary source subject";
  ICHECK_EQ(source_boundary_ref->target_subject, target_name)
      << "PlanTTKernelABI requires materialization fact target_buffer to match "
         "SpatialPlan boundary target subject";
  SpatialLiveValueRef boundary_source_live_value_ref{source_boundary_ref->source_live_value,
                                                     source_boundary_ref->source_live_value_index};
  SpatialLiveValueRef boundary_target_live_value_ref{source_boundary_ref->target_live_value,
                                                     source_boundary_ref->target_live_value_index};
  std::vector<TTLiveFormBoundaryRequest> live_boundary_graph;
  live_boundary_graph.reserve(spatial_materialization_boundaries_.size());
  for (const SpatialMaterializationBoundaryRef& boundary : spatial_materialization_boundaries_) {
    live_boundary_graph.push_back(TTLiveFormBoundaryRequest{boundary.name,
                                                            boundary.index,
                                                            boundary.source_live_value,
                                                            boundary.source_live_value_index,
                                                            boundary.target_live_value,
                                                            boundary.target_live_value_index,
                                                            boundary.event_lifetime_kind,
                                                            boundary.logical_coverage,
                                                            boundary.min_publish_pages,
                                                            boundary.max_consume_pages});
  }
  const TTLiveFormSolverResult live_form_solution = SolveFragmentCastLiveFormTransition(
      TTLiveFormSolverRequest{source_name,
                              target_name,
                              boundary_source_live_value_ref.name,
                              boundary_source_live_value_ref.index,
                              boundary_target_live_value_ref.name,
                              boundary_target_live_value_ref.index,
                              source_local_extent,
                              target_local_extent,
                              logical_element_count,
                              source_boundary_ref->event_lifetime_kind,
                              source_boundary_ref->logical_coverage,
                              source_boundary_ref->min_publish_pages,
                              source_boundary_ref->max_consume_pages,
                              fact.bridge_kind,
                              fact.materialization_kind,
                              publication_protocol,
                              source_boundary_ref->index,
                              std::move(live_boundary_graph)});

  auto has_live_form = [&](const std::string& name) {
    for (const TTLiveFormPlan& plan : tt_live_form_plans_) {
      if (static_cast<std::string>(plan->name) == name) {
        return true;
      }
    }
    return false;
  };
  auto push_live_form = [&](const TTLiveFormValueDecision& decision) {
    const std::string name = "live_form_" + decision.logical_value;
    if (has_live_form(name)) {
      return;
    }
    tt_live_form_plans_.push_back(TTLiveFormPlan(
        String(name), String(decision.logical_value), String(decision.spatial_live_value),
        decision.spatial_live_value_index, String(kernel_name), String(decision.physical_form),
        String(decision.execution_topology), decision.physical_local_extent,
        decision.logical_element_count, String(decision.ownership_kind)));
  };

  push_live_form(live_form_solution.source_value);
  push_live_form(live_form_solution.target_value);

  const std::string source_live_form = live_form_solution.materialization.source_live_form;
  const std::string produced_live_form = live_form_solution.materialization.produced_live_form;
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
        source_boundary_ref->index, String(live_form_solution.materialization.target_buffer),
        String(), String(kernel_name), String(live_form_solution.materialization.bridge_kind),
        String(live_form_solution.materialization.materialization_kind),
        String(live_form_solution.materialization.materialization_protocol),
        String(live_form_solution.materialization.publication_protocol), required_cb_indices,
        required_sync_indices, String(produced_live_form)));
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
        source_boundary_ref->live_value_edge_index,
        live_form_solution.consumer.accepts_distributed_slice,
        live_form_solution.consumer.requires_full_logical_tile, /*abi_plan_index=*/-1,
        String(target_name),
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
  std::unordered_map<std::string, std::vector<Buffer>> definition_buffers_by_identity;

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

  auto remember_definition = [&](const Buffer& buffer) {
    remember(buffer);
    if (!buffer.defined() || !IsUnsupportedResidualLocalScope(buffer)) {
      return;
    }
    const std::string identity = BufferIdentityName(buffer);
    if (identity.empty()) {
      return;
    }
    auto& group = definition_buffers_by_identity[identity];
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
        remember_definition(buffer);
      }
      return;
    }
    if (const auto* decl = node.as<tir::DeclBufferNode>()) {
      remember_definition(decl->buffer);
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

  std::unordered_set<std::string> ambiguous_definition_identities;
  std::unordered_map<std::string, Buffer> preferred_definition_by_identity;
  for (const auto& [identity, group] : definition_buffers_by_identity) {
    const VarNode* definition_data = nullptr;
    bool ambiguous = false;
    for (const Buffer& buffer : group) {
      const VarNode* data = BufferDataIdentity(buffer);
      if (data == nullptr) {
        ambiguous = true;
        break;
      }
      if (definition_data == nullptr) {
        definition_data = data;
        continue;
      }
      if (definition_data != data) {
        ambiguous = true;
        break;
      }
    }
    if (ambiguous) {
      ambiguous_definition_identities.insert(identity);
      continue;
    }
    Optional<Buffer> preferred = choose_preferred_buffer(group);
    if (preferred) {
      preferred_definition_by_identity[identity] = preferred.value();
    }
  }

  auto find_definition_for_group = [&](const std::vector<Buffer>& group) -> Optional<Buffer> {
    Optional<Buffer> selected;
    for (const Buffer& buffer : group) {
      const std::string identity = BufferIdentityName(buffer);
      if (identity.empty() || ambiguous_definition_identities.count(identity)) {
        continue;
      }
      auto definition_it = preferred_definition_by_identity.find(identity);
      if (definition_it == preferred_definition_by_identity.end()) {
        continue;
      }
      if (!selected) {
        selected = definition_it->second;
        continue;
      }
      if (!selected.value().same_as(definition_it->second)) {
        return Optional<Buffer>();
      }
    }
    return selected;
  };

  for (const auto& [data, group] : buffers_by_data) {
    Optional<Buffer> preferred = find_definition_for_group(group);
    if (!preferred) {
      preferred = choose_preferred_buffer(group);
    }
    if (!preferred) {
      continue;
    }
    compute_physical_buffers_by_data_[data] = preferred.value();
    for (const Buffer& buffer : group) {
      const std::string identity = BufferIdentityName(buffer);
      if (!identity.empty()) {
        auto definition_it = preferred_definition_by_identity.find(identity);
        if (definition_it != preferred_definition_by_identity.end() &&
            !ambiguous_definition_identities.count(identity)) {
          compute_physical_buffers_by_identity_[identity] = definition_it->second;
        } else {
          compute_physical_buffers_by_identity_[identity] = preferred.value();
        }
      }
    }
  }
  for (const auto& [identity, group] : buffers_by_identity) {
    if (compute_physical_buffers_by_identity_.count(identity)) {
      continue;
    }
    Optional<Buffer> preferred;
    auto definition_it = preferred_definition_by_identity.find(identity);
    if (definition_it != preferred_definition_by_identity.end() &&
        !ambiguous_definition_identities.count(identity)) {
      preferred = definition_it->second;
    } else {
      preferred = choose_preferred_buffer(group);
    }
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
  if (order_index >= 0) {
    buffer_live_form_order_by_cb_id_[cb_id] = order_index;
  }
  auto clear_exact_buffer = [&](const std::string& identity) {
    if (identity.empty()) {
      return false;
    }
    auto invalidated_it = invalidated_live_form_order_by_buffer_identity_.find(identity);
    if (invalidated_it != invalidated_live_form_order_by_buffer_identity_.end()) {
      if (order_index >= 0 && order_index < invalidated_it->second) {
        return false;
      }
      invalidated_live_form_order_by_buffer_identity_.erase(invalidated_it);
    }
    auto exact_order_it = exact_output_live_form_order_by_buffer_identity_.find(identity);
    if (order_index >= 0 &&
        exact_order_it != exact_output_live_form_order_by_buffer_identity_.end() &&
        exact_order_it->second > order_index) {
      return false;
    }
    exact_output_live_form_cb_by_buffer_identity_.erase(identity);
    exact_output_live_form_value_by_buffer_identity_.erase(identity);
    if (order_index >= 0) {
      exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
    }
    return true;
  };
  auto is_loop_carried_state_candidate = [&](const std::string& identity,
                                             const Buffer& candidate) -> bool {
    if (identity.empty()) {
      return false;
    }
    const Buffer physical = ResolvePhysicalComputeBuffer(candidate);
    const Buffer state_buffer = physical.defined() ? physical : candidate;
    if (!state_buffer.defined() || GetStorageScope(state_buffer) != "blackhole.acc" ||
        !IsSingleFullTileLogicalMatrix(state_buffer)) {
      return false;
    }
    if (!IsActiveLoopCarriedBuffer(candidate) && !IsCompletedLoopCarriedBuffer(candidate) &&
        !IsActiveLoopCarriedBuffer(state_buffer) && !IsCompletedLoopCarriedBuffer(state_buffer) &&
        !HasLoopCarriedExactCBState(identity)) {
      return false;
    }
    return true;
  };
  auto record_loop_carried_state = [&](const std::string& identity,
                                       const Buffer& candidate) {
    if (!is_loop_carried_state_candidate(identity, candidate)) {
      return;
    }
    ExactTiledCBValue state_value;
    state_value.buffer = candidate;
    state_value.cb_id = cb_id;
    state_value.borrowed_live = true;
    state_value.live_identity = identity;
    PopulateExactTiledCBValueShape(candidate, &state_value);
    RefineExactTiledCBValueShapeFromRequirement(&state_value);
    RememberLoopCarriedExactCBState(identity, state_value, order_index);
  };
  auto record_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (!identity.empty()) {
      const bool loop_carried_state = is_loop_carried_state_candidate(identity, candidate);
      auto order_it = buffer_live_form_order_by_buffer_identity_.find(identity);
      if (order_index >= 0 && order_it != buffer_live_form_order_by_buffer_identity_.end() &&
          order_it->second > order_index) {
        if (loop_carried_state) {
          record_loop_carried_state(identity, candidate);
        }
        return;
      }
      if (!clear_exact_buffer(identity)) {
        if (loop_carried_state) {
          record_loop_carried_state(identity, candidate);
        }
        return;
      }
      buffer_live_form_cb_by_buffer_identity_[identity] = cb_id;
      buffer_live_form_order_by_buffer_identity_[identity] = order_index;
      invalidated_live_form_order_by_buffer_identity_.erase(identity);
      local_only_live_form_buffer_identities_.erase(identity);
      record_loop_carried_state(identity, candidate);
    }
  };
  record_buffer(buffer);
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined()) {
    record_buffer(physical);
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        const bool loop_carried_state = is_loop_carried_state_candidate(identity, physical);
        if (!clear_exact_buffer(identity)) {
          if (loop_carried_state) {
            record_loop_carried_state(identity, physical);
          }
          continue;
        }
        buffer_live_form_cb_by_buffer_identity_[identity] = cb_id;
        if (order_index >= 0) {
          buffer_live_form_order_by_buffer_identity_[identity] = order_index;
        }
        local_only_live_form_buffer_identities_.erase(identity);
        record_loop_carried_state(identity, physical);
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
  exact_output_live_form_value_by_buffer_identity_.erase(identity);
  if (order_index >= 0) {
    exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
    invalidated_live_form_order_by_buffer_identity_[identity] = order_index;
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

void PlanTTKernelABI::OverwriteTiledCBLiveFormAliasesForWrite(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  const int order_index = current_lowering_order_index_;
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    if (identity.empty()) {
      continue;
    }
    buffer_live_form_cb_by_buffer_identity_.erase(identity);
    exact_output_live_form_cb_by_buffer_identity_.erase(identity);
    exact_output_live_form_value_by_buffer_identity_.erase(identity);
    local_only_live_form_buffer_identities_.erase(identity);
    if (order_index >= 0) {
      buffer_live_form_order_by_buffer_identity_[identity] = order_index;
      exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
      invalidated_live_form_order_by_buffer_identity_[identity] = order_index;
    }
  }
}

void PlanTTKernelABI::MarkLocalOnlyLiveFormAliases(const Buffer& buffer) {
  if (!buffer.defined()) {
    return;
  }
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    if (!identity.empty()) {
      local_only_live_form_buffer_identities_.insert(identity);
    }
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

void PlanTTKernelABI::InvalidateLastFragmentFillValueIdentity(
    const std::string& identity) {
  if (identity.empty()) {
    return;
  }
  last_fragment_fill_value_by_buffer_identity_.erase(identity);
  auto erase_buffer_data = [&](const Buffer& buffer) {
    if (!buffer.defined()) {
      return;
    }
    if (const VarNode* data = BufferDataIdentity(buffer)) {
      last_fragment_fill_value_by_data_.erase(data);
    }
  };
  auto buffer_it = buffer_by_identity_.find(identity);
  if (buffer_it != buffer_by_identity_.end()) {
    erase_buffer_data(buffer_it->second);
  }
  auto physical_it = compute_physical_buffers_by_identity_.find(identity);
  if (physical_it != compute_physical_buffers_by_identity_.end()) {
    erase_buffer_data(physical_it->second);
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
      selected_source_live_producer_order_by_buffer_identity_.erase(identity);
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
        selected_source_live_producer_order_by_buffer_identity_.erase(identity);
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
      selected_source_live_producer_order_by_buffer_identity_[identity] =
          current_lowering_order_index_;
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
        selected_source_live_producer_order_by_buffer_identity_[identity] =
            current_lowering_order_index_;
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

void PlanTTKernelABI::LoadDirectCopySourceBindings(const PrimFunc& func) {
  direct_copy_source_by_buffer_identity_.clear();
  buffer_by_identity_.clear();
  std::unordered_set<std::string> ambiguous_targets;
  auto remember_buffer = [&](const Buffer& buffer) {
    const std::string identity = BufferIdentityName(buffer);
    if (!identity.empty() && !buffer_by_identity_.count(identity)) {
      buffer_by_identity_.emplace(identity, buffer);
    }
  };
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    const auto* store = node.as<tir::BufferStoreNode>();
    if (store == nullptr || !IsCopyOperation(store)) {
      return;
    }
    const auto* load = store->value.as<tir::BufferLoadNode>();
    if (load == nullptr) {
      return;
    }
    remember_buffer(store->buffer);
    remember_buffer(load->buffer);
    const std::string dst = BufferIdentityName(store->buffer);
    const std::string src = BufferIdentityName(load->buffer);
    if (dst.empty() || src.empty() || dst == src) {
      return;
    }
    const std::string dst_scope = GetStorageScope(store->buffer);
    if ((dst_scope.empty() || dst_scope == "global") &&
        IsUnsupportedResidualLocalScope(load->buffer)) {
      host_buffer_by_compute_operand_buffer_[src] = dst;
    }
    auto it = direct_copy_source_by_buffer_identity_.find(dst);
    if (it == direct_copy_source_by_buffer_identity_.end()) {
      direct_copy_source_by_buffer_identity_.emplace(dst, src);
      return;
    }
    if (it->second != src) {
      ambiguous_targets.insert(dst);
    }
  });
  for (const std::string& target : ambiguous_targets) {
    direct_copy_source_by_buffer_identity_.erase(target);
  }
}

void PlanTTKernelABI::RefreshBroadcastColsSourceBuffers() {
  broadcast_cols_source_buffers_.clear();
  for (const std::string& rhs : broadcast_cols_rhs_buffers_) {
    std::string current = rhs;
    std::unordered_set<std::string> seen;
    while (!current.empty() && seen.insert(current).second) {
      broadcast_cols_source_buffers_.insert(current);
      auto it = direct_copy_source_by_buffer_identity_.find(current);
      if (it == direct_copy_source_by_buffer_identity_.end()) {
        break;
      }
      current = it->second;
    }
  }
}

bool PlanTTKernelABI::IsBroadcastColsSourceBuffer(const Buffer& buffer) const {
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    if (broadcast_cols_source_buffers_.count(identity) != 0U) {
      return true;
    }
  }
  return false;
}

bool PlanTTKernelABI::IsBroadcastColsSourceCBId(int cb_id) const {
  auto is_source_requirement = [&](int requirement_index) {
    if (requirement_index < 0 || requirement_index >= static_cast<int>(cb_requirements_.size())) {
      return false;
    }
    const CBRequirement& req = cb_requirements_.at(requirement_index);
    return broadcast_cols_source_buffers_.count(req.name) != 0U;
  };
  if (is_source_requirement(cb_id)) {
    return true;
  }
  constexpr int kTTMetalUserCBBase = 16;
  if (cb_id >= kTTMetalUserCBBase && is_source_requirement(cb_id - kTTMetalUserCBBase)) {
    return true;
  }
  if (cb_id < 0 || cb_id >= static_cast<int>(cb_requirements_.size())) {
    return false;
  }
  return false;
}

bool PlanTTKernelABI::TryCreateBroadcastColsSourceLiveExactTiledCBValue(
    const Buffer& buffer, ExactTiledCBValue* value) {
  ICHECK(value != nullptr);
  auto populate_bcast_source_live_value = [&](int cb_id, bool borrowed_live,
                                              const std::string& live_identity) {
    ICHECK_GE(cb_id, 0);
    ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
    const CBRequirement& req = cb_requirements_.at(cb_id);
    const int event_pages =
        std::max({1, req.publish_pages_per_event, req.consume_pages_per_event});
    value->buffer = buffer;
    value->cb_id = cb_id;
    value->producer_live = true;
    value->borrowed_live = borrowed_live;
    value->live_identity =
        !live_identity.empty() ? live_identity : BufferIdentityName(buffer);
    PopulateExactTiledCBValueShape(buffer, value);
    value->num_tiles = event_pages;
    value->num_elements = static_cast<int64_t>(event_pages) *
                          kBlackholeTileRows * kBlackholeTileCols;
    value->row_width = kBlackholeTileCols;
  };
  std::vector<std::string> candidates = CollectBufferFlowIdentities(buffer);
  std::unordered_set<std::string> seen(candidates.begin(), candidates.end());
  for (size_t index = 0; index < candidates.size(); ++index) {
    auto it = direct_copy_source_by_buffer_identity_.find(candidates[index]);
    if (it == direct_copy_source_by_buffer_identity_.end()) {
      continue;
    }
    if (seen.insert(it->second).second) {
      candidates.push_back(it->second);
    }
  }

  for (const std::string& identity : broadcast_cols_source_buffers_) {
    if (seen.insert(identity).second) {
      candidates.push_back(identity);
    }
  }
  std::vector<std::string> producer_candidates;
  std::unordered_set<std::string> producer_seen;
  auto add_producer_candidate = [&](const std::string& identity) {
    if (!identity.empty() && broadcast_cols_source_buffers_.count(identity) != 0U &&
        producer_seen.insert(identity).second) {
      producer_candidates.push_back(identity);
    }
  };
  for (const std::string& root : CollectBufferFlowIdentities(buffer)) {
    std::string current = root;
    std::unordered_set<std::string> chain_seen;
    bool saw_copy_source = false;
    while (!current.empty() && chain_seen.insert(current).second) {
      auto copy_it = direct_copy_source_by_buffer_identity_.find(current);
      if (copy_it == direct_copy_source_by_buffer_identity_.end()) {
        break;
      }
      saw_copy_source = true;
      current = copy_it->second;
      add_producer_candidate(current);
    }
    if (!saw_copy_source) {
      add_producer_candidate(root);
    }
  }
  for (const std::string& identity : candidates) {
    add_producer_candidate(identity);
  }

  for (const std::string& identity : candidates) {
    auto live_it = buffer_live_form_cb_by_buffer_identity_.find(identity);
    if (live_it == buffer_live_form_cb_by_buffer_identity_.end()) {
      continue;
    }
    const int cb_id = live_it->second;
    if (cb_id < 0 || cb_id >= static_cast<int>(cb_requirements_.size())) {
      continue;
    }
    const CBRequirement& req = cb_requirements_.at(cb_id);
    if (req.page_size <
        kBlackholeTileRows * kBlackholeTileCols * ExactTiledCBStorageDType(buffer->dtype).bytes()) {
      continue;
    }
    int live_order_index = -1;
    auto order_it = buffer_live_form_order_by_buffer_identity_.find(identity);
    if (order_it != buffer_live_form_order_by_buffer_identity_.end()) {
      live_order_index = order_it->second;
    }
    bool has_intervening_identity_write = false;
    if (current_lowering_order_index_ >= 0 && live_order_index >= 0) {
      auto flow_it = buffer_flow_facts_.find(identity);
      if (flow_it != buffer_flow_facts_.end()) {
        for (const BlackholeBufferFlowEvent& event : flow_it->second.events) {
          if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
              event.order_index > live_order_index &&
              event.order_index < current_lowering_order_index_) {
            has_intervening_identity_write = true;
            break;
          }
        }
      }
    }
    if (has_intervening_identity_write) {
      continue;
    }
    populate_bcast_source_live_value(cb_id, /*borrowed_live=*/true, identity);
    return true;
  }
  for (const std::string& identity : producer_candidates) {
    auto req_it = buffer_identity_to_req_index_.find(identity);
    if (req_it == buffer_identity_to_req_index_.end()) {
      continue;
    }
    const int cb_id = req_it->second;
    if (cb_id < 0 || cb_id >= static_cast<int>(cb_requirements_.size())) {
      continue;
    }
    const CBRequirement& req = cb_requirements_.at(cb_id);
    if (req.page_size <
        kBlackholeTileRows * kBlackholeTileCols * ExactTiledCBStorageDType(buffer->dtype).bytes()) {
      continue;
    }
    populate_bcast_source_live_value(cb_id, /*borrowed_live=*/false, identity);
    return true;
  }
  for (int cb_id = 0; cb_id < static_cast<int>(cb_requirements_.size()); ++cb_id) {
    const CBRequirement& req = cb_requirements_.at(cb_id);
    if (std::find(producer_candidates.begin(), producer_candidates.end(), req.name) ==
        producer_candidates.end()) {
      continue;
    }
    if (req.page_size <
        kBlackholeTileRows * kBlackholeTileCols * ExactTiledCBStorageDType(buffer->dtype).bytes()) {
      continue;
    }
    populate_bcast_source_live_value(cb_id, /*borrowed_live=*/false, req.name);
    return true;
  }
  for (const std::string& identity : producer_candidates) {
    auto buffer_it = buffer_by_identity_.find(identity);
    if (buffer_it == buffer_by_identity_.end() || !buffer_it->second.defined()) {
      continue;
    }
    const int cb_id = AllocateRequirementIndex(buffer_it->second, CBType::kIntermediate);
    const int tile_bytes =
        kBlackholeTileRows * kBlackholeTileCols * ExactTiledCBStorageDType(buffer->dtype).bytes();
    SetRequirementPageLayout(cb_id, tile_bytes, 1);
    auto& req = cb_requirements_.at(cb_id);
    req.publish_pages_per_event = std::max(req.publish_pages_per_event, 1);
    req.consume_pages_per_event = std::max(req.consume_pages_per_event, 1);
    populate_bcast_source_live_value(cb_id, /*borrowed_live=*/false, identity);
    return true;
  }
  return false;
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
    for (const auto& [identity, logical_candidate] : buffer_by_identity_) {
      if (!identity.empty() && logical_candidate.defined() &&
          SameBufferIdentity(logical_candidate, physical)) {
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

bool PlanTTKernelABI::FutureWritePrecedesFutureComputeConsume(
    const Buffer& buffer, int current_order_index) const {
  if (current_order_index < 0) {
    return false;
  }
  const std::vector<std::string> identity_list = CollectBufferFlowIdentities(buffer);
  if (identity_list.empty() || execution_ordered_stmts_.empty()) {
    return false;
  }
  const std::unordered_set<std::string> identities(identity_list.begin(), identity_list.end());
  auto writes_buffer = [&](const Stmt& stmt) {
    bool writes = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (writes) {
        return;
      }
      if (const auto* store = node.as<BufferStoreNode>()) {
        writes = identities.count(BufferIdentityName(store->buffer)) != 0U;
        return;
      }
      const auto* call = node.as<CallNode>();
      if (!call || !call->op->IsInstance<OpNode>()) {
        return;
      }
      TileOperator tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
        if (access.kind == DataflowAccessKind::kComputeProduce &&
            identities.count(BufferIdentityName(access.buffer)) != 0U) {
          writes = true;
          return;
        }
      }
    });
    return writes;
  };
  auto compute_consumes_buffer = [&](const Stmt& stmt) {
    bool consumes = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (consumes) {
        return;
      }
      const auto* call = node.as<CallNode>();
      if (!call || !call->op->IsInstance<OpNode>()) {
        return;
      }
      TileOperator tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
        if (access.kind == DataflowAccessKind::kComputeConsume &&
            identities.count(BufferIdentityName(access.buffer)) != 0U) {
          consumes = true;
          return;
        }
      }
    });
    return consumes;
  };

  for (const Stmt& stmt : execution_ordered_stmts_) {
    auto order_it = stmt_order_index_by_node_.find(stmt.get());
    const int order_index =
        order_it != stmt_order_index_by_node_.end() ? order_it->second : -1;
    if (order_index <= current_order_index) {
      continue;
    }
    if (compute_consumes_buffer(stmt)) {
      return false;
    }
    if (writes_buffer(stmt)) {
      return true;
    }
  }
  return false;
}

bool PlanTTKernelABI::FutureWritePrecedesFutureTransportConsume(
    const Buffer& buffer, int current_order_index) const {
  if (current_order_index < 0) {
    return false;
  }
  const std::vector<std::string> identity_list = CollectBufferFlowIdentities(buffer);
  if (identity_list.empty() || execution_ordered_stmts_.empty()) {
    return false;
  }
  const std::unordered_set<std::string> identities(identity_list.begin(), identity_list.end());
  auto writes_buffer = [&](const Stmt& stmt) {
    bool writes = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (writes) {
        return;
      }
      if (const auto* store = node.as<BufferStoreNode>()) {
        writes = identities.count(BufferIdentityName(store->buffer)) != 0U;
        return;
      }
      const auto* call = node.as<CallNode>();
      if (!call || !call->op->IsInstance<OpNode>()) {
        return;
      }
      TileOperator tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
        if (access.kind == DataflowAccessKind::kComputeProduce &&
            identities.count(BufferIdentityName(access.buffer)) != 0U) {
          writes = true;
          return;
        }
      }
    });
    return writes;
  };
  auto transport_consumes_buffer = [&](const Stmt& stmt) {
    bool consumes = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (consumes) {
        return;
      }
      const auto* store = node.as<BufferStoreNode>();
      if (!IsCopyOperation(store)) {
        return;
      }
      const auto* load = store->value.as<BufferLoadNode>();
      if (load != nullptr && identities.count(BufferIdentityName(load->buffer)) != 0U &&
          IsUnsupportedResidualLocalScope(load->buffer)) {
        consumes = true;
      }
    });
    return consumes;
  };

  for (const Stmt& stmt : execution_ordered_stmts_) {
    auto order_it = stmt_order_index_by_node_.find(stmt.get());
    const int order_index =
        order_it != stmt_order_index_by_node_.end() ? order_it->second : -1;
    if (order_index <= current_order_index) {
      continue;
    }
    if (transport_consumes_buffer(stmt)) {
      return false;
    }
    if (writes_buffer(stmt)) {
      return true;
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
  return ClassifyFutureLiveCBReadsBeforeNextWriteUntilOrder(
      buffer, current_order_index, /*upper_bound_order_index=*/-1);
}

PlanTTKernelABI::FutureBufferUses
PlanTTKernelABI::ClassifyFutureLiveCBReadsBeforeNextWriteUntilOrder(
    const Buffer& buffer, int current_order_index, int upper_bound_order_index) const {
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
      if (upper_bound_order_index >= 0 && event.order_index > upper_bound_order_index) {
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
      if (upper_bound_order_index >= 0 && event.order_index > upper_bound_order_index) {
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
PlanTTKernelABI::ClassifyFutureBufferIdentityReadsBeforeNextWrite(
    const std::string& buffer_identity, int current_order_index) const {
  return ClassifyFutureBufferIdentityReadsBeforeNextWriteUntilOrder(
      buffer_identity, current_order_index, /*upper_bound_order_index=*/-1);
}

PlanTTKernelABI::FutureBufferUses
PlanTTKernelABI::ClassifyFutureBufferIdentityReadsBeforeNextWriteUntilOrder(
    const std::string& buffer_identity, int current_order_index,
    int upper_bound_order_index) const {
  FutureBufferUses uses;
  if (buffer_identity.empty()) {
    return uses;
  }
  auto it = buffer_flow_facts_.find(buffer_identity);
  if (it == buffer_flow_facts_.end()) {
    return uses;
  }
  int next_write_order_index = -1;
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    if (upper_bound_order_index >= 0 && event.order_index > upper_bound_order_index) {
      continue;
    }
    if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
        (next_write_order_index < 0 || event.order_index < next_write_order_index)) {
      next_write_order_index = event.order_index;
    }
  }
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    if (upper_bound_order_index >= 0 && event.order_index > upper_bound_order_index) {
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
  return uses;
}

bool PlanTTKernelABI::BufferIdentityHasWriteAtOrder(
    const std::string& buffer_identity, int order_index) const {
  if (buffer_identity.empty() || order_index < 0) {
    return false;
  }
  auto it = buffer_flow_facts_.find(buffer_identity);
  if (it == buffer_flow_facts_.end()) {
    return false;
  }
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index == order_index &&
        event.kind == BlackholeBufferFlowEventKind::kWrite) {
      return true;
    }
  }
  return false;
}

bool PlanTTKernelABI::BufferIdentityHasComputeConsumeAtOrder(
    const std::string& buffer_identity, int order_index) const {
  if (buffer_identity.empty() || order_index < 0) {
    return false;
  }
  auto it = buffer_flow_facts_.find(buffer_identity);
  if (it == buffer_flow_facts_.end()) {
    return false;
  }
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index == order_index &&
        event.kind == BlackholeBufferFlowEventKind::kComputeConsume) {
      return true;
    }
  }
  return false;
}

bool PlanTTKernelABI::HasFutureExactLiveFormTileComputeConsume(
    const Buffer& buffer, int current_order_index) const {
  bool has_seeded_tile_compute_input = false;
  bool has_future_exact_live_form = false;
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    if (tile_compute_input_buffers_.count(identity) != 0U) {
      has_seeded_tile_compute_input = true;
    }
    auto order_it = exact_output_live_form_order_by_buffer_identity_.find(identity);
    if (order_it != exact_output_live_form_order_by_buffer_identity_.end() &&
        (current_order_index < 0 || order_it->second > current_order_index)) {
      has_future_exact_live_form = true;
    }
  }
  return has_seeded_tile_compute_input && has_future_exact_live_form;
}

int PlanTTKernelABI::FindRequirementIndexForBuffer(const Buffer& buffer) const {
  if (!buffer.defined()) {
    return -1;
  }
  auto it = buffer_to_req_.find(buffer);
  if (it != buffer_to_req_.end()) {
    return it->second;
  }
  const std::string buffer_identity = BufferIdentityName(buffer);
  if (!buffer_identity.empty()) {
    auto by_identity = buffer_identity_to_req_index_.find(buffer_identity);
    if (by_identity != buffer_identity_to_req_index_.end()) {
      return by_identity->second;
    }
  }
  auto by_data = buffer_data_to_req_index_.find(buffer->data.get());
  if (by_data != buffer_data_to_req_index_.end()) {
    return by_data->second;
  }
  return -1;
}

bool PlanTTKernelABI::ShouldRetainComputeInputBufferAcrossSerialLoop(
    const Buffer& buffer, int consumed_pages) const {
  if (!buffer.defined() || consumed_pages <= 0 || !HasRepeatingActiveSerialLoop()) {
    return false;
  }
  const int requirement_index = FindRequirementIndexForBuffer(buffer);
  if (requirement_index < 0 ||
      requirement_index >= static_cast<int>(cb_requirements_.size())) {
    return false;
  }
  const CBRequirement& req = cb_requirements_.at(requirement_index);
  if (req.type != CBType::kInput || GetStorageScope(buffer) == "blackhole.acc") {
    return false;
  }
  return true;
}

bool PlanTTKernelABI::HasRepeatingActiveSerialLoop() const {
  if (active_serial_loop_vars_.empty()) {
    return false;
  }
  for (const Var& loop_var : active_serial_loop_vars_) {
    auto thread_extent_it = thread_index_var_static_extents_.find(loop_var.get());
    if (thread_extent_it != thread_index_var_static_extents_.end()) {
      if (thread_extent_it->second > 1) {
        return true;
      }
      continue;
    }
    auto loop_extent_it = loop_var_static_extents_.find(loop_var.get());
    if (loop_extent_it != loop_var_static_extents_.end()) {
      if (loop_extent_it->second > 1) {
        return true;
      }
      continue;
    }
    return true;
  }
  return false;
}

PrimExpr PlanTTKernelABI::BuildActiveSerialLoopFinalIterationPredicate() const {
  PrimExpr condition;
  for (const Var& loop_var : active_serial_loop_vars_) {
    int64_t extent = -1;
    auto thread_extent_it = thread_index_var_static_extents_.find(loop_var.get());
    if (thread_extent_it != thread_index_var_static_extents_.end()) {
      extent = thread_extent_it->second;
    } else {
      auto loop_extent_it = loop_var_static_extents_.find(loop_var.get());
      if (loop_extent_it != loop_var_static_extents_.end()) {
        extent = loop_extent_it->second;
      }
    }
    if (extent <= 0) {
      return PrimExpr();
    }
    PrimExpr is_final =
        tir::EQ(loop_var, IntImm(loop_var.dtype(), static_cast<int64_t>(extent - 1)));
    condition = condition.defined() ? (condition && is_final) : is_final;
  }
  return condition;
}

bool PlanTTKernelABI::ShouldDeferTerminalTransportPublicationAcrossSerialLoop(
    const Buffer& buffer, int current_order_index) const {
  if (!buffer.defined() || !HasRepeatingActiveSerialLoop() ||
      serial_loop_terminal_transport_publications_stack_.empty()) {
    return false;
  }
  if (!BuildActiveSerialLoopFinalIterationPredicate().defined()) {
    return false;
  }
  const FutureBufferUses future_uses =
      ClassifyFutureBufferUses(buffer, current_order_index);
  if (future_uses.has_compute_consume) {
    return false;
  }
  if (future_uses.has_transport_consume) {
    return true;
  }
  const std::string scope = GetStorageScope(buffer);
  return scope.rfind("shared", 0) == 0 || scope.rfind("blackhole.cb", 0) == 0;
}

int PlanTTKernelABI::ReserveSerialLoopRetainedComputeInputOffset(
    const Buffer& buffer,
    const std::string& region_key,
    int pages) {
  if (serial_loop_retained_input_pop_pages_stack_.empty() ||
      serial_loop_retained_input_offsets_stack_.empty() || pages <= 0 ||
      !ShouldRetainComputeInputBufferAcrossSerialLoop(buffer, pages)) {
    return 0;
  }
  const int requirement_index = FindRequirementIndexForBuffer(buffer);
  ICHECK_GE(requirement_index, 0);
  ICHECK_LT(requirement_index, static_cast<int>(cb_requirements_.size()));
  ICHECK_EQ(serial_loop_retained_input_offsets_stack_.size(),
            serial_loop_retained_input_pop_pages_stack_.size());
  auto& retained_offsets =
      serial_loop_retained_input_offsets_stack_.back()[requirement_index];
  const std::string key =
      region_key.empty() ? BufferIdentityName(buffer) : region_key;
  if (!key.empty()) {
    auto existing = retained_offsets.find(key);
    if (existing != retained_offsets.end()) {
      return existing->second;
    }
  }
  auto& retained_pages =
      serial_loop_retained_input_pop_pages_stack_.back()[requirement_index];
  const int tile_offset = retained_pages;
  retained_pages += pages;
  if (!key.empty()) {
    retained_offsets[key] = tile_offset;
  }
  return tile_offset;
}

Stmt PlanTTKernelABI::BuildSerialLoopRetainedInputPops(
    const std::map<int, int>& pop_pages_by_requirement_index) const {
  if (pop_pages_by_requirement_index.empty()) {
    return Stmt();
  }
  std::vector<Stmt> pops;
  pops.reserve(pop_pages_by_requirement_index.size());
  for (const auto& [requirement_index, pages] : pop_pages_by_requirement_index) {
    ICHECK_GE(requirement_index, 0);
    ICHECK_LT(requirement_index, static_cast<int>(cb_requirements_.size()));
    ICHECK_GT(pages, 0);
    pops.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                     {IntImm(DataType::Int(32), requirement_index),
                                      IntImm(DataType::Int(32), pages)}));
  }
  return MaybeWrapComputeSegment(SeqStmt::Flatten(pops));
}

void PlanTTKernelABI::RecordSerialLoopTerminalTransportPublication(const Stmt& stmt) {
  if (!stmt.defined() || serial_loop_terminal_transport_publications_stack_.empty()) {
    return;
  }
  serial_loop_terminal_transport_publications_stack_.back().push_back(stmt);
}

Stmt PlanTTKernelABI::BuildSerialLoopTerminalTransportPublications(
    const std::vector<Stmt>& publications) const {
  if (publications.empty()) {
    return Stmt();
  }
  return SeqStmt::Flatten(publications);
}

Stmt PlanTTKernelABI::AppendSerialLoopLocalComputeOutputPops(
    const Stmt& body) const {
  if (!body.defined()) {
    return body;
  }
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(const std::vector<CBRequirement>& requirements)
        : requirements_(requirements),
          pushed_pages_(requirements.size(), 0),
          popped_pages_(requirements.size(), 0),
          locally_available_pages_(requirements.size(), 0),
          waits_after_local_publish_(requirements.size(), 0) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const Stmt& stmt) { VisitStmt(stmt); }

    std::vector<int> Take() const {
      std::vector<int> pop_pages(requirements_.size(), 0);
      for (size_t i = 0; i < requirements_.size(); ++i) {
        const CBRequirement& req = requirements_[i];
        if (req.type == CBType::kInput || req.flow_class == CBFlowClass::kState) {
          continue;
        }
        const int net_local_front = pushed_pages_[i] - popped_pages_[i];
        if (net_local_front <= 0 || waits_after_local_publish_[i] <= 0) {
          continue;
        }
        pop_pages[i] = net_local_front;
      }
      return pop_pages;
    }

    void VisitExpr_(const tir::CallNode* op) final {
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (cb_id >= 0 && pages > 0) {
        if (IsBlackholeOpName(op, "tl.blackhole.cb_push_back")) {
          pushed_pages_[cb_id] += pages;
          locally_available_pages_[cb_id] += pages;
        } else if (IsBlackholeOpName(op, "tl.blackhole.cb_pop_front")) {
          popped_pages_[cb_id] += pages;
          locally_available_pages_[cb_id] =
              std::max(0, locally_available_pages_[cb_id] - pages);
        } else if (IsBlackholeOpName(op, "tl.blackhole.cb_wait_front") &&
                   locally_available_pages_[cb_id] > 0) {
          waits_after_local_publish_[cb_id] =
              std::max(waits_after_local_publish_[cb_id], pages);
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

   private:
    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<tir::IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(requirements_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<tir::IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    const std::vector<CBRequirement>& requirements_;
    std::vector<int> pushed_pages_;
    std::vector<int> popped_pages_;
    std::vector<int> locally_available_pages_;
    std::vector<int> waits_after_local_publish_;
  };

  Collector collector(cb_requirements_);
  collector.Collect(body);
  const std::vector<int> pop_pages = collector.Take();
  std::vector<Stmt> stmts;
  stmts.push_back(body);
  for (size_t i = 0; i < pop_pages.size(); ++i) {
    if (pop_pages[i] > 0) {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                        {IntImm32(static_cast<int>(i)),
                                         IntImm32(pop_pages[i])}));
    }
  }
  if (stmts.size() == 1U) {
    return body;
  }
  return SeqStmt::Flatten(stmts);
}

bool PlanTTKernelABI::ShouldRetainComputeInputBuffer(const Buffer& buffer,
                                                       int current_order_index,
                                                       int consumed_pages) const {
  if (ShouldRetainComputeInputBufferAcrossSerialLoop(buffer, consumed_pages)) {
    return true;
  }
  if (FindBufferMaterializationFact(buffer) != nullptr) {
    return false;
  }
  const FutureBufferUses uses = ClassifyFutureBufferUses(buffer, current_order_index);
  if (!uses.has_compute_consume) {
    return false;
  }
  if (FutureWritePrecedesFutureComputeConsume(buffer, current_order_index)) {
    return false;
  }
  if (consumed_pages <= 0) {
    return true;
  }
  const int64_t total_elements = GetLogicalBufferElementCount(buffer);
  const int page_size = EstimateCopyPageSize(buffer);
  if (total_elements <= 0 || page_size <= 0) {
    return true;
  }
  const int64_t dtype_bytes = std::max<int64_t>(1, buffer->dtype.bytes());
  const int64_t total_bytes = total_elements * dtype_bytes;
  const int64_t total_pages = (total_bytes + page_size - 1) / page_size;
  return total_pages <= consumed_pages;
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
