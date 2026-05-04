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
 * \file lower_blackhole_exact_cb.cc
 * \brief Exact tiled-CB live-form and local materialization helpers.
 */

#include "lower_blackhole_ops.h"

#include "common/blackhole_utils.h"

#include <tvm/tir/op.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::Buffer;
using tir::Call;
using tir::AttrStmt;
using tir::Evaluate;
using tir::SeqStmt;
using tir::Stmt;
using tir::VarNode;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_fill_fragment;
using tir::builtin::blackhole_pack_fill_fragment_to_tiled_cb;
using tir::builtin::blackhole_tilize_cast_fragment_slice;
using tir::builtin::blackhole_tilize_local_fragment_slice;
using tir::builtin::blackhole_untilize_cb_front_tile_fragment;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;
constexpr const char* kBlackholeExactOutputLiveCBAttr = "blackhole.exact_output_live_cb";
constexpr const char* kBlackholeExactOutputLiveNumTilesAttr =
    "blackhole.exact_output_live_num_tiles";
constexpr const char* kBlackholeExactOutputLiveNumElementsAttr =
    "blackhole.exact_output_live_num_elements";
constexpr const char* kBlackholeExactOutputLiveRowWidthAttr =
    "blackhole.exact_output_live_row_width";

Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

int64_t ComputeStaticElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         [](int64_t lhs, int64_t rhs) { return lhs * rhs; });
}

int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  ICHECK_GE(value, 0);
  return static_cast<int>((value + divisor - 1) / divisor);
}

std::string DataTypeToDataFormatForBlackhole(DataType dtype) {
  if (dtype.is_bfloat16()) return "Float16_b";
  if (dtype.is_float16()) return "Float16";
  if (dtype.is_float() && dtype.bits() == 32) return "Float32";
  if (dtype.is_float() && dtype.bits() == 8) return "Bfp8";
  if (dtype.is_uint() && dtype.bits() == 32) return "UInt32";
  if (dtype.is_uint() && dtype.bits() == 16) return "UInt16";
  if (dtype.is_int() && dtype.bits() == 32) return "Int32";
  if (dtype.is_int() && dtype.bits() == 16) return "Int16";
  return "Float16_b";
}

std::string GetStorageScope(const Buffer& buffer) {
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

}  // namespace

Buffer PlanTTKernelABI::CreateEphemeralBufferLike(const Buffer& buffer,
                                                  const std::string& suffix) const {
  const std::string name =
      BufferIdentityName(buffer) + "_" + suffix + "_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(buffer->shape, ExactTiledCBStorageDType(buffer->dtype), name,
                          GetStorageScope(buffer));
}

Buffer PlanTTKernelABI::CreateConstantTileBuffer(DataType dtype, const std::string& suffix) const {
  Array<PrimExpr> tile_shape{IntImm32(kBlackholeTileRows), IntImm32(kBlackholeTileCols)};
  const std::string name =
      "exact_const_tile_" + suffix + "_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(tile_shape, ExactTiledCBStorageDType(dtype), name, "local.fragment");
}

DataType PlanTTKernelABI::ExactTiledCBStorageDType(DataType dtype) const {
  if (dtype.is_float() && dtype.bits() == 32) {
    return DataType::BFloat(16);
  }
  return dtype;
}

int PlanTTKernelABI::PrepareExactTiledCBRequirement(const Buffer& buffer,
                                                    CBType type) {
  const int cb_id = AllocateRequirementIndex(buffer, type);
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const int num_tiles = GetLogicalBufferTileCount(buffer);
  const DataType storage_dtype = ExactTiledCBStorageDType(buffer->dtype);
  const int tile_bytes = kBlackholeTileRows * kBlackholeTileCols * storage_dtype.bytes();
  SetRequirementPageLayout(cb_id, tile_bytes, num_tiles);
  auto& req = cb_requirements_.at(cb_id);
  req.data_format = DataTypeToDataFormatForBlackhole(storage_dtype);
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, num_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, num_tiles);
  return cb_id;
}

Stmt PlanTTKernelABI::FillLocalTileBuffer(const Buffer& buffer, const PrimExpr& value) {
  return MakeBlackholeCall(blackhole_fill_fragment(),
                           {buffer->data, IntImm32(kBlackholeTileRows * kBlackholeTileCols),
                            value});
}

void PlanTTKernelABI::PopulateExactTiledCBValueShape(const Buffer& buffer,
                                                     ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  value->num_elements = GetLogicalBufferElementCount(buffer);
  value->num_tiles = GetLogicalBufferTileCount(buffer);
  value->row_width = GetLogicalMatrixShape(buffer).second;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(buffer)) {
    auto shape_it = spec->find(String(schema_key::kShape));
    if (shape_it != spec->end()) {
      std::vector<int64_t> shape;
      for (const Integer& dim : Downcast<Array<Integer>>((*shape_it).second)) {
        shape.push_back(dim->value);
      }
      if (shape.size() >= 2U) {
        value->num_elements = ComputeStaticElementCount(shape);
        value->row_width = shape.back();
        value->num_tiles =
            std::max(1, CeilDivToInt(shape[shape.size() - 2], kBlackholeTileRows) *
                            CeilDivToInt(shape.back(), kBlackholeTileCols));
      } else if (shape.size() == 1U) {
        value->num_elements = shape.front();
        value->row_width = 1;
        value->num_tiles = std::max(1, CeilDivToInt(shape.front(), kBlackholeTileRows));
      }
    }
  }
  if (value->row_width <= 0) {
    value->row_width = kBlackholeTileCols;
  }
  if (value->num_tiles <= 0) {
    value->num_tiles = 1;
  }
  if (value->num_elements <= 0) {
    value->num_elements =
        static_cast<int64_t>(value->num_tiles) * kBlackholeTileRows * kBlackholeTileCols;
  }
}

void PlanTTKernelABI::RefineExactTiledCBValueShapeFromRequirement(
    ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  if (value->cb_id < 0 ||
      value->cb_id >= static_cast<int>(cb_requirements_.size())) {
    return;
  }
  const CBRequirement& req = cb_requirements_.at(value->cb_id);
  const int event_tiles = std::max({req.publish_pages_per_event,
                                    req.consume_pages_per_event,
                                    value->num_tiles});
  value->num_tiles = std::max(1, event_tiles);
  if (value->num_elements <= 0) {
    value->num_elements =
        static_cast<int64_t>(value->num_tiles) * kBlackholeTileRows *
        kBlackholeTileCols;
  }
}

void PlanTTKernelABI::RefineExactTiledCBValueShapeFromNumElements(
    ExactTiledCBValue* value, const PrimExpr& num_elements) {
  ICHECK(value != nullptr);
  if (!num_elements.defined() || value->cb_id < 0) {
    return;
  }
  const auto* int_imm = num_elements.as<IntImmNode>();
  if (int_imm == nullptr || int_imm->value <= 0) {
    return;
  }
  const int64_t logical_elements = int_imm->value;
  constexpr int64_t kTileElements = kBlackholeTileRows * kBlackholeTileCols;
  const int required_tiles = std::max(1, CeilDivToInt(logical_elements, kTileElements));
  if (required_tiles <= value->num_tiles && logical_elements <= value->num_elements) {
    return;
  }
  value->num_tiles = std::max(value->num_tiles, required_tiles);
  value->num_elements = std::max<int64_t>(value->num_elements, logical_elements);
  ICHECK_LT(value->cb_id, static_cast<int>(cb_requirements_.size()));
  const DataType storage_dtype =
      value->buffer.defined() ? ExactTiledCBStorageDType(value->buffer->dtype)
                              : DataType::BFloat(16);
  SetRequirementPageLayout(
      value->cb_id,
      kBlackholeTileRows * kBlackholeTileCols * storage_dtype.bytes(),
      std::max(cb_requirements_.at(value->cb_id).num_pages, value->num_tiles));
  auto& req = cb_requirements_.at(value->cb_id);
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, value->num_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, value->num_tiles);
  req.data_format = DataTypeToDataFormatForBlackhole(storage_dtype);
}

CBType PlanTTKernelABI::ExactOutputCBTypeForBuffer(
    const Buffer& buffer, int current_order_index) const {
  const FutureBufferUses uses = ClassifyFutureBufferUses(buffer, current_order_index);
  if (uses.has_transport_consume && !uses.has_compute_consume) {
    return CBType::kOutput;
  }
  return CBType::kIntermediate;
}

void PlanTTKernelABI::MarkExactTiledCBValueConsumedByTransport(
    const ExactTiledCBValue& value) {
  if (value.cb_id < 0) {
    return;
  }
  ICHECK_LT(value.cb_id, static_cast<int>(cb_requirements_.size()));
  auto& req = cb_requirements_.at(value.cb_id);
  ICHECK(req.type != CBType::kInput)
      << "PlanTTKernelABI cannot consume input CB requirement as a writer output";
  req.type = CBType::kOutput;
}

bool PlanTTKernelABI::TryCreateLiveExactTiledCBValue(const Buffer& buffer,
                                                     ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  auto find_live_cb = [&](const std::string& name) -> std::pair<int, int> {
    if (name.empty()) {
      return {-1, -1};
    }
    auto it = buffer_live_form_cb_by_buffer_identity_.find(name);
    if (it == buffer_live_form_cb_by_buffer_identity_.end()) {
      return {-1, -1};
    }
    auto order_it = buffer_live_form_order_by_buffer_identity_.find(name);
    const int order_index =
        order_it == buffer_live_form_order_by_buffer_identity_.end() ? -1 : order_it->second;
    return {it->second, order_index};
  };

  int cb_id = -1;
  int live_order_index = -1;
  std::string selected_identity;
  auto consider_identity = [&](const std::string& identity) {
    auto [candidate_cb_id, candidate_order_index] = find_live_cb(identity);
    if (candidate_cb_id < 0) {
      return;
    }
    if (cb_id < 0 || candidate_order_index > live_order_index) {
      cb_id = candidate_cb_id;
      live_order_index = candidate_order_index;
      selected_identity = identity;
    }
  };

  consider_identity(BufferIdentityName(buffer));
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined()) {
    consider_identity(BufferIdentityName(physical));
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        consider_identity(identity);
      }
    }
  }
  if (cb_id < 0) return false;
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  if (cb_requirements_.at(cb_id).initial_reserve_pages > 0) {
    return false;
  }
  const std::string requested_identity = BufferIdentityName(buffer);
  if (!requested_identity.empty()) {
    auto [requested_cb_id, _] = find_live_cb(requested_identity);
    if (requested_cb_id == cb_id) {
      selected_identity = requested_identity;
    }
  }
  if (!selected_identity.empty() &&
      local_only_live_form_buffer_identities_.count(selected_identity) != 0U) {
    return false;
  }
  if (current_lowering_order_index_ >= 0 && live_order_index > current_lowering_order_index_) {
    return false;
  }
  if (HasInterveningBufferWrite(buffer, live_order_index, current_lowering_order_index_)) {
    return false;
  }
  auto invalidated_it = invalidated_live_form_order_by_buffer_identity_.find(selected_identity);
  if (invalidated_it != invalidated_live_form_order_by_buffer_identity_.end() &&
      invalidated_it->second >= live_order_index &&
      (current_lowering_order_index_ < 0 ||
       invalidated_it->second <= current_lowering_order_index_)) {
    return false;
  }
  value->buffer = buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  value->live_identity = selected_identity;
  PopulateExactTiledCBValueShape(buffer, value);
  RefineExactTiledCBValueShapeFromRequirement(value);
  return true;
}

bool PlanTTKernelABI::TryCreateExactOutputLiveTiledCBValue(const Buffer& buffer,
                                                           ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  auto find_live_cb = [&](const std::string& name) -> std::pair<int, int> {
    if (name.empty()) {
      return {-1, -1};
    }
    auto it = exact_output_live_form_cb_by_buffer_identity_.find(name);
    if (it == exact_output_live_form_cb_by_buffer_identity_.end()) {
      return {-1, -1};
    }
    auto order_it = exact_output_live_form_order_by_buffer_identity_.find(name);
    const int order_index =
        order_it == exact_output_live_form_order_by_buffer_identity_.end() ? -1 : order_it->second;
    return {it->second, order_index};
  };

  int cb_id = -1;
  int live_order_index = -1;
  std::string selected_identity;
  auto consider_identity = [&](const std::string& identity) {
    auto [candidate_cb_id, candidate_order_index] = find_live_cb(identity);
    if (candidate_cb_id < 0) {
      return;
    }
    if (cb_id < 0 || candidate_order_index > live_order_index) {
      cb_id = candidate_cb_id;
      live_order_index = candidate_order_index;
      selected_identity = identity;
    }
  };

  consider_identity(BufferIdentityName(buffer));
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined()) {
    consider_identity(BufferIdentityName(physical));
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        consider_identity(identity);
      }
    }
  }
  if (cb_id < 0) {
    return false;
  }
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  if (cb_requirements_.at(cb_id).initial_reserve_pages > 0) {
    return false;
  }
  const std::string requested_identity = BufferIdentityName(buffer);
  if (!requested_identity.empty()) {
    auto [requested_cb_id, _] = find_live_cb(requested_identity);
    if (requested_cb_id == cb_id) {
      selected_identity = requested_identity;
    }
  }
  if (!selected_identity.empty() &&
      local_only_live_form_buffer_identities_.count(selected_identity) != 0U) {
    return false;
  }
  if (current_lowering_order_index_ >= 0 && live_order_index > current_lowering_order_index_) {
    return false;
  }
  int latest_materialized_order = -1;
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    auto materialized_order_it = buffer_live_form_order_by_buffer_identity_.find(identity);
    if (materialized_order_it != buffer_live_form_order_by_buffer_identity_.end()) {
      latest_materialized_order =
          std::max(latest_materialized_order, materialized_order_it->second);
    }
  }
  if (latest_materialized_order > live_order_index &&
      (current_lowering_order_index_ < 0 ||
       latest_materialized_order <= current_lowering_order_index_)) {
    return false;
  }
  if (HasInterveningBufferWrite(buffer, live_order_index, current_lowering_order_index_)) {
    return false;
  }
  auto invalidated_it = invalidated_live_form_order_by_buffer_identity_.find(selected_identity);
  if (invalidated_it != invalidated_live_form_order_by_buffer_identity_.end() &&
      invalidated_it->second >= live_order_index &&
      (current_lowering_order_index_ < 0 ||
       invalidated_it->second <= current_lowering_order_index_)) {
    return false;
  }
  value->buffer = buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  value->live_identity = selected_identity;
  PopulateExactTiledCBValueShape(buffer, value);
  auto value_it = exact_output_live_form_value_by_buffer_identity_.find(selected_identity);
  if (value_it != exact_output_live_form_value_by_buffer_identity_.end()) {
    const ExactTiledCBValue& live_value = value_it->second;
    if (live_value.num_tiles > 0) {
      value->num_tiles = live_value.num_tiles;
    }
    if (live_value.num_elements > 0) {
      value->num_elements = live_value.num_elements;
    }
    if (live_value.row_width > 0) {
      value->row_width = live_value.row_width;
    }
  }
  RefineExactTiledCBValueShapeFromRequirement(value);
  return true;
}

bool PlanTTKernelABI::TryCreateLoopCarriedExactInputStateCBValue(
    const Buffer& buffer, ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  if (!buffer.defined() || !IsSingleFullTileLogicalMatrix(buffer)) {
    return false;
  }
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  const Buffer state_buffer = physical.defined() ? physical : buffer;
  if (GetStorageScope(state_buffer) != "blackhole.acc") {
    return false;
  }
  if (!IsActiveLoopCarriedBuffer(buffer) && !IsCompletedLoopCarriedBuffer(buffer) &&
      !IsActiveLoopCarriedBuffer(state_buffer) && !IsCompletedLoopCarriedBuffer(state_buffer)) {
    return false;
  }
  const bool prefer_loop_carried_state =
      IsCompletedLoopCarriedBuffer(buffer) || IsCompletedLoopCarriedBuffer(state_buffer);

  int cb_id = -1;
  std::string selected_identity;
  Buffer live_form_buffer;
  auto select_candidate = [&](const std::string& identity, int candidate_cb_id,
                              const Buffer& candidate_buffer) {
    if (cb_id >= 0 || identity.empty()) {
      return;
    }
    if (candidate_cb_id < 0) {
      return;
    }
    cb_id = candidate_cb_id;
    selected_identity = identity;
    live_form_buffer = candidate_buffer;
  };
  auto consider_loop_carried_identity = [&](const std::string& identity) {
    if (cb_id >= 0 || identity.empty() ||
        local_only_live_form_buffer_identities_.count(identity) != 0U) {
      return;
    }
    const LoopCarriedExactCBState* state = FindLoopCarriedExactCBState(identity);
    if (state == nullptr) {
      return;
    }
    select_candidate(identity, state->cb_id, state->buffer);
  };
  auto consider_identity = [&](const std::string& identity) {
    if (prefer_loop_carried_state) {
      consider_loop_carried_identity(identity);
      if (cb_id >= 0) {
        return;
      }
    }
    if (cb_id >= 0 || identity.empty() ||
        local_only_live_form_buffer_identities_.count(identity) != 0U) {
      return;
    }
    auto exact_cb_it = exact_output_live_form_cb_by_buffer_identity_.find(identity);
    if (exact_cb_it != exact_output_live_form_cb_by_buffer_identity_.end()) {
      Buffer exact_buffer;
      auto exact_value_it = exact_output_live_form_value_by_buffer_identity_.find(identity);
      if (exact_value_it != exact_output_live_form_value_by_buffer_identity_.end()) {
        exact_buffer = exact_value_it->second.buffer;
      }
      select_candidate(identity, exact_cb_it->second, exact_buffer);
      if (cb_id >= 0) {
        return;
      }
    }
    auto live_cb_it = buffer_live_form_cb_by_buffer_identity_.find(identity);
    if (live_cb_it != buffer_live_form_cb_by_buffer_identity_.end()) {
      select_candidate(identity, live_cb_it->second, Buffer());
      if (cb_id >= 0) {
        return;
      }
    }
    consider_loop_carried_identity(identity);
  };

  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    consider_identity(identity);
  }
  if (physical.defined() && !physical.same_as(buffer)) {
    for (const std::string& identity : CollectBufferFlowIdentities(physical)) {
      consider_identity(identity);
    }
  }
  if (cb_id < 0) {
    return false;
  }
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  if (cb_requirements_.at(cb_id).initial_reserve_pages > 0) {
    return false;
  }
  value->buffer = live_form_buffer.defined() ? live_form_buffer : state_buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  value->live_identity = selected_identity;
  PopulateExactTiledCBValueShape(buffer, value);
  RefineExactTiledCBValueShapeFromRequirement(value);
  return true;
}

bool PlanTTKernelABI::TryCreateLoopCarriedExactOutputStateCBValue(
    const Buffer& buffer, ExactTiledCBValue* value) const {
  ICHECK(value != nullptr);
  if (!buffer.defined() || active_loop_carried_buffer_identity_stack_.empty() ||
      active_serial_loop_order_ranges_.empty() || !IsSingleFullTileLogicalMatrix(buffer)) {
    return false;
  }
  const auto& loop_range = active_serial_loop_order_ranges_.back();
  if (current_lowering_order_index_ < loop_range.first ||
      current_lowering_order_index_ > loop_range.second) {
    return false;
  }

  int cb_id = -1;
  std::string selected_identity;
  Buffer live_form_buffer;
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    const FutureBufferUses future_uses =
        ClassifyFutureBufferIdentityReadsBeforeNextWriteUntilOrder(
            identity, current_lowering_order_index_, loop_range.second);
    if (future_uses.has_compute_consume || future_uses.has_transport_consume ||
        future_uses.has_reference) {
      return false;
    }
    const LoopCarriedExactCBState* state = FindLoopCarriedExactCBState(identity);
    if (state == nullptr) {
      continue;
    }
    cb_id = state->cb_id;
    selected_identity = identity;
    live_form_buffer = state->buffer;
    break;
  }
  if (cb_id < 0) {
    return false;
  }
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  value->buffer = live_form_buffer.defined() ? live_form_buffer : buffer;
  value->cb_id = cb_id;
  value->borrowed_live = false;
  value->live_identity = selected_identity;
  PopulateExactTiledCBValueShape(buffer, value);
  RefineExactTiledCBValueShapeFromRequirement(value);
  return true;
}

bool PlanTTKernelABI::TryCreateSelectedSourceLiveExactTiledCBValue(const Buffer& buffer,
                                                                   ExactTiledCBValue* value) {
  ICHECK(value != nullptr);
  if (!select_compute_builtins_only_ || !HasSelectedSourceLiveProducer(buffer)) {
    return false;
  }
  if (!IsSingleFullTileLogicalMatrix(buffer)) {
    return false;
  }
  std::string selected_identity;
  int producer_order_index = -1;
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    if (local_only_live_form_buffer_identities_.count(identity) != 0U) {
      return false;
    }
    auto producer_order_it =
        selected_source_live_producer_order_by_buffer_identity_.find(identity);
    if (producer_order_it == selected_source_live_producer_order_by_buffer_identity_.end()) {
      continue;
    }
    if (producer_order_index < 0 || producer_order_it->second > producer_order_index) {
      producer_order_index = producer_order_it->second;
      selected_identity = identity;
    }
  }
  if (producer_order_index < 0) {
    return false;
  }
  if (current_lowering_order_index_ >= 0 && producer_order_index > current_lowering_order_index_) {
    return false;
  }
  if (HasInterveningBufferWrite(buffer, producer_order_index, current_lowering_order_index_)) {
    return false;
  }
  const int cb_id = PrepareExactTiledCBRequirement(buffer);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  if (cb_requirements_.at(cb_id).initial_reserve_pages > 0) {
    return false;
  }
  value->buffer = buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  value->live_identity = selected_identity;
  PopulateExactTiledCBValueShape(buffer, value);
  return true;
}

bool PlanTTKernelABI::BorrowedExactInputHasNoFutureUseAt(
    const ExactTiledCBValue& value, int current_order_index) const {
  if (!value.borrowed_live || !value.buffer.defined() || value.cb_id < 0) {
    return false;
  }
  if (select_compute_builtins_only_ && value.num_tiles > 1 &&
      active_loop_carried_buffer_identity_stack_.empty()) {
    return false;
  }
  if (!value.live_identity.empty() &&
      BufferIdentityHasWriteAtOrder(value.live_identity, current_order_index)) {
    return true;
  }
  int upper_bound_order_index = -1;
  if (!active_serial_loop_order_ranges_.empty()) {
    const auto& loop_range = active_serial_loop_order_ranges_.back();
    const int producer_order_index = ResolveBorrowedExactInputProducerOrder(value);
    bool has_loop_local_write = false;
    auto inspect_identity = [&](const std::string& identity) {
      if (has_loop_local_write || identity.empty()) {
        return;
      }
      auto flow_it = buffer_flow_facts_.find(identity);
      if (flow_it == buffer_flow_facts_.end()) {
        return;
      }
      for (const BlackholeBufferFlowEvent& event : flow_it->second.events) {
        if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
            event.order_index >= loop_range.first &&
            event.order_index <= current_order_index) {
          has_loop_local_write = true;
          return;
        }
      }
    };
    if (!value.live_identity.empty()) {
      inspect_identity(value.live_identity);
    }
    for (const std::string& identity : CollectBufferFlowIdentities(value.buffer)) {
      inspect_identity(identity);
    }
    const bool produced_in_active_loop =
        producer_order_index >= loop_range.first &&
        producer_order_index <= loop_range.second;
    if (loop_range.first >= 0 && loop_range.second >= loop_range.first &&
        current_order_index >= loop_range.first &&
        current_order_index <= loop_range.second &&
        (produced_in_active_loop || has_loop_local_write)) {
      upper_bound_order_index = loop_range.second;
    }
  }
  const FutureBufferUses future_uses =
      !value.live_identity.empty()
          ? ClassifyFutureBufferIdentityReadsBeforeNextWriteUntilOrder(
                value.live_identity, current_order_index, upper_bound_order_index)
          : ClassifyFutureLiveCBReadsBeforeNextWriteUntilOrder(
                value.buffer, current_order_index, upper_bound_order_index);
  return !future_uses.has_compute_consume &&
         !future_uses.has_transport_consume &&
         !future_uses.has_reference;
}

int PlanTTKernelABI::ResolveBorrowedExactInputProducerOrder(
    const ExactTiledCBValue& value) const {
  int producer_order_index = -1;
  if (value.cb_id >= 0) {
    auto exact_cb_it = exact_output_live_form_order_by_cb_id_.find(value.cb_id);
    if (exact_cb_it != exact_output_live_form_order_by_cb_id_.end()) {
      producer_order_index = std::max(producer_order_index, exact_cb_it->second);
    }
    auto live_cb_it = buffer_live_form_order_by_cb_id_.find(value.cb_id);
    if (live_cb_it != buffer_live_form_order_by_cb_id_.end()) {
      producer_order_index = std::max(producer_order_index, live_cb_it->second);
    }
  }
  auto consider_identity = [&](const std::string& identity) {
    if (identity.empty()) {
      return;
    }
    auto exact_it = exact_output_live_form_order_by_buffer_identity_.find(identity);
    if (exact_it != exact_output_live_form_order_by_buffer_identity_.end()) {
      producer_order_index = std::max(producer_order_index, exact_it->second);
    }
    auto live_it = buffer_live_form_order_by_buffer_identity_.find(identity);
    if (live_it != buffer_live_form_order_by_buffer_identity_.end()) {
      producer_order_index = std::max(producer_order_index, live_it->second);
    }
    auto selected_it = selected_source_live_producer_order_by_buffer_identity_.find(identity);
    if (selected_it != selected_source_live_producer_order_by_buffer_identity_.end()) {
      producer_order_index = std::max(producer_order_index, selected_it->second);
    }
  };
  if (!value.live_identity.empty()) {
    consider_identity(value.live_identity);
    return producer_order_index;
  }
  for (const std::string& identity : CollectBufferFlowIdentities(value.buffer)) {
    consider_identity(identity);
  }
  return producer_order_index;
}

ffi::Optional<TTExactCBReleaseEvent>
PlanTTKernelABI::RecordExactCBUseAndReleaseEvent(
    const ExactTiledCBValue& value,
    int current_order_index,
    ExactCBReleasePolicy release_policy) {
  if (value.cb_id < 0 || value.num_tiles <= 0) {
    return ffi::Optional<TTExactCBReleaseEvent>();
  }
  const std::string logical_value =
      !value.live_identity.empty() ? value.live_identity : BufferIdentityName(value.buffer);
  const int64_t virtual_value_index =
      EnsureExactCBVirtualValue(logical_value, value, current_order_index);
  if (virtual_value_index < 0) {
    return ffi::Optional<TTExactCBReleaseEvent>();
  }
  const TTExactCBVirtualValue& virtual_value =
      tt_exact_cb_virtual_values_[static_cast<size_t>(virtual_value_index)];
  const std::string kernel_name =
      !current_segment_kind_.empty()
          ? current_segment_kind_
          : (requires_compute_segment_ ? std::string("compute") : std::string("main"));
  tt_exact_cb_use_events_.push_back(TTExactCBUseEvent(
      String("exact_cb_use_" + std::to_string(tt_exact_cb_use_events_.size())),
      virtual_value->name, virtual_value_index, String(kernel_name),
      String("compute_consume"), String("input"), std::max(current_order_index, 0),
      /*requires_full_logical_tile=*/true,
      String(value.borrowed_live ? "borrow" : "consume")));
  bool emit_release = false;
  switch (release_policy) {
    case ExactCBReleasePolicy::kNever:
      return ffi::Optional<TTExactCBReleaseEvent>();
    case ExactCBReleasePolicy::kAlways:
      emit_release = true;
      break;
    case ExactCBReleasePolicy::kBorrowedLastUse:
      emit_release =
          BorrowedExactInputHasNoFutureUseAt(value, current_order_index);
      break;
  }
  if (!emit_release) {
    return ffi::Optional<TTExactCBReleaseEvent>();
  }
  const int64_t allocation_index = EnsureExactCBAllocation(
      virtual_value_index, value, current_order_index, "last_use");
  ICHECK_GE(allocation_index, 0)
      << "Exact-CB release requires a typed allocation record for "
      << (!value.live_identity.empty() ? value.live_identity
                                       : BufferIdentityName(value.buffer));
  const TTExactCBAllocation& allocation =
      tt_exact_cb_allocations_[static_cast<size_t>(allocation_index)];
  const TTExactCBReleaseEvent release(
      String("exact_cb_release_" + std::to_string(tt_exact_cb_release_events_.size())),
      allocation->name, allocation_index, allocation->cb_plan,
      allocation->cb_plan_index, std::max(current_order_index, 0),
      std::max<int64_t>(1, value.num_tiles), String("last_use"));
  tt_exact_cb_release_events_.push_back(release);
  return release;
}

void PlanTTKernelABI::RecordLoopCarriedExactCBLifecycle(
    const std::string& logical_value,
    const ExactTiledCBValue& value,
    int program_point) {
  if (logical_value.empty() || value.cb_id < 0 || value.num_tiles <= 0) {
    return;
  }
  auto lifetime_it = spatial_lifetime_kind_by_subject_.find(logical_value);
  if (lifetime_it == spatial_lifetime_kind_by_subject_.end() ||
      lifetime_it->second != "loop_carried") {
    return;
  }

  ExactTiledCBValue lifecycle_value = value;
  lifecycle_value.live_identity = logical_value;
  lifecycle_value.borrowed_live = true;
  const int64_t virtual_value_index =
      EnsureExactCBVirtualValue(logical_value, lifecycle_value, program_point);
  if (virtual_value_index < 0) {
    return;
  }
  const TTExactCBVirtualValue& virtual_value =
      tt_exact_cb_virtual_values_[static_cast<size_t>(virtual_value_index)];
  bool has_loop_use = false;
  for (const TTExactCBUseEvent& event : tt_exact_cb_use_events_) {
    if (event->virtual_value_index == virtual_value_index &&
        static_cast<std::string>(event->consumer_event) == "loop_carried_state") {
      has_loop_use = true;
      break;
    }
  }
  if (!has_loop_use) {
    const std::string kernel_name =
        !current_segment_kind_.empty()
            ? current_segment_kind_
            : (requires_compute_segment_ ? std::string("compute")
                                         : std::string("main"));
    tt_exact_cb_use_events_.push_back(TTExactCBUseEvent(
        String("exact_cb_use_" + std::to_string(tt_exact_cb_use_events_.size())),
        virtual_value->name, virtual_value_index, String(kernel_name),
        String("loop_carried_state"), String("state"),
        std::max(program_point, 0), /*requires_full_logical_tile=*/true,
        String("borrow")));
  }

  const int64_t allocation_index = EnsureExactCBAllocation(
      virtual_value_index, lifecycle_value, program_point,
      "loop_backedge_transfer");
  if (allocation_index < 0) {
    return;
  }
  for (const TTExactCBReleaseEvent& event : tt_exact_cb_release_events_) {
    if (event->allocation_index == allocation_index &&
        static_cast<std::string>(event->reason) == "loop_backedge_transfer") {
      return;
    }
  }
  const TTExactCBAllocation& allocation =
      tt_exact_cb_allocations_[static_cast<size_t>(allocation_index)];
  tt_exact_cb_release_events_.push_back(TTExactCBReleaseEvent(
      String("exact_cb_release_" + std::to_string(tt_exact_cb_release_events_.size())),
      allocation->name, allocation_index, allocation->cb_plan,
      allocation->cb_plan_index, std::max(program_point, 0),
      std::max<int64_t>(1, lifecycle_value.num_tiles),
      String("loop_backedge_transfer")));
}

void PlanTTKernelABI::RememberLoopCarriedExactCBState(
    const std::string& logical_value,
    const ExactTiledCBValue& value,
    int program_point) {
  if (logical_value.empty() || value.cb_id < 0) {
    return;
  }
  auto& state = loop_carried_exact_cb_state_by_logical_value_[logical_value];
  state.cb_id = value.cb_id;
  state.buffer = value.buffer;
  state.program_point = std::max(state.program_point, program_point);
  RecordLoopCarriedExactCBLifecycle(logical_value, value, program_point);
}

const PlanTTKernelABI::LoopCarriedExactCBState*
PlanTTKernelABI::FindLoopCarriedExactCBState(
    const std::string& logical_value) const {
  if (logical_value.empty()) {
    return nullptr;
  }
  auto it = loop_carried_exact_cb_state_by_logical_value_.find(logical_value);
  return it == loop_carried_exact_cb_state_by_logical_value_.end()
             ? nullptr
             : &it->second;
}

bool PlanTTKernelABI::HasLoopCarriedExactCBState(
    const std::string& logical_value) const {
  return FindLoopCarriedExactCBState(logical_value) != nullptr;
}

int PlanTTKernelABI::GetLoopCarriedExactCBId(
    const std::string& logical_value) const {
  const LoopCarriedExactCBState* state =
      FindLoopCarriedExactCBState(logical_value);
  return state == nullptr ? -1 : state->cb_id;
}

Buffer PlanTTKernelABI::GetLoopCarriedExactCBBuffer(
    const std::string& logical_value) const {
  const LoopCarriedExactCBState* state =
      FindLoopCarriedExactCBState(logical_value);
  return state == nullptr ? Buffer() : state->buffer;
}

void PlanTTKernelABI::MarkLoopCarriedExactCBStateCompleted(
    const std::string& logical_value) {
  if (logical_value.empty()) {
    return;
  }
  auto it = loop_carried_exact_cb_state_by_logical_value_.find(logical_value);
  if (it != loop_carried_exact_cb_state_by_logical_value_.end()) {
    it->second.completed = true;
  }
}

Stmt PlanTTKernelABI::ReleaseExactInputAfterUse(
    const ExactTiledCBValue& value, int current_order_index) {
  if (value.cb_id < 0 || value.num_tiles <= 0) {
    return Stmt();
  }
  ffi::Optional<TTExactCBReleaseEvent> release_event =
      RecordExactCBUseAndReleaseEvent(
          value, current_order_index,
          value.borrowed_live ? ExactCBReleasePolicy::kBorrowedLastUse
                              : ExactCBReleasePolicy::kAlways);
  if (!release_event) {
    if (!value.borrowed_live) {
      return MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(value.cb_id), IntImm32(value.num_tiles)});
    }
    return Stmt();
  }
  if (!value.live_identity.empty()) {
    ClearTiledCBLiveFormIdentity(value.live_identity);
  } else {
    ClearTiledCBLiveFormAliases(value.buffer);
  }
  return MakeBlackholeCall(
      blackhole_cb_pop_front(),
      {IntImm32(static_cast<int>(release_event.value()->cb_plan_index)),
       IntImm32(static_cast<int>(release_event.value()->page_count))});
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateRowReductionInputCBValue(
    const Buffer& src) {
  ExactTiledCBValue live_value;
  const bool force_local_loop_carried =
      IsActiveLoopCarriedBuffer(src) && !IsSingleFullTileLogicalMatrix(src);
  const bool prefer_completed_loop_carried_state =
      IsCompletedLoopCarriedBuffer(src) && IsSingleFullTileLogicalMatrix(src);
  if (!force_local_loop_carried && prefer_completed_loop_carried_state &&
      TryCreateLoopCarriedExactInputStateCBValue(src, &live_value)) {
    return live_value;
  }
  if (!force_local_loop_carried &&
      (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
       TryCreateLiveExactTiledCBValue(src, &live_value) ||
       TryCreateLoopCarriedExactInputStateCBValue(src, &live_value) ||
       TryCreateSelectedSourceLiveExactTiledCBValue(src, &live_value))) {
    return live_value;
  }
  return CreatePublishedExactTiledCBValue(src, "reduce_src");
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateExactInputCBValue(
    const Buffer& src, const std::string& suffix) {
  ExactTiledCBValue live_value;
  const bool force_local_loop_carried =
      IsActiveLoopCarriedBuffer(src) && !IsSingleFullTileLogicalMatrix(src);
  const bool prefer_completed_loop_carried_state =
      IsCompletedLoopCarriedBuffer(src) && IsSingleFullTileLogicalMatrix(src);
  auto remember_loop_carried_state_cb = [&]() {
    const Buffer physical_src = ResolvePhysicalComputeBuffer(src);
    const Buffer state_src = physical_src.defined() ? physical_src : src;
    if (!IsActiveLoopCarriedBuffer(src) || !IsSingleFullTileLogicalMatrix(src) ||
        GetStorageScope(state_src) != "blackhole.acc" || live_value.cb_id < 0) {
      return;
    }
    for (const std::string& identity : CollectBufferFlowIdentities(src)) {
      if (identity.empty() || HasLoopCarriedExactCBState(identity)) {
        continue;
      }
      ExactTiledCBValue state_value = live_value;
      if (!state_value.buffer.defined()) {
        state_value.buffer = src;
      }
      state_value.live_identity = identity;
      RememberLoopCarriedExactCBState(identity, state_value,
                                      current_lowering_order_index_);
    }
  };
  if (!force_local_loop_carried && prefer_completed_loop_carried_state &&
      TryCreateLoopCarriedExactInputStateCBValue(src, &live_value)) {
    return live_value;
  }
  if (!force_local_loop_carried &&
      (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
       TryCreateLiveExactTiledCBValue(src, &live_value) ||
       TryCreateLoopCarriedExactInputStateCBValue(src, &live_value) ||
       TryCreateSelectedSourceLiveExactTiledCBValue(src, &live_value))) {
    remember_loop_carried_state_cb();
    return live_value;
  }
  return CreatePublishedExactTiledCBValue(src, suffix);
}

bool PlanTTKernelABI::TryGetLastFragmentFillValue(const Buffer& buffer, PrimExpr* value) const {
  ICHECK(value != nullptr);
  auto find_by_buffer = [&](const Buffer& candidate) -> bool {
    const std::string name = BufferIdentityName(candidate);
    if (name.empty()) {
      return false;
    }
    auto it = last_fragment_fill_value_by_buffer_identity_.find(name);
    if (it != last_fragment_fill_value_by_buffer_identity_.end()) {
      *value = it->second;
      return true;
    }
    return false;
  };
  if (find_by_buffer(buffer)) {
    return true;
  }
  const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
  if (physical.defined() && !physical.same_as(buffer) && find_by_buffer(physical)) {
    return true;
  }
  if (const VarNode* data = BufferDataIdentity(buffer)) {
    auto it = last_fragment_fill_value_by_data_.find(data);
    if (it != last_fragment_fill_value_by_data_.end()) {
      *value = it->second;
      return true;
    }
  }
  if (physical.defined() && !physical.same_as(buffer)) {
    if (const VarNode* data = BufferDataIdentity(physical)) {
      auto it = last_fragment_fill_value_by_data_.find(data);
      if (it != last_fragment_fill_value_by_data_.end()) {
        *value = it->second;
        return true;
      }
    }
  }
  return false;
}

Stmt PlanTTKernelABI::PublishConstantToExactTiledCB(const Buffer& buffer,
                                                   const PrimExpr& fill_value,
                                                   const ExactTiledCBValue& cb_value) {
  ICHECK(cb_value.cb_id >= 0);
  const Buffer physical_buffer = ResolvePhysicalComputeBuffer(buffer);
  std::vector<Stmt> stmts;
  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_pack_fill_fragment_to_tiled_cb(),
      {physical_buffer->data, IntImm32(cb_value.cb_id), IntImm32(0),
       IntImm32(static_cast<int>(cb_value.num_elements)),
       IntImm32(static_cast<int>(cb_value.row_width)), fill_value}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::PublishExactInputToTiledCB(const Buffer& src,
                                                ExactTiledCBValue* cb_value) {
  ICHECK(cb_value != nullptr);
  if (cb_value->producer_live || cb_value->borrowed_live) {
    return Stmt();
  }
  const bool force_local_loop_carried =
      IsActiveLoopCarriedBuffer(src) && !IsSingleFullTileLogicalMatrix(src);
  ExactTiledCBValue live_value;
  const bool prefer_completed_loop_carried_state =
      IsCompletedLoopCarriedBuffer(src) && IsSingleFullTileLogicalMatrix(src);
  if (!force_local_loop_carried && prefer_completed_loop_carried_state &&
      TryCreateLoopCarriedExactInputStateCBValue(src, &live_value)) {
    *cb_value = live_value;
    return Stmt();
  }
  if (!force_local_loop_carried &&
      (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
       TryCreateLiveExactTiledCBValue(src, &live_value) ||
       TryCreateLoopCarriedExactInputStateCBValue(src, &live_value))) {
    *cb_value = live_value;
    return Stmt();
  }
  PrimExpr fill_value;
  if (!force_local_loop_carried && TryGetLastFragmentFillValue(src, &fill_value)) {
    return PublishConstantToExactTiledCB(src, fill_value, *cb_value);
  }
  return PublishLocalBufferToExactTiledCB(src, *cb_value);
}

void PlanTTKernelABI::RecordExactOutputLiveForm(const Buffer& dst,
                                                const ExactTiledCBValue& cb_value) {
  RecordTiledCBLiveFormAliases(dst, cb_value.cb_id);
  if (cb_value.cb_id >= 0) {
    ICHECK_LT(cb_value.cb_id, static_cast<int>(cb_requirements_.size()));
    if (current_lowering_order_index_ >= 0) {
      exact_output_live_form_order_by_cb_id_[cb_value.cb_id] =
          current_lowering_order_index_;
    }
    const FutureBufferUses future_uses =
        ClassifyFutureBufferUses(dst, current_lowering_order_index_);
    const std::string dst_scope = GetStorageScope(dst);
    const bool dst_is_cb_scope =
        dst_scope.rfind("blackhole.cb", 0) == 0 || dst_scope.rfind("shared", 0) == 0;
    if (future_uses.has_transport_consume || dst_is_cb_scope) {
      auto& req = cb_requirements_.at(cb_value.cb_id);
      req.flow_class = CBFlowClass::kRepublish;
      const int event_pages = std::max(1, cb_value.num_tiles);
      req.publish_pages_per_event =
          std::max(req.publish_pages_per_event, event_pages);
      req.consume_pages_per_event =
          std::max(req.consume_pages_per_event, event_pages);
    }
  }
  const int order_index = current_lowering_order_index_;
  auto record_exact_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (identity.empty()) {
      return;
    }
    auto invalidated_it = invalidated_live_form_order_by_buffer_identity_.find(identity);
    if (invalidated_it != invalidated_live_form_order_by_buffer_identity_.end()) {
      if (order_index >= 0 && order_index < invalidated_it->second) {
        return;
      }
      invalidated_live_form_order_by_buffer_identity_.erase(invalidated_it);
    }
    exact_output_live_form_cb_by_buffer_identity_[identity] = cb_value.cb_id;
    exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
    exact_output_live_form_value_by_buffer_identity_[identity] = cb_value;
    local_only_live_form_buffer_identities_.erase(identity);
  };
  record_exact_buffer(dst);
  const Buffer physical = ResolvePhysicalComputeBuffer(dst);
  if (physical.defined()) {
    record_exact_buffer(physical);
    for (const auto& [identity, physical_candidate] : compute_physical_buffers_by_identity_) {
      if (!identity.empty() && physical_candidate.defined() &&
          SameBufferIdentity(physical_candidate, physical)) {
        exact_output_live_form_cb_by_buffer_identity_[identity] = cb_value.cb_id;
        exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
        exact_output_live_form_value_by_buffer_identity_[identity] = cb_value;
        local_only_live_form_buffer_identities_.erase(identity);
      }
    }
  }
  InvalidateLastFragmentFillValue(dst);
}

void PlanTTKernelABI::MarkExactCBValuesOverlap(std::initializer_list<int> cb_ids) {
  std::vector<int> valid_ids;
  for (int cb_id : cb_ids) {
    if (cb_id >= 0) {
      valid_ids.push_back(cb_id);
    }
  }
  for (size_t i = 0; i < valid_ids.size(); ++i) {
    for (size_t j = i + 1; j < valid_ids.size(); ++j) {
      if (valid_ids[i] != valid_ids[j]) {
        MarkRequirementLifetimeOverlap(valid_ids[i], valid_ids[j]);
      }
    }
  }
}

Stmt PlanTTKernelABI::PublishLocalBufferToExactTiledCB(const Buffer& src,
                                                       const ExactTiledCBValue& cb_value) {
  ICHECK(cb_value.cb_id >= 0);
  const Buffer physical_src = ResolvePhysicalComputeBuffer(src);
  const bool requires_cast_bridge =
      cb_value.buffer.defined() && cb_value.buffer->dtype != physical_src->dtype;
  Array<Stmt> stmts{MakeBlackholeCall(
      blackhole_cb_reserve_back(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)})};
  if (requires_cast_bridge) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_tilize_cast_fragment_slice(),
        {cb_value.buffer->data, physical_src->data, IntImm32(cb_value.cb_id), IntImm32(0),
         IntImm32(0), IntImm32(static_cast<int>(cb_value.num_elements)),
         IntImm32(static_cast<int>(cb_value.row_width))}));
  } else {
    stmts.push_back(MakeBlackholeCall(
        blackhole_tilize_local_fragment_slice(),
        {physical_src->data, IntImm32(cb_value.cb_id), IntImm32(0),
         IntImm32(static_cast<int>(cb_value.num_elements)),
         IntImm32(static_cast<int>(cb_value.row_width))}));
  }
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_push_back(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::MaterializeExactTiledCBToLocalBuffer(const Buffer& dst,
                                                           const ExactTiledCBValue& cb_value,
                                                           bool pop_front) {
  ICHECK(cb_value.cb_id >= 0);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  Array<Stmt> stmts;
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_wait_front(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  constexpr int kTileElements = kBlackholeTileRows * kBlackholeTileCols;
  for (int tile = 0; tile < cb_value.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_untilize_cb_front_tile_fragment(),
        {physical_dst->data, IntImm32(cb_value.cb_id), IntImm32(tile),
         IntImm32(tile * kTileElements)}));
  }
  ffi::Optional<TTExactCBReleaseEvent> release_event =
      RecordExactCBUseAndReleaseEvent(
          cb_value, current_lowering_order_index_,
          pop_front ? ExactCBReleasePolicy::kAlways
                    : ExactCBReleasePolicy::kNever);
  if (pop_front) {
    if (release_event) {
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(),
          {IntImm32(static_cast<int>(release_event.value()->cb_plan_index)),
           IntImm32(static_cast<int>(release_event.value()->page_count))}));
    } else {
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
    }
  }
  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::AttachExactOutputLiveFormMarker(const Buffer& dst,
                                                      const ExactTiledCBValue& cb_value,
                                                      const Stmt& body) const {
  if (!select_compute_builtins_only_ || !dst.defined() || cb_value.cb_id < 0) {
    return body;
  }
  const PrimExpr identity = tir::StringImm(BufferIdentityName(dst));
  Stmt marked = body;
  marked = AttrStmt(identity, kBlackholeExactOutputLiveRowWidthAttr,
                    IntImm(DataType::Int(64), cb_value.row_width), marked);
  marked = AttrStmt(identity, kBlackholeExactOutputLiveNumElementsAttr,
                    IntImm(DataType::Int(64), cb_value.num_elements), marked);
  marked = AttrStmt(identity, kBlackholeExactOutputLiveNumTilesAttr,
                    IntImm32(cb_value.num_tiles), marked);
  marked = AttrStmt(identity, kBlackholeExactOutputLiveCBAttr,
                    IntImm32(cb_value.cb_id), marked);
  return marked;
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreatePublishedExactTiledCBValue(
    const Buffer& src, const std::string& suffix, CBType type) {
  ExactTiledCBValue value;
  value.buffer = CreateEphemeralBufferLike(src, suffix);
  PopulateExactTiledCBValueShape(src, &value);
  value.cb_id = PrepareExactTiledCBRequirement(value.buffer, type);
  SetRequirementPageLayout(value.cb_id,
                           kBlackholeTileRows * kBlackholeTileCols * value.buffer->dtype.bytes(),
                           value.num_tiles);
  auto& req = cb_requirements_.at(value.cb_id);
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, value.num_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, value.num_tiles);
  return value;
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateEmptyExactTiledCBValue(
    const Buffer& like_buffer, const std::string& suffix, CBType type) {
  return CreatePublishedExactTiledCBValue(like_buffer, suffix, type);
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateConstantExactTiledCBValue(
    DataType dtype, const std::string& suffix) {
  ExactTiledCBValue cb_value;
  cb_value.buffer = CreateConstantTileBuffer(dtype, suffix);
  cb_value.num_tiles = 1;
  cb_value.num_elements = kBlackholeTileRows * kBlackholeTileCols;
  cb_value.row_width = kBlackholeTileCols;
  cb_value.cb_id = PrepareExactTiledCBRequirement(cb_value.buffer);
  return cb_value;
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateReduceScalerExactTiledCBValue() {
  ExactTiledCBValue cb_value = CreateConstantExactTiledCBValue(DataType::Float(32), "reduce_scaler");
  const DataType scaler_cb_dtype = DataType::BFloat(16);
  SetRequirementPageLayout(cb_value.cb_id,
                           kBlackholeTileRows * kBlackholeTileCols * scaler_cb_dtype.bytes(),
                           cb_value.num_tiles);
  auto& req = cb_requirements_.at(cb_value.cb_id);
  req.data_format = DataTypeToDataFormatForBlackhole(scaler_cb_dtype);
  return cb_value;
}

}  // namespace tl
}  // namespace tvm
