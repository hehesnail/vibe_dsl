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

int PlanTTKernelABI::PrepareExactTiledCBRequirement(const Buffer& buffer) {
  const int cb_id = AllocateRequirementIndex(buffer, CBType::kIntermediate);
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
  auto consider_identity = [&](const std::string& identity) {
    auto [candidate_cb_id, candidate_order_index] = find_live_cb(identity);
    if (candidate_cb_id < 0) {
      return;
    }
    if (cb_id < 0 || candidate_order_index > live_order_index) {
      cb_id = candidate_cb_id;
      live_order_index = candidate_order_index;
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
  if (HasInterveningBufferWrite(buffer, live_order_index, current_lowering_order_index_)) {
    return false;
  }
  value->buffer = buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  PopulateExactTiledCBValueShape(buffer, value);
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
  auto consider_identity = [&](const std::string& identity) {
    auto [candidate_cb_id, candidate_order_index] = find_live_cb(identity);
    if (candidate_cb_id < 0) {
      return;
    }
    if (cb_id < 0 || candidate_order_index > live_order_index) {
      cb_id = candidate_cb_id;
      live_order_index = candidate_order_index;
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
  if (HasInterveningBufferWrite(buffer, live_order_index, current_lowering_order_index_)) {
    return false;
  }
  value->buffer = buffer;
  value->cb_id = cb_id;
  value->borrowed_live = true;
  PopulateExactTiledCBValueShape(buffer, value);
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
  value->buffer = buffer;
  value->cb_id = PrepareExactTiledCBRequirement(buffer);
  value->borrowed_live = true;
  PopulateExactTiledCBValueShape(buffer, value);
  return true;
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateRowReductionInputCBValue(
    const Buffer& src) {
  ExactTiledCBValue live_value;
  if (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
      TryCreateSelectedSourceLiveExactTiledCBValue(src, &live_value) ||
      TryCreateLiveExactTiledCBValue(src, &live_value)) {
    return live_value;
  }
  return CreatePublishedExactTiledCBValue(src, "reduce_src");
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateExactInputCBValue(
    const Buffer& src, const std::string& suffix) {
  ExactTiledCBValue live_value;
  if (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
      TryCreateSelectedSourceLiveExactTiledCBValue(src, &live_value) ||
      TryCreateLiveExactTiledCBValue(src, &live_value)) {
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
  if (cb_value->borrowed_live) {
    return Stmt();
  }
  ExactTiledCBValue live_value;
  if (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
      TryCreateLiveExactTiledCBValue(src, &live_value)) {
    *cb_value = live_value;
    return Stmt();
  }
  PrimExpr fill_value;
  if (TryGetLastFragmentFillValue(src, &fill_value)) {
    return PublishConstantToExactTiledCB(src, fill_value, *cb_value);
  }
  return PublishLocalBufferToExactTiledCB(src, *cb_value);
}

void PlanTTKernelABI::RecordExactOutputLiveForm(const Buffer& dst,
                                                const ExactTiledCBValue& cb_value) {
  RecordTiledCBLiveFormAliases(dst, cb_value.cb_id);
  const int order_index = current_lowering_order_index_;
  auto record_exact_buffer = [&](const Buffer& candidate) {
    const std::string identity = BufferIdentityName(candidate);
    if (identity.empty()) {
      return;
    }
    exact_output_live_form_cb_by_buffer_identity_[identity] = cb_value.cb_id;
    exact_output_live_form_order_by_buffer_identity_[identity] = order_index;
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
  if (pop_front) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  }
  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::AttachExactOutputLiveFormMarker(const Buffer& dst,
                                                      const ExactTiledCBValue& cb_value,
                                                      const Stmt& body) const {
  if (!select_compute_builtins_only_ || !dst.defined() || cb_value.cb_id < 0) {
    return body;
  }
  return AttrStmt(dst->data, kBlackholeExactOutputLiveCBAttr, IntImm32(cb_value.cb_id), body);
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreatePublishedExactTiledCBValue(
    const Buffer& src, const std::string& suffix) {
  ExactTiledCBValue value;
  value.buffer = CreateEphemeralBufferLike(src, suffix);
  PopulateExactTiledCBValueShape(src, &value);
  value.cb_id = PrepareExactTiledCBRequirement(value.buffer);
  SetRequirementPageLayout(value.cb_id,
                           kBlackholeTileRows * kBlackholeTileCols * value.buffer->dtype.bytes(),
                           value.num_tiles);
  auto& req = cb_requirements_.at(value.cb_id);
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, value.num_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, value.num_tiles);
  return value;
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreateEmptyExactTiledCBValue(
    const Buffer& like_buffer, const std::string& suffix) {
  return CreatePublishedExactTiledCBValue(like_buffer, suffix);
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
