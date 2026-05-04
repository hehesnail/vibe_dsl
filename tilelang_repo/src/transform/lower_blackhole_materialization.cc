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
 * \file lower_blackhole_materialization.cc
 * \brief Fragment/local materialization planning for Blackhole lowering.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::Allocate;
using tir::AllocateNode;
using tir::AttrStmtNode;
using tir::Buffer;
using tir::BufferLoadNode;
using tir::BufferStoreNode;
using tir::Call;
using tir::CallNode;
using tir::CastNode;
using tir::DeclBuffer;
using tir::DeclBufferNode;
using tir::Evaluate;
using tir::For;
using tir::ForNode;
using tir::SeqStmt;
using tir::SeqStmtNode;
using tir::Stmt;
using tir::Var;
using tir::VarNode;
using tir::builtin::blackhole_cast_fragment_slice;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_copy_tile_to_dst_init_short;
using tir::builtin::blackhole_pack_fill_fragment_to_tiled_cb;
using tir::builtin::blackhole_pack_reconfig_data_format;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_tilize_cast_fragment_slice;
using tir::builtin::blackhole_tilize_local_fragment_slice;
using tvm::Bool;
using tvm::DataType;
using tvm::Integer;
using tvm::IntImm;
using tvm::arith::Analyzer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
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

int64_t ProductIntegerArrayField(const Map<String, Any>& map, const char* key,
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

bool IsUnsupportedResidualLocalScope(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

bool IsVectorLocalFragmentBuffer(const Buffer& buffer) {
  return IsUnsupportedResidualLocalScope(buffer) && buffer->shape.size() == 1 &&
         !buffer->shape.empty() && !tir::is_one(buffer->shape[0]);
}

std::vector<Stmt> FlattenSeqStmtBody(const Stmt& body) {
  if (const auto* seq = body.as<SeqStmtNode>()) {
    return std::vector<Stmt>(seq->seq.begin(), seq->seq.end());
  }
  return {body};
}

const ForNode* AsUnwrappedFor(const Stmt& stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto* attr = current.as<AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto* decl = current.as<DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<AllocateNode>()) {
      current = alloc->body;
      continue;
    }
    break;
  }
  return current.as<ForNode>();
}

const BufferStoreNode* AsUnwrappedBufferStore(const Stmt& stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto* attr = current.as<AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto* decl = current.as<DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<AllocateNode>()) {
      current = alloc->body;
      continue;
    }
    break;
  }
  return current.as<BufferStoreNode>();
}

bool ExprUsesVar(const PrimExpr& expr, const Var& var) {
  bool found = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (const auto* var_node = node.as<VarNode>()) {
      if (var_node == var.get()) {
        found = true;
      }
    }
  });
  return found;
}

}  // namespace

bool PlanTTKernelABI::MatchDirectFragmentCast(const ForNode* op,
                                                FragmentCastMatch* match) const {
  if (!op || !match) {
    return false;
  }

  const auto* store = AsUnwrappedBufferStore(op->body);
  const ForNode* inner_loop = nullptr;
  const auto* inner_store = store;
  PrimExpr linear_index = op->loop_var;
  PrimExpr num_elements = op->extent;
  if (!store) {
    inner_loop = AsUnwrappedFor(op->body);
    inner_store = inner_loop ? AsUnwrappedBufferStore(inner_loop->body) : nullptr;
    if (!inner_loop || !inner_store || inner_store->indices.size() != 1) {
      return false;
    }
    linear_index = op->loop_var * inner_loop->extent + inner_loop->loop_var;
    num_elements = Analyzer().Simplify(op->extent * inner_loop->extent);
  } else if (store->indices.size() != 1) {
    return false;
  }

  if (!inner_store || !IsVectorLocalFragmentBuffer(inner_store->buffer)) {
    return false;
  }
  const auto* cast = inner_store->value.as<CastNode>();
  const auto* load = cast ? cast->value.as<BufferLoadNode>() : nullptr;
  if (!cast || !load || load->indices.size() != 1 || !IsVectorLocalFragmentBuffer(load->buffer)) {
    return false;
  }

  Analyzer analyzer;
  PrimExpr dst_offset = analyzer.Simplify(inner_store->indices[0] - linear_index);
  PrimExpr src_offset = analyzer.Simplify(load->indices[0] - linear_index);
  if (ExprUsesVar(dst_offset, op->loop_var) || ExprUsesVar(src_offset, op->loop_var)) {
    return false;
  }
  if (inner_loop &&
      (ExprUsesVar(dst_offset, inner_loop->loop_var) || ExprUsesVar(src_offset, inner_loop->loop_var))) {
    return false;
  }

  match->dst = inner_store->buffer;
  match->src = load->buffer;
  match->dst_offset = dst_offset;
  match->src_offset = src_offset;
  match->num_elements = num_elements;
  match->row_width = PrimExpr();
  const auto [logical_src_rows, logical_src_cols] = GetLogicalMatrixShape(load->buffer);
  const auto [logical_dst_rows, logical_dst_cols] = GetLogicalMatrixShape(inner_store->buffer);
  const int64_t logical_src_extent = logical_src_rows > 0 && logical_src_cols > 0
                                         ? logical_src_rows * logical_src_cols
                                         : -1;
  const int64_t logical_dst_matrix_extent =
      logical_dst_rows > 0 && logical_dst_cols > 0 ? logical_dst_rows * logical_dst_cols : -1;
  const int64_t logical_dst_vector_extent = GetLogicalVectorLength(inner_store->buffer);
  if (logical_src_extent > 0 && logical_dst_matrix_extent > 0 &&
      logical_src_extent == logical_dst_matrix_extent && tir::is_zero(match->src_offset) &&
      tir::is_zero(match->dst_offset)) {
    match->num_elements = IntImm(DataType::Int(32), logical_src_extent);
    match->row_width = IntImm(DataType::Int(32), logical_src_cols);
    return true;
  }
  if (logical_src_rows > 0 && logical_src_cols > 0 &&
      (logical_dst_vector_extent > 0 || logical_dst_matrix_extent > 0)) {
    match->row_width = IntImm(DataType::Int(32), logical_src_cols);
    if (logical_dst_vector_extent > 0 && logical_dst_matrix_extent <= 0) {
      Var thread_row_var = SelectLogicalRowThreadVar(logical_src_rows);
      if (thread_row_var.defined() && !ExprUsesVar(match->src_offset, thread_row_var)) {
        match->src_offset = analyzer.Simplify(
            thread_row_var * IntImm(DataType::Int(32), logical_src_cols) + match->src_offset);
      }
    }
  }
  return true;
}

Stmt PlanTTKernelABI::GenerateFragmentCastSequence(const FragmentCastMatch& match,
                                                    bool publish_cb,
                                                    int current_order_index,
                                                    bool allow_force_publish_from_fact) {
  InvalidateLastFragmentFillValue(match.dst);
  const bool force_publish_from_fact =
      allow_force_publish_from_fact && GetStorageScope(match.dst) == "blackhole.acc" &&
      FindBufferMaterializationFact(match.dst) != nullptr;
  const bool publish_result = publish_cb || force_publish_from_fact;
  PrimExpr num_elements_expr = match.num_elements;
  std::vector<Stmt> stmts;
  bool use_tiled_republish_materialization = false;
  PrimExpr pack_thread_direct_fill_value;
  PrimExpr tiled_republish_row_width;
  const BlackholeBufferMaterializationFact* materialization_fact = nullptr;
  int cb_id = -1;
  int num_pages = 0;
  const std::string dst_buffer_name = BufferIdentityName(match.dst);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(match.dst);
  const Buffer physical_src = ResolvePhysicalComputeBuffer(match.src);
  Buffer publish_buffer;
  bool publish_to_existing_compute_input = false;
  int consumed_compute_input_pages = 0;
  if (match.src->dtype != match.dst->dtype) {
    RecordExactComputeOpPlan("unary", "typecast_tile",
                             {{"input", match.src, "identity"},
                              {"output", match.dst, "identity"}});
  }
  if (publish_result) {
    publish_buffer = match.dst;
    for (const std::string& buffer_identity : CollectBufferFlowIdentities(match.dst)) {
      auto consumed_it = cb_consumed_compute_input_pages_by_buffer_identity_.find(buffer_identity);
      if (consumed_it != cb_consumed_compute_input_pages_by_buffer_identity_.end()) {
        consumed_compute_input_pages =
            std::max(consumed_compute_input_pages, consumed_it->second);
      }
      auto existing_req_it = buffer_identity_to_req_index_.find(buffer_identity);
      if (existing_req_it != buffer_identity_to_req_index_.end() &&
          existing_req_it->second >= 0 &&
          existing_req_it->second < static_cast<int>(cb_requirements_.size())) {
        const CBRequirement& existing_req = cb_requirements_.at(existing_req_it->second);
        if (existing_req.type == CBType::kInput &&
            (existing_req.consume_pages_per_event > 0 || consumed_compute_input_pages > 0)) {
          publish_to_existing_compute_input = true;
          auto buffer_it = buffer_by_identity_.find(buffer_identity);
          if (buffer_it != buffer_by_identity_.end() && buffer_it->second.defined()) {
            publish_buffer = buffer_it->second;
          }
        }
      }
    }
    if (GetStorageScope(match.dst) == "blackhole.acc" &&
        !publish_to_existing_compute_input) {
      publish_buffer = CreateEphemeralBufferLike(match.dst, "cast_publish");
    }
    cb_id = AllocateRequirementIndex(publish_buffer, CBType::kIntermediate);
    ICHECK_GE(cb_id, 0);
    ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
    num_pages = std::max(
        1, cb_requirements_[cb_id].publish_pages_per_event > 0
               ? cb_requirements_[cb_id].publish_pages_per_event
               : cb_requirements_[cb_id].num_pages);
    if (consumed_compute_input_pages > 0 &&
        StaticIntValueOrDefault(num_elements_expr, int64_t{0}) <= 0) {
      num_pages = std::max(num_pages, consumed_compute_input_pages);
      auto& req = cb_requirements_[cb_id];
      SetRequirementPageLayout(cb_id, req.page_size,
                               std::max(req.num_pages, num_pages));
      req.publish_pages_per_event = std::max(req.publish_pages_per_event, num_pages);
      req.consume_pages_per_event =
          std::max(req.consume_pages_per_event, consumed_compute_input_pages);
    }
    const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(match.dst);
    if (fact != nullptr) {
      materialization_fact = fact;
      use_tiled_republish_materialization =
          fact->kind == buffer_materialization::kRepublishedLogicalTile &&
          fact->bridge_kind == buffer_materialization::kTileNFacesMaterialization &&
          fact->execution_protocol == buffer_materialization::kTiledCBRepublish;
      if (use_tiled_republish_materialization) {
        auto layout_spec_it = logical_tile_layout_specs_by_buffer_.find(dst_buffer_name);
        auto apply_typed_layout_shape_spec = [&](const Map<String, Any>* spec) {
          if (spec == nullptr) {
            return;
          }
          const int64_t logical_element_count =
              ProductIntegerArrayField(*spec, schema_key::kShape, int64_t{0});
          if (logical_element_count > StaticIntValueOrDefault(num_elements_expr, int64_t{0})) {
            num_elements_expr = IntImm(DataType::Int(32),
                                       static_cast<int>(logical_element_count));
          }
          auto shape_it = spec->find(String(schema_key::kShape));
          if (shape_it != spec->end()) {
            const Array<Integer> logical_shape = Downcast<Array<Integer>>((*shape_it).second);
            if (logical_shape.size() >= 2U && logical_shape.back()->value > 0) {
              tiled_republish_row_width =
                  IntImm(DataType::Int(32), static_cast<int>(logical_shape.back()->value));
            }
          }
        };
        auto apply_typed_layout_shape = [&](const Buffer& buffer) {
          apply_typed_layout_shape_spec(FindLogicalTileLayoutSpec(buffer));
        };
        auto apply_typed_layout_shape_by_name = [&](const std::string& buffer_name) {
          auto it = logical_tile_layout_specs_by_buffer_.find(buffer_name);
          if (it != logical_tile_layout_specs_by_buffer_.end()) {
            apply_typed_layout_shape_spec(&it->second);
          }
        };
        auto apply_typed_layout_shape_from_fact = [&](const std::string& buffer_name) {
          if (!buffer_name.empty()) {
            apply_typed_layout_shape_by_name(buffer_name);
          }
        };
        if (fact->logical_row_width > 0) {
          tiled_republish_row_width =
              IntImm(DataType::Int(32), static_cast<int>(fact->logical_row_width));
        } else if (layout_spec_it != logical_tile_layout_specs_by_buffer_.end()) {
          apply_typed_layout_shape_spec(&layout_spec_it->second);
        }
        if (!tiled_republish_row_width.defined()) {
          tiled_republish_row_width = match.row_width;
          if (!tiled_republish_row_width.defined()) {
            const auto [logical_src_rows, logical_src_cols] = GetLogicalMatrixShape(match.src);
            if (logical_src_rows > 0 && logical_src_cols > 0) {
              tiled_republish_row_width = IntImm(DataType::Int(32), logical_src_cols);
            }
          }
          if (!tiled_republish_row_width.defined()) {
            const auto [logical_dst_rows, logical_dst_cols] = GetLogicalMatrixShape(match.dst);
            if (logical_dst_rows > 0 && logical_dst_cols > 0) {
              tiled_republish_row_width = IntImm(DataType::Int(32), logical_dst_cols);
            }
          }
        }
        ICHECK(tiled_republish_row_width.defined())
            << "PlanTTKernelABI requires logical_row_width fact or structural row_width "
               "for tiled republish materialization of "
            << dst_buffer_name;
        if (fact->logical_element_count > 0) {
          num_elements_expr =
              IntImm(DataType::Int(32), static_cast<int>(fact->logical_element_count));
        }
        apply_typed_layout_shape(match.src);
        apply_typed_layout_shape(match.dst);
        apply_typed_layout_shape_from_fact(fact->source_buffer);
        apply_typed_layout_shape_from_fact(fact->target_buffer);
        auto fill_value_it =
            last_fragment_fill_value_by_buffer_identity_.find(BufferIdentityName(match.src));
        auto fill_data_value_it =
            last_fragment_fill_value_by_data_.find(BufferDataIdentity(match.src));
        if (fill_value_it != last_fragment_fill_value_by_buffer_identity_.end() &&
            match.dst->dtype.is_bfloat16() && tir::is_zero(match.src_offset)) {
          pack_thread_direct_fill_value = fill_value_it->second;
        } else if (fill_data_value_it != last_fragment_fill_value_by_data_.end() &&
                   match.dst->dtype.is_bfloat16() && tir::is_zero(match.src_offset)) {
          pack_thread_direct_fill_value = fill_data_value_it->second;
        }
        RecordFragmentCastMaterializationPlans(
            match, *fact, cb_id, num_elements_expr,
            pack_thread_direct_fill_value.defined()
                ? buffer_materialization::kPackThreadDirectStore
                : buffer_materialization::kTilizeCastFragmentSlice);
      }
    }
    const int64_t logical_elements = StaticIntValueOrDefault(num_elements_expr, int64_t{0});
    if (logical_elements > 0) {
      constexpr int64_t kTileElements = 32 * 32;
      const int logical_pages = CeilDivToInt(logical_elements, kTileElements);
      num_pages = publish_to_existing_compute_input
                      ? std::max(logical_pages, consumed_compute_input_pages)
                      : logical_pages;
      const DataType publish_dtype =
          publish_buffer.defined() ? publish_buffer->dtype : match.dst->dtype;
      auto& req = cb_requirements_[cb_id];
      SetRequirementPageLayout(cb_id, 32 * 32 * publish_dtype.bytes(),
                               std::max(req.num_pages, num_pages));
      req.publish_pages_per_event = std::max(req.publish_pages_per_event, num_pages);
      req.consume_pages_per_event = std::max(req.consume_pages_per_event, num_pages);
      req.data_format = DataTypeToDataFormatForBlackhole(publish_dtype);
    }
  }
  if (use_tiled_republish_materialization) {
    ExactTiledCBValue live_source;
    auto try_exact_source_live_by_fact = [&]() -> bool {
      if (materialization_fact == nullptr || materialization_fact->source_buffer.empty()) {
        return false;
      }
      auto it =
          exact_output_live_form_cb_by_buffer_identity_.find(materialization_fact->source_buffer);
      if (it == exact_output_live_form_cb_by_buffer_identity_.end()) {
        return false;
      }
      int live_order_index = -1;
      auto order_it = exact_output_live_form_order_by_buffer_identity_.find(
          materialization_fact->source_buffer);
      if (order_it != exact_output_live_form_order_by_buffer_identity_.end()) {
        live_order_index = order_it->second;
      }
      if (HasInterveningBufferWrite(match.src, live_order_index, current_lowering_order_index_)) {
        return false;
      }
      live_source.buffer = match.src;
      live_source.cb_id = it->second;
      live_source.borrowed_live = true;
      PopulateExactTiledCBValueShape(match.src, &live_source);
      auto value_it = exact_output_live_form_value_by_buffer_identity_.find(
          materialization_fact->source_buffer);
      if (value_it != exact_output_live_form_value_by_buffer_identity_.end()) {
        const ExactTiledCBValue& value = value_it->second;
        if (value.num_tiles > 0) {
          live_source.num_tiles = value.num_tiles;
        }
        if (value.num_elements > 0) {
          live_source.num_elements = value.num_elements;
        }
        if (value.row_width > 0) {
          live_source.row_width = value.row_width;
        }
      }
      return true;
    };
    const bool can_republish_from_live_cb =
        tir::is_zero(match.src_offset) && tir::is_zero(match.dst_offset) &&
        (try_exact_source_live_by_fact() ||
         TryCreateExactOutputLiveTiledCBValue(match.src, &live_source) ||
         TryCreateLiveExactTiledCBValue(match.src, &live_source)) &&
        live_source.num_tiles == num_pages;
    if (can_republish_from_live_cb && !pack_thread_direct_fill_value.defined()) {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                        {IntImm32(live_source.cb_id), IntImm32(live_source.num_tiles)}));
      stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                        {IntImm32(cb_id), IntImm32(num_pages)}));
      MarkExactCBValuesOverlap({live_source.cb_id, cb_id});
      for (int tile = 0; tile < num_pages; ++tile) {
        stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
        stmts.push_back(
            MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(), {IntImm32(live_source.cb_id)}));
        stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                          {IntImm32(live_source.cb_id), IntImm32(tile), IntImm32(0)}));
        stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
        stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
        stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(cb_id)}));
        stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                          {IntImm32(0), IntImm32(cb_id), IntImm32(tile)}));
        stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
      }
      if (Stmt release = ReleaseExactInputAfterUse(live_source, current_order_index);
          release.defined()) {
        stmts.push_back(release);
      }
      stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cb_push_back(),
                                        {IntImm32(cb_id), IntImm32(num_pages)}));
      RecordTiledCBLiveFormAliases(match.dst, cb_id);
      return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
    }
    // The republish fact says the result becomes cb-live. Whether the logical
    // buffer also happens to use blackhole.acc storage does not imply a page has
    // already been reserved for the publish event.
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
    if (pack_thread_direct_fill_value.defined()) {
      stmts.push_back(MakeBlackholeCall(
          blackhole_pack_fill_fragment_to_tiled_cb(),
          {physical_dst->data, IntImm32(cb_id), match.dst_offset, num_elements_expr,
           tiled_republish_row_width, pack_thread_direct_fill_value}));
      last_fragment_fill_value_by_buffer_identity_.erase(BufferIdentityName(match.src));
      last_fragment_fill_value_by_data_.erase(BufferDataIdentity(match.src));
    } else {
      stmts.push_back(
          MakeBlackholeCall(tir::builtin::blackhole_tilize_cast_fragment_slice(),
                            {physical_dst->data, physical_src->data, IntImm32(cb_id),
                             match.dst_offset, match.src_offset, num_elements_expr,
                             tiled_republish_row_width}));
    }
    RecordTiledCBLiveFormAliases(match.dst, cb_id);
  } else if (publish_result) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
    stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cast_fragment_slice(),
                                      {physical_dst->data, physical_src->data, match.dst_offset,
                                       match.src_offset, num_elements_expr}));
    stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_tilize_local_fragment_slice(),
                                      {physical_dst->data, IntImm32(cb_id), match.dst_offset,
                                       num_elements_expr, match.row_width, match.dst_offset}));
    RecordTiledCBLiveFormAliases(match.dst, cb_id);
  } else {
    stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cast_fragment_slice(),
                                      {physical_dst->data, physical_src->data, match.dst_offset,
                                       match.src_offset, num_elements_expr}));
  }

  if (publish_result) {
    stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cb_push_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
  }

  return MaybeWrapComputeSegment(stmts.size() == 1 ? stmts.front() : SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::MatchDirectLocalToCBSliceLoop(const ForNode* op,
                                                      LocalToCBSliceMatch* match) const {
  if (!op || !match) {
    return false;
  }

  auto unwrap_stmt = [&](const Stmt& stmt, bool* wrapped_src_allocation) -> Stmt {
    Stmt current = stmt;
    while (true) {
      if (const auto* attr = current.as<AttrStmtNode>()) {
        current = attr->body;
        continue;
      }
      if (const auto* alloc = current.as<tir::AllocateNode>()) {
        *wrapped_src_allocation = true;
        current = alloc->body;
        continue;
      }
      if (const auto* decl = current.as<tir::DeclBufferNode>()) {
        *wrapped_src_allocation = true;
        current = decl->body;
        continue;
      }
      break;
    }
    return current;
  };

  const auto build_match = [&](const BufferStoreNode* store, const Var& vector_var,
                               const PrimExpr& vector_extent, bool wrap_src_allocation,
                               const std::vector<Stmt>& prefix_stmts) -> bool {
    if (!store || !IsCopyOperation(store) || GetCopyDirection(store) != CopyDirection::kLocalToCB) {
      return false;
    }

    const auto* load = store->value.as<BufferLoadNode>();
    if (!load || load->indices.size() != 1 || store->indices.size() != 2) {
      return false;
    }
    if (!load->indices[0].same_as(vector_var)) {
      return false;
    }
    if (!IsVectorLocalFragmentBuffer(load->buffer)) {
      return false;
    }

    Analyzer analyzer;
    PrimExpr row_extent = store->buffer->shape[1];
    PrimExpr dst_linear =
        analyzer.Simplify(store->indices[0] * row_extent + store->indices[1]);
    Map<Var, PrimExpr> slice_subst;
    slice_subst.Set(vector_var, IntImm(vector_var.dtype(), 0));
    PrimExpr base_offset = analyzer.Simplify(tir::Substitute(dst_linear - vector_var, slice_subst));
    if (ExprUsesVar(base_offset, vector_var)) {
      return false;
    }

    std::vector<Stmt> rewritten = prefix_stmts;
    match->dst = store->buffer;
    match->src = load->buffer;
    match->cast_src = Buffer();
    for (const Stmt& prefix_stmt : prefix_stmts) {
      const ForNode* cast_loop = AsUnwrappedFor(prefix_stmt);
      FragmentCastMatch cast_match;
      if (cast_loop && MatchDirectFragmentCast(cast_loop, &cast_match) &&
          SameBufferIdentity(cast_match.dst, match->src)) {
        match->cast_src = cast_match.src;
        break;
      }
    }
    match->dst_offset_elements = base_offset;
    match->num_elements = vector_extent;
    match->row_width = row_extent;
    match->wrap_src_allocation = wrap_src_allocation;
    if (rewritten.empty()) {
      match->lowered_loop_body = Stmt();
    } else {
      match->lowered_loop_body =
          rewritten.size() == 1 ? rewritten.front() : SeqStmt::Flatten(rewritten);
    }
    return true;
  };

  bool direct_wrapped_src_allocation = false;
  Stmt unwrapped_body = unwrap_stmt(op->body, &direct_wrapped_src_allocation);
  if (const auto* direct_loop = AsUnwrappedFor(unwrapped_body)) {
    const auto* store = AsUnwrappedBufferStore(direct_loop->body);
    return build_match(store, direct_loop->loop_var, direct_loop->extent,
                       direct_wrapped_src_allocation, {});
  }

  std::vector<Stmt> stmts = FlattenSeqStmtBody(unwrapped_body);
  if (stmts.empty()) {
    return false;
  }
  const auto* inner_loop = AsUnwrappedFor(stmts.back());
  const auto* store = inner_loop ? AsUnwrappedBufferStore(inner_loop->body) : nullptr;
  if (!inner_loop) {
    return false;
  }
  std::vector<Stmt> prefix(stmts.begin(), stmts.end() - 1);
  return build_match(store, inner_loop->loop_var, inner_loop->extent,
                     direct_wrapped_src_allocation, prefix);
}

namespace {

class LocalSliceCastSourceOffsetRewriter : public tir::StmtExprMutator {
 public:
  LocalSliceCastSourceOffsetRewriter(const Var& local_slice_data, const PrimExpr& src_offset)
      : local_slice_data_(local_slice_data), src_offset_(src_offset) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(tir::builtin::blackhole_cast_fragment_slice()) && op->args.size() == 5 &&
        op->args[0].same_as(local_slice_data_)) {
      Array<PrimExpr> args = op->args;
      args.Set(3, src_offset_);
      return Call(op->dtype, op->op, args, op->annotations, op->span);
    }
    return tir::StmtExprMutator::VisitExpr_(op);
  }

  Var local_slice_data_;
  PrimExpr src_offset_;
};

}  // namespace

Stmt PlanTTKernelABI::GenerateLocalToCBSliceLoopSequence(const ForNode* op,
                                                           const LocalToCBSliceMatch& match) {
  const int cb_id = AllocateRequirementIndex(match.dst, CBType::kIntermediate);
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const int page_size = std::max(1, cb_requirements_[cb_id].page_size);
  const int64_t total_elements = GetLogicalBufferElementCount(match.dst);
  const int64_t total_bytes = total_elements > 0
                                  ? total_elements * match.dst->dtype.bytes()
                                  : static_cast<int64_t>(page_size);
  const int num_pages = std::max(1, static_cast<int>((total_bytes + page_size - 1) / page_size));

  std::vector<Stmt> stmts;
  ExactTiledCBValue live_source;
  const Buffer live_source_candidate = match.cast_src.defined() ? match.cast_src : match.src;
  auto try_cast_source_live_form = [&]() {
    return TryCreateExactOutputLiveTiledCBValue(live_source_candidate, &live_source) ||
           TryCreateLiveExactTiledCBValue(live_source_candidate, &live_source);
  };
  if (live_source_candidate.defined() && try_cast_source_live_form() &&
      live_source.num_tiles == num_pages) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(live_source.cb_id), IntImm32(live_source.num_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
    MarkExactCBValuesOverlap({live_source.cb_id, cb_id});
    for (int tile = 0; tile < num_pages; ++tile) {
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(), {IntImm32(live_source.cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                        {IntImm32(live_source.cb_id), IntImm32(tile), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(cb_id), IntImm32(tile)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
    }
    if (current_lowering_order_index_ >= 0) {
      const FutureBufferUses future_uses = ClassifyFutureLiveCBReadsBeforeNextWrite(
          live_source_candidate, current_lowering_order_index_);
      if (!future_uses.has_compute_consume && !future_uses.has_transport_consume &&
          !future_uses.has_reference) {
        stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                          {IntImm32(live_source.cb_id),
                                           IntImm32(live_source.num_tiles)}));
        ClearTiledCBLiveFormAliases(live_source_candidate);
      }
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
    return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(cb_id), IntImm32(num_pages)}));
  std::vector<Stmt> loop_stmts;
  if (match.lowered_loop_body.defined()) {
    Stmt lowered_prefix = VisitStmt(match.lowered_loop_body);
    lowered_prefix =
        LocalSliceCastSourceOffsetRewriter(match.src->data, match.dst_offset_elements)(lowered_prefix);
    loop_stmts.push_back(lowered_prefix);
  }
  loop_stmts.push_back(
      MakeBlackholeCall(blackhole_tilize_local_fragment_slice(),
                        {match.src->data, IntImm32(cb_id), match.dst_offset_elements,
                         match.num_elements, match.row_width}));
  Stmt loop_body =
      loop_stmts.size() == 1 ? loop_stmts.front() : SeqStmt::Flatten(loop_stmts);
  if (match.wrap_src_allocation) {
    loop_body = tir::DeclBuffer(match.src, loop_body);
    loop_body =
        tir::Allocate(match.src->data, match.src->dtype, match.src->shape, Bool(1), loop_body);
  }
  stmts.push_back(For(op->loop_var,
                      op->min,
                      op->extent,
                      op->kind,
                      loop_body,
                      op->thread_binding,
                      op->annotations));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(cb_id), IntImm32(num_pages)}));
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}


}  // namespace tl
}  // namespace tvm
