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
 * \file lower_blackhole_tile_compute.cc
 * \brief Explicit Blackhole tile compute selection and leaf TT-Metal sequences.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_tile_compute_dag.h"
#include "common/blackhole_utils.h"
#include "runtime/thread_storage_scope.h"

#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::Buffer;
using tir::Call;
using tir::CallNode;
using tir::Evaluate;
using tir::FloorDiv;
using tir::FloorMod;
using tir::For;
using tir::ForKind;
using tir::SeqStmt;
using tir::Stmt;
using tir::Var;
using tir::builtin::blackhole_add_bcast_cols_init_short;
using tir::builtin::blackhole_add_tiles;
using tir::builtin::blackhole_add_tiles_bcast_cols;
using tir::builtin::blackhole_add_tiles_init;
using tir::builtin::blackhole_binary_op_init_common;
using tir::builtin::blackhole_binary_max_tile;
using tir::builtin::blackhole_binary_max_tile_init;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_copy_tile_to_dst_init_short;
using tir::builtin::blackhole_exp2_tile;
using tir::builtin::blackhole_exp2_tile_init;
using tir::builtin::blackhole_fill_fragment;
using tir::builtin::blackhole_mul_bcast_cols_init_short;
using tir::builtin::blackhole_mul_tiles;
using tir::builtin::blackhole_mul_tiles_bcast_cols;
using tir::builtin::blackhole_mul_tiles_init;
using tir::builtin::blackhole_pack_reconfig_data_format;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_recip_tile;
using tir::builtin::blackhole_recip_tile_init;
using tir::builtin::blackhole_reconfig_data_format;
using tir::builtin::blackhole_reduce_init;
using tir::builtin::blackhole_reduce_tile;
using tir::builtin::blackhole_reduce_uninit;
using tir::builtin::blackhole_sub_tiles;
using tir::builtin::blackhole_sub_tiles_init;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_unary_op_init_common;
using tvm::Integer;
using tvm::ffi::String;

namespace {

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

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

bool IsFloatImmValue(const PrimExpr& expr, double expected) {
  if (const auto* imm = expr.as<FloatImmNode>()) {
    return imm->value == expected;
  }
  if (const auto* cast = expr.as<CastNode>()) {
    return IsFloatImmValue(cast->value, expected);
  }
  return false;
}

bool IsInfinityExpr(const PrimExpr& expr) {
  const auto* call = expr.as<CallNode>();
  if (!call || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  Op op = Downcast<Op>(call->op);
  return op->name == "tir.infinity" || op->name == "tl.infinity";
}

bool IsZeroValue(const PrimExpr& expr) {
  return tir::is_zero(expr) || IsFloatImmValue(expr, 0.0);
}

bool IsNegInfValue(const PrimExpr& expr) {
  if (const auto* cast = expr.as<CastNode>()) {
    return IsNegInfValue(cast->value);
  }
  if (const auto* mul = expr.as<MulNode>()) {
    return ((IsFloatImmValue(mul->a, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->a)) &&
            IsFloatImmValue(mul->b, -1.0)) ||
           ((IsFloatImmValue(mul->b, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->b)) &&
            IsFloatImmValue(mul->a, -1.0));
  }
  return IsFloatImmValue(expr, -std::numeric_limits<double>::infinity());
}

PrimExpr ExactCBReadTileIndex(int num_tiles, const PrimExpr& output_tile) {
  if (num_tiles <= 1) {
    return IntImm32(0);
  }
  return FloorMod(output_tile, IntImm32(num_tiles));
}

std::string GetStorageScope(const Buffer& buffer) {
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
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

class ExactTileComputeEmitter {
 public:
  explicit ExactTileComputeEmitter(std::vector<Stmt>* stmts) : stmts_(stmts) {
    ICHECK(stmts_ != nullptr);
  }

  void Append(const Op& op, const std::vector<PrimExpr>& args) {
    stmts_->push_back(MakeBlackholeCall(op, args));
  }

  void Reserve(int cb_id, int num_tiles) {
    Append(blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void Wait(int cb_id, int num_tiles) {
    Append(blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void Pop(int cb_id, int num_tiles) {
    Append(blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void Push(int cb_id, int num_tiles) {
    Append(blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(num_tiles)});
  }

  void ReconfigDataFormat(int lhs_cb_id, int rhs_cb_id) {
    Append(blackhole_reconfig_data_format(), {IntImm32(lhs_cb_id), IntImm32(rhs_cb_id)});
  }

  void BinaryOpInitCommon(int lhs_cb_id, int rhs_cb_id, int out_cb_id) {
    Append(blackhole_binary_op_init_common(),
           {IntImm32(lhs_cb_id), IntImm32(rhs_cb_id), IntImm32(out_cb_id)});
  }

  void UnaryOpInitCommon(int input_cb_id, int out_cb_id) {
    Append(blackhole_unary_op_init_common(), {IntImm32(input_cb_id), IntImm32(out_cb_id)});
  }

  void PackTile(int out_cb_id, const PrimExpr& out_tile) {
    Append(blackhole_pack_reconfig_data_format(), {IntImm32(out_cb_id)});
    Append(blackhole_pack_tile(), {IntImm32(0), IntImm32(out_cb_id), out_tile});
  }

  void PackTile(int out_cb_id, int out_tile) {
    PackTile(out_cb_id, IntImm32(out_tile));
  }

  template <typename EmitBody>
  void EmitPackedTile(int out_cb_id, int out_tile, EmitBody emit_body) {
    EmitPackedTile(out_cb_id, IntImm32(out_tile), emit_body);
  }

  template <typename EmitBody>
  void EmitPackedTile(int out_cb_id, const PrimExpr& out_tile, EmitBody emit_body) {
    EmitPackedTileBeforePack(out_cb_id, out_tile, emit_body,
                             [](ExactTileComputeEmitter&) {});
  }

  template <typename EmitBody, typename EmitBeforePack>
  void EmitPackedTileBeforePack(int out_cb_id, int out_tile, EmitBody emit_body,
                                EmitBeforePack emit_before_pack) {
    EmitPackedTileBeforePack(out_cb_id, IntImm32(out_tile), emit_body,
                             emit_before_pack);
  }

  template <typename EmitBody, typename EmitBeforePack>
  void EmitPackedTileBeforePack(int out_cb_id, const PrimExpr& out_tile, EmitBody emit_body,
                                EmitBeforePack emit_before_pack) {
    Append(blackhole_tile_regs_acquire(), {});
    emit_body(*this);
    Append(blackhole_tile_regs_commit(), {});
    Append(blackhole_tile_regs_wait(), {});
    emit_before_pack(*this);
    PackTile(out_cb_id, out_tile);
    Append(blackhole_tile_regs_release(), {});
  }

  template <typename EmitBody>
  void EmitPackedTileLoop(int out_cb_id, int num_tiles,
                          const std::string& loop_var_name, EmitBody emit_body) {
    if (num_tiles <= 1) {
      EmitPackedTile(out_cb_id, 0, [&](ExactTileComputeEmitter& tile_emit) {
        emit_body(tile_emit, IntImm32(0));
      });
      return;
    }

    Var tile(loop_var_name, DataType::Int(32));
    std::vector<Stmt> loop_stmts;
    ExactTileComputeEmitter loop_emit(&loop_stmts);
    loop_emit.EmitPackedTile(out_cb_id, tile, [&](ExactTileComputeEmitter& tile_emit) {
      emit_body(tile_emit, tile);
    });
    stmts_->push_back(For(tile, IntImm32(0), IntImm32(num_tiles), ForKind::kSerial,
                          SeqStmt::Flatten(loop_stmts)));
  }

 private:
  std::vector<Stmt>* stmts_;
};

}  // namespace

bool PlanTTKernelABI::MatchExplicitTileReduce(const CallNode* op,
                                              RowReductionMatch* match) const {
  if (!op || !match || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != "tl.tileop.reduce") {
    return false;
  }
  ICHECK_GE(op->args.size(), 5U)
      << "tl.tileop.reduce must carry src, dst, reduce_type, dim, and clear";

  const tir::BufferRegion logical_src_region = NormalizeToBufferRegion(op->args[0]);
  const tir::BufferRegion logical_dst_region = NormalizeToBufferRegion(op->args[1]);
  const Buffer logical_src = logical_src_region->buffer;
  const Buffer logical_dst = logical_dst_region->buffer;
  const Buffer src = ResolvePhysicalComputeBuffer(logical_src);
  const Buffer dst = ResolvePhysicalComputeBuffer(logical_dst);

  const auto* kind_imm = op->args[2].as<StringImmNode>();
  ICHECK(kind_imm) << "tl.tileop.reduce requires string reduce_type";
  const std::string kind = kind_imm->value;
  ICHECK(kind == "sum" || kind == "max")
      << "Blackhole explicit tile reduce currently supports sum/max only, got "
      << kind;

  const auto* dim_imm = op->args[3].as<IntImmNode>();
  ICHECK(dim_imm) << "tl.tileop.reduce requires static dim";
  auto static_region_shape = [](const tir::BufferRegion& region) {
    std::vector<int64_t> shape;
    for (const Range& range : region->region) {
      const auto* extent = range->extent.as<IntImmNode>();
      if (extent == nullptr || extent->value <= 0) {
        return std::vector<int64_t>{};
      }
      shape.push_back(extent->value);
    }
    return shape;
  };

  std::vector<int64_t> src_shape = static_region_shape(logical_src_region);
  if (src_shape.empty()) {
    src_shape = GetLogicalBufferShape(src);
  }
  ICHECK(!src_shape.empty())
      << "Blackhole explicit tile reduce requires a static logical source shape for "
      << BufferIdentityName(src);
  int64_t dim = dim_imm->value;
  if (dim < 0) {
    dim += static_cast<int64_t>(src_shape.size());
  }
  ICHECK_GE(dim, 0) << "Blackhole explicit tile reduce dim out of range: " << dim_imm->value;
  ICHECK_LT(dim, static_cast<int64_t>(src_shape.size()))
      << "Blackhole explicit tile reduce dim " << dim << " out of rank "
      << src_shape.size();
  ICHECK_EQ(dim, static_cast<int64_t>(src_shape.size() - 1))
      << "Blackhole explicit tile reduce Phase A supports row reductions over "
      << "the innermost logical axis only";

  const auto clear_bool = op->args[4].as<Bool>();
  ICHECK(clear_bool) << "tl.tileop.reduce requires static clear bool";
  const bool clear = clear_bool.value()->value;

  std::vector<int64_t> dst_shape = static_region_shape(logical_dst_region);
  if (dst_shape.empty()) {
    dst_shape = GetLogicalBufferShape(logical_dst);
  }
  const int64_t dst_elements =
      dst_shape.empty() ? GetLogicalBufferElementCount(dst)
                        : ComputeStaticElementCount(dst_shape);
  const Buffer live_form_dst =
      ResolveRowReductionLiveFormDestination(logical_dst, dst_elements);
  match->src = src;
  match->dst = dst;
  match->live_form_dst = live_form_dst.defined() ? live_form_dst : logical_dst;
  match->num_elements = IntImm32(static_cast<int>(std::max<int64_t>(1, dst_elements)));
  match->row_width = IntImm32(static_cast<int>(std::max<int64_t>(1, src_shape[dim])));
  match->kind = kind;
  match->grouped = dst_elements > 1;
  match->clear = clear;
  match->accumulate_existing = !clear;
  return true;
}

Buffer PlanTTKernelABI::ResolveRowReductionLiveFormDestination(
    const Buffer& reduce_dst, int64_t reduce_dst_elements) const {
  if (!reduce_dst.defined() || reduce_dst_elements <= 0) {
    return Buffer();
  }

  auto is_state_scope = [](const std::string& scope) {
    if (scope == "local" || scope == "local.fragment") {
      return true;
    }
    const auto parsed = runtime::StorageScope::Create(scope);
    return parsed.rank == runtime::StorageRank::kBlackholeAccumulator;
  };
  auto is_dram_scope = [](const std::string& scope) {
    return scope.empty() || scope == "global";
  };
  auto same_logical_shape = [&](const Buffer& candidate) {
    const std::vector<int64_t> reduce_shape = GetLogicalBufferShape(reduce_dst);
    const std::vector<int64_t> candidate_shape = GetLogicalBufferShape(candidate);
    if (!reduce_shape.empty() && !candidate_shape.empty()) {
      return reduce_shape == candidate_shape;
    }
    return GetLogicalBufferElementCount(candidate) == reduce_dst_elements;
  };

  const std::string reduce_dst_identity = BufferIdentityName(reduce_dst);
  std::vector<Buffer> candidates;
  std::unordered_set<std::string> seen_candidates;
  for (const auto& [dram_identity, source_identity] : direct_copy_source_by_buffer_identity_) {
    if (source_identity.empty()) {
      continue;
    }
    auto source_it = buffer_by_identity_.find(source_identity);
    if (source_it == buffer_by_identity_.end() || !source_it->second.defined()) {
      continue;
    }
    auto dram_it = buffer_by_identity_.find(dram_identity);
    if (dram_it != buffer_by_identity_.end() && dram_it->second.defined() &&
        !is_dram_scope(GetStorageScope(dram_it->second))) {
      continue;
    }

    const Buffer& candidate = source_it->second;
    if (!is_state_scope(GetStorageScope(candidate)) || candidate->dtype != reduce_dst->dtype) {
      continue;
    }
    if (GetLogicalBufferElementCount(candidate) != reduce_dst_elements ||
        !same_logical_shape(candidate)) {
      continue;
    }

    const std::string candidate_identity = BufferIdentityName(candidate);
    if (candidate_identity.empty()) {
      continue;
    }
    if (!reduce_dst_identity.empty() && candidate_identity == reduce_dst_identity) {
      return candidate;
    }
    if (seen_candidates.insert(candidate_identity).second) {
      candidates.push_back(candidate);
    }
  }

  if (candidates.size() == 1U) {
    return candidates.front();
  }
  return Buffer();
}

bool PlanTTKernelABI::MatchExplicitTileTypecast(const CallNode* op,
                                                FragmentCastMatch* match) const {
  if (!op || !match || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName || op->args.size() < 4U) {
    return false;
  }
  const auto* operation_imm = op->args[0].as<StringImmNode>();
  if (!operation_imm ||
      operation_imm->value != blackhole_tile_compute_schema::kTypecastTile ||
      !IsBufferLikeExpr(op->args[1]) || !IsBufferLikeExpr(op->args[2])) {
    return false;
  }
  match->src = ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[1])->buffer);
  match->dst = NormalizeToBufferRegion(op->args[2])->buffer;
  match->src_offset = IntImm(DataType::Int(32), 0);
  match->dst_offset = IntImm(DataType::Int(32), 0);
  match->num_elements = op->args[3];
  match->row_width = PrimExpr();
  const auto [logical_src_rows, logical_src_cols] = GetLogicalMatrixShape(match->src);
  if (logical_src_rows > 0 && logical_src_cols > 0) {
    match->row_width = IntImm(DataType::Int(32), static_cast<int>(logical_src_cols));
    match->num_elements =
        IntImm(DataType::Int(32), static_cast<int>(logical_src_rows * logical_src_cols));
  }
  return true;
}

namespace {

bool IsTileComputeDAGOutputRoleForLowering(const std::string& role) {
  return role == "output" || role == "c";
}

}  // namespace

void PlanTTKernelABI::LoadTileComputeDAGLoweringPlan(const PrimFunc& func) {
  std::unordered_set<std::string> seeded_tile_compute_input_buffers =
      std::move(tile_compute_input_buffers_);
  tile_compute_dag_lowering_decisions_.clear();
  tile_compute_dag_lowering_decision_consumed_.clear();
  active_tile_compute_dag_lowering_decision_.reset();
  tile_compute_input_buffers_.clear();

  const BlackholeTileComputeDAG dag = BuildBlackholeTileComputeDAG(func);
  std::unordered_map<int64_t, std::string> dag_op_name_by_node;
  for (const BlackholeTileComputeDAGNode& node : dag.nodes) {
    dag_op_name_by_node[node.id] = node.op_name;
  }
  tile_compute_input_buffers_ = std::move(seeded_tile_compute_input_buffers);
  if (!dag.edges.empty()) {
    for (const BlackholeTileComputeDAGEdge& edge : dag.edges) {
      if (IsTileComputeDAGOutputRoleForLowering(edge.value_role)) {
        continue;
      }
      const std::string buffer_prefix = "buffer:";
      if (edge.value_key.rfind(buffer_prefix, 0) == 0) {
        const std::string buffer_name = edge.value_key.substr(buffer_prefix.size());
        tile_compute_input_buffers_.insert(buffer_name);
        auto node_it = dag_op_name_by_node.find(edge.consumer_node);
        if (edge.value_role == "rhs" && node_it != dag_op_name_by_node.end() &&
            node_it->second.find("_bcast_cols") != std::string::npos) {
          broadcast_cols_rhs_buffers_.insert(buffer_name);
        }
      }
    }
  }
  std::unordered_map<int64_t, std::vector<const BlackholeTileComputeDAGEdge*>>
      uses_by_producer;
  for (const BlackholeTileComputeDAGEdge& edge : dag.edges) {
    if (edge.producer_node < 0 || IsTileComputeDAGOutputRoleForLowering(edge.value_role)) {
      continue;
    }
    uses_by_producer[edge.producer_node].push_back(&edge);
  }

  for (const BlackholeTileComputeDAGNode& node : dag.nodes) {
    BlackholeTileComputeCoveringDecision covering =
        SelectBlackholeTileComputeCovering(node.op_name);
    ICHECK(covering.selected)
        << "TileCompute DAG lower plan rejected operation " << node.op_name
        << ": " << covering.reject_reason;
    TileComputeDAGLoweringDecision decision;
    decision.node_id = node.id;
    decision.operation_name = covering.operation_name;
    decision.covering = std::move(covering);
    auto uses_it = uses_by_producer.find(node.id);
    const int64_t use_count =
        uses_it == uses_by_producer.end()
            ? 0
            : static_cast<int64_t>(uses_it->second.size());
    decision.fanout_use_count = use_count;
    if (use_count < 2) {
      decision.fanout_policy = "single_use";
    } else if (node.side_effect_class == "tile_regs" ||
               node.side_effect_class == "dst" ||
               node.side_effect_class == "pack") {
      decision.fanout_policy = "materialize_before_cross_event_use";
    } else {
      decision.fanout_policy = "share_live_value";
    }
    tile_compute_dag_lowering_decisions_.push_back(std::move(decision));
  }
  tile_compute_dag_lowering_decision_consumed_.assign(
      tile_compute_dag_lowering_decisions_.size(), false);
}

BlackholeTileComputeCoveringDecision
PlanTTKernelABI::ConsumeTileComputeDAGLoweringDecision(
    const std::string& operation_name) {
  ICHECK_EQ(tile_compute_dag_lowering_decision_consumed_.size(),
            tile_compute_dag_lowering_decisions_.size())
      << "TileCompute DAG lower plan consumption state is out of sync";
  for (size_t i = 0; i < tile_compute_dag_lowering_decisions_.size(); ++i) {
    if (tile_compute_dag_lowering_decision_consumed_[i]) {
      continue;
    }
    const TileComputeDAGLoweringDecision& decision =
        tile_compute_dag_lowering_decisions_[i];
    if (decision.operation_name != operation_name) {
      continue;
    }
    tile_compute_dag_lowering_decision_consumed_[i] = true;
    active_tile_compute_dag_lowering_decision_ = decision;
    return decision.covering;
  }
  ICHECK(false) << "TileCompute DAG lower plan has no selected decision for operation "
                << operation_name;
  return BlackholeTileComputeCoveringDecision();
}

int64_t PlanTTKernelABI::CurrentTileComputeDAGNodeId() const {
  return active_tile_compute_dag_lowering_decision_
             ? active_tile_compute_dag_lowering_decision_->node_id
             : -1;
}

String PlanTTKernelABI::CurrentTileComputeDAGSourceEmitter() const {
  if (!active_tile_compute_dag_lowering_decision_ ||
      !active_tile_compute_dag_lowering_decision_->covering.source_emitter) {
    return String();
  }
  return String(ToString(
      *active_tile_compute_dag_lowering_decision_->covering.source_emitter));
}

String PlanTTKernelABI::CurrentTileComputeDAGMaterializationPolicy() const {
  return active_tile_compute_dag_lowering_decision_
             ? String(active_tile_compute_dag_lowering_decision_->covering
                          .materialization_policy)
             : String();
}

int64_t PlanTTKernelABI::CurrentTileComputeDAGFanoutUseCount() const {
  return active_tile_compute_dag_lowering_decision_
             ? active_tile_compute_dag_lowering_decision_->fanout_use_count
             : 0;
}

String PlanTTKernelABI::CurrentTileComputeDAGFanoutPolicy() const {
  return active_tile_compute_dag_lowering_decision_
             ? String(active_tile_compute_dag_lowering_decision_->fanout_policy)
             : String();
}

class BlackholeTileComputeSourceProjection {
 public:
  static Stmt Emit(PlanTTKernelABI* abi, const CallNode* op,
                   const BlackholeTileComputeCoveringDecision& covering);

 private:
  static Op RequiredSourceBuiltin(const BlackholeTileComputePattern& pattern,
                                  const char* field_name,
                                  const char* op_name);
  static Stmt EmitCustom(PlanTTKernelABI* abi, const CallNode* op,
                         const BlackholeTileComputeCoveringDecision& covering,
                         const BlackholeTileComputePattern& pattern);
  static Stmt EmitFillFragment(PlanTTKernelABI* abi, const CallNode* op,
                               const BlackholeTileComputeCoveringDecision& covering,
                               const BlackholeTileComputePattern& pattern);
  static Stmt EmitCopy(PlanTTKernelABI* abi, const CallNode* op,
                       const BlackholeTileComputeCoveringDecision& covering,
                       const BlackholeTileComputePattern& pattern);
  static Stmt EmitTypecast(PlanTTKernelABI* abi, const CallNode* op,
                           const BlackholeTileComputeCoveringDecision& covering,
                           const BlackholeTileComputePattern& pattern);
  static Stmt EmitBinaryMax(PlanTTKernelABI* abi, const CallNode* op,
                            const BlackholeTileComputeCoveringDecision& covering,
                            const BlackholeTileComputePattern& pattern);
  static Stmt EmitBinary(PlanTTKernelABI* abi, const CallNode* op,
                         const BlackholeTileComputeCoveringDecision& covering,
                         const BlackholeTileComputePattern& pattern);
  static Stmt EmitBroadcastColsBinary(
      PlanTTKernelABI* abi, const CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering,
      const BlackholeTileComputePattern& pattern);
  static Stmt EmitUnary(PlanTTKernelABI* abi, const CallNode* op,
                        const BlackholeTileComputeCoveringDecision& covering,
                        const BlackholeTileComputePattern& pattern);
  static Stmt EmitReduce(PlanTTKernelABI* abi, const CallNode* op,
                         const BlackholeTileComputeCoveringDecision& covering,
                         const BlackholeTileComputePattern& pattern);
};

size_t FindBlackholeTileComputeBufferArgIndex(
    BlackholeTileComputeOperandRole role,
    const BlackholeTileComputeCoveringDecision& covering) {
  const BlackholeTileComputePattern* pattern =
      FindBlackholeTileComputePattern(covering.operation_name);
  ICHECK(pattern != nullptr)
      << "Selected Blackhole tile compute covering references unknown operation "
      << covering.operation_name;
  for (const BlackholeTileComputeCallOperand& operand :
       pattern->blackhole_compute_operands) {
    if (operand.role == role) {
      return operand.arg_index;
    }
  }
  ICHECK(false) << "Selected Blackhole tile compute pattern "
                << covering.pattern_name << " has no explicit source argument for role "
                << ToString(role);
  return 0;
}

Buffer PlanTTKernelABI::GetBlackholeTileComputeBufferArg(
    const CallNode* op, BlackholeTileComputeOperandRole role,
    const BlackholeTileComputeCoveringDecision& covering) const {
  ICHECK(op != nullptr);
  const size_t index = FindBlackholeTileComputeBufferArgIndex(role, covering);
  ICHECK_LT(index, op->args.size())
      << "tl.tileop.blackhole_compute missing buffer argument for role "
      << ToString(role)
      << " for selected pattern " << covering.pattern_name
      << " with emitter "
      << (covering.source_emitter ? ToString(*covering.source_emitter) : "");
  return ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[index])->buffer);
}

PrimExpr PlanTTKernelABI::GetBlackholeTileComputePrimArg(
    const CallNode* op, size_t index,
    const BlackholeTileComputeCoveringDecision& covering) const {
  ICHECK(op != nullptr);
  ICHECK_LT(index, op->args.size())
      << "tl.tileop.blackhole_compute missing scalar argument " << index
      << " for selected pattern " << covering.pattern_name
      << " with emitter "
      << (covering.source_emitter ? ToString(*covering.source_emitter) : "");
  return op->args[index];
}

Stmt PlanTTKernelABI::LowerExplicitTileComputeCall(const CallNode* op) {
  if (!op || !op->op->IsInstance<OpNode>()) {
    return Stmt();
  }
  const Op call_op = Downcast<Op>(op->op);
  std::string operation;
  if (call_op->name == blackhole_tile_compute_schema::kOpName) {
    if (op->args.empty()) {
      return Stmt();
    }
    const auto* operation_imm = op->args[0].as<StringImmNode>();
    ICHECK(operation_imm) << "tl.tileop.blackhole_compute requires string operation name";
    operation = operation_imm->value;
  } else {
    RowReductionMatch reduce_match;
    if (!MatchExplicitTileReduce(op, &reduce_match)) {
      return Stmt();
    }
    operation = "reduce_tile";
  }
  const BlackholeTileComputeCoveringDecision covering =
      ConsumeTileComputeDAGLoweringDecision(operation);
  ICHECK(covering.selected)
      << "TileCompute covering rejected operation " << operation
      << ": " << covering.reject_reason;
  requires_compute_segment_ = true;
  Stmt lowered = BlackholeTileComputeSourceProjection::Emit(this, op, covering);
  active_tile_compute_dag_lowering_decision_.reset();
  return lowered;
}

Stmt BlackholeTileComputeSourceProjection::Emit(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering) {
  ICHECK(abi != nullptr);
  ICHECK(op != nullptr);
  ICHECK(covering.source_emitter)
      << "Selected Blackhole tile compute pattern " << covering.pattern_name
      << " for operation " << covering.operation_name
      << " is not admitted on the explicit tile-compute source path";
  const BlackholeTileComputePattern* pattern =
      FindBlackholeTileComputePattern(covering.operation_name);
  ICHECK(pattern != nullptr)
      << "No explicit pattern registered for selected Blackhole tile compute operation "
      << covering.operation_name;
  ICHECK(pattern->source_emitter && *pattern->source_emitter == *covering.source_emitter)
      << "Selected Blackhole tile compute pattern " << covering.pattern_name
      << " has mismatched source emitter "
      << ToString(*covering.source_emitter);
  switch (pattern->source_emitter_category) {
    case BlackholeTileComputeSourceEmitterCategory::kBinary:
      return EmitBinary(abi, op, covering, *pattern);
    case BlackholeTileComputeSourceEmitterCategory::kBroadcastColsBinary:
      return EmitBroadcastColsBinary(abi, op, covering, *pattern);
    case BlackholeTileComputeSourceEmitterCategory::kUnary:
      return EmitUnary(abi, op, covering, *pattern);
    case BlackholeTileComputeSourceEmitterCategory::kCustom:
      return EmitCustom(abi, op, covering, *pattern);
    case BlackholeTileComputeSourceEmitterCategory::kNone:
      break;
  }
  ICHECK(false) << "Selected Blackhole tile compute pattern "
                << covering.pattern_name
                << " has no source projection category";
  return Stmt();
}

Op BlackholeTileComputeSourceProjection::RequiredSourceBuiltin(
    const BlackholeTileComputePattern& pattern, const char* field_name,
    const char* op_name) {
  ICHECK(op_name != nullptr && std::string(op_name).size() > 0U)
      << "Blackhole tile compute pattern " << pattern.name
      << " is missing " << field_name
      << " for source projection category "
      << ToString(pattern.source_emitter_category);
  return Op::Get(op_name);
}

Stmt BlackholeTileComputeSourceProjection::EmitCustom(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  ICHECK(pattern.source_emitter)
      << "Custom Blackhole tile compute source projection requires source emitter";
  switch (*pattern.source_emitter) {
    case BlackholeTileComputeSourceEmitterKind::kFillFragment:
      return EmitFillFragment(abi, op, covering, pattern);
    case BlackholeTileComputeSourceEmitterKind::kCopyTile:
      return EmitCopy(abi, op, covering, pattern);
    case BlackholeTileComputeSourceEmitterKind::kTypecastTile:
      return EmitTypecast(abi, op, covering, pattern);
    case BlackholeTileComputeSourceEmitterKind::kBinaryMaxTile:
      return EmitBinaryMax(abi, op, covering, pattern);
    case BlackholeTileComputeSourceEmitterKind::kReduceTile:
      return EmitReduce(abi, op, covering, pattern);
    case BlackholeTileComputeSourceEmitterKind::kAddTiles:
    case BlackholeTileComputeSourceEmitterKind::kSubTiles:
    case BlackholeTileComputeSourceEmitterKind::kMulTiles:
    case BlackholeTileComputeSourceEmitterKind::kMulTilesBcastCols:
    case BlackholeTileComputeSourceEmitterKind::kAddTilesBcastCols:
    case BlackholeTileComputeSourceEmitterKind::kExp2Tile:
    case BlackholeTileComputeSourceEmitterKind::kRecipTile:
      break;
  }
  ICHECK(false) << "Blackhole tile compute pattern " << pattern.name
                << " uses custom source projection for non-custom emitter "
                << ToString(*pattern.source_emitter);
  return Stmt();
}

Stmt BlackholeTileComputeSourceProjection::EmitFillFragment(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  (void)pattern;
  return abi->GenerateFillTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kOutput,
                                            covering),
      abi->GetBlackholeTileComputePrimArg(op, 2, covering),
      abi->GetBlackholeTileComputePrimArg(op, 3, covering));
}

Stmt BlackholeTileComputeSourceProjection::EmitCopy(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  (void)pattern;
  return abi->GenerateCopyTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kInput,
                                            covering),
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kOutput,
                                            covering),
      abi->GetBlackholeTileComputePrimArg(op, 3, covering));
}

Stmt BlackholeTileComputeSourceProjection::EmitTypecast(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  (void)covering;
  (void)pattern;
  PlanTTKernelABI::FragmentCastMatch match;
  ICHECK(abi->MatchExplicitTileTypecast(op, &match))
      << "tl.tileop.blackhole_compute typecast_tile requires source and destination "
         "regions";
  const int cast_order_index =
      abi->ResolveCurrentBufferTransferOrder(match.src, match.dst,
                                             abi->current_lowering_order_index_);
  const bool publish_cb =
      !abi->select_compute_builtins_only_ &&
      abi->ShouldPublishBufferResult(match.dst, cast_order_index);
  std::vector<Stmt> prefix;
  std::vector<Stmt> suffix;
  if (publish_cb) {
    abi->ValidatePublishedBufferSourceEdge(match.src, match.dst);
    const bool source_has_live_cb =
        abi->buffer_live_form_cb_by_buffer_identity_.count(BufferIdentityName(match.src)) != 0U;
    const bool dst_has_republish_fact =
        abi->FindBufferMaterializationFact(match.dst) != nullptr;
    if (!(source_has_live_cb && dst_has_republish_fact)) {
      abi->AppendPublishedBufferSourceMaterialization(match.src, cast_order_index,
                                                      &prefix, &suffix);
    }
  }
  prefix.push_back(abi->GenerateFragmentCastSequence(match, publish_cb, cast_order_index));
  prefix.insert(prefix.end(), suffix.begin(), suffix.end());
  return SeqStmt::Flatten(prefix);
}

Stmt BlackholeTileComputeSourceProjection::EmitBinaryMax(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  (void)pattern;
  return abi->GenerateBinaryMaxTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kLhs,
                                            covering),
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kRhs,
                                            covering));
}

Stmt BlackholeTileComputeSourceProjection::EmitBinary(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  return abi->GenerateBinaryTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kLhs,
                                            covering),
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kRhs,
                                            covering),
      covering.operation_name,
      RequiredSourceBuiltin(pattern, "source_init_builtin",
                            pattern.source_init_builtin),
      RequiredSourceBuiltin(pattern, "source_tile_builtin",
                            pattern.source_tile_builtin));
}

Stmt BlackholeTileComputeSourceProjection::EmitBroadcastColsBinary(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  return abi->GenerateBroadcastColsBinaryTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kLhs,
                                            covering),
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kRhs,
                                            covering),
      covering.operation_name,
      RequiredSourceBuiltin(pattern, "source_init_builtin",
                            pattern.source_init_builtin),
      RequiredSourceBuiltin(pattern, "source_tile_builtin",
                            pattern.source_tile_builtin),
      abi->GetBlackholeTileComputePrimArg(op, 3, covering),
      abi->GetBlackholeTileComputePrimArg(op, 4, covering));
}

Stmt BlackholeTileComputeSourceProjection::EmitUnary(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  return abi->GenerateUnaryTileSequence(
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kInput,
                                            covering),
      abi->GetBlackholeTileComputeBufferArg(op, BlackholeTileComputeOperandRole::kOutput,
                                            covering),
      covering.operation_name,
      RequiredSourceBuiltin(pattern, "source_init_builtin",
                            pattern.source_init_builtin),
      RequiredSourceBuiltin(pattern, "source_tile_builtin",
                            pattern.source_tile_builtin),
      abi->GetBlackholeTileComputePrimArg(op, 3, covering));
}

Stmt BlackholeTileComputeSourceProjection::EmitReduce(
    PlanTTKernelABI* abi, const CallNode* op,
    const BlackholeTileComputeCoveringDecision& covering,
    const BlackholeTileComputePattern& pattern) {
  (void)covering;
  (void)pattern;
  PlanTTKernelABI::RowReductionMatch match;
  ICHECK(abi->MatchExplicitTileReduce(op, &match))
      << "Selected reduce_tile source emitter requires tl.tileop.reduce";
  return abi->GenerateRowReductionSequence(match);
}

Stmt PlanTTKernelABI::GenerateRowReductionSequence(const RowReductionMatch& match) {
  PrimExpr accumulator_fill_value;
  const bool accumulator_is_identity_fill =
      match.accumulate_existing && TryGetLastFragmentFillValue(match.dst, &accumulator_fill_value) &&
      ((match.kind == "sum" && IsZeroValue(accumulator_fill_value)) ||
       (match.kind == "max" && IsNegInfValue(accumulator_fill_value)));
  const bool accumulate_existing = match.accumulate_existing && !accumulator_is_identity_fill;
  ExactTiledCBValue src_in = CreateRowReductionInputCBValue(match.src);
  ExactTiledCBValue out;
  ExactTiledCBValue reduced;
  if (accumulate_existing) {
    reduced = CreateEmptyExactTiledCBValue(match.dst, "reduce_partial");
    out = reduced.num_tiles == 1
              ? reduced
              : CreateEmptyExactTiledCBValue(
                    match.dst, "reduce_out",
                    ExactOutputCBTypeForBuffer(match.dst, current_lowering_order_index_));
  } else {
    out = CreateEmptyExactTiledCBValue(
        match.dst, "reduce_out",
        ExactOutputCBTypeForBuffer(match.dst, current_lowering_order_index_));
    reduced = out;
  }
  auto constrain_reduce_output_shape = [&](ExactTiledCBValue* value) {
    ICHECK(value != nullptr);
    const auto* num_elements_imm = match.num_elements.as<IntImmNode>();
    if (num_elements_imm == nullptr || num_elements_imm->value <= 0) {
      return;
    }
    constexpr int64_t kTileElements = kBlackholeTileRows * kBlackholeTileCols;
    const int64_t logical_elements = num_elements_imm->value;
    const int logical_tiles = std::max(1, CeilDivToInt(logical_elements, kTileElements));
    value->num_elements = logical_elements;
    value->num_tiles = logical_tiles;
    value->row_width = 1;
    if (value->cb_id >= 0) {
      ICHECK_LT(value->cb_id, static_cast<int>(cb_requirements_.size()));
      const DataType storage_dtype =
          value->buffer.defined() ? ExactTiledCBStorageDType(value->buffer->dtype)
                                  : DataType::BFloat(16);
      SetRequirementPageLayout(
          value->cb_id,
          kBlackholeTileRows * kBlackholeTileCols * storage_dtype.bytes(),
          logical_tiles);
      auto& req = cb_requirements_.at(value->cb_id);
      req.publish_pages_per_event = logical_tiles;
      req.consume_pages_per_event = logical_tiles;
    }
  };
  constrain_reduce_output_shape(&reduced);
  if (out.cb_id != reduced.cb_id) {
    constrain_reduce_output_shape(&out);
  } else {
    out = reduced;
  }
  const bool reuse_reduced_as_output =
      accumulate_existing && out.cb_id == reduced.cb_id;
  ExactTiledCBValue dst_in;
  if (accumulate_existing) {
    dst_in = CreateExactInputCBValue(match.dst, "reduce_accum");
    ICHECK_EQ(dst_in.num_tiles, reduced.num_tiles)
        << "Blackhole accumulating row reduction requires accumulator and reduction tile counts "
           "to match";
    ICHECK_EQ(out.num_tiles, reduced.num_tiles)
        << "Blackhole accumulating row reduction requires output and reduction tile counts to "
           "match";
  }
  ExactTiledCBValue scaler = CreateReduceScalerExactTiledCBValue();
  RecordExactComputeOpPlan("reduce", "reduce_tile",
                           {{"input", match.src, "identity"},
                            {"scaler", scaler.buffer, "identity"},
                            {"output", accumulate_existing ? reduced.buffer : match.dst, "identity"}});
  if (accumulate_existing) {
    RecordExactComputeOpPlan("binary", match.kind == "sum" ? "add_tiles" : "binary_max_tile",
                             {{"lhs", match.dst, "identity"},
                              {"rhs", reduced.buffer, "identity"},
                              {"output", match.dst, "identity"}});
  }

  const Buffer scaler_local = scaler.buffer;
  const Stmt scaler_publish =
      PublishConstantToExactTiledCB(scaler_local, make_const(match.src->dtype, 1.0), scaler);

  const int tiles_per_reduction = std::max(1, src_in.num_tiles / std::max(1, out.num_tiles));
  std::vector<Stmt> stmts;
  if (Stmt publish_src = PublishExactInputToTiledCB(match.src, &src_in); publish_src.defined()) {
    stmts.push_back(publish_src);
  }
  if (accumulate_existing) {
    if (Stmt publish_dst = PublishExactInputToTiledCB(match.dst, &dst_in);
        publish_dst.defined()) {
      stmts.push_back(publish_dst);
    }
  }
  stmts.push_back(MakeBlackholeCall(blackhole_binary_op_init_common(),
                                    {IntImm32(src_in.cb_id), IntImm32(src_in.cb_id),
                                     IntImm32(scaler.cb_id)}));
  stmts.push_back(scaler_publish);
  if (accumulate_existing) {
    MarkExactCBValuesOverlap({src_in.cb_id, dst_in.cb_id, scaler.cb_id, reduced.cb_id,
                              out.cb_id});
  } else {
    MarkExactCBValuesOverlap({src_in.cb_id, scaler.cb_id, out.cb_id});
  }
  ExactTileComputeEmitter emit(&stmts);
  emit.Reserve(reduced.cb_id, reduced.num_tiles);
  emit.Wait(src_in.cb_id, src_in.num_tiles);
  emit.Wait(scaler.cb_id, scaler.num_tiles);
  emit.ReconfigDataFormat(src_in.cb_id, scaler.cb_id);
  emit.BinaryOpInitCommon(src_in.cb_id, scaler.cb_id, reduced.cb_id);
  emit.Append(
      blackhole_reduce_init(),
      {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(reduced.cb_id),
       StringImm(match.kind), StringImm("row")});
  for (int out_tile = 0; out_tile < reduced.num_tiles; ++out_tile) {
    emit.Append(blackhole_tile_regs_acquire(), {});
    for (int tile = 0; tile < tiles_per_reduction; ++tile) {
      const int src_tile = out_tile * tiles_per_reduction + tile;
      emit.Append(
          blackhole_reduce_tile(),
          {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(src_tile), IntImm32(0),
           IntImm32(0), StringImm(match.kind), StringImm("row")});
    }
    emit.Append(blackhole_tile_regs_commit(), {});
    emit.Append(blackhole_tile_regs_wait(), {});
    emit.PackTile(reduced.cb_id, out_tile);
    emit.Append(blackhole_tile_regs_release(), {});
  }
  emit.Append(blackhole_reduce_uninit(), {StringImm(match.kind), StringImm("row")});
  if (Stmt release = ReleaseExactInputAfterUse(src_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  emit.Pop(scaler.cb_id, scaler.num_tiles);
  emit.Push(reduced.cb_id, reduced.num_tiles);
  if (accumulate_existing) {
    if (!reuse_reduced_as_output) {
      emit.Reserve(out.cb_id, out.num_tiles);
    }
    emit.Wait(dst_in.cb_id, dst_in.num_tiles);
    emit.Wait(reduced.cb_id, reduced.num_tiles);
    emit.ReconfigDataFormat(dst_in.cb_id, reduced.cb_id);
    if (match.kind == "sum") {
      emit.BinaryOpInitCommon(dst_in.cb_id, reduced.cb_id, out.cb_id);
      emit.Append(blackhole_add_tiles_init(),
                  {IntImm32(dst_in.cb_id), IntImm32(reduced.cb_id)});
    } else {
      emit.UnaryOpInitCommon(dst_in.cb_id, out.cb_id);
    }
    for (int tile = 0; tile < out.num_tiles; ++tile) {
      auto emit_accumulate_body = [&](ExactTileComputeEmitter& tile_emit) {
        if (match.kind == "sum") {
          tile_emit.Append(blackhole_add_tiles(),
                           {IntImm32(dst_in.cb_id), IntImm32(reduced.cb_id),
                            IntImm32(tile), IntImm32(tile), IntImm32(0)});
        } else {
          tile_emit.Append(blackhole_copy_tile_to_dst_init_short(),
                           {IntImm32(dst_in.cb_id)});
          tile_emit.Append(blackhole_copy_tile(),
                           {IntImm32(dst_in.cb_id), IntImm32(tile), IntImm32(0)});
          tile_emit.Append(blackhole_copy_tile_to_dst_init_short(),
                           {IntImm32(reduced.cb_id)});
          tile_emit.Append(blackhole_copy_tile(),
                           {IntImm32(reduced.cb_id), IntImm32(tile), IntImm32(1)});
          tile_emit.Append(blackhole_binary_max_tile_init(), {});
          tile_emit.Append(blackhole_binary_max_tile(),
                           {IntImm32(0), IntImm32(1), IntImm32(0), StringImm("C")});
        }
      };
      if (reuse_reduced_as_output) {
        ICHECK_EQ(out.num_tiles, 1)
            << "Blackhole in-place accumulating row reduction currently requires one output tile";
        emit.EmitPackedTileBeforePack(
            out.cb_id, tile, emit_accumulate_body,
            [&](ExactTileComputeEmitter& tile_emit) {
              tile_emit.Pop(reduced.cb_id, reduced.num_tiles);
              tile_emit.Reserve(out.cb_id, out.num_tiles);
            });
      } else {
        emit.EmitPackedTile(out.cb_id, tile, emit_accumulate_body);
      }
    }
    if (Stmt release = ReleaseExactInputAfterUse(dst_in, current_lowering_order_index_);
        release.defined()) {
      stmts.push_back(release);
    }
    if (!reuse_reduced_as_output) {
      emit.Pop(reduced.cb_id, reduced.num_tiles);
    }
    emit.Push(out.cb_id, out.num_tiles);
  }
  const Buffer live_form_dst = match.live_form_dst.defined() ? match.live_form_dst : match.dst;
  auto future_reference_precedes_live_cb_consume = [&]() {
    int first_reference_order = -1;
    int first_live_cb_consume_order = -1;
    int next_write_order = -1;
    for (const std::string& identity : CollectBufferFlowIdentities(match.dst)) {
      auto it = buffer_flow_facts_.find(identity);
      if (it == buffer_flow_facts_.end()) {
        continue;
      }
      for (const BlackholeBufferFlowEvent& event : it->second.events) {
        if (event.order_index <= current_lowering_order_index_) {
          continue;
        }
        if (event.kind == BlackholeBufferFlowEventKind::kWrite &&
            (next_write_order < 0 || event.order_index < next_write_order)) {
          next_write_order = event.order_index;
        }
      }
    }
    for (const std::string& identity : CollectBufferFlowIdentities(match.dst)) {
      auto it = buffer_flow_facts_.find(identity);
      if (it == buffer_flow_facts_.end()) {
        continue;
      }
      for (const BlackholeBufferFlowEvent& event : it->second.events) {
        if (event.order_index <= current_lowering_order_index_) {
          continue;
        }
        if (next_write_order >= 0 && event.order_index > next_write_order) {
          continue;
        }
        if (event.kind == BlackholeBufferFlowEventKind::kReference &&
            (first_reference_order < 0 || event.order_index < first_reference_order)) {
          first_reference_order = event.order_index;
        }
        if ((event.kind == BlackholeBufferFlowEventKind::kComputeConsume ||
             event.kind == BlackholeBufferFlowEventKind::kTransportConsume) &&
            (first_live_cb_consume_order < 0 ||
             event.order_index < first_live_cb_consume_order)) {
          first_live_cb_consume_order = event.order_index;
        }
      }
    }
    return first_reference_order >= 0 &&
           (first_live_cb_consume_order < 0 ||
            first_reference_order < first_live_cb_consume_order);
  };
  const bool materialize_loop_carried =
      ShouldMaterializeLoopCarriedExactOutput(live_form_dst);
  if (!materialize_loop_carried && future_reference_precedes_live_cb_consume()) {
    stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out, /*pop_front=*/false));
  }
  InvalidateLastFragmentFillValue(match.dst);
  if (live_form_dst.defined() && !SameBufferIdentity(live_form_dst, match.dst)) {
    InvalidateLastFragmentFillValue(live_form_dst);
  }
  if (materialize_loop_carried) {
    if (Stmt materialize = MaterializeLoopCarriedExactOutput(live_form_dst, out);
        materialize.defined()) {
      stmts.push_back(materialize);
    }
  } else {
    RecordExactOutputLiveForm(live_form_dst, out);
  }

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(scaler_local, body);
  body = tir::Allocate(scaler_local->data, scaler_local->dtype, scaler_local->shape, Bool(1), body);
  if (!materialize_loop_carried) {
    body = AttachExactOutputLiveFormMarker(live_form_dst, out, body);
  }
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateFillTileSequence(const Buffer& dst, const PrimExpr& value,
                                               const PrimExpr& requested_num_elements) {
  const FutureBufferUses future_uses =
      ClassifyFutureBufferUses(dst, current_lowering_order_index_);
  if (GetStorageScope(dst) == "local.fragment" &&
      !future_uses.has_compute_consume && !future_uses.has_transport_consume &&
      !future_uses.has_reference && FindBufferMaterializationFact(dst) == nullptr) {
    return Evaluate(IntImm32(0));
  }

  PrimExpr num_elements = requested_num_elements;
  int64_t physical_local_extent = 0;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(dst)) {
    physical_local_extent = ProductIntegerArrayField(*spec, schema_key::kLocalShape, int64_t{0});
  }
  if (physical_local_extent > 0) {
    num_elements = IntImm(DataType::Int(32), static_cast<int>(physical_local_extent));
  }
  const int64_t logical_extent = GetLogicalBufferElementCount(dst);
  if (physical_local_extent <= 0 && logical_extent > 1) {
    bool should_promote_extent = !num_elements.defined();
    if (const auto* int_imm = num_elements.as<IntImmNode>()) {
      should_promote_extent = int_imm->value < logical_extent;
    }
    if (should_promote_extent) {
      num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }

  const bool future_write_before_compute_consume =
      FutureWritePrecedesFutureComputeConsume(dst, current_lowering_order_index_);

  ClearSelectedSourceLiveProducer(dst);
  ClearTiledCBLiveFormAliases(dst);
  InvalidateLastFragmentFillValue(dst);
  if (!future_write_before_compute_consume) {
    for (const std::string& identity : CollectBufferFlowIdentities(dst)) {
      last_fragment_fill_value_by_buffer_identity_[identity] = value;
    }
    if (const VarNode* data = BufferDataIdentity(dst)) {
      last_fragment_fill_value_by_data_[data] = value;
    }
  }
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  if (!future_write_before_compute_consume &&
      physical_dst.defined() && !physical_dst.same_as(dst)) {
    if (const VarNode* data = BufferDataIdentity(physical_dst)) {
      last_fragment_fill_value_by_data_[data] = value;
    }
  }
  RecordExactComputeOpPlan("fill", "fill_tile",
                           {{"output", dst, "identity"}});
  return MaybeWrapComputeSegment(MakeBlackholeCall(
      tir::builtin::blackhole_fill_fragment(),
      {physical_dst.defined() ? physical_dst->data : dst->data, num_elements, value}));
}

Stmt PlanTTKernelABI::GenerateCopyTileSequence(const Buffer& src, const Buffer& dst,
                                               const PrimExpr& num_elements) {
  RecordExactComputeOpPlan("copy", "copy_tile",
                           {{"input", src, "identity"},
                            {"output", dst, "identity"}});
  ExactTiledCBValue live_value;
  const bool force_local_loop_carried =
      (IsActiveLoopCarriedBuffer(src) || IsCompletedLoopCarriedBuffer(src)) &&
      !IsSingleFullTileLogicalMatrix(src);
  if (!force_local_loop_carried &&
      (TryCreateExactOutputLiveTiledCBValue(src, &live_value) ||
       TryCreateLiveExactTiledCBValue(src, &live_value))) {
    RecordTiledCBLiveFormAliases(dst, live_value.cb_id);
    InvalidateLastFragmentFillValue(dst);
    return Evaluate(IntImm32(0));
  }

  PrimExpr fill_value;
  if (!force_local_loop_carried && TryGetLastFragmentFillValue(src, &fill_value)) {
    const std::string dst_name = BufferIdentityName(dst);
    if (!dst_name.empty()) {
      ClearTiledCBLiveFormIdentity(dst_name);
      last_fragment_fill_value_by_buffer_identity_[dst_name] = fill_value;
    }
    if (const VarNode* data = BufferDataIdentity(dst)) {
      last_fragment_fill_value_by_data_[data] = fill_value;
    }
    return Evaluate(IntImm32(0));
  }

  FragmentCastMatch cast_match;
  cast_match.dst = dst;
  cast_match.src = src;
  cast_match.dst_offset = IntImm(DataType::Int(32), 0);
  cast_match.src_offset = IntImm(DataType::Int(32), 0);
  cast_match.num_elements = num_elements;
  if (!cast_match.num_elements.defined()) {
    const int64_t logical_extent = GetLogicalVectorLength(dst);
    if (logical_extent > 1) {
      cast_match.num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }
  return GenerateFragmentCastSequence(
      cast_match, /*publish_cb=*/false, /*current_order_index=*/-1,
      /*allow_force_publish_from_fact=*/!force_local_loop_carried);
}

Stmt PlanTTKernelABI::GenerateBinaryMaxTileSequence(const Buffer& dst, const Buffer& rhs) {
  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, "binary_max_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, "binary_max_rhs");
  ExactTiledCBValue out;
  if (!TryCreateLoopCarriedExactOutputStateCBValue(dst, &out)) {
    out = CreateEmptyExactTiledCBValue(
        dst, "binary_max_out",
        ExactOutputCBTypeForBuffer(dst, current_lowering_order_index_));
  }
  RecordExactComputeOpPlan("binary", "binary_max_tile",
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs, "identity"},
                            {"output", dst, "identity"}});

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.UnaryOpInitCommon(lhs_in.cb_id, out.cb_id);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(lhs_in.cb_id, lhs_in.num_tiles);
  emit.Wait(rhs_in.cb_id, rhs_in.num_tiles);
  emit.ReconfigDataFormat(lhs_in.cb_id, rhs_in.cb_id);
  emit.EmitPackedTileLoop(out.cb_id, out.num_tiles, "tile",
                          [&](ExactTileComputeEmitter& tile_emit,
                              const PrimExpr& tile) {
      const PrimExpr lhs_tile = ExactCBReadTileIndex(lhs_in.num_tiles, tile);
      const PrimExpr rhs_tile = ExactCBReadTileIndex(rhs_in.num_tiles, tile);
      tile_emit.Append(blackhole_copy_tile_to_dst_init_short(), {IntImm32(lhs_in.cb_id)});
      tile_emit.Append(blackhole_copy_tile(),
                       {IntImm32(lhs_in.cb_id), lhs_tile, IntImm32(0)});
      tile_emit.Append(blackhole_copy_tile_to_dst_init_short(), {IntImm32(rhs_in.cb_id)});
      tile_emit.Append(blackhole_copy_tile(),
                       {IntImm32(rhs_in.cb_id), rhs_tile, IntImm32(1)});
      tile_emit.Append(blackhole_binary_max_tile_init(), {});
      tile_emit.Append(blackhole_binary_max_tile(),
                       {IntImm32(0), IntImm32(1), IntImm32(0)});
    });
  if (Stmt release = ReleaseExactInputAfterUse(lhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  if (Stmt release = ReleaseExactInputAfterUse(rhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  emit.Push(out.cb_id, out.num_tiles);
  const bool materialize_loop_carried = ShouldMaterializeLoopCarriedExactOutput(dst);
  const FutureBufferUses future_uses =
      ClassifyFutureBufferUses(dst, current_lowering_order_index_);
  const bool materialize_completed_loop_carried_before_compute_consume =
      !materialize_loop_carried && IsCompletedLoopCarriedBuffer(dst) &&
      future_uses.has_compute_consume;
  const bool materialize_local_state =
      materialize_loop_carried ||
      materialize_completed_loop_carried_before_compute_consume;
  if (materialize_local_state) {
    if (Stmt materialize =
            materialize_loop_carried
                ? MaterializeLoopCarriedExactOutput(dst, out)
                : [&]() {
                    InvalidateLastFragmentFillValue(dst);
                    ClearSelectedSourceLiveProducer(dst);
                    ClearTiledCBLiveFormAliases(dst);
                    MarkLocalOnlyLiveFormAliases(dst);
                    Stmt local_materialize =
                        MaterializeExactTiledCBToLocalBuffer(dst, out, /*pop_front=*/true);
                    ClearTiledCBLiveFormAliases(dst);
                    MarkLocalOnlyLiveFormAliases(dst);
                    return local_materialize;
                  }();
        materialize.defined()) {
      stmts.push_back(materialize);
    }
  } else {
    RecordExactOutputLiveForm(dst, out);
  }
  Stmt body = SeqStmt::Flatten(stmts);
  if (!materialize_local_state) {
    body = AttachExactOutputLiveFormMarker(dst, out, body);
  }
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateBinaryTileSequence(const Buffer& dst,
                                                  const Buffer& rhs,
                                                  const std::string& operation_name,
                                                  const Op& init_op,
                                                  const Op& tile_op) {
  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, operation_name + "_lhs");
  ExactTiledCBValue rhs_in = CreateExactInputCBValue(rhs, operation_name + "_rhs");
  ExactTiledCBValue out;
  if (!TryCreateLoopCarriedExactOutputStateCBValue(dst, &out)) {
    out = CreateEmptyExactTiledCBValue(
        dst, operation_name + "_out",
        ExactOutputCBTypeForBuffer(dst, current_lowering_order_index_));
  }
  RecordExactComputeOpPlan("binary", operation_name,
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs, "identity"},
                            {"output", dst, "identity"}});

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.BinaryOpInitCommon(lhs_in.cb_id, rhs_in.cb_id, out.cb_id);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(lhs_in.cb_id, lhs_in.num_tiles);
  emit.Wait(rhs_in.cb_id, rhs_in.num_tiles);
  emit.ReconfigDataFormat(lhs_in.cb_id, rhs_in.cb_id);
  emit.Append(init_op, {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id)});
  emit.EmitPackedTileLoop(out.cb_id, out.num_tiles, "tile",
                          [&](ExactTileComputeEmitter& tile_emit,
                              const PrimExpr& tile) {
      const PrimExpr lhs_tile = ExactCBReadTileIndex(lhs_in.num_tiles, tile);
      const PrimExpr rhs_tile = ExactCBReadTileIndex(rhs_in.num_tiles, tile);
      tile_emit.Append(tile_op,
                       {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id),
                        lhs_tile, rhs_tile, IntImm32(0)});
    });
  if (Stmt release = ReleaseExactInputAfterUse(lhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  if (Stmt release = ReleaseExactInputAfterUse(rhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  emit.Push(out.cb_id, out.num_tiles);
  const bool materialize_loop_carried = ShouldMaterializeLoopCarriedExactOutput(dst);
  if (materialize_loop_carried) {
    if (Stmt materialize = MaterializeLoopCarriedExactOutput(dst, out); materialize.defined()) {
      stmts.push_back(materialize);
    }
  } else {
    RecordExactOutputLiveForm(dst, out);
  }
  Stmt body = SeqStmt::Flatten(stmts);
  if (!materialize_loop_carried) {
    body = AttachExactOutputLiveFormMarker(dst, out, body);
  }
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateBroadcastColsBinaryTileSequence(
    const Buffer& dst, const Buffer& rhs, const std::string& operation_name,
    const Op& init_op, const Op& tile_op, const PrimExpr& num_elements,
    const PrimExpr& row_width) {
  (void)num_elements;
  (void)row_width;

  const std::string rhs_identity = BufferIdentityName(rhs);
  if (!rhs_identity.empty()) {
    broadcast_cols_rhs_buffers_.insert(rhs_identity);
    RefreshBroadcastColsSourceBuffers();
  }

  ExactTiledCBValue lhs_in = CreateExactInputCBValue(dst, operation_name + "_lhs");
  ExactTiledCBValue rhs_in;
  if (!TryCreateBroadcastColsSourceLiveExactTiledCBValue(rhs, &rhs_in)) {
    rhs_in = CreateExactInputCBValue(rhs, operation_name + "_rhs");
  }
  ExactTiledCBValue out;
  if (!TryCreateLoopCarriedExactOutputStateCBValue(dst, &out)) {
    out = CreateEmptyExactTiledCBValue(
        dst, operation_name + "_out",
        ExactOutputCBTypeForBuffer(dst, current_lowering_order_index_));
  }
  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(dst);
  const bool materialize_loop_carried = ShouldMaterializeLoopCarriedExactOutput(dst);
  const bool materialize_completed_loop_carried_before_rewrite =
      !materialize_loop_carried && IsCompletedLoopCarriedBuffer(dst) && logical_rows > 1 &&
      logical_cols > 1 &&
      FutureWritePrecedesFutureComputeConsume(dst, current_lowering_order_index_);
  const int tiles_per_row =
      logical_rows > 0 && logical_cols > 0
          ? std::max(1, CeilDivToInt(logical_cols, kBlackholeTileCols))
          : std::max(1, lhs_in.num_tiles / std::max(1, rhs_in.num_tiles));

  std::vector<Stmt> stmts;
  if (Stmt publish_lhs = PublishExactInputToTiledCB(dst, &lhs_in); publish_lhs.defined()) {
    stmts.push_back(publish_lhs);
  }
  if (Stmt publish_rhs = PublishExactInputToTiledCB(rhs, &rhs_in); publish_rhs.defined()) {
    stmts.push_back(publish_rhs);
  }

  RecordExactComputeOpPlan("binary", operation_name,
                           {{"lhs", dst, "identity"},
                            {"rhs", rhs, "broadcast"},
                            {"output", dst, "identity"}});
  MarkExactCBValuesOverlap({lhs_in.cb_id, rhs_in.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.BinaryOpInitCommon(lhs_in.cb_id, rhs_in.cb_id, out.cb_id);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(lhs_in.cb_id, lhs_in.num_tiles);
  emit.Wait(rhs_in.cb_id, rhs_in.num_tiles);
  emit.ReconfigDataFormat(lhs_in.cb_id, rhs_in.cb_id);
  emit.Append(init_op, {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id)});
  emit.EmitPackedTileLoop(out.cb_id, out.num_tiles, "tile",
                          [&](ExactTileComputeEmitter& tile_emit,
                              const PrimExpr& tile) {
      const PrimExpr lhs_tile = ExactCBReadTileIndex(lhs_in.num_tiles, tile);
      const PrimExpr rhs_base_tile = tiles_per_row <= 1
                                        ? tile
                                        : FloorDiv(tile, IntImm32(tiles_per_row));
      const PrimExpr rhs_tile = ExactCBReadTileIndex(rhs_in.num_tiles, rhs_base_tile);
      tile_emit.Append(tile_op,
                       {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id),
                        lhs_tile, rhs_tile, IntImm32(0)});
    });
  if (Stmt release = ReleaseExactInputAfterUse(lhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  if (Stmt release = ReleaseExactInputAfterUse(rhs_in, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  emit.Push(out.cb_id, out.num_tiles);
  const bool materialize_local_state =
      materialize_loop_carried || materialize_completed_loop_carried_before_rewrite;
  if (materialize_local_state) {
    Stmt materialize =
        materialize_loop_carried
            ? MaterializeLoopCarriedExactOutput(dst, out)
            : [&]() {
                InvalidateLastFragmentFillValue(dst);
                ClearSelectedSourceLiveProducer(dst);
                ClearTiledCBLiveFormAliases(dst);
                MarkLocalOnlyLiveFormAliases(dst);
                Stmt local_materialize =
                    MaterializeExactTiledCBToLocalBuffer(dst, out, /*pop_front=*/true);
                ClearTiledCBLiveFormAliases(dst);
                MarkLocalOnlyLiveFormAliases(dst);
                return local_materialize;
              }();
    if (materialize.defined()) {
      stmts.push_back(materialize);
    }
  } else {
    RecordExactOutputLiveForm(dst, out);
  }
  Stmt body = SeqStmt::Flatten(stmts);
  if (!materialize_local_state) {
    body = AttachExactOutputLiveFormMarker(dst, out, body);
  }
  return MaybeWrapComputeSegment(body);
}

Stmt PlanTTKernelABI::GenerateUnaryTileSequence(
    const Buffer& input, const Buffer& output, const std::string& operation_name,
    const Op& init_op, const Op& tile_op, const PrimExpr& num_elements) {
  ExactTiledCBValue input_cb = CreateExactInputCBValue(input, operation_name + "_input");
  ExactTiledCBValue out;
  if (!TryCreateLoopCarriedExactOutputStateCBValue(output, &out)) {
    out = CreateEmptyExactTiledCBValue(
        output, operation_name + "_out",
        ExactOutputCBTypeForBuffer(output, current_lowering_order_index_));
  }
  RefineExactTiledCBValueShapeFromNumElements(&out, num_elements);
  RefineExactTiledCBValueShapeFromNumElements(
      &out, IntImm(DataType::Int(64), input_cb.num_elements));
  out.row_width = std::max(out.row_width, input_cb.row_width);
  RecordExactComputeOpPlan("unary", operation_name,
                           {{"input", input, "identity"},
                            {"output", output, "identity"}});
  std::vector<Stmt> stmts;
  if (Stmt publish_input = PublishExactInputToTiledCB(input, &input_cb);
      publish_input.defined()) {
    stmts.push_back(publish_input);
  }
  MarkExactCBValuesOverlap({input_cb.cb_id, out.cb_id});
  ExactTileComputeEmitter emit(&stmts);
  emit.UnaryOpInitCommon(input_cb.cb_id, out.cb_id);
  emit.Reserve(out.cb_id, out.num_tiles);
  emit.Wait(input_cb.cb_id, input_cb.num_tiles);
  emit.EmitPackedTileLoop(out.cb_id, out.num_tiles, "tile",
                          [&](ExactTileComputeEmitter& tile_emit,
                              const PrimExpr& tile) {
      const PrimExpr input_tile = input_cb.num_tiles == 1
                                      ? IntImm32(0)
                                      : FloorMod(tile, IntImm32(input_cb.num_tiles));
      tile_emit.Append(blackhole_copy_tile_to_dst_init_short(), {IntImm32(input_cb.cb_id)});
      tile_emit.Append(blackhole_copy_tile(),
                       {IntImm32(input_cb.cb_id), input_tile, IntImm32(0)});
      tile_emit.Append(init_op, {});
      tile_emit.Append(tile_op, {IntImm32(0)});
    });
  if (Stmt release = ReleaseExactInputAfterUse(input_cb, current_lowering_order_index_);
      release.defined()) {
    stmts.push_back(release);
  }
  emit.Push(out.cb_id, out.num_tiles);
  const bool materialize_loop_carried = ShouldMaterializeLoopCarriedExactOutput(output);
  if (materialize_loop_carried) {
    if (Stmt materialize = MaterializeLoopCarriedExactOutput(output, out); materialize.defined()) {
      stmts.push_back(materialize);
    }
  } else {
    RecordExactOutputLiveForm(output, out);
  }

  Stmt body = SeqStmt::Flatten(stmts);
  if (!materialize_loop_carried) {
    body = AttachExactOutputLiveFormMarker(output, out, body);
  }
  return MaybeWrapComputeSegment(body);
}

}  // namespace tl
}  // namespace tvm
