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
 * \file lower_blackhole_ops.cc
 * \brief Implementation of PlanTTKernelABI pass.
 *
 * Transforms TileLang high-level operations (T.copy, T.gemm, T.clear)
 * into TT-Metal builtin sequences.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_lowering_requirements.h"
#include "common/blackhole_utils.h"
#include "common/blackhole_runtime_arg_schema.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

#include <tvm/ir/attrs.h>
#include "runtime/thread_storage_scope.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <optional>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

static std::string DataTypeToDataFormatForBlackhole(DataType dtype) {
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

static std::string CBFlowClassToString(CBFlowClass flow_class) {
  switch (flow_class) {
    case CBFlowClass::kStream:
      return buffer_flow::kStream;
    case CBFlowClass::kRepublish:
      return buffer_flow::kRepublish;
    case CBFlowClass::kState:
    default:
      return buffer_flow::kState;
  }
}

static std::optional<CBFlowClass> ParseCBFlowClass(const std::string& flow_class) {
  if (flow_class == buffer_flow::kState) {
    return CBFlowClass::kState;
  }
  if (flow_class == buffer_flow::kStream) {
    return CBFlowClass::kStream;
  }
  if (flow_class == buffer_flow::kRepublish) {
    return CBFlowClass::kRepublish;
  }
  return std::nullopt;
}

static bool IsLiteralZeroValue(const PrimExpr& expr) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value == 0;
  }
  if (const auto* imm = expr.as<FloatImmNode>()) {
    return imm->value == 0.0;
  }
  if (const auto* cast = expr.as<CastNode>()) {
    return IsLiteralZeroValue(cast->value);
  }
  return false;
}

struct CBDepthEffect {
  explicit CBDepthEffect(size_t num_requirements = 0)
      : peak_extra_pages(num_requirements, 0), net_page_delta(num_requirements, 0) {}

  std::vector<int64_t> peak_extra_pages;
  std::vector<int64_t> net_page_delta;
};

static bool IsBlackholeBuiltinCall(const tir::CallNode* call,
                                   const tvm::Op& builtin,
                                   const char* op_name) {
  if (!call) {
    return false;
  }
  if (call->op.same_as(builtin)) {
    return true;
  }
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == op_name;
  }
  return false;
}

static CBDepthEffect CombineCBDepthEffectSequential(const CBDepthEffect& lhs,
                                                    const CBDepthEffect& rhs) {
  ICHECK_EQ(lhs.peak_extra_pages.size(), rhs.peak_extra_pages.size());
  CBDepthEffect combined(lhs.peak_extra_pages.size());
  for (size_t i = 0; i < lhs.peak_extra_pages.size(); ++i) {
    combined.peak_extra_pages[i] =
        std::max(lhs.peak_extra_pages[i],
                 std::max<int64_t>(0, lhs.net_page_delta[i] + rhs.peak_extra_pages[i]));
    combined.net_page_delta[i] = lhs.net_page_delta[i] + rhs.net_page_delta[i];
  }
  return combined;
}

static CBDepthEffect RepeatCBDepthEffect(const CBDepthEffect& body, int64_t extent) {
  CBDepthEffect repeated(body.peak_extra_pages.size());
  if (extent <= 0) {
    return repeated;
  }
  for (size_t i = 0; i < body.peak_extra_pages.size(); ++i) {
    repeated.net_page_delta[i] = body.net_page_delta[i] * extent;
    repeated.peak_extra_pages[i] = body.peak_extra_pages[i];
    if (body.net_page_delta[i] > 0 && extent > 1) {
      repeated.peak_extra_pages[i] += (extent - 1) * body.net_page_delta[i];
    }
  }
  return repeated;
}

static CBDepthEffect MergeCBDepthEffectBranches(const CBDepthEffect& then_effect,
                                                const CBDepthEffect& else_effect) {
  ICHECK_EQ(then_effect.peak_extra_pages.size(), else_effect.peak_extra_pages.size());
  CBDepthEffect merged(then_effect.peak_extra_pages.size());
  for (size_t i = 0; i < then_effect.peak_extra_pages.size(); ++i) {
    merged.peak_extra_pages[i] =
        std::max(then_effect.peak_extra_pages[i], else_effect.peak_extra_pages[i]);
    merged.net_page_delta[i] =
        std::max(then_effect.net_page_delta[i], else_effect.net_page_delta[i]);
  }
  return merged;
}

static CBDepthEffect AnalyzeCBDepthEffect(const tir::Stmt& stmt,
                                          size_t num_requirements,
                                          const std::string& requested_segment_kind = "",
                                          const std::string& default_segment_kind = "compute",
                                          const std::string& active_segment_kind = "") {
  CBDepthEffect empty(num_requirements);
  if (!stmt.defined()) {
    return empty;
  }

  if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
    CBDepthEffect effect(num_requirements);
    for (const tir::Stmt& child : seq->seq) {
      effect = CombineCBDepthEffectSequential(
          effect, AnalyzeCBDepthEffect(child, num_requirements, requested_segment_kind,
                                       default_segment_kind, active_segment_kind));
    }
    return effect;
  }
  if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
    if (attr->attr_key == "blackhole.segment_kind") {
      if (const auto* kind = attr->value.as<StringImmNode>()) {
        return AnalyzeCBDepthEffect(attr->body, num_requirements, requested_segment_kind,
                                    default_segment_kind, kind->value);
      }
    }
    return AnalyzeCBDepthEffect(attr->body, num_requirements, requested_segment_kind,
                                default_segment_kind, active_segment_kind);
  }
  if (const auto* let = stmt.as<tir::LetStmtNode>()) {
    return AnalyzeCBDepthEffect(let->body, num_requirements, requested_segment_kind,
                                default_segment_kind, active_segment_kind);
  }
  if (const auto* decl = stmt.as<tir::DeclBufferNode>()) {
    return AnalyzeCBDepthEffect(decl->body, num_requirements, requested_segment_kind,
                                default_segment_kind, active_segment_kind);
  }
  if (const auto* alloc = stmt.as<tir::AllocateNode>()) {
    return AnalyzeCBDepthEffect(alloc->body, num_requirements, requested_segment_kind,
                                default_segment_kind, active_segment_kind);
  }
  if (const auto* if_then_else = stmt.as<tir::IfThenElseNode>()) {
    CBDepthEffect then_effect =
        AnalyzeCBDepthEffect(if_then_else->then_case, num_requirements, requested_segment_kind,
                             default_segment_kind, active_segment_kind);
    CBDepthEffect else_effect = AnalyzeCBDepthEffect(
        if_then_else->else_case.value_or(tir::Stmt()), num_requirements, requested_segment_kind,
        default_segment_kind, active_segment_kind);
    return MergeCBDepthEffectBranches(then_effect, else_effect);
  }
  if (const auto* loop = stmt.as<tir::ForNode>()) {
    CBDepthEffect body_effect =
        AnalyzeCBDepthEffect(loop->body, num_requirements, requested_segment_kind,
                             default_segment_kind, active_segment_kind);
    if (const auto* extent = loop->extent.as<IntImmNode>()) {
      return RepeatCBDepthEffect(body_effect, extent->value);
    }
    return body_effect;
  }
  if (const auto* eval = stmt.as<tir::EvaluateNode>()) {
    const auto* call = eval->value.as<tir::CallNode>();
    if (!call || call->args.size() < 2) {
      return empty;
    }

    int64_t delta_sign = 0;
    if (IsBlackholeBuiltinCall(call, tir::builtin::blackhole_cb_reserve_back(),
                               "tl.blackhole.cb_reserve_back")) {
      delta_sign = 1;
    } else if (IsBlackholeBuiltinCall(call, tir::builtin::blackhole_cb_pop_front(),
                                      "tl.blackhole.cb_pop_front")) {
      delta_sign = -1;
    } else {
      return empty;
    }

    const std::string effective_segment_kind =
        active_segment_kind.empty() ? default_segment_kind : active_segment_kind;
    if (!requested_segment_kind.empty() && effective_segment_kind != requested_segment_kind) {
      return empty;
    }

    const auto* cb_id = call->args[0].as<IntImmNode>();
    const auto* page_count = call->args[1].as<IntImmNode>();
    if (!cb_id || !page_count || cb_id->value < 0 ||
        static_cast<size_t>(cb_id->value) >= num_requirements) {
      return empty;
    }

    CBDepthEffect effect(num_requirements);
    const int requirement_index = static_cast<int>(cb_id->value);
    const int64_t delta = delta_sign * page_count->value;
    effect.net_page_delta[requirement_index] = delta;
    if (delta > 0) {
      effect.peak_extra_pages[requirement_index] = delta;
    }
    return effect;
  }

  return empty;
}

static void UpdateCBRequirementDepthsFromLoweredBody(std::vector<CBRequirement>* requirements,
                                                     const tir::Stmt& body,
                                                     const std::string& default_segment_kind) {
  ICHECK(requirements != nullptr);
  if (requirements->empty()) {
    return;
  }
  std::vector<CBDepthEffect> effects;
  effects.push_back(AnalyzeCBDepthEffect(body, requirements->size(), "",
                                         default_segment_kind));
  effects.push_back(AnalyzeCBDepthEffect(body, requirements->size(), default_segment_kind,
                                         default_segment_kind));
  if (default_segment_kind != "fused_dataflow") {
    effects.push_back(AnalyzeCBDepthEffect(body, requirements->size(), "reader",
                                           default_segment_kind));
    effects.push_back(AnalyzeCBDepthEffect(body, requirements->size(), "compute",
                                           default_segment_kind));
    effects.push_back(AnalyzeCBDepthEffect(body, requirements->size(), "writer",
                                           default_segment_kind));
  }
  for (size_t i = 0; i < requirements->size(); ++i) {
    int max_num_pages = (*requirements)[i].num_pages;
    for (const CBDepthEffect& effect : effects) {
      max_num_pages = std::max(max_num_pages, static_cast<int>(effect.peak_extra_pages[i]));
    }
    (*requirements)[i].num_pages = max_num_pages;
  }
}

using tir::PrimFunc;
using tir::PrimFuncNode;
using tir::Stmt;
using tir::StmtExprMutator;
using tir::CallNode;
using tir::BufferStoreNode;
using tir::BufferLoadNode;
using tir::EvaluateNode;
using tir::Call;
using tir::Evaluate;
using tir::SeqStmt;
using tir::LetStmt;
using tir::Var;
using tir::For;
using tir::ForNode;
using tir::AttrStmt;
using tir::AttrStmtNode;
using tir::IterVar;
using tir::Buffer;
using tir::builtin::blackhole_mm_init;
using tir::builtin::blackhole_reconfig_data_format;
using tir::builtin::blackhole_mm_init_short;
using tir::builtin::blackhole_mm_init_short_with_dt;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_matmul_tiles;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_pack_reconfig_data_format;
using tir::builtin::blackhole_copy_tile_to_dst_init_short;
using tir::builtin::blackhole_copy_tile_to_dst_init_short_with_dt;
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_add_tiles_init;
using tir::builtin::blackhole_add_tiles;
using tir::builtin::blackhole_add_bcast_rows_init_short;
using tir::builtin::blackhole_add_bcast_cols_init_short;
using tir::builtin::blackhole_add_tiles_bcast_rows;
using tir::builtin::blackhole_add_tiles_bcast_cols;
using tir::builtin::blackhole_mul_tiles_init;
using tir::builtin::blackhole_mul_tiles;
using tir::builtin::blackhole_mul_bcast_rows_init_short;
using tir::builtin::blackhole_mul_bcast_cols_init_short;
using tir::builtin::blackhole_mul_tiles_bcast_rows;
using tir::builtin::blackhole_mul_tiles_bcast_cols;
using tir::builtin::blackhole_reduce_init;
using tir::builtin::blackhole_reduce_tile;
using tir::builtin::blackhole_reduce_uninit;
using tir::builtin::blackhole_binary_max_tile_init;
using tir::builtin::blackhole_binary_max_tile;
using tir::builtin::blackhole_div_binary_tile_init;
using tir::builtin::blackhole_div_binary_tile;
using tir::builtin::blackhole_exp2_tile_init;
using tir::builtin::blackhole_exp2_tile;
using tir::builtin::blackhole_recip_tile_init;
using tir::builtin::blackhole_recip_tile;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_add_fragment;
using tir::builtin::blackhole_add_fragment_from_cb_front;
using tir::builtin::blackhole_noc_async_read;
using tir::builtin::blackhole_noc_async_write;
using tir::builtin::blackhole_noc_async_read_barrier;
using tir::builtin::blackhole_noc_async_write_barrier;
using tir::builtin::blackhole_read_tile_to_cb;
using tir::builtin::blackhole_read_page_to_cb;
using tir::builtin::blackhole_write_tile_from_cb;
using tir::builtin::blackhole_write_page_from_cb;
using tir::builtin::blackhole_pack_untilize_slice;
using tir::builtin::blackhole_pack_untilize_tile;
using tir::builtin::blackhole_tilize_local_fragment_slice;
using tir::builtin::blackhole_tilize_cast_fragment_slice;
using tir::builtin::blackhole_pack_fill_fragment_to_tiled_cb;
using tir::builtin::blackhole_untilize_cb_front_tile;
using tir::builtin::blackhole_untilize_cb_front_tile_fragment;
using tvm::Integer;
using tvm::DataType;
using tvm::IntImm;
using tvm::DictAttrs;
using tvm::ffi::GetRef;
using tvm::ffi::String;
using ffi::String;
using tvm::ffi::Map;
using tvm::ffi::Array;
using tvm::ffi::Any;
using tvm::arith::Analyzer;

static constexpr const char* kBlackholeExactOutputLiveCBAttr =
    "blackhole.exact_output_live_cb";

// Helper to create a call to TT-Metal builtin
static Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

// Helper to create IntImm(32) expression
static PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
}

static std::optional<std::vector<int64_t>> ExtractStaticShape(const Array<PrimExpr>& shape) {
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

static int64_t ComputeStaticElementCount(const std::vector<int64_t>& shape) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    total_elements *= dim;
  }
  return total_elements;
}

static int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  if (value <= 0) {
    return 1;
  }
  return static_cast<int>((value + divisor - 1) / divisor);
}


static PrimExpr ScalarizeVectorizedIndex(const PrimExpr& index) {
  if (const auto* ramp = index.as<tir::RampNode>()) {
    return ramp->base;
  }
  return index;
}

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

static bool IsUnsupportedResidualLocalScope(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

static bool IsVectorLocalFragmentBuffer(const Buffer& buffer) {
  return IsUnsupportedResidualLocalScope(buffer) && buffer->shape.size() == 1 &&
         !buffer->shape.empty() && !tir::is_one(buffer->shape[0]);
}

static void ValidateNoResidualComputeRegionStores(const Stmt& body) {
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    if (const auto* store = node.as<BufferStoreNode>()) {
      if (store->buffer->shape.size() == 1) {
        return;
      }
      if (IsUnsupportedResidualLocalScope(store->buffer)) {
        ICHECK(false)
            << "Blackhole compute subset lowering is not implemented; residual local "
               "store remains for buffer "
            << store->buffer->name;
      }
    }
  });
}

namespace {
bool IsFragmentFillValue(const PrimExpr& expr);
bool HasResidualFragmentFill(const Stmt& body);
bool HasResidualFragmentAdd(const Stmt& body);
bool HasResidualFragmentMax(const Stmt& body);
bool HasResidualFragmentCast(const Stmt& body);
bool HasResidualScalarLoadBroadcast(const Stmt& body);
}  // namespace

static std::vector<std::string> CollectLeafUnsupportedComputeOpsFromBody(const Stmt& body) {
  std::vector<std::string> unsupported_ops;
  std::unordered_set<std::string> seen_ops;
  auto push = [&](const char* op_name) {
    if (seen_ops.insert(op_name).second) {
      unsupported_ops.push_back(op_name);
    }
  };
  if (HasResidualScalarLoadBroadcast(body)) {
    push("broadcast");
  }
  if (HasResidualFragmentFill(body)) {
    push("fill");
  }
  if (HasResidualFragmentMax(body)) {
    push("max");
  }
  if (HasResidualFragmentAdd(body)) {
    push("add");
  }
  if (HasResidualFragmentCast(body)) {
    push("cast");
  }
  return unsupported_ops;
}

static bool IsStageLocalScopeForPipelineLegality(const std::string& scope) {
  return scope.rfind("shared", 0) == 0 || scope.rfind("blackhole.cb.", 0) == 0;
}

static std::optional<int64_t> GetPipelineStageCountFromLoop(const ForNode* loop) {
  if (!loop || !loop->annotations.defined()) {
    return std::nullopt;
  }
  for (const char* key : {"num_stages", "tl_pipelined_num_stages"}) {
    if (auto value = loop->annotations.Get(key)) {
      if (const auto* imm = value.value().as<IntImmNode>()) {
        return imm->value;
      }
    }
  }
  return std::nullopt;
}

static std::optional<int64_t> InferPipelineStageCountFromStmt(const Stmt& stmt) {
  std::optional<int64_t> inferred;
  tir::PostOrderVisit(stmt, [&inferred](const ObjectRef& node) {
    auto update_from_buffer = [&inferred](const Buffer& buffer) {
      const std::string scope = buffer.scope();
      if (!IsStageLocalScopeForPipelineLegality(scope) || buffer->shape.size() < 3) {
        return;
      }
      if (const auto* imm = buffer->shape[0].as<IntImmNode>()) {
        if (imm->value > 0) {
          inferred = imm->value;
        }
      }
    };
    if (const auto* store = node.as<BufferStoreNode>()) {
      update_from_buffer(store->buffer);
      return;
    }
    if (const auto* load = node.as<BufferLoadNode>()) {
      update_from_buffer(load->buffer);
    }
  });
  return inferred;
}

static void ValidateComputePipelineLegalityFromBody(const Stmt& body) {
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* loop = node.as<ForNode>();
    if (!loop) {
      return;
    }
    std::optional<int64_t> maybe_stages = GetPipelineStageCountFromLoop(loop);
    if (!maybe_stages.has_value()) {
      maybe_stages = InferPipelineStageCountFromStmt(GetRef<Stmt>(loop));
    }
    if (!maybe_stages.has_value()) {
      return;
    }
    const int64_t stages = maybe_stages.value();
    ICHECK_LE(stages, 2)
        << "Blackhole compute pipeline legality: unsupported stage count " << stages;
  });
}

static bool HasComputeSegmentRequirement(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    if (const auto* call = node.as<CallNode>()) {
      if (call->op->IsInstance<OpNode>() &&
          (Downcast<Op>(call->op)->name == "tl.tileop.gemm_py" ||
           Downcast<Op>(call->op)->name == blackhole_tile_compute_schema::kOpName)) {
        found = true;
      }
      return;
    }
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer)) {
      return;
    }
    if (!store->value.as<BufferLoadNode>()) {
      found = true;
    }
  });
  return found;
}

static BlackholeLoweringSupportFacts BuildLoweringSupportFactsFromAnalysis(const PrimFunc& func) {
  auto spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(spatial_plan)
      << "PlanTTKernelABI requires tl.spatial_plan; run BuildSpatialPlan before lowering";
  ICHECK(func->GetAttr<Bool>(attr::kTLSpatialPlanValidated, Bool(false)).value())
      << "PlanTTKernelABI requires validated SpatialPlan; run ValidateSpatialPlan before lowering";
  return CollectBlackholeLoweringSupportFacts(func, spatial_plan.value());
}

static bool BufferMaterializationFactHasLogicalRowWidth(
    const BlackholeBufferMaterializationFact& fact) {
  return fact.logical_row_width > 0;
}

static bool BufferMaterializationFactHasLogicalElementCount(
    const BlackholeBufferMaterializationFact& fact) {
  return fact.logical_element_count > 0;
}

static int BufferMaterializationFactSpecificityScore(
    const BlackholeBufferMaterializationFact& fact) {
  int score = 0;
  if (BufferMaterializationFactHasLogicalRowWidth(fact)) {
    score += 4;
  }
  if (BufferMaterializationFactHasLogicalElementCount(fact)) {
    score += 2;
  }
  if (!fact.source_buffer.empty()) {
    score += 1;
  }
  if (fact.spatial_materialization_boundary_index >= 0) {
    score += 8;
  }
  return score;
}

static std::unordered_map<std::string, BlackholeBufferMaterializationFact>
BuildBufferMaterializationFactMap(
    const std::vector<BlackholeBufferMaterializationFact>& buffer_materialization_facts) {
  std::unordered_map<std::string, BlackholeBufferMaterializationFact> facts_by_target_buffer;
  for (const BlackholeBufferMaterializationFact& fact : buffer_materialization_facts) {
    if (!fact.target_buffer.empty()) {
      auto existing_it = facts_by_target_buffer.find(fact.target_buffer);
      if (existing_it == facts_by_target_buffer.end() ||
          BufferMaterializationFactSpecificityScore(fact) >=
              BufferMaterializationFactSpecificityScore(existing_it->second)) {
        // Keep the most specific fact for each target buffer so later
        // cast-/publish-driven facts can override generic seed entries.
        facts_by_target_buffer[fact.target_buffer] = fact;
      }
    }
  }
  return facts_by_target_buffer;
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

static int64_t StaticIntValueOrDefault(const PrimExpr& expr, int64_t default_value = 0) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return default_value;
}

static Array<Integer> ExtractStaticShape(const Buffer& buffer) {
  Array<Integer> shape;
  for (const PrimExpr& dim : buffer->shape) {
    if (const auto* imm = dim.as<IntImmNode>()) {
      shape.push_back(Integer(imm->value));
    }
  }
  return shape;
}

static Stmt StripSegmentKindMarkers(const Stmt& body) {
  class SegmentMarkerStripper : public tir::StmtMutator {
   public:
    Stmt VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == "blackhole.segment_kind") {
        return VisitStmt(op->body);
      }
      return tir::StmtMutator::VisitStmt_(op);
    }
  };

  return SegmentMarkerStripper()(body);
}

static Stmt WrapSegmentStmtIfNeeded(const std::string& current_segment_kind,
                                    const std::string& segment_kind,
                                    const Stmt& stmt) {
  if (!stmt.defined() || !current_segment_kind.empty() || segment_kind == "fused_dataflow") {
    return stmt;
  }
  auto wrap_one = [&](const Stmt& inner) {
    return AttrStmt(StringImm("blackhole.segment_kind"), "blackhole.segment_kind",
                    StringImm(segment_kind), inner);
  };
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    ffi::Array<Stmt> wrapped;
    wrapped.reserve(seq->seq.size());
    for (const Stmt& child : seq->seq) {
      wrapped.push_back(wrap_one(child));
    }
    return tir::SeqStmt(wrapped);
  }
  return wrap_one(stmt);
}

// Helper to get storage scope from buffer
static std::string GetStorageScope(const Buffer& buffer) {
  // Use the scope() method which returns ffi::String
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

PlanTTKernelABI::PlanTTKernelABI() : next_requirement_index_(0) {}

PrimFunc PlanTTKernelABI::SelectComputeBuiltins(const PrimFunc& func) {
  current_func_ = func;
  buffer_to_req_.clear();
  buffer_data_to_req_index_.clear();
  buffer_identity_to_req_index_.clear();
  cb_requirements_.clear();
  next_requirement_index_ = 0;
  logical_buffer_shapes_.clear();
  compute_physical_buffers_by_data_.clear();
  compute_physical_buffers_by_identity_.clear();
  host_buffer_by_compute_operand_buffer_.clear();
  selected_source_live_producer_buffers_.clear();
  seeded_cb_requirement_names_.clear();
  last_fragment_fill_value_by_buffer_identity_.clear();
  last_fragment_fill_value_by_data_.clear();
  LoadPhysicalComputeBufferBindings(func);
  current_segment_kind_.clear();
  thread_index_vars_.clear();
  thread_index_var_names_.clear();
  thread_index_var_static_extents_.clear();
  loop_var_static_extents_.clear();
  block_index_vars_.clear();
  block_index_var_names_.clear();
  requires_compute_segment_ = false;
  logical_tile_layout_specs_by_buffer_.clear();
  spatial_materialization_boundaries_.clear();
  spatial_materialization_boundary_position_by_index_.clear();
  buffer_materialization_facts_by_target_buffer_.clear();
  tt_compute_op_plans_.clear();
  tile_compute_dag_lowering_decisions_.clear();
  tile_compute_dag_lowering_decision_consumed_.clear();
  active_tile_compute_dag_lowering_decision_.reset();
  select_compute_builtins_only_ = true;

  auto maybe_spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(maybe_spatial_plan)
      << "PlanTTKernelABI requires tl.spatial_plan; run BuildSpatialPlan before lowering";
  LoadSpatialLiveValueBoundaries(maybe_spatial_plan.value());
  const BlackholeLoweringSupportFacts lowering_support_facts =
      BuildLoweringSupportFactsFromAnalysis(func);
  LoadLogicalBufferShapes(func, lowering_support_facts, maybe_spatial_plan.value());
  requires_compute_segment_ = HasComputeSegmentRequirement(func->body);
  LoadLogicalTileLayoutSpecs(maybe_spatial_plan.value());
  buffer_materialization_facts_by_target_buffer_ =
      BuildBufferMaterializationFactMap(
          lowering_support_facts.buffer_materialization_facts);
  LoadTileComputeDAGLoweringPlan(func);

  PrimFunc selected = func;
  selected.CopyOnWrite()->body = VisitStmt(func->body);
  UpdateCBRequirementDepthsFromLoweredBody(
      &cb_requirements_, selected->body, gemm_a_buffer_name_.empty() ? "fused_dataflow" : "compute");
  select_compute_builtins_only_ = false;
  return selected;
}

void PlanTTKernelABI::LoadLogicalBufferShapes(
    const PrimFunc& func, const BlackholeLoweringSupportFacts& lowering_support_facts,
    const SpatialPlan& spatial_plan) {
  logical_buffer_shapes_.clear();
  std::unordered_map<std::string, std::vector<int64_t>> canonical_shapes;
  std::unordered_map<std::string, int> canonical_shape_priority;
  std::unordered_map<const VarNode*, std::vector<std::string>> alias_names_by_data;
  auto register_shape = [&](const std::string& name, const std::vector<int64_t>& shape,
                            int priority) {
    if (name.empty() || shape.empty()) {
      return;
    }
    auto it = canonical_shapes.find(name);
    if (it != canonical_shapes.end()) {
      const int existing_priority = canonical_shape_priority[name];
      if (priority < existing_priority) {
        return;
      }
      if (priority == existing_priority && shape.size() <= it->second.size()) {
        return;
      }
    }
    logical_buffer_shapes_[name] = shape;
    canonical_shapes[name] = shape;
    canonical_shape_priority[name] = priority;
  };
  auto ingest_buffer = [&](const Buffer& buffer) {
    if (const auto* data = BufferDataIdentity(buffer)) {
      const std::string name = BufferIdentityName(buffer);
      if (!name.empty()) {
        auto& aliases = alias_names_by_data[data];
        if (std::find(aliases.begin(), aliases.end(), name) == aliases.end()) {
          aliases.push_back(name);
        }
      }
    }
    auto static_shape = ExtractStaticShape(buffer->shape);
    if (!static_shape.has_value()) {
      return;
    }
    register_shape(BufferIdentityName(buffer), static_shape.value(), /*priority=*/0);
  };
  for (const auto& [_, buffer] : func->buffer_map) {
    ingest_buffer(buffer);
  }
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (const auto* block = node.as<tir::BlockNode>()) {
      for (const Buffer& buffer : block->alloc_buffers) {
        ingest_buffer(buffer);
      }
      return;
    }
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      ingest_buffer(store->buffer);
      return;
    }
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      ingest_buffer(load->buffer);
    }
  });

  auto decode_shape = [&](const Any& shape_any) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    for (const Integer& dim : Downcast<Array<Integer>>(shape_any)) {
      shape.push_back(dim->value);
    }
    return shape;
  };
  auto register_tile_bridge_shape = [&](const Map<String, Any>& spec) {
    auto buffer_it = spec.find(String(schema_key::kBuffer));
    auto shape_it = spec.find(String(schema_key::kShape));
    if (buffer_it == spec.end() || shape_it == spec.end()) {
      return;
    }
    register_shape(Downcast<String>((*buffer_it).second), decode_shape((*shape_it).second),
                   /*priority=*/1);
  };
  auto register_materialization_fact_shape = [&](const BlackholeBufferMaterializationFact& fact) {
    if (fact.target_buffer.empty() || fact.logical_row_width <= 0 ||
        fact.logical_element_count <= 0) {
      return;
    }
    if (fact.logical_element_count % fact.logical_row_width != 0) {
      return;
    }
    register_shape(fact.target_buffer,
                   {fact.logical_element_count / fact.logical_row_width,
                    fact.logical_row_width},
                   /*priority=*/1);
  };
  for (const auto& [_, spec] : BuildLogicalTileLayoutSpecMap(spatial_plan)) {
    register_tile_bridge_shape(spec);
  }
  for (const BlackholeBufferMaterializationFact& fact :
       lowering_support_facts.buffer_materialization_facts) {
    register_materialization_fact_shape(fact);
  }
  for (const auto& [_, aliases] : alias_names_by_data) {
    std::vector<int64_t> shared_shape;
    for (const std::string& alias : aliases) {
      auto it = canonical_shapes.find(alias);
      if (it != canonical_shapes.end()) {
        shared_shape = it->second;
        break;
      }
    }
    if (shared_shape.empty()) {
      continue;
    }
    for (const std::string& alias : aliases) {
      logical_buffer_shapes_[alias] = shared_shape;
    }
  }
}

std::vector<int64_t> PlanTTKernelABI::GetLogicalBufferShape(const Buffer& buffer) const {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = logical_buffer_shapes_.find(buffer_identity);
  if (it != logical_buffer_shapes_.end()) {
    return it->second;
  }
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(buffer)) {
    auto shape_it = spec->find(String(schema_key::kShape));
    if (shape_it != spec->end()) {
      std::vector<int64_t> shape;
      for (const Integer& dim : Downcast<Array<Integer>>((*shape_it).second)) {
        shape.push_back(dim->value);
      }
      if (!shape.empty()) {
        return shape;
      }
    }
  }
  auto static_shape = ExtractStaticShape(buffer->shape);
  if (static_shape.has_value()) {
    return static_shape.value();
  }
  return {};
}

Array<Integer> PlanTTKernelABI::GetEncodedCurrentBufferShape(const Buffer& buffer) const {
  Array<Integer> encoded_shape;
  const std::vector<int64_t> logical_shape = GetLogicalBufferShape(buffer);
  if (!logical_shape.empty()) {
    for (int64_t dim : logical_shape) {
      encoded_shape.push_back(Integer(dim));
    }
    return encoded_shape;
  }
  return ExtractStaticShape(buffer);
}

int64_t PlanTTKernelABI::GetLogicalBufferElementCount(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (!shape.empty()) {
    return ComputeStaticElementCount(shape);
  }

  int64_t total_elements = 1;
  for (const PrimExpr& shape_dim : buffer->shape) {
    const auto* int_imm = shape_dim.as<IntImmNode>();
    if (!int_imm) {
      return 1;
    }
    total_elements *= int_imm->value;
  }
  return total_elements;
}

int PlanTTKernelABI::GetLogicalBufferTileCount(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() >= 2U) {
    const int mt = CeilDivToInt(shape[shape.size() - 2], kBlackholeTileRows);
    const int nt = CeilDivToInt(shape[shape.size() - 1], kBlackholeTileCols);
    return std::max(1, mt * nt);
  }
  constexpr int64_t kTileElements = kBlackholeTileRows * kBlackholeTileCols;
  return std::max(1, CeilDivToInt(GetLogicalBufferElementCount(buffer), kTileElements));
}

int64_t PlanTTKernelABI::GetLogicalVectorLength(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() == 1U) {
    return shape.front();
  }
  return -1;
}

std::pair<int64_t, int64_t> PlanTTKernelABI::GetLogicalMatrixShape(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() >= 2U) {
    return {shape[shape.size() - 2], shape[shape.size() - 1]};
  }
  return {-1, -1};
}

bool PlanTTKernelABI::IsSingleFullTileLogicalMatrix(const Buffer& buffer) const {
  const auto [rows, cols] = GetLogicalMatrixShape(buffer);
  return rows == kBlackholeTileRows && cols == kBlackholeTileCols;
}

PrimFunc PlanTTKernelABI::Transform(const PrimFunc& func) {
  current_func_ = func;
  buffer_to_req_.clear();
  buffer_data_to_req_index_.clear();
  buffer_identity_to_req_index_.clear();
  cb_requirements_.clear();
  accessor_descriptors_.clear();
  next_requirement_index_ = 0;
  saw_copy_op_ = false;
  needs_copy_runtime_args_ = false;
  requires_compute_segment_ = false;
  copy_input_buffer_ = Buffer();
  copy_output_buffer_ = Buffer();
  copy_input_buffer_name_.clear();
  copy_output_buffer_name_.clear();
  host_buffer_by_compute_operand_buffer_.clear();
  copy_input_shape_.clear();
  copy_output_shape_.clear();
  copy_intermediate_shape_.clear();
  thread_index_vars_.clear();
  thread_index_var_names_.clear();
  thread_index_var_static_extents_.clear();
  loop_var_static_extents_.clear();
  block_index_vars_.clear();
  block_index_var_names_.clear();
  cb_consumed_compute_input_pages_by_buffer_identity_.clear();
  cb_consumed_compute_input_use_count_by_buffer_identity_.clear();
  buffer_flow_facts_.clear();
  buffer_live_form_cb_by_buffer_identity_.clear();
  buffer_live_form_order_by_buffer_identity_.clear();
  exact_output_live_form_cb_by_buffer_identity_.clear();
  exact_output_live_form_order_by_buffer_identity_.clear();
  selected_source_live_producer_buffers_.clear();
  seeded_cb_requirement_names_.clear();
  stmt_order_index_by_node_.clear();
  current_lowering_order_index_ = -1;
  segment_plan_.clear();
  tt_kernels_.clear();
  tt_abi_plans_.clear();
  tt_live_form_plans_.clear();
  tt_materialization_plans_.clear();
  tt_consumer_binding_plans_.clear();
  logical_buffer_shapes_.clear();
  compute_physical_buffers_by_data_.clear();
  compute_physical_buffers_by_identity_.clear();
  last_fragment_fill_value_by_buffer_identity_.clear();
  last_fragment_fill_value_by_data_.clear();
  LoadPhysicalComputeBufferBindings(func);
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (const auto* attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key != tir::attr::thread_extent) {
        return;
      }
      if (const auto* iv = attr->node.as<IterVarNode>()) {
        if (std::string(iv->thread_tag).rfind("threadIdx.", 0) == 0) {
          thread_index_vars_.insert(iv->var.get());
          thread_index_var_names_.insert(iv->var->name_hint);
          if (const auto* extent = attr->value.as<IntImmNode>()) {
            thread_index_var_static_extents_[iv->var.get()] = extent->value;
          }
        } else if (std::string(iv->thread_tag).rfind("blockIdx.", 0) == 0) {
          block_index_vars_.insert(iv->var.get());
          block_index_var_names_.insert(iv->var->name_hint);
        }
      }
      return;
    }
    if (const auto* loop = node.as<ForNode>()) {
      if (const auto* extent = loop->extent.as<IntImmNode>()) {
        loop_var_static_extents_[loop->loop_var.get()] = extent->value;
      }
    }
  });
  current_segment_kind_.clear();
  read_accessor_slots_.clear();
  write_accessor_slots_.clear();
  gemm_a_buffer_ = Buffer();
  gemm_b_buffer_ = Buffer();
  gemm_c_buffer_ = Buffer();
  gemm_a_buffer_name_.clear();
  gemm_b_buffer_name_.clear();
  gemm_c_buffer_name_.clear();
  gemm_c_scope_.clear();
  gemm_has_mbarrier_ = false;
  gemm_mbarrier_buffer_ = Buffer();
  gemm_mbarrier_buffer_name_.clear();
  gemm_mbarrier_scope_.clear();
  gemm_mbarrier_index_exprs_.clear();
  gemm_a_req_index_ = -1;
  gemm_b_req_index_ = -1;
  gemm_c_req_index_ = -1;
  gemm_m_ = 0;
  gemm_n_ = 0;
  gemm_k_ = 0;
  compute_op_signatures_.clear();
  gemm_compute_op_fact_index_by_signature_.clear();
  gemm_compute_op_facts_.clear();
  tt_compute_op_plans_.clear();
  tile_compute_dag_lowering_decisions_.clear();
  tile_compute_dag_lowering_decision_consumed_.clear();
  active_tile_compute_dag_lowering_decision_.reset();
  logical_tile_layout_specs_by_buffer_.clear();
  spatial_materialization_boundaries_.clear();
  spatial_materialization_boundary_position_by_index_.clear();
  buffer_materialization_facts_by_target_buffer_.clear();
  gemm_input_buffer_num_tiles_.clear();
  gemm_transpose_a_ = false;
  gemm_transpose_b_ = false;
  gemm_policy_type_ = 0;
  gemm_clear_accum_ = false;
  gemm_k_pack_ = 1;
  gemm_wg_wait_ = 0;
  gemm_dst_full_sync_en_ = false;
  gemm_bfp8_pack_precise_ = false;
  gemm_defines_.clear();
  gemm_named_compile_args_.clear();
  gemm_a_dtype_ = DataType::Void();
  gemm_b_dtype_ = DataType::Void();
  gemm_c_dtype_ = DataType::Void();
  LoadSeededCBRequirements(func);
  LoadSeededComputeOpPlans(func);
  auto maybe_spatial_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  ICHECK(maybe_spatial_plan)
      << "PlanTTKernelABI requires tl.spatial_plan; run BuildSpatialPlan before lowering";
  LoadSpatialLiveValueBoundaries(maybe_spatial_plan.value());
  const BlackholeLoweringSupportFacts lowering_support_facts =
      BuildLoweringSupportFactsFromAnalysis(func);
  LoadLogicalBufferShapes(func, lowering_support_facts, maybe_spatial_plan.value());
  ValidateComputePipelineLegalityFromBody(func->body);
  requires_compute_segment_ = HasComputeSegmentRequirement(func->body);
  LoadLogicalTileLayoutSpecs(maybe_spatial_plan.value());
  buffer_materialization_facts_by_target_buffer_ =
      BuildBufferMaterializationFactMap(
          lowering_support_facts.buffer_materialization_facts);
  LoadTileComputeDAGLoweringPlan(func);
  LoadBufferFlowFacts(lowering_support_facts);
  stmt_order_index_by_node_ = BuildExecutionOrderIndexByStmtNode(func->body);
  const std::vector<std::string> expected_unsupported_ops =
      CollectLeafUnsupportedComputeOpsFromBody(func->body);
  // Pre-scan: register GEMM CB requirements first so their indices are stable
  // when copy stmts are visited.
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (!gemm_a_buffer_name_.empty()) {
      return;
    }
    const auto* call = node.as<CallNode>();
    if (call && IsMatmulCall(call)) {
      ExtractGemmInfo(call);
    }
  });

  // Pre-scan all GEMM calls and record blackhole.acc buffers that will later be
  // consumed through CB wait/pop semantics. Their local producers must publish
  // the reserved CB pages before the matmul sequence can make progress.
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (!call || !IsMatmulCall(call) || call->args.size() < 8) {
      return;
    }
    int m_tiles = 1;
    int n_tiles = 1;
    int k_tiles = 1;
    if (const auto* m_imm = call->args[5].as<IntImmNode>()) {
      m_tiles = CeilDivToInt(m_imm->value, kBlackholeTileRows);
    }
    if (const auto* n_imm = call->args[6].as<IntImmNode>()) {
      n_tiles = CeilDivToInt(n_imm->value, kBlackholeTileCols);
    }
    if (const auto* k_imm = call->args[7].as<IntImmNode>()) {
      k_tiles = CeilDivToInt(k_imm->value, kBlackholeTileCols);
    }
    auto record_gemm_input_tiles = [&](const PrimExpr& expr, int tile_count) {
      if (!IsBufferLikeExpr(expr)) {
        return;
      }
      tir::BufferRegion region = NormalizeToBufferRegion(expr);
      const std::string buffer_identity = BufferIdentityName(region->buffer);
      auto it = gemm_input_buffer_num_tiles_.find(buffer_identity);
      if (it == gemm_input_buffer_num_tiles_.end()) {
        gemm_input_buffer_num_tiles_[buffer_identity] = tile_count;
        return;
      }
      ICHECK_EQ(it->second, tile_count)
          << "PlanTTKernelABI requires a stable GEMM input tile contract per logical "
             "buffer identity; "
          << buffer_identity << " was seen with both " << it->second << " and " << tile_count
          << " tiles";
    };
    auto record_if_cb_consumed_fragment = [&](const PrimExpr& expr, int tile_count) {
      if (!IsBufferLikeExpr(expr)) {
        return;
      }
      tir::BufferRegion region = NormalizeToBufferRegion(expr);
      if (GetStorageScope(region->buffer) == "blackhole.acc") {
        const std::string buffer_identity = BufferIdentityName(region->buffer);
        auto it = cb_consumed_compute_input_pages_by_buffer_identity_.find(buffer_identity);
        if (it == cb_consumed_compute_input_pages_by_buffer_identity_.end()) {
          cb_consumed_compute_input_pages_by_buffer_identity_[buffer_identity] = tile_count;
        } else {
          it->second = std::max(it->second, tile_count);
        }
        cb_consumed_compute_input_use_count_by_buffer_identity_[buffer_identity] += 1;
      }
    };
    record_gemm_input_tiles(call->args[0], m_tiles * k_tiles);
    record_gemm_input_tiles(call->args[1], k_tiles * n_tiles);
    record_if_cb_consumed_fragment(call->args[0], m_tiles * k_tiles);
    record_if_cb_consumed_fragment(call->args[1], k_tiles * n_tiles);
  });

  gemm_compute_op_known_buffers_.assign(gemm_compute_op_facts_.size(), {});
  for (size_t i = 0; i < gemm_compute_op_facts_.size(); ++i) {
    const GemmComputeOpFact& fact = gemm_compute_op_facts_[i];
    auto maybe_insert = [&](const std::string& buffer) {
      if (!buffer.empty()) {
        gemm_compute_op_known_buffers_[i].insert(buffer);
      }
    };
    maybe_insert(fact.a_buffer);
    maybe_insert(fact.b_buffer);
    maybe_insert(fact.c_buffer);
  }

  // Transform the function body. Segment markers remain pass-local mechanics
  // until we have derived TTProgram slice metadata and CB depth.
  Stmt body_with_segment_markers = VisitStmt(func->body);
  UpdateCBRequirementDepthsFromLoweredBody(&cb_requirements_, body_with_segment_markers,
                                           gemm_a_buffer_name_.empty() ? "fused_dataflow"
                                                                       : "compute");
  std::vector<std::string> unresolved_unsupported_ops;
  std::unordered_set<std::string> unresolved_unsupported_seen;
  auto push_unresolved = [&](const char* op_name) {
    if (unresolved_unsupported_seen.insert(op_name).second) {
      unresolved_unsupported_ops.push_back(op_name);
    }
  };
  for (const std::string& op_name : expected_unsupported_ops) {
    if (op_name == "broadcast" &&
        HasResidualScalarLoadBroadcast(body_with_segment_markers)) {
      push_unresolved("broadcast");
      continue;
    }
    if (op_name == "fill" && HasResidualFragmentFill(body_with_segment_markers)) {
      push_unresolved("fill");
      continue;
    }
    if (op_name == "max" && HasResidualFragmentMax(body_with_segment_markers)) {
      push_unresolved("max");
      continue;
    }
    if (op_name == "add" && HasResidualFragmentAdd(body_with_segment_markers)) {
      push_unresolved("add");
      continue;
    }
    if (op_name == "cast" && HasResidualFragmentCast(body_with_segment_markers)) {
      push_unresolved("cast");
      continue;
    }
  }
  // Store TTProgram slice metadata while pass-local segment markers still exist.
  PrimFunc staged_func = func;
  staged_func.CopyOnWrite()->body = body_with_segment_markers;
  StoreSegmentPlan(staged_func);

  // Create the final function body without cross-pass segment markers.
  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = StripSegmentKindMarkers(body_with_segment_markers);
  StoreAccessorDescriptors(new_func);
  RejectUnsupportedComputeOps(unresolved_unsupported_ops);

  if (unresolved_unsupported_ops.empty()) {
    ValidateNoResidualComputeRegionStores(body_with_segment_markers);
  }

  return new_func;
}

// Get CB configuration from function attributes
PlanTTKernelABI::CBConfig PlanTTKernelABI::GetCBConfig() const {
  CBConfig config;

  // Try to get CB configuration from function attributes
  if (auto cb_in0 = current_func_->GetAttr<Integer>("tl_cb_in0")) {
    config.in0_id = cb_in0.value()->value;
  }
  if (auto cb_in1 = current_func_->GetAttr<Integer>("tl_cb_in1")) {
    config.in1_id = cb_in1.value()->value;
  }
  if (auto cb_out = current_func_->GetAttr<Integer>("tl_cb_out")) {
    config.out_id = cb_out.value()->value;
  }
  if (auto k_tiles = current_func_->GetAttr<Integer>("tl_k_tiles")) {
    config.num_k_tiles = k_tiles.value()->value;
  }

  return config;
}

int PlanTTKernelABI::AllocateRequirementIndex(const Buffer& buffer, CBType type) {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto bind_existing_requirement = [&](int requirement_index) {
    buffer_to_req_[buffer] = requirement_index;
    buffer_data_to_req_index_[buffer->data.get()] = requirement_index;
    buffer_identity_to_req_index_[buffer_identity] = requirement_index;

    auto& req = cb_requirements_.at(requirement_index);
    if (req.type == type) {
      return requirement_index;
    }
    if (req.type == CBType::kIntermediate && type != CBType::kIntermediate) {
      req.type = type;
      return requirement_index;
    }
    if (type == CBType::kIntermediate) {
      return requirement_index;
    }
    ICHECK(req.type == type)
        << "PlanTTKernelABI requires one CB type per logical buffer identity; "
        << buffer_identity << " was assigned both " << static_cast<int>(req.type)
        << " and " << static_cast<int>(type);
    return requirement_index;
  };

  auto it = buffer_to_req_.find(buffer);
  if (it != buffer_to_req_.end()) {
    return bind_existing_requirement(it->second);
  }
  auto by_data = buffer_data_to_req_index_.find(buffer->data.get());
  if (by_data != buffer_data_to_req_index_.end()) {
    return bind_existing_requirement(by_data->second);
  }
  auto by_identity = buffer_identity_to_req_index_.find(buffer_identity);
  if (by_identity != buffer_identity_to_req_index_.end()) {
    return bind_existing_requirement(by_identity->second);
  }

  const int requirement_index = next_requirement_index_++;
  buffer_to_req_[buffer] = requirement_index;
  buffer_data_to_req_index_[buffer->data.get()] = requirement_index;
  buffer_identity_to_req_index_[buffer_identity] = requirement_index;

  CBRequirement req;
  req.name = buffer_identity;
  req.type = type;
  req.lifetime_begin = requirement_index;
  req.lifetime_end = req.lifetime_begin;

  // Calculate page size from the logical buffer shape. This preserves fragment
  // tile counts even when the lowered TIR buffer handle has been scalarized or
  // flattened for pointwise codegen.
  const int64_t total_elements = GetLogicalBufferElementCount(buffer);
  const int total_bytes = static_cast<int>(total_elements * buffer->dtype.bytes());
  req.page_size = EstimateCopyPageSize(buffer);
  req.num_pages = std::max(
      2, req.page_size > 0 ? (total_bytes + req.page_size - 1) / req.page_size : 2);

  // Keep generic CB requirements on the same dtype->format contract as the
  // contract-specialized paths so bfloat16/uint payloads do not inherit the
  // CBRequirement default format by accident.
  req.data_format = DataTypeToDataFormatForBlackhole(buffer->dtype);

  if (!buffer_identity.empty()) {
    auto flow_fact_it = buffer_flow_facts_.find(buffer_identity);
    if (flow_fact_it != buffer_flow_facts_.end()) {
      req.flow_class = flow_fact_it->second.flow_class;
      if (flow_fact_it->second.publish_pages_per_event > 0) {
        req.publish_pages_per_event =
            std::max(req.publish_pages_per_event,
                     flow_fact_it->second.publish_pages_per_event);
      }
      if (flow_fact_it->second.consume_pages_per_event > 0) {
        req.consume_pages_per_event =
            std::max(req.consume_pages_per_event,
                     flow_fact_it->second.consume_pages_per_event);
      }
    }
  }

  cb_requirements_.push_back(req);
  return requirement_index;
}

void PlanTTKernelABI::SetRequirementPageLayout(int requirement_index, int page_size,
                                                 int num_pages) {
  ICHECK_GE(requirement_index, 0);
  ICHECK_LT(requirement_index, static_cast<int>(cb_requirements_.size()));
  auto& req = cb_requirements_[requirement_index];
  req.page_size = page_size;
  req.num_pages = num_pages;
}

void PlanTTKernelABI::MarkRequirementLifetimeOverlap(int lhs_requirement_index,
                                                       int rhs_requirement_index) {
  ICHECK_GE(lhs_requirement_index, 0);
  ICHECK_LT(lhs_requirement_index, static_cast<int>(cb_requirements_.size()));
  ICHECK_GE(rhs_requirement_index, 0);
  ICHECK_LT(rhs_requirement_index, static_cast<int>(cb_requirements_.size()));
  const int overlap_begin = std::min(cb_requirements_[lhs_requirement_index].lifetime_begin,
                                     cb_requirements_[rhs_requirement_index].lifetime_begin);
  const int overlap_end = std::max(cb_requirements_[lhs_requirement_index].lifetime_end,
                                   cb_requirements_[rhs_requirement_index].lifetime_end);
  cb_requirements_[lhs_requirement_index].lifetime_begin = overlap_begin;
  cb_requirements_[lhs_requirement_index].lifetime_end = overlap_end;
  cb_requirements_[rhs_requirement_index].lifetime_begin = overlap_begin;
  cb_requirements_[rhs_requirement_index].lifetime_end = overlap_end;
}

Array<TTCBPlan> PlanTTKernelABI::GetStagedCBPlans() const {
  Array<TTCBPlan> staged_cb_plans;
  for (size_t i = 0; i < cb_requirements_.size(); ++i) {
    const auto& req = cb_requirements_[i];
    const int64_t lifetime_begin = req.lifetime_begin;
    const int64_t lifetime_end = std::max(req.lifetime_begin, req.lifetime_end);
    const char* role = req.type == CBType::kInput ? "input"
                        : req.type == CBType::kOutput ? "output"
                                                      : "intermediate";
    // Until PlanTTCBAlloc assigns hardware ids, cb_id carries the dense
    // requirement slot already referenced by the lowered IR.
    staged_cb_plans.push_back(TTCBPlan(String(req.name), static_cast<int64_t>(i), String(role),
                                       req.num_pages, req.page_size, String(req.data_format),
                                       req.initial_reserve_pages,
                                       String(CBFlowClassToString(req.flow_class)),
                                       req.publish_pages_per_event,
                                       req.consume_pages_per_event, lifetime_begin,
                                       lifetime_end, Array<String>{}, Array<Integer>{}));
  }
  return staged_cb_plans;
}

void PlanTTKernelABI::LoadSeededCBRequirements(const PrimFunc& func) {
  auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  if (!staged_program) {
    return;
  }
  for (const TTCBPlan& staged_cb_plan : staged_program.value()->cb_plans) {
    CBRequirement req;
    const int req_index = static_cast<int>(staged_cb_plan->cb_id);
    ICHECK_EQ(req_index, static_cast<int>(cb_requirements_.size()))
        << "PlanTTKernelABI requires staged TTProgram cb_plans to preserve dense requirement "
           "slot ordering";
    req.name = static_cast<std::string>(staged_cb_plan->name);
    const std::string role = static_cast<std::string>(staged_cb_plan->resource_class);
    if (role == "input") {
      req.type = CBType::kInput;
    } else if (role == "output") {
      req.type = CBType::kOutput;
    } else {
      req.type = CBType::kIntermediate;
    }
    req.page_size = static_cast<int>(staged_cb_plan->page_size_bytes);
    req.num_pages = static_cast<int>(staged_cb_plan->num_pages);
    req.data_format = static_cast<std::string>(staged_cb_plan->data_format);
    req.initial_reserve_pages = static_cast<int>(staged_cb_plan->initial_reserve_pages);
    req.flow_class =
        ParseCBFlowClass(static_cast<std::string>(staged_cb_plan->flow_class))
            .value_or(CBFlowClass::kState);
    req.publish_pages_per_event = static_cast<int>(staged_cb_plan->publish_pages_per_event);
    req.consume_pages_per_event = static_cast<int>(staged_cb_plan->consume_pages_per_event);
    req.lifetime_begin = static_cast<int>(staged_cb_plan->lifetime_begin);
    req.lifetime_end = static_cast<int>(staged_cb_plan->lifetime_end);
    if (req.lifetime_end < req.lifetime_begin) {
      std::swap(req.lifetime_begin, req.lifetime_end);
    }

    cb_requirements_.push_back(req);
    if (!req.name.empty()) {
      buffer_identity_to_req_index_[req.name] = req_index;
      seeded_cb_requirement_names_.insert(req.name);
    }
  }
  next_requirement_index_ =
      std::max(next_requirement_index_, static_cast<int>(cb_requirements_.size()));
}

// Detect matmul operation using Op comparison
// Detect clear operation using Op comparison
bool PlanTTKernelABI::IsClearOperation(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.clear";
}

// Detect copy operation using buffer scopes
// Determine copy direction
static bool IsPureCopyLoopNest(const Stmt& stmt) {
  if (const auto* loop = stmt.as<ForNode>()) {
    return IsPureCopyLoopNest(loop->body);
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    return IsPureCopyLoopNest(attr->body);
  }
  if (const auto* allocate = stmt.as<AllocateNode>()) {
    return IsPureCopyLoopNest(allocate->body);
  }
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    if (seq->seq.empty()) {
      return false;
    }
    for (const Stmt& child : seq->seq) {
      if (!IsPureCopyLoopNest(child)) {
        return false;
      }
    }
    return true;
  }
  if (const auto* store = stmt.as<BufferStoreNode>()) {
    return store->value.as<BufferLoadNode>() != nullptr;
  }
  return false;
}

void PlanTTKernelABI::RejectUnsupportedComputeOps(const std::vector<std::string>& unsupported_ops) {
  if (!unsupported_ops.empty()) {
    std::ostringstream os;
    for (const std::string& op_name : unsupported_ops) {
      if (!os.str().empty()) {
        os << ", ";
      }
      os << op_name;
    }
    ICHECK(false) << "PlanTTCompute requires exact TT-Metal builtin legality before TTProgram; "
                  << "unsupported_compute_ops remain: " << os.str();
  }
}

Stmt PlanTTKernelABI::GenerateClearSequence(const CallNode* op) {
  // Clear operation: tile_regs_acquire() to zero DST registers
  // In full implementation, would also zero-fill
  return MaybeWrapComputeSegment(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
}

namespace {

bool IsRowScalarLocalFragmentBuffer(const Buffer& buffer) {
  if (!IsUnsupportedResidualLocalScope(buffer)) {
    return false;
  }
  return buffer->shape.size() == 1;
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

bool IsScalarLiteralValue(const PrimExpr& expr) {
  return expr.as<FloatImmNode>() || expr.as<IntImmNode>();
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

bool IsFragmentFillValue(const PrimExpr& expr) {
  return IsScalarLiteralValue(expr) || IsNegInfValue(expr) || IsZeroValue(expr);
}

bool HasResidualFragmentFill(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer) ||
        !IsFragmentFillValue(store->value)) {
      return;
    }
    if (store->indices.size() == 1) {
      found = true;
    }
  });
  return found;
}

bool HasResidualFragmentAdd(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer)) {
      return;
    }
    if (store->value.as<AddNode>()) {
      found = true;
    }
  });
  return found;
}

bool HasResidualFragmentMax(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer)) {
      return;
    }
    if (store->value.as<MaxNode>()) {
      found = true;
    }
  });
  return found;
}

bool HasResidualFragmentCast(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer)) {
      return;
    }
    if (store->value.as<CastNode>()) {
      found = true;
    }
  });
  return found;
}

bool HasResidualScalarLoadBroadcast(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsVectorLocalFragmentBuffer(store->buffer) || store->indices.size() != 1) {
      return;
    }
    PrimExpr grouped_scalar_index;
    auto is_grouped_scalar = [&](const PrimExpr& expr) {
      const auto* load = expr.as<BufferLoadNode>();
      if (!load || !IsRowScalarLocalFragmentBuffer(load->buffer) || load->indices.size() != 1) {
        return false;
      }
      const auto* floordiv = load->indices[0].as<FloorDivNode>();
      return floordiv && floordiv->a.same_as(store->indices[0]);
    };
    if (const auto* mul = store->value.as<MulNode>()) {
      if (is_grouped_scalar(mul->a) || is_grouped_scalar(mul->b)) {
        found = true;
      }
      return;
    }
    if (const auto* div = store->value.as<DivNode>()) {
      if (is_grouped_scalar(div->b)) {
        found = true;
      }
      return;
    }
    if (const auto* call = store->value.as<CallNode>();
        call && call->op->IsInstance<OpNode>() && call->args.size() == 1 &&
        Downcast<Op>(call->op)->name == "tir.exp2") {
      if (const auto* sub = call->args[0].as<SubNode>()) {
        if (is_grouped_scalar(sub->a) || is_grouped_scalar(sub->b)) {
          found = true;
        }
      }
    }
  });
  return found;
}

}  // namespace

// Parse a colon-separated string into fields
Stmt PlanTTKernelABI::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == kBlackholeExactOutputLiveCBAttr) {
    Stmt body = VisitStmt(op->body);
    const auto* cb_id = op->value.as<IntImmNode>();
    const auto* data = op->node.as<VarNode>();
    if (cb_id != nullptr && data != nullptr) {
      auto buffer_it = compute_physical_buffers_by_data_.find(data);
      if (buffer_it != compute_physical_buffers_by_data_.end() && buffer_it->second.defined()) {
        ExactTiledCBValue live_value;
        live_value.buffer = buffer_it->second;
        live_value.cb_id = static_cast<int>(cb_id->value);
        live_value.borrowed_live = true;
        PopulateExactTiledCBValueShape(buffer_it->second, &live_value);
        RecordExactOutputLiveForm(buffer_it->second, live_value);
      }
    }
    return body;
  }
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    const std::string thread_tag = iv->thread_tag;
    const bool zero_thread_var = thread_tag.rfind("threadIdx.", 0) == 0;
    const bool transport_thread_var = thread_tag.rfind("blockIdx.", 0) == 0;
    if (zero_thread_var) {
      thread_index_vars_.insert(iv->var.get());
      thread_index_var_names_.insert(iv->var->name_hint);
      if (const auto* extent = op->value.as<IntImmNode>()) {
        thread_index_var_static_extents_[iv->var.get()] = extent->value;
      }
    } else if (transport_thread_var) {
      block_index_vars_.insert(iv->var.get());
      block_index_var_names_.insert(iv->var->name_hint);
    }
    Stmt body = VisitStmt(op->body);
    if (zero_thread_var) {
      thread_index_vars_.erase(iv->var.get());
      thread_index_var_names_.erase(iv->var->name_hint);
    } else if (transport_thread_var) {
      block_index_vars_.erase(iv->var.get());
      block_index_var_names_.erase(iv->var->name_hint);
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }
  if (op->attr_key == "blackhole.segment_kind") {
    std::string previous_segment_kind = current_segment_kind_;
    if (const auto* kind = op->value.as<StringImmNode>()) {
      current_segment_kind_ = kind->value;
    }
    Stmt body = VisitStmt(op->body);
    current_segment_kind_ = previous_segment_kind;
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt PlanTTKernelABI::VisitStmt_(const DeclBufferNode* op) {
  if (select_compute_builtins_only_) {
    return StmtExprMutator::VisitStmt_(op);
  }
  if (GetStorageScope(op->buffer) == "blackhole.acc") {
    const int requirement_index = AllocateRequirementIndex(op->buffer, CBType::kIntermediate);
    auto& req = cb_requirements_.at(requirement_index);
    req.lifetime_begin = 0;
    req.lifetime_end = std::max(req.lifetime_end, next_requirement_index_);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt PlanTTKernelABI::VisitStmt_(const AllocateNode* op) {
  return StmtExprMutator::VisitStmt_(op);
}

Stmt PlanTTKernelABI::VisitStmt_(const SeqStmtNode* op) {
  Array<Stmt> rewritten;
  struct LoweringOrderGuard {
    int* slot;
    int previous;
    LoweringOrderGuard(int* slot, int value) : slot(slot), previous(*slot) {
      *slot = value;
    }
    ~LoweringOrderGuard() {
      *slot = previous;
    }
  };
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const auto order_it = stmt_order_index_by_node_.find(op->seq[i].get());
    const int current_order_index =
        order_it != stmt_order_index_by_node_.end() ? order_it->second : static_cast<int>(i);
    LoweringOrderGuard order_guard(&current_lowering_order_index_, current_order_index);
    auto try_lower_retained_matmul = [&](const Stmt& stmt,
                                         const FragmentCastMatch* post_merge_cast,
                                         int post_merge_cast_order_index,
                                         Stmt* lowered,
                                         bool* consumed_post_merge_cast) -> bool {
      std::vector<std::function<Stmt(Stmt)>> rewrap_stack;
      Stmt current = stmt;
      while (true) {
        if (const auto* attr = current.as<AttrStmtNode>()) {
          rewrap_stack.push_back(
              [node = attr->node, attr_key = attr->attr_key, value = attr->value](Stmt body) {
                return AttrStmt(node, attr_key, value, body);
              });
          current = attr->body;
          continue;
        }
        if (const auto* let = current.as<LetStmtNode>()) {
          rewrap_stack.push_back([var = let->var, value = let->value](Stmt body) {
            return LetStmt(var, value, body);
          });
          current = let->body;
          continue;
        }
        if (const auto* decl = current.as<DeclBufferNode>()) {
          rewrap_stack.push_back([buffer = decl->buffer](Stmt body) {
            return DeclBuffer(buffer, body);
          });
          current = decl->body;
          continue;
        }
        if (const auto* allocate = current.as<AllocateNode>()) {
          rewrap_stack.push_back([buffer_var = allocate->buffer_var, dtype = allocate->dtype,
                                  extents = allocate->extents, condition = allocate->condition,
                                  annotations = allocate->annotations](Stmt body) {
            return Allocate(buffer_var, dtype, extents, condition, body, annotations);
          });
          current = allocate->body;
          continue;
        }
        break;
      }
      const auto* eval = current.as<EvaluateNode>();
      if (!eval) {
        return false;
      }
      if (const auto* call = eval->value.as<CallNode>()) {
        if (IsMatmulCall(call)) {
          Stmt matmul = LowerMatmulCallWithFlowAnalysis(call, current_order_index,
                                                        post_merge_cast,
                                                        post_merge_cast_order_index,
                                                        consumed_post_merge_cast);
          for (auto it = rewrap_stack.rbegin(); it != rewrap_stack.rend(); ++it) {
            matmul = (*it)(matmul);
          }
          *lowered = matmul;
          return true;
        }
      }
      return false;
    };

    auto unwrap_call = [](const Stmt& stmt) -> const CallNode* {
      Stmt current = stmt;
      while (true) {
        if (const auto* attr = current.as<AttrStmtNode>()) {
          current = attr->body;
          continue;
        }
        if (const auto* let = current.as<LetStmtNode>()) {
          current = let->body;
          continue;
        }
        if (const auto* decl = current.as<DeclBufferNode>()) {
          current = decl->body;
          continue;
        }
        if (const auto* allocate = current.as<AllocateNode>()) {
          current = allocate->body;
          continue;
        }
        break;
      }
      const auto* eval = current.as<EvaluateNode>();
      return eval ? eval->value.as<CallNode>() : nullptr;
    };

    struct ExplicitFragmentFill {
      Buffer dst;
      PrimExpr value;
      PrimExpr num_elements;
      const VarNode* data = nullptr;
    };
    auto match_explicit_fragment_fill =
        [&](const Stmt& stmt, ExplicitFragmentFill* fill) -> bool {
      if (!fill) {
        return false;
      }
      const CallNode* call = unwrap_call(stmt);
      if (!call || !call->op->IsInstance<OpNode>()) {
        return false;
      }
      const Op call_op = Downcast<Op>(call->op);
      if (call_op->name == blackhole_tile_compute_schema::kOpName && call->args.size() >= 4U) {
        const auto* operation = call->args[0].as<StringImmNode>();
        if (!operation || operation->value != blackhole_tile_compute_schema::kFillTile ||
            !IsBufferLikeExpr(call->args[1]) || !IsFragmentFillValue(call->args[2])) {
          return false;
        }
        fill->dst = NormalizeToBufferRegion(call->args[1])->buffer;
        fill->value = call->args[2];
        fill->num_elements = call->args[3];
        fill->data = BufferDataIdentity(fill->dst);
        return true;
      }
      if (call_op->name != "tl.blackhole.fill_fragment" || call->args.size() < 3U ||
          !IsFragmentFillValue(call->args[2])) {
        return false;
      }
      fill->data = call->args[0].as<VarNode>();
      if (!fill->data) {
        return false;
      }
      auto it = compute_physical_buffers_by_data_.find(fill->data);
      if (it != compute_physical_buffers_by_data_.end()) {
        fill->dst = it->second;
      }
      fill->value = call->args[2];
      fill->num_elements = call->args[1];
      return true;
    };

    auto is_redundant_zero_fill_before_full_overwrite_matmul =
        [&](const Stmt& fill_stmt, const Stmt& next_stmt) -> bool {
      ExplicitFragmentFill fill_match;
      if (!match_explicit_fragment_fill(fill_stmt, &fill_match) ||
          !IsLiteralZeroValue(fill_match.value)) {
        return false;
      }

      Stmt current = next_stmt;
      while (true) {
        if (const auto* attr = current.as<AttrStmtNode>()) {
          current = attr->body;
          continue;
        }
        if (const auto* let = current.as<LetStmtNode>()) {
          current = let->body;
          continue;
        }
        if (const auto* decl = current.as<DeclBufferNode>()) {
          current = decl->body;
          continue;
        }
        if (const auto* allocate = current.as<AllocateNode>()) {
          current = allocate->body;
          continue;
        }
        break;
      }

      const auto* eval = current.as<EvaluateNode>();
      const auto* call = eval ? eval->value.as<CallNode>() : nullptr;
      if (!call || !IsMatmulCall(call) || !IsBufferLikeExpr(call->args[2])) {
        return false;
      }

      const Buffer fill_buffer = ResolvePhysicalComputeBuffer(fill_match.dst);
      const Buffer out_buffer =
          ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(call->args[2])->buffer);
      if (fill_match.dst.defined() && !SameBufferIdentity(fill_buffer, out_buffer)) {
        return false;
      }
      if (!fill_match.dst.defined() && fill_match.data != BufferDataIdentity(out_buffer)) {
        return false;
      }
      if (FindBufferMaterializationFact(out_buffer) != nullptr) {
        return false;
      }
      const auto next_order_it = stmt_order_index_by_node_.find(next_stmt.get());
      const int next_order_index =
          next_order_it != stmt_order_index_by_node_.end()
              ? next_order_it->second
              : current_order_index + 1;
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(out_buffer, next_order_index);
      return !future_uses.has_compute_consume && !future_uses.has_reference;
    };

    auto is_redundant_identity_fill_before_clear_row_reduce =
        [&](const Stmt& fill_stmt, const Stmt& next_stmt) -> bool {
      ExplicitFragmentFill fill_match;
      if (!match_explicit_fragment_fill(fill_stmt, &fill_match)) {
        return false;
      }

      Stmt current = next_stmt;
      while (true) {
        if (const auto* attr = current.as<AttrStmtNode>()) {
          current = attr->body;
          continue;
        }
        if (const auto* let = current.as<LetStmtNode>()) {
          current = let->body;
          continue;
        }
        if (const auto* decl = current.as<DeclBufferNode>()) {
          current = decl->body;
          continue;
        }
        if (const auto* allocate = current.as<AllocateNode>()) {
          current = allocate->body;
          continue;
        }
        break;
      }
      const auto* eval = current.as<EvaluateNode>();
      const auto* call = eval ? eval->value.as<CallNode>() : nullptr;
      if (call && call->op->IsInstance<OpNode>()) {
        const Op call_op = Downcast<Op>(call->op);
        if (call_op->name == "tl.tileop.reduce" && call->args.size() >= 5U &&
            fill_match.dst.defined() && IsBufferLikeExpr(call->args[1])) {
          const auto* kind_imm = call->args[2].as<StringImmNode>();
          const Buffer reduce_dst = NormalizeToBufferRegion(call->args[1])->buffer;
          const bool identity_fill =
              kind_imm != nullptr &&
              ((kind_imm->value == "sum" && IsZeroValue(fill_match.value)) ||
               (kind_imm->value == "max" && IsNegInfValue(fill_match.value)));
          if (identity_fill && SameBufferIdentity(fill_match.dst, reduce_dst)) {
            ClearSelectedSourceLiveProducer(fill_match.dst);
            ClearTiledCBLiveFormAliases(fill_match.dst);
            for (const std::string& identity : CollectBufferFlowIdentities(fill_match.dst)) {
              last_fragment_fill_value_by_buffer_identity_[identity] = fill_match.value;
            }
            if (const VarNode* data = BufferDataIdentity(fill_match.dst)) {
              last_fragment_fill_value_by_data_[data] = fill_match.value;
            }
            if (fill_match.data != nullptr) {
              last_fragment_fill_value_by_data_[fill_match.data] = fill_match.value;
            }
            return true;
          }
        }
      }
      RowReductionMatch reduce_match;
      if (!call || !MatchExplicitTileReduce(call, &reduce_match) ||
          !(reduce_match.clear || reduce_match.accumulate_existing)) {
        return false;
      }
      Buffer reduce_logical_dst;
      if (call->args.size() >= 2U && IsBufferLikeExpr(call->args[1])) {
        reduce_logical_dst = NormalizeToBufferRegion(call->args[1])->buffer;
      }

      const Buffer fill_buffer = ResolvePhysicalComputeBuffer(fill_match.dst);
      const Buffer reduce_buffer = ResolvePhysicalComputeBuffer(reduce_match.dst);
      if (fill_match.dst.defined()) {
        const bool same_target =
            SameBufferIdentity(fill_match.dst, reduce_match.dst) ||
            (reduce_logical_dst.defined() &&
             SameBufferIdentity(fill_match.dst, reduce_logical_dst)) ||
            SameBufferIdentity(fill_buffer, reduce_buffer);
        if (!same_target) {
          return false;
        }
      }
      if (!fill_match.dst.defined() && fill_match.data != BufferDataIdentity(reduce_buffer)) {
        return false;
      }
      const bool identity_fill =
          (reduce_match.kind == "sum" && IsZeroValue(fill_match.value)) ||
          (reduce_match.kind == "max" && IsNegInfValue(fill_match.value));
      if (!identity_fill) {
        return false;
      }
      if (fill_match.dst.defined()) {
        ClearSelectedSourceLiveProducer(fill_match.dst);
        ClearTiledCBLiveFormAliases(fill_match.dst);
        for (const std::string& identity : CollectBufferFlowIdentities(fill_match.dst)) {
          last_fragment_fill_value_by_buffer_identity_[identity] = fill_match.value;
        }
        if (const VarNode* data = BufferDataIdentity(fill_match.dst)) {
          last_fragment_fill_value_by_data_[data] = fill_match.value;
        }
      }
      if (fill_match.data != nullptr) {
        last_fragment_fill_value_by_data_[fill_match.data] = fill_match.value;
      }
      return true;
    };

    auto is_initial_identity_state_copy_before_clear_row_reduce =
        [&](const Stmt& copy_stmt, const Stmt& fill_stmt, const Stmt& reduce_stmt,
            const Stmt& combine_stmt) -> bool {
      auto blackhole_compute_operation = [](const CallNode* call) -> const StringImmNode* {
        if (!call || !call->op->IsInstance<OpNode>() || call->args.empty()) {
          return nullptr;
        }
        const Op call_op = Downcast<Op>(call->op);
        if (call_op->name != blackhole_tile_compute_schema::kOpName) {
          return nullptr;
        }
        return call->args[0].as<StringImmNode>();
      };
      auto buffer_arg = [&](const CallNode* call, size_t index) -> Buffer {
        if (!call || index >= call->args.size() || !IsBufferLikeExpr(call->args[index])) {
          return Buffer();
        }
        return ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(call->args[index])->buffer);
      };

      const CallNode* copy_call = unwrap_call(copy_stmt);
      const StringImmNode* copy_op = blackhole_compute_operation(copy_call);
      if (!copy_op || copy_op->value != blackhole_tile_compute_schema::kCopyTile ||
          copy_call->args.size() < 4U) {
        return false;
      }
      const Buffer copy_src = buffer_arg(copy_call, 1);
      const Buffer copy_dst = buffer_arg(copy_call, 2);
      if (!copy_src.defined() || !copy_dst.defined()) {
        return false;
      }

      const CallNode* fill_call = unwrap_call(fill_stmt);
      const StringImmNode* fill_op = blackhole_compute_operation(fill_call);
      if (!fill_op || fill_op->value != blackhole_tile_compute_schema::kFillTile ||
          fill_call->args.size() < 4U) {
        return false;
      }
      const Buffer fill_dst = buffer_arg(fill_call, 1);
      const PrimExpr fill_value = fill_call->args[2];
      if (!fill_dst.defined() || !SameBufferIdentity(copy_src, fill_dst)) {
        return false;
      }

      const CallNode* reduce_call = unwrap_call(reduce_stmt);
      RowReductionMatch reduce_match;
      if (!reduce_call || !MatchExplicitTileReduce(reduce_call, &reduce_match) ||
          !SameBufferIdentity(copy_src, reduce_match.dst)) {
        return false;
      }
      const bool identity_fill =
          (reduce_match.kind == "sum" && IsZeroValue(fill_value)) ||
          (reduce_match.kind == "max" && IsNegInfValue(fill_value));
      if (!identity_fill) {
        return false;
      }

      const CallNode* combine_call = unwrap_call(combine_stmt);
      const StringImmNode* combine_op = blackhole_compute_operation(combine_call);
      const char* expected_combine_op =
          reduce_match.kind == "sum" ? blackhole_tile_compute_schema::kAddTiles
                                     : blackhole_tile_compute_schema::kBinaryMaxTile;
      if (!combine_op || combine_op->value != expected_combine_op ||
          combine_call->args.size() < 3U) {
        return false;
      }
      const Buffer combine_dst = buffer_arg(combine_call, 1);
      const Buffer combine_rhs = buffer_arg(combine_call, 2);
      if (!combine_dst.defined() || !combine_rhs.defined() ||
          !SameBufferIdentity(copy_src, combine_dst) ||
          !SameBufferIdentity(copy_dst, combine_rhs)) {
        return false;
      }

      PrimExpr existing_fill;
      ExactTiledCBValue existing_live;
      if (TryGetLastFragmentFillValue(copy_src, &existing_fill) ||
          TryCreateLiveExactTiledCBValue(copy_src, &existing_live)) {
        return false;
      }

      ClearSelectedSourceLiveProducer(copy_dst);
      ClearTiledCBLiveFormAliases(copy_dst);
      for (const std::string& identity : CollectBufferFlowIdentities(copy_dst)) {
        last_fragment_fill_value_by_buffer_identity_[identity] = fill_value;
      }
      if (const VarNode* data = BufferDataIdentity(copy_dst)) {
        last_fragment_fill_value_by_data_[data] = fill_value;
      }
      return true;
    };

    if (!select_compute_builtins_only_) {
      Stmt retained_matmul;
      FragmentCastMatch post_merge_cast;
      const FragmentCastMatch* post_merge_cast_ptr = nullptr;
      int post_merge_cast_order_index = -1;
      if (i + 1 < op->seq.size()) {
        if (const auto* next_eval = op->seq[i + 1].as<EvaluateNode>()) {
          const auto* next_call = next_eval->value.as<CallNode>();
          if (next_call && MatchExplicitTileTypecast(next_call, &post_merge_cast)) {
            post_merge_cast_ptr = &post_merge_cast;
            const auto next_order_it = stmt_order_index_by_node_.find(next_eval);
            post_merge_cast_order_index =
                next_order_it != stmt_order_index_by_node_.end()
                    ? next_order_it->second
                    : static_cast<int>(i + 1);
            post_merge_cast_order_index = ResolveCurrentBufferTransferOrder(
                post_merge_cast.src, post_merge_cast.dst, post_merge_cast_order_index);
          }
        }
      }
      bool consumed_post_merge_cast = false;
      if (try_lower_retained_matmul(op->seq[i], post_merge_cast_ptr,
                                    post_merge_cast_order_index, &retained_matmul,
                                    &consumed_post_merge_cast)) {
        rewritten.push_back(retained_matmul);
        if (consumed_post_merge_cast) {
          ++i;
        }
        continue;
      }
    }
    if (!select_compute_builtins_only_ && i + 1 < static_cast<int>(op->seq.size()) &&
        is_redundant_zero_fill_before_full_overwrite_matmul(op->seq[i], op->seq[i + 1])) {
      continue;
    }
    if (i + 1 < static_cast<int>(op->seq.size()) &&
        is_redundant_identity_fill_before_clear_row_reduce(op->seq[i], op->seq[i + 1])) {
      continue;
    }
    if (i + 3 < static_cast<int>(op->seq.size()) &&
        is_initial_identity_state_copy_before_clear_row_reduce(
            op->seq[i], op->seq[i + 1], op->seq[i + 2], op->seq[i + 3])) {
      continue;
    }
    rewritten.push_back(VisitStmt(op->seq[i]));
  }
  return SeqStmt::Flatten(rewritten);
}

Stmt PlanTTKernelABI::VisitStmt_(const ForNode* op) {
  const bool zero_loop_var = !op->thread_binding.defined();
  const Var transport_loop_var = zero_loop_var ? op->loop_var : Var();
  if (!select_compute_builtins_only_ && IsPureCopyLoopNest(op->body)) {
    std::vector<Var> loop_stack;
    std::vector<NestedCopyMatch> matches;
    CollectNestedCopyStores(op->body, &loop_stack, &matches);
    if (!matches.empty()) {
      const NestedCopyMatch* dram_to_cb = nullptr;
      const NestedCopyMatch* cb_to_dram = nullptr;
      for (const auto& match : matches) {
        if (match.direction == CopyDirection::kDramToCB && !dram_to_cb) {
          dram_to_cb = &match;
        } else if (match.direction == CopyDirection::kCBToDram && !cb_to_dram) {
          cb_to_dram = &match;
        }
      }
      if (dram_to_cb && cb_to_dram) {
        saw_copy_op_ = true;
        std::vector<Var> loop_vars_to_zero = dram_to_cb->loop_vars;
        for (const auto& v : cb_to_dram->loop_vars) {
          if (std::find_if(loop_vars_to_zero.begin(), loop_vars_to_zero.end(),
                           [&](const Var& existing) { return existing.same_as(v); }) ==
              loop_vars_to_zero.end()) {
            loop_vars_to_zero.push_back(v);
          }
        }
        PrimExpr base_tile_index =
            InferStagedCopyBaseTileIndex(dram_to_cb->store, loop_vars_to_zero);
        return GenerateFusedStagedCopySequence(dram_to_cb->store, cb_to_dram->store,
                                               base_tile_index, loop_vars_to_zero);
      }

      bool all_staged_single_direction = true;
      std::vector<Stmt> lowered_matches;
      for (const auto& match : matches) {
        if (match.direction != CopyDirection::kDramToCB &&
            match.direction != CopyDirection::kCBToDram) {
          all_staged_single_direction = false;
          break;
        }
        std::vector<Var> loop_vars_to_zero;
        if (transport_loop_var.defined()) {
          loop_vars_to_zero.push_back(transport_loop_var);
        }
        loop_vars_to_zero.insert(loop_vars_to_zero.end(), match.loop_vars.begin(),
                                 match.loop_vars.end());
        PrimExpr base_tile_index =
            InferStagedCopyBaseTileIndex(match.store, loop_vars_to_zero);
        lowered_matches.push_back(
            GenerateStagedCopyLoopSequence(match.store, base_tile_index, loop_vars_to_zero));
      }
      if (all_staged_single_direction && !lowered_matches.empty()) {
        saw_copy_op_ = true;
        return SeqStmt::Flatten(lowered_matches);
      }
    }

    std::vector<Var> nested_loop_vars;
    if (const auto* nested_store = FindNestedCopyStore(op->body, &nested_loop_vars)) {
      CopyDirection direction = GetCopyDirection(nested_store);
      if (direction == CopyDirection::kDramToCB || direction == CopyDirection::kCBToDram) {
        saw_copy_op_ = true;
        std::vector<Var> loop_vars_to_zero;
        if (transport_loop_var.defined()) {
          loop_vars_to_zero.push_back(transport_loop_var);
        }
        loop_vars_to_zero.insert(loop_vars_to_zero.end(), nested_loop_vars.begin(),
                                 nested_loop_vars.end());
        PrimExpr base_tile_index =
            InferStagedCopyBaseTileIndex(nested_store, loop_vars_to_zero);
        return GenerateStagedCopyLoopSequence(nested_store, base_tile_index,
                                              loop_vars_to_zero);
      }
    }
    if (const auto* store = op->body.as<BufferStoreNode>()) {
      if (IsCopyOperation(store)) {
        CopyDirection direction = GetCopyDirection(store);
        if (direction == CopyDirection::kDramToCB || direction == CopyDirection::kCBToDram) {
          saw_copy_op_ = true;
          PrimExpr tile_index = InferCopyTileIndex(store, transport_loop_var);
          return GenerateCopySequence(store, tile_index);
        }
      }
    }
  }
  if (!select_compute_builtins_only_) {
    LocalToCBSliceMatch local_to_cb_match;
    if (MatchDirectLocalToCBSliceLoop(op, &local_to_cb_match)) {
      saw_copy_op_ = true;
      return GenerateLocalToCBSliceLoopSequence(op, local_to_cb_match);
    }
  }
  Stmt lowered = StmtExprMutator::VisitStmt_(op);
  return lowered;
}

// StmtExprMutator overrides
// Note: We only override specific node types and return the original node
// for unmatched patterns to avoid deep recursion that causes stack overflow.
Stmt PlanTTKernelABI::VisitStmt_(const EvaluateNode* op) {
  if (select_compute_builtins_only_) {
    if (const auto* call = op->value.as<CallNode>()) {
      FragmentCastMatch explicit_typecast_match;
      if (MatchExplicitTileTypecast(call, &explicit_typecast_match)) {
        return GetRef<Stmt>(op);
      }
      if (Stmt explicit_compute = LowerExplicitTileComputeCall(call);
          explicit_compute.defined()) {
        return explicit_compute;
      }
      if (call->op->IsInstance<OpNode>()) {
        const Op call_op = Downcast<Op>(call->op);
        if (call_op->name == "tl.blackhole.fill_fragment" && call->args.size() >= 3 &&
            IsFragmentFillValue(call->args[2])) {
          if (const auto* data = call->args[0].as<VarNode>()) {
            auto physical_it = compute_physical_buffers_by_data_.find(data);
            if (physical_it != compute_physical_buffers_by_data_.end()) {
              ClearSelectedSourceLiveProducer(physical_it->second);
              for (const std::string& identity :
                   CollectBufferFlowIdentities(physical_it->second)) {
                last_fragment_fill_value_by_buffer_identity_[identity] = call->args[2];
              }
            }
            last_fragment_fill_value_by_data_[data] = call->args[2];
          }
        }
      }
      if (IsMatmulCall(call) && call->args.size() >= 3 && IsBufferLikeExpr(call->args[2])) {
        const Buffer out_buffer = NormalizeToBufferRegion(call->args[2])->buffer;
        InvalidateLastFragmentFillValue(out_buffer);
        ClearTiledCBLiveFormAliases(out_buffer);
        ClearSelectedSourceLiveProducer(out_buffer);
        if (IsSingleFullTileMatmulOutput(call)) {
          RecordSelectedSourceLiveProducer(out_buffer);
        }
      }
    }
    return GetRef<Stmt>(op);
  }
  if (const auto* call = op->value.as<CallNode>()) {
    if (Stmt explicit_compute = LowerExplicitTileComputeCall(call);
        explicit_compute.defined()) {
      return explicit_compute;
    }
    if (call->op->IsInstance<OpNode>()) {
      const Op call_op = Downcast<Op>(call->op);
      if (call_op->name == "tl.blackhole.fill_fragment" && call->args.size() >= 3 &&
          IsFragmentFillValue(call->args[2])) {
        if (const auto* data = call->args[0].as<VarNode>()) {
          last_fragment_fill_value_by_data_[data] = call->args[2];
          auto physical_it = compute_physical_buffers_by_data_.find(data);
          if (physical_it != compute_physical_buffers_by_data_.end()) {
            for (const std::string& identity :
                 CollectBufferFlowIdentities(physical_it->second)) {
              last_fragment_fill_value_by_buffer_identity_[identity] = call->args[2];
            }
          }
        }
      }
    }
    if (IsMatmulCall(call)) {
      const auto order_it = stmt_order_index_by_node_.find(op);
      const int current_order_index =
          order_it != stmt_order_index_by_node_.end() ? order_it->second : 0;
      const int previous_order_index = current_lowering_order_index_;
      current_lowering_order_index_ = current_order_index;
      Stmt lowered = LowerMatmulCallWithFlowAnalysis(call, current_order_index);
      current_lowering_order_index_ = previous_order_index;
      return lowered;
    }
    if (IsClearOperation(call)) {
      return GenerateClearSequence(call);
    }
  }
  // Return original statement without recursion to avoid stack overflow
  // The parent class's VisitStmt_ would recursively visit child nodes,
  // which can cause deep recursion for deeply nested IR trees.
  return GetRef<Stmt>(op);
}

Stmt PlanTTKernelABI::VisitStmt_(const BufferStoreNode* op) {
  if (!select_compute_builtins_only_ && IsCopyOperation(op)) {
    saw_copy_op_ = true;
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

}  // namespace tl
}  // namespace tvm
