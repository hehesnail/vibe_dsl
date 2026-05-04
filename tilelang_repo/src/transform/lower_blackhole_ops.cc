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
#include "tir/transforms/ir_utils.h"
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>

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
using tir::IfThenElseNode;
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
static constexpr const char* kBlackholeExactOutputLiveNumTilesAttr =
    "blackhole.exact_output_live_num_tiles";
static constexpr const char* kBlackholeExactOutputLiveNumElementsAttr =
    "blackhole.exact_output_live_num_elements";
static constexpr const char* kBlackholeExactOutputLiveRowWidthAttr =
    "blackhole.exact_output_live_row_width";

static bool IsBlackholeExactOutputLiveAttr(const std::string& attr_key) {
  return attr_key == kBlackholeExactOutputLiveCBAttr ||
         attr_key == kBlackholeExactOutputLiveNumTilesAttr ||
         attr_key == kBlackholeExactOutputLiveNumElementsAttr ||
         attr_key == kBlackholeExactOutputLiveRowWidthAttr;
}

static bool StmtUsesVarOutsideDefinitionsAndExactLiveAttr(const Stmt& stmt,
                                                          const VarNode* target) {
  class Visitor final : public tir::StmtExprVisitor {
   public:
    explicit Visitor(const VarNode* target) : target_(target) {}

    bool Check(const Stmt& stmt) {
      VisitStmt(stmt);
      return found_;
    }

   private:
    void VisitStmt(const Stmt& stmt) final {
      if (found_) {
        return;
      }
      tir::StmtExprVisitor::VisitStmt(stmt);
    }

    void VisitExpr(const PrimExpr& expr) final {
      if (found_) {
        return;
      }
      tir::StmtExprVisitor::VisitExpr(expr);
    }

    void VisitStmt_(const AttrStmtNode* op) final {
      if (IsBlackholeExactOutputLiveAttr(op->attr_key)) {
        VisitStmt(op->body);
        return;
      }
      tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const AllocateNode* op) final {
      VisitStmt(op->body);
    }

    void VisitStmt_(const DeclBufferNode* op) final {
      VisitStmt(op->body);
    }

    void VisitExpr_(const VarNode* op) final {
      if (op == target_) {
        found_ = true;
      }
    }

    const VarNode* target_;
    bool found_{false};
  };

  return target != nullptr && stmt.defined() && Visitor(target).Check(stmt);
}

static bool StmtReadsBufferOutsideDefinitionsAndExactLiveAttr(const Stmt& stmt,
                                                              const Buffer& target) {
  class Visitor final : public tir::StmtExprVisitor {
   public:
    explicit Visitor(const Buffer& target) : target_(target) {}

    bool Check(const Stmt& stmt) {
      VisitStmt(stmt);
      return found_;
    }

   private:
    bool SameTarget(const Buffer& buffer) const {
      return target_.defined() && buffer.defined() && SameBufferIdentity(buffer, target_);
    }

    bool ReadsTargetArg(const PrimExpr& arg) {
      if (IsBufferLikeExpr(arg)) {
        BufferRegion region = NormalizeToBufferRegion(arg);
        return region.defined() && SameTarget(region->buffer);
      }
      VisitExpr(arg);
      return found_;
    }

    void VisitStmt(const Stmt& stmt) final {
      if (found_) {
        return;
      }
      tir::StmtExprVisitor::VisitStmt(stmt);
    }

    void VisitExpr(const PrimExpr& expr) final {
      if (found_) {
        return;
      }
      tir::StmtExprVisitor::VisitExpr(expr);
    }

    void VisitStmt_(const AttrStmtNode* op) final {
      if (IsBlackholeExactOutputLiveAttr(op->attr_key)) {
        VisitStmt(op->body);
        return;
      }
      tir::StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const AllocateNode* op) final {
      VisitStmt(op->body);
    }

    void VisitStmt_(const DeclBufferNode* op) final {
      VisitStmt(op->body);
    }

    void VisitStmt_(const BufferStoreNode* op) final {
      VisitExpr(op->value);
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);
      }
    }

    void VisitExpr_(const BufferLoadNode* op) final {
      if (SameTarget(op->buffer)) {
        found_ = true;
        return;
      }
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);
      }
    }

    void VisitExpr_(const CallNode* op) final {
      std::string op_name;
      if (op->op->IsInstance<OpNode>()) {
        op_name = Downcast<Op>(op->op)->name;
      }
      auto read_arg = [&](size_t index) {
        if (index < op->args.size() && ReadsTargetArg(op->args[index])) {
          found_ = true;
        }
      };
      if (op_name == "tl.blackhole.fill_fragment" ||
          op_name == "tl.blackhole.pack_fill_fragment_to_tiled_cb" ||
          op_name == "tl.blackhole.untilize_cb_front_tile_fragment") {
        for (size_t i = 1; i < op->args.size(); ++i) {
          read_arg(i);
          if (found_) {
            return;
          }
        }
        return;
      }
      if (op_name == "tl.blackhole.tilize_local_fragment_slice") {
        read_arg(0);
        return;
      }
      if (op_name == "tl.blackhole.tilize_cast_fragment_slice") {
        read_arg(1);
        return;
      }
      if (op_name == "tl.blackhole.add_fragment" ||
          op_name == "tl.blackhole.add_fragment_from_cb_front") {
        read_arg(0);
        read_arg(1);
        return;
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        read_arg(i);
        if (found_) {
          return;
        }
      }
    }

    Buffer target_;
    bool found_{false};
  };

  return target.defined() && stmt.defined() && Visitor(target).Check(stmt);
}

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
    if (const auto* attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == "blackhole.segment_kind") {
        if (const auto* kind = attr->value.as<StringImmNode>()) {
          if (kind->value == "compute") {
            found = true;
          }
        }
      }
      return;
    }
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

static Array<PrimExpr> FindBlackholeAccAllocationExtents(
    const Buffer& buffer,
    const std::unordered_map<std::string, Map<String, Any>>& logical_tile_layout_specs_by_buffer) {
  const std::string identity = BufferIdentityName(buffer);
  auto spec_it = logical_tile_layout_specs_by_buffer.find(identity);
  if (spec_it == logical_tile_layout_specs_by_buffer.end()) {
    return buffer->shape;
  }
  if (auto local_shape = spec_it->second.Get(String(schema_key::kLocalShape))) {
    Array<PrimExpr> local_extents = Downcast<Array<PrimExpr>>(local_shape.value());
    if (!local_extents.empty()) {
      return local_extents;
    }
  }
  return buffer->shape;
}

static std::string BlackholeAccStorageKey(const VarNode* var) {
  if (var == nullptr) {
    return "";
  }
  if (var->type_annotation.as<PointerTypeNode>() == nullptr) {
    return var->name_hint;
  }
  const std::string scope = tir::GetPtrStorageScope(GetRef<Var>(var));
  if (scope != "blackhole.acc") {
    return "";
  }
  return var->name_hint;
}

static std::string BlackholeAccStorageKey(const Buffer& buffer) {
  if (!buffer.defined() || std::string(buffer.scope()) != "blackhole.acc") {
    return "";
  }
  return BlackholeAccStorageKey(buffer->data.get());
}

static Stmt RewrapMissingBlackholeAccDefinitions(
    const Stmt& original_body, const Stmt& rewritten_body,
    const std::unordered_map<std::string, Map<String, Any>>& logical_tile_layout_specs_by_buffer) {
  struct AccDefinition {
    Var var;
    DataType dtype;
    Array<PrimExpr> extents;
    PrimExpr condition;
    Map<String, Any> annotations;
    Buffer buffer;
    bool has_lexical_allocation{false};
  };

  std::unordered_map<const VarNode*, size_t> definition_index_by_data;
  std::unordered_map<std::string, size_t> definition_index_by_storage_key;
  std::vector<AccDefinition> definitions;
  auto bind_definition_key = [&](size_t definition_index, const VarNode* data,
                                 const std::string& storage_key) {
    if (data != nullptr) {
      definition_index_by_data[data] = definition_index;
    }
    if (!storage_key.empty()) {
      definition_index_by_storage_key[storage_key] = definition_index;
    }
  };
  auto remember_buffer = [&](const Buffer& buffer) {
    if (!buffer.defined() || std::string(buffer.scope()) != "blackhole.acc") {
      return;
    }
    const VarNode* data = buffer->data.get();
    const std::string storage_key = BlackholeAccStorageKey(buffer);
    auto [it, inserted] = definition_index_by_data.emplace(data, definitions.size());
    if (inserted) {
      definitions.push_back(AccDefinition{buffer->data, buffer->dtype,
                                          FindBlackholeAccAllocationExtents(
                                              buffer, logical_tile_layout_specs_by_buffer),
                                          Bool(1),
                                          Map<String, Any>(), buffer, false});
      bind_definition_key(it->second, data, storage_key);
      return;
    }
    AccDefinition& definition = definitions[it->second];
    if (!definition.buffer.defined()) {
      definition.buffer = buffer;
    }
  };
  auto remember_allocate = [&](const tir::AllocateNode* allocate) {
    if (allocate == nullptr) {
      return;
    }
    const std::string storage_key = BlackholeAccStorageKey(allocate->buffer_var.get());
    if (storage_key.empty()) {
      return;
    }
    size_t definition_index = definitions.size();
    auto definition_it = definition_index_by_data.find(allocate->buffer_var.get());
    if (definition_it != definition_index_by_data.end()) {
      definition_index = definition_it->second;
    } else {
      auto key_it = definition_index_by_storage_key.find(storage_key);
      if (key_it != definition_index_by_storage_key.end()) {
        definition_index = key_it->second;
      }
    }
    if (definition_index >= definitions.size()) {
      definition_index = definitions.size();
      definitions.push_back(AccDefinition{allocate->buffer_var, allocate->dtype,
                                          allocate->extents, allocate->condition,
                                          allocate->annotations, Buffer(), true});
    }
    AccDefinition& definition = definitions[definition_index];
    definition.var = allocate->buffer_var;
    definition.dtype = allocate->dtype;
    definition.extents = allocate->extents;
    definition.condition = allocate->condition;
    definition.annotations = allocate->annotations;
    definition.has_lexical_allocation = true;
    bind_definition_key(definition_index, allocate->buffer_var.get(), storage_key);
  };
  tir::PostOrderVisit(original_body, [&](const ObjectRef& node) {
    if (const auto* block = node.as<tir::BlockNode>()) {
      for (const Buffer& buffer : block->alloc_buffers) {
        remember_buffer(buffer);
      }
      return;
    }
    if (const auto* decl = node.as<tir::DeclBufferNode>()) {
      remember_buffer(decl->buffer);
      return;
    }
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      remember_buffer(store->buffer);
      return;
    }
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      remember_buffer(load->buffer);
      return;
    }
    if (const auto* call = node.as<tir::CallNode>()) {
      for (const PrimExpr& arg : call->args) {
        if (!IsBufferLikeExpr(arg)) {
          continue;
        }
        BufferRegion region = NormalizeToBufferRegion(arg);
        if (region.defined()) {
          remember_buffer(region->buffer);
        }
      }
      return;
    }
    remember_allocate(node.as<tir::AllocateNode>());
  });
  if (definitions.empty()) {
    return rewritten_body;
  }

  class UnboundAccUseCollector final : public tir::StmtExprVisitor {
   public:
    UnboundAccUseCollector(
        const std::unordered_map<const VarNode*, size_t>& definition_index_by_data,
        const std::unordered_map<std::string, size_t>& definition_index_by_storage_key,
        const std::vector<AccDefinition>& definitions,
        std::unordered_set<size_t>* unbound_definition_indices)
        : definition_index_by_data_(definition_index_by_data),
          definition_index_by_storage_key_(definition_index_by_storage_key),
          definitions_(definitions),
          unbound_definition_indices_(unbound_definition_indices) {}

    void Check(const Stmt& stmt) { VisitStmt(stmt); }

   private:
    void PushBinding(const VarNode* data, const std::string& storage_key) {
      if (data != nullptr) {
        ++active_data_bindings_[data];
      }
      if (!storage_key.empty()) {
        ++active_storage_bindings_[storage_key];
      }
    }

    void PopBinding(const VarNode* data, const std::string& storage_key) {
      if (data != nullptr) {
        auto data_it = active_data_bindings_.find(data);
        ICHECK(data_it != active_data_bindings_.end());
        if (--data_it->second == 0) {
          active_data_bindings_.erase(data_it);
        }
      }
      if (!storage_key.empty()) {
        auto key_it = active_storage_bindings_.find(storage_key);
        ICHECK(key_it != active_storage_bindings_.end());
        if (--key_it->second == 0) {
          active_storage_bindings_.erase(key_it);
        }
      }
    }

    bool HasActiveBinding(const VarNode* data, const std::string& storage_key) const {
      if (data != nullptr && active_data_bindings_.count(data) != 0U) {
        return true;
      }
      return !storage_key.empty() && active_storage_bindings_.count(storage_key) != 0U;
    }

    std::optional<size_t> ResolveDefinitionIndex(const VarNode* data,
                                                 const std::string& storage_key) const {
      auto data_it = definition_index_by_data_.find(data);
      if (data_it != definition_index_by_data_.end()) {
        return data_it->second;
      }
      if (!storage_key.empty()) {
        auto key_it = definition_index_by_storage_key_.find(storage_key);
        if (key_it != definition_index_by_storage_key_.end()) {
          return key_it->second;
        }
      }
      return std::nullopt;
    }

    void VisitStmt_(const tir::AllocateNode* op) final {
      const std::string storage_key = BlackholeAccStorageKey(op->buffer_var.get());
      PushBinding(op->buffer_var.get(), storage_key);
      VisitStmt(op->body);
      PopBinding(op->buffer_var.get(), storage_key);
    }

    void VisitStmt_(const tir::DeclBufferNode* op) final {
      const bool is_acc_buffer =
          op->buffer.defined() && std::string(op->buffer.scope()) == "blackhole.acc";
      const VarNode* data = is_acc_buffer ? op->buffer->data.get() : nullptr;
      const std::string storage_key = is_acc_buffer ? BlackholeAccStorageKey(op->buffer) : "";
      PushBinding(data, storage_key);
      VisitStmt(op->body);
      PopBinding(data, storage_key);
    }

    void VisitExpr_(const VarNode* op) final {
      const std::string storage_key = BlackholeAccStorageKey(op);
      std::optional<size_t> definition_index = ResolveDefinitionIndex(op, storage_key);
      if (!definition_index.has_value()) {
        return;
      }
      const AccDefinition& definition = definitions_[definition_index.value()];
      if (HasActiveBinding(op, storage_key)) {
        return;
      }
      // Executable materialization can expose a blackhole.acc Buffer as a
      // direct builtin handle even when the original TIR carried only a buffer
      // definition, not a lexical AllocateNode.  That buffer is still explicit
      // IR state and must be rewrapped before codegen sees the handle.
      if (!definition.has_lexical_allocation && !definition.buffer.defined()) {
        return;
      }
      unbound_definition_indices_->insert(definition_index.value());
    }

    const std::unordered_map<const VarNode*, size_t>& definition_index_by_data_;
    const std::unordered_map<std::string, size_t>& definition_index_by_storage_key_;
    const std::vector<AccDefinition>& definitions_;
    std::unordered_set<size_t>* unbound_definition_indices_;
    std::unordered_map<const VarNode*, int> active_data_bindings_;
    std::unordered_map<std::string, int> active_storage_bindings_;
  };

  std::unordered_set<size_t> unbound_definition_indices;
  UnboundAccUseCollector(definition_index_by_data, definition_index_by_storage_key,
                         definitions, &unbound_definition_indices)
      .Check(rewritten_body);
  if (unbound_definition_indices.empty()) {
    return rewritten_body;
  }

  Stmt body = rewritten_body;
  for (int i = static_cast<int>(definitions.size()) - 1; i >= 0; --i) {
    const AccDefinition& definition = definitions[i];
    if (unbound_definition_indices.count(static_cast<size_t>(i)) == 0U) {
      continue;
    }
    if (definition.buffer.defined()) {
      body = DeclBuffer(definition.buffer, body);
    }
    body = Allocate(definition.var, definition.dtype, definition.extents,
                    definition.condition, body, definition.annotations);
  }
  return body;
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

static bool IsNoOpStmt(const Stmt& stmt) {
  if (!stmt.defined()) {
    return true;
  }
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<IntImmNode>() != nullptr ||
           eval->value.as<FloatImmNode>() != nullptr;
  }
  if (const auto* seq = stmt.as<SeqStmtNode>()) {
    return std::all_of(seq->seq.begin(), seq->seq.end(),
                       [](const Stmt& child) { return IsNoOpStmt(child); });
  }
  return false;
}

static bool IsTrackedBlackholeAccVar(
    const VarNode* var,
    const std::unordered_set<const VarNode*>& blackhole_acc_data_vars) {
  if (var == nullptr) {
    return false;
  }
  return blackhole_acc_data_vars.count(var) != 0U ||
         !BlackholeAccStorageKey(var).empty();
}

static std::optional<const VarNode*> BlackholeAccDataVarFromExpr(
    const PrimExpr& expr,
    const std::unordered_set<const VarNode*>& blackhole_acc_data_vars) {
  if (const auto* var = expr.as<VarNode>()) {
    if (IsTrackedBlackholeAccVar(var, blackhole_acc_data_vars)) {
      return var;
    }
  }
  if (!IsBufferLikeExpr(expr)) {
    return std::nullopt;
  }
  BufferRegion region = NormalizeToBufferRegion(expr);
  if (!region.defined() || !region->buffer.defined() ||
      GetStorageScope(region->buffer) != "blackhole.acc") {
    return std::nullopt;
  }
  return region->buffer->data.get();
}

static std::unordered_set<const VarNode*> CollectBlackholeAccDataVars(const Stmt& body) {
  std::unordered_set<const VarNode*> vars;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    if (const auto* allocate = node.as<AllocateNode>()) {
      if (!BlackholeAccStorageKey(allocate->buffer_var.get()).empty()) {
        vars.insert(allocate->buffer_var.get());
      }
      return;
    }
    if (const auto* decl = node.as<DeclBufferNode>()) {
      if (decl->buffer.defined() && GetStorageScope(decl->buffer) == "blackhole.acc") {
        vars.insert(decl->buffer->data.get());
      }
      return;
    }
    if (const auto* load = node.as<BufferLoadNode>()) {
      if (load->buffer.defined() && GetStorageScope(load->buffer) == "blackhole.acc") {
        vars.insert(load->buffer->data.get());
      }
      return;
    }
    if (const auto* store = node.as<BufferStoreNode>()) {
      if (store->buffer.defined() && GetStorageScope(store->buffer) == "blackhole.acc") {
        vars.insert(store->buffer->data.get());
      }
      return;
    }
    if (const auto* call = node.as<CallNode>()) {
      for (const PrimExpr& arg : call->args) {
        std::optional<const VarNode*> data = BlackholeAccDataVarFromExpr(arg, vars);
        if (data.has_value()) {
          vars.insert(data.value());
        }
      }
    }
  });
  return vars;
}

static bool IsBlackholeTileComputeFillTile(const CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>() || call->args.size() < 2U) {
    return false;
  }
  const Op call_op = Downcast<Op>(call->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName) {
    return false;
  }
  const auto* operation = call->args[0].as<StringImmNode>();
  return operation != nullptr && operation->value == blackhole_tile_compute_schema::kFillTile;
}

static bool IsDeadBlackholeAccFillCall(
    const CallNode* call,
    const std::unordered_set<const VarNode*>& blackhole_acc_data_vars,
    const std::unordered_set<const VarNode*>& live_blackhole_acc_data_vars) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(call->op);
  std::optional<const VarNode*> dst_data;
  if (call_op->name == "tl.blackhole.fill_fragment" && !call->args.empty()) {
    dst_data = BlackholeAccDataVarFromExpr(call->args[0], blackhole_acc_data_vars);
  } else if (IsBlackholeTileComputeFillTile(call)) {
    dst_data = BlackholeAccDataVarFromExpr(call->args[1], blackhole_acc_data_vars);
  }
  return dst_data.has_value() &&
         live_blackhole_acc_data_vars.count(dst_data.value()) == 0U;
}

class LiveBlackholeAccUseCollector final : public tir::StmtExprVisitor {
 public:
  explicit LiveBlackholeAccUseCollector(
      std::unordered_set<const VarNode*> blackhole_acc_data_vars)
      : blackhole_acc_data_vars_(std::move(blackhole_acc_data_vars)) {}

  std::unordered_set<const VarNode*> Collect(const Stmt& stmt) {
    VisitStmt(stmt);
    return live_blackhole_acc_data_vars_;
  }

 private:
  void MarkIfTracked(const VarNode* var) {
    if (IsTrackedBlackholeAccVar(var, blackhole_acc_data_vars_)) {
      live_blackhole_acc_data_vars_.insert(var);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (IsBlackholeExactOutputLiveAttr(op->attr_key)) {
      VisitStmt(op->body);
      return;
    }
    if (const auto* var = op->node.as<VarNode>()) {
      if (IsTrackedBlackholeAccVar(var, blackhole_acc_data_vars_)) {
        VisitExpr(op->value);
        VisitStmt(op->body);
        return;
      }
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    VisitExpr(op->condition);
    VisitStmt(op->body);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    VisitStmt(op->body);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitExpr(op->value);
    for (const PrimExpr& index : op->indices) {
      VisitExpr(index);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.defined() && GetStorageScope(op->buffer) == "blackhole.acc") {
      MarkIfTracked(op->buffer->data.get());
    }
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final {
    MarkIfTracked(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op->IsInstance<OpNode>()) {
      const Op call_op = Downcast<Op>(op->op);
      if (call_op->name == "tl.blackhole.fill_fragment" && !op->args.empty()) {
        for (size_t i = 1; i < op->args.size(); ++i) {
          VisitExpr(op->args[i]);
        }
        return;
      }
      if (IsBlackholeTileComputeFillTile(op)) {
        VisitExpr(op->args[0]);
        for (size_t i = 2; i < op->args.size(); ++i) {
          VisitExpr(op->args[i]);
        }
        return;
      }
      if (call_op->name == "tl.blackhole.pack_fill_fragment_to_tiled_cb" &&
          op->args.size() >= 6U) {
        for (size_t i = 1; i < op->args.size(); ++i) {
          VisitExpr(op->args[i]);
        }
        return;
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        VisitExpr(op->args[i]);
      }
      return;
    }
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<const VarNode*> blackhole_acc_data_vars_;
  std::unordered_set<const VarNode*> live_blackhole_acc_data_vars_;
};

class DeadBlackholeAccPruner final : public tir::StmtExprMutator {
 public:
  DeadBlackholeAccPruner(
      std::unordered_set<const VarNode*> blackhole_acc_data_vars,
      std::unordered_set<const VarNode*> live_blackhole_acc_data_vars)
      : blackhole_acc_data_vars_(std::move(blackhole_acc_data_vars)),
        live_blackhole_acc_data_vars_(std::move(live_blackhole_acc_data_vars)) {}

 private:
  bool IsLive(const VarNode* var) const {
    return IsTrackedBlackholeAccVar(var, blackhole_acc_data_vars_) &&
           live_blackhole_acc_data_vars_.count(var) != 0U;
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> rewritten;
    for (const Stmt& child : op->seq) {
      Stmt lowered = VisitStmt(child);
      if (!IsNoOpStmt(lowered)) {
        rewritten.push_back(lowered);
      }
    }
    if (rewritten.empty()) {
      return Evaluate(IntImm32(0));
    }
    return SeqStmt::Flatten(rewritten);
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (IsDeadBlackholeAccFillCall(call, blackhole_acc_data_vars_,
                                     live_blackhole_acc_data_vars_)) {
        return Evaluate(IntImm32(0));
      }
    }
    return tir::StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt body = VisitStmt(op->body);
    if (IsTrackedBlackholeAccVar(op->buffer_var.get(), blackhole_acc_data_vars_) &&
        !IsLive(op->buffer_var.get())) {
      return body;
    }
    return Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body,
                    op->annotations);
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Stmt body = VisitStmt(op->body);
    if (op->buffer.defined() && GetStorageScope(op->buffer) == "blackhole.acc" &&
        !IsLive(op->buffer->data.get())) {
      return body;
    }
    return DeclBuffer(op->buffer, body);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (op->buffer.defined() && GetStorageScope(op->buffer) == "blackhole.acc" &&
        !IsLive(op->buffer->data.get())) {
      return Evaluate(IntImm32(0));
    }
    return tir::StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt body = VisitStmt(op->body);
    if (IsNoOpStmt(body)) {
      return body;
    }
    if (const auto* var = op->node.as<VarNode>()) {
      if (IsTrackedBlackholeAccVar(var, blackhole_acc_data_vars_) && !IsLive(var)) {
        return body;
      }
    }
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }

  std::unordered_set<const VarNode*> blackhole_acc_data_vars_;
  std::unordered_set<const VarNode*> live_blackhole_acc_data_vars_;
};

static Stmt PruneDeadBlackholeAccFragmentFillsAndDefinitions(const Stmt& body) {
  std::unordered_set<const VarNode*> blackhole_acc_data_vars = CollectBlackholeAccDataVars(body);
  if (blackhole_acc_data_vars.empty()) {
    return body;
  }
  std::unordered_set<const VarNode*> live_blackhole_acc_data_vars =
      LiveBlackholeAccUseCollector(blackhole_acc_data_vars).Collect(body);
  return DeadBlackholeAccPruner(std::move(blackhole_acc_data_vars),
                                std::move(live_blackhole_acc_data_vars))(body);
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
  direct_copy_source_by_buffer_identity_.clear();
  buffer_by_identity_.clear();
  broadcast_cols_rhs_buffers_.clear();
  broadcast_cols_source_buffers_.clear();
  selected_source_live_producer_buffers_.clear();
  selected_source_live_producer_order_by_buffer_identity_.clear();
  seeded_cb_requirement_names_.clear();
  loop_carried_exact_cb_state_by_logical_value_.clear();
  tt_exact_cb_virtual_values_.clear();
  tt_exact_cb_use_events_.clear();
  tt_exact_cb_live_intervals_.clear();
  tt_exact_cb_allocations_.clear();
  tt_exact_cb_release_events_.clear();
  tt_exact_cb_live_form_index_by_logical_value_.clear();
  tt_exact_cb_virtual_index_by_key_.clear();
  tt_exact_cb_allocation_index_by_key_.clear();
  last_fragment_fill_value_by_buffer_identity_.clear();
  last_fragment_fill_value_by_data_.clear();
  LoadPhysicalComputeBufferBindings(func);
  current_segment_kind_.clear();
  thread_index_vars_.clear();
  thread_index_var_names_.clear();
  thread_index_var_static_extents_.clear();
  loop_var_static_extents_.clear();
  active_serial_loop_vars_.clear();
  active_serial_loop_order_ranges_.clear();
  block_index_vars_.clear();
  block_index_var_names_.clear();
  cb_consumed_compute_input_pages_by_buffer_identity_.clear();
  cb_consumed_compute_input_use_count_by_buffer_identity_.clear();
  buffer_flow_facts_.clear();
  execution_ordered_stmts_.clear();
  buffer_live_form_cb_by_buffer_identity_.clear();
  buffer_live_form_order_by_buffer_identity_.clear();
  buffer_live_form_order_by_cb_id_.clear();
  exact_output_live_form_cb_by_buffer_identity_.clear();
  exact_output_live_form_order_by_buffer_identity_.clear();
  exact_output_live_form_order_by_cb_id_.clear();
  exact_output_live_form_value_by_buffer_identity_.clear();
  invalidated_live_form_order_by_buffer_identity_.clear();
  local_only_live_form_buffer_identities_.clear();
  stmt_order_index_by_node_.clear();
  current_lowering_order_index_ = -1;
  requires_compute_segment_ = false;
  logical_tile_layout_specs_by_buffer_.clear();
  spatial_materialization_boundaries_.clear();
  spatial_materialization_boundary_position_by_index_.clear();
  spatial_live_value_by_subject_.clear();
  spatial_lifetime_kind_by_subject_.clear();
  buffer_materialization_facts_by_target_buffer_.clear();
  tt_compute_op_plans_.clear();
  tile_compute_dag_lowering_decisions_.clear();
  tile_compute_dag_lowering_decision_consumed_.clear();
  active_tile_compute_dag_lowering_decision_.reset();
  tile_compute_input_buffers_.clear();
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
  LoadBufferFlowFacts(lowering_support_facts);
  LoadTileComputeDAGLoweringPlan(func);
  LoadDirectCopySourceBindings(func);
  RefreshBroadcastColsSourceBuffers();
  execution_ordered_stmts_ = CollectExecutionOrderedStmts(func->body);
  stmt_order_index_by_node_ = BuildExecutionOrderIndexByStmtNode(func->body);

  PrimFunc selected = func;
  Stmt selected_body = VisitStmt(func->body);
  selected_body = RewrapMissingBlackholeAccDefinitions(
      func->body, selected_body, logical_tile_layout_specs_by_buffer_);
  selected.CopyOnWrite()->body = selected_body;
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
  logical_grid_z_ = 1;
  copy_input_buffer_ = Buffer();
  copy_output_buffer_ = Buffer();
  copy_input_buffer_name_.clear();
  copy_output_buffer_name_.clear();
  copy_input_buffer_names_.clear();
  copy_output_buffer_names_.clear();
  host_buffer_by_compute_operand_buffer_.clear();
  direct_copy_source_by_buffer_identity_.clear();
  buffer_by_identity_.clear();
  broadcast_cols_rhs_buffers_.clear();
  broadcast_cols_source_buffers_.clear();
  copy_input_shape_.clear();
  copy_output_shape_.clear();
  copy_intermediate_shape_.clear();
  thread_index_vars_.clear();
  thread_index_var_names_.clear();
  thread_index_var_static_extents_.clear();
  loop_var_static_extents_.clear();
  active_serial_loop_vars_.clear();
  active_serial_loop_order_ranges_.clear();
  block_index_vars_.clear();
  block_index_var_names_.clear();
  cb_consumed_compute_input_pages_by_buffer_identity_.clear();
  cb_consumed_compute_input_use_count_by_buffer_identity_.clear();
  buffer_flow_facts_.clear();
  execution_ordered_stmts_.clear();
  buffer_live_form_cb_by_buffer_identity_.clear();
  buffer_live_form_order_by_buffer_identity_.clear();
  buffer_live_form_order_by_cb_id_.clear();
  exact_output_live_form_cb_by_buffer_identity_.clear();
  exact_output_live_form_order_by_buffer_identity_.clear();
  exact_output_live_form_order_by_cb_id_.clear();
  exact_output_live_form_value_by_buffer_identity_.clear();
  invalidated_live_form_order_by_buffer_identity_.clear();
  local_only_live_form_buffer_identities_.clear();
  active_loop_carried_buffer_identity_stack_.clear();
  loop_carried_exact_cb_state_by_logical_value_.clear();
  selected_source_live_producer_buffers_.clear();
  selected_source_live_producer_order_by_buffer_identity_.clear();
  seeded_cb_requirement_names_.clear();
  stmt_order_index_by_node_.clear();
  current_lowering_order_index_ = -1;
  segment_plan_.clear();
  read_accessor_slots_.clear();
  write_accessor_slots_.clear();
  fused_dataflow_accessor_slots_.clear();
  tt_kernels_.clear();
  tt_abi_plans_.clear();
  tt_live_form_plans_.clear();
  tt_materialization_plans_.clear();
  tt_consumer_binding_plans_.clear();
  tt_exact_cb_virtual_values_.clear();
  tt_exact_cb_use_events_.clear();
  tt_exact_cb_live_intervals_.clear();
  tt_exact_cb_allocations_.clear();
  tt_exact_cb_release_events_.clear();
  tt_exact_cb_live_form_index_by_logical_value_.clear();
  tt_exact_cb_virtual_index_by_key_.clear();
  tt_exact_cb_allocation_index_by_key_.clear();
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
  tile_compute_input_buffers_.clear();
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
  if (auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram)) {
    if (!staged_program.value()->core_groups.empty()) {
      logical_grid_z_ =
          std::max<int64_t>(1, staged_program.value()->core_groups[0]->logical_grid_z);
    }
  }
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
  LoadDirectCopySourceBindings(func);
  RefreshBroadcastColsSourceBuffers();
  execution_ordered_stmts_ = CollectExecutionOrderedStmts(func->body);
  stmt_order_index_by_node_ = BuildExecutionOrderIndexByStmtNode(func->body);
  LoadExactOutputLiveFormMarkers(func->body);
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
        for (const std::string& buffer_identity : CollectBufferFlowIdentities(region->buffer)) {
          auto it = cb_consumed_compute_input_pages_by_buffer_identity_.find(buffer_identity);
          if (it == cb_consumed_compute_input_pages_by_buffer_identity_.end()) {
            cb_consumed_compute_input_pages_by_buffer_identity_[buffer_identity] = tile_count;
          } else {
            it->second = std::max(it->second, tile_count);
          }
          cb_consumed_compute_input_use_count_by_buffer_identity_[buffer_identity] += 1;
        }
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
  Stmt final_body = StripSegmentKindMarkers(body_with_segment_markers);
  final_body = RewrapMissingBlackholeAccDefinitions(
      func->body, final_body, logical_tile_layout_specs_by_buffer_);
  final_body = PruneDeadBlackholeAccFragmentFillsAndDefinitions(final_body);
  new_func.CopyOnWrite()->body = final_body;
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
  if (!buffer_identity.empty()) {
    auto by_identity = buffer_identity_to_req_index_.find(buffer_identity);
    if (by_identity != buffer_identity_to_req_index_.end()) {
      return bind_existing_requirement(by_identity->second);
    }
  }
  auto by_data = buffer_data_to_req_index_.find(buffer->data.get());
  if (by_data != buffer_data_to_req_index_.end()) {
    const CBRequirement& existing_req = cb_requirements_.at(by_data->second);
    if (buffer_identity.empty() || existing_req.name.empty() ||
        existing_req.name == buffer_identity) {
      return bind_existing_requirement(by_data->second);
    }
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
  if (const auto* decl_buffer = stmt.as<DeclBufferNode>()) {
    return IsPureCopyLoopNest(decl_buffer->body);
  }
  if (const auto* if_then_else = stmt.as<IfThenElseNode>()) {
    if (!IsPureCopyLoopNest(if_then_else->then_case)) {
      return false;
    }
    return !if_then_else->else_case.defined() ||
           IsPureCopyLoopNest(if_then_else->else_case.value());
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

bool IsValueIndexSelectionMaskCastStore(const BufferStoreNode* store) {
  if (!store || !store->buffer->dtype.is_bfloat16()) {
    return false;
  }
  const auto* cast = store->value.as<CastNode>();
  if (!cast || !cast->dtype.is_bfloat16()) {
    return false;
  }
  PrimExpr true_value;
  PrimExpr false_value;
  if (const auto* select = cast->value.as<SelectNode>()) {
    true_value = select->true_value;
    false_value = select->false_value;
  } else if (const auto* call = cast->value.as<CallNode>()) {
    if (!call->op->IsInstance<OpNode>() ||
        Downcast<Op>(call->op)->name != "tir.if_then_else" ||
        call->args.size() != 3U) {
      return false;
    }
    true_value = call->args[1];
    false_value = call->args[2];
  } else {
    return false;
  }
  if (!IsFloatImmValue(true_value, -10000.0)) {
    return false;
  }
  PrimExpr retained_value = false_value;
  if (const auto* retained_cast = retained_value.as<CastNode>()) {
    retained_value = retained_cast->value;
  }
  const auto* retained_load = retained_value.as<BufferLoadNode>();
  return retained_load != nullptr &&
         SameBufferIdentity(retained_load->buffer, store->buffer);
}

bool HasResidualFragmentCast(const Stmt& body) {
  bool found = false;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* store = node.as<BufferStoreNode>();
    if (!store || !IsUnsupportedResidualLocalScope(store->buffer)) {
      return;
    }
    if (store->value.as<CastNode>()) {
      if (IsValueIndexSelectionMaskCastStore(store)) {
        return;
      }
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
  if (op->attr_key == kBlackholeExactOutputLiveNumTilesAttr ||
      op->attr_key == kBlackholeExactOutputLiveNumElementsAttr ||
      op->attr_key == kBlackholeExactOutputLiveRowWidthAttr) {
    Stmt body = VisitStmt(op->body);
    const auto* value = op->value.as<IntImmNode>();
    if (value == nullptr) {
      return body;
    }
    auto record_identity = [&](const std::string& identity) {
      if (identity.empty()) {
        return;
      }
      auto& live_value = exact_output_live_form_value_by_buffer_identity_[identity];
      if (op->attr_key == kBlackholeExactOutputLiveNumTilesAttr) {
        live_value.num_tiles = static_cast<int>(value->value);
      } else if (op->attr_key == kBlackholeExactOutputLiveNumElementsAttr) {
        live_value.num_elements = value->value;
      } else {
        live_value.row_width = value->value;
      }
    };
    if (const auto* identity = op->node.as<StringImmNode>()) {
      record_identity(identity->value);
    } else if (const auto* data = op->node.as<VarNode>()) {
      auto buffer_it = compute_physical_buffers_by_data_.find(data);
      if (buffer_it != compute_physical_buffers_by_data_.end() &&
          buffer_it->second.defined()) {
        record_identity(BufferIdentityName(buffer_it->second));
      } else {
        record_identity(data->name_hint);
      }
    }
    return body;
  }
  if (op->attr_key == kBlackholeExactOutputLiveCBAttr) {
    Stmt body = VisitStmt(op->body);
    const auto* cb_id = op->value.as<IntImmNode>();
    const auto* data = op->node.as<VarNode>();
    auto is_active_loop_carried_identity = [&](const std::string& identity) {
      if (identity.empty()) {
        return false;
      }
      for (auto stack_it = active_loop_carried_buffer_identity_stack_.rbegin();
           stack_it != active_loop_carried_buffer_identity_stack_.rend(); ++stack_it) {
        if (stack_it->count(identity) != 0U) {
          return true;
        }
      }
      return false;
    };
    auto record_identity = [&](const std::string& identity) {
      if (identity.empty() || cb_id == nullptr) {
        return;
      }
      if (is_active_loop_carried_identity(identity)) {
        return;
      }
      auto invalidated_it = invalidated_live_form_order_by_buffer_identity_.find(identity);
      if (invalidated_it != invalidated_live_form_order_by_buffer_identity_.end()) {
        if (current_lowering_order_index_ >= 0 &&
            current_lowering_order_index_ < invalidated_it->second) {
          return;
        }
        invalidated_live_form_order_by_buffer_identity_.erase(invalidated_it);
      }
      const auto tombstone_order_it =
          exact_output_live_form_order_by_buffer_identity_.find(identity);
      const bool has_live_cb =
          exact_output_live_form_cb_by_buffer_identity_.find(identity) !=
          exact_output_live_form_cb_by_buffer_identity_.end();
      if (!has_live_cb && current_lowering_order_index_ >= 0 &&
          tombstone_order_it != exact_output_live_form_order_by_buffer_identity_.end() &&
          tombstone_order_it->second >= current_lowering_order_index_) {
        return;
      }
      exact_output_live_form_cb_by_buffer_identity_[identity] =
          static_cast<int>(cb_id->value);
      exact_output_live_form_value_by_buffer_identity_[identity].cb_id =
          static_cast<int>(cb_id->value);
      local_only_live_form_buffer_identities_.erase(identity);
      if (current_lowering_order_index_ >= 0) {
        exact_output_live_form_order_by_buffer_identity_[identity] =
            current_lowering_order_index_;
      }
    };
    if (cb_id != nullptr) {
      if (const auto* identity = op->node.as<StringImmNode>()) {
        record_identity(identity->value);
      }
    }
    if (cb_id != nullptr && data != nullptr) {
      auto buffer_it = compute_physical_buffers_by_data_.find(data);
      if (buffer_it != compute_physical_buffers_by_data_.end() && buffer_it->second.defined()) {
        ExactTiledCBValue live_value;
        live_value.buffer = buffer_it->second;
        live_value.cb_id = static_cast<int>(cb_id->value);
        live_value.borrowed_live = true;
        live_value.live_identity = BufferIdentityName(buffer_it->second);
        PopulateExactTiledCBValueShape(buffer_it->second, &live_value);
        RecordExactOutputLiveForm(buffer_it->second, live_value);
      } else {
        record_identity(data->name_hint);
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
    if (!select_compute_builtins_only_ && zero_thread_var) {
      active_serial_loop_vars_.push_back(iv->var);
    }
    Stmt body = VisitStmt(op->body);
    if (!select_compute_builtins_only_ && zero_thread_var) {
      active_serial_loop_vars_.pop_back();
    }
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

void PlanTTKernelABI::LoadExactOutputLiveFormMarkers(const Stmt& body) {
  auto marker_identity = [&](const AttrStmtNode* attr) -> std::string {
    if (const auto* identity = attr->node.as<StringImmNode>()) {
      return identity->value;
    }
    if (const auto* data = attr->node.as<VarNode>()) {
      auto buffer_it = compute_physical_buffers_by_data_.find(data);
      if (buffer_it != compute_physical_buffers_by_data_.end() &&
          buffer_it->second.defined()) {
        return BufferIdentityName(buffer_it->second);
      }
      return data->name_hint;
    }
    return "";
  };
  auto marker_order = [&](const AttrStmtNode* attr) -> int {
    auto order_it = stmt_order_index_by_node_.find(attr);
    if (order_it != stmt_order_index_by_node_.end()) {
      return order_it->second;
    }
    int last_body_order = -1;
    tir::PostOrderVisit(attr->body, [&](const ObjectRef& node) {
      auto body_order_it = stmt_order_index_by_node_.find(node.get());
      if (body_order_it != stmt_order_index_by_node_.end()) {
        last_body_order = std::max(last_body_order, body_order_it->second);
      }
    });
    return last_body_order;
  };
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* attr = node.as<AttrStmtNode>();
    if (attr == nullptr || !IsBlackholeExactOutputLiveAttr(attr->attr_key)) {
      return;
    }
    const std::string identity = marker_identity(attr);
    if (identity.empty()) {
      return;
    }
    const auto* int_value = attr->value.as<IntImmNode>();
    if (int_value == nullptr) {
      return;
    }
    auto& live_value = exact_output_live_form_value_by_buffer_identity_[identity];
    if (attr->attr_key == kBlackholeExactOutputLiveCBAttr) {
      const int cb_id = static_cast<int>(int_value->value);
      const int order = marker_order(attr);
      const auto existing_order_it =
          exact_output_live_form_order_by_buffer_identity_.find(identity);
      const bool replaces_existing =
          existing_order_it == exact_output_live_form_order_by_buffer_identity_.end() ||
          existing_order_it->second < 0 || order < 0 || order >= existing_order_it->second;
      if (replaces_existing) {
        exact_output_live_form_cb_by_buffer_identity_[identity] = cb_id;
        live_value.cb_id = cb_id;
        if (order >= 0) {
          exact_output_live_form_order_by_buffer_identity_[identity] = order;
        }
      }
    } else if (attr->attr_key == kBlackholeExactOutputLiveNumTilesAttr) {
      live_value.num_tiles = static_cast<int>(int_value->value);
    } else if (attr->attr_key == kBlackholeExactOutputLiveNumElementsAttr) {
      live_value.num_elements = int_value->value;
    } else if (attr->attr_key == kBlackholeExactOutputLiveRowWidthAttr) {
      live_value.row_width = int_value->value;
    }
  });
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

    auto preserve_definition_wrappers_as_noop = [](const Stmt& stmt) -> Stmt {
      std::vector<std::function<Stmt(Stmt)>> rewrap_stack;
      bool saw_definition_wrapper = false;
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
          saw_definition_wrapper = true;
          rewrap_stack.push_back([buffer = decl->buffer](Stmt body) {
            return DeclBuffer(buffer, body);
          });
          current = decl->body;
          continue;
        }
        if (const auto* allocate = current.as<AllocateNode>()) {
          saw_definition_wrapper = true;
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
      if (!saw_definition_wrapper) {
        return Stmt();
      }
      Stmt preserved = Evaluate(IntImm32(0));
      for (auto it = rewrap_stack.rbegin(); it != rewrap_stack.rend(); ++it) {
        preserved = (*it)(preserved);
      }
      return preserved;
    };

    auto preserve_definitions_or_drop = [&](const Stmt& stmt) {
      Stmt preserved = preserve_definition_wrappers_as_noop(stmt);
      if (preserved.defined()) {
        rewritten.push_back(preserved);
      }
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

    auto record_fragment_fill_fact = [&](const ExplicitFragmentFill& fill_match) {
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
    };

    auto is_representable_fragment_fill_before_typecast_materialization =
        [&](const Stmt& fill_stmt, const Stmt& typecast_stmt) -> bool {
      ExplicitFragmentFill fill_match;
      if (!match_explicit_fragment_fill(fill_stmt, &fill_match) ||
          !fill_match.dst.defined()) {
        return false;
      }

      const CallNode* typecast_call = unwrap_call(typecast_stmt);
      FragmentCastMatch cast_match;
      if (!MatchExplicitTileTypecast(typecast_call, &cast_match) ||
          !tir::is_zero(cast_match.src_offset) ||
          !cast_match.dst->dtype.is_bfloat16()) {
        return false;
      }

      Buffer fill_dst = ResolvePhysicalComputeBuffer(fill_match.dst);
      if (!fill_dst.defined()) {
        fill_dst = fill_match.dst;
      }
      if (!SameBufferIdentity(fill_dst, cast_match.src)) {
        return false;
      }

      const BlackholeBufferMaterializationFact* fact =
          FindBufferMaterializationFact(cast_match.dst);
      if (fact == nullptr ||
          fact->kind != buffer_materialization::kRepublishedLogicalTile ||
          fact->bridge_kind != buffer_materialization::kTileNFacesMaterialization ||
          fact->execution_protocol != buffer_materialization::kTiledCBRepublish) {
        return false;
      }

      const auto typecast_order_it = stmt_order_index_by_node_.find(typecast_stmt.get());
      const int typecast_order_index =
          typecast_order_it != stmt_order_index_by_node_.end()
              ? typecast_order_it->second
              : current_order_index + 1;
      const int cast_order_index = ResolveCurrentBufferTransferOrder(
          cast_match.src, cast_match.dst, typecast_order_index);
      const FutureBufferUses later_uses =
          ClassifyFutureBufferUses(fill_match.dst, cast_order_index);
      if (later_uses.has_compute_consume || later_uses.has_transport_consume ||
          later_uses.has_reference) {
        return false;
      }

      record_fragment_fill_fact(fill_match);
      for (const std::string& identity : CollectBufferFlowIdentities(cast_match.src)) {
        last_fragment_fill_value_by_buffer_identity_[identity] = fill_match.value;
      }
      if (const VarNode* data = BufferDataIdentity(cast_match.src)) {
        last_fragment_fill_value_by_data_[data] = fill_match.value;
      }
      return true;
    };

    auto is_representable_fragment_fill_before_next_write =
        [&](const Stmt& fill_stmt, int stmt_index) -> bool {
      ExplicitFragmentFill fill_match;
      if (!match_explicit_fragment_fill(fill_stmt, &fill_match) || !fill_match.dst.defined()) {
        return false;
      }
      auto future_loop_carries_fill_dst = [&]() -> bool {
        const std::vector<std::string> fill_identities =
            CollectBufferFlowIdentities(fill_match.dst);
        for (int next_index = stmt_index + 1;
             next_index < static_cast<int>(op->seq.size()); ++next_index) {
          const auto* loop = op->seq[next_index].as<ForNode>();
          if (loop == nullptr) {
            continue;
          }
          const std::unordered_set<std::string> loop_carried =
              CollectLoopCarriedBufferIdentities(loop->body);
          for (const std::string& identity : fill_identities) {
            if (!identity.empty() && loop_carried.count(identity) != 0U) {
              return true;
            }
          }
        }
        return false;
      };
      if (future_loop_carries_fill_dst()) {
        return false;
      }
      const Buffer physical_dst = ResolvePhysicalComputeBuffer(fill_match.dst);
      const Buffer dst = physical_dst.defined() ? physical_dst : fill_match.dst;
      const std::string scope = GetStorageScope(dst);
      if (scope != "blackhole.acc" && scope != "local.fragment") {
        return false;
      }
      if (!IsNegInfValue(fill_match.value)) {
        return false;
      }
      const VarNode* data = fill_match.data != nullptr ? fill_match.data : BufferDataIdentity(dst);
      for (int next_index = stmt_index + 1;
           next_index < static_cast<int>(op->seq.size()); ++next_index) {
        if (StmtUsesVarOutsideDefinitionsAndExactLiveAttr(op->seq[next_index], data) ||
            StmtReadsBufferOutsideDefinitionsAndExactLiveAttr(op->seq[next_index], dst)) {
          return false;
        }
      }
      record_fragment_fill_fact(fill_match);
      return true;
    };

    auto is_initial_identity_state_copy_before_clear_row_reduce =
        [&](const Stmt& copy_stmt, const Stmt& fill_stmt, const Stmt& reduce_stmt,
            const Stmt& combine_stmt, int stmt_index) -> bool {
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
      if (IsActiveLoopCarriedBuffer(copy_src)) {
        return false;
      }
      auto prior_serial_loop_carries_copy_src = [&]() -> bool {
        const std::vector<std::string> copy_src_identities =
            CollectBufferFlowIdentities(copy_src);
        for (int prev_index = 0; prev_index < stmt_index; ++prev_index) {
          const auto* loop = op->seq[prev_index].as<ForNode>();
          if (loop == nullptr) {
            continue;
          }
          const std::unordered_set<std::string> loop_carried =
              CollectLoopCarriedBufferIdentities(loop->body);
          for (const std::string& identity : copy_src_identities) {
            if (!identity.empty() && loop_carried.count(identity) != 0U) {
              return true;
            }
          }
        }
        return false;
      };
      if (prior_serial_loop_carries_copy_src()) {
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
      preserve_definitions_or_drop(op->seq[i]);
      continue;
    }
    if (i + 1 < static_cast<int>(op->seq.size()) &&
        is_redundant_identity_fill_before_clear_row_reduce(op->seq[i], op->seq[i + 1])) {
      preserve_definitions_or_drop(op->seq[i]);
      continue;
    }
    if (!select_compute_builtins_only_ &&
        i + 1 < static_cast<int>(op->seq.size()) &&
        is_representable_fragment_fill_before_typecast_materialization(op->seq[i], op->seq[i + 1])) {
      preserve_definitions_or_drop(op->seq[i]);
      continue;
    }
    if (is_representable_fragment_fill_before_next_write(op->seq[i], i)) {
      preserve_definitions_or_drop(op->seq[i]);
      continue;
    }
    if (i + 3 < static_cast<int>(op->seq.size()) &&
        is_initial_identity_state_copy_before_clear_row_reduce(
            op->seq[i], op->seq[i + 1], op->seq[i + 2], op->seq[i + 3], i)) {
      preserve_definitions_or_drop(op->seq[i]);
      continue;
    }
    rewritten.push_back(VisitStmt(op->seq[i]));
  }
  return SeqStmt::Flatten(rewritten);
}

std::unordered_set<std::string> PlanTTKernelABI::CollectLoopCarriedBufferIdentities(
    const Stmt& body) const {
  struct AccessState {
    bool saw_write = false;
    bool read_before_write = false;
    bool has_write = false;
  };
  std::unordered_map<std::string, AccessState> state_by_identity;

  auto record_buffer = [&](const Buffer& buffer, bool is_write) {
    if (!buffer.defined()) {
      return;
    }
    for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
      if (identity.empty()) {
        continue;
      }
      AccessState& state = state_by_identity[identity];
      if (is_write) {
        state.saw_write = true;
        state.has_write = true;
      } else if (!state.saw_write) {
        state.read_before_write = true;
      }
    }
  };

  auto record_buffer_arg = [&](const CallNode* call, size_t index, bool is_write) {
    if (call == nullptr || index >= call->args.size()) {
      return;
    }
    if (IsBufferLikeExpr(call->args[index])) {
      record_buffer(NormalizeToBufferRegion(call->args[index])->buffer, is_write);
      return;
    }
    if (const auto* data = call->args[index].as<VarNode>()) {
      auto physical_it = compute_physical_buffers_by_data_.find(data);
      if (physical_it != compute_physical_buffers_by_data_.end()) {
        record_buffer(physical_it->second, is_write);
      }
    }
  };

  class Collector : public StmtExprVisitor {
   public:
    Collector(const PlanTTKernelABI* abi,
              std::function<void(const Buffer&, bool)> record_buffer,
              std::function<void(const CallNode*, size_t, bool)> record_buffer_arg)
        : abi_(abi),
          record_buffer_(std::move(record_buffer)),
          record_buffer_arg_(std::move(record_buffer_arg)) {}

    void VisitStmt_(const BufferStoreNode* op) final {
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->value);
      record_buffer_(op->buffer, /*is_write=*/true);
    }

    void VisitExpr_(const BufferLoadNode* op) final {
      for (const PrimExpr& index : op->indices) {
        VisitExpr(index);
      }
      record_buffer_(op->buffer, /*is_write=*/false);
    }

    void VisitExpr_(const CallNode* op) final {
      bool handled = false;
      if (op->op->IsInstance<OpNode>()) {
        const Op call_op = Downcast<Op>(op->op);
        if (call_op->name == "tl.blackhole.cast_fragment_slice" && op->args.size() >= 2U) {
          record_buffer_arg_(op, 1, /*is_write=*/false);
          record_buffer_arg_(op, 0, /*is_write=*/true);
          handled = true;
        } else if (call_op->name == "tl.blackhole.fill_fragment" && !op->args.empty()) {
          record_buffer_arg_(op, 0, /*is_write=*/true);
          handled = true;
        }
        if (call_op->name == blackhole_tile_compute_schema::kOpName && !op->args.empty()) {
          if (const auto* operation = op->args[0].as<StringImmNode>()) {
            if (const BlackholeTileComputePattern* pattern =
                    FindBlackholeTileComputePattern(operation->value)) {
              for (const BlackholeTileComputeCallOperand& operand :
                   pattern->blackhole_compute_operands) {
                switch (operand.role) {
                  case BlackholeTileComputeOperandRole::kInput:
                  case BlackholeTileComputeOperandRole::kLhs:
                  case BlackholeTileComputeOperandRole::kRhs:
                  case BlackholeTileComputeOperandRole::kA:
                  case BlackholeTileComputeOperandRole::kB:
                  case BlackholeTileComputeOperandRole::kScaler:
                    record_buffer_arg_(op, operand.arg_index, /*is_write=*/false);
                    break;
                  case BlackholeTileComputeOperandRole::kOutput:
                  case BlackholeTileComputeOperandRole::kC:
                    break;
                }
              }
              for (const BlackholeTileComputeCallOperand& operand :
                   pattern->blackhole_compute_operands) {
                if (operand.role == BlackholeTileComputeOperandRole::kOutput ||
                    operand.role == BlackholeTileComputeOperandRole::kC) {
                  record_buffer_arg_(op, operand.arg_index, /*is_write=*/true);
                }
              }
              handled = true;
            }
          }
        }
      }

      RowReductionMatch reduce_match;
      if (abi_->MatchExplicitTileReduce(op, &reduce_match)) {
        record_buffer_(reduce_match.src, /*is_write=*/false);
        if (reduce_match.accumulate_existing) {
          record_buffer_(reduce_match.dst, /*is_write=*/false);
        }
        record_buffer_(reduce_match.dst, /*is_write=*/true);
        handled = true;
      }

      FragmentCastMatch cast_match;
      if (abi_->MatchExplicitTileTypecast(op, &cast_match)) {
        record_buffer_(cast_match.src, /*is_write=*/false);
        record_buffer_(cast_match.dst, /*is_write=*/true);
        handled = true;
      }

      if (abi_->IsMatmulCall(op)) {
        record_buffer_arg_(op, 0, /*is_write=*/false);
        record_buffer_arg_(op, 1, /*is_write=*/false);
        record_buffer_arg_(op, 2, /*is_write=*/true);
        handled = true;
      }

      if (!handled) {
        StmtExprVisitor::VisitExpr_(op);
      }
    }

   private:
    const PlanTTKernelABI* abi_;
    std::function<void(const Buffer&, bool)> record_buffer_;
    std::function<void(const CallNode*, size_t, bool)> record_buffer_arg_;
  };

  Collector collector(this, record_buffer, record_buffer_arg);
  collector(body);

  std::unordered_set<std::string> loop_carried;
  for (const auto& [identity, state] : state_by_identity) {
    if (state.read_before_write && state.has_write) {
      loop_carried.insert(identity);
    }
  }
  return loop_carried;
}

Stmt PlanTTKernelABI::InitializeLoopCarriedExactLiveForms(
    const std::unordered_set<std::string>& loop_carried_identities) {
  std::vector<Stmt> stmts;
  std::unordered_set<int> initialized_cb_ids;
  auto loop_carried_program_point = [&]() {
    int program_point = current_lowering_order_index_;
    if (!active_serial_loop_order_ranges_.empty()) {
      const auto& loop_range = active_serial_loop_order_ranges_.back();
      if (loop_range.second >= 0) {
        program_point = loop_range.second;
      }
    }
    return program_point;
  };

  auto resolve_identity_buffer = [&](const std::string& identity) -> Buffer {
    auto physical_it = compute_physical_buffers_by_identity_.find(identity);
    if (physical_it != compute_physical_buffers_by_identity_.end() &&
        physical_it->second.defined()) {
      return physical_it->second;
    }
    auto buffer_it = buffer_by_identity_.find(identity);
    if (buffer_it != buffer_by_identity_.end() && buffer_it->second.defined()) {
      return buffer_it->second;
    }
    return Buffer();
  };

  for (const auto& [identity, cb_id] : exact_output_live_form_cb_by_buffer_identity_) {
    if (identity.empty() || HasLoopCarriedExactCBState(identity)) {
      continue;
    }
    Buffer buffer = resolve_identity_buffer(identity);
    if (!buffer.defined() || GetStorageScope(buffer) != "blackhole.acc" ||
        !IsSingleFullTileLogicalMatrix(buffer)) {
      continue;
    }
    ExactTiledCBValue state_value;
    state_value.buffer = buffer;
    state_value.cb_id = cb_id;
    state_value.borrowed_live = true;
    state_value.live_identity = identity;
    PopulateExactTiledCBValueShape(buffer, &state_value);
    RefineExactTiledCBValueShapeFromRequirement(&state_value);
    RememberLoopCarriedExactCBState(identity, state_value,
                                    loop_carried_program_point());
  }

  for (const std::string& identity : loop_carried_identities) {
    if (identity.empty() || HasLoopCarriedExactCBState(identity)) {
      continue;
    }
    Buffer buffer = resolve_identity_buffer(identity);
    if (!buffer.defined() || !IsSingleFullTileLogicalMatrix(buffer)) {
      continue;
    }

    PrimExpr fill_value;
    auto fill_it = last_fragment_fill_value_by_buffer_identity_.find(identity);
    if (fill_it != last_fragment_fill_value_by_buffer_identity_.end()) {
      fill_value = fill_it->second;
    } else if (!TryGetLastFragmentFillValue(buffer, &fill_value)) {
      ExactTiledCBValue existing_value;
      if (TryCreateExactOutputLiveTiledCBValue(buffer, &existing_value) ||
          TryCreateLiveExactTiledCBValue(buffer, &existing_value)) {
        if (!existing_value.buffer.defined()) {
          existing_value.buffer = buffer;
        }
        RememberLoopCarriedExactCBState(identity, existing_value,
                                        loop_carried_program_point());
        for (const std::string& alias : CollectBufferFlowIdentities(buffer)) {
          if (!alias.empty() && loop_carried_identities.count(alias) != 0U) {
            ExactTiledCBValue alias_value = existing_value;
            alias_value.live_identity = alias;
            RememberLoopCarriedExactCBState(alias, alias_value,
                                            loop_carried_program_point());
          }
        }
      }
      continue;
    }

    const std::string live_form_name =
        BufferIdentityName(buffer) + "_loop_carried_live_form_" +
        std::to_string(next_requirement_index_);
    Buffer live_form_buffer =
        tir::decl_buffer(buffer->shape, ExactTiledCBStorageDType(buffer->dtype),
                         live_form_name, GetStorageScope(buffer));
    const int live_form_cb_id = AllocateRequirementIndex(live_form_buffer, CBType::kIntermediate);
    ExactTiledCBValue initial_value;
    initial_value.buffer = buffer;
    initial_value.cb_id = live_form_cb_id;
    initial_value.borrowed_live = true;
    initial_value.live_identity = identity;
    PopulateExactTiledCBValueShape(buffer, &initial_value);
    RefineExactTiledCBValueShapeFromRequirement(&initial_value);

    const DataType storage_dtype = ExactTiledCBStorageDType(buffer->dtype);
    SetRequirementPageLayout(live_form_cb_id,
                             kBlackholeTileRows * kBlackholeTileCols * storage_dtype.bytes(),
                             initial_value.num_tiles);
    auto& req = cb_requirements_.at(live_form_cb_id);
    req.data_format = DataTypeToDataFormatForBlackhole(storage_dtype);
    req.flow_class = CBFlowClass::kState;
    req.publish_pages_per_event =
        std::max(req.publish_pages_per_event, initial_value.num_tiles);
    req.consume_pages_per_event =
        std::max(req.consume_pages_per_event, initial_value.num_tiles);

    ExactTiledCBValue live_form_state = initial_value;
    live_form_state.buffer = live_form_buffer;
    RememberLoopCarriedExactCBState(identity, live_form_state,
                                    loop_carried_program_point());
    for (const std::string& alias : CollectBufferFlowIdentities(buffer)) {
      if (!alias.empty() && loop_carried_identities.count(alias) != 0U) {
        ExactTiledCBValue alias_value = live_form_state;
        alias_value.live_identity = alias;
        RememberLoopCarriedExactCBState(alias, alias_value,
                                        loop_carried_program_point());
      }
    }
    if (initialized_cb_ids.insert(live_form_cb_id).second) {
      Stmt publish = PublishConstantToExactTiledCB(buffer, fill_value, initial_value);
      stmts.push_back(AttachExactOutputLiveFormMarker(buffer, initial_value, publish));
    }
    RecordTiledCBLiveFormAliases(buffer, live_form_cb_id);
  }

  if (stmts.empty()) {
    return Stmt();
  }
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::IsActiveLoopCarriedBuffer(const Buffer& buffer) const {
  if (!buffer.defined() || active_loop_carried_buffer_identity_stack_.empty()) {
    return false;
  }
  const std::vector<std::string> identities = CollectBufferFlowIdentities(buffer);
  for (auto stack_it = active_loop_carried_buffer_identity_stack_.rbegin();
       stack_it != active_loop_carried_buffer_identity_stack_.rend(); ++stack_it) {
    for (const std::string& identity : identities) {
      if (!identity.empty() && stack_it->count(identity) != 0U) {
        return true;
      }
    }
  }
  return false;
}

bool PlanTTKernelABI::IsCompletedLoopCarriedBuffer(const Buffer& buffer) const {
  if (!buffer.defined() || loop_carried_exact_cb_state_by_logical_value_.empty()) {
    return false;
  }
  const std::vector<std::string> identities = CollectBufferFlowIdentities(buffer);
  for (const std::string& identity : identities) {
    const LoopCarriedExactCBState* state = FindLoopCarriedExactCBState(identity);
    if (state != nullptr && state->completed) {
      return true;
    }
  }
  return false;
}

bool PlanTTKernelABI::ShouldMaterializeLoopCarriedExactOutput(const Buffer& dst) const {
  if (IsActiveLoopCarriedBuffer(dst)) {
    if (IsSingleFullTileLogicalMatrix(dst)) {
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(dst, current_lowering_order_index_);
      return future_uses.has_reference ||
             FutureWritePrecedesFutureComputeConsume(dst, current_lowering_order_index_);
    }
    return true;
  }
  if (!IsCompletedLoopCarriedBuffer(dst)) {
    return false;
  }
  const FutureBufferUses future_uses =
      ClassifyFutureBufferUses(dst, current_lowering_order_index_);
  return future_uses.has_reference ||
         FutureWritePrecedesFutureComputeConsume(dst, current_lowering_order_index_);
}

Stmt PlanTTKernelABI::MaterializeLoopCarriedExactOutput(
    const Buffer& dst, const ExactTiledCBValue& cb_value) {
  if (!ShouldMaterializeLoopCarriedExactOutput(dst)) {
    return Stmt();
  }
  InvalidateLastFragmentFillValue(dst);
  ClearSelectedSourceLiveProducer(dst);
  ClearTiledCBLiveFormAliases(dst);
  MarkLocalOnlyLiveFormAliases(dst);
  Stmt materialize = MaterializeExactTiledCBToLocalBuffer(dst, cb_value, /*pop_front=*/true);
  ClearTiledCBLiveFormAliases(dst);
  MarkLocalOnlyLiveFormAliases(dst);
  return materialize;
}

Stmt PlanTTKernelABI::VisitStmt_(const ForNode* op) {
  const bool zero_loop_var = !op->thread_binding.defined();
  const Var transport_loop_var = zero_loop_var ? op->loop_var : Var();
  if (select_compute_builtins_only_ && IsPureCopyLoopNest(op->body)) {
    std::vector<Var> loop_stack;
    std::vector<NestedCopyMatch> matches;
    CollectNestedCopyStores(op->body, &loop_stack, &matches);
    for (const NestedCopyMatch& match : matches) {
      if (match.direction != CopyDirection::kCBToLocal || match.store == nullptr) {
        continue;
      }
      const auto* load = match.store->value.as<BufferLoadNode>();
      if (load == nullptr) {
        continue;
      }
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(match.store->buffer, current_lowering_order_index_);
      bool has_tile_compute_input_use = future_uses.has_compute_consume;
      for (const std::string& identity : CollectBufferFlowIdentities(match.store->buffer)) {
        if (tile_compute_input_buffers_.count(identity) != 0U) {
          has_tile_compute_input_use = true;
          break;
        }
      }
      if (!has_tile_compute_input_use) {
        continue;
      }
      const int src_cb_id =
          AllocateRequirementIndex(match.store->buffer, CBType::kIntermediate);
      auto& req = cb_requirements_.at(src_cb_id);
      req.lifetime_end = std::max(req.lifetime_end, next_requirement_index_);
      RecordTiledCBLiveFormAliases(match.store->buffer, src_cb_id);
    }
  }
  if (!select_compute_builtins_only_ && IsPureCopyLoopNest(op->body)) {
    std::vector<Var> loop_stack;
    std::vector<NestedCopyMatch> matches;
    CollectNestedCopyStores(op->body, &loop_stack, &matches);
    if (!matches.empty()) {
      if (matches.size() == 1U && matches[0].direction == CopyDirection::kDramToCB &&
          IsBroadcastColsSourceBuffer(matches[0].store->buffer)) {
        saw_copy_op_ = true;
        std::vector<Var> loop_vars_to_zero;
        if (transport_loop_var.defined()) {
          loop_vars_to_zero.push_back(transport_loop_var);
        }
        loop_vars_to_zero.insert(loop_vars_to_zero.end(), matches[0].loop_vars.begin(),
                                 matches[0].loop_vars.end());
        return GenerateCopySequence(matches[0].store, loop_vars_to_zero);
      }

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
            match.direction != CopyDirection::kCBToDram &&
            match.direction != CopyDirection::kDramToLocal) {
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
      if (direction == CopyDirection::kDramToCB ||
          direction == CopyDirection::kCBToDram ||
          direction == CopyDirection::kDramToLocal) {
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
      if (direction == CopyDirection::kCBToLocal) {
        saw_copy_op_ = true;
        return GenerateCopySequence(nested_store);
      }
    }
    if (const auto* store = op->body.as<BufferStoreNode>()) {
      if (IsCopyOperation(store)) {
        CopyDirection direction = GetCopyDirection(store);
        if (direction == CopyDirection::kDramToCB ||
            direction == CopyDirection::kCBToDram ||
            direction == CopyDirection::kDramToLocal) {
          saw_copy_op_ = true;
          PrimExpr tile_index = InferCopyTileIndex(store, transport_loop_var);
          return GenerateCopySequence(store, tile_index);
        }
        if (direction == CopyDirection::kCBToLocal) {
          saw_copy_op_ = true;
          return GenerateCopySequence(store);
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
  if (!select_compute_builtins_only_) {
    active_serial_loop_vars_.push_back(op->loop_var);
  }
  auto compute_loop_body_order_range = [&]() -> std::pair<int, int> {
    int min_order = std::numeric_limits<int>::max();
    int max_order = -1;
    for (const Stmt& stmt : CollectExecutionOrderedStmts(op->body)) {
      auto order_it = stmt_order_index_by_node_.find(stmt.get());
      if (order_it == stmt_order_index_by_node_.end()) {
        continue;
      }
      min_order = std::min(min_order, order_it->second);
      max_order = std::max(max_order, order_it->second);
    }
    if (max_order < 0) {
      return {-1, -1};
    }
    return {min_order, max_order};
  };
  active_serial_loop_order_ranges_.push_back(compute_loop_body_order_range());
  const std::unordered_set<std::string> loop_carried_identities =
      CollectLoopCarriedBufferIdentities(op->body);
  active_loop_carried_buffer_identity_stack_.push_back(loop_carried_identities);
  Stmt loop_carried_init = InitializeLoopCarriedExactLiveForms(loop_carried_identities);
  Stmt lowered = StmtExprMutator::VisitStmt_(op);
  auto resolve_loop_carried_identity_buffer = [&](const std::string& identity) -> Buffer {
    Buffer state_buffer = GetLoopCarriedExactCBBuffer(identity);
    if (state_buffer.defined()) {
      return state_buffer;
    }
    auto buffer_it = buffer_by_identity_.find(identity);
    if (buffer_it != buffer_by_identity_.end() && buffer_it->second.defined()) {
      return buffer_it->second;
    }
    auto physical_it = compute_physical_buffers_by_identity_.find(identity);
    if (physical_it != compute_physical_buffers_by_identity_.end() &&
        physical_it->second.defined()) {
      return physical_it->second;
    }
    return Buffer();
  };
  auto should_drop_loop_carried_exit_pop = [&](const Stmt& stmt) -> bool {
    const auto* eval = stmt.as<EvaluateNode>();
    if (eval == nullptr) {
      return false;
    }
    const auto* call = eval->value.as<CallNode>();
    if (call == nullptr || !call->op->IsInstance<OpNode>() || call->args.size() < 2U) {
      return false;
    }
    const Op call_op = Downcast<Op>(call->op);
    if (!call_op.same_as(blackhole_cb_pop_front())) {
      return false;
    }
    const auto* cb_id = call->args[0].as<IntImmNode>();
    if (cb_id == nullptr) {
      return false;
    }
    for (const std::string& identity : loop_carried_identities) {
      if (GetLoopCarriedExactCBId(identity) != cb_id->value) {
        continue;
      }
      Buffer buffer = resolve_loop_carried_identity_buffer(identity);
      if (!buffer.defined()) {
        continue;
      }
      const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
      const Buffer state_buffer = physical.defined() ? physical : buffer;
      if (!state_buffer.defined() || GetStorageScope(state_buffer) != "blackhole.acc" ||
          !IsSingleFullTileLogicalMatrix(state_buffer)) {
        continue;
      }
      const auto& loop_range = active_serial_loop_order_ranges_.back();
      const int future_query_order = loop_range.second >= 0 ? loop_range.second
                                                            : current_lowering_order_index_;
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(buffer, future_query_order);
      if (future_uses.has_compute_consume || future_uses.has_transport_consume ||
          future_uses.has_reference) {
        return true;
      }
    }
    return false;
  };
  if (const auto* seq = lowered.as<SeqStmtNode>()) {
    if (!seq->seq.empty() && should_drop_loop_carried_exit_pop(seq->seq.back())) {
      Array<Stmt> kept;
      for (size_t i = 0; i + 1 < seq->seq.size(); ++i) {
        kept.push_back(seq->seq[i]);
      }
      lowered = SeqStmt::Flatten(kept);
    }
  }
  active_loop_carried_buffer_identity_stack_.pop_back();
  active_serial_loop_order_ranges_.pop_back();
  for (const std::string& identity : loop_carried_identities) {
    InvalidateLastFragmentFillValueIdentity(identity);
    MarkLoopCarriedExactCBStateCompleted(identity);
  }
  if (!select_compute_builtins_only_) {
    active_serial_loop_vars_.pop_back();
  }
  if (loop_carried_init.defined()) {
    std::vector<Stmt> loop_with_init{loop_carried_init, lowered};
    return SeqStmt::Flatten(loop_with_init);
  }
  return lowered;
}

// StmtExprMutator overrides
// Note: We only override specific node types and return the original node
// for unmatched patterns to avoid deep recursion that causes stack overflow.
Stmt PlanTTKernelABI::VisitStmt_(const EvaluateNode* op) {
  auto should_drop_completed_loop_carried_state_pop = [&](const CallNode* call) -> bool {
    if (call == nullptr || !active_loop_carried_buffer_identity_stack_.empty() ||
        !call->op->IsInstance<OpNode>() || call->args.size() < 2U) {
      return false;
    }
    const Op call_op = Downcast<Op>(call->op);
    if (!call_op.same_as(blackhole_cb_pop_front())) {
      return false;
    }
    const auto* cb_id = call->args[0].as<IntImmNode>();
    if (cb_id == nullptr) {
      return false;
    }
    for (const auto& [identity, state] : loop_carried_exact_cb_state_by_logical_value_) {
      if (state.cb_id != cb_id->value || !state.completed) {
        continue;
      }
      Buffer buffer;
      auto buffer_it = buffer_by_identity_.find(identity);
      if (buffer_it != buffer_by_identity_.end() && buffer_it->second.defined()) {
        buffer = buffer_it->second;
      } else if (state.buffer.defined()) {
        buffer = state.buffer;
      } else {
        auto physical_it = compute_physical_buffers_by_identity_.find(identity);
        if (physical_it != compute_physical_buffers_by_identity_.end() &&
            physical_it->second.defined()) {
          buffer = physical_it->second;
        }
      }
      if (!buffer.defined()) {
        continue;
      }
      const Buffer physical = ResolvePhysicalComputeBuffer(buffer);
      const Buffer state_buffer = physical.defined() ? physical : buffer;
      if (!state_buffer.defined() || GetStorageScope(state_buffer) != "blackhole.acc" ||
          !IsSingleFullTileLogicalMatrix(state_buffer)) {
        continue;
      }
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(buffer, current_lowering_order_index_);
      if (future_uses.has_compute_consume || future_uses.has_transport_consume ||
          future_uses.has_reference) {
        return true;
      }
    }
    return false;
  };
  if (select_compute_builtins_only_) {
    if (const auto* call = op->value.as<CallNode>()) {
      if (should_drop_completed_loop_carried_state_pop(call)) {
        return Evaluate(IntImm32(0));
      }
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
        const auto* clear_accum = call->args.size() > 9 ? call->args[9].as<IntImmNode>() : nullptr;
        const bool matmul_publishes_c_directly = clear_accum != nullptr && clear_accum->value != 0;
        const bool selected_source_shape_is_stable =
            matmul_publishes_c_directly || active_loop_carried_buffer_identity_stack_.empty();
        if (IsSingleFullTileMatmulOutput(call) && selected_source_shape_is_stable &&
            !IsActiveLoopCarriedBuffer(out_buffer)) {
          RecordSelectedSourceLiveProducer(out_buffer);
        }
      }
    }
    return GetRef<Stmt>(op);
  }
  if (const auto* call = op->value.as<CallNode>()) {
    if (should_drop_completed_loop_carried_state_pop(call)) {
      return Evaluate(IntImm32(0));
    }
    if (Stmt explicit_compute = LowerExplicitTileComputeCall(call);
        explicit_compute.defined()) {
      return explicit_compute;
    }
    if (call->op->IsInstance<OpNode>()) {
      const Op call_op = Downcast<Op>(call->op);
      if (call_op.same_as(blackhole_cb_pop_front()) && call->args.size() >= 1) {
        if (const auto* cb_id = call->args[0].as<IntImmNode>()) {
          if (IsBroadcastColsSourceCBId(static_cast<int>(cb_id->value))) {
            return Evaluate(IntImm32(0));
          }
        }
      }
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
    const CopyDirection direction = GetCopyDirection(op);
    if (direction == CopyDirection::kDramToCB && IsBroadcastColsSourceBuffer(op->buffer) &&
        !active_serial_loop_vars_.empty()) {
      Stmt lowered = GenerateCopySequence(op, active_serial_loop_vars_);
      PrimExpr condition;
      for (const Var& loop_var : active_serial_loop_vars_) {
        const PrimExpr is_zero = tir::EQ(loop_var, IntImm32(0));
        condition = condition.defined() ? (condition && is_zero) : is_zero;
      }
      return tir::IfThenElse(condition, lowered);
    }
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

}  // namespace tl
}  // namespace tvm
