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
 * \brief Implementation of LowerBlackholeOps pass.
 *
 * Transforms TileLang high-level operations (T.copy, T.gemm, T.clear)
 * into TT-Metal builtin sequences.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_utils.h"
#include "common/blackhole_runtime_arg_schema.h"
#include "common/companion_base.h"
#include "common/semantic_program.h"
#include "common/spatial_program.h"
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

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

static std::string GemmWarpPolicyTypeToStringForBlackhole(int policy_type) {
  switch (policy_type) {
    case 0:
      return "Square";
    case 1:
      return "FullRow";
    case 2:
      return "FullCol";
    case 3:
      return "Free";
    default:
      return "Unknown";
  }
}

static std::string PrimExprToCompactString(const PrimExpr& expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

static std::string EncodeGemmContractSignature(
    const std::string& a_buffer, const std::string& b_buffer, const std::string& c_buffer, int m,
    int n, int k, bool transpose_a, bool transpose_b, int policy_type, bool clear_accum,
    int k_pack, int wg_wait, bool dst_full_sync_en, bool bfp8_pack_precise,
    const std::vector<std::pair<std::string, std::string>>& defines,
    const std::vector<std::pair<std::string, uint32_t>>& named_compile_args,
    const std::string& mbarrier_buffer, const std::string& mbarrier_scope,
    const std::vector<std::string>& mbarrier_index_exprs) {
  std::ostringstream os;
  os << a_buffer << "|" << b_buffer << "|" << c_buffer << "|" << m << "|" << n << "|" << k
     << "|" << transpose_a << "|" << transpose_b << "|" << policy_type << "|" << clear_accum
     << "|" << k_pack << "|" << wg_wait << "|" << dst_full_sync_en << "|"
     << bfp8_pack_precise << "|" << mbarrier_buffer << "|" << mbarrier_scope;
  for (const auto& [name, value] : defines) {
    os << "|define:" << name << "=" << value;
  }
  for (const auto& [name, value] : named_compile_args) {
    os << "|arg:" << name << "=" << value;
  }
  for (const auto& expr : mbarrier_index_exprs) {
    os << "|mbar:" << expr;
  }
  return os.str();
}

static Array<Any> EncodeNamedStringPairs(
    const std::vector<std::pair<std::string, std::string>>& entries) {
  Array<Any> encoded_entries;
  for (const auto& [name, value] : entries) {
    Map<String, Any> entry;
    entry.Set("name", String(name));
    entry.Set("value", String(value));
    encoded_entries.push_back(entry);
  }
  return encoded_entries;
}

static Array<Any> EncodeNamedUint32Pairs(
    const std::vector<std::pair<std::string, uint32_t>>& entries) {
  Array<Any> encoded_entries;
  for (const auto& [name, value] : entries) {
    Map<String, Any> entry;
    entry.Set("name", String(name));
    entry.Set("value", Integer(static_cast<int>(value)));
    encoded_entries.push_back(entry);
  }
  return encoded_entries;
}

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
      return fragment_flow::kStream;
    case CBFlowClass::kRepublish:
      return fragment_flow::kRepublish;
    case CBFlowClass::kState:
    default:
      return fragment_flow::kState;
  }
}

static std::optional<CBFlowClass> ParseCBFlowClass(const std::string& flow_class) {
  if (flow_class == fragment_flow::kState) {
    return CBFlowClass::kState;
  }
  if (flow_class == fragment_flow::kStream) {
    return CBFlowClass::kStream;
  }
  if (flow_class == fragment_flow::kRepublish) {
    return CBFlowClass::kRepublish;
  }
  return std::nullopt;
}

static Map<String, Any> BuildGemmContractPayload(
    const std::string& a_buffer, const std::string& b_buffer, const std::string& c_buffer, int m,
    int n, int k, bool transpose_a, bool transpose_b, DataType a_dtype, DataType b_dtype,
    DataType c_dtype) {
  Map<String, Any> gemm_contract;
  gemm_contract.Set("a_buffer", String(a_buffer));
  gemm_contract.Set("b_buffer", String(b_buffer));
  gemm_contract.Set("c_buffer", String(c_buffer));
  gemm_contract.Set("M", Integer(m));
  gemm_contract.Set("N", Integer(n));
  gemm_contract.Set("K", Integer(k));
  gemm_contract.Set("transpose_A", Bool(transpose_a));
  gemm_contract.Set("transpose_B", Bool(transpose_b));
  gemm_contract.Set("a_tensor_dtype", String(DataTypeToDataFormatForBlackhole(a_dtype)));
  gemm_contract.Set("b_tensor_dtype", String(DataTypeToDataFormatForBlackhole(b_dtype)));
  gemm_contract.Set("c_tensor_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  gemm_contract.Set("a_cb_dtype", String(DataTypeToDataFormatForBlackhole(a_dtype)));
  gemm_contract.Set("b_cb_dtype", String(DataTypeToDataFormatForBlackhole(b_dtype)));
  gemm_contract.Set("c_cb_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  gemm_contract.Set("accumulator_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  return gemm_contract;
}

static Map<String, Any> BuildComputeContractPayload(
    const std::string& a_buffer, const std::string& b_buffer, const std::string& c_buffer, int m,
    int n, int k, bool transpose_a, bool transpose_b, int policy_type, bool clear_accum,
    int k_pack, int wg_wait, bool dst_full_sync_en, bool bfp8_pack_precise,
    const std::vector<std::pair<std::string, std::string>>& defines,
    const std::vector<std::pair<std::string, uint32_t>>& named_compile_args,
    const std::string& mbarrier_buffer, const std::string& mbarrier_scope,
    const std::vector<std::string>& mbarrier_index_exprs, DataType a_dtype, DataType b_dtype,
    DataType c_dtype) {
  Map<String, Any> compute_contract;
  compute_contract.Set("enabled", Bool(true));
  compute_contract.Set("kind", String("gemm"));
  compute_contract.Set("a_buffer", String(a_buffer));
  compute_contract.Set("b_buffer", String(b_buffer));
  compute_contract.Set("c_buffer", String(c_buffer));
  compute_contract.Set("M", Integer(m));
  compute_contract.Set("N", Integer(n));
  compute_contract.Set("K", Integer(k));
  compute_contract.Set("Mt", Integer(m / 32));
  compute_contract.Set("Nt", Integer(n / 32));
  compute_contract.Set("Kt", Integer(k / 32));
  compute_contract.Set("block_m_tiles", Integer(m / 32));
  compute_contract.Set("block_n_tiles", Integer(n / 32));
  compute_contract.Set("block_k_tiles", Integer(k / 32));
  compute_contract.Set("subblock_m_tiles", Integer(m / 32));
  compute_contract.Set("subblock_n_tiles", Integer(n / 32));
  compute_contract.Set("transpose_A", Bool(transpose_a));
  compute_contract.Set("transpose_B", Bool(transpose_b));
  compute_contract.Set("policy_type", Integer(policy_type));
  compute_contract.Set("policy_name", String(GemmWarpPolicyTypeToStringForBlackhole(policy_type)));
  compute_contract.Set("has_mbarrier", Bool(!mbarrier_buffer.empty()));
  compute_contract.Set("mbarrier_buffer", String(mbarrier_buffer));
  compute_contract.Set("mbarrier_scope", String(mbarrier_scope));
  Array<Any> encoded_mbarrier_index_exprs;
  for (const auto& expr : mbarrier_index_exprs) {
    encoded_mbarrier_index_exprs.push_back(String(expr));
  }
  compute_contract.Set("mbarrier_index_exprs", encoded_mbarrier_index_exprs);
  compute_contract.Set("a_tensor_dtype", String(DataTypeToDataFormatForBlackhole(a_dtype)));
  compute_contract.Set("b_tensor_dtype", String(DataTypeToDataFormatForBlackhole(b_dtype)));
  compute_contract.Set("c_tensor_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  compute_contract.Set("a_cb_dtype", String(DataTypeToDataFormatForBlackhole(a_dtype)));
  compute_contract.Set("b_cb_dtype", String(DataTypeToDataFormatForBlackhole(b_dtype)));
  compute_contract.Set("c_cb_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  compute_contract.Set("accumulator_dtype", String(DataTypeToDataFormatForBlackhole(c_dtype)));
  compute_contract.Set("math_fidelity", String("HiFi4"));
  compute_contract.Set("fp32_dest_acc_en", Bool(true));
  compute_contract.Set("dst_full_sync_en", Bool(dst_full_sync_en));
  compute_contract.Set("math_approx_mode", Bool(false));
  compute_contract.Set("unpack_to_dest_mode", Array<Any>{});
  compute_contract.Set("bfp8_pack_precise", Bool(bfp8_pack_precise));
  compute_contract.Set("defines", EncodeNamedStringPairs(defines));
  compute_contract.Set("named_compile_args", EncodeNamedUint32Pairs(named_compile_args));
  compute_contract.Set("clear_accum", Bool(clear_accum));
  compute_contract.Set("k_pack", Integer(k_pack));
  compute_contract.Set("wg_wait", Integer(wg_wait));
  return compute_contract;
}

static Map<String, Any> MakeComputeEpilogueOpPayload(const char* kind,
                                                     const std::string& dst_buffer) {
  Map<String, Any> op_payload;
  op_payload.Set("kind", String(kind));
  if (!dst_buffer.empty()) {
    op_payload.Set("dst_buffer", String(dst_buffer));
  }
  return op_payload;
}

static void SetOptionalBufferField(Map<String, Any>* payload, const char* key,
                                   const Buffer& buffer) {
  if (buffer.defined()) {
    payload->Set(String(key), String(BufferIdentityName(buffer)));
  }
}

static void SetOptionalExprField(Map<String, Any>* payload, const char* key, const PrimExpr& expr) {
  if (expr.defined()) {
    payload->Set(String(key), String(PrimExprToCompactString(expr)));
  }
}

static std::string MakeBlackholeRuntimeArgIdentity(const std::string& kind, const std::string& name,
                                                   const std::string& buffer_name = "") {
  if (!buffer_name.empty()) {
    return kind + ":" + buffer_name;
  }
  return !kind.empty() ? kind : name;
}

static std::string MakeSegmentBufferKey(const std::string& segment_kind,
                                        const tir::Buffer& buffer) {
  std::ostringstream os;
  os << segment_kind << ":"
     << reinterpret_cast<uintptr_t>(BufferDataIdentity(buffer));
  return os.str();
}

static bool IsBufferAddrRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr32" || kind == "input_buffer_addr" ||
         kind == "output_buffer_addr32" || kind == "output_buffer_addr";
}

static Array<Any> EnsureSegmentBufferRuntimeArgs(const std::string& segment_kind,
                                                 const Array<Any>& accessors,
                                                 const Optional<Any>& runtime_args_opt) {
  const bool is_reader = segment_kind == "reader";
  const bool is_writer = segment_kind == "writer";
  if (!is_reader && !is_writer) {
    return runtime_args_opt ? Downcast<Array<Any>>(runtime_args_opt.value()) : Array<Any>();
  }

  Array<Any> existing_runtime_args =
      runtime_args_opt ? Downcast<Array<Any>>(runtime_args_opt.value()) : Array<Any>();
  Array<Any> buffer_args;
  Array<Any> other_args;
  std::vector<std::string> bound_buffers;

  for (const auto& arg_item : existing_runtime_args) {
    auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (arg.empty()) {
      other_args.push_back(arg_item);
      continue;
    }
    const std::string arg_kind =
        arg.Get("kind") ? static_cast<std::string>(Downcast<String>(arg.Get("kind").value()))
                        : std::string();
    if (IsBufferAddrRuntimeArgKind(arg_kind) && arg.Get("buffer")) {
      bound_buffers.push_back(static_cast<std::string>(Downcast<String>(arg.Get("buffer").value())));
      buffer_args.push_back(arg_item);
    } else {
      other_args.push_back(arg_item);
    }
  }

  auto has_bound_buffer = [&](const std::string& buffer_name) {
    return std::find(bound_buffers.begin(), bound_buffers.end(), buffer_name) != bound_buffers.end();
  };

  for (const auto& accessor_item : accessors) {
    auto accessor = accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (accessor.empty() || !accessor.Get("buffer")) {
      continue;
    }
    const std::string buffer_name =
        static_cast<std::string>(Downcast<String>(accessor.Get("buffer").value()));
    if (buffer_name.empty() || has_bound_buffer(buffer_name)) {
      continue;
    }

    Map<String, Any> arg;
    const std::string arg_kind = is_reader ? "input_buffer_addr32" : "output_buffer_addr32";
    const std::string arg_name = buffer_name + "_addr";
    arg.Set("name", String(arg_name));
    arg.Set("kind", String(arg_kind));
    arg.Set("dtype", String("uint32"));
    arg.Set("buffer", String(buffer_name));
    arg.Set("identity", String(MakeBlackholeRuntimeArgIdentity(arg_kind, arg_name, buffer_name)));
    buffer_args.push_back(arg);
    bound_buffers.push_back(buffer_name);
  }

  Array<Any> runtime_args;
  for (const auto& item : buffer_args) {
    runtime_args.push_back(item);
  }
  for (const auto& item : other_args) {
    runtime_args.push_back(item);
  }
  return runtime_args;
}

static void ValidateStagedStickCopyPageAlignedOffset(arith::Analyzer* analyzer,
                                                     const PrimExpr& transport_col,
                                                     int64_t page_cols) {
  ICHECK_GT(page_cols, 0);
  const PrimExpr page_cols_expr = IntImm(DataType::Int(32), static_cast<int>(page_cols));
  ICHECK(analyzer->CanProve(
      tir::FloorMod(transport_col, page_cols_expr) == IntImm(DataType::Int(32), 0)))
      << "Blackhole staged stick copy direct-path boundary requires page-aligned transport "
         "offsets, but got column offset "
      << transport_col << " for page width " << page_cols;
}

static void ValidateStagedStickCopyGlobalWidthDivisible(int64_t global_cols, int64_t shared_cols) {
  ICHECK_EQ(global_cols % shared_cols, 0)
      << "Blackhole staged stick copy direct-path boundary requires global width divisible by "
         "shared width";
}

static void ValidateStagedStickCopyTransportPageAlignment(int page_bytes) {
  ICHECK_EQ(page_bytes % 64, 0)
      << "Blackhole staged stick copy direct-path boundary requires a 64B-aligned transport "
         "page size, but got "
      << page_bytes << " bytes";
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
using tir::builtin::blackhole_copy_tile_from_cb;
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
using tir::builtin::blackhole_write_local_slice_to_cb;
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

static Map<String, Any> MakeCompileTimeArgSpec(const std::string& name,
                                               const std::string& kind,
                                               const std::string& dtype,
                                               int offset,
                                               int count,
                                               const std::string& segment_role,
                                               const std::string& buffer = "",
                                               const std::vector<uint32_t>& values = {},
                                               int args_config_bits = 0,
                                               int transport_page_size_bytes = 0,
                                               const std::string& layout = "",
                                               const std::string& memory_space = "") {
  Map<String, Any> spec;
  spec.Set("name", String(name));
  spec.Set("kind", String(kind));
  spec.Set("dtype", String(dtype));
  spec.Set("offset", Integer(offset));
  spec.Set("count", Integer(count));
  if (!buffer.empty()) {
    spec.Set("buffer", String(buffer));
  }
  if (!segment_role.empty()) {
    spec.Set("segment_role", String(segment_role));
  }
  if (!values.empty()) {
    Array<Any> encoded_values;
    for (uint32_t value : values) {
      encoded_values.push_back(Integer(static_cast<int>(value)));
    }
    spec.Set("values", encoded_values);
  }
  if (args_config_bits != 0) {
    spec.Set("args_config_bits", Integer(args_config_bits));
  }
  if (transport_page_size_bytes > 0) {
    spec.Set("transport_page_size", Integer(transport_page_size_bytes));
  }
  if (!layout.empty()) {
    spec.Set("layout", String(layout));
  }
  if (!memory_space.empty()) {
    spec.Set("memory_space", String(memory_space));
  }
  return spec;
}

static Map<String, Any> MakePerWorkArgSpec(const std::string& arg_kind,
                                           const std::string& value_kind,
                                           const std::string& buffer = "",
                                           uint32_t constant_value = 0) {
  Map<String, Any> spec;
  spec.Set(String(blackhole_runtime_arg_schema::kArgKind), String(arg_kind));
  spec.Set(String(blackhole_runtime_arg_schema::kArgIdentity),
           String(MakeBlackholeRuntimeArgIdentity(arg_kind, arg_kind, buffer)));
  if (!buffer.empty()) {
    spec.Set(String(blackhole_runtime_arg_schema::kBuffer), String(buffer));
  }
  spec.Set(String(blackhole_runtime_arg_schema::kValueKind), String(value_kind));
  if (value_kind == blackhole_runtime_arg_schema::kValueConstant) {
    spec.Set(String(blackhole_runtime_arg_schema::kConstantValue),
             Integer(static_cast<int64_t>(constant_value)));
  }
  return spec;
}

static int MakeAccessorArgsConfigBits(const std::string& layout,
                                      const std::string& memory_space) {
  int bits = 0;
  if (layout == "sharded") {
    bits |= 1;
  }
  if (memory_space == "dram") {
    bits |= 2;
  }
  return bits;
}

static Map<String, Any> MakeLaunchSpec(const std::string& core_type,
                                       const std::string& processor,
                                       const std::string& noc) {
  Map<String, Any> spec;
  spec.Set("core_type", String(core_type));
  spec.Set("processor", String(processor));
  spec.Set("noc", String(noc));
  return spec;
}

static void BuildTTKernelAndABISeeds(const Array<Any>& segment_plan,
                                     const Array<Any>& top_runtime_args,
                                     const Array<Any>& top_common_runtime_args,
                                     Array<TTKernel>* kernels_out,
                                     Array<TTABIPlan>* abi_plans_out) {
  Array<TTKernel> kernels;
  Array<TTABIPlan> abi_plans;
  int index = 0;
  for (const Any& item : segment_plan) {
    Map<String, Any> segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (segment.empty()) {
      continue;
    }
    String kernel_name =
        segment.Get("name") ? Downcast<String>(segment.Get("name").value())
                            : String("kernel_" + std::to_string(index));
    String kernel_kind =
        segment.Get("kind") ? Downcast<String>(segment.Get("kind").value())
                            : String("fused_dataflow");
    String core_type =
        segment.Get("core_type") ? Downcast<String>(segment.Get("core_type").value())
                                 : String("brisc");
    Array<Any> runtime_args =
        segment.Get("runtime_args") ? Downcast<Array<Any>>(segment.Get("runtime_args").value())
                                    : Array<Any>();
    Array<Any> common_runtime_args =
        segment.Get("common_runtime_args")
            ? Downcast<Array<Any>>(segment.Get("common_runtime_args").value())
            : Array<Any>();
    const bool uses_top_level_runtime_args = runtime_args.empty() && !top_runtime_args.empty();
    const bool uses_top_level_common_runtime_args =
        common_runtime_args.empty() && !top_common_runtime_args.empty();
    if (runtime_args.empty()) {
      runtime_args = top_runtime_args;
    }
    if (common_runtime_args.empty()) {
      common_runtime_args = top_common_runtime_args;
    }
    segment.Set("tt_uses_top_level_runtime_args", Bool(uses_top_level_runtime_args));
    segment.Set("tt_uses_top_level_common_runtime_args",
                Bool(uses_top_level_common_runtime_args));
    Array<Any> compile_time_arg_specs =
        segment.Get("compile_time_arg_specs")
            ? Downcast<Array<Any>>(segment.Get("compile_time_arg_specs").value())
            : Array<Any>();
    Array<Any> accessors =
        segment.Get("accessors") ? Downcast<Array<Any>>(segment.Get("accessors").value())
                                 : Array<Any>();
    Array<Any> semaphore_bindings =
        segment.Get("semaphore_bindings")
            ? Downcast<Array<Any>>(segment.Get("semaphore_bindings").value())
            : Array<Any>();
    abi_plans.push_back(TTABIPlan(String("abi_" + std::to_string(index)), kernel_name, runtime_args,
                                  common_runtime_args, compile_time_arg_specs, accessors,
                                  semaphore_bindings, segment));
    kernels.push_back(TTKernel(kernel_name, kernel_kind, core_type, index, segment));
    ++index;
  }
  *kernels_out = kernels;
  *abi_plans_out = abi_plans;
}

static tir::PrimFunc StripLegacyTTBridgeProjectionAttrs(tir::PrimFunc func) {
  static const char* kLegacyBridgeAttrs[] = {
      "blackhole.segment_plan",
      "blackhole.runtime_args",
      "blackhole.common_runtime_args",
      "blackhole.per_work_arg_specs",
      "blackhole.gemm_contract",
      "blackhole.compute_contract",
      "blackhole.direct_runtime_unsupported_reasons",
  };
  for (const char* key : kLegacyBridgeAttrs) {
    func = tvm::WithoutAttr(std::move(func), key);
  }
  return func;
}

static PrimExpr ScalarizeVectorizedIndex(const PrimExpr& index) {
  if (const auto* ramp = index.as<tir::RampNode>()) {
    return ramp->base;
  }
  return index;
}

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

struct StagedCopyTransportGeometry {
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  int64_t global_rows = 0;
  int64_t global_cols = 0;
  bool use_page_transport = false;
  int subtile_rows = 0;
  int subtile_cols = 0;
  int tile_bytes = 0;
  int page_bytes = 0;
  int l1_stick_stride = 0;
  int shared_bytes = 0;
};

struct StagedCopyGlobalIndexInfo {
  PrimExpr base_row;
  PrimExpr base_col;
  PrimExpr outer_slice_index{IntImm(DataType::Int(32), 0)};
  int64_t global_rows = 0;
  int64_t global_cols = 0;
};

static std::pair<int64_t, int64_t> ResolveStaticShape2DFromBufferOrMetadata(
    const Buffer& buffer,
    const Array<Integer>& fallback_shape,
    const char* static_shape_message,
    const char* rank2_message) {
  if (buffer->shape.size() >= 2U) {
    const size_t rank = buffer->shape.size();
    const auto* rows_imm = buffer->shape[rank - 2].as<IntImmNode>();
    const auto* cols_imm = buffer->shape[rank - 1].as<IntImmNode>();
    ICHECK(rows_imm && cols_imm) << static_shape_message;
    return {rows_imm->value, cols_imm->value};
  }
  ICHECK_GE(fallback_shape.size(), 2U) << rank2_message;
  return {fallback_shape[0]->value, fallback_shape[1]->value};
}

static std::pair<int64_t, int64_t> ResolveStaticShape2DFromBufferAxesOrMetadata(
    const Buffer& buffer,
    const Array<Integer>& fallback_shape,
    int row_axis,
    int col_axis,
    const char* static_shape_message,
    const char* rank2_message) {
  if (buffer->shape.size() > static_cast<size_t>(std::max(row_axis, col_axis))) {
    const auto* rows_imm = buffer->shape[row_axis].as<IntImmNode>();
    const auto* cols_imm = buffer->shape[col_axis].as<IntImmNode>();
    ICHECK(rows_imm && cols_imm) << static_shape_message;
    return {rows_imm->value, cols_imm->value};
  }
  ICHECK_GE(fallback_shape.size(), 2U) << rank2_message;
  return {fallback_shape[0]->value, fallback_shape[1]->value};
}

static int64_t ResolveStaticExtentForAxisFromBufferOrMetadata(
    const Buffer& buffer,
    const Array<Integer>& fallback_shape,
    int axis,
    const char* static_shape_message) {
  if (buffer->shape.size() > static_cast<size_t>(axis)) {
    const auto* extent_imm = buffer->shape[axis].as<IntImmNode>();
    ICHECK(extent_imm) << static_shape_message;
    return extent_imm->value;
  }
  ICHECK(fallback_shape.size() > static_cast<size_t>(axis)) << static_shape_message;
  return fallback_shape[axis]->value;
}

static StagedCopyTransportGeometry BuildStagedCopyTransportGeometry(
    const Buffer& shared_buffer,
    int64_t shared_rows,
    int64_t shared_cols,
    int64_t global_rows,
    int64_t global_cols,
    bool use_page_transport) {
  ICHECK_EQ(shared_rows % kBlackholeTileRows, 0)
      << "Blackhole staged copy currently expects shared tile height aligned to 32";
  if (!use_page_transport) {
    ICHECK_EQ(shared_cols % kBlackholeTileCols, 0)
        << "Blackhole staged copy currently expects shared tile width aligned to 32";
    ICHECK_EQ(global_cols % kBlackholeTileCols, 0)
        << "Blackhole staged copy currently expects global width aligned to 32";
  }

  StagedCopyTransportGeometry geometry;
  geometry.shared_rows = shared_rows;
  geometry.shared_cols = shared_cols;
  geometry.global_rows = global_rows;
  geometry.global_cols = global_cols;
  geometry.use_page_transport = use_page_transport;
  geometry.subtile_rows = static_cast<int>(shared_rows / kBlackholeTileRows);
  geometry.subtile_cols =
      use_page_transport ? 1 : static_cast<int>(shared_cols / kBlackholeTileCols);
  geometry.tile_bytes =
      kBlackholeTileRows * kBlackholeTileCols * shared_buffer->dtype.bytes();
  geometry.page_bytes = static_cast<int>(shared_cols * shared_buffer->dtype.bytes());
  if (use_page_transport) {
    ValidateStagedStickCopyTransportPageAlignment(geometry.page_bytes);
  }
  geometry.l1_stick_stride = geometry.page_bytes;
  geometry.shared_bytes = static_cast<int>(shared_rows * geometry.l1_stick_stride);
  return geometry;
}

static std::pair<int64_t, int64_t> ResolveStagedCopySharedShape(
    const Buffer& shared_buffer,
    const Array<Integer>& fallback_shape,
    bool segmented_gemm,
    bool transpose_b_reader,
    bool accumulator_like_src,
    int64_t gemm_m,
    int64_t gemm_n,
    int64_t gemm_k) {
  if (transpose_b_reader) {
    ICHECK_GT(gemm_k, 0);
    ICHECK_GT(gemm_n, 0);
    return {gemm_k, gemm_n};
  }
  if (segmented_gemm && accumulator_like_src) {
    ICHECK_GT(gemm_m, 0);
    ICHECK_GT(gemm_n, 0);
    return {gemm_m, gemm_n};
  }
  return ResolveStaticShape2DFromBufferOrMetadata(
      shared_buffer, fallback_shape,
      "Blackhole staged copy currently expects static shared tile shapes",
      "Blackhole staged copy currently expects rank-2 shared tiles");
}

template <typename ZeroIndexFn>
static StagedCopyGlobalIndexInfo ResolveStagedCopyGlobalIndexInfo(
    const Buffer& global_buffer,
    const Array<PrimExpr>& global_indices,
    const Array<Integer>& fallback_shape,
    int row_axis,
    int col_axis,
    const char* static_shape_message,
    const char* rank2_message,
    ZeroIndexFn zero_index,
    Analyzer* analyzer) {
  StagedCopyGlobalIndexInfo info;
  std::tie(info.global_rows, info.global_cols) = ResolveStaticShape2DFromBufferAxesOrMetadata(
      global_buffer, fallback_shape, row_axis, col_axis, static_shape_message, rank2_message);
  if (global_indices.size() > static_cast<size_t>(std::max(row_axis, col_axis))) {
    PrimExpr outer_slice_index = IntImm(DataType::Int(32), 0);
    bool has_outer_axis = false;
    for (size_t axis = 0; axis < global_indices.size(); ++axis) {
      if (static_cast<int>(axis) == row_axis || static_cast<int>(axis) == col_axis) {
        continue;
      }
      const int64_t axis_extent = ResolveStaticExtentForAxisFromBufferOrMetadata(
          global_buffer, fallback_shape, static_cast<int>(axis), static_shape_message);
      PrimExpr axis_index = zero_index(global_indices[axis]);
      if (!has_outer_axis) {
        outer_slice_index = axis_index;
        has_outer_axis = true;
      } else {
        outer_slice_index =
            analyzer->Simplify(outer_slice_index * IntImm32(static_cast<int>(axis_extent)) +
                               axis_index);
      }
    }
    info.outer_slice_index = outer_slice_index;
    info.base_row = zero_index(global_indices[row_axis]);
    info.base_col = zero_index(global_indices[col_axis]);
    return info;
  }
  if (global_indices.size() == 1U) {
    PrimExpr linear_index = ScalarizeVectorizedIndex(global_indices[0]);
    PrimExpr row_index =
        analyzer->Simplify(tir::FloorDiv(linear_index, IntImm32(info.global_cols)));
    PrimExpr col_index =
        analyzer->Simplify(tir::FloorMod(linear_index, IntImm32(info.global_cols)));
    info.base_row = zero_index(row_index);
    info.base_col = zero_index(col_index);
    return info;
  }
  LOG(FATAL) << "Blackhole staged copy currently expects rank-2 tiled regions";
}

static PrimExpr LinearizeStagedCopyTransportIndex(Analyzer* analyzer,
                                                  const PrimExpr& transport_row,
                                                  const PrimExpr& transport_col,
                                                  const PrimExpr& outer_slice_index,
                                                  const StagedCopyTransportGeometry& geometry) {
  PrimExpr slice_offset = IntImm(DataType::Int(32), 0);
  if (geometry.use_page_transport) {
    ValidateStagedStickCopyPageAlignedOffset(analyzer, transport_col, geometry.shared_cols);
    ValidateStagedStickCopyGlobalWidthDivisible(geometry.global_cols, geometry.shared_cols);
    PrimExpr page_col =
        analyzer->Simplify(tir::FloorDiv(transport_col, IntImm32(geometry.shared_cols)));
    PrimExpr pages_per_row =
        IntImm32(static_cast<int>(geometry.global_cols / geometry.shared_cols));
    PrimExpr pages_per_slice =
        IntImm32(static_cast<int>(geometry.global_rows * (geometry.global_cols / geometry.shared_cols)));
    slice_offset = analyzer->Simplify(outer_slice_index * pages_per_slice);
    return analyzer->Simplify(slice_offset + transport_row * pages_per_row + page_col);
  }

  PrimExpr tile_row =
      analyzer->Simplify(tir::FloorDiv(transport_row, IntImm32(kBlackholeTileRows)));
  PrimExpr tile_col =
      analyzer->Simplify(tir::FloorDiv(transport_col, IntImm32(kBlackholeTileCols)));
  PrimExpr tiles_per_row =
      IntImm32(static_cast<int>(geometry.global_cols / kBlackholeTileCols));
  PrimExpr tiles_per_slice =
      IntImm32(static_cast<int>((geometry.global_rows / kBlackholeTileRows) *
                                (geometry.global_cols / kBlackholeTileCols)));
  slice_offset = analyzer->Simplify(outer_slice_index * tiles_per_slice);
  return analyzer->Simplify(slice_offset + tile_row * tiles_per_row + tile_col);
}

static bool IsUnsupportedResidualLocalScope(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

static bool IsVectorLocalFragmentBuffer(const Buffer& buffer) {
  return IsUnsupportedResidualLocalScope(buffer) && buffer->shape.size() == 1 &&
         !buffer->shape.empty() && !tir::is_one(buffer->shape[0]);
}

static void ValidateNoResidualFragmentCompute(const Stmt& body) {
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    if (const auto* store = node.as<BufferStoreNode>()) {
      if (IsUnsupportedResidualLocalScope(store->buffer)) {
        ICHECK(false)
            << "Blackhole fragment compute subset lowering is not implemented; residual local "
               "store remains for buffer "
            << store->buffer->name;
      }
    }
  });
}

static bool HasUnsupportedFragmentOpsInRequirements(const Map<String, Any>& lowering_requirements) {
  if (auto fragment_ops = lowering_requirements.Get("fragment_op_kinds")) {
    for (const auto& item : Downcast<Array<Any>>(fragment_ops.value())) {
      const std::string op_name = Downcast<String>(item);
      if (op_name == "row_reduction" || op_name == "row_broadcast" ||
          op_name == "pointwise_chain") {
        return true;
      }
    }
  }
  return false;
}

namespace {
bool IsFragmentFillValue(const PrimExpr& expr);
bool HasResidualFragmentFill(const Stmt& body);
bool HasResidualFragmentAdd(const Stmt& body);
bool HasResidualFragmentMax(const Stmt& body);
bool HasResidualFragmentCast(const Stmt& body);
bool HasResidualRowBroadcast(const Stmt& body);
}

static int CountLoweredRowReductionBuiltins(const Stmt& body) {
  int count = 0;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    const Op op = Downcast<Op>(call->op);
    if (op.same_as(tir::builtin::blackhole_reduce_row()) ||
        op->name == "tl.blackhole.reduce_row") {
      ++count;
    }
  });
  return count;
}

static int CountLoweredRowBroadcastBuiltins(const Stmt& body) {
  int count = 0;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    const Op op = Downcast<Op>(call->op);
    if (op.same_as(tir::builtin::blackhole_mul_row_bcast()) ||
        op.same_as(tir::builtin::blackhole_mul_grouped_row_bcast()) ||
        op.same_as(tir::builtin::blackhole_div_row_bcast()) ||
        op.same_as(tir::builtin::blackhole_div_grouped_row_bcast()) ||
        op.same_as(tir::builtin::blackhole_exp2_row_bcast_affine()) ||
        op.same_as(tir::builtin::blackhole_exp2_grouped_row_bcast_affine()) ||
        op->name == "tl.blackhole.mul_row_bcast" ||
        op->name == "tl.blackhole.mul_grouped_row_bcast" ||
        op->name == "tl.blackhole.div_row_bcast" ||
        op->name == "tl.blackhole.div_grouped_row_bcast" ||
        op->name == "tl.blackhole.exp2_row_bcast_affine" ||
        op->name == "tl.blackhole.exp2_grouped_row_bcast_affine") {
      ++count;
    }
  });
  return count;
}

static Map<String, Any> PruneSatisfiedLoweringRequirements(const Map<String, Any>& lowering_requirements,
                                                           const Stmt& body) {
  auto row_targets_opt = lowering_requirements.Get("row_reduction_targets");
  auto row_broadcast_sources_opt = lowering_requirements.Get("row_broadcast_sources");
  auto pointwise_ops_opt = lowering_requirements.Get("pointwise_op_kinds");
  if (!row_targets_opt.has_value() && !row_broadcast_sources_opt.has_value() &&
      !pointwise_ops_opt.has_value()) {
    return lowering_requirements;
  }

  bool row_reduction_satisfied = !row_targets_opt.has_value();
  if (row_targets_opt.has_value()) {
    const int target_count = static_cast<int>(Downcast<Array<Any>>(row_targets_opt.value()).size());
    row_reduction_satisfied =
        target_count == 0 || CountLoweredRowReductionBuiltins(body) >= target_count;
  }

  bool row_broadcast_satisfied = !row_broadcast_sources_opt.has_value();
  if (row_broadcast_sources_opt.has_value()) {
    const int source_count =
        static_cast<int>(Downcast<Array<Any>>(row_broadcast_sources_opt.value()).size());
    row_broadcast_satisfied =
        source_count == 0 || !HasResidualRowBroadcast(body);
  }

  if (!row_reduction_satisfied && !row_broadcast_satisfied) {
    return lowering_requirements;
  }

  Map<String, Any> pruned;
  for (const auto& [key, value] : lowering_requirements) {
    if (key == "row_reduction_targets" && row_reduction_satisfied) {
      continue;
    }
    if (key == "row_broadcast_sources" && row_broadcast_satisfied) {
      continue;
    }
    if (key == "fragment_op_kinds") {
      Array<Any> kept_ops;
      for (const auto& item : Downcast<Array<Any>>(value)) {
        const std::string op_name = Downcast<String>(item);
        if (op_name == "row_reduction" && row_reduction_satisfied) continue;
        if (op_name == "row_broadcast" && row_broadcast_satisfied) continue;
        kept_ops.push_back(item);
      }
      pruned.Set(key, kept_ops);
      continue;
    }
    if (key == "pointwise_op_kinds") {
      Array<Any> kept_ops;
      for (const auto& item : Downcast<Array<Any>>(value)) {
        const std::string op_name = Downcast<String>(item);
        if (op_name == "fill" && !HasResidualFragmentFill(body)) {
          continue;
        }
        if (op_name == "add" && !HasResidualFragmentAdd(body)) {
          continue;
        }
        if (op_name == "max" && !HasResidualFragmentMax(body)) {
          continue;
        }
        if (op_name == "cast" && !HasResidualFragmentCast(body)) {
          continue;
        }
        kept_ops.push_back(item);
      }
      pruned.Set(key, kept_ops);
      continue;
    }
    pruned.Set(key, value);
  }
  return pruned;
}

static void ValidateFragmentPipelineLegality(const Map<String, Any>& lowering_requirements) {
  if (auto pipeline_stage_counts = lowering_requirements.Get("pipeline_stage_counts")) {
    for (const auto& item : Downcast<Array<Any>>(pipeline_stage_counts.value())) {
      const int stage_count = Downcast<Integer>(item)->value;
      ICHECK_LE(stage_count, 2)
          << "Blackhole fragment pipeline legality: unsupported stage count " << stage_count;
    }
  }
}

static void ValidateFragmentPipelineLegalityFromBody(const Stmt& body) {
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* loop = node.as<ForNode>();
    if (!loop || !loop->annotations.defined()) {
      return;
    }
    auto stage_count = loop->annotations.Get("num_stages");
    if (!stage_count.has_value()) {
      return;
    }
    const int64_t stages = Downcast<Integer>(stage_count.value())->value;
    ICHECK_LE(stages, 2)
        << "Blackhole fragment pipeline legality: unsupported stage count " << stages;
  });
}

static std::optional<Array<Any>> GetSpatialWorkAxesFromProgram(const SpatialProgram& program) {
  if (!program.defined()) {
    return std::nullopt;
  }
  for (const SpatialLayout& layout : program->layouts) {
    if (!layout->axes.empty()) {
      Array<Any> axes;
      for (const String& axis : layout->axes) {
        axes.push_back(axis);
      }
      return axes;
    }
  }
  for (const WorkPartition& partition : program->work_partitions) {
    if (!partition->axes.empty()) {
      Array<Any> axes;
      for (const String& axis : partition->axes) {
        axes.push_back(axis);
      }
      return axes;
    }
  }
  return std::nullopt;
}

static int GetSpatialDerivedIndexExprCountFromProgram(const SpatialProgram& program) {
  if (!program.defined()) {
    return 0;
  }
  for (const SpatialLayout& layout : program->layouts) {
    if (static_cast<std::string>(layout->kind) == "indexed") {
      return 1;
    }
    for (const String& trait : layout->traits) {
      if (static_cast<std::string>(trait) == "derived_indices") {
        return 1;
      }
    }
  }
  return 0;
}

static int GetWorkDependentLoopBoundCountFromProgram(const SpatialProgram& program) {
  if (!program.defined()) {
    return 0;
  }
  for (const WorkPartition& partition : program->work_partitions) {
    auto maybe_loop_bounds = partition->payload.Get(String(schema_key::kWorkDependentLoopBounds));
    if (!maybe_loop_bounds) {
      continue;
    }
    return static_cast<int>(Downcast<Array<Any>>(maybe_loop_bounds.value()).size());
  }
  return 0;
}

static bool HasTrait(const Array<String>& traits, const char* expected) {
  for (const String& trait : traits) {
    if (static_cast<std::string>(trait) == expected) {
      return true;
    }
  }
  return false;
}

static void CollectPipelineStageInfoFromSpatialProgram(const SpatialProgram& program,
                                                       Array<Any>* stage_counts,
                                                       Array<Any>* loop_vars) {
  std::unordered_set<std::string> seen_loop_vars;
  for (const ResourceIntent& intent : program->resource_intents) {
    if (static_cast<std::string>(intent->kind) != "synchronization_support" ||
        !HasTrait(intent->traits, "pipeline_contract")) {
      continue;
    }
    auto maybe_pipeline_stages = intent->payload.Get(String(schema_key::kPipelineStages));
    if (!maybe_pipeline_stages) {
      continue;
    }
    for (const Any& stage_any : Downcast<Array<Any>>(maybe_pipeline_stages.value())) {
      auto stage = stage_any.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (auto num_stages = stage.Get(String(schema_key::kNumStages))) {
        stage_counts->push_back(Downcast<Integer>(num_stages.value()));
      }
      if (auto loop_var = stage.Get(String(schema_key::kLoopVar))) {
        const std::string loop_var_name = Downcast<String>(loop_var.value());
        if (!loop_var_name.empty() && seen_loop_vars.insert(loop_var_name).second) {
          loop_vars->push_back(loop_var.value());
        }
      }
    }
  }
}

static void SetArrayRequirementIfMissing(Map<String, Any>* lowering_requirements,
                                         const String& key, const Array<Any>& values) {
  if (!values.empty() && !lowering_requirements->count(key)) {
    lowering_requirements->Set(key, values);
  }
}

static void CollectFragmentRequirementsFromSpatialProgram(const SpatialProgram& program,
                                                         Map<String, Any>* lowering_requirements) {
  for (const ResourceIntent& intent : program->resource_intents) {
    if (static_cast<std::string>(intent->kind) != "lowering_support" ||
        !HasTrait(intent->traits, "fragment_contract")) {
      continue;
    }
    if (auto fragment_ops = intent->payload.Get(String(schema_key::kFragmentOpKinds))) {
      SetArrayRequirementIfMissing(lowering_requirements, String(schema_key::kFragmentOpKinds),
                                   Downcast<Array<Any>>(fragment_ops.value()));
    }
    if (auto row_targets = intent->payload.Get(String(schema_key::kRowReductionTargets))) {
      SetArrayRequirementIfMissing(lowering_requirements, String(schema_key::kRowReductionTargets),
                                   Downcast<Array<Any>>(row_targets.value()));
    }
    if (auto row_broadcasts =
            intent->payload.Get(String(schema_key::kRowBroadcastSources))) {
      SetArrayRequirementIfMissing(lowering_requirements,
                                   String(schema_key::kRowBroadcastSources),
                                   Downcast<Array<Any>>(row_broadcasts.value()));
    }
    if (auto pointwise_ops = intent->payload.Get(String(schema_key::kPointwiseOpKinds))) {
      SetArrayRequirementIfMissing(lowering_requirements, String(schema_key::kPointwiseOpKinds),
                                   Downcast<Array<Any>>(pointwise_ops.value()));
    }
    if (auto loop_carried =
            intent->payload.Get(String(schema_key::kFragmentLoopCarriedState))) {
      SetArrayRequirementIfMissing(lowering_requirements,
                                   String(schema_key::kFragmentLoopCarriedState),
                                   Downcast<Array<Any>>(loop_carried.value()));
    }
    if (auto materialization_contracts =
            intent->payload.Get(String(schema_key::kFragmentMaterializationContracts))) {
      SetArrayRequirementIfMissing(lowering_requirements,
                                   String(schema_key::kFragmentMaterializationContracts),
                                   Downcast<Array<Any>>(materialization_contracts.value()));
    }
    if (auto flow_contracts =
            intent->payload.Get(String(schema_key::kFragmentBufferFlowContracts))) {
      SetArrayRequirementIfMissing(lowering_requirements,
                                   String(schema_key::kFragmentBufferFlowContracts),
                                   Downcast<Array<Any>>(flow_contracts.value()));
    }
  }
}

static Map<String, Any> BuildLoweringRequirementsFromAnalysis(const PrimFunc& func) {
  Map<String, Any> lowering_requirements;
  auto spatial_program = func->GetAttr<SpatialProgram>(attr::kTLSpatialProgram);
  ICHECK(spatial_program)
      << "LowerBlackholeOps requires tl.spatial_program; run LowerToSpatialProgram and "
         "ValidateSpatialProgram before lowering";
  auto semantic_program = func->GetAttr<SemanticProgram>(attr::kTLSemanticProgram);

  const SpatialProgram& program = spatial_program.value();
  if (auto axes = GetSpatialWorkAxesFromProgram(program)) {
    lowering_requirements.Set("work_axes", axes.value());
  }
  const int derived_index_expr_count = GetSpatialDerivedIndexExprCountFromProgram(program);
  if (derived_index_expr_count > 0) {
    lowering_requirements.Set("derived_index_expr_count", Integer(derived_index_expr_count));
  }
  const int work_dependent_loop_bound_count = GetWorkDependentLoopBoundCountFromProgram(program);
  if (work_dependent_loop_bound_count > 0) {
    lowering_requirements.Set("work_dependent_loop_bound_count",
                              Integer(work_dependent_loop_bound_count));
  }
  if (!program->phases.empty()) {
    lowering_requirements.Set("spatial_phase_count",
                              Integer(static_cast<int>(program->phases.size())));
  }
  if (!program->channels.empty()) {
    lowering_requirements.Set("spatial_channel_count",
                              Integer(static_cast<int>(program->channels.size())));
  }
  Array<Any> phase_boundary_states;
  std::unordered_set<std::string> seen_phase_boundary_states;
  for (const ResourceIntent& intent : program->resource_intents) {
    if (static_cast<std::string>(intent->kind) != "phase_boundary_materialization") {
      continue;
    }
    ICHECK(str(intent->target_kind) == spatial_contract::kSemanticStateTarget &&
           intent->target_index >= 0)
        << "LowerBlackholeOps requires phase-boundary intents to carry semantic_state "
           "target_kind/target_index contract";
    ICHECK(semantic_program)
        << "LowerBlackholeOps requires tl.semantic_program when consuming phase-boundary "
           "state contracts";
    ICHECK_LT(intent->target_index, semantic_program.value()->states.size())
        << "LowerBlackholeOps found phase-boundary intent with invalid target_index";
    const std::string state_name =
        static_cast<std::string>(semantic_program.value()->states[intent->target_index]->name);
    if (state_name.empty() || !seen_phase_boundary_states.insert(state_name).second) {
      continue;
    }
    phase_boundary_states.push_back(String(state_name));
  }
  if (!phase_boundary_states.empty()) {
    lowering_requirements.Set("spatial_phase_boundary_states", phase_boundary_states);
  }
  Array<Any> stage_counts;
  Array<Any> loop_vars;
  CollectPipelineStageInfoFromSpatialProgram(program, &stage_counts, &loop_vars);
  if (!stage_counts.empty()) {
    lowering_requirements.Set("pipeline_stage_counts", stage_counts);
  }
  if (!loop_vars.empty()) {
    lowering_requirements.Set("pipeline_loop_vars", loop_vars);
  }
  CollectFragmentRequirementsFromSpatialProgram(program, &lowering_requirements);

  return lowering_requirements;
}

static std::unordered_map<std::string, Map<String, Any>>
BuildFragmentMaterializationContractMap(const Map<String, Any>& lowering_requirements) {
  std::unordered_map<std::string, Map<String, Any>> contracts_by_target_buffer;
  auto maybe_contracts =
      lowering_requirements.Get(String(schema_key::kFragmentMaterializationContracts));
  if (!maybe_contracts) {
    return contracts_by_target_buffer;
  }
  for (const Any& contract_any : Downcast<Array<Any>>(maybe_contracts.value())) {
    Map<String, Any> contract = Downcast<Map<String, Any>>(contract_any);
    auto target_it = contract.find(String(schema_key::kTargetBuffer));
    if (target_it == contract.end()) {
      continue;
    }
    const std::string target_buffer = Downcast<String>((*target_it).second);
    if (!target_buffer.empty()) {
      contracts_by_target_buffer.emplace(target_buffer, contract);
    }
  }
  return contracts_by_target_buffer;
}

void LowerBlackholeOps::LoadBufferFlowContracts(const Map<String, Any>& lowering_requirements) {
  buffer_flow_contracts_.clear();
  auto maybe_contracts =
      lowering_requirements.Get(String(schema_key::kFragmentBufferFlowContracts));
  if (!maybe_contracts) {
    return;
  }
  for (const Any& contract_any : Downcast<Array<Any>>(maybe_contracts.value())) {
    Map<String, Any> encoded_contract = Downcast<Map<String, Any>>(contract_any);
    auto buffer_it = encoded_contract.find(String(schema_key::kBuffer));
    auto flow_class_it = encoded_contract.find(String(schema_key::kFlowClass));
    if (buffer_it == encoded_contract.end() || flow_class_it == encoded_contract.end()) {
      continue;
    }
    const std::string buffer_name = Downcast<String>((*buffer_it).second);
    auto parsed_flow_class = ParseCBFlowClass(Downcast<String>((*flow_class_it).second));
    ICHECK(parsed_flow_class.has_value())
        << "LowerBlackholeOps requires a known fragment buffer flow_class for " << buffer_name;

    BufferFlowContract contract;
    contract.flow_class = parsed_flow_class.value();
    if (auto publish = encoded_contract.Get(String(schema_key::kPublishGranule))) {
      contract.publish_pages_per_event =
          std::max(1, static_cast<int>(Downcast<Integer>(publish.value())->value));
    }
    if (auto consume = encoded_contract.Get(String(schema_key::kConsumeGranule))) {
      contract.consume_pages_per_event =
          std::max(1, static_cast<int>(Downcast<Integer>(consume.value())->value));
    }
    if (auto events = encoded_contract.Get(String(schema_key::kEvents))) {
      for (const Any& event_any : Downcast<Array<Any>>(events.value())) {
        Map<String, Any> encoded_event = Downcast<Map<String, Any>>(event_any);
        auto maybe_kind = encoded_event.Get(String(schema_key::kKind));
        auto maybe_order_index = encoded_event.Get(String(schema_key::kOrderIndex));
        if (!maybe_kind || !maybe_order_index) {
          continue;
        }
        BufferFlowEvent event;
        event.order_index = Downcast<Integer>(maybe_order_index.value())->value;
        const std::string kind = Downcast<String>(maybe_kind.value());
        if (kind == fragment_flow::kWrite) {
          event.kind = BufferFlowEventKind::kWrite;
        } else if (kind == fragment_flow::kComputeConsume) {
          event.kind = BufferFlowEventKind::kComputeConsume;
        } else if (kind == fragment_flow::kTransportConsume) {
          event.kind = BufferFlowEventKind::kTransportConsume;
        } else {
          event.kind = BufferFlowEventKind::kReference;
        }
        contract.events.push_back(event);
      }
    }
    buffer_flow_contracts_.emplace(buffer_name, std::move(contract));
  }
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

LowerBlackholeOps::LowerBlackholeOps() : next_requirement_index_(0) {}

void LowerBlackholeOps::LoadLogicalBufferShapes(const PrimFunc& func) {
  logical_buffer_shapes_.clear();
  auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest);
  if (!manifest.has_value()) {
    return;
  }
  std::unordered_map<std::string, std::vector<int64_t>> canonical_shapes;
  auto register_shape = [&](const std::string& name, const std::vector<int64_t>& shape) {
    if (name.empty() || shape.empty()) {
      return;
    }
    logical_buffer_shapes_[name] = shape;
    canonical_shapes[name] = shape;
  };
  auto buffers_it = manifest.value().find(manifest_key::kBuffers);
  if (buffers_it != manifest.value().end()) {
    for (const Any& buffer_any : Downcast<Array<Any>>((*buffers_it).second)) {
      auto descriptor = Downcast<Map<String, Any>>(buffer_any);
      auto name = descriptor.Get(String(schema_key::kName));
      auto shape = descriptor.Get(String(schema_key::kShape));
      if (!name.has_value() || !shape.has_value()) {
        continue;
      }
      auto static_shape = ExtractStaticShape(Downcast<Array<PrimExpr>>(shape.value()));
      if (!static_shape.has_value()) {
        continue;
      }
      register_shape(static_cast<std::string>(Downcast<String>(name.value())),
                     static_shape.value());
    }
  }

  auto structural_regions_it = manifest.value().find(manifest_key::kStructuralRegions);
  if (structural_regions_it == manifest.value().end()) {
    return;
  }

  auto infer_shape_from_names = [&](const Array<Any>& names) -> std::vector<int64_t> {
    for (const Any& name_any : names) {
      const std::string name = static_cast<std::string>(Downcast<String>(name_any));
      auto it = canonical_shapes.find(name);
      if (it != canonical_shapes.end()) {
        return it->second;
      }
    }
    return {};
  };

  auto infer_shape_from_buffers = [&](const Array<Any>& buffers) -> std::vector<int64_t> {
    for (const Any& buffer_any : buffers) {
      auto buffer = buffer_any.try_cast<Buffer>();
      if (!buffer.has_value()) {
        continue;
      }
      const std::string identity = BufferIdentityName(buffer.value());
      auto it = logical_buffer_shapes_.find(identity);
      if (it != logical_buffer_shapes_.end()) {
        return it->second;
      }
    }
    return {};
  };

  const Array<Any> structural_regions = Downcast<Array<Any>>((*structural_regions_it).second);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const Any& region_any : structural_regions) {
      auto region = Downcast<Map<String, Any>>(region_any);
      auto update_sources_it = region.find(manifest_key::kUpdateSources);
      if (update_sources_it == region.end()) {
        continue;
      }
      for (const Any& update_any : Downcast<Array<Any>>((*update_sources_it).second)) {
        auto update = Downcast<Map<String, Any>>(update_any);
        auto target_it = update.find(schema_key::kTarget);
        if (target_it == update.end()) {
          continue;
        }
        const std::string target_name =
            static_cast<std::string>(Downcast<String>((*target_it).second));
        if (target_name.empty() || canonical_shapes.count(target_name)) {
          continue;
        }
        std::vector<int64_t> inferred_shape;
        auto sources_it = update.find(schema_key::kSources);
        if (sources_it != update.end()) {
          inferred_shape = infer_shape_from_names(Downcast<Array<Any>>((*sources_it).second));
        }
        if (inferred_shape.empty()) {
          auto source_states_it = update.find(schema_key::kSourceStates);
          if (source_states_it != update.end()) {
            inferred_shape =
                infer_shape_from_names(Downcast<Array<Any>>((*source_states_it).second));
          }
        }
        if (inferred_shape.empty()) {
          auto source_buffers_it = update.find(schema_key::kSourceBuffers);
          if (source_buffers_it != update.end()) {
            inferred_shape =
                infer_shape_from_buffers(Downcast<Array<Any>>((*source_buffers_it).second));
          }
        }
        if (!inferred_shape.empty()) {
          canonical_shapes[target_name] = inferred_shape;
          logical_buffer_shapes_[target_name] = inferred_shape;
          changed = true;
        }
      }
    }
  }

  for (const Any& region_any : structural_regions) {
    auto region = Downcast<Map<String, Any>>(region_any);
    auto fragment_buffers_it = region.find(manifest_key::kFragmentBuffers);
    if (fragment_buffers_it == region.end()) {
      continue;
    }
    for (const Any& fragment_any : Downcast<Array<Any>>((*fragment_buffers_it).second)) {
      auto fragment = Downcast<Map<String, Any>>(fragment_any);
      auto name_it = fragment.find(schema_key::kName);
      auto buffer_it = fragment.find(schema_key::kBuffer);
      if (name_it == fragment.end() || buffer_it == fragment.end()) {
        continue;
      }
      const std::string canonical_name =
          static_cast<std::string>(Downcast<String>((*name_it).second));
      auto shape_it = canonical_shapes.find(canonical_name);
      auto buffer = (*buffer_it).second.try_cast<Buffer>();
      if (shape_it == canonical_shapes.end() || !buffer.has_value()) {
        continue;
      }
      logical_buffer_shapes_[BufferIdentityName(buffer.value())] = shape_it->second;
    }
  }
}

std::vector<int64_t> LowerBlackholeOps::GetLogicalBufferShape(const Buffer& buffer) const {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = logical_buffer_shapes_.find(buffer_identity);
  if (it != logical_buffer_shapes_.end()) {
    return it->second;
  }
  auto static_shape = ExtractStaticShape(buffer->shape);
  if (static_shape.has_value()) {
    return static_shape.value();
  }
  return {};
}

int64_t LowerBlackholeOps::GetLogicalBufferElementCount(const Buffer& buffer) const {
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

int LowerBlackholeOps::GetLogicalBufferTileCount(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() >= 2U) {
    const int mt = CeilDivToInt(shape[shape.size() - 2], kBlackholeTileRows);
    const int nt = CeilDivToInt(shape[shape.size() - 1], kBlackholeTileCols);
    return std::max(1, mt * nt);
  }
  constexpr int64_t kTileElements = kBlackholeTileRows * kBlackholeTileCols;
  return std::max(1, CeilDivToInt(GetLogicalBufferElementCount(buffer), kTileElements));
}

int64_t LowerBlackholeOps::GetLogicalVectorLength(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() == 1U) {
    return shape.front();
  }
  return -1;
}

std::pair<int64_t, int64_t> LowerBlackholeOps::GetLogicalMatrixShape(const Buffer& buffer) const {
  const std::vector<int64_t> shape = GetLogicalBufferShape(buffer);
  if (shape.size() >= 2U) {
    return {shape[shape.size() - 2], shape[shape.size() - 1]};
  }
  return {-1, -1};
}

PrimFunc LowerBlackholeOps::Transform(const PrimFunc& func) {
  current_func_ = func;
  buffer_to_req_.clear();
  buffer_data_to_req_index_.clear();
  buffer_identity_to_req_index_.clear();
  cb_requirements_.clear();
  accessor_descriptors_.clear();
  next_requirement_index_ = 0;
  saw_copy_op_ = false;
  needs_copy_runtime_args_ = false;
  copy_input_buffer_ = Buffer();
  copy_output_buffer_ = Buffer();
  copy_input_buffer_name_.clear();
  copy_output_buffer_name_.clear();
  copy_input_shape_.clear();
  copy_output_shape_.clear();
  copy_intermediate_shape_.clear();
  thread_index_vars_.clear();
  thread_index_var_names_.clear();
  thread_index_var_static_extents_.clear();
  block_index_vars_.clear();
  block_index_var_names_.clear();
  cb_consumed_fragment_pages_by_buffer_identity_.clear();
  cb_consumed_fragment_use_count_by_buffer_identity_.clear();
  buffer_flow_contracts_.clear();
  stmt_order_index_by_node_.clear();
  logical_buffer_shapes_.clear();
  LoadLogicalBufferShapes(func);
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
  gemm_contract_signatures_.clear();
  compute_contract_payload_index_by_signature_.clear();
  multi_gemm_contract_payloads_.clear();
  multi_compute_contract_payloads_.clear();
  compute_epilogue_payloads_.clear();
  active_compute_contract_payload_index_ = -1;
  compute_epilogue_payloads_flat_.clear();
  fragment_materialization_contracts_by_target_buffer_.clear();
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
  ValidateFragmentPipelineLegalityFromBody(func->body);
  Map<String, Any> lowering_requirements = BuildLoweringRequirementsFromAnalysis(func);
  fragment_materialization_contracts_by_target_buffer_ =
      BuildFragmentMaterializationContractMap(lowering_requirements);
  LoadBufferFlowContracts(lowering_requirements);
  stmt_order_index_by_node_ = BuildExecutionOrderIndexByStmtNode(func->body);
  if (!lowering_requirements.empty()) {
    ValidateFragmentPipelineLegality(lowering_requirements);
  }

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
          << "LowerBlackholeOps requires a stable GEMM input tile contract per logical "
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
        auto it = cb_consumed_fragment_pages_by_buffer_identity_.find(buffer_identity);
        if (it == cb_consumed_fragment_pages_by_buffer_identity_.end()) {
          cb_consumed_fragment_pages_by_buffer_identity_[buffer_identity] = tile_count;
        } else {
          it->second = std::max(it->second, tile_count);
        }
        cb_consumed_fragment_use_count_by_buffer_identity_[buffer_identity] += 1;
      }
    };
    record_gemm_input_tiles(call->args[0], m_tiles * k_tiles);
    record_gemm_input_tiles(call->args[1], k_tiles * n_tiles);
    record_if_cb_consumed_fragment(call->args[0], m_tiles * k_tiles);
    record_if_cb_consumed_fragment(call->args[1], k_tiles * n_tiles);
  });

  compute_epilogue_payloads_.assign(multi_compute_contract_payloads_.size(), {});
  compute_contract_known_buffers_.assign(multi_compute_contract_payloads_.size(), {});
  for (size_t i = 0; i < multi_compute_contract_payloads_.size(); ++i) {
    auto maybe_insert = [&](const char* key) {
      if (auto value = multi_compute_contract_payloads_[i].Get(String(key))) {
        compute_contract_known_buffers_[i].insert(
            static_cast<std::string>(Downcast<String>(value.value())));
      }
    };
    maybe_insert("a_buffer");
    maybe_insert("b_buffer");
    maybe_insert("c_buffer");
  }

  // Transform the function body
  Stmt body = VisitStmt(func->body);
  UpdateCBRequirementDepthsFromLoweredBody(
      &cb_requirements_, body, gemm_a_buffer_name_.empty() ? "fused_dataflow" : "compute");
  lowering_requirements = PruneSatisfiedLoweringRequirements(lowering_requirements, body);

  // Create new function with transformed body
  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = body;

  // Store CB requirements in function attributes for PlanBlackholeCB
  StoreCBRequirements(new_func);
  StoreRuntimeArgs(new_func);
  StoreSegmentPlan(new_func);
  StoreGemmContract(new_func);
  StoreAccessorDescriptors(new_func);

  if (!lowering_requirements.empty()) {
    Map<String, Any> attrs =
        new_func->attrs.defined() ? new_func->attrs->dict : Map<String, Any>();
    attrs.Set("blackhole.lowering_requirements", lowering_requirements);
    new_func.CopyOnWrite()->attrs = DictAttrs(attrs);
  }

  if (!HasUnsupportedFragmentOpsInRequirements(lowering_requirements)) {
    ValidateNoResidualFragmentCompute(body);
  }

  return new_func;
}

// Get CB configuration from function attributes
LowerBlackholeOps::CBConfig LowerBlackholeOps::GetCBConfig() const {
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

int LowerBlackholeOps::AllocateRequirementIndex(const Buffer& buffer, CBType type) {
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
        << "LowerBlackholeOps requires one CB type per logical buffer identity; "
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

  cb_requirements_.push_back(req);
  return requirement_index;
}

int LowerBlackholeOps::EstimateCopyPageSize(const Buffer& buffer) const {
  const int64_t total_elements = GetLogicalBufferElementCount(buffer);
  if (total_elements <= 0) {
    return 2048;
  }

  const int64_t dtype_bytes = buffer->dtype.bytes();
  const int64_t total_bytes = total_elements * dtype_bytes;
  const int64_t default_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * dtype_bytes;
  return static_cast<int>(std::max<int64_t>(dtype_bytes, std::min(total_bytes, default_tile_bytes)));
}

void LowerBlackholeOps::SetRequirementPageLayout(int requirement_index, int page_size,
                                                 int num_pages) {
  ICHECK_GE(requirement_index, 0);
  ICHECK_LT(requirement_index, static_cast<int>(cb_requirements_.size()));
  auto& req = cb_requirements_[requirement_index];
  req.page_size = page_size;
  req.num_pages = num_pages;
}

bool LowerBlackholeOps::UseStagedCopyPageTransport(const Buffer& shared_buffer) const {
  if (shared_buffer->shape.size() < 2U) {
    return false;
  }
  const auto* rows_imm = shared_buffer->shape[0].as<IntImmNode>();
  const auto* cols_imm = shared_buffer->shape[1].as<IntImmNode>();
  if (!rows_imm || !cols_imm) {
    return false;
  }
  return rows_imm->value > 0 && rows_imm->value % kBlackholeTileRows == 0 &&
         cols_imm->value > 0 &&
         cols_imm->value % kBlackholeTileCols != 0;
}

// Store CB requirements in function attributes
void LowerBlackholeOps::StoreCBRequirements(PrimFunc& func) {
  if (cb_requirements_.empty()) {
    return;
  }

  // Get existing attributes
  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  // Build CB requirements array
  Array<Any> cb_reqs;
  for (size_t i = 0; i < cb_requirements_.size(); ++i) {
    const auto& req = cb_requirements_[i];
    Map<String, Any> req_map;
    req_map.Set("requirement_index", Integer(static_cast<int>(i)));
    req_map.Set("name", String(req.name));
    req_map.Set("type", String(req.type == CBType::kInput ? "input" :
                               req.type == CBType::kOutput ? "output" : "intermediate"));
    req_map.Set("page_size", Integer(req.page_size));
    req_map.Set("num_pages", Integer(req.num_pages));
    if (req.initial_reserve_pages > 0) {
      req_map.Set("initial_reserve_pages", Integer(req.initial_reserve_pages));
    }
    req_map.Set("flow_class", String(CBFlowClassToString(req.flow_class)));
    if (req.publish_pages_per_event > 0) {
      req_map.Set("publish_pages_per_event", Integer(req.publish_pages_per_event));
    }
    if (req.consume_pages_per_event > 0) {
      req_map.Set("consume_pages_per_event", Integer(req.consume_pages_per_event));
    }
    req_map.Set("data_format", String(req.data_format));
    req_map.Set("lifetime_begin", Integer(req.lifetime_begin));
    req_map.Set("lifetime_end", Integer(std::max(req.lifetime_begin, req.lifetime_end)));

    cb_reqs.push_back(req_map);
  }

  attrs.Set("blackhole.cb_requirements", cb_reqs);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

void LowerBlackholeOps::StoreRuntimeArgs(PrimFunc& func) {
  if (!needs_copy_runtime_args_) {
    return;
  }
  if (func->GetAttr<Array<Any>>("blackhole.segment_plan")) {
    return;
  }

  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  Array<Any> runtime_args;
  Array<Any> per_work_arg_specs;
  auto push_arg = [&](const std::string& name, const char* kind, const char* dtype,
                      const std::string& buffer_name = "") {
    Map<String, Any> arg_map;
    arg_map.Set("name", String(name));
    arg_map.Set("kind", String(kind));
    arg_map.Set("dtype", String(dtype));
    if (!buffer_name.empty()) {
      arg_map.Set("buffer", String(buffer_name));
    }
    arg_map.Set("identity", String(MakeBlackholeRuntimeArgIdentity(kind, name, buffer_name)));
    runtime_args.push_back(arg_map);
  };

  const std::string input_buffer_name =
      copy_input_buffer_.defined() ? BufferIdentityName(copy_input_buffer_) : copy_input_buffer_name_;
  const std::string output_buffer_name = copy_output_buffer_.defined()
                                             ? BufferIdentityName(copy_output_buffer_)
                                             : copy_output_buffer_name_;
  const std::string input_arg_name =
      input_buffer_name.empty() ? "input_addr" : input_buffer_name + "_addr";
  const std::string output_arg_name =
      output_buffer_name.empty() ? "output_addr" : output_buffer_name + "_addr";

  push_arg(input_arg_name, "input_buffer_addr32", "uint32", input_buffer_name);
  push_arg(output_arg_name, "output_buffer_addr32", "uint32", output_buffer_name);
  push_arg("work_linear_id", "work_linear_id", "uint32");
  push_arg("a_tile_start_id", "a_tile_start_id", "uint32");
  push_arg("a_tile_num_tiles", "a_tile_num_tiles", "uint32");
  push_arg("a_tile_stride", "a_tile_stride", "uint32");
  push_arg("output_tile_start_id", "output_tile_start_id", "uint32");
  push_arg("output_tile_num_tiles", "output_tile_num_tiles", "uint32");
  push_arg("output_tile_stride", "output_tile_stride", "uint32");
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "a_tile_start_id", blackhole_runtime_arg_schema::kValueCurrentWorkLinearId,
      input_buffer_name));
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "a_tile_num_tiles", blackhole_runtime_arg_schema::kValueConstant, input_buffer_name, 1));
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "a_tile_stride", blackhole_runtime_arg_schema::kValueConstant, input_buffer_name, 1));
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "output_tile_start_id", blackhole_runtime_arg_schema::kValueCurrentWorkLinearId,
      output_buffer_name));
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "output_tile_num_tiles", blackhole_runtime_arg_schema::kValueConstant, output_buffer_name,
      1));
  per_work_arg_specs.push_back(MakePerWorkArgSpec(
      "output_tile_stride", blackhole_runtime_arg_schema::kValueConstant, output_buffer_name, 1));

  attrs.Set("blackhole.runtime_args", runtime_args);
  attrs.Set("blackhole.per_work_arg_specs", per_work_arg_specs);
  Map<String, Any> tt_program_payload =
      attrs.Get(attr::kTLTTProgramPayload)
          ? Downcast<Map<String, Any>>(attrs.Get(attr::kTLTTProgramPayload).value())
          : Map<String, Any>();
  tt_program_payload.Set(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs),
                         per_work_arg_specs);
  attrs.Set(attr::kTLTTProgramPayload, tt_program_payload);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

void LowerBlackholeOps::StoreSegmentPlan(PrimFunc& func) {
  // If SplitBlackholeKernels already wrote the segment plan, do not overwrite.
  if (func->GetAttr<Array<Any>>("blackhole.segment_plan")) return;

  if (!needs_copy_runtime_args_) {
    return;
  }

  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  Array<Any> kernels;
  Map<String, Any> kernel;
  kernel.Set("name", String("main"));
  kernel.Set("kind", String("fused_dataflow"));
  kernel.Set("core_type", String("brisc"));
  kernels.push_back(kernel);

  attrs.Set("blackhole.segment_plan", kernels);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

void LowerBlackholeOps::StoreGemmContract(PrimFunc& func) {
  if (multi_gemm_contract_payloads_.empty() || multi_compute_contract_payloads_.empty()) {
    return;
  }

  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }
  Map<String, Any> tt_program_payload =
      func->GetAttr<Map<String, Any>>(attr::kTLTTProgramPayload).value_or(Map<String, Any>());

  if (gemm_contract_signatures_.size() > 1) {
    Array<Any> multi_gemm_contracts;
    for (const auto& contract : multi_gemm_contract_payloads_) {
      multi_gemm_contracts.push_back(contract);
    }
    Array<Any> multi_compute_contracts;
    for (const auto& contract : multi_compute_contract_payloads_) {
      multi_compute_contracts.push_back(contract);
    }
    tt_program_payload.Set("multi_gemm_contracts", multi_gemm_contracts);
    tt_program_payload.Set("multi_compute_contracts", multi_compute_contracts);
    if (!compute_epilogue_payloads_flat_.empty()) {
      Array<Any> compute_epilogue_ops;
      for (const auto& op_payload : compute_epilogue_payloads_flat_) {
        compute_epilogue_ops.push_back(op_payload);
      }
      tt_program_payload.Set("compute_epilogue_ops", compute_epilogue_ops);
    }
    attrs.Set(attr::kTLTTProgramPayload, tt_program_payload);
    func.CopyOnWrite()->attrs = DictAttrs(attrs);
    return;
  }

  std::string a_buffer = gemm_a_buffer_name_;
  std::string b_buffer = gemm_b_buffer_name_;
  std::string c_buffer = gemm_c_buffer_name_;
  if (auto segment_plan = func->GetAttr<Array<Any>>("blackhole.segment_plan")) {
    for (const auto& item : segment_plan.value()) {
      auto segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (segment.empty()) {
        continue;
      }
      const std::string kind = segment.Get("kind")
                                   ? static_cast<std::string>(Downcast<String>(segment.Get("kind").value()))
                                   : std::string();
      auto runtime_args = segment.Get("runtime_args");
      if (!runtime_args) {
        continue;
      }
      for (const auto& arg_item : Downcast<Array<Any>>(runtime_args.value())) {
        auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (arg.empty() || !arg.Get("buffer")) {
          continue;
        }
        const std::string buffer_name = Downcast<String>(arg.Get("buffer").value());
        const std::string arg_kind = arg.Get("kind")
                                         ? static_cast<std::string>(Downcast<String>(arg.Get("kind").value()))
                                         : std::string();
        if (kind == "reader" && arg_kind == "input_buffer_addr32") {
          if (a_buffer == gemm_a_buffer_name_) {
            a_buffer = buffer_name;
          } else if (b_buffer == gemm_b_buffer_name_) {
            b_buffer = buffer_name;
          }
        } else if (kind == "writer" && arg_kind == "output_buffer_addr32") {
          c_buffer = buffer_name;
        }
      }
    }
  }

  Map<String, Any> gemm_contract = BuildGemmContractPayload(
      a_buffer, b_buffer, c_buffer, gemm_m_, gemm_n_, gemm_k_, gemm_transpose_a_,
      gemm_transpose_b_, gemm_a_dtype_, gemm_b_dtype_, gemm_c_dtype_);
  Map<String, Any> compute_contract = BuildComputeContractPayload(
      a_buffer, b_buffer, c_buffer, gemm_m_, gemm_n_, gemm_k_, gemm_transpose_a_,
      gemm_transpose_b_, gemm_policy_type_, gemm_clear_accum_, gemm_k_pack_, gemm_wg_wait_,
      gemm_dst_full_sync_en_, gemm_bfp8_pack_precise_, gemm_defines_, gemm_named_compile_args_,
      gemm_mbarrier_buffer_name_, gemm_mbarrier_scope_, gemm_mbarrier_index_exprs_, gemm_a_dtype_,
      gemm_b_dtype_, gemm_c_dtype_);

  attrs.Set("blackhole.gemm_contract", gemm_contract);
  attrs.Set("blackhole.compute_contract", compute_contract);
  tt_program_payload.Set("gemm_contract", gemm_contract);
  tt_program_payload.Set("compute_contract", compute_contract);
  if (!compute_epilogue_payloads_flat_.empty()) {
    Array<Any> compute_epilogue_ops;
    for (const auto& op_payload : compute_epilogue_payloads_flat_) {
      compute_epilogue_ops.push_back(op_payload);
    }
    tt_program_payload.Set("compute_epilogue_ops", compute_epilogue_ops);
  }
  attrs.Set(attr::kTLTTProgramPayload, tt_program_payload);
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

void LowerBlackholeOps::StoreAccessorDescriptors(PrimFunc& func) {
  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  auto make_compute_config_from_contract = [&]() -> Map<String, Any> {
    Map<String, Any> compute_config;
    compute_config.Set("math_fidelity", String("HiFi4"));
    compute_config.Set("fp32_dest_acc_en", Bool(true));
    compute_config.Set("dst_full_sync_en", Bool(gemm_dst_full_sync_en_));
    compute_config.Set("math_approx_mode", Bool(false));
    compute_config.Set("unpack_to_dest_mode", Array<Any>{});
    compute_config.Set("bfp8_pack_precise", Bool(gemm_bfp8_pack_precise_));
    compute_config.Set("defines", EncodeNamedStringPairs(gemm_defines_));
    compute_config.Set("named_compile_args", EncodeNamedUint32Pairs(gemm_named_compile_args_));
    compute_config.Set("clear_accum", Bool(gemm_clear_accum_));
    compute_config.Set("k_pack", Integer(gemm_k_pack_));
    compute_config.Set("wg_wait", Integer(gemm_wg_wait_));
    compute_config.Set("policy_type", Integer(gemm_policy_type_));
    compute_config.Set("policy_name", String(GemmWarpPolicyTypeToStringForBlackhole(gemm_policy_type_)));
    return compute_config;
  };

  auto make_launch_spec = [](const std::string& core_type) -> Map<String, Any> {
    if (core_type == "brisc") {
      return MakeLaunchSpec(core_type, "riscv_0", "riscv_0_default");
    }
    if (core_type == "ncrisc") {
      return MakeLaunchSpec(core_type, "riscv_1", "riscv_1_default");
    }
    if (core_type == "trisc") {
      return MakeLaunchSpec(core_type, "", "");
    }
    return Map<String, Any>();
  };

  auto make_accessor_cta_specs = [&](const std::string& kind, const Array<Any>& accessors) {
    Array<Any> compile_time_arg_specs;
    for (const auto& accessor_item : accessors) {
      auto accessor = accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (accessor.empty()) {
        continue;
      }
      const std::string buffer =
          accessor.Get("buffer") ? static_cast<std::string>(Downcast<String>(accessor.Get("buffer").value()))
                                  : std::string();
      const int compile_time_arg_offset =
          accessor.Get("compile_time_arg_offset")
              ? Downcast<Integer>(accessor.Get("compile_time_arg_offset").value()).IntValue()
              : (accessor.Get("slot")
                     ? Downcast<Integer>(accessor.Get("slot").value()).IntValue()
                     : 0);
      const int compile_time_arg_count =
          accessor.Get("compile_time_arg_count")
              ? Downcast<Integer>(accessor.Get("compile_time_arg_count").value()).IntValue()
              : 2;
      const std::string layout =
          accessor.Get("layout")
              ? static_cast<std::string>(Downcast<String>(accessor.Get("layout").value()))
              : std::string("interleaved");
      const std::string memory_space =
          accessor.Get("memory_space")
              ? static_cast<std::string>(Downcast<String>(accessor.Get("memory_space").value()))
              : std::string("dram");
      const int args_config_bits =
          accessor.Get("args_config_bits")
              ? Downcast<Integer>(accessor.Get("args_config_bits").value()).IntValue()
              : MakeAccessorArgsConfigBits(layout, memory_space);
      const int transport_page_size =
          accessor.Get("transport_page_size")
              ? Downcast<Integer>(accessor.Get("transport_page_size").value()).IntValue()
              : 0;

      compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
          buffer,
          "interleaved_accessor_cta",
          "uint32",
          compile_time_arg_offset,
          compile_time_arg_count,
          kind,
          buffer,
          {},
          args_config_bits,
          transport_page_size,
          layout,
          memory_space));
    }
    return compile_time_arg_specs;
  };

  std::unordered_map<std::string, int> accessor_transport_page_sizes;
  PostOrderVisit(func->body, [&](const ObjectRef& node_ref) {
    const auto* call = node_ref.as<CallNode>();
    if (call == nullptr || !call->op->IsInstance<OpNode>() || call->args.size() < 4) {
      return;
    }
    const std::string op_name = Downcast<Op>(call->op)->name;
    const auto* page_bytes = call->args[3].as<IntImmNode>();
    if (page_bytes == nullptr) {
      return;
    }
    if (op_name == "tl.blackhole.read_page_to_cb") {
      if (!copy_input_buffer_name_.empty()) {
        accessor_transport_page_sizes[copy_input_buffer_name_] = page_bytes->value;
      }
    } else if (op_name == "tl.blackhole.write_page_from_cb") {
      if (!copy_output_buffer_name_.empty()) {
        accessor_transport_page_sizes[copy_output_buffer_name_] = page_bytes->value;
      }
    }
  });

  auto make_gemm_compute_cta_specs = [&]() {
    Array<Any> compile_time_arg_specs;
    if (gemm_a_buffer_name_.empty() || gemm_b_buffer_name_.empty() || gemm_c_buffer_name_.empty()) {
      return compile_time_arg_specs;
    }
    ICHECK_EQ(gemm_m_ % kBlackholeTileRows, 0)
        << "Blackhole GEMM compile-time ABI requires M to be 32-aligned";
    ICHECK_EQ(gemm_k_ % kBlackholeTileCols, 0)
        << "Blackhole GEMM compile-time ABI requires K to be 32-aligned";
    ICHECK_EQ(gemm_n_ % kBlackholeTileCols, 0)
        << "Blackhole GEMM compile-time ABI requires N to be 32-aligned";
    const int mt = gemm_m_ / kBlackholeTileRows;
    const int kt = gemm_k_ / kBlackholeTileCols;
    const int nt = gemm_n_ / kBlackholeTileCols;
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_shape",
        "gemm_shape",
        "uint32",
        0,
        3,
        "compute",
        "",
        {static_cast<uint32_t>(mt), static_cast<uint32_t>(kt), static_cast<uint32_t>(nt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_transpose_flags",
        "gemm_transpose_flags",
        "uint32",
        3,
        2,
        "compute",
        "",
        {static_cast<uint32_t>(gemm_transpose_a_ ? 1 : 0),
         static_cast<uint32_t>(gemm_transpose_b_ ? 1 : 0)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_block_shape",
        "gemm_block_shape",
        "uint32",
        5,
        3,
        "compute",
        "",
        {static_cast<uint32_t>(mt), static_cast<uint32_t>(nt), static_cast<uint32_t>(kt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_subblock_shape",
        "gemm_subblock_shape",
        "uint32",
        8,
        2,
        "compute",
        "",
        {static_cast<uint32_t>(mt), static_cast<uint32_t>(nt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_clear_accum",
        "gemm_clear_accum",
        "uint32",
        10,
        1,
        "compute",
        "",
        {static_cast<uint32_t>(gemm_clear_accum_ ? 1 : 0)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_k_pack",
        "gemm_k_pack",
        "uint32",
        11,
        1,
        "compute",
        "",
        {static_cast<uint32_t>(gemm_k_pack_)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_wg_wait",
        "gemm_wg_wait",
        "uint32",
        12,
        1,
        "compute",
        "",
        {static_cast<uint32_t>(gemm_wg_wait_)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_policy",
        "gemm_policy",
        "uint32",
        13,
        1,
        "compute",
        "",
        {static_cast<uint32_t>(gemm_policy_type_)}));
    return compile_time_arg_specs;
  };

  auto make_segment_per_work_arg_specs = [&](const std::string& kind,
                                             const Array<Any>& runtime_args,
                                             const Optional<Any>& existing_specs_opt) {
    Array<Any> per_work_arg_specs =
        existing_specs_opt ? Downcast<Array<Any>>(existing_specs_opt.value()) : Array<Any>();

    auto runtime_args_contain_kind = [&](const char* arg_kind) {
      return std::any_of(runtime_args.begin(), runtime_args.end(), [&](const Any& arg_item) {
        auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        return arg.Get("kind") &&
               static_cast<std::string>(Downcast<String>(arg.Get("kind").value())) == arg_kind;
      });
    };
    auto has_spec = [&](const char* arg_kind) {
      return std::any_of(per_work_arg_specs.begin(), per_work_arg_specs.end(), [&](const Any& item) {
        auto spec = item.as<Map<String, Any>>().value_or(Map<String, Any>());
        return spec.Get(String(blackhole_runtime_arg_schema::kArgKind)) &&
               static_cast<std::string>(Downcast<String>(
                   spec.Get(String(blackhole_runtime_arg_schema::kArgKind)).value())) == arg_kind;
      });
    };
    auto push_if_needed = [&](const Map<String, Any>& spec) {
      const std::string arg_kind = static_cast<std::string>(
          Downcast<String>(spec.Get(String(blackhole_runtime_arg_schema::kArgKind)).value()));
      if (!has_spec(arg_kind.c_str())) {
        per_work_arg_specs.push_back(spec);
      }
    };

    if (kind == "reader") {
      if (runtime_args_contain_kind("a_tile_start_id")) {
        push_if_needed(MakePerWorkArgSpec(
            "a_tile_start_id", blackhole_runtime_arg_schema::kValueLogicalBlockY,
            gemm_a_buffer_name_));
      }
      if (runtime_args_contain_kind("a_tile_num_tiles")) {
        push_if_needed(MakePerWorkArgSpec(
            "a_tile_num_tiles", blackhole_runtime_arg_schema::kValueGemmNumKTiles,
            gemm_a_buffer_name_));
      }
      if (runtime_args_contain_kind("a_tile_stride")) {
        push_if_needed(MakePerWorkArgSpec(
            "a_tile_stride", blackhole_runtime_arg_schema::kValueConstant, gemm_a_buffer_name_,
            1));
      }
      if (runtime_args_contain_kind("b_tile_start_id")) {
        push_if_needed(MakePerWorkArgSpec(
            "b_tile_start_id", blackhole_runtime_arg_schema::kValueLogicalBlockX,
            gemm_b_buffer_name_));
      }
      if (runtime_args_contain_kind("b_tile_num_tiles")) {
        push_if_needed(MakePerWorkArgSpec(
            "b_tile_num_tiles", blackhole_runtime_arg_schema::kValueGemmNumKTiles,
            gemm_b_buffer_name_));
      }
      if (runtime_args_contain_kind("b_tile_stride")) {
        push_if_needed(MakePerWorkArgSpec(
            "b_tile_stride", blackhole_runtime_arg_schema::kValueGemmLogicalNTiles,
            gemm_b_buffer_name_));
      }
    }
    if (kind == "reader" || kind == "compute") {
      if (runtime_args_contain_kind("k_tile_start_id")) {
        push_if_needed(MakePerWorkArgSpec(
            "k_tile_start_id", blackhole_runtime_arg_schema::kValueConstant, "", 0));
      }
      if (runtime_args_contain_kind("num_k_tiles")) {
        push_if_needed(MakePerWorkArgSpec(
            "num_k_tiles", blackhole_runtime_arg_schema::kValueGemmNumKTiles));
      }
    }
    if (kind == "writer") {
      if (runtime_args_contain_kind("output_tile_start_id")) {
        push_if_needed(MakePerWorkArgSpec(
            "output_tile_start_id", blackhole_runtime_arg_schema::kValueCurrentWorkLinearId,
            gemm_c_buffer_name_));
      }
      if (runtime_args_contain_kind("output_tile_num_tiles")) {
        push_if_needed(MakePerWorkArgSpec(
            "output_tile_num_tiles", blackhole_runtime_arg_schema::kValueConstant,
            gemm_c_buffer_name_, 1));
      }
      if (runtime_args_contain_kind("output_tile_stride")) {
        push_if_needed(MakePerWorkArgSpec(
            "output_tile_stride", blackhole_runtime_arg_schema::kValueConstant,
            gemm_c_buffer_name_, 1));
      }
    }
    return per_work_arg_specs;
  };

  if (auto segment_plan = func->GetAttr<Array<Any>>("blackhole.segment_plan")) {
    Array<Any> rewritten_segments;
    for (const auto& item : segment_plan.value()) {
      auto segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (segment.empty()) {
        rewritten_segments.push_back(item);
        continue;
      }
      const std::string kind =
          segment.Get("kind")
              ? static_cast<std::string>(Downcast<String>(segment.Get("kind").value()))
              : std::string();
      Array<Any> accessors;
      Array<Any> compile_time_arg_specs;
      if (auto accessor_items = segment.Get("accessors")) {
        for (const auto& accessor_item : Downcast<Array<Any>>(accessor_items.value())) {
          auto accessor = accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
          if (accessor.empty()) {
            accessors.push_back(accessor_item);
            continue;
          }

          const int compile_time_arg_offset =
              accessor.Get("compile_time_arg_offset")
                  ? Downcast<Integer>(accessor.Get("compile_time_arg_offset").value()).IntValue()
                  : (accessor.Get("slot")
                         ? Downcast<Integer>(accessor.Get("slot").value()).IntValue()
                         : 0);
          const int compile_time_arg_count =
              accessor.Get("compile_time_arg_count")
                  ? Downcast<Integer>(accessor.Get("compile_time_arg_count").value()).IntValue()
                  : 2;
          const int common_runtime_arg_offset =
              accessor.Get("common_runtime_arg_offset")
                  ? Downcast<Integer>(accessor.Get("common_runtime_arg_offset").value()).IntValue()
                  : 0;
          const int common_runtime_arg_count =
              accessor.Get("common_runtime_arg_count")
                  ? Downcast<Integer>(accessor.Get("common_runtime_arg_count").value()).IntValue()
                  : 0;
          const std::string layout =
              accessor.Get("layout")
                  ? static_cast<std::string>(Downcast<String>(accessor.Get("layout").value()))
                  : std::string("interleaved");
          const std::string memory_space =
              accessor.Get("memory_space")
                  ? static_cast<std::string>(Downcast<String>(accessor.Get("memory_space").value()))
                  : std::string("dram");
          const int args_config_bits =
              accessor.Get("args_config_bits")
                  ? Downcast<Integer>(accessor.Get("args_config_bits").value()).IntValue()
                  : MakeAccessorArgsConfigBits(layout, memory_space);
          const int transport_page_size =
              accessor.Get("transport_page_size")
                  ? Downcast<Integer>(accessor.Get("transport_page_size").value()).IntValue()
                  : 0;
          int resolved_transport_page_size = transport_page_size;
          if (resolved_transport_page_size == 0 && accessor.Get("buffer")) {
            const std::string buffer_name =
                static_cast<std::string>(Downcast<String>(accessor.Get("buffer").value()));
            auto transport_it = accessor_transport_page_sizes.find(buffer_name);
            if (transport_it != accessor_transport_page_sizes.end()) {
              resolved_transport_page_size = transport_it->second;
            }
            auto desc_it = std::find_if(
                accessor_descriptors_.begin(), accessor_descriptors_.end(),
                [&](const AccessorDescriptor& desc) {
                  return desc.segment_kind == kind && desc.buffer_name == buffer_name &&
                         desc.compile_time_arg_offset == compile_time_arg_offset &&
                         desc.compile_time_arg_count == compile_time_arg_count &&
                         desc.common_runtime_arg_offset == common_runtime_arg_offset &&
                         desc.common_runtime_arg_count == common_runtime_arg_count &&
                         desc.args_config_bits == args_config_bits;
                });
            if (desc_it != accessor_descriptors_.end()) {
              if (desc_it->transport_page_size_bytes > 0) {
                resolved_transport_page_size = desc_it->transport_page_size_bytes;
              }
            }
          }

          accessor.Set("slot", Integer(compile_time_arg_offset));
          accessor.Set("compile_time_arg_offset", Integer(compile_time_arg_offset));
          accessor.Set("compile_time_arg_count", Integer(compile_time_arg_count));
          accessor.Set("common_runtime_arg_offset", Integer(common_runtime_arg_offset));
          accessor.Set("common_runtime_arg_count", Integer(common_runtime_arg_count));
          accessor.Set("args_config_bits", Integer(args_config_bits));
          if (resolved_transport_page_size > 0) {
            accessor.Set("transport_page_size", Integer(resolved_transport_page_size));
          }
          accessor.Set("layout", String(layout));
          accessor.Set("memory_space", String(memory_space));
          accessors.push_back(accessor);
        }
      }
      if (accessors.empty()) {
        accessors = EncodeAccessorDescriptors(kind);
      }
      Array<Any> runtime_args = EnsureSegmentBufferRuntimeArgs(kind, accessors, segment.Get("runtime_args"));
      Array<Any> per_work_arg_specs = make_segment_per_work_arg_specs(
          kind, runtime_args, segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs)));
      compile_time_arg_specs = make_accessor_cta_specs(kind, accessors);
      if (kind == "compute") {
        if (gemm_contract_signatures_.size() == 1) {
          auto gemm_compile_time_arg_specs = make_gemm_compute_cta_specs();
          for (const auto& spec : gemm_compile_time_arg_specs) {
            compile_time_arg_specs.push_back(spec);
          }
        }
        segment.Set("compute_config", make_compute_config_from_contract());
      }
      segment.Set("accessors", accessors);
      if (!runtime_args.empty()) {
        segment.Set("runtime_args", runtime_args);
      }
      if (!per_work_arg_specs.empty()) {
        segment.Set(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs), per_work_arg_specs);
      }
      segment.Set("compile_time_arg_specs", compile_time_arg_specs);
      Map<String, Any> launch_spec =
          make_launch_spec(segment.Get("core_type")
                               ? static_cast<std::string>(Downcast<String>(segment.Get("core_type").value()))
                               : std::string());
      if (!launch_spec.empty()) {
        segment.Set("launch_spec", launch_spec);
      }
      segment.Set("common_runtime_args", EncodeCommonRuntimeArgs(kind));
      rewritten_segments.push_back(segment);
    }
    attrs.Set("blackhole.segment_plan", rewritten_segments);

    auto aggregate_runtime_args = [&](const char* field_name) {
      Array<Any> aggregated;
      std::unordered_set<std::string> seen_identities;
      for (const auto& segment_item : rewritten_segments) {
        auto segment = segment_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (segment.empty()) {
          continue;
        }
        auto args_it = segment.Get(field_name);
        if (!args_it) {
          continue;
        }
        for (const auto& arg_item : Downcast<Array<Any>>(args_it.value())) {
          auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
          if (arg.empty()) {
            continue;
          }
          const std::string identity =
              arg.Get("identity")
                  ? static_cast<std::string>(Downcast<String>(arg.Get("identity").value()))
                  : std::string();
          if (!identity.empty() && !seen_identities.insert(identity).second) {
            continue;
          }
          aggregated.push_back(arg);
        }
      }
      return aggregated;
    };

    Array<Any> aggregated_runtime_args = aggregate_runtime_args("runtime_args");
    if (!aggregated_runtime_args.empty()) {
      attrs.Set("blackhole.runtime_args", aggregated_runtime_args);
    }
    Array<Any> aggregated_common_runtime_args = aggregate_runtime_args("common_runtime_args");
    if (!aggregated_common_runtime_args.empty()) {
      attrs.Set("blackhole.common_runtime_args", aggregated_common_runtime_args);
    }
    auto aggregate_per_work_arg_specs = [&]() {
      Array<Any> aggregated;
      std::unordered_set<std::string> seen_identities;
      for (const auto& segment_item : rewritten_segments) {
        auto segment = segment_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (segment.empty()) {
          continue;
        }
        auto specs_it = segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs));
        if (!specs_it) {
          continue;
        }
        for (const auto& spec_item : Downcast<Array<Any>>(specs_it.value())) {
          auto spec = spec_item.as<Map<String, Any>>().value_or(Map<String, Any>());
          if (spec.empty()) {
            continue;
          }
          const std::string identity =
              spec.Get(String(blackhole_runtime_arg_schema::kArgIdentity))
                  ? static_cast<std::string>(Downcast<String>(
                        spec.Get(String(blackhole_runtime_arg_schema::kArgIdentity)).value()))
                  : (spec.Get(String(blackhole_runtime_arg_schema::kArgKind))
                         ? static_cast<std::string>(Downcast<String>(
                               spec.Get(String(blackhole_runtime_arg_schema::kArgKind)).value()))
                         : std::string());
          if (!identity.empty() && !seen_identities.insert(identity).second) {
            continue;
          }
          aggregated.push_back(spec_item);
        }
      }
      return aggregated;
    };
    Array<Any> aggregated_per_work_arg_specs = aggregate_per_work_arg_specs();
    if (!aggregated_per_work_arg_specs.empty()) {
      attrs.Set("blackhole.per_work_arg_specs", aggregated_per_work_arg_specs);
      Map<String, Any> tt_program_payload =
          attrs.Get(attr::kTLTTProgramPayload)
              ? Downcast<Map<String, Any>>(attrs.Get(attr::kTLTTProgramPayload).value())
              : Map<String, Any>();
      tt_program_payload.Set(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs),
                             aggregated_per_work_arg_specs);
      attrs.Set(attr::kTLTTProgramPayload, tt_program_payload);
    }
    Array<Any> top_level_runtime_args =
        attrs.Get("blackhole.runtime_args")
            ? Downcast<Array<Any>>(attrs.Get("blackhole.runtime_args").value())
            : Array<Any>();
    Array<Any> top_level_common_runtime_args =
        attrs.Get("blackhole.common_runtime_args")
            ? Downcast<Array<Any>>(attrs.Get("blackhole.common_runtime_args").value())
            : Array<Any>();
    Array<TTKernel> tt_kernels;
    Array<TTABIPlan> tt_abi_plans;
    BuildTTKernelAndABISeeds(rewritten_segments, top_level_runtime_args,
                             top_level_common_runtime_args, &tt_kernels, &tt_abi_plans);
    attrs.Set(attr::kTLTTKernelSeeds, tt_kernels);
    attrs.Set(attr::kTLTTABIPlans, tt_abi_plans);
  }

  func.CopyOnWrite()->attrs = DictAttrs(attrs);
  if (func->GetAttr<Array<TTKernel>>(attr::kTLTTKernelSeeds).has_value() &&
      func->GetAttr<Array<TTABIPlan>>(attr::kTLTTABIPlans).has_value()) {
    func = StripLegacyTTBridgeProjectionAttrs(std::move(func));
  }
}

Array<Any> LowerBlackholeOps::EncodeAccessorDescriptors(const std::string& segment_kind) const {
  Array<Any> accessors;
  for (const auto& desc : accessor_descriptors_) {
    if (desc.segment_kind != segment_kind) {
      continue;
    }
    Map<String, Any> accessor;
    accessor.Set("buffer", String(desc.buffer_name));
    accessor.Set("compile_time_arg_offset", Integer(desc.compile_time_arg_offset));
    accessor.Set("compile_time_arg_count", Integer(desc.compile_time_arg_count));
    accessor.Set("common_runtime_arg_offset", Integer(desc.common_runtime_arg_offset));
    accessor.Set("common_runtime_arg_count", Integer(desc.common_runtime_arg_count));
    accessor.Set("args_config_bits", Integer(desc.args_config_bits));
    if (desc.transport_page_size_bytes > 0) {
      accessor.Set("transport_page_size", Integer(desc.transport_page_size_bytes));
    }
    accessor.Set("layout", String(desc.layout));
    accessor.Set("memory_space", String(desc.memory_space));
    if (!desc.host_axis_order.empty()) {
      Array<Any> axis_order;
      for (int64_t axis : desc.host_axis_order) {
        axis_order.push_back(Integer(axis));
      }
      accessor.Set("host_axis_order", axis_order);
    }
    if (desc.transpose_2d) {
      accessor.Set("transpose_2d", Bool(true));
    }
    accessors.push_back(accessor);
  }
  return accessors;
}

Array<Any> LowerBlackholeOps::EncodeCommonRuntimeArgs(const std::string& segment_kind) const {
  (void)segment_kind;
  return Array<Any>{};
}

// Detect matmul operation using Op comparison
bool LowerBlackholeOps::IsMatmulCall(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.tileop.gemm_py";
}

std::string LowerBlackholeOps::DataTypeToDataFormat(DataType dtype) {
  return DataTypeToDataFormatForBlackhole(dtype);
}

void LowerBlackholeOps::ExtractGemmInfo(const CallNode* op) {
  // tl.tileop.gemm_py args layout (from gemm_op.py _gemm_impl):
  //   [0]=A_region, [1]=B_region, [2]=C_region,
  //   [3]=transA, [4]=transB, [5]=M, [6]=N, [7]=K, ...
  // Optional Blackhole-only producer payload may continue after the existing
  // core ABI without affecting GemmPy/Gemm lowering:
  //   [19]=dst_full_sync_en, [20]=bfp8_pack_precise, [21]=define_count,
  //   then StringImm name/value define pairs, then named_compile_arg_count,
  //   then StringImm/IntImm named compile-arg pairs.
  const auto& args = op->args;
  ICHECK_GE(args.size(), 8U) << "tl.tileop.gemm_py expects at least 8 args";

  tir::BufferRegion a_region = NormalizeToBufferRegion(args[0]);
  tir::BufferRegion b_region = NormalizeToBufferRegion(args[1]);
  tir::BufferRegion c_region = NormalizeToBufferRegion(args[2]);

  gemm_a_buffer_ = a_region->buffer;
  gemm_b_buffer_ = b_region->buffer;
  gemm_c_buffer_ = c_region->buffer;
  gemm_a_buffer_name_ = BufferIdentityName(a_region->buffer);
  gemm_b_buffer_name_ = BufferIdentityName(b_region->buffer);
  gemm_c_buffer_name_ = BufferIdentityName(c_region->buffer);
  gemm_c_scope_ = GetStorageScope(c_region->buffer);
  gemm_a_dtype_ = a_region->buffer->dtype;
  gemm_b_dtype_ = b_region->buffer->dtype;
  gemm_c_dtype_ = c_region->buffer->dtype;
  if (const auto* imm = args[3].as<IntImmNode>()) gemm_transpose_a_ = imm->value != 0;
  if (const auto* imm = args[4].as<IntImmNode>()) gemm_transpose_b_ = imm->value != 0;
  if (const auto* imm = args[8].as<IntImmNode>()) gemm_policy_type_ = static_cast<int>(imm->value);
  if (const auto* imm = args[9].as<IntImmNode>()) gemm_clear_accum_ = imm->value != 0;
  if (const auto* imm = args[14].as<IntImmNode>()) gemm_k_pack_ = static_cast<int>(imm->value);
  if (const auto* imm = args[15].as<IntImmNode>()) gemm_wg_wait_ = static_cast<int>(imm->value);
  gemm_dst_full_sync_en_ = false;
  if (args.size() > 19) {
    if (const auto* imm = args[19].as<IntImmNode>()) {
      gemm_dst_full_sync_en_ = imm->value != 0;
    }
  }
  gemm_bfp8_pack_precise_ = false;
  if (args.size() > 20) {
    if (const auto* imm = args[20].as<IntImmNode>()) {
      gemm_bfp8_pack_precise_ = imm->value != 0;
    }
  }
  gemm_defines_.clear();
  int arg_index = 21;
  int define_count = 0;
  if (args.size() > arg_index) {
    if (const auto* imm = args[arg_index].as<IntImmNode>()) {
      define_count = static_cast<int>(imm->value);
      ++arg_index;
    }
  }
  for (int i = 0; i < define_count; ++i) {
    ICHECK_LT(arg_index + 1, static_cast<int>(args.size()))
        << "blackhole GEMM define payload is truncated";
    const auto* name = args[arg_index].as<tir::StringImmNode>();
    const auto* value = args[arg_index + 1].as<tir::StringImmNode>();
    ICHECK(name && value) << "blackhole GEMM defines must be encoded as StringImm pairs";
    gemm_defines_.emplace_back(name->value, value->value);
    arg_index += 2;
  }
  gemm_named_compile_args_.clear();
  int named_compile_arg_count = 0;
  if (args.size() > arg_index) {
    if (const auto* imm = args[arg_index].as<IntImmNode>()) {
      named_compile_arg_count = static_cast<int>(imm->value);
      ++arg_index;
    }
  }
  for (int i = 0; i < named_compile_arg_count; ++i) {
    ICHECK_LT(arg_index + 1, static_cast<int>(args.size()))
        << "blackhole GEMM named compile arg payload is truncated";
    const auto* name = args[arg_index].as<tir::StringImmNode>();
    const auto* value = args[arg_index + 1].as<IntImmNode>();
    ICHECK(name && value)
        << "blackhole GEMM named compile args must be encoded as StringImm/IntImm pairs";
    gemm_named_compile_args_.emplace_back(name->value, static_cast<uint32_t>(value->value));
    arg_index += 2;
  }
  if (args.size() > 16 && IsBufferLikeExpr(args[16])) {
    tir::BufferRegion mbar_region = NormalizeToBufferRegion(args[16]);
    gemm_has_mbarrier_ = true;
    gemm_mbarrier_buffer_ = mbar_region->buffer;
    gemm_mbarrier_buffer_name_ = BufferIdentityName(mbar_region->buffer);
    gemm_mbarrier_scope_ = GetStorageScope(mbar_region->buffer);
    gemm_mbarrier_index_exprs_.clear();
    for (const auto& range : mbar_region->region) {
      gemm_mbarrier_index_exprs_.push_back(PrimExprToCompactString(range->min));
    }
  } else {
    gemm_has_mbarrier_ = false;
    gemm_mbarrier_buffer_ = Buffer();
    gemm_mbarrier_buffer_name_.clear();
    gemm_mbarrier_scope_.clear();
    gemm_mbarrier_index_exprs_.clear();
  }

  if (const auto* imm = args[5].as<IntImmNode>()) gemm_m_ = static_cast<int>(imm->value);
  if (const auto* imm = args[6].as<IntImmNode>()) gemm_n_ = static_cast<int>(imm->value);
  if (const auto* imm = args[7].as<IntImmNode>()) gemm_k_ = static_cast<int>(imm->value);
  const std::string signature = EncodeGemmContractSignature(
      gemm_a_buffer_name_, gemm_b_buffer_name_, gemm_c_buffer_name_, gemm_m_, gemm_n_, gemm_k_,
      gemm_transpose_a_, gemm_transpose_b_, gemm_policy_type_, gemm_clear_accum_, gemm_k_pack_,
      gemm_wg_wait_, gemm_dst_full_sync_en_, gemm_bfp8_pack_precise_, gemm_defines_,
      gemm_named_compile_args_, gemm_mbarrier_buffer_name_, gemm_mbarrier_scope_,
      gemm_mbarrier_index_exprs_);
  if (gemm_contract_signatures_.insert(signature).second) {
    compute_contract_payload_index_by_signature_[signature] =
        static_cast<int>(multi_compute_contract_payloads_.size());
    multi_gemm_contract_payloads_.push_back(BuildGemmContractPayload(
        gemm_a_buffer_name_, gemm_b_buffer_name_, gemm_c_buffer_name_, gemm_m_, gemm_n_, gemm_k_,
        gemm_transpose_a_, gemm_transpose_b_, gemm_a_dtype_, gemm_b_dtype_, gemm_c_dtype_));
    multi_compute_contract_payloads_.push_back(BuildComputeContractPayload(
        gemm_a_buffer_name_, gemm_b_buffer_name_, gemm_c_buffer_name_, gemm_m_, gemm_n_, gemm_k_,
        gemm_transpose_a_, gemm_transpose_b_, gemm_policy_type_, gemm_clear_accum_, gemm_k_pack_,
        gemm_wg_wait_, gemm_dst_full_sync_en_, gemm_bfp8_pack_precise_, gemm_defines_,
        gemm_named_compile_args_, gemm_mbarrier_buffer_name_, gemm_mbarrier_scope_,
        gemm_mbarrier_index_exprs_, gemm_a_dtype_, gemm_b_dtype_, gemm_c_dtype_));
  }

  // Register GEMM requirements.  The final cb_id is a planner decision and must
  // be consumed later from blackhole.cb_bindings rather than assumed here.
  ICHECK(gemm_a_dtype_ == gemm_b_dtype_)
      << "Blackhole GEMM currently requires matching A/B tensor dtypes";
  const int ab_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * gemm_a_dtype_.bytes();
  const int c_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * gemm_c_dtype_.bytes();
  const int num_m_tiles = CeilDivToInt(gemm_m_, kBlackholeTileRows);
  const int num_n_tiles = CeilDivToInt(gemm_n_, kBlackholeTileCols);
  const int num_k_tiles = CeilDivToInt(gemm_k_, kBlackholeTileCols);

  gemm_a_req_index_ = AllocateRequirementIndex(a_region->buffer, CBType::kInput);
  gemm_b_req_index_ = AllocateRequirementIndex(b_region->buffer, CBType::kInput);
  gemm_c_req_index_ = AllocateRequirementIndex(c_region->buffer, CBType::kOutput);

  auto set_requirement_fields = [&](int requirement_index, int page_size, int num_pages,
                                    const std::string& data_format) {
    ICHECK_GE(requirement_index, 0);
    ICHECK_LT(requirement_index, static_cast<int>(cb_requirements_.size()));
    auto& req = cb_requirements_[requirement_index];
    req.page_size = page_size;
    req.num_pages = num_pages;
    req.data_format = data_format;
  };
  auto maybe_double_buffer_republished_fragment_input = [&](int requirement_index,
                                                            const Buffer& buffer,
                                                            int consumed_pages) {
    if (!buffer.defined() || GetStorageScope(buffer) != "blackhole.acc") {
      return;
    }
    const std::string buffer_identity = BufferIdentityName(buffer);
    CBFlowClass flow_class = CBFlowClass::kStream;
    if (auto contract_it = buffer_flow_contracts_.find(buffer_identity);
        contract_it != buffer_flow_contracts_.end()) {
      flow_class = contract_it->second.flow_class;
    } else if (auto use_count_it =
                   cb_consumed_fragment_use_count_by_buffer_identity_.find(buffer_identity);
               use_count_it != cb_consumed_fragment_use_count_by_buffer_identity_.end() &&
               use_count_it->second > 1) {
      flow_class = CBFlowClass::kRepublish;
    }
    auto& req = cb_requirements_[requirement_index];
    req.flow_class = flow_class;
    req.publish_pages_per_event = std::max(req.publish_pages_per_event, std::max(1, consumed_pages));
    req.consume_pages_per_event = std::max(req.consume_pages_per_event, std::max(1, consumed_pages));
    if (flow_class == CBFlowClass::kRepublish) {
      // Compute-produced inputs that are republished later in the same kernel
      // keep a second page set resident so the later reserve_back rotates onto
      // fresh storage instead of reusing the just-consumed page.
      req.num_pages = std::max(req.num_pages, 2 * std::max(1, consumed_pages));
      req.initial_reserve_pages = std::max(1, consumed_pages);
    }
  };

  set_requirement_fields(gemm_a_req_index_, ab_tile_bytes, num_m_tiles * num_k_tiles,
                         DataTypeToDataFormat(gemm_a_dtype_));
  set_requirement_fields(gemm_b_req_index_, ab_tile_bytes, num_k_tiles * num_n_tiles,
                         DataTypeToDataFormat(gemm_b_dtype_));
  set_requirement_fields(gemm_c_req_index_, c_tile_bytes, num_m_tiles * num_n_tiles,
                         DataTypeToDataFormat(gemm_c_dtype_));
  maybe_double_buffer_republished_fragment_input(gemm_a_req_index_, a_region->buffer,
                                                 num_m_tiles * num_k_tiles);
  maybe_double_buffer_republished_fragment_input(gemm_b_req_index_, b_region->buffer,
                                                 num_k_tiles * num_n_tiles);
  cb_requirements_[gemm_a_req_index_].lifetime_begin = 0;
  cb_requirements_[gemm_a_req_index_].lifetime_end = 0;
  cb_requirements_[gemm_b_req_index_].lifetime_begin = 0;
  cb_requirements_[gemm_b_req_index_].lifetime_end = 0;
  cb_requirements_[gemm_c_req_index_].lifetime_begin = 1;
  cb_requirements_[gemm_c_req_index_].lifetime_end = 1;
}

// Detect clear operation using Op comparison
bool LowerBlackholeOps::IsClearOperation(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.clear";
}

// Detect copy operation using buffer scopes
bool LowerBlackholeOps::IsCopyOperation(const BufferStoreNode* op) const {
  // Check if this is a BufferStore where value is a BufferLoad from another buffer
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    return !op->buffer.same_as(load->buffer);
  }
  return false;
}

// Determine copy direction
CopyDirection LowerBlackholeOps::GetCopyDirection(const BufferStoreNode* op) const {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return CopyDirection::kUnknown;

  std::string dst_scope = GetStorageScope(op->buffer);
  std::string src_scope = GetStorageScope(load->buffer);

  // Helper to check if scope indicates CB (shared memory or canonicalized blackhole.cb.*)
  auto isCBScope = [](const std::string& scope) {
    if (scope.rfind("shared", 0) == 0) return true;
    auto s = runtime::StorageScope::Create(scope);
    return s.rank == runtime::StorageRank::kBlackholeCB;
  };

  // Helper to check if scope indicates DRAM (global memory)
  auto isDRAMScope = [](const std::string& scope) {
    return scope.empty() || scope == "global";
  };

  auto isAccumulatorLikeScope = [](const std::string& scope) {
    if (scope.rfind("local", 0) == 0) return true;
    auto s = runtime::StorageScope::Create(scope);
    return s.rank == runtime::StorageRank::kBlackholeAccumulator;
  };

  if (current_func_->GetAttr<Array<Any>>("blackhole.segment_plan").has_value() &&
      isAccumulatorLikeScope(src_scope) && isDRAMScope(dst_scope)) {
    return CopyDirection::kCBToDram;
  }

  // DRAM -> CB (global -> shared)
  if (isDRAMScope(src_scope) && isCBScope(dst_scope)) {
    return CopyDirection::kDramToCB;
  }

  // DRAM -> DRAM (global -> global)
  if (isDRAMScope(src_scope) && isDRAMScope(dst_scope)) {
    return CopyDirection::kDramToDram;
  }

  // CB -> DRAM (shared -> global)
  if (isCBScope(src_scope) && isDRAMScope(dst_scope)) {
    return CopyDirection::kCBToDram;
  }

  // CB -> CB (shared -> shared)
  if (isCBScope(src_scope) && isCBScope(dst_scope)) {
    return CopyDirection::kCBToCB;
  }

  // local/accumulator -> CB (fragment/local staging -> shared/CB)
  if (isAccumulatorLikeScope(src_scope) && isCBScope(dst_scope)) {
    return CopyDirection::kLocalToCB;
  }

  return CopyDirection::kUnknown;
}

PrimExpr LowerBlackholeOps::ZeroThreadAndLoopVars(const PrimExpr& expr,
                                                  const Var& loop_var) const {
  if (!loop_var.defined()) {
    return ZeroThreadAndLoopVars(expr, std::vector<Var>{});
  }
  return ZeroThreadAndLoopVars(expr, std::vector<Var>{loop_var});
}

PrimExpr LowerBlackholeOps::ZeroThreadAndLoopVars(const PrimExpr& expr,
                                                  const std::vector<Var>& loop_vars) const {
  Map<Var, PrimExpr> subst_map;
  for (const auto& loop_var : loop_vars) {
    if (loop_var.defined()) {
      subst_map.Set(loop_var, IntImm(loop_var.dtype(), 0));
    }
  }
  for (const auto* thread_var : thread_index_vars_) {
    subst_map.Set(GetRef<Var>(thread_var), IntImm(thread_var->dtype, 0));
  }
  if (!thread_index_var_names_.empty()) {
    tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
      if (const auto* var = node.as<tir::VarNode>()) {
        if (!thread_index_vars_.count(var) &&
            thread_index_var_names_.count(var->name_hint)) {
          subst_map.Set(GetRef<Var>(var), IntImm(var->dtype, 0));
        }
      }
    });
  }
  if (subst_map.empty()) {
    return expr;
  }
  Analyzer analyzer;
  return analyzer.Simplify(tir::Substitute(expr, subst_map));
}

bool LowerBlackholeOps::ExprUsesTransportVar(const PrimExpr& expr,
                                             const std::vector<Var>& loop_vars) const {
  bool uses_transport_var = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (const auto* var = node.as<tir::VarNode>()) {
      if (thread_index_vars_.count(var)) {
        uses_transport_var = true;
        return;
      }
      if (thread_index_var_names_.count(var->name_hint)) {
        uses_transport_var = true;
        return;
      }
      if (block_index_vars_.count(var)) {
        uses_transport_var = true;
        return;
      }
      if (block_index_var_names_.count(var->name_hint)) {
        uses_transport_var = true;
        return;
      }
      for (const auto& loop_var : loop_vars) {
        if (loop_var.defined() && var == loop_var.get()) {
          uses_transport_var = true;
          return;
        }
      }
    }
  });
  return uses_transport_var;
}

Var LowerBlackholeOps::SelectLogicalRowThreadVar(int64_t logical_rows) const {
  std::vector<Var> exact_extent_matches;
  std::vector<Var> non_unit_matches;
  for (const auto* thread_var : thread_index_vars_) {
    auto extent_it = thread_index_var_static_extents_.find(thread_var);
    if (extent_it == thread_index_var_static_extents_.end()) {
      continue;
    }
    const int64_t extent = extent_it->second;
    if (extent <= 1) {
      continue;
    }
    Var var = GetRef<Var>(thread_var);
    non_unit_matches.push_back(var);
    if (logical_rows > 0 && extent == logical_rows) {
      exact_extent_matches.push_back(var);
    }
  }
  if (exact_extent_matches.size() == 1) {
    return exact_extent_matches.front();
  }
  if (non_unit_matches.size() == 1) {
    return non_unit_matches.front();
  }
  return Var();
}

std::pair<int, int> LowerBlackholeOps::SelectStagedCopyTransportAxes(
    const Array<PrimExpr>& global_indices, const std::vector<Var>& loop_vars) const {
  std::vector<int> transport_axes;
  for (size_t i = 0; i < global_indices.size(); ++i) {
    if (ExprUsesTransportVar(global_indices[i], loop_vars)) {
      transport_axes.push_back(static_cast<int>(i));
    }
  }
  if (transport_axes.size() >= 2U) {
    return {transport_axes.front(), transport_axes.back()};
  }
  return {0, 1};
}

std::vector<int64_t> LowerBlackholeOps::BuildStagedCopyHostAxisOrder(
    const Array<PrimExpr>& global_indices, const Array<Integer>& global_shape, int row_axis,
    int col_axis) const {
  const size_t ndim = !global_shape.empty() ? global_shape.size() : global_indices.size();
  if (ndim < 2 || row_axis < 0 || col_axis < 0 ||
      row_axis >= static_cast<int>(ndim) || col_axis >= static_cast<int>(ndim) ||
      row_axis == col_axis) {
    return {};
  }

  std::vector<int64_t> axis_order;
  axis_order.reserve(ndim);
  for (size_t axis = 0; axis < ndim; ++axis) {
    if (static_cast<int>(axis) == row_axis || static_cast<int>(axis) == col_axis) {
      continue;
    }
    axis_order.push_back(static_cast<int64_t>(axis));
  }
  axis_order.push_back(static_cast<int64_t>(row_axis));
  axis_order.push_back(static_cast<int64_t>(col_axis));
  return axis_order;
}

PrimExpr LowerBlackholeOps::InferCopyTileIndex(const BufferStoreNode* op,
                                               const Var& loop_var) const {
  const auto* load = op->value.as<BufferLoadNode>();
  ICHECK(load) << "InferCopyTileIndex requires BufferLoad copy source";

  CopyDirection direction = GetCopyDirection(op);
  const bool segmented_gemm = !gemm_a_buffer_name_.empty();
  const bool accumulator_like_src =
      direction == CopyDirection::kCBToDram &&
      (GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
       runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
           runtime::StorageRank::kBlackholeAccumulator);
  const bool transpose_b_reader = false;
  const Buffer& global_buffer =
      direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;
  const Buffer& shared_buffer =
      direction == CopyDirection::kDramToCB ? op->buffer : load->buffer;
  const Array<Integer>& global_shape =
      direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
  const auto [row_axis, col_axis] =
      SelectStagedCopyTransportAxes(global_indices, std::vector<Var>{loop_var});

  Analyzer analyzer;
  const StagedCopyGlobalIndexInfo global_info = ResolveStagedCopyGlobalIndexInfo(
      global_buffer, global_indices, global_shape, row_axis, col_axis,
      "Blackhole staged copy currently expects static global buffer shape",
      "Blackhole staged copy requires rank-2 shape metadata after FlattenBuffer",
      [&](const PrimExpr& expr) { return ZeroThreadAndLoopVars(expr, loop_var); }, &analyzer);
  const bool use_page_transport = UseStagedCopyPageTransport(shared_buffer);
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, copy_intermediate_shape_, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, global_info.global_rows, global_info.global_cols,
      use_page_transport);
  return LinearizeStagedCopyTransportIndex(
      &analyzer, global_info.base_row, global_info.base_col, global_info.outer_slice_index,
      geometry);
}

PrimExpr LowerBlackholeOps::InferStagedCopyBaseTileIndex(
    const BufferStoreNode* op, const std::vector<Var>& loop_vars_to_zero) const {
  const auto* load = op->value.as<BufferLoadNode>();
  ICHECK(load) << "InferStagedCopyBaseTileIndex requires BufferLoad copy source";

  CopyDirection direction = GetCopyDirection(op);
  const Buffer& global_buffer =
      direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;
  const bool is_gemm_b_input =
      direction == CopyDirection::kDramToCB &&
      ((gemm_b_buffer_.defined() && SameBufferIdentity(op->buffer, gemm_b_buffer_)) ||
       (buffer_to_req_.count(op->buffer) && buffer_to_req_.at(op->buffer) == gemm_b_req_index_));
  const bool segmented_gemm = !gemm_a_buffer_name_.empty();
  const bool accumulator_like_src =
      direction == CopyDirection::kCBToDram &&
      (GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
       runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
           runtime::StorageRank::kBlackholeAccumulator);

  Analyzer analyzer;
  const bool transpose_b_reader = gemm_transpose_b_ && is_gemm_b_input;
  const Buffer& shared_buffer =
      direction == CopyDirection::kDramToCB ? op->buffer : load->buffer;
  const Array<Integer>& global_shape =
      direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
  const auto [row_axis, col_axis] =
      SelectStagedCopyTransportAxes(global_indices, loop_vars_to_zero);
  const StagedCopyGlobalIndexInfo global_info = ResolveStagedCopyGlobalIndexInfo(
      global_buffer, global_indices, global_shape, row_axis, col_axis,
      "Blackhole staged copy currently expects static global buffer shape",
      "Blackhole staged copy requires rank-2 shape metadata after FlattenBuffer",
      [&](const PrimExpr& expr) { return ZeroThreadAndLoopVars(expr, loop_vars_to_zero); },
      &analyzer);
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, copy_intermediate_shape_, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);
  const int64_t effective_global_rows =
      transpose_b_reader ? global_info.global_cols : global_info.global_rows;
  const int64_t effective_global_cols =
      transpose_b_reader ? global_info.global_rows : global_info.global_cols;
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, effective_global_rows, effective_global_cols,
      UseStagedCopyPageTransport(shared_buffer));
  const PrimExpr transport_row = transpose_b_reader ? global_info.base_col : global_info.base_row;
  const PrimExpr transport_col = transpose_b_reader ? global_info.base_row : global_info.base_col;
  return LinearizeStagedCopyTransportIndex(&analyzer, transport_row, transport_col,
                                           global_info.outer_slice_index, geometry);
}

const BufferStoreNode* LowerBlackholeOps::FindNestedCopyStore(
    const Stmt& stmt, std::vector<Var>* nested_loop_vars) const {
  if (const auto* store = stmt.as<BufferStoreNode>()) {
    return IsCopyOperation(store) ? store : nullptr;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    return FindNestedCopyStore(attr->body, nested_loop_vars);
  }
  if (const auto* allocate = stmt.as<tir::AllocateNode>()) {
    return FindNestedCopyStore(allocate->body, nested_loop_vars);
  }
  if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
    for (const auto& child : seq->seq) {
      std::vector<Var> child_loop_vars = *nested_loop_vars;
      if (const BufferStoreNode* store = FindNestedCopyStore(child, &child_loop_vars)) {
        *nested_loop_vars = std::move(child_loop_vars);
        return store;
      }
    }
    return nullptr;
  }
  if (const auto* loop = stmt.as<ForNode>()) {
    const bool zero_loop_var = !loop->thread_binding.defined();
    if (zero_loop_var) {
      nested_loop_vars->push_back(loop->loop_var);
    }
    const BufferStoreNode* store = FindNestedCopyStore(loop->body, nested_loop_vars);
    if (!store && zero_loop_var) {
      nested_loop_vars->pop_back();
    }
    return store;
  }
  return nullptr;
}

void LowerBlackholeOps::CollectNestedCopyStores(const Stmt& stmt,
                                                std::vector<Var>* loop_stack,
                                                std::vector<NestedCopyMatch>* matches) const {
  if (const auto* store = stmt.as<BufferStoreNode>()) {
    if (IsCopyOperation(store)) {
      matches->push_back({store, *loop_stack, GetCopyDirection(store)});
    }
    return;
  }
  if (const auto* attr = stmt.as<AttrStmtNode>()) {
    CollectNestedCopyStores(attr->body, loop_stack, matches);
    return;
  }
  if (const auto* allocate = stmt.as<tir::AllocateNode>()) {
    CollectNestedCopyStores(allocate->body, loop_stack, matches);
    return;
  }
  if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
    for (const auto& child : seq->seq) {
      CollectNestedCopyStores(child, loop_stack, matches);
    }
    return;
  }
  if (const auto* loop = stmt.as<ForNode>()) {
    const bool zero_loop_var = !loop->thread_binding.defined();
    if (zero_loop_var) {
      loop_stack->push_back(loop->loop_var);
    }
    CollectNestedCopyStores(loop->body, loop_stack, matches);
    if (zero_loop_var) {
      loop_stack->pop_back();
    }
  }
}

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

void LowerBlackholeOps::RecordStagedCopyBufferBinding(const BufferStoreNode* op,
                                                      CopyDirection direction) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return;
  }
  needs_copy_runtime_args_ = true;
  if (direction == CopyDirection::kDramToCB) {
    copy_input_buffer_ = load->buffer;
    copy_input_buffer_name_ = BufferIdentityName(load->buffer);
  } else if (direction == CopyDirection::kCBToDram) {
    copy_output_buffer_ = op->buffer;
    copy_output_buffer_name_ = BufferIdentityName(op->buffer);
  }
}

void LowerBlackholeOps::RecordDramToDramCopy(const BufferStoreNode* op) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return;

  auto ensure_requirement = [&](const Buffer& buffer, CBType type) {
    auto it = buffer_to_req_.find(buffer);
    if (it != buffer_to_req_.end()) {
      return;
    }
    const int requirement_index = AllocateRequirementIndex(buffer, type);
    auto& req = cb_requirements_.at(requirement_index);
    req.num_pages = 1;
    req.data_format = DataTypeToDataFormatForBlackhole(buffer->dtype);
  };

  ensure_requirement(load->buffer, CBType::kInput);
  ensure_requirement(op->buffer, CBType::kOutput);
  needs_copy_runtime_args_ = true;
  copy_input_buffer_ = load->buffer;
  copy_output_buffer_ = op->buffer;
  copy_input_buffer_name_ = BufferIdentityName(load->buffer);
  copy_output_buffer_name_ = BufferIdentityName(op->buffer);
}

void LowerBlackholeOps::RegisterAccessor(const std::string& segment_kind,
                                         const Buffer& buffer,
                                         int compile_time_arg_offset,
                                         int compile_time_arg_count,
                                         int common_runtime_arg_offset,
                                         int common_runtime_arg_count,
                                         int args_config_bits,
                                         int transport_page_size_bytes,
                                         std::vector<int64_t> host_axis_order,
                                         bool transpose_2d) {
  auto it = std::find_if(accessor_descriptors_.begin(), accessor_descriptors_.end(),
                         [&](const AccessorDescriptor& desc) {
                           return desc.segment_kind == segment_kind &&
                                  SameBufferIdentity(desc.buffer, buffer) &&
                                  desc.compile_time_arg_offset == compile_time_arg_offset &&
                                  desc.compile_time_arg_count == compile_time_arg_count &&
                                  desc.common_runtime_arg_offset == common_runtime_arg_offset &&
                                  desc.common_runtime_arg_count == common_runtime_arg_count &&
                                  desc.args_config_bits == args_config_bits &&
                                  desc.transport_page_size_bytes == transport_page_size_bytes &&
                                  desc.host_axis_order == host_axis_order &&
                                  desc.transpose_2d == transpose_2d;
                         });
  if (it != accessor_descriptors_.end()) {
    return;
  }
  accessor_descriptors_.push_back(AccessorDescriptor{segment_kind,
                                                     buffer,
                                                     BufferIdentityName(buffer),
                                                     compile_time_arg_offset,
                                                     compile_time_arg_count,
                                                     common_runtime_arg_offset,
                                                     common_runtime_arg_count,
                                                     args_config_bits,
                                                     transport_page_size_bytes,
                                                     "interleaved",
                                                     "dram",
                                                     std::move(host_axis_order),
                                                     transpose_2d});
}

std::string LowerBlackholeOps::ResolveAccessorSegmentKind(CopyDirection direction) const {
  if (!current_segment_kind_.empty()) {
    return current_segment_kind_;
  }
  if (direction == CopyDirection::kDramToCB) {
    return !gemm_a_buffer_name_.empty() ? "reader" : "fused_dataflow";
  }
  if (direction == CopyDirection::kCBToDram || direction == CopyDirection::kLocalToCB) {
    return !gemm_a_buffer_name_.empty() ? "writer" : "fused_dataflow";
  }
  return "fused_dataflow";
}

int LowerBlackholeOps::GetOrAllocateSegmentAccessorSlot(
    std::unordered_map<std::string, int>* slot_map, const std::string& segment_kind,
    const Buffer& buffer) {
  const std::string key = MakeSegmentBufferKey(segment_kind, buffer);
  auto it = slot_map->find(key);
  if (it != slot_map->end()) {
    return it->second;
  }
  int next_slot = 0;
  for (const auto& [existing_key, slot] : *slot_map) {
    if (existing_key.rfind(segment_kind + ":", 0) == 0) {
      next_slot = std::max(next_slot, slot + 2);
    }
  }
  slot_map->emplace(key, next_slot);
  return next_slot;
}

int LowerBlackholeOps::GetReadAccessorSlot(const std::string& segment_kind, const Buffer& buffer,
                                           CopyDirection direction) {
  if (segment_kind == "fused_dataflow") {
    if (copy_input_buffer_.defined() && SameBufferIdentity(buffer, copy_input_buffer_)) {
      return 0;
    }
    return 0;
  }
  if (direction == CopyDirection::kDramToCB) {
    return GetOrAllocateSegmentAccessorSlot(&read_accessor_slots_, segment_kind, buffer);
  }
  return 0;
}

int LowerBlackholeOps::GetWriteAccessorSlot(const std::string& segment_kind, const Buffer& buffer,
                                            CopyDirection direction) {
  if (segment_kind == "fused_dataflow") {
    if (copy_output_buffer_.defined() && SameBufferIdentity(buffer, copy_output_buffer_)) {
      return 2;
    }
    return 0;
  }
  if (direction == CopyDirection::kCBToDram) {
    return GetOrAllocateSegmentAccessorSlot(&write_accessor_slots_, segment_kind, buffer);
  }
  return 0;
}

void LowerBlackholeOps::ActivateCurrentComputeContractPayload() {
  const std::string signature = EncodeGemmContractSignature(
      gemm_a_buffer_name_, gemm_b_buffer_name_, gemm_c_buffer_name_, gemm_m_, gemm_n_, gemm_k_,
      gemm_transpose_a_, gemm_transpose_b_, gemm_policy_type_, gemm_clear_accum_, gemm_k_pack_,
      gemm_wg_wait_, gemm_dst_full_sync_en_, gemm_bfp8_pack_precise_, gemm_defines_,
      gemm_named_compile_args_, gemm_mbarrier_buffer_name_, gemm_mbarrier_scope_,
      gemm_mbarrier_index_exprs_);
  auto it = compute_contract_payload_index_by_signature_.find(signature);
  if (it == compute_contract_payload_index_by_signature_.end()) {
    active_compute_contract_payload_index_ = -1;
    return;
  }
  active_compute_contract_payload_index_ = it->second;
}

void LowerBlackholeOps::RecordComputeEpilogueOp(Map<String, Any> op_payload) {
  compute_epilogue_payloads_flat_.push_back(op_payload);
  if (compute_epilogue_payloads_.empty()) {
    return;
  }

  std::vector<std::string> candidate_buffers;
  auto collect_buffer = [&](const char* key) {
    if (auto value = op_payload.Get(String(key))) {
      std::string buffer = Downcast<String>(value.value());
      if (!buffer.empty()) {
        candidate_buffers.push_back(std::move(buffer));
      }
    }
  };
  collect_buffer("dst_buffer");
  collect_buffer("src_buffer");
  collect_buffer("scalar_buffer");
  collect_buffer("lhs_buffer");
  collect_buffer("rhs_buffer");
  collect_buffer("add_buffer");

  int best_index = -1;
  int best_score = -1;
  for (size_t i = 0; i < multi_compute_contract_payloads_.size(); ++i) {
    const auto& contract = multi_compute_contract_payloads_[i];
    const std::string a_buffer =
        contract.Get(String("a_buffer"))
            ? static_cast<std::string>(Downcast<String>(contract.Get(String("a_buffer")).value()))
            : std::string();
    const std::string b_buffer =
        contract.Get(String("b_buffer"))
            ? static_cast<std::string>(Downcast<String>(contract.Get(String("b_buffer")).value()))
            : std::string();
    const std::string c_buffer =
        contract.Get(String("c_buffer"))
            ? static_cast<std::string>(Downcast<String>(contract.Get(String("c_buffer")).value()))
            : std::string();
    int score = 0;
    for (const auto& buffer : candidate_buffers) {
      if (!c_buffer.empty() && buffer == c_buffer) {
        score += 8;
      } else if ((!a_buffer.empty() && buffer == a_buffer) ||
                 (!b_buffer.empty() && buffer == b_buffer)) {
        score += 4;
      } else if (i < compute_contract_known_buffers_.size() &&
                 compute_contract_known_buffers_[i].count(buffer)) {
        score += 2;
      }
    }
    if (score > best_score) {
      best_score = score;
      best_index = static_cast<int>(i);
    }
  }

  if (best_score <= 0) {
    best_index = active_compute_contract_payload_index_;
  }
  if (best_index < 0 || best_index >= static_cast<int>(compute_epilogue_payloads_.size())) {
    return;
  }
  if (best_index < static_cast<int>(compute_contract_known_buffers_.size())) {
    if (auto value = op_payload.Get(String("dst_buffer"))) {
      compute_contract_known_buffers_[static_cast<size_t>(best_index)].insert(
          static_cast<std::string>(Downcast<String>(value.value())));
    }
  }
  compute_epilogue_payloads_[static_cast<size_t>(best_index)].push_back(std::move(op_payload));
}

Stmt LowerBlackholeOps::GenerateMatmulSequence(const CallNode* op,
                                              bool retain_in0,
                                              bool retain_in1,
                                              bool publish_out,
                                              bool reacquire_in0,
                                              bool reacquire_in1) {
  ICHECK_GE(gemm_a_req_index_, 0);
  ICHECK_GE(gemm_b_req_index_, 0);
  ICHECK_GE(gemm_c_req_index_, 0);
  ActivateCurrentComputeContractPayload();
  if (!gemm_clear_accum_) {
    return GenerateAccumulatingMatmulSequence(op, retain_in0, retain_in1,
                                              reacquire_in0, reacquire_in1);
  }
  return GenerateMatmulSequenceForOutputRequirement(gemm_c_req_index_, retain_in0, retain_in1,
                                                    publish_out, publish_out, reacquire_in0,
                                                    reacquire_in1);
}

Stmt LowerBlackholeOps::GenerateMatmulSequenceForOutputRequirement(int out_req_index,
                                                                  bool retain_in0,
                                                                  bool retain_in1,
                                                                  bool reserve_out,
                                                                  bool publish_out,
                                                                  bool reacquire_in0,
                                                                  bool reacquire_in1) {
  const int in0_id = gemm_a_req_index_;
  const int in1_id = gemm_b_req_index_;
  const int out_id = out_req_index;
  const int num_m_tiles = CeilDivToInt(gemm_m_, kBlackholeTileRows);
  const int num_n_tiles = CeilDivToInt(gemm_n_, kBlackholeTileCols);
  const int num_k_tiles = CeilDivToInt(gemm_k_, kBlackholeTileCols);
  const int num_a_tiles = num_m_tiles * num_k_tiles;
  const int num_b_tiles = num_k_tiles * num_n_tiles;
  const int num_c_tiles = num_m_tiles * num_n_tiles;
  ICHECK_LE(num_c_tiles, 16)
      << "Blackhole matmul lowering currently supports at most 16 output tiles per GEMM, but "
         "saw "
      << num_c_tiles << " for " << gemm_c_buffer_name_;

  std::vector<Stmt> stmts;
  std::vector<Stmt> deferred_reacquire_stmts;

  // 1. Initialize MM engine
  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(in0_id), IntImm32(in1_id)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_mm_init(),
      {IntImm32(in0_id), IntImm32(in1_id), IntImm32(out_id)}));

  // 2. Acquire tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));

  // 3. Wait for the full staged tile sets and execute the logical tile grid.
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_wait_front(),
      {IntImm32(in0_id), IntImm32(num_a_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_wait_front(),
      {IntImm32(in1_id), IntImm32(num_b_tiles)}));

  for (int mt = 0; mt < num_m_tiles; ++mt) {
    for (int nt = 0; nt < num_n_tiles; ++nt) {
      for (int kt = 0; kt < num_k_tiles; ++kt) {
        stmts.push_back(MakeBlackholeCall(
            blackhole_matmul_tiles(),
            {IntImm32(in0_id), IntImm32(in1_id), IntImm32(mt * num_k_tiles + kt),
             IntImm32(kt * num_n_tiles + nt), IntImm32(mt * num_n_tiles + nt)}));
      }
    }
  }
  if (!retain_in0) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(in0_id), IntImm32(num_a_tiles)}));
    if (reacquire_in0) {
      deferred_reacquire_stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(),
          {IntImm32(in0_id), IntImm32(num_a_tiles)}));
    }
  }
  if (!retain_in1) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(in1_id), IntImm32(num_b_tiles)}));
    if (reacquire_in1) {
      deferred_reacquire_stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(),
          {IntImm32(in1_id), IntImm32(num_b_tiles)}));
    }
  }

  // 4-5. Commit and wait
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));

  // 6-8. Pack and push output
  if (reserve_out) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_reserve_back(),
        {IntImm32(out_id), IntImm32(num_c_tiles)}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out_id)}));
  for (int out_tile = 0; out_tile < num_c_tiles; ++out_tile) {
    std::vector<PrimExpr> pack_args = {IntImm32(out_tile), IntImm32(out_id)};
    if (!publish_out) {
      pack_args.push_back(IntImm32(out_tile));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(), pack_args));
  }
  if (publish_out) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_push_back(),
        {IntImm32(out_id), IntImm32(num_c_tiles)}));
  }

  // 9. Release tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  for (const Stmt& reacquire : deferred_reacquire_stmts) {
    stmts.push_back(reacquire);
  }

  return SeqStmt::Flatten(stmts);
}

Buffer LowerBlackholeOps::CreateClearAccumPartialsBuffer(const Buffer& buffer) {
  const std::string partials_name =
      BufferIdentityName(buffer) + "_clear_accum_partials_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(buffer->shape, buffer->dtype, partials_name, GetStorageScope(buffer));
}

bool LowerBlackholeOps::ClearAccumReloadNeedsDataFormatReconfig() const {
  return gemm_c_dtype_ != gemm_b_dtype_;
}

Stmt LowerBlackholeOps::GenerateAddFragmentSequence(const Buffer& dst,
                                                    const Buffer& src,
                                                    const PrimExpr& num_elements) {
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("add_fragment", BufferIdentityName(dst));
  op_payload.Set("src_buffer", String(BufferIdentityName(src)));
  SetOptionalExprField(&op_payload, "num_elements_expr", num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
  return MakeBlackholeCall(blackhole_add_fragment(), {dst->data, src->data, num_elements});
}

Stmt LowerBlackholeOps::GenerateAddFragmentFromCBFrontSequence(const Buffer& dst,
                                                               int src_cb_id,
                                                               const PrimExpr& num_elements,
                                                               const Buffer& src_buffer) {
  const std::string dst_buffer_name = BufferIdentityName(dst);
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("add_fragment_from_cb_front", dst_buffer_name);
  op_payload.Set("src_buffer", String(BufferIdentityName(src_buffer)));
  auto contract_it = fragment_materialization_contracts_by_target_buffer_.find(dst_buffer_name);
  ICHECK(contract_it != fragment_materialization_contracts_by_target_buffer_.end())
      << "LowerBlackholeOps requires fragment_materialization_contract in SpatialProgram for "
         "add_fragment_from_cb_front destination "
      << dst_buffer_name;
  op_payload.Set(String(schema_key::kFragmentMaterializationContract), contract_it->second);
  SetOptionalExprField(&op_payload, "num_elements_expr", num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
  return MakeBlackholeCall(blackhole_add_fragment_from_cb_front(),
                           {dst->data, IntImm32(src_cb_id), num_elements});
}

Stmt LowerBlackholeOps::GenerateMatmulSequenceWithPartialReload(int out_req_index,
                                                                int partials_cb_id,
                                                                bool retain_in0,
                                                                bool retain_in1,
                                                                bool reserve_out,
                                                                bool publish_out,
                                                                bool reacquire_in0,
                                                                bool reacquire_in1) {
  const int in0_id = gemm_a_req_index_;
  const int in1_id = gemm_b_req_index_;
  const int out_id = out_req_index;
  const int num_m_tiles = CeilDivToInt(gemm_m_, kBlackholeTileRows);
  const int num_n_tiles = CeilDivToInt(gemm_n_, kBlackholeTileCols);
  const int num_k_tiles = CeilDivToInt(gemm_k_, kBlackholeTileCols);
  const int num_a_tiles = num_m_tiles * num_k_tiles;
  const int num_b_tiles = num_k_tiles * num_n_tiles;
  const int num_c_tiles = num_m_tiles * num_n_tiles;
  ICHECK_LE(num_c_tiles, 16)
      << "Blackhole matmul lowering currently supports at most 16 output tiles per GEMM, but "
         "saw "
      << num_c_tiles << " for " << gemm_c_buffer_name_;

  std::vector<Stmt> stmts;
  std::vector<Stmt> deferred_reacquire_stmts;

  stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                    {IntImm32(in0_id), IntImm32(in1_id)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mm_init(),
                                    {IntImm32(in0_id), IntImm32(in1_id), IntImm32(out_id)}));
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(partials_cb_id), IntImm32(num_c_tiles)}));
  if (ClearAccumReloadNeedsDataFormatReconfig()) {
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short_with_dt(),
                                      {IntImm32(in1_id), IntImm32(partials_cb_id)}));
  } else {
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(),
                                      {IntImm32(partials_cb_id)}));
  }
  for (int tile = 0; tile < num_c_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_from_cb(),
                                      {IntImm32(partials_cb_id), IntImm32(tile), IntImm32(tile)}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(partials_cb_id), IntImm32(num_c_tiles)}));
  if (ClearAccumReloadNeedsDataFormatReconfig()) {
    stmts.push_back(MakeBlackholeCall(blackhole_mm_init_short_with_dt(),
                                      {IntImm32(in0_id), IntImm32(in1_id), IntImm32(partials_cb_id)}));
  } else {
    stmts.push_back(
        MakeBlackholeCall(blackhole_mm_init_short(), {IntImm32(in0_id), IntImm32(in1_id)}));
  }

  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(in0_id), IntImm32(num_a_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(in1_id), IntImm32(num_b_tiles)}));

  for (int mt = 0; mt < num_m_tiles; ++mt) {
    for (int nt = 0; nt < num_n_tiles; ++nt) {
      for (int kt = 0; kt < num_k_tiles; ++kt) {
        stmts.push_back(MakeBlackholeCall(
            blackhole_matmul_tiles(),
            {IntImm32(in0_id), IntImm32(in1_id), IntImm32(mt * num_k_tiles + kt),
             IntImm32(kt * num_n_tiles + nt), IntImm32(mt * num_n_tiles + nt)}));
      }
    }
  }
  if (!retain_in0) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(in0_id), IntImm32(num_a_tiles)}));
    if (reacquire_in0) {
      deferred_reacquire_stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(in0_id), IntImm32(num_a_tiles)}));
    }
  }
  if (!retain_in1) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(in1_id), IntImm32(num_b_tiles)}));
    if (reacquire_in1) {
      deferred_reacquire_stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(in1_id), IntImm32(num_b_tiles)}));
    }
  }

  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
  if (reserve_out) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(out_id), IntImm32(num_c_tiles)}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out_id)}));
  for (int out_tile = 0; out_tile < num_c_tiles; ++out_tile) {
    std::vector<PrimExpr> pack_args = {IntImm32(out_tile), IntImm32(out_id)};
    if (!publish_out) {
      pack_args.push_back(IntImm32(out_tile));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(), pack_args));
  }
  if (publish_out) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(out_id), IntImm32(num_c_tiles)}));
  }

  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  for (const Stmt& reacquire : deferred_reacquire_stmts) {
    stmts.push_back(reacquire);
  }
  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateAccumulatingMatmulSequence(const CallNode* op,
                                                           bool retain_in0,
                                                           bool retain_in1,
                                                           bool reacquire_in0,
                                                           bool reacquire_in1) {
  ICHECK(op != nullptr);
  ICHECK(IsUnsupportedResidualLocalScope(gemm_c_buffer_))
      << "Blackhole clear_accum=false lowering currently requires a local fragment destination, "
         "but "
      << gemm_c_buffer_name_ << " uses scope " << GetStorageScope(gemm_c_buffer_);

  const int num_m_tiles = CeilDivToInt(gemm_m_, kBlackholeTileRows);
  const int num_n_tiles = CeilDivToInt(gemm_n_, kBlackholeTileCols);
  const int num_c_tiles = num_m_tiles * num_n_tiles;
  const int c_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * gemm_c_dtype_.bytes();

  Buffer scratch_buffer = CreateClearAccumPartialsBuffer(gemm_c_buffer_);
  const int scratch_req_index = AllocateRequirementIndex(scratch_buffer, CBType::kIntermediate);
  SetRequirementPageLayout(scratch_req_index, c_tile_bytes, num_c_tiles);
  auto& scratch_req = cb_requirements_.at(scratch_req_index);
  scratch_req.data_format = DataTypeToDataFormat(gemm_c_dtype_);
  scratch_req.flow_class = CBFlowClass::kStream;
  scratch_req.publish_pages_per_event =
      std::max(scratch_req.publish_pages_per_event, num_c_tiles);
  scratch_req.consume_pages_per_event =
      std::max(scratch_req.consume_pages_per_event, num_c_tiles);

  const int64_t num_elements = GetLogicalBufferElementCount(gemm_c_buffer_);
  ICHECK_GT(num_elements, 0)
      << "Blackhole clear_accum=false lowering requires a static logical element count for "
      << gemm_c_buffer_name_;
  ICHECK_LE(num_elements, std::numeric_limits<int>::max())
      << "Blackhole clear_accum=false lowering requires num_elements to fit in int32 for "
      << gemm_c_buffer_name_;

  std::vector<Stmt> stmts;
  stmts.push_back(
      GenerateMatmulSequenceForOutputRequirement(scratch_req_index, retain_in0, retain_in1,
                                                 /*reserve_out=*/true, /*publish_out=*/true,
                                                 reacquire_in0, reacquire_in1));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(),
                        {IntImm32(scratch_req_index), IntImm32(num_c_tiles)}));
  stmts.push_back(GenerateAddFragmentFromCBFrontSequence(
      gemm_c_buffer_, scratch_req_index, IntImm32(static_cast<int>(num_elements)), scratch_buffer));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(),
                        {IntImm32(scratch_req_index), IntImm32(num_c_tiles)}));

  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateCopySequence(const BufferStoreNode* op) {
  CopyDirection direction = GetCopyDirection(op);

  if (direction == CopyDirection::kUnknown) {
    LOG(WARNING) << "LowerBlackholeOps: Unknown copy direction, falling back";
    return StmtExprMutator::VisitStmt_(op);
  }

  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return StmtExprMutator::VisitStmt_(op);

  std::vector<Stmt> stmts;

  switch (direction) {
    case CopyDirection::kDramToCB: {
      // Staged DRAM -> shared copies should be collapsed at loop granularity.
      return GetRef<Stmt>(op);
    }

    case CopyDirection::kDramToDram: {
      // Stage 2 transition path: reconnect pure copy to builtin-driven TIR first.
      // Execution may still temporarily rely on the minimal runtime emitter, but
      // the copy semantics should now exist explicitly in the lowered TIR body.
      RecordDramToDramCopy(op);

      const auto* load = op->value.as<BufferLoadNode>();
      if (!load) {
        return GetRef<Stmt>(op);
      }

      const int src_cb_id = buffer_to_req_.at(load->buffer);
      const int dst_cb_id = buffer_to_req_.at(op->buffer);
      const int tile_bytes = EstimateCopyPageSize(load->buffer);

      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(src_cb_id), IntImm32(1)}));
      const int input_accessor_slot =
          GetReadAccessorSlot("fused_dataflow", load->buffer, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {load->buffer->data, IntImm32(0), IntImm32(src_cb_id), IntImm32(tile_bytes),
           IntImm32(input_accessor_slot)}));
      RegisterAccessor("fused_dataflow", load->buffer, input_accessor_slot, 2, 0, 0, 2,
                       tile_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(src_cb_id), IntImm32(1)}));

      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(src_cb_id), IntImm32(1)}));
      const int output_accessor_slot =
          GetWriteAccessorSlot("fused_dataflow", op->buffer, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_tile_from_cb(),
          {IntImm32(src_cb_id), op->buffer->data, IntImm32(0), IntImm32(tile_bytes),
           IntImm32(output_accessor_slot)}));
      RegisterAccessor("fused_dataflow", op->buffer, output_accessor_slot, 2, 0, 0, 2,
                       tile_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(src_cb_id), IntImm32(1)}));

      return SeqStmt::Flatten(stmts);
    }

    case CopyDirection::kCBToDram: {
      // Staged shared -> DRAM copies should be collapsed at loop granularity.
      return GetRef<Stmt>(op);
    }

    case CopyDirection::kCBToCB: {
      // CB -> CB (local copy)
      int src_cb_id = AllocateRequirementIndex(load->buffer, CBType::kIntermediate);
      int dst_cb_id = AllocateRequirementIndex(op->buffer, CBType::kIntermediate);

      // cb_wait_front(src_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(src_cb_id), IntImm32(1)}));

      // cb_reserve_back(dst_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(dst_cb_id), IntImm32(1)}));

      // Note: local copy would use memcpy or similar
      // For now, just pop and push markers

      // cb_push_back(dst_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(dst_cb_id), IntImm32(1)}));

      // cb_pop_front(src_cb_id, 1)
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(src_cb_id), IntImm32(1)}));
      break;
    }

    default:
      return StmtExprMutator::VisitStmt_(op);
  }

  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateCopySequence(const BufferStoreNode* op,
                                             const PrimExpr& tile_index) {
  CopyDirection direction = GetCopyDirection(op);
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return GetRef<Stmt>(op);
  }

  std::vector<Stmt> stmts;
  auto maybe_wrap_segment_stmt = [&](const std::string& segment_kind, Stmt stmt) -> Stmt {
    if (current_segment_kind_.empty() && segment_kind != "fused_dataflow") {
      return AttrStmt(StringImm("blackhole.segment_kind"), "blackhole.segment_kind",
                      StringImm(segment_kind), stmt);
    }
    return stmt;
  };
  switch (direction) {
    case CopyDirection::kDramToCB: {
      const std::string segment_kind = ResolveAccessorSegmentKind(direction);
      const bool segmented_gemm = segment_kind == "reader";
      int cb_id = AllocateRequirementIndex(
          op->buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
      int tile_bytes = EstimateCopyPageSize(op->buffer);
      RecordStagedCopyBufferBinding(op, direction);
      const Array<PrimExpr>& global_indices = load->indices;
      const Array<Integer>& global_shape = copy_input_shape_;
      const auto [row_axis, col_axis] =
          SelectStagedCopyTransportAxes(global_indices, {});
      const std::vector<int64_t> host_axis_order =
          BuildStagedCopyHostAxisOrder(global_indices, global_shape, row_axis, col_axis);
      const int accessor_slot = GetReadAccessorSlot(segment_kind, load->buffer, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {load->buffer->data, tile_index, IntImm32(cb_id), IntImm32(tile_bytes),
           IntImm32(accessor_slot)}));
      RegisterAccessor(segment_kind, load->buffer,
                       accessor_slot, 2, 0, 0, 2, tile_bytes, host_axis_order);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      return maybe_wrap_segment_stmt(segment_kind, SeqStmt::Flatten(stmts));
    }
    case CopyDirection::kCBToDram: {
      const std::string segment_kind = ResolveAccessorSegmentKind(direction);
      const bool segmented_gemm = segment_kind == "writer";
      const bool accumulator_like_src =
          GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
          runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
              runtime::StorageRank::kBlackholeAccumulator;
      int cb_id = AllocateRequirementIndex(
          load->buffer,
          (segmented_gemm && accumulator_like_src) ? CBType::kOutput : CBType::kIntermediate);
      int tile_bytes = EstimateCopyPageSize(load->buffer);
      RecordStagedCopyBufferBinding(op, direction);
      const Array<PrimExpr>& global_indices = op->indices;
      const Array<Integer>& global_shape = copy_output_shape_;
      const auto [row_axis, col_axis] =
          SelectStagedCopyTransportAxes(global_indices, {});
      const std::vector<int64_t> host_axis_order =
          BuildStagedCopyHostAxisOrder(global_indices, global_shape, row_axis, col_axis);
      const int accessor_slot = GetWriteAccessorSlot(segment_kind, op->buffer, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_tile_from_cb(),
          {IntImm32(cb_id), op->buffer->data, tile_index, IntImm32(tile_bytes),
           IntImm32(accessor_slot)}));
      RegisterAccessor(segment_kind, op->buffer,
                       accessor_slot, 2, 0, 0, 2, tile_bytes, host_axis_order);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      return maybe_wrap_segment_stmt(segment_kind, SeqStmt::Flatten(stmts));
    }
    default:
      return GenerateCopySequence(op);
  }
}

Stmt LowerBlackholeOps::GenerateStagedCopyLoopSequence(const BufferStoreNode* op,
                                                       const PrimExpr& base_tile_index) {
  CopyDirection direction = GetCopyDirection(op);
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return GetRef<Stmt>(op);
  }

  const std::string segment_kind = ResolveAccessorSegmentKind(direction);
  const bool segmented_gemm = segment_kind == "reader" || segment_kind == "writer";
  const bool accumulator_like_src =
      direction == CopyDirection::kCBToDram &&
      (GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
       runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
           runtime::StorageRank::kBlackholeAccumulator);
  const bool transpose_b_reader =
      direction == CopyDirection::kDramToCB && segmented_gemm && gemm_transpose_b_ &&
      ((gemm_b_buffer_.defined() && SameBufferIdentity(op->buffer, gemm_b_buffer_)) ||
       (buffer_to_req_.count(op->buffer) && buffer_to_req_.at(op->buffer) == gemm_b_req_index_));

  const Buffer& shared_buffer =
      direction == CopyDirection::kDramToCB ? op->buffer : load->buffer;
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, copy_intermediate_shape_, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);

  const bool use_page_transport = UseStagedCopyPageTransport(shared_buffer);
  const Buffer& global_buffer = direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;
  const Array<Integer>& global_shape =
      direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
  const auto [row_axis, col_axis] =
      SelectStagedCopyTransportAxes(global_indices, {});
  const std::vector<int64_t> host_axis_order =
      BuildStagedCopyHostAxisOrder(global_indices, global_shape, row_axis, col_axis);
  int64_t global_rows = 0;
  int64_t global_cols = 0;
  std::tie(global_rows, global_cols) = ResolveStaticShape2DFromBufferAxesOrMetadata(
      global_buffer, global_shape, row_axis, col_axis,
      "Blackhole staged copy currently expects static global buffer shape",
      "Blackhole staged copy requires rank-2 global shape metadata after FlattenBuffer");
  int64_t effective_global_rows = global_rows;
  if (transpose_b_reader) {
    effective_global_rows = global_cols;
    global_cols = global_rows;
  }
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, effective_global_rows, global_cols,
      use_page_transport);
  if (geometry.use_page_transport) {
    ValidateStagedStickCopyGlobalWidthDivisible(geometry.global_cols, geometry.shared_cols);
  }
  int segmented_reader_tile_limit = -1;
  if (direction == CopyDirection::kDramToCB && segmented_gemm && !geometry.use_page_transport) {
    auto it = gemm_input_buffer_num_tiles_.find(BufferIdentityName(shared_buffer));
    if (it != gemm_input_buffer_num_tiles_.end()) {
      segmented_reader_tile_limit = it->second;
    }
  }
  const int tiles_per_row = static_cast<int>(geometry.global_cols / kBlackholeTileCols);
  const int pages_per_row =
      geometry.use_page_transport ? static_cast<int>(geometry.global_cols / geometry.shared_cols) : 0;
  Analyzer analyzer;

  std::vector<Stmt> stmts;
  auto maybe_wrap_segment_stmt = [&](Stmt stmt) -> Stmt {
    if (current_segment_kind_.empty() && segment_kind != "fused_dataflow") {
      return AttrStmt(StringImm("blackhole.segment_kind"), "blackhole.segment_kind",
                      StringImm(segment_kind), stmt);
    }
    return stmt;
  };
  auto make_tile_index = [&](int subtile_row, int subtile_col) -> PrimExpr {
    PrimExpr tile_index = base_tile_index;
    if (subtile_row != 0) {
      tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_row * tiles_per_row));
    }
    if (subtile_col != 0) {
      tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_col));
    }
    return tile_index;
  };

  auto make_page_index = [&](int page_row) -> PrimExpr {
    PrimExpr page_index = base_tile_index;
    if (page_row != 0) {
      page_index = analyzer.Simplify(page_index + IntImm32(page_row * pages_per_row));
    }
    return page_index;
  };

  if (direction == CopyDirection::kDramToCB) {
    int cb_id = AllocateRequirementIndex(
        op->buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
    RecordStagedCopyBufferBinding(op, direction);
    const int accessor_slot = GetReadAccessorSlot(segment_kind, load->buffer, direction);
    if (use_page_transport) {
      SetRequirementPageLayout(cb_id, geometry.shared_bytes, 1);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      for (int page_row = 0; page_row < shared_rows; ++page_row) {
        PrimExpr page_index = make_page_index(page_row);
        stmts.push_back(MakeBlackholeCall(
            blackhole_read_page_to_cb(), {load->buffer->data, page_index, IntImm32(cb_id),
                                          IntImm32(geometry.page_bytes), IntImm32(accessor_slot),
                                          IntImm32(page_row * geometry.l1_stick_stride)}));
        RegisterAccessor(segment_kind, load->buffer,
                         accessor_slot, 2, 0, 0, 2, geometry.page_bytes, host_axis_order,
                         transpose_b_reader);
      }
      stmts.push_back(MakeBlackholeCall(
          blackhole_noc_async_read_barrier(), {}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
    }
    const int total_subtiles = geometry.subtile_rows * geometry.subtile_cols;
    int tile_emit_count = total_subtiles;
    if (segmented_reader_tile_limit > 0) {
      ICHECK_LE(segmented_reader_tile_limit, total_subtiles)
          << "LowerBlackholeOps segmented reader transport exceeds staged copy shape for buffer "
          << BufferIdentityName(shared_buffer);
      tile_emit_count = segmented_reader_tile_limit;
    }
    ICHECK_GT(geometry.subtile_cols, 0);
    for (int subtile_ordinal = 0; subtile_ordinal < tile_emit_count; ++subtile_ordinal) {
      const int subtile_row = subtile_ordinal / geometry.subtile_cols;
      const int subtile_col = subtile_ordinal % geometry.subtile_cols;
      PrimExpr tile_index = make_tile_index(subtile_row, subtile_col);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {load->buffer->data, tile_index, IntImm32(cb_id), IntImm32(geometry.tile_bytes),
           IntImm32(accessor_slot)}));
      RegisterAccessor(segment_kind, load->buffer,
                       accessor_slot, 2, 0, 0, 2, geometry.tile_bytes, host_axis_order,
                       transpose_b_reader);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
    }
    return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
  }

  if (direction == CopyDirection::kCBToDram) {
    int cb_id = AllocateRequirementIndex(
        load->buffer,
        (segmented_gemm && accumulator_like_src) ? CBType::kOutput : CBType::kIntermediate);
    RecordStagedCopyBufferBinding(op, direction);
    const int accessor_slot = GetWriteAccessorSlot(segment_kind, op->buffer, direction);
    if (use_page_transport) {
      SetRequirementPageLayout(cb_id, geometry.shared_bytes, 1);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
      for (int page_row = 0; page_row < shared_rows; ++page_row) {
        PrimExpr page_index = make_page_index(page_row);
        stmts.push_back(MakeBlackholeCall(
            blackhole_write_page_from_cb(), {IntImm32(cb_id), op->buffer->data, page_index,
                                             IntImm32(geometry.page_bytes),
                                             IntImm32(accessor_slot),
                                             IntImm32(page_row * geometry.l1_stick_stride)}));
        RegisterAccessor(segment_kind, op->buffer,
                         accessor_slot, 2, 0, 0, 2, geometry.page_bytes, host_axis_order);
      }
      stmts.push_back(MakeBlackholeCall(
          blackhole_noc_async_write_barrier(), {}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
    }
    for (int subtile_row = 0; subtile_row < geometry.subtile_rows; ++subtile_row) {
      for (int subtile_col = 0; subtile_col < geometry.subtile_cols; ++subtile_col) {
        PrimExpr tile_index = make_tile_index(subtile_row, subtile_col);
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_write_tile_from_cb(),
            {IntImm32(cb_id), op->buffer->data, tile_index, IntImm32(geometry.tile_bytes),
             IntImm32(accessor_slot)}));
        RegisterAccessor(segment_kind, op->buffer,
                         accessor_slot, 2, 0, 0, 2, geometry.tile_bytes, host_axis_order);
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      }
    }
    return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
  }

  return GenerateCopySequence(op);
}

Stmt LowerBlackholeOps::GenerateFusedStagedCopySequence(const BufferStoreNode* dram_to_cb,
                                                        const BufferStoreNode* cb_to_dram,
                                                        const PrimExpr& base_tile_index) {
  const auto* dram_load = dram_to_cb->value.as<BufferLoadNode>();
  const auto* cb_load = cb_to_dram->value.as<BufferLoadNode>();
  if (!dram_load || !cb_load) {
    return GetRef<Stmt>(dram_to_cb);
  }

  const Buffer& shared_buffer = dram_to_cb->buffer;
  ICHECK(shared_buffer.same_as(cb_load->buffer))
      << "Fused staged copy expects DRAM->shared and shared->DRAM to use the same shared buffer";
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, copy_intermediate_shape_, /*segmented_gemm=*/false,
      /*transpose_b_reader=*/false, /*accumulator_like_src=*/false, gemm_m_, gemm_n_, gemm_k_);
  const bool use_page_transport = UseStagedCopyPageTransport(shared_buffer);
  int64_t global_rows = 0;
  int64_t global_cols = 0;
  std::tie(global_rows, global_cols) = ResolveStaticShape2DFromBufferOrMetadata(
      dram_load->buffer, copy_input_shape_,
      "Blackhole staged copy currently expects static global buffer shape",
      "Blackhole staged copy requires rank-2 global shape metadata after FlattenBuffer");
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, global_rows, global_cols, use_page_transport);
  if (geometry.use_page_transport) {
    ValidateStagedStickCopyGlobalWidthDivisible(geometry.global_cols, geometry.shared_cols);
  }
  const int tiles_per_row =
      geometry.use_page_transport ? static_cast<int>(geometry.global_cols / geometry.shared_cols)
                                  : static_cast<int>(geometry.global_cols / kBlackholeTileCols);
  const int shared_pages = static_cast<int>(geometry.shared_rows);
  const int cb_id = AllocateRequirementIndex(shared_buffer, CBType::kIntermediate);
  if (use_page_transport) {
    SetRequirementPageLayout(cb_id, geometry.page_bytes, shared_pages);
  }
  RecordStagedCopyBufferBinding(dram_to_cb, CopyDirection::kDramToCB);
  RecordStagedCopyBufferBinding(cb_to_dram, CopyDirection::kCBToDram);

  Analyzer analyzer;
  std::vector<Stmt> stmts;
  if (use_page_transport) {
    const int global_row_bytes =
        static_cast<int>(geometry.global_cols * dram_load->buffer->dtype.bytes());
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(shared_pages)}));
    for (int page_row = 0; page_row < geometry.shared_rows; ++page_row) {
      PrimExpr buffer_byte_offset =
          analyzer.Simplify(base_tile_index * IntImm32(geometry.page_bytes));
      if (page_row != 0) {
        buffer_byte_offset =
            analyzer.Simplify(buffer_byte_offset + IntImm32(page_row * global_row_bytes));
      }
      const int input_accessor_slot =
          GetReadAccessorSlot("fused_dataflow", dram_load->buffer, CopyDirection::kDramToCB);
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_page_to_cb(),
          {dram_load->buffer->data, buffer_byte_offset, IntImm32(cb_id),
           IntImm32(geometry.page_bytes), IntImm32(input_accessor_slot),
           IntImm32(page_row * geometry.l1_stick_stride)}));
      RegisterAccessor("fused_dataflow", dram_load->buffer, input_accessor_slot, 2, 0, 0, 2,
                       geometry.page_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_noc_async_read_barrier(), {}));
    }
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(shared_pages)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(shared_pages)}));
    for (int page_row = 0; page_row < geometry.shared_rows; ++page_row) {
      PrimExpr buffer_byte_offset =
          analyzer.Simplify(base_tile_index * IntImm32(geometry.page_bytes));
      if (page_row != 0) {
        buffer_byte_offset =
            analyzer.Simplify(buffer_byte_offset + IntImm32(page_row * global_row_bytes));
      }
      const int output_accessor_slot =
          GetWriteAccessorSlot("fused_dataflow", cb_to_dram->buffer, CopyDirection::kCBToDram);
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_page_from_cb(),
          {IntImm32(cb_id), cb_to_dram->buffer->data, buffer_byte_offset,
           IntImm32(geometry.page_bytes), IntImm32(output_accessor_slot),
           IntImm32(page_row * geometry.l1_stick_stride)}));
      RegisterAccessor("fused_dataflow", cb_to_dram->buffer, output_accessor_slot, 2, 0, 0, 2,
                       geometry.page_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_noc_async_write_barrier(), {}));
    }
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(shared_pages)}));
    return SeqStmt::Flatten(stmts);
  }
  for (int subtile_row = 0; subtile_row < geometry.subtile_rows; ++subtile_row) {
    for (int subtile_col = 0; subtile_col < geometry.subtile_cols; ++subtile_col) {
      PrimExpr tile_index = base_tile_index;
      if (subtile_row != 0) {
        tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_row * tiles_per_row));
      }
      if (subtile_col != 0) {
        tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_col));
      }
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      const int input_accessor_slot =
          GetReadAccessorSlot("fused_dataflow", dram_load->buffer, CopyDirection::kDramToCB);
      stmts.push_back(MakeBlackholeCall(
          use_page_transport ? blackhole_read_page_to_cb() : blackhole_read_tile_to_cb(),
          {dram_load->buffer->data, tile_index, IntImm32(cb_id),
           IntImm32(use_page_transport ? geometry.page_bytes : geometry.tile_bytes),
           IntImm32(input_accessor_slot)}));
      RegisterAccessor("fused_dataflow", dram_load->buffer, input_accessor_slot, 2, 0, 0, 2,
                       use_page_transport ? geometry.page_bytes : geometry.tile_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
      const int output_accessor_slot =
          GetWriteAccessorSlot("fused_dataflow", cb_to_dram->buffer, CopyDirection::kCBToDram);
      stmts.push_back(MakeBlackholeCall(
          use_page_transport ? blackhole_write_page_from_cb() : blackhole_write_tile_from_cb(),
          {IntImm32(cb_id), cb_to_dram->buffer->data, tile_index,
           IntImm32(use_page_transport ? geometry.page_bytes : geometry.tile_bytes),
           IntImm32(output_accessor_slot)}));
      RegisterAccessor("fused_dataflow", cb_to_dram->buffer, output_accessor_slot, 2, 0, 0, 2,
                       use_page_transport ? geometry.page_bytes : geometry.tile_bytes);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
    }
  }

  return SeqStmt::Flatten(stmts);
}

Stmt LowerBlackholeOps::GenerateClearSequence(const CallNode* op) {
  // Clear operation: tile_regs_acquire() to zero DST registers
  // In full implementation, would also zero-fill
  return MakeBlackholeCall(blackhole_tile_regs_acquire(), {});
}

namespace {

bool IsScalarLocalFragmentBuffer(const Buffer& buffer) {
  if (!IsUnsupportedResidualLocalScope(buffer)) {
    return false;
  }
  return buffer->shape.size() == 1 && tir::is_one(buffer->shape[0]);
}

bool IsRowScalarLocalFragmentBuffer(const Buffer& buffer) {
  if (!IsUnsupportedResidualLocalScope(buffer)) {
    return false;
  }
  return buffer->shape.size() == 1;
}

bool MatchScalarAccumulatorUpdate(const BufferStoreNode* store,
                                  const Buffer& accum_buffer,
                                  const Var& reduce_var,
                                  Buffer* src_buffer,
                                  std::string* kind) {
  auto same_buffer = [](const Buffer& lhs, const Buffer& rhs) { return SameBufferIdentity(lhs, rhs); };
  auto same_loop_index = [&](const PrimExpr& index) {
    if (index.same_as(reduce_var)) {
      return true;
    }
    const auto* index_var = index.as<VarNode>();
    return index_var &&
           (index_var == reduce_var.get() || index_var->name_hint == reduce_var->name_hint);
  };
  if (!store || !same_buffer(store->buffer, accum_buffer) || store->indices.size() != 1 ||
      !tir::is_zero(store->indices[0])) {
    return false;
  }

  auto try_match_binary = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                              const char* expected_kind) -> bool {
    const auto* accum_load = lhs.as<BufferLoadNode>();
    const auto* src_load = rhs.as<BufferLoadNode>();
    if (!accum_load || !src_load) {
      return false;
    }
    if (!same_buffer(accum_load->buffer, accum_buffer) || accum_load->indices.size() != 1 ||
        !tir::is_zero(accum_load->indices[0])) {
      return false;
    }
    if (same_buffer(src_load->buffer, accum_buffer) || src_load->indices.size() != 1 ||
        !same_loop_index(src_load->indices[0])) {
      return false;
    }
    *src_buffer = src_load->buffer;
    *kind = expected_kind;
    return true;
  };

  if (const auto* add = store->value.as<AddNode>()) {
    return try_match_binary(add->a, add->b, "sum") || try_match_binary(add->b, add->a, "sum");
  }
  if (const auto* max = store->value.as<MaxNode>()) {
    return try_match_binary(max->a, max->b, "max") || try_match_binary(max->b, max->a, "max");
  }
  return false;
}

bool MatchReductionFinalizeStore(const BufferStoreNode* store,
                                 const Buffer& tmp_buffer,
                                 Buffer* dst_buffer,
                                 std::string* kind) {
  auto same_buffer = [](const Buffer& lhs, const Buffer& rhs) { return SameBufferIdentity(lhs, rhs); };
  if (!store || store->buffer->shape.size() != 1 || store->indices.size() != 1 ||
      !tir::is_zero(store->indices[0]) || !IsScalarLocalFragmentBuffer(store->buffer)) {
    return false;
  }

  auto try_match_binary = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                              const char* expected_kind) -> bool {
    const auto* dst_load = lhs.as<BufferLoadNode>();
    const auto* tmp_load = rhs.as<BufferLoadNode>();
    if (!dst_load || !tmp_load) {
      return false;
    }
    if (!same_buffer(dst_load->buffer, store->buffer) || dst_load->indices.size() != 1 ||
        !tir::is_zero(dst_load->indices[0])) {
      return false;
    }
    if (!same_buffer(tmp_load->buffer, tmp_buffer) || tmp_load->indices.size() != 1 ||
        !tir::is_zero(tmp_load->indices[0])) {
      return false;
    }
    *dst_buffer = store->buffer;
    *kind = expected_kind;
    return true;
  };

  if (const auto* add = store->value.as<AddNode>()) {
    return try_match_binary(add->a, add->b, "sum") || try_match_binary(add->b, add->a, "sum");
  }
  if (const auto* max = store->value.as<MaxNode>()) {
    return try_match_binary(max->a, max->b, "max") || try_match_binary(max->b, max->a, "max");
  }
  return false;
}

std::vector<Stmt> FlattenSeqStmtBody(const Stmt& body) {
  if (const auto* seq = body.as<SeqStmtNode>()) {
    return std::vector<Stmt>(seq->seq.begin(), seq->seq.end());
  }
  return {body};
}

std::vector<Stmt> FlattenSingletonLoopBody(const Stmt& body) {
  if (const auto* loop = body.as<ForNode>()) {
    if (tir::is_one(loop->extent)) {
      return FlattenSeqStmtBody(loop->body);
    }
  }
  return FlattenSeqStmtBody(body);
}

const ForNode* AsUnwrappedFor(const Stmt& stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto* attr = current.as<AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto* decl = current.as<tir::DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<tir::AllocateNode>()) {
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
    if (const auto* decl = current.as<tir::DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<tir::AllocateNode>()) {
      current = alloc->body;
      continue;
    }
    break;
  }
  return current.as<BufferStoreNode>();
}

bool IsFloatImmValue(const PrimExpr& expr, double expected) {
  if (const auto* imm = expr.as<FloatImmNode>()) {
    return imm->value == expected;
  }
  return false;
}

bool IsInfinityExpr(const PrimExpr& expr) {
  const auto* call = expr.as<CallNode>();
  if (!call || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  Op op = Downcast<Op>(call->op);
  return op->name == "tir.infinity";
}

bool IsScalarLiteralValue(const PrimExpr& expr) {
  return expr.as<FloatImmNode>() || expr.as<IntImmNode>();
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

bool IsZeroValue(const PrimExpr& expr) {
  return tir::is_zero(expr) || IsFloatImmValue(expr, 0.0);
}

bool IsNegInfValue(const PrimExpr& expr) {
  if (const auto* mul = expr.as<MulNode>()) {
    return ((IsFloatImmValue(mul->a, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->a)) &&
            IsFloatImmValue(mul->b, -1.0)) ||
           ((IsFloatImmValue(mul->b, std::numeric_limits<double>::infinity()) ||
             IsInfinityExpr(mul->b)) &&
            IsFloatImmValue(mul->a, -1.0));
  }
  return IsFloatImmValue(expr, -std::numeric_limits<double>::infinity()) || IsInfinityExpr(expr);
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

bool HasResidualRowBroadcast(const Stmt& body) {
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

bool MatchSelfIndexedVectorLoad(const PrimExpr& expr,
                                const Buffer& dst_buffer,
                                const Var& loop_var) {
  const auto* load = expr.as<BufferLoadNode>();
  return load && SameBufferIdentity(load->buffer, dst_buffer) &&
         load->indices.size() == 1 && load->indices[0].same_as(loop_var);
}

bool MatchScalarFragmentLoad(const PrimExpr& expr, Buffer* scalar_buffer) {
  const auto* load = expr.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || !tir::is_zero(load->indices[0]) ||
      !IsScalarLocalFragmentBuffer(load->buffer)) {
    return false;
  }
  *scalar_buffer = load->buffer;
  return true;
}

bool MatchGroupedScalarFragmentLoad(const PrimExpr& expr,
                                    const Var& loop_var,
                                    Buffer* scalar_buffer,
                                    PrimExpr* row_width) {
  const auto* load = expr.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || !IsRowScalarLocalFragmentBuffer(load->buffer)) {
    return false;
  }
  if (const auto* floordiv = load->indices[0].as<FloorDivNode>()) {
    if (!floordiv->a.same_as(loop_var)) {
      return false;
    }
    const auto* width_imm = floordiv->b.as<IntImmNode>();
    if (!width_imm || width_imm->value <= 0) {
      return false;
    }
    *scalar_buffer = load->buffer;
    *row_width = floordiv->b;
    return true;
  }
  if (const auto* shift = load->indices[0].as<CallNode>()) {
    if (!shift->op->IsInstance<OpNode>()) {
      return false;
    }
    const Op shift_op = Downcast<Op>(shift->op);
    if (shift_op->name != "tir.shift_right" || shift->args.size() != 2 ||
        !shift->args[0].same_as(loop_var)) {
      return false;
    }
    const auto* shift_imm = shift->args[1].as<IntImmNode>();
    if (!shift_imm || shift_imm->value < 0 || shift_imm->value >= 31) {
      return false;
    }
    *scalar_buffer = load->buffer;
    *row_width = IntImm(DataType::Int(32), 1 << shift_imm->value);
    return true;
  }
  return false;
}

bool MatchScalarBufferLoadFrom(const PrimExpr& expr, const Buffer& buffer) {
  const auto* load = expr.as<BufferLoadNode>();
  return load && load->indices.size() == 1 && tir::is_zero(load->indices[0]) &&
         SameBufferIdentity(load->buffer, buffer);
}

bool MatchIndexedRowStateLoad(const PrimExpr& expr, const Buffer& buffer, const Var& loop_var) {
  const auto* load = expr.as<BufferLoadNode>();
  return load && load->indices.size() == 1 && load->indices[0].same_as(loop_var) &&
         SameBufferIdentity(load->buffer, buffer);
}

bool MatchExp2Call(const PrimExpr& expr, PrimExpr* arg) {
  const auto* call = expr.as<CallNode>();
  if (!call || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op op = Downcast<Op>(call->op);
  if (op->name == "tir.exp2") {
    if (call->args.size() != 1) {
      return false;
    }
    if (arg != nullptr) {
      *arg = call->args[0];
    }
    return true;
  }

  const bool is_extern_exp2 =
      (op.same_as(tir::builtin::call_pure_extern()) || op.same_as(tir::builtin::call_extern())) &&
      call->args.size() == 2;
  if (!is_extern_exp2) {
    return false;
  }

  const auto* callee = call->args[0].as<StringImmNode>();
  if (!callee) {
    return false;
  }

  const std::string extern_name = callee->value;
  if (extern_name != "exp2" && extern_name != "exp2f" && extern_name != "exp2l" &&
      extern_name != "__exp2f") {
    return false;
  }

  if (arg != nullptr) {
    *arg = call->args[1];
  }
  return true;
}

bool MatchScaledSelfIndexedVectorLoad(const PrimExpr& expr,
                                      const Buffer& dst_buffer,
                                      const Var& loop_var,
                                      PrimExpr* scale) {
  if (MatchSelfIndexedVectorLoad(expr, dst_buffer, loop_var)) {
    *scale = make_const(expr.dtype(), 1.0);
    return true;
  }
  const auto* mul = expr.as<MulNode>();
  if (!mul) {
    return false;
  }
  if (MatchSelfIndexedVectorLoad(mul->a, dst_buffer, loop_var) && IsScalarLiteralValue(mul->b) &&
      !ExprUsesVar(mul->b, loop_var)) {
    *scale = mul->b;
    return true;
  }
  if (MatchSelfIndexedVectorLoad(mul->b, dst_buffer, loop_var) && IsScalarLiteralValue(mul->a) &&
      !ExprUsesVar(mul->a, loop_var)) {
    *scale = mul->a;
    return true;
  }
  return false;
}

bool MatchScaledScalarFragmentLoad(const PrimExpr& expr, Buffer* scalar_buffer, PrimExpr* scale) {
  if (MatchScalarFragmentLoad(expr, scalar_buffer)) {
    *scale = make_const(expr.dtype(), 1.0);
    return true;
  }
  const auto* mul = expr.as<MulNode>();
  if (!mul) {
    return false;
  }
  if (MatchScalarFragmentLoad(mul->a, scalar_buffer) && IsScalarLiteralValue(mul->b)) {
    *scale = mul->b;
    return true;
  }
  if (MatchScalarFragmentLoad(mul->b, scalar_buffer) && IsScalarLiteralValue(mul->a)) {
    *scale = mul->a;
    return true;
  }
  return false;
}

bool MatchScaledGroupedScalarFragmentLoad(const PrimExpr& expr,
                                          const Var& loop_var,
                                          Buffer* scalar_buffer,
                                          PrimExpr* row_width,
                                          PrimExpr* scale) {
  if (MatchGroupedScalarFragmentLoad(expr, loop_var, scalar_buffer, row_width)) {
    *scale = make_const(expr.dtype(), 1.0);
    return true;
  }
  const auto* mul = expr.as<MulNode>();
  if (!mul) {
    return false;
  }
  PrimExpr local_row_width;
  if (MatchGroupedScalarFragmentLoad(mul->a, loop_var, scalar_buffer, &local_row_width) &&
      IsScalarLiteralValue(mul->b)) {
    *row_width = local_row_width;
    *scale = mul->b;
    return true;
  }
  if (MatchGroupedScalarFragmentLoad(mul->b, loop_var, scalar_buffer, &local_row_width) &&
      IsScalarLiteralValue(mul->a)) {
    *row_width = local_row_width;
    *scale = mul->a;
    return true;
  }
  return false;
}

}  // namespace

bool LowerBlackholeOps::MatchDirectRowReduction(const ForNode* op, RowReductionMatch* match) const {
  if (!op || !match || !tir::is_one(op->extent)) {
    return false;
  }
  std::vector<Stmt> stmts = FlattenSingletonLoopBody(op->body);
  if (stmts.size() != 2) {
    return false;
  }

  const auto* init_store = AsUnwrappedBufferStore(stmts[0]);
  const auto* reduce_loop = AsUnwrappedFor(stmts[1]);
  if (!init_store || !reduce_loop || !IsScalarLocalFragmentBuffer(init_store->buffer) ||
      init_store->indices.size() != 1 || !tir::is_zero(init_store->indices[0])) {
    return false;
  }
  Buffer src_buffer;
  std::string kind;
  if (!MatchScalarAccumulatorUpdate(AsUnwrappedBufferStore(reduce_loop->body), init_store->buffer,
                                    reduce_loop->loop_var, &src_buffer, &kind)) {
    return false;
  }

  if (kind == "sum" && !IsZeroValue(init_store->value)) {
    return false;
  }
  if (kind == "max" && !IsNegInfValue(init_store->value)) {
    return false;
  }

  match->src = src_buffer;
  match->dst = init_store->buffer;
  match->num_elements = reduce_loop->extent;
  match->row_width = reduce_loop->extent;
  match->kind = kind;
  match->grouped = false;
  match->clear = true;
  return true;
}

bool LowerBlackholeOps::MatchAllocatedRowReduction(const AllocateNode* op,
                                                   RowReductionMatch* match) const {
  if (!op || !match || op->extents.size() != 1 || !tir::is_one(op->extents[0])) {
    return false;
  }
  std::vector<Stmt> stmts = FlattenSingletonLoopBody(op->body);
  if (stmts.size() != 3) {
    return false;
  }

  const auto* init_store = AsUnwrappedBufferStore(stmts[0]);
  const auto* reduce_loop = AsUnwrappedFor(stmts[1]);
  const auto* finalize_store = AsUnwrappedBufferStore(stmts[2]);
  if (init_store) {
  }
  if (!init_store || !reduce_loop || !finalize_store || init_store->indices.size() != 1 ||
      !tir::is_zero(init_store->indices[0])) {
    return false;
  }
  if (BufferDataIdentity(init_store->buffer) != op->buffer_var.get()) {
    return false;
  }

  Buffer src_buffer;
  std::string reduction_kind;
  if (!MatchScalarAccumulatorUpdate(AsUnwrappedBufferStore(reduce_loop->body), init_store->buffer,
                                    reduce_loop->loop_var, &src_buffer, &reduction_kind)) {
    return false;
  }

  Buffer dst_buffer;
  std::string finalize_kind;
  if (!MatchReductionFinalizeStore(finalize_store, init_store->buffer, &dst_buffer, &finalize_kind) ||
      finalize_kind != reduction_kind) {
    return false;
  }

  match->src = src_buffer;
  match->dst = dst_buffer;
  match->num_elements = reduce_loop->extent;
  match->row_width = reduce_loop->extent;
  match->kind = reduction_kind;
  match->grouped = false;
  match->clear = false;
  return true;
}

bool LowerBlackholeOps::MatchGroupedRowReduction(const ForNode* op,
                                                 RowReductionMatch* match) const {
  if (!op || !match) {
    return false;
  }
  std::vector<Stmt> stmts = FlattenSingletonLoopBody(op->body);
  if (stmts.size() != 3 && stmts.size() != 4) {
    return false;
  }

  const auto* init_store = AsUnwrappedBufferStore(stmts[0]);
  const auto* reduce_loop = AsUnwrappedFor(stmts[1]);
  const auto* allreduce_store = AsUnwrappedBufferStore(stmts[2]);
  const auto* finalize_store = stmts.size() == 4 ? AsUnwrappedBufferStore(stmts[3]) : nullptr;
  if (!init_store || !reduce_loop || !allreduce_store ||
      !IsRowScalarLocalFragmentBuffer(init_store->buffer) || init_store->indices.size() != 1 ||
      !init_store->indices[0].same_as(op->loop_var)) {
    return false;
  }

  auto same_buffer = [](const Buffer& lhs, const Buffer& rhs) { return SameBufferIdentity(lhs, rhs); };

  Buffer src_buffer;
  std::string reduction_kind;
  auto try_match_reduce = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                              const char* expected_kind) -> bool {
    const auto* accum_load = lhs.as<BufferLoadNode>();
    const auto* src_load = rhs.as<BufferLoadNode>();
    if (!accum_load || !src_load || !same_buffer(accum_load->buffer, init_store->buffer) ||
        accum_load->indices.size() != 1 || !accum_load->indices[0].same_as(op->loop_var) ||
        src_load->indices.size() != 1) {
      return false;
    }
    Analyzer analyzer;
    PrimExpr expected_index = op->loop_var * reduce_loop->extent + reduce_loop->loop_var;
    if (!tir::is_zero(analyzer.Simplify(src_load->indices[0] - expected_index))) {
      return false;
    }
    src_buffer = src_load->buffer;
    reduction_kind = expected_kind;
    return true;
  };

  const auto* reduce_store = AsUnwrappedBufferStore(reduce_loop->body);
  if (!reduce_store || !same_buffer(reduce_store->buffer, init_store->buffer) ||
      reduce_store->indices.size() != 1 || !reduce_store->indices[0].same_as(op->loop_var)) {
    return false;
  }
  if (const auto* add = reduce_store->value.as<AddNode>()) {
    if (!(try_match_reduce(add->a, add->b, "sum") || try_match_reduce(add->b, add->a, "sum"))) {
      return false;
    }
  } else if (const auto* max = reduce_store->value.as<MaxNode>()) {
    if (!(try_match_reduce(max->a, max->b, "max") || try_match_reduce(max->b, max->a, "max"))) {
      return false;
    }
  } else {
    return false;
  }

  const auto* allreduce_call = allreduce_store->value.as<CallNode>();
  if (!same_buffer(allreduce_store->buffer, init_store->buffer) || allreduce_store->indices.size() != 1 ||
      !allreduce_store->indices[0].same_as(op->loop_var) || !allreduce_call ||
      !allreduce_call->op.same_as(tir::builtin::call_extern())) {
    return false;
  }

  Buffer dst_buffer = init_store->buffer;
  bool clear = true;
  if (finalize_store != nullptr) {
    auto try_match_finalize = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                                  const char* expected_kind) -> bool {
      const auto* dst_load = lhs.as<BufferLoadNode>();
      const auto* tmp_load = rhs.as<BufferLoadNode>();
      if (!dst_load || !tmp_load || dst_load->indices.size() != 1 || tmp_load->indices.size() != 1 ||
          !dst_load->indices[0].same_as(op->loop_var) || !tmp_load->indices[0].same_as(op->loop_var) ||
          !same_buffer(tmp_load->buffer, init_store->buffer)) {
        return false;
      }
      dst_buffer = finalize_store->buffer;
      return same_buffer(dst_load->buffer, finalize_store->buffer) && reduction_kind == expected_kind;
    };

    if (const auto* add = finalize_store->value.as<AddNode>()) {
      if (!(try_match_finalize(add->a, add->b, "sum") ||
            try_match_finalize(add->b, add->a, "sum"))) {
        return false;
      }
    } else if (const auto* max = finalize_store->value.as<MaxNode>()) {
      if (!(try_match_finalize(max->a, max->b, "max") ||
            try_match_finalize(max->b, max->a, "max"))) {
        return false;
      }
    } else {
      return false;
    }
    clear = false;
  }

  if (reduction_kind == "sum" && !IsZeroValue(init_store->value)) {
    return false;
  }
  if (reduction_kind == "max" && !IsNegInfValue(init_store->value)) {
    return false;
  }

  match->src = src_buffer;
  match->dst = dst_buffer;
  match->num_elements = op->extent;
  match->row_width = reduce_loop->extent;
  match->kind = reduction_kind;
  match->grouped = true;
  match->clear = clear;
  return true;
}

Stmt LowerBlackholeOps::GenerateRowReductionSequence(const RowReductionMatch& match) {
  RowReductionMatch lowered_match = match;
  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(match.src);
  const int64_t logical_dst_extent = GetLogicalVectorLength(match.dst);
  if (logical_rows > 0 && logical_cols > 0 && logical_dst_extent == logical_rows) {
    lowered_match.grouped = true;
    lowered_match.num_elements = IntImm(DataType::Int(32), logical_rows * logical_cols);
    lowered_match.row_width = IntImm(DataType::Int(32), logical_cols);
  }

  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("reduce_row", BufferIdentityName(lowered_match.dst));
  op_payload.Set("src_buffer", String(BufferIdentityName(lowered_match.src)));
  op_payload.Set("reduce_kind", String(lowered_match.kind));
  op_payload.Set("grouped", Bool(lowered_match.grouped));
  op_payload.Set("clear", Bool(lowered_match.clear));
  SetOptionalExprField(&op_payload, "num_elements_expr", lowered_match.num_elements);
  SetOptionalExprField(&op_payload, "row_width_expr", lowered_match.row_width);
  RecordComputeEpilogueOp(std::move(op_payload));
  if (lowered_match.grouped) {
    return MakeBlackholeCall(tir::builtin::blackhole_reduce_row(),
                             {lowered_match.src->data, lowered_match.dst->data,
                              lowered_match.num_elements, lowered_match.row_width,
                              StringImm(lowered_match.kind), Bool(lowered_match.clear)});
  }
  return MakeBlackholeCall(tir::builtin::blackhole_reduce_row(),
                           {lowered_match.src->data, lowered_match.dst->data,
                            lowered_match.num_elements, StringImm(lowered_match.kind),
                            Bool(lowered_match.clear)});
}

bool LowerBlackholeOps::MatchDirectRowBroadcast(const ForNode* op,
                                                RowBroadcastMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsVectorLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }

  auto fill_match = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                        const char* kind) -> bool {
    Buffer scalar;
    PrimExpr row_width;
    if (!MatchSelfIndexedVectorLoad(lhs, store->buffer, op->loop_var)) {
      return false;
    }
    match->dst = store->buffer;
    match->num_elements = op->extent;
    match->kind = kind;
    match->grouped = false;
    if (MatchScalarFragmentLoad(rhs, &scalar)) {
      match->scalar = scalar;
      match->row_width = op->extent;
      return true;
    }
    if (MatchGroupedScalarFragmentLoad(rhs, op->loop_var, &scalar, &row_width)) {
      match->scalar = scalar;
      match->row_width = row_width;
      match->grouped = true;
      return true;
    }
    return false;
  };

  if (const auto* mul = store->value.as<MulNode>()) {
    return fill_match(mul->a, mul->b, "mul") || fill_match(mul->b, mul->a, "mul");
  }
  if (const auto* div = store->value.as<DivNode>()) {
    return fill_match(div->a, div->b, "div");
  }
  return false;
}

Stmt LowerBlackholeOps::GenerateRowBroadcastSequence(const RowBroadcastMatch& match) {
  RowBroadcastMatch lowered_match = match;
  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(match.dst);
  const int64_t logical_scalar_extent = GetLogicalVectorLength(match.scalar);
  if (logical_rows > 0 && logical_cols > 0 && logical_scalar_extent == logical_rows) {
    lowered_match.grouped = true;
    lowered_match.num_elements = IntImm(DataType::Int(32), logical_rows * logical_cols);
    lowered_match.row_width = IntImm(DataType::Int(32), logical_cols);
  }

  const char* op_kind = nullptr;
  Op op;
  if (lowered_match.kind == "mul") {
    op = lowered_match.grouped ? tir::builtin::blackhole_mul_grouped_row_bcast()
                       : tir::builtin::blackhole_mul_row_bcast();
    op_kind = lowered_match.grouped ? "mul_grouped_row_bcast" : "mul_row_bcast";
  } else {
    op = lowered_match.grouped ? tir::builtin::blackhole_div_grouped_row_bcast()
                       : tir::builtin::blackhole_div_row_bcast();
    op_kind = lowered_match.grouped ? "div_grouped_row_bcast" : "div_row_bcast";
  }
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload(op_kind, BufferIdentityName(lowered_match.dst));
  op_payload.Set("scalar_buffer", String(BufferIdentityName(lowered_match.scalar)));
  op_payload.Set("grouped", Bool(lowered_match.grouped));
  SetOptionalExprField(&op_payload, "num_elements_expr", lowered_match.num_elements);
  SetOptionalExprField(&op_payload, "row_width_expr", lowered_match.row_width);
  RecordComputeEpilogueOp(std::move(op_payload));
  if (lowered_match.grouped) {
    return MakeBlackholeCall(
        op, {lowered_match.dst->data, lowered_match.scalar->data, lowered_match.num_elements,
             lowered_match.row_width});
  }
  return MakeBlackholeCall(
      op, {lowered_match.dst->data, lowered_match.scalar->data, lowered_match.num_elements});
}

bool LowerBlackholeOps::MatchScalarFmaStore(const BufferStoreNode* op,
                                            ScalarFmaMatch* match) const {
  if (!op || !match || !IsScalarLocalFragmentBuffer(op->buffer) || op->indices.size() != 1 ||
      !tir::is_zero(op->indices[0])) {
    return false;
  }
  const auto* add = op->value.as<AddNode>();
  if (!add) {
    return false;
  }

  auto try_match = [&](const PrimExpr& mul_expr, const PrimExpr& add_expr) -> bool {
    const auto* mul = mul_expr.as<MulNode>();
    Buffer mul_lhs;
    Buffer mul_rhs;
    Buffer add_buffer;
    if (!mul || !MatchScalarFragmentLoad(mul->a, &mul_lhs) ||
        !MatchScalarFragmentLoad(mul->b, &mul_rhs) ||
        !MatchScalarFragmentLoad(add_expr, &add_buffer)) {
      return false;
    }
    if (!MatchScalarBufferLoadFrom(mul->a, op->buffer) &&
        !MatchScalarBufferLoadFrom(mul->b, op->buffer)) {
      return false;
    }
    match->dst = op->buffer;
    match->lhs = mul_lhs;
    match->rhs = mul_rhs;
    match->add = add_buffer;
    match->num_elements = PrimExpr();
    return true;
  };

  return try_match(add->a, add->b) || try_match(add->b, add->a);
}

bool LowerBlackholeOps::MatchGroupedScalarFmaLoop(const ForNode* op,
                                                  ScalarFmaMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsRowScalarLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }
  const auto* add = store->value.as<AddNode>();
  if (!add) {
    return false;
  }

  auto try_match = [&](const PrimExpr& mul_expr, const PrimExpr& add_expr) -> bool {
    const auto* mul = mul_expr.as<MulNode>();
    Buffer mul_lhs;
    Buffer mul_rhs;
    Buffer add_buffer;
    if (!mul || !MatchIndexedRowStateLoad(mul->a, store->buffer, op->loop_var) &&
                    !MatchIndexedRowStateLoad(mul->b, store->buffer, op->loop_var)) {
      return false;
    }
    const PrimExpr& other_mul = MatchIndexedRowStateLoad(mul->a, store->buffer, op->loop_var) ? mul->b : mul->a;
    const PrimExpr& self_mul = MatchIndexedRowStateLoad(mul->a, store->buffer, op->loop_var) ? mul->a : mul->b;
    if (!MatchIndexedRowStateLoad(self_mul, store->buffer, op->loop_var)) {
      return false;
    }
    const auto* other_load = other_mul.as<BufferLoadNode>();
    const auto* add_load = add_expr.as<BufferLoadNode>();
    if (!other_load || !add_load || other_load->indices.size() != 1 || add_load->indices.size() != 1 ||
        !other_load->indices[0].same_as(op->loop_var) || !add_load->indices[0].same_as(op->loop_var) ||
        !IsRowScalarLocalFragmentBuffer(other_load->buffer) ||
        !IsRowScalarLocalFragmentBuffer(add_load->buffer)) {
      return false;
    }
    mul_lhs = store->buffer;
    mul_rhs = other_load->buffer;
    add_buffer = add_load->buffer;
    match->dst = store->buffer;
    match->lhs = mul_lhs;
    match->rhs = mul_rhs;
    match->add = add_buffer;
    match->num_elements = PrimExpr();
    return true;
  };

  return try_match(add->a, add->b) || try_match(add->b, add->a);
}

Stmt LowerBlackholeOps::GenerateScalarFmaSequence(const ScalarFmaMatch& match) {
  ScalarFmaMatch lowered_match = match;
  const int64_t logical_extent = GetLogicalVectorLength(match.dst);
  if (!lowered_match.num_elements.defined() && logical_extent > 1) {
    lowered_match.num_elements = IntImm(DataType::Int(32), logical_extent);
  }

  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("scalar_fma", BufferIdentityName(lowered_match.dst));
  op_payload.Set("lhs_buffer", String(BufferIdentityName(lowered_match.lhs)));
  op_payload.Set("rhs_buffer", String(BufferIdentityName(lowered_match.rhs)));
  op_payload.Set("add_buffer", String(BufferIdentityName(lowered_match.add)));
  SetOptionalExprField(&op_payload, "num_elements_expr", lowered_match.num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
  if (lowered_match.num_elements.defined()) {
    return MakeBlackholeCall(tir::builtin::blackhole_scalar_fma(),
                             {lowered_match.dst->data, lowered_match.lhs->data,
                              lowered_match.rhs->data, lowered_match.add->data,
                              lowered_match.num_elements});
  }
  return MakeBlackholeCall(tir::builtin::blackhole_scalar_fma(),
                           {lowered_match.dst->data, lowered_match.lhs->data,
                            lowered_match.rhs->data, lowered_match.add->data});
}

bool LowerBlackholeOps::MatchExp2RowBroadcastAffine(const ForNode* op,
                                                    Exp2RowBroadcastAffineMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsVectorLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }

  PrimExpr exp2_arg;
  if (!MatchExp2Call(store->value, &exp2_arg)) {
    return false;
  }
  const auto* sub = exp2_arg.as<SubNode>();
  if (!sub) {
    return false;
  }

  Buffer scalar;
  PrimExpr dst_scale;
  PrimExpr scalar_scale;
  PrimExpr row_width;
  if (!MatchScaledSelfIndexedVectorLoad(sub->a, store->buffer, op->loop_var, &dst_scale)) {
    return false;
  }

  match->dst = store->buffer;
  match->num_elements = op->extent;
  match->dst_scale = dst_scale;
  Analyzer analyzer;
  match->grouped = false;
  if (MatchScaledScalarFragmentLoad(sub->b, &scalar, &scalar_scale)) {
    match->scalar = scalar;
    match->row_width = op->extent;
    match->scalar_scale = analyzer.Simplify(-scalar_scale);
    return true;
  }
  if (MatchScaledGroupedScalarFragmentLoad(sub->b, op->loop_var, &scalar, &row_width,
                                           &scalar_scale)) {
    match->scalar = scalar;
    match->row_width = row_width;
    match->grouped = true;
    match->scalar_scale = analyzer.Simplify(-scalar_scale);
    return true;
  }
  return false;
}

Stmt LowerBlackholeOps::GenerateExp2RowBroadcastAffineSequence(
    const Exp2RowBroadcastAffineMatch& match) {
  Exp2RowBroadcastAffineMatch lowered_match = match;
  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(match.dst);
  const int64_t logical_scalar_extent = GetLogicalVectorLength(match.scalar);
  if (logical_rows > 0 && logical_cols > 0 && logical_scalar_extent == logical_rows) {
    lowered_match.grouped = true;
    lowered_match.num_elements = IntImm(DataType::Int(32), logical_rows * logical_cols);
    lowered_match.row_width = IntImm(DataType::Int(32), logical_cols);
  }

  Map<String, Any> op_payload = MakeComputeEpilogueOpPayload(
      lowered_match.grouped ? "exp2_grouped_row_bcast_affine" : "exp2_row_bcast_affine",
      BufferIdentityName(lowered_match.dst));
  op_payload.Set("scalar_buffer", String(BufferIdentityName(lowered_match.scalar)));
  op_payload.Set("grouped", Bool(lowered_match.grouped));
  SetOptionalExprField(&op_payload, "num_elements_expr", lowered_match.num_elements);
  SetOptionalExprField(&op_payload, "row_width_expr", lowered_match.row_width);
  SetOptionalExprField(&op_payload, "dst_scale_expr", lowered_match.dst_scale);
  SetOptionalExprField(&op_payload, "scalar_scale_expr", lowered_match.scalar_scale);
  RecordComputeEpilogueOp(std::move(op_payload));
  if (lowered_match.grouped) {
    return MakeBlackholeCall(tir::builtin::blackhole_exp2_grouped_row_bcast_affine(),
                             {lowered_match.dst->data, lowered_match.scalar->data,
                              lowered_match.num_elements, lowered_match.row_width,
                              lowered_match.dst_scale, lowered_match.scalar_scale});
  }
  return MakeBlackholeCall(tir::builtin::blackhole_exp2_row_bcast_affine(),
                           {lowered_match.dst->data, lowered_match.scalar->data,
                            lowered_match.num_elements, lowered_match.dst_scale,
                            lowered_match.scalar_scale});
}

bool LowerBlackholeOps::MatchScalarExp2AffineStore(const BufferStoreNode* op,
                                                   ScalarExp2AffineMatch* match) const {
  if (!op || !match || !IsScalarLocalFragmentBuffer(op->buffer) || op->indices.size() != 1 ||
      !tir::is_zero(op->indices[0])) {
    return false;
  }

  PrimExpr exp2_arg;
  if (!MatchExp2Call(op->value, &exp2_arg)) {
    return false;
  }
  const auto* sub = exp2_arg.as<SubNode>();
  if (!sub) {
    return false;
  }

  Buffer lhs;
  Buffer rhs;
  PrimExpr lhs_scale;
  PrimExpr rhs_scale;
  if (!MatchScaledScalarFragmentLoad(sub->a, &lhs, &lhs_scale) ||
      !MatchScaledScalarFragmentLoad(sub->b, &rhs, &rhs_scale)) {
    return false;
  }

  match->dst = op->buffer;
  match->lhs = lhs;
  match->rhs = rhs;
  match->lhs_scale = lhs_scale;
  Analyzer analyzer;
  match->rhs_scale = analyzer.Simplify(-rhs_scale);
  return true;
}

bool LowerBlackholeOps::MatchGroupedScalarExp2AffineLoop(const ForNode* op,
                                                         ScalarExp2AffineMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsRowScalarLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }

  PrimExpr exp2_arg;
  if (!MatchExp2Call(store->value, &exp2_arg)) {
    return false;
  }
  const auto* sub = exp2_arg.as<SubNode>();
  if (!sub) {
    return false;
  }

  auto match_scaled_row_state_load = [&](const PrimExpr& expr, Buffer* buffer, PrimExpr* scale) {
    if (MatchIndexedRowStateLoad(expr, store->buffer, op->loop_var)) {
      *buffer = store->buffer;
      *scale = make_const(expr.dtype(), 1.0);
      return true;
    }
    const auto* load = expr.as<BufferLoadNode>();
    if (load && load->indices.size() == 1 && load->indices[0].same_as(op->loop_var) &&
        IsRowScalarLocalFragmentBuffer(load->buffer)) {
      *buffer = load->buffer;
      *scale = make_const(expr.dtype(), 1.0);
      return true;
    }
    const auto* mul = expr.as<MulNode>();
    if (!mul || !IsScalarLiteralValue(mul->a) && !IsScalarLiteralValue(mul->b)) {
      return false;
    }
    const PrimExpr& maybe_load = IsScalarLiteralValue(mul->a) ? mul->b : mul->a;
    const PrimExpr& maybe_scale = IsScalarLiteralValue(mul->a) ? mul->a : mul->b;
    const auto* load_expr = maybe_load.as<BufferLoadNode>();
    if (!load_expr || load_expr->indices.size() != 1 ||
        !load_expr->indices[0].same_as(op->loop_var) ||
        !IsRowScalarLocalFragmentBuffer(load_expr->buffer)) {
      return false;
    }
    *buffer = load_expr->buffer;
    *scale = maybe_scale;
    return true;
  };

  Buffer lhs;
  Buffer rhs;
  PrimExpr lhs_scale;
  PrimExpr rhs_scale;
  if (!match_scaled_row_state_load(sub->a, &lhs, &lhs_scale) ||
      !match_scaled_row_state_load(sub->b, &rhs, &rhs_scale)) {
    return false;
  }

  Analyzer analyzer;
  match->dst = store->buffer;
  match->lhs = lhs;
  match->rhs = rhs;
  match->lhs_scale = lhs_scale;
  match->rhs_scale = analyzer.Simplify(-rhs_scale);
  return true;
}

Stmt LowerBlackholeOps::GenerateScalarExp2AffineSequence(const ScalarExp2AffineMatch& match) {
  const int64_t logical_extent = GetLogicalVectorLength(match.dst);
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("scalar_exp2_affine", BufferIdentityName(match.dst));
  op_payload.Set("lhs_buffer", String(BufferIdentityName(match.lhs)));
  op_payload.Set("rhs_buffer", String(BufferIdentityName(match.rhs)));
  SetOptionalExprField(&op_payload, "lhs_scale_expr", match.lhs_scale);
  SetOptionalExprField(&op_payload, "rhs_scale_expr", match.rhs_scale);
  if (logical_extent > 1) {
    SetOptionalExprField(&op_payload, "num_elements_expr",
                         IntImm(DataType::Int(32), logical_extent));
  }
  RecordComputeEpilogueOp(std::move(op_payload));
  if (logical_extent > 1) {
    return MakeBlackholeCall(tir::builtin::blackhole_scalar_exp2_affine(),
                             {match.dst->data, match.lhs->data, match.rhs->data,
                              match.lhs_scale, match.rhs_scale,
                              IntImm(DataType::Int(32), logical_extent)});
  }
  return MakeBlackholeCall(tir::builtin::blackhole_scalar_exp2_affine(),
                           {match.dst->data, match.lhs->data, match.rhs->data,
                            match.lhs_scale, match.rhs_scale});
}

bool LowerBlackholeOps::MatchDirectFragmentFill(const ForNode* op,
                                                FragmentFillMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsVectorLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var) || !IsFragmentFillValue(store->value)) {
    const auto* inner_loop = AsUnwrappedFor(op->body);
    const auto* inner_store = inner_loop ? AsUnwrappedBufferStore(inner_loop->body) : nullptr;
    if (!inner_loop || !inner_store || !IsVectorLocalFragmentBuffer(inner_store->buffer) ||
        inner_store->indices.size() != 1 || !IsFragmentFillValue(inner_store->value)) {
      return false;
    }
    Analyzer analyzer;
    auto match_linearized_index = [&](const PrimExpr& expr) -> bool {
      PrimExpr expected = op->loop_var * inner_loop->extent + inner_loop->loop_var;
      return tir::is_zero(analyzer.Simplify(expr - expected));
    };
    if (!match_linearized_index(inner_store->indices[0])) {
      return false;
    }
    match->dst = inner_store->buffer;
    match->num_elements = analyzer.Simplify(op->extent * inner_loop->extent);
    match->value = inner_store->value;
    return true;
  }
  match->dst = store->buffer;
  match->num_elements = op->extent;
  match->value = store->value;
  return true;
}

bool LowerBlackholeOps::MatchScalarFragmentFillStore(const BufferStoreNode* op,
                                                     FragmentFillMatch* match) const {
  if (!op || !match || !IsScalarLocalFragmentBuffer(op->buffer) || op->indices.size() != 1 ||
      !tir::is_zero(op->indices[0]) || !IsFragmentFillValue(op->value)) {
    return false;
  }
  match->dst = op->buffer;
  match->num_elements = IntImm(DataType::Int(32), 1);
  match->value = op->value;
  return true;
}

Stmt LowerBlackholeOps::GenerateFragmentFillSequence(const FragmentFillMatch& match) {
  PrimExpr num_elements = match.num_elements;
  const int64_t logical_extent = GetLogicalBufferElementCount(match.dst);
  if (logical_extent > 1) {
    bool should_promote_extent = !num_elements.defined();
    if (const auto* int_imm = num_elements.as<IntImmNode>()) {
      should_promote_extent = int_imm->value < logical_extent;
    }
    if (should_promote_extent) {
      num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }
  return MakeBlackholeCall(tir::builtin::blackhole_fill_fragment(),
                           {match.dst->data, num_elements, match.value});
}

bool LowerBlackholeOps::MatchScalarMaxStore(const BufferStoreNode* op,
                                            ScalarMaxMatch* match) const {
  if (!op || !match || !IsScalarLocalFragmentBuffer(op->buffer) || op->indices.size() != 1 ||
      !tir::is_zero(op->indices[0])) {
    return false;
  }
  const auto* max = op->value.as<MaxNode>();
  if (!max) {
    return false;
  }

  auto try_match = [&](const PrimExpr& self_expr, const PrimExpr& other_expr) -> bool {
    Buffer other;
    if (!MatchScalarBufferLoadFrom(self_expr, op->buffer) ||
        !MatchScalarFragmentLoad(other_expr, &other)) {
      return false;
    }
    match->dst = op->buffer;
    match->src = other;
    match->num_elements = PrimExpr();
    return true;
  };

  return try_match(max->a, max->b) || try_match(max->b, max->a);
}

bool LowerBlackholeOps::MatchGroupedScalarMaxLoop(const ForNode* op,
                                                  ScalarMaxMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsRowScalarLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }
  const auto* max = store->value.as<MaxNode>();
  if (!max) {
    return false;
  }

  auto try_match = [&](const PrimExpr& self_expr, const PrimExpr& other_expr) -> bool {
    const auto* other_load = other_expr.as<BufferLoadNode>();
    if (!MatchIndexedRowStateLoad(self_expr, store->buffer, op->loop_var) || !other_load ||
        other_load->indices.size() != 1 || !other_load->indices[0].same_as(op->loop_var) ||
        !IsRowScalarLocalFragmentBuffer(other_load->buffer)) {
      return false;
    }
    match->dst = store->buffer;
    match->src = other_load->buffer;
    match->num_elements = PrimExpr();
    return true;
  };

  return try_match(max->a, max->b) || try_match(max->b, max->a);
}

Stmt LowerBlackholeOps::GenerateScalarMaxSequence(const ScalarMaxMatch& match) {
  ScalarMaxMatch lowered_match = match;
  const int64_t logical_extent = GetLogicalVectorLength(match.dst);
  if (!lowered_match.num_elements.defined() && logical_extent > 1) {
    lowered_match.num_elements = IntImm(DataType::Int(32), logical_extent);
  }

  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("scalar_max", BufferIdentityName(lowered_match.dst));
  op_payload.Set("src_buffer", String(BufferIdentityName(lowered_match.src)));
  SetOptionalExprField(&op_payload, "num_elements_expr", lowered_match.num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
  if (lowered_match.num_elements.defined()) {
    return MakeBlackholeCall(tir::builtin::blackhole_scalar_max(),
                             {lowered_match.dst->data, lowered_match.src->data,
                              lowered_match.num_elements});
  }
  return MakeBlackholeCall(tir::builtin::blackhole_scalar_max(),
                           {lowered_match.dst->data, lowered_match.src->data});
}

bool LowerBlackholeOps::MatchScalarFragmentCopyStore(const BufferStoreNode* op,
                                                     ScalarFragmentCopyMatch* match) const {
  if (!op || !match || !IsScalarLocalFragmentBuffer(op->buffer) || op->indices.size() != 1 ||
      !tir::is_zero(op->indices[0])) {
    return false;
  }
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || !tir::is_zero(load->indices[0]) ||
      !IsScalarLocalFragmentBuffer(load->buffer) || SameBufferIdentity(op->buffer, load->buffer) ||
      op->buffer->dtype != load->buffer->dtype) {
    return false;
  }
  const int64_t logical_extent = GetLogicalVectorLength(op->buffer);
  if (logical_extent <= 1) {
    return false;
  }
  match->dst = op->buffer;
  match->src = load->buffer;
  match->num_elements = IntImm(DataType::Int(32), logical_extent);
  return true;
}

bool LowerBlackholeOps::MatchGroupedScalarFragmentCopyLoop(const ForNode* op,
                                                           ScalarFragmentCopyMatch* match) const {
  if (!op || !match) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  if (!store || !IsRowScalarLocalFragmentBuffer(store->buffer) || store->indices.size() != 1 ||
      !store->indices[0].same_as(op->loop_var)) {
    return false;
  }
  const auto* load = store->value.as<BufferLoadNode>();
  if (!load || load->indices.size() != 1 || !load->indices[0].same_as(op->loop_var) ||
      !IsRowScalarLocalFragmentBuffer(load->buffer) ||
      SameBufferIdentity(store->buffer, load->buffer) ||
      store->buffer->dtype != load->buffer->dtype) {
    return false;
  }
  const int64_t logical_extent = GetLogicalVectorLength(store->buffer);
  if (logical_extent <= 1) {
    return false;
  }
  match->dst = store->buffer;
  match->src = load->buffer;
  match->num_elements = IntImm(DataType::Int(32), logical_extent);
  return true;
}

Stmt LowerBlackholeOps::GenerateScalarFragmentCopySequence(const ScalarFragmentCopyMatch& match) {
  FragmentCastMatch cast_match;
  cast_match.dst = match.dst;
  cast_match.src = match.src;
  cast_match.dst_offset = IntImm(DataType::Int(32), 0);
  cast_match.src_offset = IntImm(DataType::Int(32), 0);
  cast_match.num_elements = match.num_elements;
  if (!cast_match.num_elements.defined()) {
    const int64_t logical_extent = GetLogicalVectorLength(match.dst);
    if (logical_extent > 1) {
      cast_match.num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }
  return GenerateFragmentCastSequence(cast_match, /*publish_cb=*/false);
}

bool LowerBlackholeOps::MatchDirectFragmentCast(const ForNode* op,
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
  const auto [logical_src_rows, logical_src_cols] = GetLogicalMatrixShape(load->buffer);
  const auto [logical_dst_rows, logical_dst_cols] = GetLogicalMatrixShape(inner_store->buffer);
  const int64_t logical_dst_extent = GetLogicalVectorLength(inner_store->buffer);
  const int64_t logical_src_extent = logical_src_rows > 0 && logical_src_cols > 0
                                         ? logical_src_rows * logical_src_cols
                                         : -1;
  const int64_t logical_matrix_cast_extent =
      logical_dst_rows > 0 && logical_dst_cols > 0 ? logical_dst_rows * logical_dst_cols : -1;
  if (logical_src_extent > 0 && logical_matrix_cast_extent > 0 &&
      logical_src_extent == logical_matrix_cast_extent && logical_src_extent > logical_dst_extent &&
      tir::is_zero(match->src_offset) && tir::is_zero(match->dst_offset)) {
    match->num_elements = IntImm(DataType::Int(32), logical_src_extent);
    return true;
  }
  if (logical_src_rows > 0 && logical_src_cols > 0 && logical_dst_extent > 0) {
    Var thread_row_var = SelectLogicalRowThreadVar(logical_src_rows);
    if (thread_row_var.defined() && !ExprUsesVar(match->src_offset, thread_row_var)) {
      match->src_offset = analyzer.Simplify(thread_row_var * IntImm(DataType::Int(32), logical_src_cols) +
                                            match->src_offset);
    }
  }
  return true;
}

Stmt LowerBlackholeOps::GenerateFragmentCastSequence(const FragmentCastMatch& match,
                                                    bool publish_cb) {
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("cast_fragment_slice", BufferIdentityName(match.dst));
  op_payload.Set("src_buffer", String(BufferIdentityName(match.src)));
  op_payload.Set("publish_cb", Bool(publish_cb));
  SetOptionalExprField(&op_payload, "dst_offset_expr", match.dst_offset);
  SetOptionalExprField(&op_payload, "src_offset_expr", match.src_offset);
  SetOptionalExprField(&op_payload, "num_elements_expr", match.num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
  std::vector<Stmt> stmts;
  stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cast_fragment_slice(),
                                    {match.dst->data, match.src->data, match.dst_offset,
                                     match.src_offset, match.num_elements}));

  if (publish_cb) {
    const int cb_id = AllocateRequirementIndex(match.dst, CBType::kIntermediate);
    ICHECK_GE(cb_id, 0);
    ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
    int num_pages = std::max(1, cb_requirements_[cb_id].num_pages);
    const std::string buffer_identity = BufferIdentityName(match.dst);
    auto it = cb_consumed_fragment_pages_by_buffer_identity_.find(buffer_identity);
    if (it != cb_consumed_fragment_pages_by_buffer_identity_.end()) {
      num_pages = std::max(1, it->second);
    }
    stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_cb_push_back(),
                                      {IntImm32(cb_id), IntImm32(num_pages)}));
  }

  return stmts.size() == 1 ? stmts.front() : SeqStmt::Flatten(stmts);
}

bool LowerBlackholeOps::ShouldRetainComputeInputBuffer(const Buffer& buffer,
                                                       int current_order_index) const {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = buffer_flow_contracts_.find(buffer_identity);
  if (it == buffer_flow_contracts_.end()) {
    return false;
  }
  for (const BufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    if (event.kind == BufferFlowEventKind::kComputeConsume) {
      return true;
    }
    if (event.kind == BufferFlowEventKind::kWrite) {
      return false;
    }
  }
  return false;
}

bool LowerBlackholeOps::ShouldReacquireComputeInputBuffer(const Buffer& buffer,
                                                          int current_order_index) const {
  if (GetStorageScope(buffer) != "blackhole.acc") {
    return false;
  }
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = buffer_flow_contracts_.find(buffer_identity);
  if (it == buffer_flow_contracts_.end()) {
    return false;
  }
  for (const BufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    return event.kind == BufferFlowEventKind::kWrite;
  }
  return false;
}

bool LowerBlackholeOps::ShouldPublishBufferResult(const Buffer& buffer,
                                                  int current_order_index) const {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = buffer_flow_contracts_.find(buffer_identity);
  if (it == buffer_flow_contracts_.end()) {
    return false;
  }
  for (const BufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    if (event.kind == BufferFlowEventKind::kComputeConsume ||
        event.kind == BufferFlowEventKind::kTransportConsume) {
      return true;
    }
    if (event.kind == BufferFlowEventKind::kWrite) {
      return false;
    }
  }
  return false;
}

bool LowerBlackholeOps::MatchDirectLocalToCBSliceLoop(const ForNode* op,
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
    match->dst_offset_elements = base_offset;
    match->num_elements = vector_extent;
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

Stmt LowerBlackholeOps::GenerateLocalToCBSliceLoopSequence(const ForNode* op,
                                                           const LocalToCBSliceMatch& match) {
  Map<String, Any> op_payload =
      MakeComputeEpilogueOpPayload("write_local_slice_to_cb", BufferIdentityName(match.dst));
  op_payload.Set("src_buffer", String(BufferIdentityName(match.src)));
  SetOptionalExprField(&op_payload, "dst_offset_expr", match.dst_offset_elements);
  SetOptionalExprField(&op_payload, "num_elements_expr", match.num_elements);
  RecordComputeEpilogueOp(std::move(op_payload));
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
  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(cb_id), IntImm32(num_pages)}));
  std::vector<Stmt> loop_stmts;
  if (match.lowered_loop_body.defined()) {
    Stmt lowered_prefix = VisitStmt(match.lowered_loop_body);
    lowered_prefix =
        LocalSliceCastSourceOffsetRewriter(match.src->data, match.dst_offset_elements)(lowered_prefix);
    loop_stmts.push_back(lowered_prefix);
  }
  loop_stmts.push_back(MakeBlackholeCall(blackhole_write_local_slice_to_cb(),
                                         {match.src->data, IntImm32(cb_id),
                                          match.dst_offset_elements, match.num_elements}));
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
  return SeqStmt::Flatten(stmts);
}

// Parse a colon-separated string into fields
Stmt LowerBlackholeOps::VisitStmt_(const AttrStmtNode* op) {
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

Stmt LowerBlackholeOps::VisitStmt_(const DeclBufferNode* op) {
  if (GetStorageScope(op->buffer) == "blackhole.acc") {
    const int requirement_index = AllocateRequirementIndex(op->buffer, CBType::kIntermediate);
    auto& req = cb_requirements_.at(requirement_index);
    req.lifetime_begin = 0;
    req.lifetime_end = std::max(req.lifetime_end, next_requirement_index_);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt LowerBlackholeOps::VisitStmt_(const AllocateNode* op) {
  RowReductionMatch pre_lower_match;
  if (MatchAllocatedRowReduction(op, &pre_lower_match)) {
    return GenerateRowReductionSequence(pre_lower_match);
  }
  Stmt lowered = StmtExprMutator::VisitStmt_(op);
  if (const auto* allocate = lowered.as<AllocateNode>()) {
    RowReductionMatch match;
    if (MatchAllocatedRowReduction(allocate, &match)) {
      return GenerateRowReductionSequence(match);
    }
  }
  return lowered;
}

Stmt LowerBlackholeOps::VisitStmt_(const SeqStmtNode* op) {
  Array<Stmt> rewritten;
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const auto order_it = stmt_order_index_by_node_.find(op->seq[i].get());
    const int current_order_index =
        order_it != stmt_order_index_by_node_.end() ? order_it->second : static_cast<int>(i);
    auto try_lower_retained_matmul = [&](const Stmt& stmt, Stmt* lowered) -> bool {
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
          ExtractGemmInfo(call);
          bool retain_in0 = false;
          bool retain_in1 = false;
          bool reacquire_in0 = false;
          bool reacquire_in1 = false;
          if (IsBufferLikeExpr(call->args[0])) {
            const Buffer in0_buffer = NormalizeToBufferRegion(call->args[0])->buffer;
            retain_in0 = ShouldRetainComputeInputBuffer(in0_buffer, current_order_index);
            if (!retain_in0) {
              reacquire_in0 =
                  ShouldReacquireComputeInputBuffer(in0_buffer, current_order_index);
            }
          }
          if (IsBufferLikeExpr(call->args[1])) {
            const Buffer in1_buffer = NormalizeToBufferRegion(call->args[1])->buffer;
            retain_in1 = ShouldRetainComputeInputBuffer(in1_buffer, current_order_index);
            if (!retain_in1) {
              reacquire_in1 =
                  ShouldReacquireComputeInputBuffer(in1_buffer, current_order_index);
            }
          }
          bool publish_out = true;
          if (IsBufferLikeExpr(call->args[2])) {
            publish_out = ShouldPublishBufferResult(
                NormalizeToBufferRegion(call->args[2])->buffer, current_order_index);
          }
          Stmt matmul = GenerateMatmulSequence(call, retain_in0, retain_in1, publish_out,
                                               reacquire_in0, reacquire_in1);
          for (auto it = rewrap_stack.rbegin(); it != rewrap_stack.rend(); ++it) {
            matmul = (*it)(matmul);
          }
          *lowered = matmul;
          return true;
        }
      }
      return false;
    };

    Stmt retained_matmul;
    if (try_lower_retained_matmul(op->seq[i], &retained_matmul)) {
      rewritten.push_back(retained_matmul);
      continue;
    }
    if (const auto* direct_cast_loop = AsUnwrappedFor(op->seq[i])) {
      FragmentCastMatch cast_match;
      if (MatchDirectFragmentCast(direct_cast_loop, &cast_match)) {
        const bool publish_cb = ShouldPublishBufferResult(cast_match.dst, current_order_index);
        rewritten.push_back(GenerateFragmentCastSequence(cast_match, publish_cb));
        continue;
      }
    }
    if (i + 1 < op->seq.size()) {
      const Stmt& init_stmt = op->seq[i];
      const Stmt& reduce_stmt = op->seq[i + 1];
      const auto* init_store = AsUnwrappedBufferStore(init_stmt);
      const auto* reduce_loop = AsUnwrappedFor(reduce_stmt);
      if (init_store && reduce_loop && IsScalarLocalFragmentBuffer(init_store->buffer) &&
          init_store->indices.size() == 1 && tir::is_zero(init_store->indices[0])) {
        Buffer src_buffer;
        std::string kind;
        if (MatchScalarAccumulatorUpdate(AsUnwrappedBufferStore(reduce_loop->body),
                                         init_store->buffer,
                                         reduce_loop->loop_var,
                                         &src_buffer,
                                         &kind) &&
            ((kind == "sum" && IsZeroValue(init_store->value)) ||
             (kind == "max" && IsNegInfValue(init_store->value)))) {
          RowReductionMatch match;
          match.src = src_buffer;
          match.dst = init_store->buffer;
          match.num_elements = reduce_loop->extent;
          match.kind = kind;
          match.clear = true;
          rewritten.push_back(GenerateRowReductionSequence(match));
          ++i;
          continue;
        }
      }
    }
    rewritten.push_back(VisitStmt(op->seq[i]));
  }
  return SeqStmt::Flatten(rewritten);
}

Stmt LowerBlackholeOps::VisitStmt_(const ForNode* op) {
  const bool zero_loop_var = !op->thread_binding.defined();
  const Var transport_loop_var = zero_loop_var ? op->loop_var : Var();
  if (auto ann = op->annotations.Get(String("blackhole.copy_semantics"))) {
    Map<String, Any> sem = ann->as<Map<String, Any>>().value_or(Map<String, Any>());

    auto get_string = [&sem](const char* key) -> std::string {
      auto opt = sem.Get(String(key));
      if (!opt.has_value()) {
        return "";
      }
      auto str_opt = opt.value().try_cast<String>();
      return str_opt.has_value() ? std::string(str_opt.value()) : "";
    };
    auto get_shape = [&sem](const char* key) -> Array<Integer> {
      auto opt = sem.Get(String(key));
      if (!opt.has_value()) {
        return {};
      }
      return opt.value().try_cast<Array<Integer>>().value_or(Array<Integer>{});
    };
    auto get_buffer = [&sem](const char* key) -> Buffer {
      auto opt = sem.Get(String(key));
      if (!opt.has_value()) {
        return Buffer();
      }
      return opt.value().try_cast<Buffer>().value_or(Buffer());
    };

    const std::string kind = get_string(schema_key::kKind);
    const std::string direction = get_string(schema_key::kDirection);

    if (kind == "fused_staged_copy") {
      copy_input_buffer_ = get_buffer(schema_key::kSrcBufferRef);
      copy_output_buffer_ = get_buffer(schema_key::kDstBufferRef);
      copy_input_buffer_name_ = copy_input_buffer_.defined()
                                    ? BufferIdentityName(copy_input_buffer_)
                                    : get_string(schema_key::kSrcBuffer);
      copy_output_buffer_name_ = copy_output_buffer_.defined()
                                     ? BufferIdentityName(copy_output_buffer_)
                                     : get_string(schema_key::kDstBuffer);
      copy_input_shape_ = get_shape(schema_key::kSrcShape);
      copy_output_shape_ = get_shape(schema_key::kDstShape);
      copy_intermediate_shape_ = get_shape(schema_key::kMidShape);
    } else if (direction == "dram_to_cb") {
      copy_input_buffer_ = get_buffer(schema_key::kSrcBufferRef);
      copy_input_buffer_name_ = copy_input_buffer_.defined()
                                    ? BufferIdentityName(copy_input_buffer_)
                                    : get_string(schema_key::kSrcBuffer);
      copy_input_shape_ = get_shape(schema_key::kSrcShape);
      copy_intermediate_shape_ = get_shape(schema_key::kMidShape);
    } else if (direction == "cb_to_dram") {
      copy_output_buffer_ = get_buffer(schema_key::kDstBufferRef);
      copy_output_buffer_name_ = copy_output_buffer_.defined()
                                     ? BufferIdentityName(copy_output_buffer_)
                                     : get_string(schema_key::kDstBuffer);
      copy_output_shape_ = get_shape(schema_key::kDstShape);
      copy_intermediate_shape_ = get_shape(schema_key::kMidShape);
    } else if (direction == "dram_to_dram") {
      copy_input_buffer_ = get_buffer(schema_key::kSrcBufferRef);
      copy_output_buffer_ = get_buffer(schema_key::kDstBufferRef);
      copy_input_buffer_name_ = copy_input_buffer_.defined()
                                    ? BufferIdentityName(copy_input_buffer_)
                                    : get_string(schema_key::kSrcBuffer);
      copy_output_buffer_name_ = copy_output_buffer_.defined()
                                     ? BufferIdentityName(copy_output_buffer_)
                                     : get_string(schema_key::kDstBuffer);
      copy_input_shape_ = get_shape(schema_key::kSrcShape);
      copy_output_shape_ = get_shape(schema_key::kDstShape);
    }

    needs_copy_runtime_args_ = true;
    saw_copy_op_ = true;
  }

  if (IsPureCopyLoopNest(op->body)) {
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
                                               base_tile_index);
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
        lowered_matches.push_back(GenerateStagedCopyLoopSequence(match.store, base_tile_index));
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
        return GenerateStagedCopyLoopSequence(nested_store, base_tile_index);
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
  RowReductionMatch direct_row_reduction_match;
  if (MatchDirectRowReduction(op, &direct_row_reduction_match)) {
    return GenerateRowReductionSequence(direct_row_reduction_match);
  }
  RowReductionMatch grouped_row_reduction_match;
  if (MatchGroupedRowReduction(op, &grouped_row_reduction_match)) {
    return GenerateRowReductionSequence(grouped_row_reduction_match);
  }
  FragmentFillMatch direct_fill_match;
  if (MatchDirectFragmentFill(op, &direct_fill_match)) {
    return GenerateFragmentFillSequence(direct_fill_match);
  }
  ScalarFragmentCopyMatch grouped_scalar_copy_match;
  if (MatchGroupedScalarFragmentCopyLoop(op, &grouped_scalar_copy_match)) {
    return GenerateScalarFragmentCopySequence(grouped_scalar_copy_match);
  }
  ScalarMaxMatch grouped_scalar_max_match;
  if (MatchGroupedScalarMaxLoop(op, &grouped_scalar_max_match)) {
    return GenerateScalarMaxSequence(grouped_scalar_max_match);
  }
  ScalarExp2AffineMatch grouped_scalar_exp2_affine_match;
  if (MatchGroupedScalarExp2AffineLoop(op, &grouped_scalar_exp2_affine_match)) {
    return GenerateScalarExp2AffineSequence(grouped_scalar_exp2_affine_match);
  }
  ScalarFmaMatch grouped_scalar_fma_match;
  if (MatchGroupedScalarFmaLoop(op, &grouped_scalar_fma_match)) {
    return GenerateScalarFmaSequence(grouped_scalar_fma_match);
  }
  FragmentCastMatch direct_cast_match;
  if (MatchDirectFragmentCast(op, &direct_cast_match)) {
    return GenerateFragmentCastSequence(direct_cast_match);
  }
  LocalToCBSliceMatch local_to_cb_match;
  if (MatchDirectLocalToCBSliceLoop(op, &local_to_cb_match)) {
    saw_copy_op_ = true;
    return GenerateLocalToCBSliceLoopSequence(op, local_to_cb_match);
  }
  Exp2RowBroadcastAffineMatch direct_exp2_row_broadcast_match;
  if (MatchExp2RowBroadcastAffine(op, &direct_exp2_row_broadcast_match)) {
    return GenerateExp2RowBroadcastAffineSequence(direct_exp2_row_broadcast_match);
  }
  RowBroadcastMatch direct_row_broadcast_match;
  if (MatchDirectRowBroadcast(op, &direct_row_broadcast_match)) {
    return GenerateRowBroadcastSequence(direct_row_broadcast_match);
  }
  Stmt lowered = StmtExprMutator::VisitStmt_(op);
  return lowered;
}

// StmtExprMutator overrides
// Note: We only override specific node types and return the original node
// for unmatched patterns to avoid deep recursion that causes stack overflow.
Stmt LowerBlackholeOps::VisitStmt_(const EvaluateNode* op) {
  if (const auto* call = op->value.as<CallNode>()) {
    if (IsMatmulCall(call)) {
      ExtractGemmInfo(call);
      return GenerateMatmulSequence(call);
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

Stmt LowerBlackholeOps::VisitStmt_(const BufferStoreNode* op) {
  FragmentFillMatch fill_match;
  if (MatchScalarFragmentFillStore(op, &fill_match)) {
    return GenerateFragmentFillSequence(fill_match);
  }
  ScalarFragmentCopyMatch scalar_copy_match;
  if (MatchScalarFragmentCopyStore(op, &scalar_copy_match)) {
    return GenerateScalarFragmentCopySequence(scalar_copy_match);
  }
  ScalarMaxMatch scalar_max_match;
  if (MatchScalarMaxStore(op, &scalar_max_match)) {
    return GenerateScalarMaxSequence(scalar_max_match);
  }
  ScalarExp2AffineMatch scalar_exp2_affine_match;
  if (MatchScalarExp2AffineStore(op, &scalar_exp2_affine_match)) {
    return GenerateScalarExp2AffineSequence(scalar_exp2_affine_match);
  }
  ScalarFmaMatch scalar_fma_match;
  if (MatchScalarFmaStore(op, &scalar_fma_match)) {
    return GenerateScalarFmaSequence(scalar_fma_match);
  }
  if (IsCopyOperation(op)) {
    saw_copy_op_ = true;
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

}  // namespace tl
}  // namespace tvm
