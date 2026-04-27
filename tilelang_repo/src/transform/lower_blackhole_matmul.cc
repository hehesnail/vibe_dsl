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
 * \file lower_blackhole_matmul.cc
 * \brief GEMM and accumulator merge lowering for Blackhole TT-Metal builtins.
 */

#include "lower_blackhole_ops.h"

#include "../op/utils.h"
#include "common/blackhole_utils.h"

#include <tvm/tir/op.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::Buffer;
using tir::Call;
using tir::CallNode;
using tir::Evaluate;
using tir::SeqStmt;
using tir::Stmt;
using tir::VarNode;
using tir::builtin::blackhole_add_fragment;
using tir::builtin::blackhole_add_fragment_from_cb_front;
using tir::builtin::blackhole_add_tiles;
using tir::builtin::blackhole_add_tiles_init;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_copy_tile_to_dst_init_short;
using tir::builtin::blackhole_copy_tile_to_dst_init_short_with_dt;
using tir::builtin::blackhole_matmul_tiles;
using tir::builtin::blackhole_mm_init;
using tir::builtin::blackhole_mm_init_short;
using tir::builtin::blackhole_mm_init_short_with_dt;
using tir::builtin::blackhole_pack_reconfig_data_format;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_pack_untilize_tile;
using tir::builtin::blackhole_reconfig_data_format;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_untilize_cb_front_tile_fragment;
using tvm::DataType;
using tvm::IntImm;

namespace {

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

Stmt MakeBlackholeCall(const Op& op, const std::vector<PrimExpr>& args) {
  return Evaluate(Call(DataType::Int(32), op, args));
}

PrimExpr IntImm32(int value) {
  return IntImm(DataType::Int(32), value);
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

int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  if (value <= 0) {
    return 1;
  }
  return static_cast<int>((value + divisor - 1) / divisor);
}

int64_t StaticIntValueOrDefault(const PrimExpr& expr, int64_t default_value = 0) {
  if (const auto* imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return default_value;
}

bool IsLiteralZeroValue(const PrimExpr& expr) {
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

bool IsUnsupportedResidualLocalScope(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

static std::string PrimExprToCompactString(const PrimExpr& expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

static std::string EncodeGemmComputeOpSignature(
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

static GemmComputeOpFact BuildGemmComputeOpFact(
    const std::string& a_buffer, const std::string& b_buffer, const std::string& c_buffer, int m,
    int n, int k, bool transpose_a, bool transpose_b, int policy_type, bool clear_accum,
    int k_pack, int wg_wait, bool dst_full_sync_en, bool bfp8_pack_precise,
    const std::vector<std::pair<std::string, std::string>>& defines,
    const std::vector<std::pair<std::string, uint32_t>>& named_compile_args,
    const std::string& mbarrier_buffer, const std::string& mbarrier_scope,
    const std::vector<std::string>& mbarrier_index_exprs, DataType a_dtype, DataType b_dtype,
    DataType c_dtype) {
  GemmComputeOpFact fact;
  fact.a_buffer = a_buffer;
  fact.b_buffer = b_buffer;
  fact.c_buffer = c_buffer;
  fact.m = m;
  fact.n = n;
  fact.k = k;
  fact.transpose_a = transpose_a;
  fact.transpose_b = transpose_b;
  fact.policy_type = policy_type;
  fact.clear_accum = clear_accum;
  fact.k_pack = k_pack;
  fact.wg_wait = wg_wait;
  fact.dst_full_sync_en = dst_full_sync_en;
  fact.bfp8_pack_precise = bfp8_pack_precise;
  fact.defines = defines;
  fact.named_compile_args = named_compile_args;
  fact.mbarrier_buffer = mbarrier_buffer;
  fact.mbarrier_scope = mbarrier_scope;
  fact.mbarrier_index_exprs = mbarrier_index_exprs;
  fact.a_dtype = a_dtype;
  fact.b_dtype = b_dtype;
  fact.c_dtype = c_dtype;
  return fact;
}

}  // namespace

bool PlanTTKernelABI::IsMatmulCall(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.tileop.gemm_py";
}

bool PlanTTKernelABI::IsSingleFullTileMatmulOutput(const CallNode* op) const {
  if (!IsMatmulCall(op) || op->args.size() < 7) {
    return false;
  }
  const auto* m = op->args[5].as<IntImmNode>();
  const auto* n = op->args[6].as<IntImmNode>();
  return m != nullptr && n != nullptr && m->value == kBlackholeTileRows &&
         n->value == kBlackholeTileCols;
}

std::string PlanTTKernelABI::DataTypeToDataFormat(DataType dtype) {
  return DataTypeToDataFormatForBlackhole(dtype);
}

void PlanTTKernelABI::ExtractGemmInfo(const CallNode* op) {
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
  Buffer physical_c_buffer = ResolvePhysicalComputeBuffer(c_region->buffer);

  gemm_a_buffer_ = a_region->buffer;
  gemm_b_buffer_ = b_region->buffer;
  gemm_c_buffer_ = physical_c_buffer;
  gemm_a_buffer_name_ = BufferIdentityName(a_region->buffer);
  gemm_b_buffer_name_ = BufferIdentityName(b_region->buffer);
  gemm_c_buffer_name_ = BufferIdentityName(physical_c_buffer);
  gemm_c_scope_ = GetStorageScope(physical_c_buffer);
  gemm_a_dtype_ = a_region->buffer->dtype;
  gemm_b_dtype_ = b_region->buffer->dtype;
  gemm_c_dtype_ = physical_c_buffer->dtype;
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
  const std::string signature = EncodeGemmComputeOpSignature(
      gemm_a_buffer_name_, gemm_b_buffer_name_, gemm_c_buffer_name_, gemm_m_, gemm_n_, gemm_k_,
      gemm_transpose_a_, gemm_transpose_b_, gemm_policy_type_, gemm_clear_accum_, gemm_k_pack_,
      gemm_wg_wait_, gemm_dst_full_sync_en_, gemm_bfp8_pack_precise_, gemm_defines_,
      gemm_named_compile_args_, gemm_mbarrier_buffer_name_, gemm_mbarrier_scope_,
      gemm_mbarrier_index_exprs_);
  if (compute_op_signatures_.insert(signature).second) {
    gemm_compute_op_fact_index_by_signature_[signature] =
        static_cast<int>(gemm_compute_op_facts_.size());
    gemm_compute_op_facts_.push_back(BuildGemmComputeOpFact(
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
  const DataType gemm_c_cb_dtype =
      BufferUsesTiledCBLiveForm(c_region->buffer) ? ExactTiledCBStorageDType(gemm_c_dtype_)
                                                  : gemm_c_dtype_;
  const int c_cb_tile_bytes =
      kBlackholeTileRows * kBlackholeTileCols * gemm_c_cb_dtype.bytes();
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
    if (auto flow_fact_it = buffer_flow_facts_.find(buffer_identity);
        flow_fact_it != buffer_flow_facts_.end()) {
      flow_class = flow_fact_it->second.flow_class;
    } else if (auto use_count_it =
                   cb_consumed_compute_input_use_count_by_buffer_identity_.find(buffer_identity);
               use_count_it != cb_consumed_compute_input_use_count_by_buffer_identity_.end() &&
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
  set_requirement_fields(gemm_c_req_index_, c_cb_tile_bytes, num_m_tiles * num_n_tiles,
                         DataTypeToDataFormat(gemm_c_cb_dtype));
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

Stmt PlanTTKernelABI::LowerMatmulCallWithFlowAnalysis(
    const CallNode* op, int current_order_index, const FragmentCastMatch* post_merge_cast,
    int post_merge_cast_order_index, bool* consumed_post_merge_cast) {
  ICHECK(op != nullptr);
  if (consumed_post_merge_cast != nullptr) {
    *consumed_post_merge_cast = false;
  }
  ExtractGemmInfo(op);

  bool retain_in0 = false;
  bool retain_in1 = false;
  bool reacquire_in0 = false;
  bool reacquire_in1 = false;
  if (IsBufferLikeExpr(op->args[0])) {
    const Buffer in0_buffer = ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[0])->buffer);
    retain_in0 = ShouldRetainComputeInputBuffer(in0_buffer, current_order_index);
    if (!retain_in0) {
      reacquire_in0 = ShouldReacquireComputeInputBuffer(in0_buffer, current_order_index);
    }
  }
  if (IsBufferLikeExpr(op->args[1])) {
    const Buffer in1_buffer = ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[1])->buffer);
    retain_in1 = ShouldRetainComputeInputBuffer(in1_buffer, current_order_index);
    if (!retain_in1) {
      reacquire_in1 = ShouldReacquireComputeInputBuffer(in1_buffer, current_order_index);
    }
  }

  bool publish_out = true;
  bool publish_transport_out = true;
  bool preserve_out_local_state = false;
  if (IsBufferLikeExpr(op->args[2])) {
    const Buffer out_buffer = ResolvePhysicalComputeBuffer(NormalizeToBufferRegion(op->args[2])->buffer);
    const FutureBufferUses future_uses =
        ClassifyFutureBufferUses(out_buffer, current_order_index);
    const bool has_zero_preclear = HasZeroFragmentFillFact(out_buffer);
    bool adjacent_post_merge_cast_is_only_compute_consumer = false;
    if (post_merge_cast != nullptr && post_merge_cast_order_index >= 0 &&
        post_merge_cast->src.defined() && SameBufferIdentity(post_merge_cast->src, out_buffer)) {
      const FutureBufferUses uses_after_cast =
          ClassifyFutureBufferUses(out_buffer, post_merge_cast_order_index);
      adjacent_post_merge_cast_is_only_compute_consumer =
          !uses_after_cast.has_compute_consume && !uses_after_cast.has_transport_consume &&
          !uses_after_cast.has_reference;
    }
    const bool post_merge_cast_can_consume_live_cb =
        post_merge_cast != nullptr && adjacent_post_merge_cast_is_only_compute_consumer &&
        !has_zero_preclear;
    const bool needs_accumulator_merge =
        FindBufferMaterializationFact(out_buffer) != nullptr ||
        (future_uses.has_compute_consume && !post_merge_cast_can_consume_live_cb);
    ExactTiledCBValue live_accumulator;
    const bool has_live_accumulator =
        !gemm_clear_accum_ && !needs_accumulator_merge &&
        TryCreateLiveExactTiledCBValue(out_buffer, &live_accumulator);
    if (!gemm_clear_accum_ && !needs_accumulator_merge) {
      gemm_clear_accum_ = !has_live_accumulator;
    }
    publish_out = future_uses.has_compute_consume || future_uses.has_transport_consume;
    publish_transport_out = future_uses.has_transport_consume;
    preserve_out_local_state = future_uses.has_compute_consume || future_uses.has_reference;
  }

  const bool can_publish_post_merge_cast =
      post_merge_cast != nullptr && HasZeroFragmentFillFact(gemm_c_buffer_) &&
      CanPublishPostMergeCastWithPackTile(*post_merge_cast, post_merge_cast_order_index);
  if (can_publish_post_merge_cast) {
    const FutureBufferUses source_uses_after_cast =
        ClassifyFutureBufferUses(post_merge_cast->src, post_merge_cast_order_index);
    if (!source_uses_after_cast.has_compute_consume &&
        !source_uses_after_cast.has_transport_consume &&
        !source_uses_after_cast.has_reference) {
      preserve_out_local_state = false;
    }
    if (consumed_post_merge_cast != nullptr) {
      *consumed_post_merge_cast = true;
    }
  } else {
    post_merge_cast = nullptr;
    post_merge_cast_order_index = -1;
  }

  return GenerateMatmulSequence(op, retain_in0, retain_in1, publish_out,
                                publish_transport_out, preserve_out_local_state, reacquire_in0,
                                reacquire_in1, post_merge_cast, post_merge_cast_order_index);
}

Stmt PlanTTKernelABI::GenerateMatmulSequence(const CallNode* op,
                                               bool retain_in0,
                                               bool retain_in1,
                                               bool publish_out,
                                               bool publish_transport_out,
                                               bool preserve_out_local_state,
                                               bool reacquire_in0,
                                               bool reacquire_in1,
                                               const FragmentCastMatch* post_merge_cast,
                                               int post_merge_cast_order_index) {
  ICHECK_GE(gemm_a_req_index_, 0);
  ICHECK_GE(gemm_b_req_index_, 0);
  ICHECK_GE(gemm_c_req_index_, 0);
  Buffer logical_gemm_c_buffer;
  if (IsBufferLikeExpr(op->args[2])) {
    logical_gemm_c_buffer = NormalizeToBufferRegion(op->args[2])->buffer;
  }
  const bool merge_with_zero_reload = !gemm_clear_accum_ && HasZeroFragmentFillFact(gemm_c_buffer_);
  InvalidateLastFragmentFillValue(gemm_c_buffer_);
  if (logical_gemm_c_buffer.defined()) {
    InvalidateLastFragmentFillValue(logical_gemm_c_buffer);
  }

  if (!gemm_clear_accum_) {
    Stmt body =
        GenerateAccumulatingMatmulSequence(op, retain_in0, retain_in1, publish_transport_out,
                                           preserve_out_local_state, reacquire_in0, reacquire_in1,
                                           post_merge_cast, post_merge_cast_order_index,
                                           merge_with_zero_reload);
    if (logical_gemm_c_buffer.defined() && preserve_out_local_state) {
      ExactTiledCBValue live_value;
      if (TryCreateLiveExactTiledCBValue(gemm_c_buffer_, &live_value)) {
        RecordTiledCBLiveFormAliases(logical_gemm_c_buffer, live_value.cb_id);
      }
    }
    return MaybeWrapComputeSegment(body);
  }
  const bool publish_live_form_cb =
      preserve_out_local_state && BufferUsesTiledCBLiveForm(gemm_c_buffer_);
  if (publish_live_form_cb) {
    RecordTiledCBLiveFormAliases(gemm_c_buffer_, gemm_c_req_index_);
    if (logical_gemm_c_buffer.defined()) {
      RecordTiledCBLiveFormAliases(logical_gemm_c_buffer, gemm_c_req_index_);
    }
  }
  if (publish_out) {
    RecordTiledCBLiveFormAliases(gemm_c_buffer_, gemm_c_req_index_);
    if (logical_gemm_c_buffer.defined()) {
      RecordTiledCBLiveFormAliases(logical_gemm_c_buffer, gemm_c_req_index_);
    }
  }
  return MaybeWrapComputeSegment(
      GenerateMatmulSequenceForOutputRequirement(gemm_c_req_index_, retain_in0, retain_in1,
                                                 publish_out || publish_live_form_cb,
                                                 publish_out || publish_live_form_cb,
                                                 reacquire_in0, reacquire_in1));
}

Stmt PlanTTKernelABI::GenerateMatmulSequenceForOutputRequirement(int out_req_index,
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
  stmts.push_back(
      MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out_id)}));
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

Buffer PlanTTKernelABI::CreateClearAccumPartialsBuffer(const Buffer& buffer) {
  const std::string partials_name =
      BufferIdentityName(buffer) + "_clear_accum_partials_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(buffer->shape, buffer->dtype, partials_name, GetStorageScope(buffer));
}

Buffer PlanTTKernelABI::CreateFragmentMergeReloadBuffer(const Buffer& buffer) {
  const std::string reload_name = BufferIdentityName(buffer) + "_fragment_merge_reload_" +
                                  std::to_string(next_requirement_index_);
  return tir::decl_buffer(buffer->shape, buffer->dtype, reload_name, GetStorageScope(buffer));
}

bool PlanTTKernelABI::ClearAccumReloadNeedsDataFormatReconfig() const {
  return gemm_c_dtype_ != gemm_b_dtype_;
}

Stmt PlanTTKernelABI::GenerateAddFragmentSequence(const Buffer& dst,
                                                    const Buffer& src,
  const PrimExpr& num_elements) {
  InvalidateLastFragmentFillValue(dst);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  const Buffer physical_src = ResolvePhysicalComputeBuffer(src);
  return MaybeWrapComputeSegment(MakeBlackholeCall(
      blackhole_add_fragment(), {physical_dst->data, physical_src->data, num_elements}));
}

Stmt PlanTTKernelABI::GenerateAddFragmentFromCBFrontSequence(const Buffer& dst,
                                                               int src_cb_id,
                                                               const PrimExpr& num_elements,
                                                               const Buffer& src_buffer) {
  InvalidateLastFragmentFillValue(dst);
  const std::string dst_buffer_name = BufferIdentityName(dst);
  const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(dst);
  ICHECK(fact != nullptr)
      << "PlanTTKernelABI requires buffer materialization fact for "
         "add_fragment_from_cb_front destination "
      << dst_buffer_name;
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  return MaybeWrapComputeSegment(MakeBlackholeCall(
      blackhole_add_fragment_from_cb_front(),
      {physical_dst->data, IntImm32(src_cb_id), num_elements}));
}

Stmt PlanTTKernelABI::GenerateMergeFragmentTilesSequence(const Buffer& dst,
                                                           int partials_cb_id,
                                                           const Buffer& partials_buffer,
                                                           int reload_cb_id,
                                                           const Buffer& reload_buffer,
                                                           int live_form_cb_id,
                                                           const Buffer& live_form_buffer,
                                                           const PrimExpr& num_elements,
                                                           int num_c_tiles,
                                                           bool materialize_live_form_to_local_state,
                                                           int publish_cb_id,
                                                           int materialized_cast_cb_id,
                                                           bool merge_with_zero_reload,
                                                           int live_reload_cb_id,
                                                           const Buffer& live_reload_buffer) {
  InvalidateLastFragmentFillValue(dst);
  const std::string dst_buffer_name = BufferIdentityName(dst);
  const bool use_live_reload = !merge_with_zero_reload && live_reload_cb_id >= 0;
  const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(dst);
  if (!merge_with_zero_reload && !use_live_reload) {
    ICHECK(fact != nullptr)
        << "PlanTTKernelABI requires buffer materialization fact or exact live-CB state for "
           "merge_fragment_tiles destination "
        << dst_buffer_name;
  }
  if (fact != nullptr) {
    ICHECK(!fact->bridge_kind.empty())
        << "PlanTTKernelABI requires bridge_kind in buffer materialization fact for "
        << dst_buffer_name;
    ICHECK(!fact->execution_protocol.empty())
        << "PlanTTKernelABI requires execution_protocol in buffer materialization fact for "
        << dst_buffer_name;
    ICHECK_EQ(fact->bridge_kind, "tile_nfaces_materialization")
        << "PlanTTKernelABI does not support buffer-materialization bridge_kind "
        << fact->bridge_kind
        << " for " << dst_buffer_name;
    ICHECK_EQ(fact->execution_protocol, "dst_cb_binary_pack")
        << "PlanTTKernelABI does not support buffer-materialization execution_protocol "
        << fact->execution_protocol << " for " << dst_buffer_name;
  }
  ICHECK_GT(gemm_c_dtype_.bytes(), 0)
      << "Blackhole accumulator-merge lowering requires a valid destination dtype for "
      << dst_buffer_name;
  const int tile_elements = (kBlackholeTileRows * kBlackholeTileCols * gemm_c_dtype_.bytes()) /
                            gemm_c_dtype_.bytes();
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  if (!merge_with_zero_reload) {
    if (use_live_reload) {
      ICHECK(live_reload_buffer.defined())
          << "Blackhole live accumulator-merge lowering requires a live reload buffer for "
          << dst_buffer_name;
    } else {
      ICHECK_GE(reload_cb_id, 0)
          << "Blackhole accumulator-merge lowering requires a reload CB for "
          << dst_buffer_name;
      ICHECK(reload_buffer.defined())
          << "Blackhole accumulator-merge lowering requires a reload buffer for "
          << dst_buffer_name;
    }
    RecordExactComputeOpPlan("binary", "add_tiles",
                             {{use_live_reload ? "lhs" : "lhs",
                               use_live_reload ? live_reload_buffer : reload_buffer, "identity"},
                              {"rhs", partials_buffer, "identity"},
                              {"output", dst, "identity"}});
  }

  std::vector<Stmt> stmts;
  if (!merge_with_zero_reload) {
    if (use_live_reload) {
      stmts.push_back(MakeBlackholeCall(
          blackhole_reconfig_data_format(),
          {IntImm32(live_reload_cb_id), IntImm32(partials_cb_id)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_add_tiles_init(),
          {IntImm32(live_reload_cb_id), IntImm32(partials_cb_id)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(live_reload_cb_id), IntImm32(num_c_tiles)}));
    } else {
      stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                        {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
      for (int tile = 0; tile < num_c_tiles; ++tile) {
          stmts.push_back(MakeBlackholeCall(
              blackhole_pack_untilize_tile(),
              {physical_dst->data, IntImm32(reload_cb_id), IntImm32(tile),
               IntImm32(tile * tile_elements)}));
      }
      stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                        {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
      stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                        {IntImm32(reload_cb_id), IntImm32(partials_cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                        {IntImm32(reload_cb_id), IntImm32(partials_cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                        {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
    }
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                    {IntImm32(partials_cb_id), IntImm32(num_c_tiles)}));
  if (live_form_cb_id >= 0) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(live_form_cb_id), IntImm32(num_c_tiles)}));
  }
  if (materialized_cast_cb_id >= 0) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_reserve_back(), {IntImm32(materialized_cast_cb_id), IntImm32(num_c_tiles)}));
  }
  if (materialize_live_form_to_local_state) {
    ICHECK_GE(live_form_cb_id, 0)
        << "PlanTTKernelABI requires explicit live_form_cb_id when materializing merged "
           "fragment local state for "
        << dst_buffer_name;
  }
  for (int tile = 0; tile < num_c_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    if (merge_with_zero_reload) {
      if (ClearAccumReloadNeedsDataFormatReconfig()) {
        stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short_with_dt(),
                                          {IntImm32(gemm_b_req_index_), IntImm32(partials_cb_id)}));
      } else {
        stmts.push_back(MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(),
                                          {IntImm32(partials_cb_id)}));
      }
      stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                        {IntImm32(partials_cb_id), IntImm32(tile), IntImm32(0)}));
    } else {
      if (use_live_reload) {
        stmts.push_back(MakeBlackholeCall(
            blackhole_add_tiles(),
            {IntImm32(live_reload_cb_id), IntImm32(partials_cb_id),
             IntImm32(tile), IntImm32(tile), IntImm32(0)}));
      } else {
        stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                          {IntImm32(reload_cb_id), IntImm32(partials_cb_id),
                                           IntImm32(tile), IntImm32(tile), IntImm32(0)}));
      }
    }
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    if (live_form_cb_id >= 0) {
      stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                        {IntImm32(live_form_cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(live_form_cb_id)}));
    }
    if (publish_cb_id >= 0) {
      stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                        {IntImm32(publish_cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(publish_cb_id)}));
    }
    if (materialized_cast_cb_id >= 0) {
      stmts.push_back(MakeBlackholeCall(blackhole_pack_reconfig_data_format(),
                                        {IntImm32(materialized_cast_cb_id)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_pack_tile(), {IntImm32(0), IntImm32(materialized_cast_cb_id)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  if (!merge_with_zero_reload && !use_live_reload) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
  }
  if (use_live_reload) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(live_reload_cb_id),
                                       IntImm32(num_c_tiles)}));
  }
  stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                    {IntImm32(partials_cb_id), IntImm32(num_c_tiles)}));
  if (live_form_cb_id >= 0) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(live_form_cb_id), IntImm32(num_c_tiles)}));
  }
  if (materialize_live_form_to_local_state) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(live_form_cb_id), IntImm32(num_c_tiles)}));
    for (int tile = 0; tile < num_c_tiles; ++tile) {
      stmts.push_back(MakeBlackholeCall(blackhole_untilize_cb_front_tile_fragment(),
                                        {physical_dst->data, IntImm32(live_form_cb_id), IntImm32(tile),
                                         IntImm32(tile * tile_elements)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(live_form_cb_id), IntImm32(num_c_tiles)}));
  }
  if (publish_cb_id >= 0) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(publish_cb_id), IntImm32(num_c_tiles)}));
  }
  if (materialized_cast_cb_id >= 0) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_push_back(), {IntImm32(materialized_cast_cb_id), IntImm32(num_c_tiles)}));
  }
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

Stmt PlanTTKernelABI::GenerateMatmulSequenceWithPartialReload(int out_req_index,
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
  stmts.push_back(
      MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out_id)}));
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
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
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
  Stmt body = SeqStmt::Flatten(stmts);
  if (publish_out && gemm_c_buffer_.defined()) {
    ExactTiledCBValue live_value;
    live_value.buffer = gemm_c_buffer_;
    live_value.cb_id = out_id;
    live_value.borrowed_live = true;
    PopulateExactTiledCBValueShape(gemm_c_buffer_, &live_value);
    body = AttachExactOutputLiveFormMarker(gemm_c_buffer_, live_value, body);
  }
  return body;
}

bool PlanTTKernelABI::CanPublishPostMergeCastWithPackTile(const FragmentCastMatch& match,
                                                          int cast_order_index) const {
  if (gemm_clear_accum_) {
    return false;
  }
  if (!match.src.defined() || !match.dst.defined() || !gemm_c_buffer_.defined() ||
      !SameBufferIdentity(match.src, gemm_c_buffer_) || !tir::is_zero(match.src_offset) ||
      !tir::is_zero(match.dst_offset) || !match.dst->dtype.is_bfloat16()) {
    return false;
  }
  if (gemm_m_ <= 0 || gemm_n_ <= 0 || gemm_m_ % kBlackholeTileRows != 0 ||
      gemm_n_ % kBlackholeTileCols != 0) {
    return false;
  }
  const int64_t logical_elements = static_cast<int64_t>(gemm_m_) * gemm_n_;
  const int64_t cast_elements = StaticIntValueOrDefault(match.num_elements, -1);
  if (cast_elements != logical_elements ||
      GetLogicalBufferElementCount(match.src) != logical_elements ||
      GetLogicalBufferElementCount(match.dst) != logical_elements) {
    return false;
  }
  const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(match.dst);
  if (fact == nullptr) {
    return false;
  }
  return fact->kind == buffer_materialization::kRepublishedLogicalTile &&
         fact->bridge_kind == buffer_materialization::kTileNFacesMaterialization &&
         fact->execution_protocol == buffer_materialization::kTiledCBRepublish &&
         fact->result_live_form == buffer_live_form::kTiledCB;
}

bool PlanTTKernelABI::HasZeroFragmentFillFact(const Buffer& buffer) const {
  for (const std::string& identity : CollectBufferFlowIdentities(buffer)) {
    auto fill_it = last_fragment_fill_value_by_buffer_identity_.find(identity);
    if (fill_it != last_fragment_fill_value_by_buffer_identity_.end()) {
      return IsLiteralZeroValue(fill_it->second);
    }
  }
  if (const VarNode* data = BufferDataIdentity(buffer)) {
    auto fill_it = last_fragment_fill_value_by_data_.find(data);
    if (fill_it != last_fragment_fill_value_by_data_.end()) {
      return IsLiteralZeroValue(fill_it->second);
    }
  }
  return false;
}

int PlanTTKernelABI::PreparePostMergeCastPublishCB(const FragmentCastMatch& match,
                                                   int num_c_tiles) {
  const int cb_id = AllocateRequirementIndex(match.dst, CBType::kIntermediate);
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const int target_tile_bytes =
      kBlackholeTileRows * kBlackholeTileCols * match.dst->dtype.bytes();
  SetRequirementPageLayout(cb_id, target_tile_bytes, num_c_tiles);
  auto& req = cb_requirements_.at(cb_id);
  req.data_format = DataTypeToDataFormatForBlackhole(match.dst->dtype);
  req.flow_class = CBFlowClass::kStream;
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, num_c_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, num_c_tiles);
  return cb_id;
}

Stmt PlanTTKernelABI::GenerateAccumulatingMatmulSequence(const CallNode* op,
                                                           bool retain_in0,
                                                           bool retain_in1,
                                                           bool publish_transport_out,
                                                           bool preserve_out_local_state,
                                                           bool reacquire_in0,
                                                           bool reacquire_in1,
                                                           const FragmentCastMatch* post_merge_cast,
                                                           int post_merge_cast_order_index,
                                                           bool merge_with_zero_reload) {
  ICHECK(op != nullptr);
  ICHECK(IsUnsupportedResidualLocalScope(gemm_c_buffer_))
      << "Blackhole clear_accum=false lowering currently requires a compute-local accumulator "
         "destination, but "
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

  ExactTiledCBValue live_reload_value;
  const bool use_live_reload =
      !merge_with_zero_reload && TryCreateLiveExactTiledCBValue(gemm_c_buffer_, &live_reload_value);

  Buffer reload_buffer;
  int reload_req_index = -1;
  if (!merge_with_zero_reload && !use_live_reload) {
    reload_buffer = CreateFragmentMergeReloadBuffer(gemm_c_buffer_);
    reload_req_index = AllocateRequirementIndex(reload_buffer, CBType::kIntermediate);
    SetRequirementPageLayout(reload_req_index, c_tile_bytes, num_c_tiles);
    auto& reload_req = cb_requirements_.at(reload_req_index);
    reload_req.data_format = DataTypeToDataFormat(gemm_c_dtype_);
    reload_req.flow_class = CBFlowClass::kStream;
    reload_req.publish_pages_per_event =
        std::max(reload_req.publish_pages_per_event, num_c_tiles);
    reload_req.consume_pages_per_event =
        std::max(reload_req.consume_pages_per_event, num_c_tiles);
    MarkRequirementLifetimeOverlap(scratch_req_index, reload_req_index);
  }
  if (use_live_reload) {
    MarkRequirementLifetimeOverlap(scratch_req_index, live_reload_value.cb_id);
  }

  Buffer live_form_buffer;
  int live_form_req_index = -1;
  bool materialize_live_form_to_local_state = false;
  const bool use_tiled_cb_live_form =
      preserve_out_local_state &&
      (BufferUsesTiledCBLiveForm(gemm_c_buffer_) || use_live_reload);
  if (preserve_out_local_state) {
    const std::string output_identity = BufferIdentityName(gemm_c_buffer_);
    const bool reuse_seeded_source_live_cb =
        use_tiled_cb_live_form && seeded_cb_requirement_names_.count(output_identity) != 0U &&
        gemm_c_req_index_ >= 0 && IsSingleFullTileLogicalMatrix(gemm_c_buffer_) &&
        (!use_live_reload || gemm_c_req_index_ != live_reload_value.cb_id);
    if (reuse_seeded_source_live_cb) {
      live_form_buffer = gemm_c_buffer_;
      live_form_req_index = gemm_c_req_index_;
    } else {
      const std::string live_form_name =
          output_identity + "_fragment_merge_live_form_" + std::to_string(next_requirement_index_);
      live_form_buffer =
          tir::decl_buffer(gemm_c_buffer_->shape, gemm_c_buffer_->dtype, live_form_name,
                           GetStorageScope(gemm_c_buffer_));
      live_form_req_index = AllocateRequirementIndex(live_form_buffer, CBType::kIntermediate);
    }
    const DataType live_form_storage_dtype = ExactTiledCBStorageDType(gemm_c_dtype_);
    const int live_form_tile_bytes =
        kBlackholeTileRows * kBlackholeTileCols * live_form_storage_dtype.bytes();
    SetRequirementPageLayout(live_form_req_index, live_form_tile_bytes, num_c_tiles);
    auto& live_form_req = cb_requirements_.at(live_form_req_index);
    live_form_req.data_format = DataTypeToDataFormat(live_form_storage_dtype);
    live_form_req.flow_class = CBFlowClass::kStream;
    if (use_live_reload) {
      live_form_req.flow_class = CBFlowClass::kState;
    }
    live_form_req.publish_pages_per_event =
        std::max(live_form_req.publish_pages_per_event, num_c_tiles);
    live_form_req.consume_pages_per_event =
        std::max(live_form_req.consume_pages_per_event, num_c_tiles);
    MarkRequirementLifetimeOverlap(scratch_req_index, live_form_req_index);
    if (reload_req_index >= 0) {
      MarkRequirementLifetimeOverlap(reload_req_index, live_form_req_index);
    }
    if (use_live_reload && live_reload_value.cb_id != live_form_req_index) {
      MarkRequirementLifetimeOverlap(live_reload_value.cb_id, live_form_req_index);
    }
    materialize_live_form_to_local_state = !use_tiled_cb_live_form;
  }

  int materialized_cast_req_index = -1;
  if (post_merge_cast != nullptr &&
      CanPublishPostMergeCastWithPackTile(*post_merge_cast, post_merge_cast_order_index)) {
    materialized_cast_req_index = PreparePostMergeCastPublishCB(*post_merge_cast, num_c_tiles);
    MarkRequirementLifetimeOverlap(scratch_req_index, materialized_cast_req_index);
    if (reload_req_index >= 0) {
      MarkRequirementLifetimeOverlap(reload_req_index, materialized_cast_req_index);
    }
    if (use_live_reload && live_reload_value.cb_id != materialized_cast_req_index) {
      MarkRequirementLifetimeOverlap(live_reload_value.cb_id, materialized_cast_req_index);
    }
    if (live_form_req_index >= 0) {
      MarkRequirementLifetimeOverlap(live_form_req_index, materialized_cast_req_index);
    }
    const BlackholeBufferMaterializationFact* fact =
        FindBufferMaterializationFact(post_merge_cast->dst);
    ICHECK(fact != nullptr)
        << "PlanTTKernelABI requires a materialization fact for post-merge cast target "
        << BufferIdentityName(post_merge_cast->dst);
    PrimExpr cast_num_elements = post_merge_cast->num_elements;
    if (fact->logical_element_count > 0) {
      cast_num_elements = IntImm(
          DataType::Int(32), static_cast<int>(fact->logical_element_count));
    }
    RecordFragmentCastMaterializationPlans(
        *post_merge_cast, *fact, materialized_cast_req_index, cast_num_elements,
        buffer_materialization::kPackTile);
    RecordTiledCBLiveFormAliases(post_merge_cast->dst, materialized_cast_req_index);
  }

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
  stmts.push_back(GenerateMergeFragmentTilesSequence(
      gemm_c_buffer_, scratch_req_index, scratch_buffer, reload_req_index, reload_buffer,
      live_form_req_index, live_form_buffer, IntImm32(static_cast<int>(num_elements)),
      num_c_tiles, materialize_live_form_to_local_state,
      publish_transport_out ? gemm_c_req_index_ : -1, materialized_cast_req_index,
      merge_with_zero_reload, use_live_reload ? live_reload_value.cb_id : -1,
      use_live_reload ? live_reload_value.buffer : Buffer()));
  if (use_tiled_cb_live_form) {
    RecordTiledCBLiveFormAliases(gemm_c_buffer_, live_form_req_index);
  }

  return SeqStmt::Flatten(stmts);
}

}  // namespace tl
}  // namespace tvm
