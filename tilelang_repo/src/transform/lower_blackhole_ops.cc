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

static Array<TTKernelDefine> EncodeTTKernelDefines(
    const std::vector<std::pair<std::string, std::string>>& entries) {
  Array<TTKernelDefine> encoded_entries;
  for (const auto& [name, value] : entries) {
    encoded_entries.push_back(TTKernelDefine(String(name), String(value)));
  }
  return encoded_entries;
}

static Array<TTKernelNamedCompileArg> EncodeTTKernelNamedCompileArgs(
    const std::vector<std::pair<std::string, uint32_t>>& entries) {
  Array<TTKernelNamedCompileArg> encoded_entries;
  for (const auto& [name, value] : entries) {
    encoded_entries.push_back(TTKernelNamedCompileArg(String(name), value));
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

static Array<Integer> BuildIntegerArray(std::initializer_list<int64_t> values) {
  Array<Integer> result;
  for (int64_t value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

static Array<String> BuildStringArray(std::initializer_list<const char*> values) {
  Array<String> result;
  for (const char* value : values) {
    result.push_back(String(value));
  }
  return result;
}

static String ResolveComputeOperandHostBuffer(
    const std::unordered_map<std::string, std::string>& host_buffer_by_operand,
    const std::string& buffer) {
  auto it = host_buffer_by_operand.find(buffer);
  if (it != host_buffer_by_operand.end() && !it->second.empty()) {
    return String(it->second);
  }
  return String(buffer);
}

static TTComputeOperandBindingPlan BuildGemmComputeOperandBindingPlan(
    const GemmComputeOpFact& fact,
    const std::unordered_map<std::string, std::string>& host_buffer_by_operand,
    const char* role) {
  std::string buffer;
  DataType dtype;
  bool transpose = false;
  const std::string role_string(role);
  if (role_string == "a") {
    buffer = fact.a_buffer;
    dtype = fact.a_dtype;
    transpose = fact.transpose_a;
  } else if (role_string == "b") {
    buffer = fact.b_buffer;
    dtype = fact.b_dtype;
    transpose = fact.transpose_b;
  } else {
    buffer = fact.c_buffer;
    dtype = fact.c_dtype;
  }
  const String host_buffer = ResolveComputeOperandHostBuffer(host_buffer_by_operand, buffer);
  const String data_format = String(DataTypeToDataFormatForBlackhole(dtype));

  return TTComputeOperandBindingPlan(
      String(role), String(buffer), host_buffer, data_format, data_format,
      String(transpose ? "transpose" : "identity"));
}

static TTComputeOpPlan BuildTTComputeOpPlanFromFact(
    const GemmComputeOpFact& fact,
    const std::unordered_map<std::string, std::string>& host_buffer_by_operand,
    const String& kernel_name, int64_t kernel_plan_index, int64_t ordinal) {
  Array<TTComputeOperandBindingPlan> operand_bindings;
  const TTComputeOperandBindingPlan a_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "a");
  const TTComputeOperandBindingPlan b_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "b");
  const TTComputeOperandBindingPlan c_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "c");
  operand_bindings.push_back(a_binding);
  operand_bindings.push_back(b_binding);
  operand_bindings.push_back(c_binding);

  const int64_t mt = fact.m > 0 ? fact.m / 32 : 0;
  const int64_t nt = fact.n > 0 ? fact.n / 32 : 0;
  const int64_t kt = fact.k > 0 ? fact.k / 32 : 0;

  Array<String> mbarrier_index_exprs;
  for (const auto& expr : fact.mbarrier_index_exprs) {
    mbarrier_index_exprs.push_back(String(expr));
  }

  const std::string name = "compute_op_" + static_cast<std::string>(kernel_name) + "_" +
                           std::to_string(ordinal);
  return TTComputeOpPlan(
      String(name), kernel_name, kernel_plan_index, String("gemm"), String("matmul_tiles"),
      Bool(true), operand_bindings, BuildStringArray({"M", "N", "K"}),
      BuildIntegerArray({fact.m, fact.n, fact.k}), BuildIntegerArray({mt, nt, kt}),
      BuildIntegerArray({mt, nt, kt}), BuildIntegerArray({mt, nt}),
      String(DataTypeToDataFormatForBlackhole(fact.c_dtype)),
      String(fact.mbarrier_buffer), String(fact.mbarrier_scope), mbarrier_index_exprs);
}

static void SetOptionalBufferField(Map<String, Any>* payload, const char* key,
                                   const Buffer& buffer) {
  if (buffer.defined()) {
    payload->Set(String(key), String(BufferIdentityName(buffer)));
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

static std::string GetRuntimeArgKind(const Any& arg_item) {
  auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
  if (arg.empty() || !arg.Get("kind")) {
    return "";
  }
  return static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
}

static std::string GetRuntimeArgBufferName(const Any& arg_item) {
  auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
  if (arg.empty() || !arg.Get("buffer")) {
    return "";
  }
  return static_cast<std::string>(Downcast<String>(arg.Get("buffer").value()));
}

static int FindRuntimeArgIndex(const Array<Any>& runtime_args, const std::string& kind,
                               const std::string& buffer_name = "") {
  for (int i = 0, n = static_cast<int>(runtime_args.size()); i < n; ++i) {
    if (GetRuntimeArgKind(runtime_args[i]) != kind) {
      continue;
    }
    if (!buffer_name.empty()) {
      const std::string existing_buffer = GetRuntimeArgBufferName(runtime_args[i]);
      if (!existing_buffer.empty() && existing_buffer != buffer_name) {
        continue;
      }
    }
    return i;
  }
  return -1;
}

static Map<String, Any> MakeRuntimeArg(const std::string& name, const std::string& kind,
                                       const std::string& dtype,
                                       const std::string& buffer_name = "") {
  Map<String, Any> arg;
  arg.Set("name", String(name));
  arg.Set("kind", String(kind));
  arg.Set("dtype", String(dtype));
  if (!buffer_name.empty()) {
    arg.Set("buffer", String(buffer_name));
  }
  arg.Set("identity", String(MakeBlackholeRuntimeArgIdentity(kind, name, buffer_name)));
  return arg;
}

static std::string ResolveAccessorBufferNameByCompileTimeArgOffset(const Array<Any>& accessors,
                                                                   int offset) {
  for (const auto& accessor_item : accessors) {
    auto accessor = accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (accessor.empty() || !accessor.Get("buffer")) {
      continue;
    }
    if (!accessor.Get("compile_time_arg_offset")) {
      continue;
    }
    const int accessor_offset =
        Downcast<Integer>(accessor.Get("compile_time_arg_offset").value()).IntValue();
    if (accessor_offset == offset) {
      return static_cast<std::string>(Downcast<String>(accessor.Get("buffer").value()));
    }
  }
  return "";
}

static Array<Any> EnsureSegmentBufferRuntimeArgs(const std::string& segment_kind,
                                                 const Array<Any>& accessors,
                                                 const Optional<Any>& runtime_args_opt,
                                                 const std::string& input_buffer_name = "",
                                                 const std::string& output_buffer_name = "") {
  Array<Any> existing_runtime_args =
      runtime_args_opt ? Downcast<Array<Any>>(runtime_args_opt.value()) : Array<Any>();
  if (segment_kind == "fused_dataflow") {
    std::string resolved_input_buffer_name =
        !input_buffer_name.empty()
            ? input_buffer_name
            : ResolveAccessorBufferNameByCompileTimeArgOffset(accessors, 0);
    std::string resolved_output_buffer_name = !output_buffer_name.empty()
                                                  ? output_buffer_name
                                                  : ResolveAccessorBufferNameByCompileTimeArgOffset(
                                                        accessors,
                                                        resolved_input_buffer_name.empty() ? 0 : 2);
    std::vector<bool> consumed(existing_runtime_args.size(), false);
    Array<Any> runtime_args;
    auto push_existing_or_synthesized = [&](const std::string& kind, const std::string& name,
                                            const std::string& buffer_name = "") {
      const int existing_index = FindRuntimeArgIndex(existing_runtime_args, kind, buffer_name);
      if (existing_index >= 0) {
        runtime_args.push_back(existing_runtime_args[existing_index]);
        consumed[existing_index] = true;
        return;
      }
      runtime_args.push_back(MakeRuntimeArg(name, kind, "uint32", buffer_name));
    };

    if (!resolved_input_buffer_name.empty()) {
      push_existing_or_synthesized("input_buffer_addr32",
                                   resolved_input_buffer_name + "_addr",
                                   resolved_input_buffer_name);
    }
    if (!resolved_output_buffer_name.empty()) {
      push_existing_or_synthesized("output_buffer_addr32",
                                   resolved_output_buffer_name + "_addr",
                                   resolved_output_buffer_name);
    }
    push_existing_or_synthesized("work_linear_id", "work_linear_id");
    if (!resolved_input_buffer_name.empty()) {
      push_existing_or_synthesized("a_tile_start_id", "a_tile_start_id");
      push_existing_or_synthesized("a_tile_num_tiles", "a_tile_num_tiles");
      push_existing_or_synthesized("a_tile_stride", "a_tile_stride");
    }
    if (!resolved_output_buffer_name.empty()) {
      push_existing_or_synthesized("output_tile_start_id", "output_tile_start_id");
      push_existing_or_synthesized("output_tile_num_tiles", "output_tile_num_tiles");
      push_existing_or_synthesized("output_tile_stride", "output_tile_stride");
    }
    for (int i = 0, n = static_cast<int>(existing_runtime_args.size()); i < n; ++i) {
      if (!consumed[i]) {
        runtime_args.push_back(existing_runtime_args[i]);
      }
    }
    return runtime_args;
  }

  const bool is_reader = segment_kind == "reader";
  const bool is_writer = segment_kind == "writer";
  if (!is_reader && !is_writer) {
    return existing_runtime_args;
  }

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
  auto append_runtime_arg_if_missing = [&](const std::string& name, const std::string& kind,
                                           const std::string& buffer_name = "") {
    if (FindRuntimeArgIndex(runtime_args, kind, buffer_name) >= 0) {
      return;
    }
    runtime_args.push_back(MakeRuntimeArg(name, kind, "uint32", buffer_name));
  };
  if (is_reader) {
    std::vector<std::string> input_buffers;
    for (const auto& item : buffer_args) {
      auto arg = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (arg.empty() || !arg.Get("kind")) {
        continue;
      }
      const std::string arg_kind =
          static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
      if (arg_kind != "input_buffer_addr32" && arg_kind != "input_buffer_addr") {
        continue;
      }
      input_buffers.push_back(
          arg.Get("buffer") ? static_cast<std::string>(Downcast<String>(arg.Get("buffer").value()))
                            : std::string());
    }
    if (!input_buffers.empty()) {
      append_runtime_arg_if_missing("a_tile_start_id", "a_tile_start_id", input_buffers[0]);
      append_runtime_arg_if_missing("a_tile_num_tiles", "a_tile_num_tiles", input_buffers[0]);
      append_runtime_arg_if_missing("a_tile_stride", "a_tile_stride", input_buffers[0]);
    }
    if (input_buffers.size() > 1) {
      append_runtime_arg_if_missing("b_tile_start_id", "b_tile_start_id", input_buffers[1]);
      append_runtime_arg_if_missing("b_tile_num_tiles", "b_tile_num_tiles", input_buffers[1]);
      append_runtime_arg_if_missing("b_tile_stride", "b_tile_stride", input_buffers[1]);
    }
    append_runtime_arg_if_missing("k_tile_start_id", "k_tile_start_id");
    append_runtime_arg_if_missing("num_k_tiles", "num_k_tiles");
  } else if (is_writer) {
    std::string resolved_output_buffer_name = !output_buffer_name.empty() ? output_buffer_name : "";
    if (resolved_output_buffer_name.empty()) {
      for (const auto& item : buffer_args) {
        auto arg = item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (arg.empty() || !arg.Get("kind")) {
          continue;
        }
        const std::string arg_kind =
            static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
        if (arg_kind != "output_buffer_addr32" && arg_kind != "output_buffer_addr") {
          continue;
        }
        if (arg.Get("buffer")) {
          resolved_output_buffer_name =
              static_cast<std::string>(Downcast<String>(arg.Get("buffer").value()));
        }
        break;
      }
    }
    append_runtime_arg_if_missing("output_tile_start_id", "output_tile_start_id",
                                  resolved_output_buffer_name);
    append_runtime_arg_if_missing("output_tile_num_tiles", "output_tile_num_tiles",
                                  resolved_output_buffer_name);
    append_runtime_arg_if_missing("output_tile_stride", "output_tile_stride",
                                  resolved_output_buffer_name);
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
using tir::builtin::blackhole_copy_tile;
using tir::builtin::blackhole_add_tiles_init;
using tir::builtin::blackhole_add_tiles;
using tir::builtin::blackhole_add_bcast_rows_init_short;
using tir::builtin::blackhole_add_tiles_bcast_rows;
using tir::builtin::blackhole_mul_tiles_init;
using tir::builtin::blackhole_mul_tiles;
using tir::builtin::blackhole_mul_bcast_rows_init_short;
using tir::builtin::blackhole_mul_tiles_bcast_rows;
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
using tir::builtin::blackhole_write_local_slice_to_cb;
using tir::builtin::blackhole_write_local_fragment_tile_to_cb;
using tir::builtin::blackhole_write_local_fragment_slice_to_tiled_cb;
using tir::builtin::blackhole_cast_fragment_slice_to_tiled_cb;
using tir::builtin::blackhole_pack_fill_fragment_to_tiled_cb;
using tir::builtin::blackhole_read_cb_front_tile_to_local;
using tir::builtin::blackhole_read_cb_front_tile_to_local_fragment;
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
                                               const std::string& memory_space = "",
                                               std::vector<int64_t> host_axis_order = {},
                                               bool transpose_2d = false) {
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
  if (!host_axis_order.empty()) {
    Array<Any> axis_order;
    for (int64_t axis : host_axis_order) {
      axis_order.push_back(Integer(axis));
    }
    spec.Set("host_axis_order", axis_order);
  }
  if (transpose_2d) {
    spec.Set("transpose_2d", Bool(true));
  }
  return spec;
}

static TTPerWorkArgSpec MakePerWorkArgSpec(const std::string& arg_kind,
                                           const std::string& arg_identity,
                                           const std::string& descriptor_kind,
                                           const std::string& value_source,
                                           const std::string& buffer = "",
                                           uint32_t constant_value = 0) {
  return TTPerWorkArgSpec(String(arg_kind), String(arg_identity), String(buffer),
                          String(descriptor_kind), String(value_source),
                          static_cast<int64_t>(constant_value));
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

static TTKernelLaunchSpec MakeLaunchSpec(const std::string& core_type,
                                         const std::string& processor,
                                         const std::string& noc) {
  return TTKernelLaunchSpec(String(core_type), String(processor), String(noc));
}

static TTKernelComputeConfig MakeEmptyComputeConfig() {
  return TTKernelComputeConfig(String(""), false, false, false, Array<String>{}, false,
                               Array<TTKernelDefine>{}, Array<TTKernelNamedCompileArg>{},
                               false, 1, 0, 0, String(""));
}

static Map<String, Any> AsStringAnyMap(const Any& item) {
  return item.as<Map<String, Any>>().value_or(Map<String, Any>());
}

static String GetMapString(const Map<String, Any>& item, const char* key) {
  if (auto value = item.Get(key)) {
    return Downcast<String>(value.value());
  }
  return String("");
}

static int64_t GetMapInteger(const Map<String, Any>& item, const char* key,
                             int64_t default_value = 0) {
  if (auto value = item.Get(key)) {
    return Downcast<Integer>(value.value()).IntValue();
  }
  return default_value;
}

static bool GetMapBool(const Map<String, Any>& item, const char* key,
                       bool default_value = false) {
  if (auto value = item.Get(key)) {
    return Downcast<Bool>(value.value());
  }
  return default_value;
}

static Array<Integer> GetMapIntegerArray(const Map<String, Any>& item, const char* key) {
  Array<Integer> values;
  if (auto value = item.Get(key)) {
    for (const Any& element : Downcast<Array<Any>>(value.value())) {
      values.push_back(Downcast<Integer>(element));
    }
  }
  return values;
}

static TTRuntimeArgSpec DecodeRuntimeArgSpec(const Any& item) {
  const Map<String, Any> arg = AsStringAnyMap(item);
  return TTRuntimeArgSpec(GetMapString(arg, "name"), GetMapString(arg, "kind"),
                          GetMapString(arg, "dtype"), GetMapString(arg, "buffer"),
                          GetMapString(arg, "identity"), GetMapInteger(arg, "core_x", -1),
                          GetMapInteger(arg, "core_y", -1));
}

static Array<TTRuntimeArgSpec> DecodeRuntimeArgSpecs(const Optional<Any>& items_opt) {
  Array<TTRuntimeArgSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any& item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeRuntimeArgSpec(item));
  }
  return specs;
}

static TTCompileTimeArgSpec DecodeCompileTimeArgSpec(const Any& item) {
  const Map<String, Any> spec = AsStringAnyMap(item);
  return TTCompileTimeArgSpec(
      GetMapString(spec, "name"), GetMapString(spec, "kind"), GetMapString(spec, "dtype"),
      GetMapInteger(spec, "offset"), GetMapInteger(spec, "count"),
      GetMapString(spec, "buffer"), GetMapString(spec, "segment_role"),
      GetMapIntegerArray(spec, "values"), GetMapInteger(spec, "args_config_bits"),
      GetMapInteger(spec, "transport_page_size"), GetMapString(spec, "layout"),
      GetMapString(spec, "memory_space"), GetMapIntegerArray(spec, "host_axis_order"),
      GetMapBool(spec, "transpose_2d"));
}

static Array<TTCompileTimeArgSpec> DecodeCompileTimeArgSpecs(const Optional<Any>& items_opt) {
  Array<TTCompileTimeArgSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any& item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeCompileTimeArgSpec(item));
  }
  return specs;
}

static TTAccessorSpec DecodeAccessorSpec(const Any& item) {
  const Map<String, Any> accessor = AsStringAnyMap(item);
  return TTAccessorSpec(
      GetMapString(accessor, "buffer"), GetMapInteger(accessor, "compile_time_arg_offset"),
      GetMapInteger(accessor, "compile_time_arg_count"),
      GetMapInteger(accessor, "common_runtime_arg_offset"),
      GetMapInteger(accessor, "common_runtime_arg_count"),
      GetMapInteger(accessor, "args_config_bits"),
      GetMapInteger(accessor, "transport_page_size"), GetMapString(accessor, "layout"),
      GetMapString(accessor, "memory_space"), GetMapIntegerArray(accessor, "host_axis_order"),
      GetMapBool(accessor, "transpose_2d"));
}

static Array<TTAccessorSpec> DecodeAccessorSpecs(const Optional<Any>& items_opt) {
  Array<TTAccessorSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any& item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeAccessorSpec(item));
  }
  return specs;
}

static TTSemaphoreBindingSpec DecodeSemaphoreBindingSpec(const Any& item) {
  const Map<String, Any> binding = AsStringAnyMap(item);
  return TTSemaphoreBindingSpec(GetMapString(binding, "name"),
                                GetMapInteger(binding, "semaphore_id"),
                                GetMapString(binding, "arg_kind"));
}

static Array<TTSemaphoreBindingSpec> DecodeSemaphoreBindingSpecs(
    const Optional<Any>& items_opt) {
  Array<TTSemaphoreBindingSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any& item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeSemaphoreBindingSpec(item));
  }
  return specs;
}

static void BuildTTKernelAndABISeeds(const Array<Any>& segment_plan, Array<TTKernel>* kernels_out,
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
    Array<TTRuntimeArgSpec> runtime_args = DecodeRuntimeArgSpecs(segment.Get("runtime_args"));
    Array<TTRuntimeArgSpec> common_runtime_args =
        DecodeRuntimeArgSpecs(segment.Get("common_runtime_args"));
    Array<TTCompileTimeArgSpec> compile_time_arg_specs =
        DecodeCompileTimeArgSpecs(segment.Get("compile_time_arg_specs"));
    Array<TTAccessorSpec> accessors = DecodeAccessorSpecs(segment.Get("accessors"));
    Array<TTSemaphoreBindingSpec> semaphore_bindings =
        DecodeSemaphoreBindingSpecs(segment.Get("semaphore_bindings"));
    TTKernelLaunchSpec launch_spec =
        segment.Get("launch_spec") ? Downcast<TTKernelLaunchSpec>(segment.Get("launch_spec").value())
                                   : TTKernelLaunchSpec(String(""), String(""), String(""));
    TTKernelComputeConfig compute_config =
        segment.Get("compute_config")
            ? Downcast<TTKernelComputeConfig>(segment.Get("compute_config").value())
            : MakeEmptyComputeConfig();
    Array<TTPerWorkArgSpec> per_work_arg_specs =
        segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs))
            ? Downcast<Array<TTPerWorkArgSpec>>(
                  segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs)).value())
            : Array<TTPerWorkArgSpec>();
    abi_plans.push_back(TTABIPlan(String("abi_" + std::to_string(index)), kernel_name, runtime_args,
                                  common_runtime_args, compile_time_arg_specs, accessors,
                                  semaphore_bindings));
    kernels.push_back(TTKernel(kernel_name, kernel_kind, core_type, index, launch_spec,
                               compute_config, per_work_arg_specs));
    ++index;
  }
  *kernels_out = kernels;
  *abi_plans_out = abi_plans;
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
    std::pair<int64_t, int64_t> logical_matrix_shape,
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
  if (logical_matrix_shape.first > 0 && logical_matrix_shape.second > 0) {
    return logical_matrix_shape;
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
bool HasResidualRowBroadcast(const Stmt& body);
}  // namespace

static std::vector<std::string> CollectLeafUnsupportedComputeOpsFromBody(const Stmt& body) {
  std::vector<std::string> unsupported_ops;
  std::unordered_set<std::string> seen_ops;
  auto push = [&](const char* op_name) {
    if (seen_ops.insert(op_name).second) {
      unsupported_ops.push_back(op_name);
    }
  };
  if (HasResidualRowBroadcast(body)) {
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

static int CountLoweredRowReductionBuiltins(const Stmt& body) {
  int count = 0;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    const std::string& op_name = Downcast<Op>(call->op)->name;
    if (op_name == "tl.blackhole.reduce_init" || op_name == "tl.blackhole.reduce_tile" ||
        op_name == "tl.blackhole.reduce_uninit") {
      ++count;
    }
  });
  return count;
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
      if (call->op->IsInstance<OpNode>() && Downcast<Op>(call->op)->name == "tl.tileop.gemm_py") {
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

static std::vector<std::string> CollectSegmentKindsFromBody(const Stmt& body) {
  class SegmentKindCollector : public tir::StmtVisitor {
   public:
    void VisitStmt_(const tir::AttrStmtNode* op) final {
      if (op->attr_key == "blackhole.segment_kind") {
        if (const auto* kind = op->value.as<tir::StringImmNode>()) {
          const std::string segment_kind = kind->value;
          if (seen_.insert(segment_kind).second) {
            segment_kinds_.push_back(segment_kind);
          }
        }
      }
      tir::StmtVisitor::VisitStmt_(op);
    }

    const std::vector<std::string>& segment_kinds() const { return segment_kinds_; }

   private:
    std::unordered_set<std::string> seen_;
    std::vector<std::string> segment_kinds_;
  };

  SegmentKindCollector collector;
  collector(body);
  return collector.segment_kinds();
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

static std::string CoreTypeForSegmentKind(const std::string& segment_kind) {
  if (segment_kind == "reader") {
    return "brisc";
  }
  if (segment_kind == "compute") {
    return "trisc";
  }
  if (segment_kind == "writer") {
    return "ncrisc";
  }
  return "brisc";
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

void PlanTTKernelABI::LoadLogicalTileLayoutSpecs(const SpatialPlan& spatial_plan) {
  logical_tile_layout_specs_by_buffer_ = BuildLogicalTileLayoutSpecMap(spatial_plan);
}

void PlanTTKernelABI::LoadSpatialLiveValueBoundaries(const SpatialPlan& plan) {
  spatial_live_value_by_subject_.clear();
  spatial_materialization_boundary_by_source_target_.clear();
  std::unordered_map<std::string, std::string> subject_by_live_value;

  for (int64_t i = 0; i < static_cast<int64_t>(plan->live_values.size()); ++i) {
    const LiveValue& live_value = plan->live_values[i];
    const std::string subject = static_cast<std::string>(live_value->subject);
    if (subject.empty()) {
      continue;
    }
    if (spatial_live_value_by_subject_.find(subject) == spatial_live_value_by_subject_.end()) {
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
    prefix->push_back(MakeBlackholeCall(blackhole_read_cb_front_tile_to_local_fragment(),
                                        {physical_src->data, IntImm32(cb_id), IntImm32(tile),
                                         IntImm32(tile * tile_elements)}));
  }

  const FutureBufferUses future_uses = ClassifyFutureBufferUses(src, current_order_index);
  if (!future_uses.has_compute_consume && !future_uses.has_transport_consume &&
      !future_uses.has_reference) {
    suffix->push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(num_tiles)}));
    buffer_live_form_cb_by_buffer_identity_.erase(live_form_it);
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
  spatial_live_value_by_subject_.clear();
  spatial_materialization_boundary_by_source_target_.clear();
  tt_compute_op_plans_.clear();
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

std::optional<std::pair<int64_t, int64_t>>
PlanTTKernelABI::InferStagedCopySharedShapeFromTransportCoverage(
    const BufferStoreNode* op, const std::vector<Var>& loop_vars_to_zero) const {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return std::nullopt;
  }

  const CopyDirection direction = GetCopyDirection(op);
  if (direction != CopyDirection::kDramToCB && direction != CopyDirection::kCBToDram) {
    return std::nullopt;
  }

  const Buffer& global_buffer =
      direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;
  const std::vector<int64_t> logical_global_shape = GetLogicalBufferShape(global_buffer);
  if (logical_global_shape.size() < 2U) {
    return std::nullopt;
  }

  const int64_t global_cols = logical_global_shape.back();
  if (global_cols <= 0) {
    return std::nullopt;
  }

  PrimExpr row_expr;
  PrimExpr col_expr;
  const auto [row_axis, col_axis] =
      SelectStagedCopyTransportAxes(global_indices, loop_vars_to_zero);
  if (global_indices.size() > static_cast<size_t>(std::max(row_axis, col_axis))) {
    row_expr = ScalarizeVectorizedIndex(global_indices[row_axis]);
    col_expr = ScalarizeVectorizedIndex(global_indices[col_axis]);
  } else if (global_indices.size() == 1U) {
    PrimExpr linear_index = ScalarizeVectorizedIndex(global_indices[0]);
    const PrimExpr global_cols_imm = IntImm(linear_index.dtype(), global_cols);
    row_expr = FloorDiv(linear_index, global_cols_imm);
    col_expr = FloorMod(linear_index, global_cols_imm);
  } else {
    return std::nullopt;
  }

  Analyzer analyzer;
  for (const auto& [var_ptr, extent] : thread_index_var_static_extents_) {
    if (extent <= 0) {
      return std::nullopt;
    }
    analyzer.Bind(GetRef<Var>(var_ptr),
                  Range::FromMinExtent(IntImm(DataType::Int(32), 0),
                                       IntImm(DataType::Int(32), extent)));
  }
  for (const Var& loop_var : loop_vars_to_zero) {
    auto it = loop_var_static_extents_.find(loop_var.get());
    if (it == loop_var_static_extents_.end() || it->second <= 0) {
      return std::nullopt;
    }
    analyzer.Bind(loop_var, Range::FromMinExtent(IntImm(loop_var.dtype(), 0),
                                                 IntImm(loop_var.dtype(), it->second)));
  }

  auto zero_transport_vars = [&](const PrimExpr& expr) {
    return ZeroThreadAndLoopVars(expr, loop_vars_to_zero);
  };
  const PrimExpr row_offset = analyzer.Simplify(row_expr - zero_transport_vars(row_expr));
  const PrimExpr col_offset = analyzer.Simplify(col_expr - zero_transport_vars(col_expr));
  const arith::ConstIntBound row_bounds = analyzer.const_int_bound(row_offset);
  const arith::ConstIntBound col_bounds = analyzer.const_int_bound(col_offset);
  if (row_bounds->min_value == arith::ConstIntBound::kNegInf ||
      row_bounds->max_value == arith::ConstIntBound::kPosInf ||
      col_bounds->min_value == arith::ConstIntBound::kNegInf ||
      col_bounds->max_value == arith::ConstIntBound::kPosInf) {
    return std::nullopt;
  }

  const int64_t vector_lanes = std::max<int>(1, op->value.dtype().lanes());
  const int64_t shared_rows = row_bounds->max_value - row_bounds->min_value + 1;
  const int64_t shared_cols = col_bounds->max_value - col_bounds->min_value + vector_lanes;
  if (shared_rows <= 0 || shared_cols <= 0) {
    return std::nullopt;
  }
  return std::make_pair(shared_rows, shared_cols);
}

Array<Integer> PlanTTKernelABI::GetEncodedCurrentStagedCopySharedShape(
    const BufferStoreNode* op, const std::vector<Var>& loop_vars_to_zero) const {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return {};
  }
  const CopyDirection direction = GetCopyDirection(op);
  const Buffer& shared_buffer =
      direction == CopyDirection::kDramToCB ? op->buffer : load->buffer;
  Array<Integer> shared_shape = GetEncodedCurrentBufferShape(shared_buffer);
  if (shared_shape.size() >= 2U) {
    return shared_shape;
  }
  auto inferred_shape = InferStagedCopySharedShapeFromTransportCoverage(op, loop_vars_to_zero);
  if (!inferred_shape.has_value()) {
    return shared_shape;
  }
  shared_shape.clear();
  shared_shape.push_back(Integer(inferred_shape.value().first));
  shared_shape.push_back(Integer(inferred_shape.value().second));
  return shared_shape;
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
  stmt_order_index_by_node_.clear();
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
  logical_tile_layout_specs_by_buffer_.clear();
  spatial_live_value_by_subject_.clear();
  spatial_materialization_boundary_by_source_target_.clear();
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
  LoadBufferFlowFacts(lowering_support_facts);
  stmt_order_index_by_node_ = BuildExecutionOrderIndexByStmtNode(func->body);
  const std::vector<std::string> expected_unsupported_ops =
      CollectLeafUnsupportedComputeOpsFromBody(func->body);
  int expected_row_reduction_count = 0;
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    RowReductionMatch match;
    if (const auto* loop = node.as<ForNode>()) {
      if (MatchDirectRowReduction(loop, &match) || MatchGroupedRowReduction(loop, &match)) {
        ++expected_row_reduction_count;
      }
      return;
    }
    if (const auto* allocate = node.as<AllocateNode>()) {
      if (MatchAllocatedRowReduction(allocate, &match)) {
        ++expected_row_reduction_count;
      }
    }
  });
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
  if (expected_row_reduction_count > CountLoweredRowReductionBuiltins(body_with_segment_markers)) {
    push_unresolved("reduction");
  }
  for (const std::string& op_name : expected_unsupported_ops) {
    if (op_name == "broadcast" && HasResidualRowBroadcast(body_with_segment_markers)) {
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

int PlanTTKernelABI::EstimateCopyPageSize(const Buffer& buffer) const {
  const int64_t total_elements = GetLogicalBufferElementCount(buffer);
  if (total_elements <= 0) {
    return 2048;
  }

  const int64_t dtype_bytes = buffer->dtype.bytes();
  const int64_t total_bytes = total_elements * dtype_bytes;
  const int64_t default_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * dtype_bytes;
  return static_cast<int>(std::max<int64_t>(dtype_bytes, std::min(total_bytes, default_tile_bytes)));
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

bool PlanTTKernelABI::UseStagedCopyPageTransport(const Buffer& shared_buffer) const {
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
    }
  }
  next_requirement_index_ =
      std::max(next_requirement_index_, static_cast<int>(cb_requirements_.size()));
}

void PlanTTKernelABI::LoadSeededComputeOpPlans(const PrimFunc& func) {
  auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  if (!staged_program) {
    return;
  }
  for (const TTComputeOpPlan& plan : staged_program.value()->compute_op_plans) {
    tt_compute_op_plans_.push_back(plan);
  }
}

void PlanTTKernelABI::StoreSegmentPlan(PrimFunc& func) {
  const std::vector<std::string> segment_kinds = CollectSegmentKindsFromBody(func->body);
  if (segment_kinds.empty() && !needs_copy_runtime_args_ && !requires_compute_segment_) {
    segment_plan_ = Array<Any>();
    return;
  }
  ICHECK(!requires_compute_segment_ || !segment_kinds.empty())
      << "PlanTTKernelABI requires explicit segment_kind truth for compute-bearing fragment "
         "workloads; do not recover them as fused_dataflow";

  Array<Any> kernels;
  if (segment_kinds.empty()) {
    Map<String, Any> kernel;
    kernel.Set("name", String("main"));
    kernel.Set("kind", String("fused_dataflow"));
    kernel.Set("core_type", String("brisc"));
    kernels.push_back(kernel);
  } else {
    for (const std::string& kind : segment_kinds) {
      Map<String, Any> kernel;
      kernel.Set("name", String(kind));
      kernel.Set("kind", String(kind));
      kernel.Set("core_type", String(CoreTypeForSegmentKind(kind)));
      kernels.push_back(kernel);
    }
  }
  segment_plan_ = kernels;
}

void PlanTTKernelABI::StoreAccessorDescriptors(PrimFunc& func) {
  auto make_compute_config_from_gemm_state = [&]() -> TTKernelComputeConfig {
    return TTKernelComputeConfig(
        String("HiFi4"), true, gemm_dst_full_sync_en_, false, Array<String>{},
        gemm_bfp8_pack_precise_, EncodeTTKernelDefines(gemm_defines_),
        EncodeTTKernelNamedCompileArgs(gemm_named_compile_args_), gemm_clear_accum_,
        gemm_k_pack_, gemm_wg_wait_, gemm_policy_type_,
        String(GemmWarpPolicyTypeToStringForBlackhole(gemm_policy_type_)));
  };

  auto make_launch_spec = [](const std::string& core_type) -> TTKernelLaunchSpec {
    if (core_type == "brisc") {
      return MakeLaunchSpec(core_type, "riscv_0", "riscv_0_default");
    }
    if (core_type == "ncrisc") {
      return MakeLaunchSpec(core_type, "riscv_1", "riscv_1_default");
    }
    if (core_type == "trisc") {
      return MakeLaunchSpec(core_type, "", "");
    }
    return TTKernelLaunchSpec(String(""), String(""), String(""));
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
      ICHECK(accessor.Get("compile_time_arg_offset"))
          << "PlanTTKernelABI requires accessor compile_time_arg_offset for " << buffer;
      ICHECK(accessor.Get("compile_time_arg_count"))
          << "PlanTTKernelABI requires accessor compile_time_arg_count for " << buffer;
      ICHECK(accessor.Get("layout")) << "PlanTTKernelABI requires accessor layout for " << buffer;
      ICHECK(accessor.Get("memory_space"))
          << "PlanTTKernelABI requires accessor memory_space for " << buffer;
      ICHECK(accessor.Get("args_config_bits"))
          << "PlanTTKernelABI requires accessor args_config_bits for " << buffer;
      const int compile_time_arg_offset =
          Downcast<Integer>(accessor.Get("compile_time_arg_offset").value()).IntValue();
      const int compile_time_arg_count =
          Downcast<Integer>(accessor.Get("compile_time_arg_count").value()).IntValue();
      const std::string layout =
          static_cast<std::string>(Downcast<String>(accessor.Get("layout").value()));
      const std::string memory_space =
          static_cast<std::string>(Downcast<String>(accessor.Get("memory_space").value()));
      const int args_config_bits =
          Downcast<Integer>(accessor.Get("args_config_bits").value()).IntValue();
      const int transport_page_size =
          accessor.Get("transport_page_size")
              ? Downcast<Integer>(accessor.Get("transport_page_size").value()).IntValue()
              : 0;
      std::vector<int64_t> host_axis_order;
      if (auto axis_order_value = accessor.Get("host_axis_order")) {
        for (const Any& axis : Downcast<Array<Any>>(axis_order_value.value())) {
          host_axis_order.push_back(Downcast<Integer>(axis).IntValue());
        }
      }
      const bool transpose_2d =
          accessor.Get("transpose_2d") ? Downcast<Bool>(accessor.Get("transpose_2d").value())
                                       : false;

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
          memory_space,
          host_axis_order,
          transpose_2d));
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
    Array<TTPerWorkArgSpec> per_work_arg_specs =
        existing_specs_opt ? Downcast<Array<TTPerWorkArgSpec>>(existing_specs_opt.value())
                           : Array<TTPerWorkArgSpec>();
    const std::string copy_input_buffer_name =
        copy_input_buffer_.defined() ? BufferIdentityName(copy_input_buffer_) : copy_input_buffer_name_;
    const std::string copy_output_buffer_name =
        copy_output_buffer_.defined() ? BufferIdentityName(copy_output_buffer_)
                                      : copy_output_buffer_name_;

    auto runtime_arg_identity_for_kind = [&](const char* arg_kind) -> std::string {
      for (const Any& arg_item : runtime_args) {
        auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (!arg.Get("kind") ||
            static_cast<std::string>(Downcast<String>(arg.Get("kind").value())) != arg_kind) {
          continue;
        }
        if (auto identity = arg.Get("identity")) {
          return Downcast<String>(identity.value());
        }
        return MakeBlackholeRuntimeArgIdentity(arg_kind, arg_kind);
      }
      return "";
    };
    auto runtime_args_contain_kind = [&](const char* arg_kind) {
      return !runtime_arg_identity_for_kind(arg_kind).empty();
    };
    auto upsert_spec = [&](const TTPerWorkArgSpec& spec) {
      const std::string arg_identity = static_cast<std::string>(spec->arg_identity);
      for (int i = 0; i < per_work_arg_specs.size(); ++i) {
        const TTPerWorkArgSpec& existing = per_work_arg_specs[i];
        const std::string existing_arg_identity = static_cast<std::string>(existing->arg_identity);
        if (existing_arg_identity == arg_identity) {
          per_work_arg_specs.Set(i, spec);
          return;
        }
      }
      per_work_arg_specs.push_back(spec);
    };

    if (kind == "fused_dataflow") {
      if (runtime_args_contain_kind("a_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_start_id", runtime_arg_identity_for_kind("a_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            copy_input_buffer_name));
      }
      if (runtime_args_contain_kind("a_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_num_tiles", runtime_arg_identity_for_kind("a_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_input_buffer_name, 1));
      }
      if (runtime_args_contain_kind("a_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_stride", runtime_arg_identity_for_kind("a_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_input_buffer_name, 1));
      }
      if (runtime_args_contain_kind("output_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_start_id", runtime_arg_identity_for_kind("output_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            copy_output_buffer_name));
      }
      if (runtime_args_contain_kind("output_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_num_tiles", runtime_arg_identity_for_kind("output_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_output_buffer_name, 1));
      }
      if (runtime_args_contain_kind("output_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_stride", runtime_arg_identity_for_kind("output_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_output_buffer_name, 1));
      }
    }
    if (kind == "reader") {
      if (runtime_args_contain_kind("a_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_start_id", runtime_arg_identity_for_kind("a_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceLogicalBlockY,
            gemm_a_buffer_name_));
      }
      if (runtime_args_contain_kind("a_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_num_tiles", runtime_arg_identity_for_kind("a_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles,
            gemm_a_buffer_name_));
      }
      if (runtime_args_contain_kind("a_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_stride", runtime_arg_identity_for_kind("a_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant, gemm_a_buffer_name_,
            1));
      }
      if (runtime_args_contain_kind("b_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_start_id", runtime_arg_identity_for_kind("b_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceLogicalBlockX,
            gemm_b_buffer_name_));
      }
      if (runtime_args_contain_kind("b_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_num_tiles", runtime_arg_identity_for_kind("b_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles,
            gemm_b_buffer_name_));
      }
      if (runtime_args_contain_kind("b_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_stride", runtime_arg_identity_for_kind("b_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceComputeLogicalNTiles,
            gemm_b_buffer_name_));
      }
    }
    if (kind == "reader" || kind == "compute") {
      if (runtime_args_contain_kind("k_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "k_tile_start_id", runtime_arg_identity_for_kind("k_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorKTileStart,
            blackhole_runtime_arg_schema::kValueSourceConstant, "", 0));
      }
      if (runtime_args_contain_kind("num_k_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "num_k_tiles", runtime_arg_identity_for_kind("num_k_tiles"),
            blackhole_runtime_arg_schema::kDescriptorKTileCount,
            blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles));
      }
    }
    if (kind == "writer") {
      if (runtime_args_contain_kind("output_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_start_id", runtime_arg_identity_for_kind("output_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            gemm_c_buffer_name_));
      }
      if (runtime_args_contain_kind("output_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_num_tiles", runtime_arg_identity_for_kind("output_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_c_buffer_name_, 1));
      }
      if (runtime_args_contain_kind("output_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_stride", runtime_arg_identity_for_kind("output_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_c_buffer_name_, 1));
      }
    }
    return per_work_arg_specs;
  };

  if (!segment_plan_.empty()) {
    Array<Any> rewritten_segments;
    for (const auto& item : segment_plan_) {
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

          const std::string buffer_name =
              accessor.Get("buffer")
                  ? static_cast<std::string>(Downcast<String>(accessor.Get("buffer").value()))
                  : std::string();
          ICHECK(accessor.Get("compile_time_arg_offset"))
              << "PlanTTKernelABI requires accessor compile_time_arg_offset for " << buffer_name;
          ICHECK(accessor.Get("compile_time_arg_count"))
              << "PlanTTKernelABI requires accessor compile_time_arg_count for " << buffer_name;
          ICHECK(accessor.Get("common_runtime_arg_offset"))
              << "PlanTTKernelABI requires accessor common_runtime_arg_offset for " << buffer_name;
          ICHECK(accessor.Get("common_runtime_arg_count"))
              << "PlanTTKernelABI requires accessor common_runtime_arg_count for " << buffer_name;
          ICHECK(accessor.Get("layout"))
              << "PlanTTKernelABI requires accessor layout for " << buffer_name;
          ICHECK(accessor.Get("memory_space"))
              << "PlanTTKernelABI requires accessor memory_space for " << buffer_name;
          ICHECK(accessor.Get("args_config_bits"))
              << "PlanTTKernelABI requires accessor args_config_bits for " << buffer_name;
          const int compile_time_arg_offset =
              Downcast<Integer>(accessor.Get("compile_time_arg_offset").value()).IntValue();
          const int compile_time_arg_count =
              Downcast<Integer>(accessor.Get("compile_time_arg_count").value()).IntValue();
          const int common_runtime_arg_offset =
              Downcast<Integer>(accessor.Get("common_runtime_arg_offset").value()).IntValue();
          const int common_runtime_arg_count =
              Downcast<Integer>(accessor.Get("common_runtime_arg_count").value()).IntValue();
          const std::string layout =
              static_cast<std::string>(Downcast<String>(accessor.Get("layout").value()));
          const std::string memory_space =
              static_cast<std::string>(Downcast<String>(accessor.Get("memory_space").value()));
          const int args_config_bits =
              Downcast<Integer>(accessor.Get("args_config_bits").value()).IntValue();
          const int transport_page_size =
              accessor.Get("transport_page_size")
                  ? Downcast<Integer>(accessor.Get("transport_page_size").value()).IntValue()
                  : 0;
          int resolved_transport_page_size = transport_page_size;
          if (resolved_transport_page_size == 0 && accessor.Get("buffer")) {
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
      Array<Any> runtime_args = EnsureSegmentBufferRuntimeArgs(
          kind, accessors, segment.Get("runtime_args"),
          copy_input_buffer_.defined() ? BufferIdentityName(copy_input_buffer_) : copy_input_buffer_name_,
          copy_output_buffer_.defined() ? BufferIdentityName(copy_output_buffer_)
                                        : copy_output_buffer_name_);
      Array<TTPerWorkArgSpec> per_work_arg_specs = make_segment_per_work_arg_specs(
          kind, runtime_args, segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs)));
      compile_time_arg_specs = make_accessor_cta_specs(kind, accessors);
      if (kind == "compute") {
        if (compute_op_signatures_.size() == 1) {
          auto gemm_compile_time_arg_specs = make_gemm_compute_cta_specs();
          for (const auto& spec : gemm_compile_time_arg_specs) {
            compile_time_arg_specs.push_back(spec);
          }
        }
        segment.Set("compute_config", make_compute_config_from_gemm_state());
      }
      segment.Set("accessors", accessors);
      if (!runtime_args.empty()) {
        segment.Set("runtime_args", runtime_args);
      }
      if (!per_work_arg_specs.empty()) {
        segment.Set(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs), per_work_arg_specs);
      }
      segment.Set("compile_time_arg_specs", compile_time_arg_specs);
      TTKernelLaunchSpec launch_spec =
          make_launch_spec(segment.Get("core_type")
                               ? static_cast<std::string>(Downcast<String>(segment.Get("core_type").value()))
                               : std::string());
      if (!launch_spec->core_type.empty()) {
        segment.Set("launch_spec", launch_spec);
      }
      segment.Set("common_runtime_args", EncodeCommonRuntimeArgs(kind));
      rewritten_segments.push_back(segment);
    }
    segment_plan_ = rewritten_segments;
    FinalizeMaterializationPlanHostBuffers();
    String compute_kernel_name;
    for (const Any& item : segment_plan_) {
      auto segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (segment.empty()) {
        continue;
      }
      const std::string kind =
          segment.Get("kind") ? static_cast<std::string>(Downcast<String>(segment.Get("kind").value()))
                              : std::string();
      const std::string core_type =
          segment.Get("core_type")
              ? static_cast<std::string>(Downcast<String>(segment.Get("core_type").value()))
              : std::string();
      if (kind == "compute" || core_type == "trisc") {
        compute_kernel_name =
            segment.Get("name") ? Downcast<String>(segment.Get("name").value()) : String("compute");
        break;
      }
    }
    Array<TTComputeOpPlan> exact_compute_op_plans = tt_compute_op_plans_;
    tt_compute_op_plans_.clear();
    for (const TTComputeOpPlan& plan : exact_compute_op_plans) {
      tt_compute_op_plans_.push_back(plan);
    }
    if (!gemm_compute_op_facts_.empty()) {
      ICHECK(!compute_kernel_name.empty())
      << "PlanTTKernelABI produced GEMM compute op facts without a compute kernel segment";
      int64_t ordinal = 0;
      for (const auto& fact : gemm_compute_op_facts_) {
        tt_compute_op_plans_.push_back(BuildTTComputeOpPlanFromFact(
            fact, host_buffer_by_compute_operand_buffer_, compute_kernel_name,
            /*kernel_plan_index=*/-1, ordinal++));
      }
    }
    BuildTTKernelAndABISeeds(segment_plan_, &tt_kernels_, &tt_abi_plans_);
    FinalizeConsumerBindingABIIndices();
  }
}

Array<Any> PlanTTKernelABI::EncodeAccessorDescriptors(const std::string& segment_kind) const {
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

Array<Any> PlanTTKernelABI::EncodeCommonRuntimeArgs(const std::string& segment_kind) const {
  (void)segment_kind;
  return Array<Any>{};
}

// Detect matmul operation using Op comparison
bool PlanTTKernelABI::IsMatmulCall(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.tileop.gemm_py";
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
bool PlanTTKernelABI::IsClearOperation(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.clear";
}

// Detect copy operation using buffer scopes
bool PlanTTKernelABI::IsCopyOperation(const BufferStoreNode* op) const {
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    if (op->buffer.same_as(load->buffer)) {
      return false;
    }
    return GetCopyDirection(op) != CopyDirection::kUnknown;
  }
  return false;
}

// Determine copy direction
CopyDirection PlanTTKernelABI::GetCopyDirection(const BufferStoreNode* op) const {
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

  if (isAccumulatorLikeScope(src_scope) && isDRAMScope(dst_scope)) {
    const std::string src_name = BufferIdentityName(load->buffer);
    const bool has_flow_fact =
        !src_name.empty() && buffer_flow_facts_.count(src_name) != 0U;
    const bool has_materialization_fact =
        FindBufferMaterializationFact(load->buffer) != nullptr;
    const bool has_explicit_tiled_live_form =
        !src_name.empty() &&
        buffer_live_form_cb_by_buffer_identity_.count(src_name) != 0U;
    const bool has_multi_element_logical_shape = GetLogicalBufferElementCount(load->buffer) > 1;
    if (has_flow_fact || has_materialization_fact || has_explicit_tiled_live_form ||
        has_multi_element_logical_shape) {
      return CopyDirection::kCBToDram;
    }
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

PrimExpr PlanTTKernelABI::ZeroThreadAndLoopVars(const PrimExpr& expr,
                                                  const Var& loop_var) const {
  if (!loop_var.defined()) {
    return ZeroThreadAndLoopVars(expr, std::vector<Var>{});
  }
  return ZeroThreadAndLoopVars(expr, std::vector<Var>{loop_var});
}

PrimExpr PlanTTKernelABI::ZeroThreadAndLoopVars(const PrimExpr& expr,
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

bool PlanTTKernelABI::ExprUsesTransportVar(const PrimExpr& expr,
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

Var PlanTTKernelABI::SelectLogicalRowThreadVar(int64_t logical_rows) const {
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

std::pair<int, int> PlanTTKernelABI::SelectStagedCopyTransportAxes(
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

std::vector<int64_t> PlanTTKernelABI::BuildStagedCopyHostAxisOrder(
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

PrimExpr PlanTTKernelABI::InferCopyTileIndex(const BufferStoreNode* op,
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
  const Array<Integer> global_shape = GetEncodedCurrentBufferShape(global_buffer);
  const Array<Integer> shared_shape =
      GetEncodedCurrentStagedCopySharedShape(op, loop_var.defined() ? std::vector<Var>{loop_var}
                                                                    : std::vector<Var>{});
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
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, global_info.global_rows, global_info.global_cols,
      use_page_transport);
  return LinearizeStagedCopyTransportIndex(
      &analyzer, global_info.base_row, global_info.base_col, global_info.outer_slice_index,
      geometry);
}

PrimExpr PlanTTKernelABI::InferStagedCopyBaseTileIndex(
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
  const Array<Integer> global_shape = GetEncodedCurrentBufferShape(global_buffer);
  const Array<Integer> shared_shape =
      GetEncodedCurrentStagedCopySharedShape(op, loop_vars_to_zero);
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
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, segmented_gemm, transpose_b_reader,
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

const BufferStoreNode* PlanTTKernelABI::FindNestedCopyStore(
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

void PlanTTKernelABI::CollectNestedCopyStores(const Stmt& stmt,
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

void PlanTTKernelABI::RecordStagedCopyBufferBinding(const BufferStoreNode* op,
                                                      CopyDirection direction) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return;
  }
  needs_copy_runtime_args_ = true;
  if (direction == CopyDirection::kDramToCB) {
    copy_input_buffer_ = load->buffer;
    copy_input_buffer_name_ = BufferIdentityName(load->buffer);
    copy_input_shape_ = GetEncodedCurrentBufferShape(load->buffer);
    copy_intermediate_shape_ = GetEncodedCurrentBufferShape(op->buffer);
    host_buffer_by_compute_operand_buffer_[BufferIdentityName(op->buffer)] =
        BufferIdentityName(load->buffer);
  } else if (direction == CopyDirection::kCBToDram) {
    copy_output_buffer_ = op->buffer;
    copy_output_buffer_name_ = BufferIdentityName(op->buffer);
    copy_output_shape_ = GetEncodedCurrentBufferShape(op->buffer);
    copy_intermediate_shape_ = GetEncodedCurrentBufferShape(load->buffer);
    host_buffer_by_compute_operand_buffer_[BufferIdentityName(load->buffer)] =
        BufferIdentityName(op->buffer);
  }
}

std::string PlanTTKernelABI::ResolveHostBufferForComputeOperand(const Buffer& buffer) const {
  const std::string buffer_name = BufferIdentityName(buffer);
  auto it = host_buffer_by_compute_operand_buffer_.find(buffer_name);
  if (it != host_buffer_by_compute_operand_buffer_.end() && !it->second.empty()) {
    return it->second;
  }
  return "";
}

std::string PlanTTKernelABI::ComputeKernelNameForCurrentPlan() const {
  if (!current_segment_kind_.empty()) {
    return current_segment_kind_;
  }
  return requires_compute_segment_ ? std::string("compute") : std::string("main");
}

void PlanTTKernelABI::RecordExactComputeOpPlan(
    const std::string& kind, const std::string& operation_name,
    const std::vector<ComputeOperandPlanSeed>& operands) {
  if (operation_name.empty() || operands.empty()) {
    return;
  }

  Array<TTComputeOperandBindingPlan> operand_bindings;
  const Buffer* output_buffer = nullptr;
  for (const ComputeOperandPlanSeed& operand : operands) {
    if (!operand.buffer.defined()) {
      continue;
    }
    const std::string buffer_name = BufferIdentityName(operand.buffer);
    if (buffer_name.empty()) {
      continue;
    }
    if (operand.role == "output" || operand.role == "c") {
      output_buffer = &operand.buffer;
    }
    const std::string data_format = DataTypeToDataFormatForBlackhole(operand.buffer->dtype);
    operand_bindings.push_back(TTComputeOperandBindingPlan(
        String(operand.role), String(buffer_name),
        String(ResolveHostBufferForComputeOperand(operand.buffer)), String(data_format),
        String(data_format),
        String(operand.transform_kind.empty() ? "identity" : operand.transform_kind)));
  }
  if (operand_bindings.empty()) {
    return;
  }

  if (output_buffer == nullptr) {
    output_buffer = &operands.back().buffer;
  }

  Array<String> problem_shape_axes;
  Array<Integer> problem_shape;
  Array<Integer> tile_shape;
  std::string accumulator_dtype;
  if (output_buffer != nullptr && output_buffer->defined()) {
    const std::vector<int64_t> logical_shape = GetLogicalBufferShape(*output_buffer);
    for (size_t i = 0; i < logical_shape.size(); ++i) {
      if (logical_shape.size() == 1U) {
        problem_shape_axes.push_back(String("elements"));
      } else if (logical_shape.size() == 2U) {
        problem_shape_axes.push_back(String(i == 0 ? "rows" : "cols"));
      } else {
        problem_shape_axes.push_back(String("dim" + std::to_string(i)));
      }
      problem_shape.push_back(Integer(logical_shape[i]));
    }
    if (logical_shape.size() >= 2U) {
      const int64_t rows = logical_shape[logical_shape.size() - 2];
      const int64_t cols = logical_shape[logical_shape.size() - 1];
      tile_shape.push_back(Integer(std::max<int64_t>(1, CeilDivToInt(rows, kBlackholeTileRows))));
      tile_shape.push_back(Integer(std::max<int64_t>(1, CeilDivToInt(cols, kBlackholeTileCols))));
    } else if (logical_shape.size() == 1U) {
      tile_shape.push_back(Integer(std::max<int64_t>(
          1, CeilDivToInt(logical_shape[0], kBlackholeTileRows * kBlackholeTileCols))));
    }
    accumulator_dtype = DataTypeToDataFormatForBlackhole((*output_buffer)->dtype);
  }

  const std::string kernel_name = ComputeKernelNameForCurrentPlan();
  const std::string plan_name = "compute_op_" + kernel_name + "_" + operation_name + "_" +
                                std::to_string(tt_compute_op_plans_.size());
  tt_compute_op_plans_.push_back(TTComputeOpPlan(
      String(plan_name), String(kernel_name), /*kernel_plan_index=*/-1, String(kind),
      String(operation_name), Bool(true), operand_bindings, problem_shape_axes, problem_shape,
      tile_shape, Array<Integer>{}, Array<Integer>{}, String(accumulator_dtype),
      String(""), String(""), Array<String>{}));
}

void PlanTTKernelABI::RecordDramToDramCopy(const BufferStoreNode* op) {
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
  copy_input_shape_ = GetEncodedCurrentBufferShape(load->buffer);
  copy_output_shape_ = GetEncodedCurrentBufferShape(op->buffer);
}

void PlanTTKernelABI::RegisterAccessor(const std::string& segment_kind,
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

std::string PlanTTKernelABI::ResolveAccessorSegmentKind(CopyDirection direction) const {
  if (!current_segment_kind_.empty()) {
    return current_segment_kind_;
  }
  if (requires_compute_segment_) {
    if (direction == CopyDirection::kDramToCB) {
      return "reader";
    }
    if (direction == CopyDirection::kCBToDram) {
      return "writer";
    }
    return "fused_dataflow";
  }
  if (direction == CopyDirection::kDramToCB) {
    return !gemm_a_buffer_name_.empty() ? "reader" : "fused_dataflow";
  }
  if (direction == CopyDirection::kCBToDram || direction == CopyDirection::kLocalToCB) {
    return !gemm_a_buffer_name_.empty() ? "writer" : "fused_dataflow";
  }
  return "fused_dataflow";
}

int PlanTTKernelABI::GetOrAllocateSegmentAccessorSlot(
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

int PlanTTKernelABI::GetReadAccessorSlot(const std::string& segment_kind, const Buffer& buffer,
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

int PlanTTKernelABI::GetWriteAccessorSlot(const std::string& segment_kind, const Buffer& buffer,
                                            CopyDirection direction) {
  if (segment_kind == "fused_dataflow") {
    const bool has_input_transport =
        copy_input_buffer_.defined() || !copy_input_buffer_name_.empty();
    if ((copy_output_buffer_.defined() && SameBufferIdentity(buffer, copy_output_buffer_)) ||
        (!copy_output_buffer_name_.empty() &&
         BufferIdentityName(buffer) == copy_output_buffer_name_)) {
      return has_input_transport ? 2 : 0;
    }
    return 0;
  }
  if (direction == CopyDirection::kCBToDram) {
    return GetOrAllocateSegmentAccessorSlot(&write_accessor_slots_, segment_kind, buffer);
  }
  return 0;
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
    const bool needs_accumulator_merge = FindBufferMaterializationFact(out_buffer) != nullptr;
    if (!gemm_clear_accum_ && !needs_accumulator_merge) {
      gemm_clear_accum_ = true;
    }
    const FutureBufferUses future_uses =
        ClassifyFutureBufferUses(out_buffer, current_order_index);
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
  const bool merge_with_zero_reload = !gemm_clear_accum_ && HasZeroFragmentFillFact(gemm_c_buffer_);
  InvalidateLastFragmentFillValue(gemm_c_buffer_);
  if (!gemm_clear_accum_) {
    return MaybeWrapComputeSegment(
        GenerateAccumulatingMatmulSequence(op, retain_in0, retain_in1, publish_transport_out,
                                           preserve_out_local_state, reacquire_in0, reacquire_in1,
                                           post_merge_cast, post_merge_cast_order_index,
                                           merge_with_zero_reload));
  }
  const bool publish_live_form_cb =
      preserve_out_local_state && BufferUsesTiledCBLiveForm(gemm_c_buffer_);
  if (publish_live_form_cb) {
    buffer_live_form_cb_by_buffer_identity_[BufferIdentityName(gemm_c_buffer_)] =
        gemm_c_req_index_;
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

Buffer PlanTTKernelABI::CreateEphemeralBufferLike(const Buffer& buffer,
                                                  const std::string& suffix) const {
  const std::string name =
      BufferIdentityName(buffer) + "_" + suffix + "_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(buffer->shape, buffer->dtype, name, GetStorageScope(buffer));
}

Buffer PlanTTKernelABI::CreateConstantTileBuffer(DataType dtype, const std::string& suffix) const {
  Array<PrimExpr> tile_shape{IntImm32(kBlackholeTileRows), IntImm32(kBlackholeTileCols)};
  const std::string name = "exact_const_tile_" + suffix + "_" + std::to_string(next_requirement_index_);
  return tir::decl_buffer(tile_shape, dtype, name, "local.fragment");
}

int PlanTTKernelABI::PrepareExactTiledCBRequirement(const Buffer& buffer) {
  const int cb_id = AllocateRequirementIndex(buffer, CBType::kIntermediate);
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const int num_tiles = GetLogicalBufferTileCount(buffer);
  const int tile_bytes = kBlackholeTileRows * kBlackholeTileCols * buffer->dtype.bytes();
  SetRequirementPageLayout(cb_id, tile_bytes, num_tiles);
  auto& req = cb_requirements_.at(cb_id);
  req.data_format = DataTypeToDataFormatForBlackhole(buffer->dtype);
  req.publish_pages_per_event = std::max(req.publish_pages_per_event, num_tiles);
  req.consume_pages_per_event = std::max(req.consume_pages_per_event, num_tiles);
  return cb_id;
}

Stmt PlanTTKernelABI::FillLocalTileBuffer(const Buffer& buffer, const PrimExpr& value) {
  return MakeBlackholeCall(
      tir::builtin::blackhole_fill_fragment(),
      {buffer->data, IntImm32(kBlackholeTileRows * kBlackholeTileCols), value});
}

Stmt PlanTTKernelABI::PublishLocalBufferToExactTiledCB(const Buffer& src,
                                                       const ExactTiledCBValue& cb_value) {
  ICHECK(cb_value.cb_id >= 0);
  const Buffer physical_src = ResolvePhysicalComputeBuffer(src);
  Array<Stmt> stmts{
      MakeBlackholeCall(blackhole_cb_reserve_back(),
                        {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}),
      MakeBlackholeCall(blackhole_write_local_fragment_slice_to_tiled_cb(),
                        {physical_src->data, IntImm32(cb_value.cb_id), IntImm32(0),
                         IntImm32(static_cast<int>(cb_value.num_elements)),
                         IntImm32(static_cast<int>(cb_value.row_width))}),
      MakeBlackholeCall(blackhole_cb_push_back(),
                        {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}),
  };
  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::MaterializeExactTiledCBToLocalBuffer(const Buffer& dst,
                                                           const ExactTiledCBValue& cb_value,
                                                           bool pop_front) {
  ICHECK(cb_value.cb_id >= 0);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  Array<Stmt> stmts;
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  constexpr int kTileElements = kBlackholeTileRows * kBlackholeTileCols;
  for (int tile = 0; tile < cb_value.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(
        blackhole_read_cb_front_tile_to_local_fragment(),
        {physical_dst->data, IntImm32(cb_value.cb_id), IntImm32(tile), IntImm32(tile * kTileElements)}));
  }
  if (pop_front) {
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(cb_value.cb_id), IntImm32(cb_value.num_tiles)}));
  }
  return SeqStmt::Flatten(stmts);
}

PlanTTKernelABI::ExactTiledCBValue PlanTTKernelABI::CreatePublishedExactTiledCBValue(
    const Buffer& src, const std::string& suffix) {
  ExactTiledCBValue value;
  value.buffer = CreateEphemeralBufferLike(src, suffix);
  value.num_elements = GetLogicalBufferElementCount(src);
  value.num_tiles = GetLogicalBufferTileCount(src);
  value.row_width = GetLogicalMatrixShape(src).second;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(src)) {
    auto shape_it = spec->find(String(schema_key::kShape));
    if (shape_it != spec->end()) {
      std::vector<int64_t> shape;
      for (const Integer& dim : Downcast<Array<Integer>>((*shape_it).second)) {
        shape.push_back(dim->value);
      }
      if (shape.size() >= 2U) {
        value.num_elements = ComputeStaticElementCount(shape);
        value.row_width = shape.back();
        value.num_tiles = std::max(1, CeilDivToInt(shape[shape.size() - 2], kBlackholeTileRows) *
                                          CeilDivToInt(shape.back(), kBlackholeTileCols));
      } else if (shape.size() == 1U) {
        value.num_elements = shape.front();
        value.row_width = 1;
        value.num_tiles = std::max(1, CeilDivToInt(shape.front(), kBlackholeTileRows));
      }
    }
  }
  if (value.row_width <= 0) {
    value.row_width = kBlackholeTileCols;
  }
  value.cb_id = PrepareExactTiledCBRequirement(value.buffer);
  SetRequirementPageLayout(value.cb_id,
                           kBlackholeTileRows * kBlackholeTileCols * src->dtype.bytes(),
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
                                                           bool merge_with_zero_reload) {
  InvalidateLastFragmentFillValue(dst);
  const std::string dst_buffer_name = BufferIdentityName(dst);
  const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(dst);
  ICHECK(fact != nullptr)
      << "PlanTTKernelABI requires buffer materialization fact for "
         "merge_fragment_tiles destination "
      << dst_buffer_name;
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
  ICHECK_GT(gemm_c_dtype_.bytes(), 0)
      << "Blackhole accumulator-merge lowering requires a valid destination dtype for "
      << dst_buffer_name;
  const int tile_elements = (kBlackholeTileRows * kBlackholeTileCols * gemm_c_dtype_.bytes()) /
                            gemm_c_dtype_.bytes();
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(dst);
  if (!merge_with_zero_reload) {
    RecordExactComputeOpPlan("binary", "add_tiles",
                             {{"lhs", reload_buffer, "identity"},
                              {"rhs", partials_buffer, "identity"},
                              {"output", dst, "identity"}});
  }

  std::vector<Stmt> stmts;
  if (!merge_with_zero_reload) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                      {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
    for (int tile = 0; tile < num_c_tiles; ++tile) {
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_local_fragment_tile_to_cb(),
          {physical_dst->data, IntImm32(reload_cb_id), IntImm32(tile),
           IntImm32(tile * tile_elements)}));
    }
    stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                      {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
    stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                      {IntImm32(reload_cb_id), IntImm32(partials_cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_reconfig_data_format(),
                                      {IntImm32(reload_cb_id), IntImm32(partials_cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_cb_wait_front(),
                                      {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
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
      stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                        {IntImm32(reload_cb_id), IntImm32(partials_cb_id),
                                         IntImm32(tile), IntImm32(tile), IntImm32(0)}));
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
  if (!merge_with_zero_reload) {
    stmts.push_back(MakeBlackholeCall(blackhole_cb_pop_front(),
                                      {IntImm32(reload_cb_id), IntImm32(num_c_tiles)}));
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
      stmts.push_back(MakeBlackholeCall(blackhole_read_cb_front_tile_to_local_fragment(),
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
  const FutureBufferUses dst_future_uses = ClassifyFutureBufferUses(match.dst, cast_order_index);
  if (!dst_future_uses.has_transport_consume || dst_future_uses.has_compute_consume ||
      dst_future_uses.has_reference) {
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
  const std::string identity = BufferIdentityName(buffer);
  if (!identity.empty()) {
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

  Buffer reload_buffer = CreateFragmentMergeReloadBuffer(gemm_c_buffer_);
  const int reload_req_index = AllocateRequirementIndex(reload_buffer, CBType::kIntermediate);
  SetRequirementPageLayout(reload_req_index, c_tile_bytes, num_c_tiles);
  auto& reload_req = cb_requirements_.at(reload_req_index);
  reload_req.data_format = DataTypeToDataFormat(gemm_c_dtype_);
  reload_req.flow_class = CBFlowClass::kStream;
  reload_req.publish_pages_per_event =
      std::max(reload_req.publish_pages_per_event, num_c_tiles);
  reload_req.consume_pages_per_event =
      std::max(reload_req.consume_pages_per_event, num_c_tiles);
  MarkRequirementLifetimeOverlap(scratch_req_index, reload_req_index);

  Buffer live_form_buffer;
  int live_form_req_index = -1;
  bool materialize_live_form_to_local_state = false;
  const bool use_tiled_cb_live_form =
      preserve_out_local_state && BufferUsesTiledCBLiveForm(gemm_c_buffer_);
  if (preserve_out_local_state) {
    const std::string live_form_name =
        BufferIdentityName(gemm_c_buffer_) + "_fragment_merge_live_form_" +
        std::to_string(next_requirement_index_);
    live_form_buffer =
        tir::decl_buffer(gemm_c_buffer_->shape, gemm_c_buffer_->dtype, live_form_name,
                         GetStorageScope(gemm_c_buffer_));
    live_form_req_index = AllocateRequirementIndex(live_form_buffer, CBType::kIntermediate);
    SetRequirementPageLayout(live_form_req_index, c_tile_bytes, num_c_tiles);
    auto& live_form_req = cb_requirements_.at(live_form_req_index);
    live_form_req.data_format = DataTypeToDataFormat(gemm_c_dtype_);
    live_form_req.flow_class = CBFlowClass::kStream;
    live_form_req.publish_pages_per_event =
        std::max(live_form_req.publish_pages_per_event, num_c_tiles);
    live_form_req.consume_pages_per_event =
        std::max(live_form_req.consume_pages_per_event, num_c_tiles);
    MarkRequirementLifetimeOverlap(scratch_req_index, live_form_req_index);
    MarkRequirementLifetimeOverlap(reload_req_index, live_form_req_index);
    materialize_live_form_to_local_state = !use_tiled_cb_live_form;
  }

  int materialized_cast_req_index = -1;
  if (post_merge_cast != nullptr &&
      CanPublishPostMergeCastWithPackTile(*post_merge_cast, post_merge_cast_order_index)) {
    materialized_cast_req_index = PreparePostMergeCastPublishCB(*post_merge_cast, num_c_tiles);
    MarkRequirementLifetimeOverlap(scratch_req_index, materialized_cast_req_index);
    MarkRequirementLifetimeOverlap(reload_req_index, materialized_cast_req_index);
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
    buffer_live_form_cb_by_buffer_identity_[BufferIdentityName(post_merge_cast->dst)] =
        materialized_cast_req_index;
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
      merge_with_zero_reload));
  if (use_tiled_cb_live_form) {
    buffer_live_form_cb_by_buffer_identity_[BufferIdentityName(gemm_c_buffer_)] =
        live_form_req_index;
  }

  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::GenerateCopySequence(const BufferStoreNode* op) {
  CopyDirection direction = GetCopyDirection(op);

  ICHECK(direction != CopyDirection::kUnknown)
      << "PlanTTKernelABI copy lowering requires an explicit copy-direction classification";

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

Stmt PlanTTKernelABI::GenerateCopySequence(const BufferStoreNode* op,
                                             const PrimExpr& tile_index) {
  CopyDirection direction = GetCopyDirection(op);
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return GetRef<Stmt>(op);
  }

  std::vector<Stmt> stmts;
  auto maybe_wrap_segment_stmt = [&](const std::string& segment_kind, Stmt stmt) -> Stmt {
    return WrapSegmentStmtIfNeeded(current_segment_kind_, segment_kind, stmt);
  };
  switch (direction) {
    case CopyDirection::kDramToCB: {
      const std::string segment_kind = ResolveAccessorSegmentKind(direction);
      const bool segmented_gemm = !gemm_a_buffer_name_.empty() && segment_kind == "reader";
      int cb_id = AllocateRequirementIndex(
          op->buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
      int tile_bytes = EstimateCopyPageSize(op->buffer);
      RecordStagedCopyBufferBinding(op, direction);
      const Array<PrimExpr>& global_indices = load->indices;
      const Array<Integer> global_shape = GetEncodedCurrentBufferShape(load->buffer);
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
      const bool segmented_gemm = !gemm_a_buffer_name_.empty() && segment_kind == "writer";
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
      const Array<Integer> global_shape = GetEncodedCurrentBufferShape(op->buffer);
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

Stmt PlanTTKernelABI::GenerateStagedCopyLoopSequence(
    const BufferStoreNode* op, const PrimExpr& base_tile_index,
    const std::vector<Var>& loop_vars_to_zero) {
  CopyDirection direction = GetCopyDirection(op);
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return GetRef<Stmt>(op);
  }

  const std::string segment_kind = ResolveAccessorSegmentKind(direction);
  const bool segmented_gemm =
      !gemm_a_buffer_name_.empty() && (segment_kind == "reader" || segment_kind == "writer");
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
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  const Array<Integer> shared_shape =
      GetEncodedCurrentStagedCopySharedShape(op, loop_vars_to_zero);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);

  const bool use_page_transport = UseStagedCopyPageTransport(shared_buffer);
  const Buffer& global_buffer = direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;
  const Array<Integer> global_shape = GetEncodedCurrentBufferShape(global_buffer);
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
    return WrapSegmentStmtIfNeeded(current_segment_kind_, segment_kind, stmt);
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
          << "PlanTTKernelABI segmented reader transport exceeds staged copy shape for buffer "
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

Stmt PlanTTKernelABI::GenerateFusedStagedCopySequence(
    const BufferStoreNode* dram_to_cb, const BufferStoreNode* cb_to_dram,
    const PrimExpr& base_tile_index, const std::vector<Var>& loop_vars_to_zero) {
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
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  const Array<Integer> shared_shape =
      GetEncodedCurrentStagedCopySharedShape(dram_to_cb, loop_vars_to_zero);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, /*segmented_gemm=*/false,
      /*transpose_b_reader=*/false, /*accumulator_like_src=*/false, gemm_m_, gemm_n_, gemm_k_);
  const bool use_page_transport = UseStagedCopyPageTransport(shared_buffer);
  const Array<Integer> global_shape = GetEncodedCurrentBufferShape(dram_load->buffer);
  int64_t global_rows = 0;
  int64_t global_cols = 0;
  std::tie(global_rows, global_cols) = ResolveStaticShape2DFromBufferOrMetadata(
      dram_load->buffer, global_shape,
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

Stmt PlanTTKernelABI::GenerateClearSequence(const CallNode* op) {
  // Clear operation: tile_regs_acquire() to zero DST registers
  // In full implementation, would also zero-fill
  return MaybeWrapComputeSegment(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
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

bool PlanTTKernelABI::MatchDirectRowReduction(const ForNode* op, RowReductionMatch* match) const {
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

bool PlanTTKernelABI::MatchAllocatedRowReduction(const AllocateNode* op,
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

bool PlanTTKernelABI::MatchGroupedRowReduction(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateRowReductionSequence(const RowReductionMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue src_in = CreatePublishedExactTiledCBValue(match.src, "reduce_src");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "reduce_out");
  ExactTiledCBValue scaler = CreateConstantExactTiledCBValue(match.src->dtype, "reduce_scaler");
  RecordExactComputeOpPlan("reduce", "reduce_tile",
                           {{"input", match.src, "identity"},
                            {"scaler", scaler.buffer, "identity"},
                            {"output", match.dst, "identity"}});

  const Buffer scaler_local = scaler.buffer;
  const Stmt scaler_fill = FillLocalTileBuffer(scaler_local, make_const(match.src->dtype, 1.0));
  const Stmt scaler_publish = PublishLocalBufferToExactTiledCB(scaler_local, scaler);

  const int tiles_per_reduction = std::max(1, src_in.num_tiles / std::max(1, out.num_tiles));
  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.src, src_in));
  stmts.push_back(scaler_fill);
  stmts.push_back(scaler_publish);
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scaler.cb_id), IntImm32(scaler.num_tiles)}));
  for (int out_tile = 0; out_tile < out.num_tiles; ++out_tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_reduce_init(),
        {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(out.cb_id),
         StringImm(match.kind), StringImm("row")}));
    for (int tile = 0; tile < tiles_per_reduction; ++tile) {
      const int src_tile = out_tile * tiles_per_reduction + tile;
      stmts.push_back(MakeBlackholeCall(
          blackhole_reduce_tile(),
          {IntImm32(src_in.cb_id), IntImm32(scaler.cb_id), IntImm32(src_tile), IntImm32(0),
           IntImm32(0), StringImm(match.kind), StringImm("row")}));
    }
    stmts.push_back(
        MakeBlackholeCall(blackhole_reduce_uninit(), {StringImm(match.kind), StringImm("row")}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(out.cb_id), IntImm32(out_tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scaler.cb_id), IntImm32(scaler.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(scaler_local, body);
  body = tir::Allocate(scaler_local->data, scaler_local->dtype, scaler_local->shape, Bool(1), body);
  return MaybeWrapComputeSegment(body);
}

bool PlanTTKernelABI::MatchDirectRowBroadcast(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateRowBroadcastSequence(const RowBroadcastMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue dst_in = CreatePublishedExactTiledCBValue(match.dst, "row_bcast_dst");
  ExactTiledCBValue scalar_in = CreatePublishedExactTiledCBValue(match.scalar, "row_bcast_scalar");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "row_bcast_out");

  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.dst, dst_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.scalar, scalar_in));

  ExactTiledCBValue* scalar_operand = &scalar_in;
  ExactTiledCBValue reciprocal;
  if (match.kind == "div") {
    reciprocal = CreateEmptyExactTiledCBValue(match.scalar, "row_bcast_recip");
    RecordExactComputeOpPlan("unary", "recip_tile",
                             {{"input", match.scalar, "identity"},
                              {"output", reciprocal.buffer, "identity"}});
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(reciprocal.cb_id), IntImm32(reciprocal.num_tiles)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scalar_in.cb_id), IntImm32(scalar_in.num_tiles)}));
    for (int tile = 0; tile < reciprocal.num_tiles; ++tile) {
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(), {IntImm32(scalar_in.cb_id)}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_copy_tile(), {IntImm32(scalar_in.cb_id), IntImm32(tile), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_recip_tile_init(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_recip_tile(), {IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
      stmts.push_back(
          MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(reciprocal.cb_id)}));
      stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                        {IntImm32(0), IntImm32(reciprocal.cb_id), IntImm32(tile)}));
      stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
    }
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scalar_in.cb_id), IntImm32(scalar_in.num_tiles)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(reciprocal.cb_id), IntImm32(reciprocal.num_tiles)}));
    scalar_operand = &reciprocal;
  }

  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(match.dst);
  const int tiles_per_row =
      logical_rows > 0 && logical_cols > 0 ? std::max(1, CeilDivToInt(logical_cols, kBlackholeTileCols))
                                           : std::max(1, dst_in.num_tiles / std::max(1, scalar_operand->num_tiles));
  RecordExactComputeOpPlan("binary", "mul_tiles_bcast_rows",
                           {{"lhs", match.dst, "identity"},
                            {"rhs", scalar_operand->buffer, "broadcast"},
                            {"output", match.dst, "identity"}});

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scalar_operand->cb_id), IntImm32(scalar_operand->num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_bcast_rows_init_short(),
                                    {IntImm32(dst_in.cb_id), IntImm32(scalar_operand->cb_id)}));
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    const int rhs_tile = tile / tiles_per_row;
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_bcast_rows(),
                                      {IntImm32(dst_in.cb_id), IntImm32(scalar_operand->cb_id),
                                       IntImm32(tile), IntImm32(rhs_tile), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  if (match.kind == "mul") {
    stmts.push_back(
        MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scalar_in.cb_id), IntImm32(scalar_in.num_tiles)}));
  } else {
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(), {IntImm32(reciprocal.cb_id), IntImm32(reciprocal.num_tiles)}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::MatchScalarFmaStore(const BufferStoreNode* op,
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

bool PlanTTKernelABI::MatchGroupedScalarFmaLoop(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateScalarFmaSequence(const ScalarFmaMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue lhs_in = CreatePublishedExactTiledCBValue(match.lhs, "scalar_fma_lhs");
  ExactTiledCBValue rhs_in = CreatePublishedExactTiledCBValue(match.rhs, "scalar_fma_rhs");
  ExactTiledCBValue add_in = CreatePublishedExactTiledCBValue(match.add, "scalar_fma_add");
  ExactTiledCBValue product = CreateEmptyExactTiledCBValue(match.dst, "scalar_fma_product");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "scalar_fma_out");
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", match.lhs, "identity"},
                            {"rhs", match.rhs, "identity"},
                            {"output", product.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "add_tiles",
                           {{"lhs", product.buffer, "identity"},
                            {"rhs", match.add, "identity"},
                            {"output", match.dst, "identity"}});

  ICHECK_EQ(lhs_in.num_tiles, rhs_in.num_tiles);
  ICHECK_EQ(lhs_in.num_tiles, add_in.num_tiles);
  ICHECK_EQ(lhs_in.num_tiles, out.num_tiles);

  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.lhs, lhs_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.rhs, rhs_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.add, add_in));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(product.cb_id), IntImm32(product.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id)}));
  for (int tile = 0; tile < product.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(lhs_in.cb_id), IntImm32(rhs_in.cb_id),
                                       IntImm32(tile), IntImm32(tile), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(product.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(product.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(product.cb_id), IntImm32(product.num_tiles)}));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(product.cb_id), IntImm32(product.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(add_in.cb_id), IntImm32(add_in.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                    {IntImm32(product.cb_id), IntImm32(add_in.cb_id)}));
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                      {IntImm32(product.cb_id), IntImm32(add_in.cb_id),
                                       IntImm32(tile), IntImm32(tile), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(product.cb_id), IntImm32(product.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(add_in.cb_id), IntImm32(add_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::MatchExp2RowBroadcastAffine(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateExp2RowBroadcastAffineSequence(
    const Exp2RowBroadcastAffineMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue dst_in = CreatePublishedExactTiledCBValue(match.dst, "exp2_affine_dst");
  ExactTiledCBValue scalar_in = CreatePublishedExactTiledCBValue(match.scalar, "exp2_affine_scalar");
  ExactTiledCBValue dst_scale = CreateConstantExactTiledCBValue(match.dst->dtype, "exp2_affine_dst_scale");
  ExactTiledCBValue scalar_scale =
      CreateConstantExactTiledCBValue(match.scalar->dtype, "exp2_affine_scalar_scale");
  ExactTiledCBValue scaled_dst = CreateEmptyExactTiledCBValue(match.dst, "exp2_affine_scaled_dst");
  ExactTiledCBValue scaled_scalar =
      CreateEmptyExactTiledCBValue(match.scalar, "exp2_affine_scaled_scalar");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "exp2_affine_out");
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", match.dst, "identity"},
                            {"rhs", dst_scale.buffer, "identity"},
                            {"output", scaled_dst.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", match.scalar, "identity"},
                            {"rhs", scalar_scale.buffer, "identity"},
                            {"output", scaled_scalar.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "add_tiles_bcast_rows",
                           {{"lhs", scaled_dst.buffer, "identity"},
                            {"rhs", scaled_scalar.buffer, "broadcast"},
                            {"output", out.buffer, "identity"}});
  RecordExactComputeOpPlan("unary", "exp2_tile",
                           {{"input", out.buffer, "identity"},
                            {"output", match.dst, "identity"}});

  const auto [logical_rows, logical_cols] = GetLogicalMatrixShape(match.dst);
  const int tiles_per_row =
      logical_rows > 0 && logical_cols > 0 ? std::max(1, CeilDivToInt(logical_cols, kBlackholeTileCols))
                                           : std::max(1, dst_in.num_tiles / std::max(1, scalar_in.num_tiles));

  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.dst, dst_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.scalar, scalar_in));
  stmts.push_back(FillLocalTileBuffer(dst_scale.buffer, match.dst_scale));
  stmts.push_back(PublishLocalBufferToExactTiledCB(dst_scale.buffer, dst_scale));
  stmts.push_back(FillLocalTileBuffer(scalar_scale.buffer, match.scalar_scale));
  stmts.push_back(PublishLocalBufferToExactTiledCB(scalar_scale.buffer, scalar_scale));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(scaled_dst.cb_id), IntImm32(scaled_dst.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(dst_scale.cb_id), IntImm32(dst_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(dst_in.cb_id), IntImm32(dst_scale.cb_id)}));
  for (int tile = 0; tile < scaled_dst.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(dst_in.cb_id), IntImm32(dst_scale.cb_id),
                                       IntImm32(tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(scaled_dst.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(scaled_dst.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(dst_scale.cb_id), IntImm32(dst_scale.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(scaled_dst.cb_id), IntImm32(scaled_dst.num_tiles)}));

  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(scaled_scalar.cb_id), IntImm32(scaled_scalar.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scalar_in.cb_id), IntImm32(scalar_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scalar_scale.cb_id), IntImm32(scalar_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(scalar_in.cb_id), IntImm32(scalar_scale.cb_id)}));
  for (int tile = 0; tile < scaled_scalar.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(scalar_in.cb_id), IntImm32(scalar_scale.cb_id),
                                       IntImm32(tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(scaled_scalar.cb_id)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_pack_tile(), {IntImm32(0), IntImm32(scaled_scalar.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scalar_in.cb_id), IntImm32(scalar_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(scalar_scale.cb_id), IntImm32(scalar_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_push_back(), {IntImm32(scaled_scalar.cb_id), IntImm32(scaled_scalar.num_tiles)}));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scaled_dst.cb_id), IntImm32(scaled_dst.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_wait_front(), {IntImm32(scaled_scalar.cb_id), IntImm32(scaled_scalar.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_add_bcast_rows_init_short(),
                                    {IntImm32(scaled_dst.cb_id), IntImm32(scaled_scalar.cb_id)}));
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    const int rhs_tile = tile / tiles_per_row;
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_bcast_rows(),
                                      {IntImm32(scaled_dst.cb_id), IntImm32(scaled_scalar.cb_id),
                                       IntImm32(tile), IntImm32(rhs_tile), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile_init(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile(), {IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_pop_front(), {IntImm32(scaled_dst.cb_id), IntImm32(scaled_dst.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_pop_front(), {IntImm32(scaled_scalar.cb_id), IntImm32(scaled_scalar.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(dst_scale.buffer, body);
  body = tir::DeclBuffer(scalar_scale.buffer, body);
  body = tir::Allocate(scalar_scale.buffer->data, scalar_scale.buffer->dtype,
                       scalar_scale.buffer->shape, Bool(1), body);
  body = tir::Allocate(dst_scale.buffer->data, dst_scale.buffer->dtype, dst_scale.buffer->shape,
                       Bool(1), body);
  return MaybeWrapComputeSegment(body);
}

bool PlanTTKernelABI::MatchScalarExp2AffineStore(const BufferStoreNode* op,
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

bool PlanTTKernelABI::MatchGroupedScalarExp2AffineLoop(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateScalarExp2AffineSequence(const ScalarExp2AffineMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue lhs_in = CreatePublishedExactTiledCBValue(match.lhs, "scalar_exp2_lhs");
  ExactTiledCBValue rhs_in = CreatePublishedExactTiledCBValue(match.rhs, "scalar_exp2_rhs");
  ExactTiledCBValue lhs_scale = CreateConstantExactTiledCBValue(match.lhs->dtype, "scalar_exp2_lhs_scale");
  ExactTiledCBValue rhs_scale = CreateConstantExactTiledCBValue(match.rhs->dtype, "scalar_exp2_rhs_scale");
  ExactTiledCBValue scaled_lhs =
      CreateEmptyExactTiledCBValue(match.dst, "scalar_exp2_scaled_lhs");
  ExactTiledCBValue scaled_rhs =
      CreateEmptyExactTiledCBValue(match.dst, "scalar_exp2_scaled_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "scalar_exp2_out");
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", match.lhs, "identity"},
                            {"rhs", lhs_scale.buffer, "identity"},
                            {"output", scaled_lhs.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "mul_tiles",
                           {{"lhs", match.rhs, "identity"},
                            {"rhs", rhs_scale.buffer, "identity"},
                            {"output", scaled_rhs.buffer, "identity"}});
  RecordExactComputeOpPlan("binary", "add_tiles",
                           {{"lhs", scaled_lhs.buffer, "identity"},
                            {"rhs", scaled_rhs.buffer, "identity"},
                            {"output", out.buffer, "identity"}});
  RecordExactComputeOpPlan("unary", "exp2_tile",
                           {{"input", out.buffer, "identity"},
                            {"output", match.dst, "identity"}});

  ICHECK_EQ(lhs_in.num_tiles, rhs_in.num_tiles);
  ICHECK_EQ(lhs_in.num_tiles, out.num_tiles);

  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.lhs, lhs_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.rhs, rhs_in));
  stmts.push_back(FillLocalTileBuffer(lhs_scale.buffer, match.lhs_scale));
  stmts.push_back(PublishLocalBufferToExactTiledCB(lhs_scale.buffer, lhs_scale));
  stmts.push_back(FillLocalTileBuffer(rhs_scale.buffer, match.rhs_scale));
  stmts.push_back(PublishLocalBufferToExactTiledCB(rhs_scale.buffer, rhs_scale));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(lhs_scale.cb_id), IntImm32(lhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(lhs_in.cb_id), IntImm32(lhs_scale.cb_id)}));
  for (int tile = 0; tile < scaled_lhs.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(lhs_in.cb_id), IntImm32(lhs_scale.cb_id),
                                       IntImm32(tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(scaled_lhs.cb_id)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_pack_tile(), {IntImm32(0), IntImm32(scaled_lhs.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(lhs_in.cb_id), IntImm32(lhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(lhs_scale.cb_id), IntImm32(lhs_scale.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_lhs.num_tiles)}));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(scaled_rhs.cb_id), IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(rhs_scale.cb_id), IntImm32(rhs_scale.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles_init(),
                                    {IntImm32(rhs_in.cb_id), IntImm32(rhs_scale.cb_id)}));
  for (int tile = 0; tile < scaled_rhs.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_mul_tiles(),
                                      {IntImm32(rhs_in.cb_id), IntImm32(rhs_scale.cb_id),
                                       IntImm32(tile), IntImm32(0), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(scaled_rhs.cb_id)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_pack_tile(), {IntImm32(0), IntImm32(scaled_rhs.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(rhs_in.cb_id), IntImm32(rhs_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(rhs_scale.cb_id), IntImm32(rhs_scale.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(scaled_rhs.cb_id), IntImm32(scaled_rhs.num_tiles)}));

  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(scaled_rhs.cb_id), IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(blackhole_add_tiles_init(),
                                    {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_rhs.cb_id)}));
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_add_tiles(),
                                      {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_rhs.cb_id),
                                       IntImm32(tile), IntImm32(tile), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile_init(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_exp2_tile(), {IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_tile(), {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_pop_front(), {IntImm32(scaled_lhs.cb_id), IntImm32(scaled_lhs.num_tiles)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_pop_front(), {IntImm32(scaled_rhs.cb_id), IntImm32(scaled_rhs.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));

  Stmt body = SeqStmt::Flatten(stmts);
  body = tir::DeclBuffer(lhs_scale.buffer, body);
  body = tir::DeclBuffer(rhs_scale.buffer, body);
  body = tir::Allocate(rhs_scale.buffer->data, rhs_scale.buffer->dtype,
                       rhs_scale.buffer->shape, Bool(1), body);
  body = tir::Allocate(lhs_scale.buffer->data, lhs_scale.buffer->dtype,
                       lhs_scale.buffer->shape, Bool(1), body);
  return MaybeWrapComputeSegment(body);
}

bool PlanTTKernelABI::MatchDirectFragmentFill(const ForNode* op,
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

bool PlanTTKernelABI::MatchScalarFragmentFillStore(const BufferStoreNode* op,
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

Stmt PlanTTKernelABI::GenerateFragmentFillSequence(const FragmentFillMatch& match) {
  PrimExpr num_elements = match.num_elements;
  int64_t physical_local_extent = 0;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(match.dst)) {
    physical_local_extent = ProductIntegerArrayField(*spec, schema_key::kLocalShape, int64_t{0});
  }
  if (physical_local_extent > 0) {
    num_elements = IntImm(DataType::Int(32), static_cast<int>(physical_local_extent));
  }
  const int64_t logical_extent = GetLogicalBufferElementCount(match.dst);
  if (physical_local_extent <= 0 && logical_extent > 1) {
    bool should_promote_extent = !num_elements.defined();
    if (const auto* int_imm = num_elements.as<IntImmNode>()) {
      should_promote_extent = int_imm->value < logical_extent;
    }
    if (should_promote_extent) {
      num_elements = IntImm(DataType::Int(32), logical_extent);
    }
  }
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(match.dst);
  last_fragment_fill_value_by_buffer_identity_[BufferIdentityName(match.dst)] = match.value;
  if (const VarNode* data = BufferDataIdentity(match.dst)) {
    last_fragment_fill_value_by_data_[data] = match.value;
  }
  return MaybeWrapComputeSegment(MakeBlackholeCall(
      tir::builtin::blackhole_fill_fragment(), {physical_dst->data, num_elements, match.value}));
}

Stmt PlanTTKernelABI::GenerateFragmentFillCastPublishSequence(
    const FragmentFillMatch& fill_match, const FragmentCastMatch& cast_match,
    int current_order_index) {
  (void)current_order_index;
  if (!fill_match.dst.defined() || !cast_match.src.defined() || !cast_match.dst.defined() ||
      !SameBufferIdentity(fill_match.dst, cast_match.src) ||
      !cast_match.dst->dtype.is_bfloat16() || !tir::is_zero(cast_match.src_offset)) {
    return Stmt();
  }
  const BlackholeBufferMaterializationFact* fact =
      FindBufferMaterializationFact(cast_match.dst);
  if (fact == nullptr) {
    return Stmt();
  }
  if (fact->kind != buffer_materialization::kRepublishedLogicalTile ||
      fact->bridge_kind != buffer_materialization::kTileNFacesMaterialization ||
      fact->execution_protocol != buffer_materialization::kTiledCBRepublish) {
    return Stmt();
  }

  PrimExpr num_elements_expr = cast_match.num_elements;
  if (fact->logical_element_count > 0) {
    num_elements_expr = IntImm(DataType::Int(32),
                               static_cast<int>(fact->logical_element_count));
  }

  PrimExpr row_width = cast_match.row_width;
  if (fact->logical_row_width > 0) {
    row_width = IntImm(DataType::Int(32), static_cast<int>(fact->logical_row_width));
  }
  auto apply_typed_layout_shape_spec = [&](const Map<String, Any>* spec) {
    if (spec == nullptr) {
      return;
    }
    const int64_t logical_element_count =
        ProductIntegerArrayField(*spec, schema_key::kShape, int64_t{0});
    if (logical_element_count > StaticIntValueOrDefault(num_elements_expr, int64_t{0})) {
      num_elements_expr = IntImm(DataType::Int(32), static_cast<int>(logical_element_count));
    }
    auto shape_it = spec->find(String(schema_key::kShape));
    if (shape_it != spec->end()) {
      const Array<Integer> logical_shape = Downcast<Array<Integer>>((*shape_it).second);
      if (logical_shape.size() >= 2U && logical_shape.back()->value > 0) {
        row_width = IntImm(DataType::Int(32), static_cast<int>(logical_shape.back()->value));
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
  apply_typed_layout_shape(cast_match.src);
  apply_typed_layout_shape(cast_match.dst);
  apply_typed_layout_shape_from_fact(fact->source_buffer);
  apply_typed_layout_shape_from_fact(fact->target_buffer);
  ICHECK(row_width.defined())
      << "PlanTTKernelABI requires logical row_width for pack-thread direct-store "
         "fragment fill publication of "
      << BufferIdentityName(cast_match.dst);

  const int cb_id = AllocateRequirementIndex(cast_match.dst, CBType::kIntermediate);
  ICHECK_GE(cb_id, 0);
  ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
  const int num_pages = std::max(
      1, cb_requirements_[cb_id].publish_pages_per_event > 0
             ? cb_requirements_[cb_id].publish_pages_per_event
             : cb_requirements_[cb_id].num_pages);

  RecordFragmentCastMaterializationPlans(
      cast_match, *fact, cb_id, num_elements_expr,
      buffer_materialization::kPackThreadDirectStore);

  PrimExpr local_fill_elements = fill_match.num_elements;
  if (const Map<String, Any>* spec = FindLogicalTileLayoutSpec(fill_match.dst)) {
    const int64_t local_extent =
        ProductIntegerArrayField(*spec, schema_key::kLocalShape, int64_t{0});
    if (local_extent > 0) {
      local_fill_elements = IntImm(DataType::Int(32), static_cast<int>(local_extent));
    }
  }

  const Buffer physical_src = ResolvePhysicalComputeBuffer(fill_match.dst);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(cast_match.dst);
  std::vector<Stmt> stmts;
  stmts.push_back(MakeBlackholeCall(tir::builtin::blackhole_fill_fragment(),
                                    {physical_src->data, local_fill_elements,
                                     fill_match.value}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_reserve_back(),
                                    {IntImm32(cb_id), IntImm32(num_pages)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_pack_fill_fragment_to_tiled_cb(),
      {physical_dst->data, IntImm32(cb_id), cast_match.dst_offset, num_elements_expr,
       row_width, fill_match.value}));
  stmts.push_back(MakeBlackholeCall(blackhole_cb_push_back(),
                                    {IntImm32(cb_id), IntImm32(num_pages)}));
  buffer_live_form_cb_by_buffer_identity_[BufferIdentityName(cast_match.dst)] = cb_id;
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::MatchScalarMaxStore(const BufferStoreNode* op,
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

bool PlanTTKernelABI::MatchGroupedScalarMaxLoop(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateScalarMaxSequence(const ScalarMaxMatch& match) {
  InvalidateLastFragmentFillValue(match.dst);
  ExactTiledCBValue dst_in = CreatePublishedExactTiledCBValue(match.dst, "scalar_max_lhs");
  ExactTiledCBValue src_in = CreatePublishedExactTiledCBValue(match.src, "scalar_max_rhs");
  ExactTiledCBValue out = CreateEmptyExactTiledCBValue(match.dst, "scalar_max_out");
  RecordExactComputeOpPlan("binary", "binary_max_tile",
                           {{"lhs", match.dst, "identity"},
                            {"rhs", match.src, "identity"},
                            {"output", match.dst, "identity"}});

  std::vector<Stmt> stmts;
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.dst, dst_in));
  stmts.push_back(PublishLocalBufferToExactTiledCB(match.src, src_in));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_reserve_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_wait_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  for (int tile = 0; tile < out.num_tiles; ++tile) {
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(), {IntImm32(dst_in.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                      {IntImm32(dst_in.cb_id), IntImm32(tile), IntImm32(0)}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_copy_tile_to_dst_init_short(), {IntImm32(src_in.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_copy_tile(),
                                      {IntImm32(src_in.cb_id), IntImm32(tile), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(blackhole_binary_max_tile_init(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_binary_max_tile(),
                                      {IntImm32(0), IntImm32(1), IntImm32(0)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));
    stmts.push_back(
        MakeBlackholeCall(blackhole_pack_reconfig_data_format(), {IntImm32(out.cb_id)}));
    stmts.push_back(MakeBlackholeCall(blackhole_pack_tile(),
                                      {IntImm32(0), IntImm32(out.cb_id), IntImm32(tile)}));
    stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));
  }
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(dst_in.cb_id), IntImm32(dst_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_pop_front(), {IntImm32(src_in.cb_id), IntImm32(src_in.num_tiles)}));
  stmts.push_back(
      MakeBlackholeCall(blackhole_cb_push_back(), {IntImm32(out.cb_id), IntImm32(out.num_tiles)}));
  stmts.push_back(MaterializeExactTiledCBToLocalBuffer(match.dst, out));
  return MaybeWrapComputeSegment(SeqStmt::Flatten(stmts));
}

bool PlanTTKernelABI::MatchScalarFragmentCopyStore(const BufferStoreNode* op,
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

bool PlanTTKernelABI::MatchGroupedScalarFragmentCopyLoop(const ForNode* op,
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

Stmt PlanTTKernelABI::GenerateScalarFragmentCopySequence(const ScalarFragmentCopyMatch& match) {
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
                                                    bool publish_cb) {
  InvalidateLastFragmentFillValue(match.dst);
  const bool force_publish_from_fact =
      GetStorageScope(match.dst) == "blackhole.acc" &&
      FindBufferMaterializationFact(match.dst) != nullptr;
  const bool publish_result = publish_cb || force_publish_from_fact;
  PrimExpr num_elements_expr = match.num_elements;
  std::vector<Stmt> stmts;
  bool use_tiled_republish_materialization = false;
  PrimExpr pack_thread_direct_fill_value;
  PrimExpr tiled_republish_row_width;
  int cb_id = -1;
  int num_pages = 0;
  const std::string dst_buffer_name = BufferIdentityName(match.dst);
  const Buffer physical_dst = ResolvePhysicalComputeBuffer(match.dst);
  const Buffer physical_src = ResolvePhysicalComputeBuffer(match.src);
  if (publish_result) {
    cb_id = AllocateRequirementIndex(match.dst, CBType::kIntermediate);
    ICHECK_GE(cb_id, 0);
    ICHECK_LT(cb_id, static_cast<int>(cb_requirements_.size()));
    num_pages = std::max(
        1, cb_requirements_[cb_id].publish_pages_per_event > 0
               ? cb_requirements_[cb_id].publish_pages_per_event
               : cb_requirements_[cb_id].num_pages);
    const std::string buffer_identity = BufferIdentityName(match.dst);
    auto it = cb_consumed_compute_input_pages_by_buffer_identity_.find(buffer_identity);
    if (it != cb_consumed_compute_input_pages_by_buffer_identity_.end() &&
        cb_requirements_[cb_id].publish_pages_per_event <= 0) {
      num_pages = std::max(1, it->second);
    }
    const BlackholeBufferMaterializationFact* fact = FindBufferMaterializationFact(match.dst);
    if (fact != nullptr) {
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
                : buffer_materialization::kMailboxWritePtr);
      }
    }
  }
  if (use_tiled_republish_materialization) {
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
          MakeBlackholeCall(tir::builtin::blackhole_cast_fragment_slice_to_tiled_cb(),
                            {physical_dst->data, physical_src->data, IntImm32(cb_id),
                             match.dst_offset, match.src_offset, num_elements_expr,
                             tiled_republish_row_width}));
    }
    buffer_live_form_cb_by_buffer_identity_[dst_buffer_name] = cb_id;
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

PlanTTKernelABI::FutureBufferUses PlanTTKernelABI::ClassifyFutureBufferUses(
    const Buffer& buffer, int current_order_index) const {
  FutureBufferUses uses;
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = buffer_flow_facts_.find(buffer_identity);
  if (it == buffer_flow_facts_.end()) {
    return uses;
  }
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    if (event.kind == BlackholeBufferFlowEventKind::kWrite) {
      break;
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

bool PlanTTKernelABI::ShouldRetainComputeInputBuffer(const Buffer& buffer,
                                                       int current_order_index) const {
  return ClassifyFutureBufferUses(buffer, current_order_index).has_compute_consume;
}

bool PlanTTKernelABI::ShouldReacquireComputeInputBuffer(const Buffer& buffer,
                                                          int current_order_index) const {
  if (GetStorageScope(buffer) != "blackhole.acc") {
    return false;
  }
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto it = buffer_flow_facts_.find(buffer_identity);
  if (it == buffer_flow_facts_.end()) {
    return false;
  }
  for (const BlackholeBufferFlowEvent& event : it->second.events) {
    if (event.order_index <= current_order_index) {
      continue;
    }
    return event.kind == BlackholeBufferFlowEventKind::kWrite;
  }
  return false;
}

bool PlanTTKernelABI::ShouldPublishBufferResult(const Buffer& buffer,
                                                  int current_order_index) const {
  const FutureBufferUses uses = ClassifyFutureBufferUses(buffer, current_order_index);
  return uses.has_compute_consume || uses.has_transport_consume;
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
      MakeBlackholeCall(blackhole_write_local_fragment_slice_to_tiled_cb(),
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

// Parse a colon-separated string into fields
Stmt PlanTTKernelABI::VisitStmt_(const AttrStmtNode* op) {
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

Stmt PlanTTKernelABI::VisitStmt_(const SeqStmtNode* op) {
  Array<Stmt> rewritten;
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const auto order_it = stmt_order_index_by_node_.find(op->seq[i].get());
    const int current_order_index =
        order_it != stmt_order_index_by_node_.end() ? order_it->second : static_cast<int>(i);
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

    auto is_redundant_zero_fill_before_full_overwrite_matmul =
        [&](const Stmt& fill_stmt, const Stmt& next_stmt) -> bool {
      FragmentFillMatch fill_match;
      const VarNode* fill_data = nullptr;
      auto match_zero_fill = [&](const Stmt& stmt) -> bool {
        const auto* fill_loop = AsUnwrappedFor(stmt);
        if (fill_loop && MatchDirectFragmentFill(fill_loop, &fill_match)) {
          fill_data = BufferDataIdentity(fill_match.dst);
          return IsLiteralZeroValue(fill_match.value);
        }
        const auto* fill_store = AsUnwrappedBufferStore(stmt);
        if (fill_store && MatchScalarFragmentFillStore(fill_store, &fill_match)) {
          fill_data = BufferDataIdentity(fill_match.dst);
          return IsLiteralZeroValue(fill_match.value);
        }

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
        const auto* call = eval ? eval->value.as<CallNode>() : nullptr;
        if (!call || !call->op->IsInstance<OpNode>()) {
          return false;
        }
        Op call_op = Downcast<Op>(call->op);
        if (call_op->name != "tl.blackhole.fill_fragment" || call->args.size() < 3) {
          return false;
        }
        fill_data = call->args[0].as<VarNode>();
        if (!fill_data || !IsLiteralZeroValue(call->args[2])) {
          return false;
        }
        auto it = compute_physical_buffers_by_data_.find(fill_data);
        if (it != compute_physical_buffers_by_data_.end()) {
          fill_match.dst = it->second;
        }
        fill_match.num_elements = call->args[1];
        fill_match.value = call->args[2];
        return true;
      };
      if (!match_zero_fill(fill_stmt)) {
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
      if (!fill_match.dst.defined() && fill_data != BufferDataIdentity(out_buffer)) {
        return false;
      }
      return FindBufferMaterializationFact(out_buffer) == nullptr;
    };

    if (!select_compute_builtins_only_) {
      Stmt retained_matmul;
      FragmentCastMatch post_merge_cast;
      const FragmentCastMatch* post_merge_cast_ptr = nullptr;
      int post_merge_cast_order_index = -1;
      if (i + 1 < op->seq.size()) {
        if (const auto* next_cast_loop = AsUnwrappedFor(op->seq[i + 1])) {
          if (MatchDirectFragmentCast(next_cast_loop, &post_merge_cast)) {
            post_merge_cast_ptr = &post_merge_cast;
            const auto next_order_it = stmt_order_index_by_node_.find(op->seq[i + 1].get());
            post_merge_cast_order_index =
                next_order_it != stmt_order_index_by_node_.end()
                    ? next_order_it->second
                    : static_cast<int>(i + 1);
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
    if (!select_compute_builtins_only_ && i + 1 < op->seq.size()) {
      const auto* fill_loop = AsUnwrappedFor(op->seq[i]);
      const auto* cast_loop = AsUnwrappedFor(op->seq[i + 1]);
      FragmentFillMatch fill_match;
      FragmentCastMatch cast_match;
      if (fill_loop && cast_loop && MatchDirectFragmentFill(fill_loop, &fill_match) &&
          MatchDirectFragmentCast(cast_loop, &cast_match)) {
        Stmt fused =
            GenerateFragmentFillCastPublishSequence(fill_match, cast_match, current_order_index);
        if (fused.defined()) {
          rewritten.push_back(fused);
          ++i;
          continue;
        }
      }
    }
    if (!select_compute_builtins_only_) {
      if (const auto* direct_cast_loop = AsUnwrappedFor(op->seq[i])) {
        FragmentCastMatch cast_match;
        if (MatchDirectFragmentCast(direct_cast_loop, &cast_match)) {
          std::vector<Stmt> prefix;
          std::vector<Stmt> suffix;
          ValidatePublishedBufferSourceEdge(cast_match.src, cast_match.dst);
          AppendPublishedBufferSourceMaterialization(cast_match.src, current_order_index, &prefix,
                                                     &suffix);
          const bool publish_cb = ShouldPublishBufferResult(cast_match.dst, current_order_index);
          prefix.push_back(GenerateFragmentCastSequence(cast_match, publish_cb));
          prefix.insert(prefix.end(), suffix.begin(), suffix.end());
          rewritten.push_back(SeqStmt::Flatten(prefix));
          continue;
        }
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
  if (!select_compute_builtins_only_) {
    FragmentCastMatch direct_cast_match;
    if (MatchDirectFragmentCast(op, &direct_cast_match)) {
      return GenerateFragmentCastSequence(direct_cast_match);
    }
    LocalToCBSliceMatch local_to_cb_match;
    if (MatchDirectLocalToCBSliceLoop(op, &local_to_cb_match)) {
      saw_copy_op_ = true;
      return GenerateLocalToCBSliceLoopSequence(op, local_to_cb_match);
    }
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
Stmt PlanTTKernelABI::VisitStmt_(const EvaluateNode* op) {
  if (select_compute_builtins_only_) {
    return GetRef<Stmt>(op);
  }
  if (const auto* call = op->value.as<CallNode>()) {
    if (call->op->IsInstance<OpNode>()) {
      const Op call_op = Downcast<Op>(call->op);
      if (call_op->name == "tl.blackhole.fill_fragment" && call->args.size() >= 3 &&
          IsFragmentFillValue(call->args[2])) {
        if (const auto* data = call->args[0].as<VarNode>()) {
          last_fragment_fill_value_by_data_[data] = call->args[2];
          auto physical_it = compute_physical_buffers_by_data_.find(data);
          if (physical_it != compute_physical_buffers_by_data_.end()) {
            const std::string identity = BufferIdentityName(physical_it->second);
            if (!identity.empty()) {
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
      return LowerMatmulCallWithFlowAnalysis(call, current_order_index);
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
  if (!select_compute_builtins_only_ && IsCopyOperation(op)) {
    saw_copy_op_ = true;
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

}  // namespace tl
}  // namespace tvm
