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
 * \file codegen_blackhole.cc
 * \brief Generate TT-Metal code for Blackhole backend.
 */

#include "codegen_blackhole.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../layout/layout.h"
#include "../transform/common/blackhole_runtime_arg_schema.h"
#include "../tir/builtin_blackhole.h"
#include "tt_program_projection.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

namespace tvm {
namespace tl {

namespace {

bool IsBufferAddressRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr32" || kind == "input_buffer_addr" ||
         kind == "output_buffer_addr32" || kind == "output_buffer_addr";
}

std::string RequireStringImm(const tvm::PrimExpr& expr, const char* op_name,
                             const char* arg_name) {
  const auto* value = expr.as<tvm::tir::StringImmNode>();
  ICHECK(value) << op_name << " expects " << arg_name << " to be a string literal";
  return value->value;
}

const char* ReduceKindToTTMetal(const std::string& reduce_kind, const char* op_name) {
  if (reduce_kind == "sum") {
    return "PoolType::SUM";
  }
  if (reduce_kind == "max") {
    return "PoolType::MAX";
  }
  ICHECK(false) << op_name << " got unsupported reduce kind " << reduce_kind;
  return "";
}

const char* ReduceDimToTTMetal(const std::string& reduce_dim, const char* op_name) {
  if (reduce_dim == "row") {
    return "ReduceDim::REDUCE_ROW";
  }
  if (reduce_dim == "col") {
    return "ReduceDim::REDUCE_COL";
  }
  ICHECK(false) << op_name << " got unsupported reduce dim " << reduce_dim;
  return "";
}

const tvm::tir::VarNode* AsHandleVar(const tvm::PrimExpr& expr) {
  if (const auto* var = expr.as<tvm::tir::VarNode>()) {
    return var;
  }
  return nullptr;
}

std::vector<tvm::tir::Stmt> FlattenTopLevelSeq(const tvm::tir::Stmt& stmt) {
  if (const auto* seq = stmt.as<tvm::tir::SeqStmtNode>()) {
    return std::vector<tvm::tir::Stmt>(seq->seq.begin(), seq->seq.end());
  }
  return {stmt};
}

bool ExtractThreadScopedCBStaging(const tvm::tir::Stmt& stmt,
                                  const tvm::tir::VarNode* thread_var,
                                  tvm::tir::Stmt* once_prefix,
                                  tvm::tir::Stmt* threaded_body,
                                  tvm::tir::Stmt* once_suffix) {
  auto is_blackhole_builtin = [](const tvm::tir::CallNode* call, const tvm::Op& builtin,
                                 const char* op_name) {
    if (!call) {
      return false;
    }
    if (call->op.same_as(builtin)) {
      return true;
    }
    if (const auto* op = call->op.as<tvm::OpNode>()) {
      return op->name == op_name;
    }
    return false;
  };

  tvm::tir::Stmt current = stmt;
  while (true) {
    if (const auto* attr = current.as<tvm::tir::AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto* let = current.as<tvm::tir::LetStmtNode>()) {
      current = let->body;
      continue;
    }
    if (const auto* decl = current.as<tvm::tir::DeclBufferNode>()) {
      current = decl->body;
      continue;
    }
    if (const auto* alloc = current.as<tvm::tir::AllocateNode>()) {
      current = alloc->body;
      continue;
    }
    break;
  }

  const auto* seq = current.as<tvm::tir::SeqStmtNode>();
  if (!seq || seq->seq.size() < 3) {
    return false;
  }
  const auto* reserve_eval = seq->seq.front().as<tvm::tir::EvaluateNode>();
  const auto* push_eval = seq->seq.back().as<tvm::tir::EvaluateNode>();
  if (!reserve_eval || !push_eval) {
    return false;
  }
  const auto* reserve_call = reserve_eval->value.as<tvm::tir::CallNode>();
  const auto* push_call = push_eval->value.as<tvm::tir::CallNode>();
  if (!is_blackhole_builtin(reserve_call, tir::builtin::blackhole_cb_reserve_back(),
                            "tl.blackhole.cb_reserve_back") ||
      !is_blackhole_builtin(push_call, tir::builtin::blackhole_cb_push_back(),
                            "tl.blackhole.cb_push_back")) {
    return false;
  }

  std::vector<tvm::tir::Stmt> middle(seq->seq.begin() + 1, seq->seq.end() - 1);
  if (middle.empty()) {
    return false;
  }
  tvm::tir::Stmt middle_stmt =
      middle.size() == 1 ? middle.front() : tvm::tir::SeqStmt::Flatten(middle);
  const bool uses_thread_var = tir::UsesVar(
      middle_stmt, [thread_var](const tvm::tir::VarNode* var) { return var == thread_var; });
  if (!uses_thread_var) {
    return false;
  }

  *once_prefix = seq->seq.front();
  *threaded_body = middle_stmt;
  *once_suffix = seq->seq.back();
  return true;
}

struct ThreadEmissionPiece {
  tvm::tir::Stmt stmt;
  bool uses_thread_var{false};
};

std::vector<ThreadEmissionPiece> BuildThreadEmissionPieces(const tvm::tir::Stmt& stmt,
                                                           const tvm::tir::VarNode* thread_var) {
  auto add_piece = [](std::vector<ThreadEmissionPiece>* pieces, const tvm::tir::Stmt& piece,
                      bool uses_thread_var) {
    if (!piece.defined()) {
      return;
    }
    if (const auto* seq = piece.as<tvm::tir::SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return;
      }
    }
    pieces->push_back(ThreadEmissionPiece{piece, uses_thread_var});
  };

  std::vector<ThreadEmissionPiece> pieces;
  for (const auto& top_level_stmt : FlattenTopLevelSeq(stmt)) {
    tvm::tir::Stmt once_prefix;
    tvm::tir::Stmt threaded_body;
    tvm::tir::Stmt once_suffix;
    if (ExtractThreadScopedCBStaging(top_level_stmt, thread_var, &once_prefix, &threaded_body,
                                     &once_suffix)) {
      add_piece(&pieces, once_prefix, /*uses_thread_var=*/false);
      add_piece(&pieces, threaded_body, /*uses_thread_var=*/true);
      add_piece(&pieces, once_suffix, /*uses_thread_var=*/false);
      continue;
    }

    const bool uses_thread_var = tir::UsesVar(
        top_level_stmt, [thread_var](const tvm::tir::VarNode* var) { return var == thread_var; });
    add_piece(&pieces, top_level_stmt, uses_thread_var);
  }
  return pieces;
}

ffi::Array<ffi::Any> AggregateSegmentRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  ffi::Array<ffi::Any> aggregated;
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (segment_plan.empty()) {
    return aggregated;
  }

  std::unordered_set<std::string> seen_runtime_args;
  for (const auto& item : segment_plan) {
    auto segment = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (segment.empty()) {
      continue;
    }
    auto runtime_args_it = segment.Get("runtime_args");
    if (!runtime_args_it.has_value()) {
      continue;
    }
    for (const auto& arg_item : Downcast<tvm::ffi::Array<tvm::ffi::Any>>(runtime_args_it.value())) {
      auto arg = arg_item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      std::string identity;
      std::string kind;
      if (auto v = arg.Get("identity")) {
        identity = Downcast<tvm::ffi::String>(v.value());
      }
      if (auto v = arg.Get("kind")) {
        kind = Downcast<tvm::ffi::String>(v.value());
      }
      std::string dedupe_key =
          !identity.empty() && !kind.empty() ? identity + ":" + kind : identity;
      if (!dedupe_key.empty() && !seen_runtime_args.insert(dedupe_key).second) {
        continue;
      }
      aggregated.push_back(arg);
    }
  }
  return aggregated;
}

ffi::Array<ffi::Any> GetRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  return AggregateSegmentRuntimeArgsForCodegen(f);
}

ffi::Array<ffi::Any> GetPerWorkArgSpecsForCodegen(const tvm::tir::PrimFunc& f) {
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (segment_plan.size() != 1) {
    return ffi::Array<ffi::Any>();
  }
  auto segment = segment_plan[0].as<ffi::Map<ffi::String, ffi::Any>>().value_or(
      ffi::Map<ffi::String, ffi::Any>());
  if (auto v = segment.Get(
          ffi::String(::tvm::tl::blackhole_runtime_arg_schema::kPerWorkArgSpecs))) {
    return Downcast<ffi::Array<ffi::Any>>(v.value());
  }
  return ffi::Array<ffi::Any>();
}

ffi::Array<ffi::Any> GetCBConfigsForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCBConfigsFromExecutable(f, "Blackhole codegen");
}

ffi::Map<ffi::String, ffi::Any> GetCorePlanForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCorePlanFromExecutable(f, "Blackhole codegen");
}

std::string GetCoreTypeForCodegen(const tvm::tir::PrimFunc& f) {
  auto segment_plan = tt_program_projection::GetSegmentPlanFromExecutable(f, "Blackhole codegen");
  if (!segment_plan.empty()) {
    auto segment = segment_plan[0].as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (auto value = segment.Get("core_type")) {
      return Downcast<ffi::String>(value.value());
    }
  }
  return "";
}

bool HasRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  return !GetRuntimeArgsForCodegen(f).empty();
}

}  // namespace

CodeGenBlackhole::CodeGenBlackhole()
    : headers_emitted_(false),
      core_type_(CoreType::kBRISC),
      need_dataflow_api_h_(false),
      need_compute_api_h_(false),
      emit_debug_waypoints_(false) {}

void CodeGenBlackhole::Init(bool output_ssa, bool emit_asserts,
                            bool emit_fwd_func_decl, std::string target_str,
                            const std::unordered_set<std::string> &devices) {
  CodeGenCHost::Init(output_ssa, emit_asserts, emit_fwd_func_decl,
                     target_str, devices);

  // Reset state for new CodeGen instance
  headers_emitted_ = false;
  core_type_ = CoreType::kBRISC;
  need_dataflow_api_h_ = false;
  need_compute_api_h_ = false;
  emit_debug_waypoints_ = std::getenv("TILELANG_BLACKHOLE_DEBUG_WAYPOINTS") != nullptr;
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_kind_.clear();
  runtime_arg_vars_by_identity_.clear();
  runtime_arg_vars_by_name_.clear();
  per_work_arg_bindings_by_identity_.clear();
  per_work_arg_bindings_.clear();
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_id_by_requirement_name_.clear();
  cb_num_pages_by_requirement_name_.clear();
  cb_publish_pages_by_requirement_name_.clear();
  cb_initial_reserve_pages_by_requirement_name_.clear();
  thread_idx_x_expr_.clear();
  logical_grid_x_ = 1;
  logical_grid_y_ = 1;
  linearization_ = "row_major";
}

std::string CodeGenBlackhole::GetKernelCode() const {
  // Return the kernel code with TT-Metal headers but without TVM-specific headers
  // decl_stream now contains TT-Metal headers (dataflow_api.h, etc.)
  // stream contains the actual kernel implementation
  const std::string body = stream.str();
  std::ostringstream kernel_code;
  kernel_code << decl_stream.str();
  if (body.find("tilelang_cb_write_ptr_bytes_direct(") != std::string::npos) {
    kernel_code << "ALWI uint32_t tilelang_cb_write_ptr_bytes_direct(uint32_t cb_id) {\n";
    kernel_code << "  return get_local_cb_interface(cb_id).fifo_wr_ptr << 4;\n";
    kernel_code << "}\n";
  }
  kernel_code << body;
  return kernel_code.str();
}

void CodeGenBlackhole::AddFunction(const tvm::GlobalVar &gvar,
                                   const tvm::tir::PrimFunc &f) {
  // Emit TT-Metal headers for kernel code (per-instance, not static)
  if (!headers_emitted_) {
    // Clear decl_stream to remove TVM headers added by CodeGenCHost::Init
    decl_stream.str("");
    decl_stream.clear();

    decl_stream << "// TT-Metal kernel generated by TileLang\n";
    decl_stream << "#include <cstdint>\n";
    decl_stream << "#include <cmath>\n";
    decl_stream << "#include <limits>\n";
    decl_stream << "\n";
    decl_stream << "template <typename To, typename From>\n";
    decl_stream << "static inline To tilelang_bit_cast(From value) {\n";
    decl_stream << "  static_assert(sizeof(To) == sizeof(From), \"tilelang_bit_cast requires equal-sized types\");\n";
    decl_stream << "  To out;\n";
    decl_stream << "  __builtin_memcpy(&out, &value, sizeof(To));\n";
    decl_stream << "  return out;\n";
    decl_stream << "}\n";
    decl_stream << "\n";

    // Detect core type from function attributes (IR-driven, not function name)
    std::string core_type_str = GetCoreTypeForCodegen(f);
    if (core_type_str == "brisc") {
      core_type_ = CoreType::kBRISC;
    } else if (core_type_str == "ncrisc") {
      core_type_ = CoreType::kNCRISC;
    } else if (core_type_str == "trisc") {
      core_type_ = CoreType::kTRISC;
    } else {
      ICHECK(false) << "Blackhole codegen requires executable segment core_type, got '"
                    << core_type_str << "'";
    }

    // Include appropriate API header based on core type
    switch (core_type_) {
      case CoreType::kBRISC:
      case CoreType::kNCRISC:
        decl_stream << "// DataMovement kernel API (BRISC/NCRISC)\n";
        decl_stream << "#include \"api/dataflow/dataflow_api.h\"\n";
        decl_stream << "#include \"experimental/circular_buffer.h\"\n";
        decl_stream << "#include \"experimental/tensor.h\"\n";
        break;
      case CoreType::kTRISC:
        decl_stream << "// Compute kernel API (TRISC)\n";
        decl_stream << "#ifndef REDUCE_OP\n";
        decl_stream << "#define REDUCE_OP PoolType::SUM\n";
        decl_stream << "#endif\n";
        decl_stream << "#ifndef REDUCE_DIM\n";
        decl_stream << "#define REDUCE_DIM ReduceDim::REDUCE_ROW\n";
        decl_stream << "#endif\n";
        decl_stream << "#include \"api/compute/pack.h\"\n";
        decl_stream << "#include \"api/compute/reconfig_data_format.h\"\n";
        decl_stream << "#include \"api/compute/tile_move_copy.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_binary.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/eltwise_unary.h\"\n";
        decl_stream << "#include \"api/compute/bcast.h\"\n";
        decl_stream << "#include \"api/compute/binary_max_min.h\"\n";
        decl_stream << "#include \"api/compute/reduce.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/fill.h\"\n";
        decl_stream << "#include \"api/compute/eltwise_unary/recip.h\"\n";
        decl_stream << "#include \"api/compute/compute_kernel_api.h\"\n";
        decl_stream << "#include \"api/compute/matmul.h\"\n";
        decl_stream << "#include \"api/debug/waypoint.h\"\n";
        decl_stream << "#include \"experimental/circular_buffer.h\"\n";
        decl_stream << "#include \"hostdevcommon/kernel_structs.h\"\n";
	        decl_stream << "using half = _Float16;\n";
	        decl_stream << "static constexpr float inff = std::numeric_limits<float>::infinity();\n";
	        decl_stream << "ALWI uint32_t tilelang_bitcast_float_to_u32(float value) {\n";
	        decl_stream << "  return tilelang_bit_cast<uint32_t>(value);\n";
	        decl_stream << "}\n";
        decl_stream << "ALWI uint16_t tilelang_float_to_half_bits(float value) {\n";
        decl_stream << "  const uint32_t bits = tilelang_bitcast_float_to_u32(value);\n";
        decl_stream << "  const uint32_t sign = (bits >> 16) & 0x8000u;\n";
        decl_stream << "  const uint32_t exponent = (bits >> 23) & 0xffu;\n";
        decl_stream << "  uint32_t mantissa = bits & 0x7fffffu;\n";
        decl_stream << "  if (exponent == 0xffu) {\n";
        decl_stream << "    if (mantissa == 0u) {\n";
        decl_stream << "      return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "    }\n";
        decl_stream << "    mantissa >>= 13;\n";
        decl_stream << "    return static_cast<uint16_t>(sign | 0x7c00u | mantissa | (mantissa == 0u));\n";
        decl_stream << "  }\n";
        decl_stream << "  int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;\n";
        decl_stream << "  if (half_exponent >= 31) {\n";
        decl_stream << "    return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "  }\n";
        decl_stream << "  if (half_exponent <= 0) {\n";
        decl_stream << "    if (half_exponent < -10) {\n";
        decl_stream << "      return static_cast<uint16_t>(sign);\n";
        decl_stream << "    }\n";
        decl_stream << "    mantissa |= 0x800000u;\n";
        decl_stream << "    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);\n";
        decl_stream << "    uint32_t half_mantissa = mantissa >> shift;\n";
        decl_stream << "    const uint32_t round_bit = 1u << (shift - 1);\n";
        decl_stream << "    const uint32_t remainder = mantissa & (round_bit - 1u);\n";
        decl_stream << "    const bool round_up = (mantissa & round_bit) != 0u && (remainder != 0u || (half_mantissa & 1u) != 0u);\n";
        decl_stream << "    if (round_up) {\n";
        decl_stream << "      ++half_mantissa;\n";
        decl_stream << "    }\n";
        decl_stream << "    return static_cast<uint16_t>(sign | half_mantissa);\n";
        decl_stream << "  }\n";
        decl_stream << "  uint32_t half_mantissa = mantissa >> 13;\n";
        decl_stream << "  const uint32_t remainder = mantissa & 0x1fffu;\n";
        decl_stream << "  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0u)) {\n";
        decl_stream << "    ++half_mantissa;\n";
        decl_stream << "    if (half_mantissa == 0x400u) {\n";
        decl_stream << "      half_mantissa = 0u;\n";
        decl_stream << "      ++half_exponent;\n";
        decl_stream << "      if (half_exponent >= 31) {\n";
        decl_stream << "        return static_cast<uint16_t>(sign | 0x7c00u);\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10) | (half_mantissa & 0x3ffu));\n";
	        decl_stream << "}\n";
	        decl_stream << "ALWI uint16_t tilelang_float_to_bfloat_bits(float value) {\n";
	        decl_stream << "  const uint32_t bits = tilelang_bit_cast<uint32_t>(value);\n";
	        decl_stream << "  const uint32_t lsb = (bits >> 16) & 1u;\n";
	        decl_stream << "  const uint32_t rounding_bias = 0x7fffu + lsb;\n";
	        decl_stream << "  return static_cast<uint16_t>((bits + rounding_bias) >> 16);\n";
	        decl_stream << "}\n";
        decl_stream << "ALWI float tilelang_fast_exp2f(float x) {\n";
        decl_stream << "  if (x <= -126.0f) { return 0.0f; }\n";
        decl_stream << "  if (x >= 126.0f) { x = 126.0f; }\n";
        decl_stream << "  int ipart = static_cast<int>(x);\n";
        decl_stream << "  if (static_cast<float>(ipart) > x) { --ipart; }\n";
        decl_stream << "  const float fpart = x - static_cast<float>(ipart);\n";
        decl_stream << "  const float poly = 1.0f + fpart * (0.69314718f + fpart * (0.24022651f + fpart * (0.05550411f + fpart * 0.00961813f)));\n";
	        decl_stream << "  const uint32_t exponent_bits = static_cast<uint32_t>(ipart + 127) << 23;\n";
	        decl_stream << "  return tilelang_bit_cast<float>(exponent_bits) * poly;\n";
	        decl_stream << "}\n";
        decl_stream << "template <typename T>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_fill_fragment(T* dst, uint32_t num_elements, T value) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = value; }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_add_fragment(DstT* dst, const SrcT* src, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = static_cast<DstT>(dst[i] + static_cast<DstT>(src[i])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_cast_fragment_slice(DstT* dst, const SrcT* src, uint32_t dst_offset, uint32_t src_offset, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[dst_offset + i] = static_cast<DstT>(src[src_offset + i]); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_tilize_fragment_tile_nfaces(const BitsT* src, BitsT* dst) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  uint32_t dst_index = 0;\n";
        decl_stream << "  for (uint32_t face_y = 0; face_y < kTileRows / kFaceRows; ++face_y) {\n";
        decl_stream << "    for (uint32_t face_x = 0; face_x < kTileCols / kFaceCols; ++face_x) {\n";
        decl_stream << "      for (uint32_t row = 0; row < kFaceRows; ++row) {\n";
        decl_stream << "        const BitsT* src_row = src + (face_y * kFaceRows + row) * kTileCols + face_x * kFaceCols;\n";
        decl_stream << "        for (uint32_t col = 0; col < kFaceCols; ++col) {\n";
        decl_stream << "          dst[dst_index++] = src_row[col];\n";
        decl_stream << "        }\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_tilize_fragment_slice_nfaces(const BitsT* src, BitsT* dst, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  const uint32_t tiles_per_row = row_width / kTileCols;\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) {\n";
        decl_stream << "    const uint32_t logical_index = dst_offset_elements + i;\n";
        decl_stream << "    const uint32_t global_row = logical_index / row_width;\n";
        decl_stream << "    const uint32_t global_col = logical_index % row_width;\n";
        decl_stream << "    const uint32_t tile_row = global_row / kTileRows;\n";
        decl_stream << "    const uint32_t tile_col = global_col / kTileCols;\n";
        decl_stream << "    const uint32_t row_in_tile = global_row % kTileRows;\n";
        decl_stream << "    const uint32_t col_in_tile = global_col % kTileCols;\n";
        decl_stream << "    const uint32_t face_row = row_in_tile / kFaceRows;\n";
        decl_stream << "    const uint32_t face_col = col_in_tile / kFaceCols;\n";
        decl_stream << "    const uint32_t row_in_face = row_in_tile % kFaceRows;\n";
        decl_stream << "    const uint32_t col_in_face = col_in_tile % kFaceCols;\n";
        decl_stream << "    const uint32_t tile_index = tile_row * tiles_per_row + tile_col;\n";
        decl_stream << "    const uint32_t tiled_index = tile_index * 1024u + face_row * (kFaceRows * kTileCols) + face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face;\n";
        decl_stream << "    dst[tiled_index] = src[i];\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "ALWI void tilelang_pack_fill_bfloat16_tiled_cb(uint32_t cb_id, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width, float value) {\n";
        decl_stream << "  (void)dst_offset_elements; (void)row_width;\n";
        decl_stream << "  const uint32_t num_tiles = (num_elements + 1023u) / 1024u;\n";
        decl_stream << "  fill_tile_init();\n";
        decl_stream << "  for (uint32_t tile = 0; tile < num_tiles; ++tile) {\n";
        decl_stream << "    tile_regs_acquire();\n";
        decl_stream << "    fill_tile(0, value);\n";
        decl_stream << "    tile_regs_commit();\n";
        decl_stream << "    tile_regs_wait();\n";
        decl_stream << "    pack_reconfig_data_format(cb_id);\n";
        decl_stream << "    pack_tile(0, cb_id, tile);\n";
        decl_stream << "    tile_regs_release();\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "ALWI void tilelang_pack_fill_float32_tiled_cb(uint32_t cb_id, uint32_t dst_offset_elements, uint32_t num_elements, uint32_t row_width, float value) {\n";
        decl_stream << "  (void)dst_offset_elements; (void)row_width;\n";
        decl_stream << "  const uint32_t num_tiles = (num_elements + 1023u) / 1024u;\n";
        decl_stream << "  fill_tile_init();\n";
        decl_stream << "  for (uint32_t tile = 0; tile < num_tiles; ++tile) {\n";
        decl_stream << "    tile_regs_acquire();\n";
        decl_stream << "    fill_tile(0, value);\n";
        decl_stream << "    tile_regs_commit();\n";
        decl_stream << "    tile_regs_wait();\n";
        decl_stream << "    pack_reconfig_data_format(cb_id);\n";
        decl_stream << "    pack_tile(0, cb_id, tile);\n";
        decl_stream << "    tile_regs_release();\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename BitsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_untilize_fragment_tile_nfaces(const BitsT* src, BitsT* dst) {\n";
        decl_stream << "  constexpr uint32_t kTileRows = 32;\n";
        decl_stream << "  constexpr uint32_t kTileCols = 32;\n";
        decl_stream << "  constexpr uint32_t kFaceRows = 16;\n";
        decl_stream << "  constexpr uint32_t kFaceCols = 16;\n";
        decl_stream << "  uint32_t src_index = 0;\n";
        decl_stream << "  for (uint32_t face_y = 0; face_y < kTileRows / kFaceRows; ++face_y) {\n";
        decl_stream << "    for (uint32_t face_x = 0; face_x < kTileCols / kFaceCols; ++face_x) {\n";
        decl_stream << "      for (uint32_t row = 0; row < kFaceRows; ++row) {\n";
        decl_stream << "        BitsT* dst_row = dst + (face_y * kFaceRows + row) * kTileCols + face_x * kFaceCols;\n";
        decl_stream << "        for (uint32_t col = 0; col < kFaceCols; ++col) {\n";
        decl_stream << "          dst_row[col] = src[src_index++];\n";
        decl_stream << "        }\n";
        decl_stream << "      }\n";
        decl_stream << "    }\n";
        decl_stream << "  }\n";
        decl_stream << "}\n";
        break;
      default:
        ICHECK(false) << "Blackhole codegen reached unknown core_type enum";
        break;
    }
    decl_stream << "\n";
    headers_emitted_ = true;
  }

  // Generate TT-Metal kernel_main function using IR visitor
  GenerateGenericKernelMain(f, gvar->name_hint);
}

void CodeGenBlackhole::GenerateGenericKernelMain(const tvm::tir::PrimFunc &f,
                                                  const std::string &func_name) {
  // Add function name as comment
  stream << "// Kernel: " << func_name << "\n";

  // Generate kernel_main entry point (TT-Metal convention)
  stream << "void kernel_main() {\n";

  // Generate argument loading code
  // TT-Metal kernels use get_arg_val<uint32_t>(arg_index) to read arguments
  stream << "  // Load kernel arguments from runtime\n";
  LoadCorePlan(f);
  LoadLogicalTileLayouts(f);
  if (HasRuntimeArgsForCodegen(f)) {
    EmitRuntimeArgLoads(f);
    this->VisitStmt(f->body);
    stream << "}\n\n";
    return;
  }

  int arg_idx = 0;
  for (size_t i = 0; i < f->params.size(); ++i) {
    const auto &param = f->params[i];
    std::string param_name = param->name_hint;
    tvm::DataType dtype = param->dtype;

    // Store parameter info for use in kernel body
    var_idmap_[param.get()] = param_name;

    if (dtype.is_handle()) {
      // Buffer argument - load as 64-bit address from two 32-bit args
      stream << "  // Argument " << arg_idx << ": " << param_name
             << " (buffer pointer)\n";
      stream << "  uint32_t " << param_name << "_lo = get_arg_val<uint32_t>("
             << arg_idx++ << ");\n";
      stream << "  uint32_t " << param_name << "_hi = get_arg_val<uint32_t>("
             << arg_idx++ << ");\n";
      stream << "  uint64_t " << param_name << "_addr = ((uint64_t)"
             << param_name << "_hi << 32) | " << param_name << "_lo;\n";
      // Use void* for handle types (buffer pointers)
      stream << "  void* " << param_name << " = (void*)(uintptr_t)"
             << param_name << "_addr;\n";
    } else if (dtype.is_int() || dtype.is_uint()) {
      // Integer scalar argument
      stream << "  // Argument " << arg_idx << ": " << param_name << " ("
             << dtype << ")\n";
      stream << "  " << dtype << " " << param_name
             << " = get_arg_val<uint32_t>(" << arg_idx++ << ");\n";
    } else if (dtype.is_float()) {
      ICHECK_EQ(dtype.bits(), 32)
          << "Blackhole codegen supports only 32-bit float scalar runtime arguments, got "
          << dtype;
      // Float scalar argument - passed as bits in uint32_t
      stream << "  // Argument " << arg_idx << ": " << param_name << " ("
             << dtype << ")\n";
      stream << "  uint32_t " << param_name << "_bits = get_arg_val<uint32_t>("
             << arg_idx++ << ");\n";
      stream << "  " << dtype << " " << param_name
             << " = tilelang_bit_cast<" << dtype << ">(" << param_name
             << "_bits);\n";
    } else {
      // Other types - default to uint32_t
      stream << "  // Argument " << arg_idx << ": " << param_name << " ("
             << dtype << ")\n";
      stream << "  uint32_t " << param_name
             << " = get_arg_val<uint32_t>(" << arg_idx++ << ");\n";
    }
  }
  stream << "\n";

  // Visit function body
  this->VisitStmt(f->body);

  stream << "}\n\n";
}

void CodeGenBlackhole::LoadCorePlan(const tvm::tir::PrimFunc &f) {
  logical_grid_x_ = 1;
  logical_grid_y_ = 1;
  linearization_ = "row_major";

  auto core_plan = GetCorePlanForCodegen(f);
  if (core_plan.empty()) {
    return;
  }

  if (auto v = core_plan.Get("logical_grid_x")) {
    logical_grid_x_ = Downcast<tvm::Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_x")) {
    logical_grid_x_ = Downcast<tvm::Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("logical_grid_y")) {
    logical_grid_y_ = Downcast<tvm::Integer>(v.value()).IntValue();
  } else if (auto v = core_plan.Get("grid_y")) {
    logical_grid_y_ = Downcast<tvm::Integer>(v.value()).IntValue();
  }
  if (auto v = core_plan.Get("linearization")) {
    linearization_ = Downcast<tvm::ffi::String>(v.value());
  }
}

void CodeGenBlackhole::LoadLogicalTileLayouts(const tvm::tir::PrimFunc& f) {
  logical_tile_layout_bindings_by_buffer_name_.clear();
  auto ingest_spec = [&](const ffi::Map<ffi::String, ffi::Any>& spec) {
    auto maybe_buffer = spec.Get(ffi::String(schema_key::kBuffer));
    if (!maybe_buffer) {
      return;
    }
    LogicalTileLayoutBinding binding;
    binding.buffer_name = Downcast<ffi::String>(maybe_buffer.value());
    if (auto v = spec.Get(ffi::String("logical_shape"))) {
      binding.logical_shape = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String("local_shape"))) {
      binding.local_shape = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kInverseLogicalIndexVars))) {
      binding.inverse_logical_index_vars = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kInverseLogicalIndexExprs))) {
      binding.inverse_logical_index_exprs = Downcast<ffi::Array<tvm::PrimExpr>>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kThreadExtent))) {
      binding.thread_extent = Downcast<tvm::PrimExpr>(v.value());
    }
    if (auto v = spec.Get(ffi::String(schema_key::kReplicateExtent))) {
      binding.replicate_extent = Downcast<tvm::PrimExpr>(v.value());
    }
    if (binding.buffer_name.empty()) {
      return;
    }
    auto [it, inserted] =
        logical_tile_layout_bindings_by_buffer_name_.emplace(binding.buffer_name, binding);
    if (!inserted) {
      ICHECK(StructuralEqual()(it->second.logical_shape, binding.logical_shape))
          << "Blackhole codegen requires a single logical bridge shape per buffer; "
          << binding.buffer_name;
      ICHECK(StructuralEqual()(it->second.local_shape, binding.local_shape))
          << "Blackhole codegen requires a single local bridge shape per buffer; "
          << binding.buffer_name;
    }
  };

  for (const ffi::Any& item_any : tt_program_projection::GetExecutableArrayField(
           f, "Blackhole codegen", tt_program_projection::executable_key::kBufferDistributionPlans)) {
    auto item = item_any.as<ffi::Map<ffi::String, ffi::Any>>().value_or(
        ffi::Map<ffi::String, ffi::Any>());
    if (item.empty() || !item.Get("logical_shape")) {
      continue;
    }
    ingest_spec(item);
  }
}

const CodeGenBlackhole::LogicalTileLayoutBinding* CodeGenBlackhole::FindLogicalTileLayoutBinding(
    const tvm::tir::VarNode* var) const {
  if (var == nullptr) {
    return nullptr;
  }
  auto it = logical_tile_layout_bindings_by_buffer_name_.find(var->name_hint);
  if (it == logical_tile_layout_bindings_by_buffer_name_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool CodeGenBlackhole::LogicalTileLayoutRequiresGenericBridge(
    const LogicalTileLayoutBinding& binding) const {
  return !binding.inverse_logical_index_exprs.empty() && !binding.local_shape.empty();
}

void CodeGenBlackhole::EmitRuntimeArgLoads(const tvm::tir::PrimFunc &f) {
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_kind_.clear();
  runtime_arg_vars_by_identity_.clear();
  runtime_arg_vars_by_name_.clear();
  per_work_arg_bindings_by_identity_.clear();
  per_work_arg_bindings_.clear();
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_id_by_requirement_name_.clear();
  cb_num_pages_by_requirement_name_.clear();
  cb_publish_pages_by_requirement_name_.clear();
  cb_initial_reserve_pages_by_requirement_name_.clear();
  auto cb_configs = GetCBConfigsForCodegen(f);
  if (!cb_configs.empty()) {
    for (const auto &item : cb_configs) {
      auto cb_info = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (cb_info.empty()) {
        continue;
      }
      int cb_id = -1;
      int page_size = 0;
      int num_pages = 1;
      int publish_pages_per_event = 0;
      int initial_reserve_pages = 0;
      if (auto v = cb_info.Get("cb_id")) {
        cb_id = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("page_size")) {
        page_size = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("num_pages")) {
        num_pages = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("publish_pages_per_event")) {
        publish_pages_per_event = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("initial_reserve_pages")) {
        initial_reserve_pages = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (cb_id >= 0) {
        cb_page_size_by_id_[cb_id] = page_size;
        cb_num_pages_by_id_[cb_id] = std::max(1, num_pages);
        if (auto requirement_names = cb_info.Get("requirement_names")) {
          for (const auto& requirement_name_any :
               Downcast<tvm::ffi::Array<tvm::ffi::Any>>(requirement_names.value())) {
            const std::string requirement_name =
                Downcast<tvm::ffi::String>(requirement_name_any);
            cb_id_by_requirement_name_[requirement_name] = cb_id;
            cb_num_pages_by_requirement_name_[requirement_name] = std::max(1, num_pages);
            if (publish_pages_per_event > 0) {
              cb_publish_pages_by_requirement_name_[requirement_name] =
                  std::max(1, publish_pages_per_event);
            }
            if (initial_reserve_pages > 0) {
              cb_initial_reserve_pages_by_requirement_name_[requirement_name] =
                  std::max(1, initial_reserve_pages);
            }
          }
        }
      }
    }
  }

  ffi::Array<ffi::Any> runtime_args = GetRuntimeArgsForCodegen(f);
  ffi::Array<ffi::Any> per_work_arg_specs = GetPerWorkArgSpecsForCodegen(f);
  for (const auto& item : per_work_arg_specs) {
    auto spec = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (spec.empty()) {
      continue;
    }
    std::string arg_kind;
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgKind)) {
      arg_kind = Downcast<tvm::ffi::String>(v.value());
    }
    PerWorkArgSpecBinding binding;
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kArgIdentity)) {
      binding.arg_identity = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kDescriptorKind)) {
      binding.descriptor_kind = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kValueSource)) {
      binding.value_source = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = spec.Get(::tvm::tl::blackhole_runtime_arg_schema::kConstantValue)) {
      binding.constant_value = Downcast<tvm::Integer>(v.value()).IntValue();
    }
    ICHECK(!binding.arg_identity.empty())
        << "Blackhole codegen requires per-work descriptor arg_identity";
    ICHECK(!binding.descriptor_kind.empty())
        << "Blackhole codegen requires per-work descriptor_kind for "
        << binding.arg_identity;
    ICHECK(!binding.value_source.empty())
        << "Blackhole codegen requires per-work value_source for " << binding.arg_identity;
    per_work_arg_bindings_by_identity_[binding.arg_identity] = binding;
    per_work_arg_bindings_.push_back(std::move(binding));
  }
  ICHECK(!runtime_args.empty())
      << "Blackhole codegen requires executable kernel runtime args";
  if (logical_grid_x_ > 1 || logical_grid_y_ > 1) {
    for (const auto& item : runtime_args) {
      auto arg = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
          tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
      if (arg.empty()) {
        continue;
      }
      std::string arg_kind;
      if (auto v = arg.Get("kind")) {
        arg_kind = Downcast<tvm::ffi::String>(v.value());
      }
      std::string arg_identity;
      if (auto v = arg.Get("identity")) {
        arg_identity = Downcast<tvm::ffi::String>(v.value());
      }
      const bool requires_explicit_per_work_binding =
          arg_kind == "a_tile_start_id" || arg_kind == "a_tile_num_tiles" ||
          arg_kind == "a_tile_stride" || arg_kind == "b_tile_start_id" ||
          arg_kind == "b_tile_num_tiles" || arg_kind == "b_tile_stride" ||
          arg_kind == "output_tile_start_id" || arg_kind == "output_tile_num_tiles" ||
          arg_kind == "output_tile_stride" || arg_kind == "k_tile_start_id" ||
          arg_kind == "num_k_tiles";
      if (!requires_explicit_per_work_binding) {
        continue;
      }
      ICHECK(!arg_identity.empty())
          << "Blackhole codegen requires runtime arg identity before per-work binding for "
          << arg_kind;
      ICHECK(per_work_arg_bindings_by_identity_.count(arg_identity))
          << "Blackhole codegen requires explicit per-work arg binding for runtime arg kind '"
          << arg_kind << "' identity '" << arg_identity
          << "' on multi-work kernels; codegen must not recover block/tile semantics "
          << "from work_linear_id or implicit runtime-arg inference";
    }
  }

  std::unordered_map<std::string, const tvm::tir::VarNode *> buffer_vars_by_name;
  auto record_handle_dtype = [&](const tvm::tir::VarNode* var,
                                 std::optional<DataType> dtype = std::nullopt) {
    if (var == nullptr) {
      return;
    }
    if (dtype.has_value()) {
      handle_data_type_[var] = dtype.value();
      return;
    }
    if (const auto* ptr = var->type_annotation.as<PointerTypeNode>()) {
      if (const auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        handle_data_type_[var] = prim->dtype;
      }
    }
  };
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      buffer_vars_by_name[param->name_hint] = param.get();
      record_handle_dtype(param.get());
    }
  }
  for (const auto &kv : f->buffer_map) {
    const auto &buffer = kv.second;
    buffer_vars_by_name[buffer->name] = buffer->data.get();
    record_handle_dtype(buffer->data.get(), buffer->dtype);
  }
  // Packed Blackhole entrypoints can arrive after MakePackedAPI, where the
  // public function params are no longer the original A/B handles and
  // buffer_map may be empty.  Recover exact runtime-backed buffer vars from the
  // actual TIR body so builtins like read_tile_to_cb(A, ...) still bind.
  tir::PostOrderVisit(f->body, [&](const tvm::runtime::ObjectRef &node) {
    if (const auto *store = node.as<tvm::tir::BufferStoreNode>()) {
      buffer_vars_by_name[store->buffer->name] = store->buffer->data.get();
      record_handle_dtype(store->buffer->data.get(), store->buffer->dtype);
      if (const auto *load = store->value.as<tvm::tir::BufferLoadNode>()) {
        buffer_vars_by_name[load->buffer->name] = load->buffer->data.get();
        record_handle_dtype(load->buffer->data.get(), load->buffer->dtype);
      }
      return;
    }
    if (const auto *load = node.as<tvm::tir::BufferLoadNode>()) {
      buffer_vars_by_name[load->buffer->name] = load->buffer->data.get();
      record_handle_dtype(load->buffer->data.get(), load->buffer->dtype);
      return;
    }
    const auto *call = node.as<tvm::tir::CallNode>();
    if (!call || !call->op->IsInstance<tvm::OpNode>()) {
      return;
    }
    tvm::Op call_op = Downcast<tvm::Op>(call->op);
    const std::string op_name = call_op->name;
    if (op_name == "tl.blackhole.read_tile_to_cb") {
      if (const auto *buffer_var = call->args[0].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
        record_handle_dtype(buffer_var);
      }
      return;
    }
    if (op_name == "tl.blackhole.read_page_to_cb" ||
        op_name == "tl.blackhole.read_bcast_cols_to_cb") {
      if (const auto *buffer_var = call->args[0].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
        record_handle_dtype(buffer_var);
      }
      return;
    }
    if (op_name == "tl.blackhole.write_tile_from_cb") {
      if (const auto *buffer_var = call->args[1].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
        record_handle_dtype(buffer_var);
      }
      return;
    }
    if (op_name == "tl.blackhole.write_page_from_cb") {
      if (const auto *buffer_var = call->args[1].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
        record_handle_dtype(buffer_var);
      }
    }
  });

  int arg_idx = 0;
  for (const auto &item : runtime_args) {
    auto arg_info = item.as<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>().value_or(
        tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>());
    if (arg_info.empty()) {
      continue;
    }

    std::string arg_name = "arg" + std::to_string(arg_idx);
    std::string arg_kind;
    if (auto v = arg_info.Get("name")) {
      arg_name = Downcast<tvm::ffi::String>(v.value());
    }
    if (auto v = arg_info.Get("kind")) {
      arg_kind = Downcast<tvm::ffi::String>(v.value());
    }

    stream << "  uint32_t " << arg_name << " = get_arg_val<uint32_t>(" << arg_idx << ");\n";
    runtime_arg_vars_by_name_[arg_name] = arg_name;
    if (!arg_kind.empty() && !runtime_arg_vars_by_kind_.count(arg_kind)) {
      runtime_arg_vars_by_kind_[arg_kind] = arg_name;
    }
    if (auto v = arg_info.Get("identity")) {
      const std::string arg_identity = Downcast<tvm::ffi::String>(v.value());
      if (!arg_identity.empty() && !runtime_arg_vars_by_identity_.count(arg_identity)) {
        runtime_arg_vars_by_identity_[arg_identity] = arg_name;
      }
    }

    if (IsBufferAddressRuntimeArgKind(arg_kind)) {
      auto buffer_it = arg_info.Get("buffer");
      ICHECK(buffer_it.has_value())
          << "Blackhole codegen requires explicit buffer binding for runtime arg "
          << arg_name << " kind=" << arg_kind;
      const std::string bound_buffer_name = Downcast<tvm::ffi::String>(buffer_it.value());
      ICHECK(!bound_buffer_name.empty())
          << "Blackhole codegen requires non-empty buffer binding for runtime arg "
          << arg_name << " kind=" << arg_kind;
      auto var_it = buffer_vars_by_name.find(bound_buffer_name);
      ICHECK(var_it != buffer_vars_by_name.end())
          << "Blackhole codegen requires runtime arg " << arg_name << " kind=" << arg_kind
          << " buffer=" << bound_buffer_name
          << " to match a formal/TIR buffer identity";
      auto [var_binding_it, var_inserted] =
          buffer_runtime_arg_map_.emplace(var_it->second, arg_name);
      ICHECK(var_inserted || var_binding_it->second == arg_name)
          << "Blackhole codegen buffer " << bound_buffer_name
          << " has conflicting runtime arg bindings " << var_binding_it->second
          << " and " << arg_name;
      auto [name_binding_it, name_inserted] =
          buffer_runtime_arg_map_by_name_.emplace(bound_buffer_name, arg_name);
      ICHECK(name_inserted || name_binding_it->second == arg_name)
          << "Blackhole codegen buffer " << bound_buffer_name
          << " has conflicting runtime arg name bindings " << name_binding_it->second
          << " and " << arg_name;
    }
    ++arg_idx;
  }
  stream << "\n";

  if (!cb_num_pages_by_id_.empty()) {
    stream << "\n";
  }
}

std::string CodeGenBlackhole::GetRuntimeArgVarByKind(const std::string &kind) const {
  auto it = runtime_arg_vars_by_kind_.find(kind);
  ICHECK(it != runtime_arg_vars_by_kind_.end()) << "Missing runtime arg binding for kind: " << kind;
  return it->second;
}

std::string CodeGenBlackhole::GetRuntimeArgVarForBuffer(
    const tvm::PrimExpr &buffer_expr, const char* preferred_kind) const {
  const auto *buffer_var = buffer_expr.as<tvm::tir::VarNode>();
  ICHECK(buffer_var) << "Expected buffer data var in runtime-arg-backed Blackhole builtin";
  auto it = buffer_runtime_arg_map_.find(buffer_var);
  if (it != buffer_runtime_arg_map_.end()) {
    return it->second;
  }
  auto by_name = buffer_runtime_arg_map_by_name_.find(buffer_var->name_hint);
  if (by_name != buffer_runtime_arg_map_by_name_.end()) {
    return by_name->second;
  }

  std::ostringstream available_names;
  bool first = true;
  for (const auto& kv : runtime_arg_vars_by_name_) {
    if (!first) {
      available_names << ", ";
    }
    available_names << kv.first;
    first = false;
  }
  std::ostringstream bound_buffers;
  first = true;
  for (const auto& kv : buffer_runtime_arg_map_by_name_) {
    if (!first) {
      bound_buffers << ", ";
    }
    bound_buffers << kv.first << "->" << kv.second;
    first = false;
  }
  ICHECK(false) << "Missing runtime arg binding for buffer var: " << buffer_var->name_hint
                << ", preferred_kind=" << (preferred_kind ? preferred_kind : "<none>")
                << ", available arg vars=[" << available_names.str() << "]"
                << ", bound buffers=[" << bound_buffers.str() << "]";
  return "";
}

std::optional<DataType> CodeGenBlackhole::TryResolveHandleDataType(
    const tvm::tir::VarNode* var) const {
  if (!var) {
    return std::nullopt;
  }
  if (auto it = handle_data_type_.find(var); it != handle_data_type_.end()) {
    return it->second;
  }
  if (const auto* ptr = var->type_annotation.as<PointerTypeNode>()) {
    if (const auto* prim = ptr->element_type.as<PrimTypeNode>()) {
      return prim->dtype;
    }
  }
  return std::nullopt;
}

DataType CodeGenBlackhole::ResolveHandleDataType(const tvm::tir::VarNode* var, const char* op_name,
                                                 const char* role) const {
  auto maybe_dtype = TryResolveHandleDataType(var);
  ICHECK(maybe_dtype.has_value()) << "Missing " << role << " handle dtype for " << op_name;
  return maybe_dtype.value();
}

bool CodeGenBlackhole::TryPrintCBBackedHandleVar(const tvm::tir::VarNode* var,
                                                 std::ostream& os) const {
  if (var == nullptr || var_idmap_.count(var) != 0U) {
    return false;
  }
  if (!var->type_annotation.as<PointerTypeNode>()) {
    return false;
  }
  if (tvm::tir::GetPtrStorageScope(GetRef<tvm::tir::Var>(var)) != "blackhole.acc") {
    return false;
  }
  auto cb_it = cb_id_by_requirement_name_.find(var->name_hint);
  if (cb_it == cb_id_by_requirement_name_.end()) {
    return false;
  }
  os << "tilelang_cb_write_ptr_bytes_direct(" << cb_it->second << ")";
  return true;
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::VarNode* op, std::ostream& os) {
  if (TryPrintCBBackedHandleVar(op, os)) {
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

int CodeGenBlackhole::ResolveCBId(const tvm::PrimExpr &expr) const {
  const auto *cb_id_imm = expr.as<tvm::tir::IntImmNode>();
  ICHECK(cb_id_imm) << "Blackhole CB operations currently expect constant cb_id";
  const int cb_id = static_cast<int>(cb_id_imm->value);
  ICHECK_GE(cb_id, 0) << "Blackhole codegen expects final cb_id, but saw placeholder " << cb_id;
  return cb_id;
}

void CodeGenBlackhole::PrintResolvedCBId(const tvm::PrimExpr &expr, std::ostream &os) const {
  os << ResolveCBId(expr);
}

void CodeGenBlackhole::PrintPackReconfigDataFormatForCB(int cb_id, std::ostream& os) {
  need_compute_api_h_ = true;
  os << "pack_reconfig_data_format<true>(" << cb_id << ")";
}

int CodeGenBlackhole::GetCBPageSize(int cb_id) const {
  auto it = cb_page_size_by_id_.find(cb_id);
  ICHECK(it != cb_page_size_by_id_.end()) << "Missing CB page size for cb_id=" << cb_id;
  return it->second;
}

int CodeGenBlackhole::GetCBNumPages(int cb_id) const {
  auto it = cb_num_pages_by_id_.find(cb_id);
  ICHECK(it != cb_num_pages_by_id_.end()) << "Missing CB num_pages for cb_id=" << cb_id;
  return it->second;
}

std::string CodeGenBlackhole::GetCBHeadVar(int cb_id) const {
  return "cb_head_" + std::to_string(cb_id);
}

std::string CodeGenBlackhole::GetCBTailVar(int cb_id) const {
  return "cb_tail_" + std::to_string(cb_id);
}

void CodeGenBlackhole::MaybeEmitMathWaypoint(std::ostream& os, const char* code) {
  if (!emit_debug_waypoints_ || core_type_ != CoreType::kTRISC || code == nullptr) {
    return;
  }
  os << "; MATH({ WAYPOINT(\"" << code << "\"); })";
}

void CodeGenBlackhole::RegisterActiveCBWritePtrBinding(int cb_id, const std::string& var_name,
                                                       const std::string& type_name) {
  auto& bindings = active_cb_write_ptr_bindings_[cb_id];
  auto it = std::find_if(bindings.begin(), bindings.end(),
                         [&](const ActiveCBWritePtrBinding& binding) {
                           return binding.var_name == var_name;
                         });
  if (it == bindings.end()) {
    bindings.push_back(ActiveCBWritePtrBinding{var_name, type_name});
    return;
  }
  it->type_name = type_name;
}

void CodeGenBlackhole::UnregisterActiveCBWritePtrBinding(int cb_id,
                                                         const std::string& var_name) {
  auto it = active_cb_write_ptr_bindings_.find(cb_id);
  if (it == active_cb_write_ptr_bindings_.end()) {
    return;
  }
  auto& bindings = it->second;
  bindings.erase(std::remove_if(bindings.begin(), bindings.end(),
                                [&](const ActiveCBWritePtrBinding& binding) {
                                  return binding.var_name == var_name;
                                }),
                 bindings.end());
  if (bindings.empty()) {
    active_cb_write_ptr_bindings_.erase(it);
  }
}

void CodeGenBlackhole::EmitActiveCBWritePtrRefreshes(int cb_id) {
  auto it = active_cb_write_ptr_bindings_.find(cb_id);
  if (it == active_cb_write_ptr_bindings_.end()) {
    return;
  }
  for (const ActiveCBWritePtrBinding& binding : it->second) {
    PrintIndent();
    stream << binding.var_name << " = reinterpret_cast<" << binding.type_name
           << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << "));\n";
  }
}

// ============================================================================
// Visitor Implementation for TT-Metal Builtin Calls
// ============================================================================

void CodeGenBlackhole::VisitExpr_(const tvm::tir::CallNode *op,
                                  std::ostream &os) {
  if (op->op->IsInstance<OpNode>()) {
    Op call_op = Downcast<Op>(op->op);
    if (call_op->name == "tl.infinity") {
      std::ostringstream dtype_os;
      PrintType(op->dtype, dtype_os);
      os << "static_cast<" << dtype_os.str() << ">(1.0f / 0.0f)";
      return;
    }
    if (call_op->name == "tir.exp2") {
      std::ostringstream dtype_os;
      PrintType(op->dtype, dtype_os);
      os << "static_cast<" << dtype_os.str() << ">(tilelang_fast_exp2f(static_cast<float>(";
      PrintExpr(op->args[0], os);
      os << ")))";
      return;
    }
    if ((call_op->name == "tir.call_pure_extern" || call_op->name == "tir.call_extern") &&
        op->args.size() >= 2) {
      if (const auto* callee = op->args[0].as<tvm::tir::StringImmNode>()) {
        const std::string callee_name = callee->value;
        if (callee_name == "exp2f" || callee_name == "exp2") {
          std::ostringstream dtype_os;
          PrintType(op->dtype, dtype_os);
          os << "static_cast<" << dtype_os.str() << ">(tilelang_fast_exp2f(static_cast<float>(";
          PrintExpr(op->args[1], os);
          os << ")))";
          return;
        }
      }
    }
  }
  // Try to handle TT-Metal builtin calls
  if (HandleBlackholeBuiltin(op, os)) {
    return;
  }
  // Fall back to parent class for other calls
  CodeGenCHost::VisitExpr_(op, os);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::EvaluateNode *op) {
  // Handle TT-Metal builtin calls in Evaluate statements
  if (const auto *call = op->value.as<tvm::tir::CallNode>()) {
    std::ostringstream os;
    if (HandleBlackholeBuiltin(call, os)) {
      // This is a Blackhole builtin - print it as a statement
      PrintIndent();
      stream << os.str() << ";\n";
      bool is_cb_reserve_back = call->op.same_as(tir::builtin::blackhole_cb_reserve_back());
      if (!is_cb_reserve_back) {
        if (const auto* builtin = call->op.as<OpNode>()) {
          is_cb_reserve_back = builtin->name == "tl.blackhole.cb_reserve_back";
        }
      }
      if (is_cb_reserve_back) {
        EmitActiveCBWritePtrRefreshes(ResolveCBId(call->args[0]));
      }
      return;
    }
  }
  // Fall back to grandparent class (tvm::codegen::CodeGenC) for non-builtin expressions
  // We need to call the grandparent directly since CodeGenCHost doesn't override VisitStmt_ for EvaluateNode
  tvm::codegen::CodeGenC::VisitStmt_(op);
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::AllocateNode *op) {
  std::string scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  RegisterHandleType(op->buffer_var.get(), op->dtype);

  const bool runtime_managed_storage =
      scope == "shared" || scope == "shared.dyn" || scope == "shared.barrier" ||
      scope.rfind("blackhole.cb", 0) == 0;
  const bool compute_local_fragment_storage =
      scope == "blackhole.acc" && core_type_ == CoreType::kTRISC;
  const bool cb_backed_accumulator =
      compute_local_fragment_storage &&
      cb_id_by_requirement_name_.find(op->buffer_var->name_hint) != cb_id_by_requirement_name_.end();

  if (runtime_managed_storage || (scope == "blackhole.acc" && !compute_local_fragment_storage)) {
    // Blackhole shared / CB allocations are runtime/device-managed
    // resources, not C arrays inside the generated kernel body.  The
    // blackhole.acc scope only materializes inside TRISC kernels, where it can
    // be either CB-backed accumulator storage or ordinary compute-local stack
    // storage depending on whether TT planning assigned a CB requirement.
    this->PrintStmt(op->body);
    return;
  }

  ICHECK(!tvm::tir::is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (cb_backed_accumulator) {
    auto cb_it = cb_id_by_requirement_name_.find(op->buffer_var->name_hint);
    const int cb_id = cb_it->second;
    const int num_pages = cb_num_pages_by_requirement_name_.count(op->buffer_var->name_hint)
                              ? cb_num_pages_by_requirement_name_.at(op->buffer_var->name_hint)
                              : GetCBNumPages(cb_id);
    const int initial_reserve_pages =
        cb_initial_reserve_pages_by_requirement_name_.count(op->buffer_var->name_hint)
            ? cb_initial_reserve_pages_by_requirement_name_.at(op->buffer_var->name_hint)
            : (cb_publish_pages_by_requirement_name_.count(op->buffer_var->name_hint)
                   ? cb_publish_pages_by_requirement_name_.at(op->buffer_var->name_hint)
                   : num_pages);

    std::ostringstream dtype_os;
    PrintType(op->dtype, dtype_os);

    PrintIndent();
    stream << "cb_reserve_back(" << cb_id << ", " << initial_reserve_pages << ");\n";
    PrintIndent();
    stream << dtype_os.str() << "* " << vid << " = reinterpret_cast<" << dtype_os.str()
           << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << "));\n";

    std::optional<std::string> prev_var_id;
    if (auto it = var_idmap_.find(op->buffer_var.get()); it != var_idmap_.end()) {
      prev_var_id = it->second;
    }
    var_idmap_[op->buffer_var.get()] = vid;
    RegisterActiveCBWritePtrBinding(cb_id, vid, dtype_os.str());
    this->PrintStmt(op->body);
    UnregisterActiveCBWritePtrBinding(cb_id, vid);
    if (prev_var_id) {
      var_idmap_[op->buffer_var.get()] = *prev_var_id;
    } else {
      var_idmap_.erase(op->buffer_var.get());
    }
    return;
  }

  PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  PrintStorageScope(scope, stream);
  PrintType(op->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  std::optional<std::string> prev_var_id;
  if (auto it = var_idmap_.find(op->buffer_var.get()); it != var_idmap_.end()) {
    prev_var_id = it->second;
  }
  var_idmap_[op->buffer_var.get()] = vid;
  this->PrintStmt(op->body);
  if (prev_var_id) {
    var_idmap_[op->buffer_var.get()] = *prev_var_id;
  } else {
    var_idmap_.erase(op->buffer_var.get());
  }
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::FloorDivNode *op,
                                   std::ostream &os) {
  // FloorDiv is not implemented in base CodeGenC
  // For Blackhole, we can implement it as regular division for positive integers
  // Or use a more complex expression: ((a >= 0 ? a : a - b + 1) / b)
  // For simplicity, we use regular division assuming positive values
  // TODO: Add proper floor div handling for negative values if needed
  os << "(";
  VisitExpr(op->a, os);
  os << " / ";
  VisitExpr(op->b, os);
  os << ")";
}

void CodeGenBlackhole::VisitExpr_(const tvm::tir::FloorModNode *op,
                                   std::ostream &os) {
  // FloorMod is not implemented in base CodeGenC
  // For Blackhole, implement as regular modulo for positive integers
  // TODO: Add proper floor mod handling for negative values if needed
  os << "(";
  VisitExpr(op->a, os);
  os << " % ";
  VisitExpr(op->b, os);
  os << ")";
}

void CodeGenBlackhole::BindThreadIndex(const tvm::tir::IterVar &iv) {
  // For Blackhole, we need to handle thread/block indices differently than CUDA
  // Blackhole uses a different parallelism model based on Tensix cores

  if (var_idmap_.count(iv->var.get())) {
    return;
  }

  std::string thread_tag = iv->thread_tag;
  auto runtime_arg_for_binding = [&](const PerWorkArgSpecBinding& binding)
      -> std::optional<std::string> {
    auto it = runtime_arg_vars_by_identity_.find(binding.arg_identity);
    if (it == runtime_arg_vars_by_identity_.end()) {
      return std::nullopt;
    }
    return it->second;
  };
  const bool row_major_grid = linearization_ == "row_major" && logical_grid_x_ > 0;
  auto resolve_explicit_axis = [&](bool want_x) -> std::optional<std::string> {
    for (const auto& binding : per_work_arg_bindings_) {
      if (binding.descriptor_kind !=
          ::tvm::tl::blackhole_runtime_arg_schema::kDescriptorTileStart) {
        continue;
      }
      auto arg_var = runtime_arg_for_binding(binding);
      if (!arg_var.has_value()) {
        continue;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
        if (want_x) {
          if (row_major_grid) {
            return "(" + arg_var.value() + " % " + std::to_string(logical_grid_x_) + ")";
          }
          return arg_var;
        }
        if (row_major_grid) {
          return "(" + arg_var.value() + " / " + std::to_string(logical_grid_x_) + ")";
        }
        return std::string("0 /* explicit_linear_work_descriptor_y */");
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockX) {
        if (want_x) {
          return arg_var;
        }
        continue;
      }
      if (binding.value_source ==
          ::tvm::tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockY) {
        if (!want_x) {
          return arg_var;
        }
        continue;
      }
    }
    return std::nullopt;
  };
  const auto explicit_block_x = resolve_explicit_axis(/*want_x=*/true);
  const auto explicit_block_y = resolve_explicit_axis(/*want_x=*/false);
  const bool has_explicit_work_descriptor =
      explicit_block_x.has_value() || explicit_block_y.has_value();

  // Map CUDA-style thread indices to Blackhole concepts
  // For staged single-core execution, block coordinates must come from the
  // strongest explicit work contract available. If the ABI already carries a
  // buffer-specific tile descriptor, consume that descriptor directly.
  if (thread_tag == "blockIdx.x") {
    if (explicit_block_x.has_value()) {
      var_idmap_[iv->var.get()] = explicit_block_x.value();
    } else if (has_explicit_work_descriptor) {
      var_idmap_[iv->var.get()] = "0 /* explicit_work_descriptor_x */";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_x */";
    }
  } else if (thread_tag == "blockIdx.y") {
    if (explicit_block_y.has_value()) {
      var_idmap_[iv->var.get()] = explicit_block_y.value();
    } else if (has_explicit_work_descriptor) {
      var_idmap_[iv->var.get()] = "0 /* explicit_work_descriptor_y */";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_y */";
    }
  } else if (thread_tag == "blockIdx.z") {
    var_idmap_[iv->var.get()] = "0 /* core_z */";
  } else if (thread_tag == "threadIdx.x") {
    // For Blackhole, threadIdx.x could map to worker threads within a core
    // For now, use the variable name directly
    var_idmap_[iv->var.get()] = iv->var->name_hint;
    thread_idx_x_expr_ = iv->var->name_hint;
  } else if (thread_tag == "threadIdx.y") {
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  } else if (thread_tag == "threadIdx.z") {
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  } else {
    // Unknown thread tag - use the variable name
    var_idmap_[iv->var.get()] = iv->var->name_hint;
  }
}

void CodeGenBlackhole::PrintStorageScope(const std::string &scope,
                                          std::ostream &os) {
  // Blackhole uses different memory model than CUDA
  // - "global" -> DRAM (no keyword needed)
  // - "shared" / "shared.dyn" -> Circular Buffer (CB) - handled separately
  // - "blackhole.cb.*" -> runtime/device-managed resource
  // - "blackhole.acc" -> compute-local stack storage emitted in TRISC kernels
  // - "local" -> Local registers (no keyword needed)
  // - "warp" / "warp::sync" -> Not applicable for Blackhole

  if (scope == "shared" || scope == "shared.dyn" ||
      scope == "shared.barrier" ||
      scope.rfind("blackhole.cb", 0) == 0) {
    // For Blackhole, shared memory is allocated as Circular Buffers
    // and emitted outside the generated C body.
    os << "/* blackhole managed resource */ ";
  } else if (scope == "local") {
    // Local scope doesn't need a qualifier in C++
    // Variables are local by default
  } else if (scope == "global") {
    // Global memory - no qualifier needed
  } else if (scope.find("warp") == 0) {
    // Warp scope not applicable for Blackhole
    // Blackhole doesn't have warps like CUDA
  } else {
    // Unknown scope - add a comment
    os << "/* scope: " << scope << " */ ";
  }
}

void CodeGenBlackhole::VisitStmt_(const tvm::tir::AttrStmtNode *op) {
  // Handle Blackhole-specific attribute statements
  // For TT-Metal kernels, we handle specific attr_keys differently

  if (op->attr_key == tir::attr::thread_extent) {
    // For thread_extent, we need to bind the thread index variable
    // This is similar to CUDA but maps to Blackhole core/thread model
    auto iv = Downcast<tvm::tir::IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      const std::string thread_tag = iv->thread_tag;
      const bool is_thread_idx = thread_tag.rfind("threadIdx.", 0) == 0;
      if (is_thread_idx) {
        const bool thread_var_used = tir::UsesVar(
            op->body, [thread_var = iv->var.get()](const tir::VarNode* var) {
              return var == thread_var;
            });
        std::optional<std::string> prev_var_id;
        if (auto it = var_idmap_.find(iv->var.get()); it != var_idmap_.end()) {
          prev_var_id = it->second;
        }
        auto restore_thread_var = [&]() {
          if (prev_var_id) {
            var_idmap_[iv->var.get()] = *prev_var_id;
          } else {
            var_idmap_.erase(iv->var.get());
          }
        };
        std::vector<std::pair<const tvm::tir::VarNode*, std::optional<std::string>>>
            nested_thread_prev_ids;
        auto restore_nested_thread_vars = [&]() {
          for (auto it = nested_thread_prev_ids.rbegin(); it != nested_thread_prev_ids.rend();
               ++it) {
            if (it->second) {
              var_idmap_[it->first] = *(it->second);
            } else {
              var_idmap_.erase(it->first);
            }
          }
        };
        auto emit_with_thread_binding = [&](const std::string& binding,
                                            const tvm::tir::Stmt& stmt) {
          var_idmap_[iv->var.get()] = binding;
          this->VisitStmt(stmt);
        };
        tvm::tir::Stmt partition_body = op->body;
        while (const auto* nested_attr = partition_body.as<tvm::tir::AttrStmtNode>()) {
          if (nested_attr->attr_key != tir::attr::thread_extent) {
            break;
          }
          auto nested_iv = Downcast<tvm::tir::IterVar>(nested_attr->node);
          const std::string nested_tag = nested_iv->thread_tag;
          const bool nested_is_unit_thread =
              nested_tag.rfind("threadIdx.", 0) == 0 && tir::is_one(nested_attr->value);
          if (!nested_is_unit_thread) {
            break;
          }
          std::optional<std::string> nested_prev_var_id;
          if (auto it = var_idmap_.find(nested_iv->var.get()); it != var_idmap_.end()) {
            nested_prev_var_id = it->second;
          }
          nested_thread_prev_ids.push_back({nested_iv->var.get(), nested_prev_var_id});
          var_idmap_[nested_iv->var.get()] = "0";
          partition_body = nested_attr->body;
        }
        if (!thread_var_used || tir::is_one(op->value)) {
          emit_with_thread_binding("0", partition_body);
          restore_nested_thread_vars();
          restore_thread_var();
          return;
        } else {
          const std::vector<ThreadEmissionPiece> pieces =
              BuildThreadEmissionPieces(partition_body, iv->var.get());
          const bool has_threaded_piece =
              std::any_of(pieces.begin(), pieces.end(), [](const ThreadEmissionPiece& piece) {
                return piece.uses_thread_var;
              });
          if (!has_threaded_piece) {
            emit_with_thread_binding("0", partition_body);
            restore_nested_thread_vars();
            restore_thread_var();
            return;
          }

          auto emit_thread_loop = [&](const std::vector<tvm::tir::Stmt>& loop_body_stmts) {
            if (loop_body_stmts.empty()) {
              return;
            }
            std::ostringstream dtype_os;
            PrintType(iv->var.dtype(), dtype_os);
            const std::string loop_var = iv->var->name_hint;
            var_idmap_[iv->var.get()] = loop_var;
            PrintIndent();
            stream << "for (" << dtype_os.str() << " " << loop_var << " = 0; " << loop_var
                   << " < ";
            PrintExpr(op->value, stream);
            stream << "; ++" << loop_var << ") {\n";
            int scope_id = BeginScope();
            tvm::tir::Stmt loop_body =
                loop_body_stmts.size() == 1 ? loop_body_stmts.front()
                                            : tvm::tir::SeqStmt::Flatten(loop_body_stmts);
            this->VisitStmt(loop_body);
            EndScope(scope_id);
            PrintIndent();
            stream << "}\n";
          };

          std::vector<tvm::tir::Stmt> pending_threaded_stmts;
          for (const auto& piece : pieces) {
            if (piece.uses_thread_var) {
              pending_threaded_stmts.push_back(piece.stmt);
              continue;
            }
            emit_thread_loop(pending_threaded_stmts);
            pending_threaded_stmts.clear();
            emit_with_thread_binding("0", piece.stmt);
          }
          emit_thread_loop(pending_threaded_stmts);
          restore_nested_thread_vars();
          restore_thread_var();
          return;
        }
      }
    }
    if (!var_idmap_.count(iv->var.get())) {
      BindThreadIndex(iv);
    }
    this->VisitStmt(op->body);
  } else if (op->attr_key == tir::attr::virtual_thread ||
             op->attr_key == tir::attr::coproc_scope ||
             op->attr_key == tir::attr::coproc_uop_scope) {
    // For virtual_thread and coproc attributes, just visit the body
    // These are CUDA-specific constructs that don't directly apply to Blackhole
    this->VisitStmt(op->body);
  } else if (op->attr_key == tir::attr::realize_scope ||
             op->attr_key == tir::attr::storage_alignment) {
    // Storage scope/alignment annotations - just visit the body
    // The Blackhole CB (circular buffer) system handles this differently
    this->VisitStmt(op->body);
  } else if (op->attr_key == "pragma_unroll") {
    // Unroll pragma - just visit the body
    // Blackhole compiler handles unrolling via TT-Metal
    this->VisitStmt(op->body);
  } else if (op->attr_key == "pragma") {
    // Generic pragma - skip for now
    this->VisitStmt(op->body);
  } else {
    // For all other attributes, fall back to parent class
    CodeGenCHost::VisitStmt_(op);
  }
}

bool CodeGenBlackhole::HandleBlackholeBuiltin(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  std::string op_name = call_op->name;

  // Check for TT-Metal builtin prefix
  const std::string prefix = "tl.blackhole.";
  if (op_name.find(prefix) != 0) return false;

  std::string builtin_name = op_name.substr(prefix.length());

  // Handle each builtin type
  if (builtin_name == "cb_reserve_back") {
    PrintCBReserveBack(op, os);
    return true;
  } else if (builtin_name == "cb_push_back") {
    PrintCBPushBack(op, os);
    return true;
  } else if (builtin_name == "cb_wait_front") {
    PrintCBWaitFront(op, os);
    return true;
  } else if (builtin_name == "cb_pop_front") {
    PrintCBPopFront(op, os);
    return true;
  } else if (builtin_name == "noc_async_read") {
    PrintNOCAsyncRead(op, os);
    return true;
  } else if (builtin_name == "noc_async_write") {
    PrintNOCAsyncWrite(op, os);
    return true;
  } else if (builtin_name == "noc_async_read_barrier") {
    PrintNOCReadBarrier(os);
    return true;
  } else if (builtin_name == "noc_async_write_barrier") {
    PrintNOCWriteBarrier(os);
    return true;
  } else if (builtin_name == "read_tile_to_cb") {
    PrintReadTileToCB(op, os);
    return true;
  } else if (builtin_name == "read_page_to_cb") {
    PrintReadPageToCB(op, os);
    return true;
  } else if (builtin_name == "read_bcast_cols_to_cb") {
    PrintReadBcastColsToCB(op, os);
    return true;
  } else if (builtin_name == "write_tile_from_cb") {
    PrintWriteTileFromCB(op, os);
    return true;
  } else if (builtin_name == "write_page_from_cb") {
    PrintWritePageFromCB(op, os);
    return true;
  } else if (builtin_name == "get_semaphore") {
    PrintGetSemaphore(op, os);
    return true;
  } else if (builtin_name == "runtime_arg_u32") {
    PrintRuntimeArgU32(op, os);
    return true;
  } else if (builtin_name == "semaphore_wait") {
    PrintSemaphoreWait(op, os);
    return true;
  } else if (builtin_name == "semaphore_set") {
    PrintSemaphoreSet(op, os);
    return true;
  } else if (builtin_name == "semaphore_inc_remote") {
    PrintSemaphoreIncRemote(op, os);
    return true;
  } else if (builtin_name == "semaphore_set_remote") {
    PrintSemaphoreSetRemote(op, os);
    return true;
  } else if (builtin_name == "mm_init") {
    PrintMMInit(op, os);
    return true;
  } else if (builtin_name == "reconfig_data_format") {
    PrintReconfigDataFormat(op, os);
    return true;
  } else if (builtin_name == "mm_init_short") {
    PrintMMInitShort(op, os);
    return true;
  } else if (builtin_name == "mm_init_short_with_dt") {
    PrintMMInitShortWithDT(op, os);
    return true;
  } else if (builtin_name == "matmul_tiles") {
    PrintMatmulTiles(op, os);
    return true;
  } else if (builtin_name == "tile_regs_acquire") {
    PrintTileRegsAcquire(os);
    return true;
  } else if (builtin_name == "tile_regs_commit") {
    PrintTileRegsCommit(os);
    return true;
  } else if (builtin_name == "tile_regs_wait") {
    PrintTileRegsWait(os);
    return true;
  } else if (builtin_name == "tile_regs_release") {
    PrintTileRegsRelease(os);
    return true;
  } else if (builtin_name == "pack_tile") {
    PrintPackTile(op, os);
    return true;
  } else if (builtin_name == "pack_reconfig_data_format") {
    PrintPackReconfigDataFormat(op, os);
    return true;
  } else if (builtin_name == "copy_tile_to_dst_init_short") {
    PrintCopyTileToDstInitShort(op, os);
    return true;
  } else if (builtin_name == "copy_tile_to_dst_init_short_with_dt") {
    PrintCopyTileToDstInitShortWithDT(op, os);
    return true;
  } else if (builtin_name == "copy_tile") {
    PrintCopyTile(op, os);
    return true;
  } else if (builtin_name == "binary_op_init_common") {
    PrintBinaryOpInitCommon(op, os);
    return true;
  } else if (builtin_name == "unary_op_init_common") {
    PrintUnaryOpInitCommon(op, os);
    return true;
  } else if (builtin_name == "add_tiles_init") {
    PrintAddTilesInit(op, os);
    return true;
  } else if (builtin_name == "add_tiles") {
    PrintAddTiles(op, os);
    return true;
  } else if (builtin_name == "add_bcast_rows_init_short") {
    PrintAddBcastRowsInitShort(op, os);
    return true;
  } else if (builtin_name == "add_bcast_cols_init_short") {
    PrintAddBcastColsInitShort(op, os);
    return true;
  } else if (builtin_name == "add_tiles_bcast_rows") {
    PrintAddTilesBcastRows(op, os);
    return true;
  } else if (builtin_name == "add_tiles_bcast_cols") {
    PrintAddTilesBcastCols(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_init") {
    PrintMulTilesInit(op, os);
    return true;
  } else if (builtin_name == "mul_tiles") {
    PrintMulTiles(op, os);
    return true;
  } else if (builtin_name == "mul_bcast_rows_init_short") {
    PrintMulBcastRowsInitShort(op, os);
    return true;
  } else if (builtin_name == "mul_bcast_cols_init_short") {
    PrintMulBcastColsInitShort(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_bcast_rows") {
    PrintMulTilesBcastRows(op, os);
    return true;
  } else if (builtin_name == "mul_tiles_bcast_cols") {
    PrintMulTilesBcastCols(op, os);
    return true;
  } else if (builtin_name == "reduce_init") {
    PrintReduceInit(op, os);
    return true;
  } else if (builtin_name == "reduce_tile") {
    PrintReduceTile(op, os);
    return true;
  } else if (builtin_name == "reduce_uninit") {
    PrintReduceUninit(op, os);
    return true;
  } else if (builtin_name == "binary_max_tile_init") {
    PrintBinaryMaxTileInit(op, os);
    return true;
  } else if (builtin_name == "binary_max_tile") {
    PrintBinaryMaxTile(op, os);
    return true;
  } else if (builtin_name == "div_binary_tile_init") {
    PrintDivBinaryTileInit(op, os);
    return true;
  } else if (builtin_name == "div_binary_tile") {
    PrintDivBinaryTile(op, os);
    return true;
  } else if (builtin_name == "exp_tile_init") {
    PrintExpTileInit(op, os);
    return true;
  } else if (builtin_name == "exp_tile") {
    PrintExpTile(op, os);
    return true;
  } else if (builtin_name == "exp2_tile_init") {
    PrintExp2TileInit(op, os);
    return true;
  } else if (builtin_name == "exp2_tile") {
    PrintExp2Tile(op, os);
    return true;
  } else if (builtin_name == "recip_tile_init") {
    PrintRecipTileInit(op, os);
    return true;
  } else if (builtin_name == "recip_tile") {
    PrintRecipTile(op, os);
    return true;
  } else if (builtin_name == "fill_fragment") {
    PrintFillFragment(op, os);
    return true;
  } else if (builtin_name == "add_fragment") {
    PrintAddFragment(op, os);
    return true;
  } else if (builtin_name == "add_fragment_from_cb_front") {
    PrintAddFragmentFromCBFront(op, os);
    return true;
  } else if (builtin_name == "pack_untilize_slice") {
    PrintPackUntilizeSlice(op, os);
    return true;
  } else if (builtin_name == "pack_untilize_tile") {
    PrintPackUntilizeTile(op, os);
    return true;
  } else if (builtin_name == "tilize_local_fragment_slice") {
    PrintTilizeLocalFragmentSlice(op, os);
    return true;
  } else if (builtin_name == "tilize_cast_fragment_slice") {
    PrintTilizeCastFragmentSlice(op, os);
    return true;
  } else if (builtin_name == "pack_fill_fragment_to_tiled_cb") {
    PrintPackFillFragmentToTiledCB(op, os);
    return true;
  } else if (builtin_name == "untilize_cb_front_tile") {
    PrintUntilizeCBFrontTile(op, os);
    return true;
  } else if (builtin_name == "untilize_cb_front_tile_fragment") {
    PrintUntilizeCBFrontTileFragment(op, os);
    return true;
  } else if (builtin_name == "cast_fragment_slice") {
    PrintCastFragmentSlice(op, os);
    return true;
  }

  return false;
}

// ============================================================================
// TT-Metal Builtin Print Functions
// ============================================================================

void CodeGenBlackhole::PrintCBReserveBack(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_reserve_back(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPushBack(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "cb_push_back(";
  os << cb_id;
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintCBWaitFront(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "cb_wait_front(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);  // num_tiles
  os << ")";
}

void CodeGenBlackhole::PrintCBPopFront(const tvm::tir::CallNode *op,
                                       std::ostream &os) {
  need_dataflow_api_h_ = true;
  const int cb_id = ResolveCBId(op->args[0]);
  os << "cb_pop_front(";
  os << cb_id;
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncRead(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCAsyncWrite(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write(";
  PrintExpr(op->args[0], os);  // src_addr
  os << ", ";
  PrintExpr(op->args[1], os);  // dst_addr
  os << ", ";
  PrintExpr(op->args[2], os);  // size
  os << ")";
}

void CodeGenBlackhole::PrintNOCReadBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_read_barrier()";
}

void CodeGenBlackhole::PrintNOCWriteBarrier(std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_async_write_barrier()";
}

namespace {

int ResolveCompileTimeAccessorOffset(const tvm::tir::CallNode* op,
                                     int arg_index,
                                     const char* builtin_name) {
  const auto* accessor_offset = op->args[arg_index].as<tvm::tir::IntImmNode>();
  ICHECK(accessor_offset)
      << "Blackhole codegen currently supports only compile-time-only accessor slots; "
      << builtin_name << " expects constant accessor compile-time offset";
  return static_cast<int>(accessor_offset->value);
}

void EmitTensorAccessorGenerator(std::ostream& os,
                                 const char* prefix,
                                 int accessor_offset,
                                 const std::string& addr_var,
                                 const std::string& size_expr = "") {
  os << "; constexpr auto " << prefix << "_accessor_args = TensorAccessorArgs<"
     << accessor_offset << ">(); const auto " << prefix << "_gen = TensorAccessor("
     << prefix << "_accessor_args, " << addr_var;
  if (!size_expr.empty()) {
    os << ", " << size_expr;
  }
  os << "); ";
}

}  // namespace

void CodeGenBlackhole::PrintReadTileToCB(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveCompileTimeAccessorOffset(op, /*arg_index=*/4, "tl.blackhole.read_tile_to_cb");
  os << "{ ";
  os << "const uint32_t tile_index = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t tile_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ")";
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var, "tile_bytes");
  os << "noc_async_read_tile(tile_index, src_gen, cb_l1_addr); ";
  os << "noc_async_read_barrier(); }";
}

void CodeGenBlackhole::PrintWriteTileFromCB(const tvm::tir::CallNode *op,
                                            std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string dst_addr_var = GetRuntimeArgVarForBuffer(op->args[1], "output_buffer_addr");
  const int cb_id = ResolveCBId(op->args[0]);
  const int accessor_offset =
      ResolveCompileTimeAccessorOffset(op, /*arg_index=*/4, "tl.blackhole.write_tile_from_cb");
  os << "{ ";
  os << "const uint32_t tile_index = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t tile_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ")";
  EmitTensorAccessorGenerator(os, "dst", accessor_offset, dst_addr_var, "tile_bytes");
  os << "noc_async_write_tile(tile_index, dst_gen, cb_l1_addr); ";
  os << "noc_async_write_barrier(); }";
}

void CodeGenBlackhole::PrintReadPageToCB(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveCompileTimeAccessorOffset(op, /*arg_index=*/4, "tl.blackhole.read_page_to_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << ") + ";
  PrintExpr(op->args[5], os);
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var);
  os << "const uint64_t src_noc_addr = src_gen.get_noc_addr(page_id); ";
  os << "noc_async_read(src_noc_addr, cb_l1_addr, page_bytes); }";
}

void CodeGenBlackhole::PrintReadBcastColsToCB(const tvm::tir::CallNode *op,
                                              std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string src_addr_var = GetRuntimeArgVarForBuffer(op->args[0], "input_buffer_addr");
  const int cb_id = ResolveCBId(op->args[2]);
  const int accessor_offset =
      ResolveCompileTimeAccessorOffset(op, /*arg_index=*/4,
                                       "tl.blackhole.read_bcast_cols_to_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[1], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t vector_len = ";
  PrintExpr(op->args[5], os);
  os << "; const uint32_t cb_l1_addr = get_write_ptr(" << cb_id << "); ";
  EmitTensorAccessorGenerator(os, "src", accessor_offset, src_addr_var);
  os << "const uint64_t src_noc_addr = src_gen.get_noc_addr(page_id); "
        "volatile uint16_t* dst_bits = reinterpret_cast<volatile uint16_t*>(cb_l1_addr); "
        "const uint32_t scratch_byte_offset = 2048u - page_bytes; "
        "const uint32_t scratch_l1_addr = cb_l1_addr + scratch_byte_offset; "
        "noc_async_read(src_noc_addr, scratch_l1_addr, page_bytes); "
        "noc_async_read_barrier(); "
        "const uint32_t scratch_element_offset = scratch_byte_offset / 2u; "
        "const uint32_t page_elements = page_bytes / 2u; "
        "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
        "constexpr uint32_t kTileCols = 32; "
        "const uint32_t rows = vector_len < 32u ? vector_len : 32u; "
        "for (uint32_t i = 0; i < 1024u; ++i) { "
        "if (i < scratch_element_offset || i >= scratch_element_offset + page_elements) { "
        "dst_bits[i] = 0; } } "
        "for (uint32_t row = 0; row < rows; ++row) { "
        "if (row >= page_elements) { continue; } "
        "const uint32_t row_in_tile = row; "
        "const uint32_t face_row = row_in_tile / kFaceRows; "
        "const uint32_t row_in_face = row_in_tile % kFaceRows; "
        "const uint32_t dst_element = "
        "face_row * (kFaceRows * kTileCols) + row_in_face * kFaceCols; "
        "dst_bits[dst_element] = dst_bits[scratch_element_offset + row]; "
        "} "
        "for (uint32_t i = 0; i < page_elements; ++i) { "
        "dst_bits[scratch_element_offset + i] = 0; } }";
}

void CodeGenBlackhole::PrintWritePageFromCB(const tvm::tir::CallNode *op,
                                            std::ostream &os) {
  need_dataflow_api_h_ = true;
  const std::string dst_addr_var = GetRuntimeArgVarForBuffer(op->args[1], "output_buffer_addr");
  const int cb_id = ResolveCBId(op->args[0]);
  const int accessor_offset =
      ResolveCompileTimeAccessorOffset(op, /*arg_index=*/4, "tl.blackhole.write_page_from_cb");
  os << "{ ";
  os << "const uint32_t page_id = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t page_bytes = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t cb_l1_addr = get_read_ptr(" << cb_id << ") + ";
  PrintExpr(op->args[5], os);
  EmitTensorAccessorGenerator(os, "dst", accessor_offset, dst_addr_var);
  os << "const uint64_t dst_noc_addr = dst_gen.get_noc_addr(page_id); ";
  os << "noc_async_write(cb_l1_addr, dst_noc_addr, page_bytes); }";
}

void CodeGenBlackhole::PrintMMInit(const tvm::tir::CallNode *op,
                                   std::ostream &os) {
  need_compute_api_h_ = true;
  os << "mm_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintReconfigDataFormat(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.reconfig_data_format expects 2 arguments";
  os << "reconfig_data_format(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMMInitShort(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.mm_init_short expects 2 arguments";
  os << "mm_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMMInitShortWithDT(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3) << "tl.blackhole.mm_init_short_with_dt expects 3 arguments";
  os << "mm_init_short_with_dt(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintMatmulTiles(const tvm::tir::CallNode *op,
                                        std::ostream &os) {
  need_compute_api_h_ = true;
  os << "matmul_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);  // in0_tile_index
  os << ", ";
  PrintExpr(op->args[3], os);  // in1_tile_index
  os << ", ";
  PrintExpr(op->args[4], os);  // dst_tile_index
  os << ")";
}

void CodeGenBlackhole::PrintTileRegsAcquire(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_acquire()";
}

void CodeGenBlackhole::PrintTileRegsCommit(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_commit()";
}

void CodeGenBlackhole::PrintTileRegsWait(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_wait()";
}

void CodeGenBlackhole::PrintTileRegsRelease(std::ostream &os) {
  need_compute_api_h_ = true;
  os << "tile_regs_release()";
}

void CodeGenBlackhole::PrintPackTile(const tvm::tir::CallNode *op,
                                     std::ostream &os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.pack_tile expects 2 or 3 arguments";
  os << "pack_tile(";
  PrintExpr(op->args[0], os);  // src_tile_index
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  if (op->args.size() == 3) {
    os << ", ";
    PrintExpr(op->args[2], os);  // dst_tile_index
  }
  os << ")";
}

void CodeGenBlackhole::PrintPackReconfigDataFormat(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  ICHECK_EQ(op->args.size(), 1)
      << "tl.blackhole.pack_reconfig_data_format expects 1 argument";
  PrintPackReconfigDataFormatForCB(ResolveCBId(op->args[0]), os);
}

void CodeGenBlackhole::PrintCopyTileToDstInitShort(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.copy_tile_to_dst_init_short expects 1 argument";
  os << "copy_tile_to_dst_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintCopyTileToDstInitShortWithDT(const tvm::tir::CallNode* op,
                                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.copy_tile_to_dst_init_short_with_dt expects 2 arguments";
  os << "copy_tile_to_dst_init_short_with_dt(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintCopyTile(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3) << "tl.blackhole.copy_tile expects 3 arguments";
  os << "copy_tile(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintBinaryOpInitCommon(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3)
      << "tl.blackhole.binary_op_init_common expects 3 arguments";
  os << "binary_op_init_common(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintUnaryOpInitCommon(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.unary_op_init_common expects 2 arguments";
  os << "unary_op_init_common(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.add_tiles_init expects 2 or 3 arguments";
  os << "add_tiles_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  if (op->args.size() == 3) {
    os << ", ";
    PrintExpr(op->args[2], os);
  }
  os << ")";
}

void CodeGenBlackhole::PrintAddTiles(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles expects 5 arguments";
  os << "add_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddBcastRowsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.add_bcast_rows_init_short expects 2 arguments";
  os << "add_bcast_rows_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddBcastColsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.add_bcast_cols_init_short expects 2 arguments";
  os << "add_bcast_cols_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesBcastRows(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles_bcast_rows expects 5 arguments";
  os << "add_tiles_bcast_rows(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintAddTilesBcastCols(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.add_tiles_bcast_cols expects 5 arguments";
  os << "add_tiles_bcast_cols(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.mul_tiles_init expects 2 arguments";
  os << "mul_tiles_init(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTiles(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles expects 5 arguments";
  os << "mul_tiles(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulBcastRowsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.mul_bcast_rows_init_short expects 2 arguments";
  os << "mul_bcast_rows_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulBcastColsInitShort(const tvm::tir::CallNode* op,
                                                  std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2)
      << "tl.blackhole.mul_bcast_cols_init_short expects 2 arguments";
  os << "mul_bcast_cols_init_short(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesBcastRows(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles_bcast_rows expects 5 arguments";
  os << "mul_tiles_bcast_rows(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintMulTilesBcastCols(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.mul_tiles_bcast_cols expects 5 arguments";
  os << "mul_tiles_bcast<BroadcastType::COL>(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceInit(const tvm::tir::CallNode* op,
                                       std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 5) << "tl.blackhole.reduce_init expects 5 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[3], "tl.blackhole.reduce_init",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[4], "tl.blackhole.reduce_init",
                                                  "reduce_dim");
  os << "reduce_init<" << ReduceKindToTTMetal(reduce_kind, "tl.blackhole.reduce_init") << ", "
     << ReduceDimToTTMetal(reduce_dim, "tl.blackhole.reduce_init") << ">(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintResolvedCBId(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceTile(const tvm::tir::CallNode* op,
                                       std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 7) << "tl.blackhole.reduce_tile expects 7 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[5], "tl.blackhole.reduce_tile",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[6], "tl.blackhole.reduce_tile",
                                                  "reduce_dim");
  os << "reduce_tile<" << ReduceKindToTTMetal(reduce_kind, "tl.blackhole.reduce_tile") << ", "
     << ReduceDimToTTMetal(reduce_dim, "tl.blackhole.reduce_tile") << ">(";
  PrintResolvedCBId(op->args[0], os);
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << ", ";
  PrintExpr(op->args[4], os);
  os << ")";
}

void CodeGenBlackhole::PrintReduceUninit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 2) << "tl.blackhole.reduce_uninit expects 2 arguments";
  const std::string reduce_kind = RequireStringImm(op->args[0], "tl.blackhole.reduce_uninit",
                                                   "reduce_kind");
  const std::string reduce_dim = RequireStringImm(op->args[1], "tl.blackhole.reduce_uninit",
                                                  "reduce_dim");
  (void)reduce_kind;
  (void)reduce_dim;
  os << "reduce_uninit<false>()";
}

void CodeGenBlackhole::PrintBinaryMaxTileInit(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "binary_max_tile_init()";
}

void CodeGenBlackhole::PrintBinaryMaxTile(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 3 || op->args.size() == 4)
      << "tl.blackhole.binary_max_tile expects 3 or 4 arguments";
  os << "binary_max_tile(";
  PrintExpr(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  if (op->args.size() == 4) {
    const auto* mode = op->args[3].as<StringImmNode>();
    ICHECK(mode != nullptr)
        << "tl.blackhole.binary_max_tile vector_mode must be a string literal";
    os << ", (int)VectorMode::" << mode->value;
  }
  os << ")";
}

void CodeGenBlackhole::PrintDivBinaryTileInit(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "div_binary_tile_init()";
}

void CodeGenBlackhole::PrintDivBinaryTile(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 3) << "tl.blackhole.div_binary_tile expects 3 arguments";
  os << "div_binary_tile(";
  PrintExpr(op->args[0], os);
  os << ", ";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ")";
}

void CodeGenBlackhole::PrintExpTileInit(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "exp_tile_init()";
}

void CodeGenBlackhole::PrintExpTile(const tvm::tir::CallNode* op,
                                    std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.exp_tile expects 1 argument";
  os << "exp_tile(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintExp2TileInit(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  (void)op;
  need_compute_api_h_ = true;
  os << "exp2_tile_init()";
}

void CodeGenBlackhole::PrintExp2Tile(const tvm::tir::CallNode* op,
                                     std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK_EQ(op->args.size(), 1) << "tl.blackhole.exp2_tile expects 1 argument";
  os << "exp2_tile(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintRecipTileInit(const tvm::tir::CallNode* op,
                                          std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.empty() || op->args.size() == 1)
      << "tl.blackhole.recip_tile_init expects 0 or 1 arguments";
  os << "recip_tile_init";
  if (op->args.size() == 1) {
    const auto* legacy = op->args[0].as<IntImmNode>();
    ICHECK(legacy != nullptr && (legacy->value == 0 || legacy->value == 1))
        << "tl.blackhole.recip_tile_init legacy_compat must be literal 0 or 1";
    os << "<" << (legacy->value != 0 ? "true" : "false") << ">";
  }
  os << "()";
}

void CodeGenBlackhole::PrintRecipTile(const tvm::tir::CallNode* op,
                                      std::ostream& os) {
  need_compute_api_h_ = true;
  ICHECK(op->args.size() == 1 || op->args.size() == 2 || op->args.size() == 3)
      << "tl.blackhole.recip_tile expects 1, 2, or 3 arguments";
  os << "recip_tile";
  if (op->args.size() == 3) {
    const auto* legacy = op->args[2].as<IntImmNode>();
    ICHECK(legacy != nullptr && (legacy->value == 0 || legacy->value == 1))
        << "tl.blackhole.recip_tile legacy_compat must be literal 0 or 1";
    os << "<" << (legacy->value != 0 ? "true" : "false") << ">";
  }
  os << "(";
  PrintExpr(op->args[0], os);
  if (op->args.size() >= 2) {
    const auto* mode = op->args[1].as<StringImmNode>();
    ICHECK(mode != nullptr)
        << "tl.blackhole.recip_tile vector_mode must be a string literal";
    os << ", (int)VectorMode::" << mode->value;
  }
  os << ")";
}

void CodeGenBlackhole::PrintFillFragment(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var) << "tl.blackhole.fill_fragment expects a direct destination handle var";
  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.fill_fragment", "destination");

  std::ostringstream dtype_os;
  PrintType(dst_dtype, dtype_os);

  os << "MATH({ " << dtype_os.str() << "* dst = reinterpret_cast<" << dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[1], os);
  os << "; const " << dtype_os.str() << " value = static_cast<" << dtype_os.str() << ">(";
  PrintExpr(op->args[2], os);
  os << "); tilelang_fill_fragment(dst, num_elements, value); })";
  MaybeEmitMathWaypoint(os, "FILL");
}

void CodeGenBlackhole::PrintAddFragment(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.add_fragment expects direct source/destination handle vars";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.add_fragment", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.add_fragment", "source");

  std::ostringstream dst_dtype_os;
  std::ostringstream src_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  PrintType(src_dtype, src_dtype_os);

  os << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_add_fragment(dst, src, num_elements); })";
}

void CodeGenBlackhole::PrintAddFragmentFromCBFront(const tvm::tir::CallNode* op,
                                                   std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var) << "tl.blackhole.add_fragment_from_cb_front expects a direct destination handle var";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.add_fragment_from_cb_front", "destination");

  std::ostringstream dst_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  const int cb_id = ResolveCBId(op->args[1]);
  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id
     << "); const " << dst_dtype_os.str() << "* src = reinterpret_cast<const "
     << dst_dtype_os.str() << "*>(cb_front_" << cb_id << ".get_tile_address(0)); "
     << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_add_fragment(dst, src, num_elements); }) }";
  MaybeEmitMathWaypoint(os, "AFCB");
}

void CodeGenBlackhole::PrintPackUntilizeSlice(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var) << "tl.blackhole.pack_untilize_slice expects a direct source handle var";
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.pack_untilize_slice", "source");

  std::ostringstream src_dtype_os;
  PrintType(src_dtype, src_dtype_os);

  const int cb_id = ResolveCBId(op->args[1]);
  const PrimExpr src_offset = op->args.size() >= 5 ? op->args[4] : IntImm(DataType::Int(32), 0);
  const bool raw_16bit_float_copy = src_dtype.is_float16() || src_dtype.is_bfloat16();
  if (raw_16bit_float_copy) {
    os << "{ const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); uint16_t* dst_bits = reinterpret_cast<uint16_t*>(tilelang_cb_write_ptr_bytes_direct("
       << cb_id << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(src_offset, os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[3], os);
    os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset_elements + i] = src_bits[src_offset_elements + i]; } }) }";
    return;
  }

  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << src_dtype_os.str() << "* dst = reinterpret_cast<" << src_dtype_os.str()
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(src_offset, os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[3], os);
  os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
     << "dst[dst_offset_elements + i] = src[src_offset_elements + i]; } }) }";
}

void CodeGenBlackhole::PrintPackUntilizeTile(const tvm::tir::CallNode* op,
                                             std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var)
      << "tl.blackhole.pack_untilize_tile expects a direct source handle var";
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.pack_untilize_tile", "source");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = src_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.pack_untilize_tile requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";

  os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_tile_index = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; MATH({ tilelang_tilize_fragment_tile_nfaces<" << bits_type << ">(src_bits + src_offset_elements, "
     << "dst_bits + dst_tile_index * 1024u); }) }";
}

void CodeGenBlackhole::PrintTilizeLocalFragmentSlice(const tvm::tir::CallNode* op,
                                                     std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var)
      << "tl.blackhole.tilize_local_fragment_slice expects a direct source handle var";
  const DataType src_dtype = ResolveHandleDataType(
      src_var, "tl.blackhole.tilize_local_fragment_slice", "source");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = src_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.tilize_local_fragment_slice requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";
  const PrimExpr src_offset = op->args.size() >= 6 ? op->args[5] : IntImm(DataType::Int(32), 0);
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(src_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic fragment->tiled CB bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic fragment->tiled CB bridge requires inverse logical index "
           "expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic fragment->tiled CB bridge requires at least two inverse "
             "layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr = tir::Substitute(
        binding->inverse_logical_index_exprs[binding->inverse_logical_index_exprs.size() >= 2 ? 1
                                                                                               : 0],
        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic fragment->tiled CB bridge requires trailing inverse "
             "logical indices to be zero for "
          << binding->buffer_name;
    }
    os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
    PrintExpr(op->args[0], os);
    os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
       << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
       << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t row_width = ";
    PrintExpr(op->args[4], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(src_offset, os);
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "const uint32_t tiles_per_row = row_width / kTileCols; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * row_width + logical_col; "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + num_elements) { continue; } "
          "const uint32_t tile_row = logical_row / kTileRows; "
          "const uint32_t tile_col = logical_col / kTileCols; "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
          "const uint32_t tiled_index = tile_index * 1024u + "
          "face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
          "dst_bits[tiled_index] = src_bits[src_offset_elements + __tl_local_i]; } }) }";
    return;
  }

  os << "{ const " << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[4], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(src_offset, os);
  os << "; MATH({ tilelang_tilize_fragment_slice_nfaces<" << bits_type
     << ">(src_bits + src_offset_elements, dst_bits, dst_offset_elements, num_elements, row_width); }) }";
}

void CodeGenBlackhole::PrintTilizeCastFragmentSlice(const tvm::tir::CallNode* op,
                                                    std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var)
      << "tl.blackhole.tilize_cast_fragment_slice expects a direct destination handle var";
  ICHECK(src_var)
      << "tl.blackhole.tilize_cast_fragment_slice expects a direct source handle var";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.tilize_cast_fragment_slice", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.tilize_cast_fragment_slice", "source");
  const int cb_id = ResolveCBId(op->args[2]);

  std::ostringstream src_dtype_os;
  PrintType(src_dtype, src_dtype_os);

  std::string dst_bits_type;
  auto make_convert_expr = [&](const std::string& index_name) {
    const std::string src_expr = "static_cast<float>(src[src_offset_elements + " + index_name + "])";
    if (dst_dtype.is_bfloat16()) {
      return "tilelang_float_to_bfloat_bits(" + src_expr + ")";
    }
    if (dst_dtype.is_float16()) {
      return "tilelang_float_to_half_bits(" + src_expr + ")";
    }
    if (dst_dtype.is_float() && dst_dtype.bits() == 32) {
      return "tilelang_bitcast_float_to_u32(" + src_expr + ")";
    }
    return std::string();
  };
  if (dst_dtype.is_bfloat16()) {
    dst_bits_type = "uint16_t";
  } else if (dst_dtype.is_float16()) {
    dst_bits_type = "uint16_t";
  } else if (dst_dtype.is_float() && dst_dtype.bits() == 32) {
    dst_bits_type = "uint32_t";
  } else {
    ICHECK(false)
        << "tl.blackhole.tilize_cast_fragment_slice currently supports only float16, "
           "bfloat16, or float32 destination dtypes";
  }
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(src_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic cast-fragment->tiled CB bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic cast-fragment->tiled CB bridge requires inverse logical "
           "index expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic cast-fragment->tiled CB bridge requires at least two "
             "inverse layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr = tir::Substitute(
        binding->inverse_logical_index_exprs[binding->inverse_logical_index_exprs.size() >= 2 ? 1
                                                                                               : 0],
        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic cast-fragment->tiled CB bridge requires trailing "
             "inverse logical indices to be zero for "
          << binding->buffer_name;
    }
    os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
       << src_dtype_os.str() << "*>(";
    PrintExpr(op->args[1], os);
    os << "); " << dst_bits_type << "* dst_bits = reinterpret_cast<" << dst_bits_type
       << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
       << ")); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t src_offset_elements = ";
    PrintExpr(op->args[4], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[5], os);
    os << "; const uint32_t row_width = ";
    PrintExpr(op->args[6], os);
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "const uint32_t tiles_per_row = row_width / kTileCols; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * row_width + logical_col; "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + num_elements) { continue; } "
          "const uint32_t tile_row = logical_row / kTileRows; "
          "const uint32_t tile_col = logical_col / kTileCols; "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
          "const uint32_t tiled_index = tile_index * 1024u + "
          "face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
          "dst_bits[tiled_index] = ";
    os << make_convert_expr("__tl_local_i");
    os << "; } }) }";
    return;
  }

  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); " << dst_bits_type << "* dst_bits = reinterpret_cast<" << dst_bits_type
     << "*>(tilelang_cb_write_ptr_bytes_direct(" << cb_id
     << ")); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t src_offset_elements = ";
  PrintExpr(op->args[4], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[5], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[6], os);
  os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
        "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
        "const uint32_t tiles_per_row = row_width / kTileCols; "
        "for (uint32_t i = 0; i < num_elements; ++i) { "
        "const uint32_t logical_index = dst_offset_elements + i; "
        "const uint32_t global_row = logical_index / row_width; "
        "const uint32_t global_col = logical_index % row_width; "
        "const uint32_t tile_row = global_row / kTileRows; "
        "const uint32_t tile_col = global_col / kTileCols; "
        "const uint32_t row_in_tile = global_row % kTileRows; "
        "const uint32_t col_in_tile = global_col % kTileCols; "
        "const uint32_t face_row = row_in_tile / kFaceRows; "
        "const uint32_t face_col = col_in_tile / kFaceCols; "
        "const uint32_t row_in_face = row_in_tile % kFaceRows; "
        "const uint32_t col_in_face = col_in_tile % kFaceCols; "
        "const uint32_t tile_index = tile_row * tiles_per_row + tile_col; "
        "const uint32_t tiled_index = tile_index * 1024u + face_row * (kFaceRows * kTileCols) + "
        "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
        "dst_bits[tiled_index] = ";
  os << make_convert_expr("i");
  os << "; } }) }";
}

void CodeGenBlackhole::PrintPackFillFragmentToTiledCB(const tvm::tir::CallNode* op,
                                                      std::ostream& os) {
  ICHECK_EQ(op->args.size(), 6)
      << "tl.blackhole.pack_fill_fragment_to_tiled_cb expects 6 arguments";
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.pack_fill_fragment_to_tiled_cb expects a direct destination handle var";
  const DataType dst_dtype = ResolveHandleDataType(
      dst_var, "tl.blackhole.pack_fill_fragment_to_tiled_cb", "destination");
  const int cb_id = ResolveCBId(op->args[1]);
  if (!dst_dtype.is_bfloat16() && !(dst_dtype.is_float() && dst_dtype.bits() == 32)) {
    ICHECK(false) << "tl.blackhole.pack_fill_fragment_to_tiled_cb currently admits bf16 or "
                     "float32 publication";
  }
  os << "{ (void)(";
  PrintExpr(op->args[2], os);
  os << "); (void)(";
  PrintExpr(op->args[4], os);
  os << "); const uint32_t num_tiles = (static_cast<uint32_t>(";
  PrintExpr(op->args[3], os);
  os << ") + 1023u) / 1024u; fill_tile_init(); "
        "for (uint32_t tile = 0; tile < num_tiles; ++tile) { "
        "tile_regs_acquire(); fill_tile(0, static_cast<float>(";
  PrintExpr(op->args[5], os);
  os << ")); tile_regs_commit(); tile_regs_wait(); ";
  PrintPackReconfigDataFormatForCB(cb_id, os);
  os << "; pack_tile(0, " << cb_id << ", tile); tile_regs_release(); } }";
}

void CodeGenBlackhole::PrintUntilizeCBFrontTile(const tvm::tir::CallNode* op,
                                                std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.untilize_cb_front_tile expects a direct destination handle var";
  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.untilize_cb_front_tile", "destination");

  std::ostringstream dst_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  const int cb_id = ResolveCBId(op->args[1]);
  const bool raw_16bit_float_copy = dst_dtype.is_float16() || dst_dtype.is_bfloat16();

  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); ";
  if (raw_16bit_float_copy) {
    os << "const uint16_t* src_bits = reinterpret_cast<const uint16_t*>(cb_front_" << cb_id
       << ".get_tile_address(";
    PrintExpr(op->args[2], os);
    os << ")); uint16_t* dst_bits = reinterpret_cast<uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[4], os);
    os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset_elements + i] = src_bits[i]; } }) }";
    return;
  }

  os << "const " << dst_dtype_os.str() << "* src = reinterpret_cast<const "
     << dst_dtype_os.str() << "*>(cb_front_" << cb_id << ".get_tile_address(";
  PrintExpr(op->args[2], os);
  os << ")); " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[4], os);
  os << "; MATH({ for (uint32_t i = 0; i < num_elements; ++i) { "
     << "dst[dst_offset_elements + i] = src[i]; } }) }";
}

void CodeGenBlackhole::PrintUntilizeCBFrontTileFragment(const tvm::tir::CallNode* op,
                                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var)
      << "tl.blackhole.untilize_cb_front_tile_fragment expects a direct destination handle var";
  const DataType dst_dtype = ResolveHandleDataType(
      dst_var, "tl.blackhole.untilize_cb_front_tile_fragment", "destination");

  const int cb_id = ResolveCBId(op->args[1]);
  const int bit_width = dst_dtype.bits();
  ICHECK(bit_width == 16 || bit_width == 32)
      << "tl.blackhole.untilize_cb_front_tile_fragment requires 16-bit or 32-bit element dtype";
  const char* bits_type = bit_width == 16 ? "uint16_t" : "uint32_t";
  if (const LogicalTileLayoutBinding* binding = FindLogicalTileLayoutBinding(dst_var);
      binding != nullptr && LogicalTileLayoutRequiresGenericBridge(*binding)) {
    ICHECK_EQ(binding->local_shape.size(), 1)
        << "Blackhole codegen generic tiled CB->fragment bridge currently requires a 1-D "
           "local_shape for "
        << binding->buffer_name;
    ICHECK(!binding->inverse_logical_index_exprs.empty())
        << "Blackhole codegen generic tiled CB->fragment bridge requires inverse logical index "
           "expressions for "
        << binding->buffer_name;
    tvm::ffi::Optional<Var> thread_index_var;
    PrimExpr thread_index_expr;
    if (thread_idx_x_expr_.empty()) {
      thread_index_expr = IntImm(DataType::Int(32), 0);
    } else {
      thread_index_var = Var(thread_idx_x_expr_, DataType::Int(32));
      thread_index_expr = thread_index_var.value();
    }
    const Var local_index_var("__tl_local_i", DataType::Int(32));
    Map<Var, PrimExpr> subst;
    if (!binding->inverse_logical_index_vars.empty()) {
      ICHECK_GE(binding->inverse_logical_index_vars.size(), 2)
          << "Blackhole codegen generic tiled CB->fragment bridge requires at least two inverse "
             "layout vars for "
          << binding->buffer_name;
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[0]), local_index_var);
      subst.Set(Downcast<Var>(binding->inverse_logical_index_vars[1]), thread_index_expr);
    } else {
      subst.Set(::tvm::tl::InputPlaceholder(0), local_index_var);
      subst.Set(::tvm::tl::InputPlaceholder(1), thread_index_expr);
    }
    const PrimExpr logical_row_expr =
        binding->inverse_logical_index_exprs.size() >= 2
            ? tir::Substitute(binding->inverse_logical_index_exprs[0], subst)
            : IntImm(DataType::Int(32), 0);
    const PrimExpr logical_col_expr = tir::Substitute(
        binding->inverse_logical_index_exprs[binding->inverse_logical_index_exprs.size() >= 2 ? 1
                                                                                               : 0],
        subst);
    for (size_t i = 2; i < binding->inverse_logical_index_exprs.size(); ++i) {
      ICHECK(tir::is_zero(binding->inverse_logical_index_exprs[i]))
          << "Blackhole codegen generic tiled CB->fragment bridge requires trailing inverse "
             "logical indices to be zero for "
          << binding->buffer_name;
    }
    os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); const "
       << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(cb_front_"
       << cb_id << ".get_tile_address(";
    PrintExpr(op->args[2], os);
    os << ")); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type << "*>(";
    PrintExpr(op->args[0], os);
    os << "); const uint32_t dst_offset_elements = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t logical_row_width = ";
    if (binding->logical_shape.size() >= 2) {
      PrintExpr(binding->logical_shape[1], os);
    } else {
      os << "32u";
    }
    os << "; const uint32_t local_extent = ";
    PrintExpr(binding->local_shape[0], os);
    os << "; const uint32_t thread_idx_x = ";
    var_idmap_[local_index_var.get()] = local_index_var->name_hint;
    if (thread_index_var.defined()) {
      var_idmap_[thread_index_var.value().get()] = thread_idx_x_expr_;
    }
    PrintExpr(thread_index_expr, os);
    os << "; MATH({ constexpr uint32_t kTileRows = 32; constexpr uint32_t kTileCols = 32; "
          "constexpr uint32_t kFaceRows = 16; constexpr uint32_t kFaceCols = 16; "
          "for (uint32_t __tl_local_i = 0; __tl_local_i < local_extent; ++__tl_local_i) { "
          "const uint32_t logical_row = ";
    PrintExpr(logical_row_expr, os);
    os << "; const uint32_t logical_col = ";
    PrintExpr(logical_col_expr, os);
    var_idmap_.erase(local_index_var.get());
    if (thread_index_var.defined()) {
      var_idmap_.erase(thread_index_var.value().get());
    }
    os << "; const uint32_t logical_index = logical_row * logical_row_width + logical_col; "
          "if (logical_index < dst_offset_elements || "
          "logical_index >= dst_offset_elements + 1024u) { continue; } "
          "const uint32_t row_in_tile = logical_row % kTileRows; "
          "const uint32_t col_in_tile = logical_col % kTileCols; "
          "const uint32_t face_row = row_in_tile / kFaceRows; "
          "const uint32_t face_col = col_in_tile / kFaceCols; "
          "const uint32_t row_in_face = row_in_tile % kFaceRows; "
          "const uint32_t col_in_face = col_in_tile % kFaceCols; "
          "const uint32_t tiled_index = face_row * (kFaceRows * kTileCols) + "
          "face_col * (kFaceRows * kFaceCols) + row_in_face * kFaceCols + col_in_face; "
          "dst_bits[__tl_local_i] = src_bits[tiled_index]; } }) }";
    return;
  }

  os << "{ experimental::CircularBuffer cb_front_" << cb_id << "(" << cb_id << "); const "
     << bits_type << "* src_bits = reinterpret_cast<const " << bits_type << "*>(cb_front_"
     << cb_id << ".get_tile_address(";
  PrintExpr(op->args[2], os);
  os << ")); " << bits_type << "* dst_bits = reinterpret_cast<" << bits_type << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[3], os);
  os << "; MATH({ tilelang_untilize_fragment_tile_nfaces<" << bits_type
     << ">(src_bits, dst_bits + dst_offset_elements); }) }";
}

void CodeGenBlackhole::PrintCastFragmentSlice(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.cast_fragment_slice expects direct source/destination handle vars";

  const DataType dst_dtype =
      ResolveHandleDataType(dst_var, "tl.blackhole.cast_fragment_slice", "destination");
  const DataType src_dtype =
      ResolveHandleDataType(src_var, "tl.blackhole.cast_fragment_slice", "source");

  std::ostringstream dst_dtype_os;
  std::ostringstream src_dtype_os;
  PrintType(dst_dtype, dst_dtype_os);
  PrintType(src_dtype, src_dtype_os);
  const bool fp32_to_16bit_float_cast =
      (dst_dtype.is_float16() || dst_dtype.is_bfloat16()) && src_dtype.is_float() &&
      src_dtype.bits() == 32;
  if (fp32_to_16bit_float_cast) {
    const char* cast_bits_helper = dst_dtype.is_bfloat16() ? "tilelang_float_to_bfloat_bits"
                                                           : "tilelang_float_to_half_bits";
    os << "MATH({ uint16_t* dst_bits = reinterpret_cast<uint16_t*>(";
    PrintExpr(op->args[0], os);
    os << "); const float* src = reinterpret_cast<const float*>(";
    PrintExpr(op->args[1], os);
    os << "); const uint32_t dst_offset = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t src_offset = ";
    PrintExpr(op->args[3], os);
    os << "; const uint32_t num_elements = ";
    PrintExpr(op->args[4], os);
    os << "; for (uint32_t i = 0; i < num_elements; ++i) { "
       << "dst_bits[dst_offset + i] = " << cast_bits_helper
       << "(src[src_offset + i]); } })";
    MaybeEmitMathWaypoint(os, "CAST");
    return;
  }

  os << "MATH({ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str()
     << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t dst_offset = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t src_offset = ";
  PrintExpr(op->args[3], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[4], os);
  os << "; tilelang_cast_fragment_slice(dst, src, dst_offset, src_offset, num_elements); })";
  MaybeEmitMathWaypoint(os, "CAST");
}

void CodeGenBlackhole::PrintKernelAttributes() {
  // Print kernel-specific attributes for TT-Metal
  // This is a placeholder for future kernel attribute emission
}

void CodeGenBlackhole::PrintCBDeclare(const std::string &name,
                                      tvm::DataType dtype, int num_pages,
                                      int page_size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// CB declaration: " << name << "\n";
  PrintIndent();
  stream << "// TODO: Implement CB allocation\n";
}

void CodeGenBlackhole::PrintCBWaitFront(const std::string &name,
                                        int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_wait_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPopFront(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_pop_front(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBReserveBack(const std::string &name,
                                          int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_reserve_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintCBPushBack(const std::string &name, int num_tiles) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "cb_push_back(" << name << ", " << num_tiles << ");\n";
}

void CodeGenBlackhole::PrintNOCRead(const std::string &src_addr,
                                    const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC read: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWrite(const std::string &src_addr,
                                     const std::string &dst_addr, int size) {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "// NOC write: " << src_addr << " -> " << dst_addr << " (" << size
         << " bytes)\n";
}

void CodeGenBlackhole::PrintNOCWait() {
  need_dataflow_api_h_ = true;
  PrintIndent();
  stream << "noc_async_read_barrier();\n";
}

void CodeGenBlackhole::PrintGetSemaphore(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "get_semaphore(";
  PrintExpr(op->args[0], os);
  os << ")";
}

void CodeGenBlackhole::PrintRuntimeArgU32(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  const auto* arg_name = op->args[0].as<tvm::tir::StringImmNode>();
  ICHECK(arg_name) << "tl.blackhole.runtime_arg_u32 expects a string literal name";
  auto it = runtime_arg_vars_by_name_.find(arg_name->value);
  ICHECK(it != runtime_arg_vars_by_name_.end())
      << "Missing runtime arg binding for name: " << arg_name->value;
  os << it->second;
}

void CodeGenBlackhole::PrintSemaphoreWait(const tvm::tir::CallNode *op,
                                          std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreSet(const tvm::tir::CallNode *op,
                                         std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreIncRemote(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_inc(get_noc_addr(";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[0], os);
  os << "), ";
  PrintExpr(op->args[3], os);
  os << ")";
}

void CodeGenBlackhole::PrintSemaphoreSetRemote(const tvm::tir::CallNode *op,
                                               std::ostream &os) {
  need_dataflow_api_h_ = true;
  os << "noc_semaphore_set_remote(";
  PrintExpr(op->args[0], os);
  os << ", get_noc_addr(";
  PrintExpr(op->args[1], os);
  os << ", ";
  PrintExpr(op->args[2], os);
  os << ", ";
  PrintExpr(op->args[3], os);
  os << "))";
}

}  // namespace tl
}  // namespace tvm
