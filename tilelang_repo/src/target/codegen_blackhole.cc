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
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../tir/builtin_blackhole.h"
#include "tt_program_projection.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

namespace tvm {
namespace tl {

namespace {

const tvm::tir::VarNode* AsHandleVar(const tvm::PrimExpr& expr) {
  if (const auto* var = expr.as<tvm::tir::VarNode>()) {
    return var;
  }
  return nullptr;
}

void ValidateNoUnsupportedFragmentRequirementsForCodegen(const tvm::tir::PrimFunc& f) {
  auto lowering_requirements =
      f->GetAttr<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>("blackhole.lowering_requirements");
  if (!lowering_requirements) {
    return;
  }

  std::vector<std::string> unsupported_ops;
  std::unordered_set<std::string> seen_ops;
  if (auto fragment_ops = lowering_requirements.value().Get("fragment_op_kinds")) {
    for (const auto& item : Downcast<tvm::ffi::Array<tvm::ffi::Any>>(fragment_ops.value())) {
      const std::string op_name = Downcast<tvm::ffi::String>(item);
      if ((op_name == "row_reduction" || op_name == "row_broadcast") &&
          seen_ops.insert(op_name).second) {
        unsupported_ops.push_back(op_name);
      }
    }
  }
  if (auto pointwise_ops = lowering_requirements.value().Get("pointwise_op_kinds")) {
    for (const auto& item : Downcast<tvm::ffi::Array<tvm::ffi::Any>>(pointwise_ops.value())) {
      const std::string op_name = Downcast<tvm::ffi::String>(item);
      if ((op_name == "fill" || op_name == "max" || op_name == "add" || op_name == "cast") &&
          seen_ops.insert(op_name).second) {
        unsupported_ops.push_back(op_name);
      }
    }
  }
  if (!unsupported_ops.empty()) {
    std::ostringstream os;
    for (size_t i = 0; i < unsupported_ops.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << unsupported_ops[i];
    }
    ICHECK(false) << "Blackhole fragment compute subset lowering is not implemented for ops ["
                  << os.str() << "]";
  }
}

ffi::Array<ffi::Any> AggregateSegmentRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  ffi::Array<ffi::Any> aggregated;
  auto segment_plan =
      tt_program_projection::GetSegmentPlanFromTTProgram(f, "Blackhole codegen");
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
  return tt_program_projection::GetRuntimeArgsFromTTProgram(f, "Blackhole codegen");
}

ffi::Array<ffi::Any> GetCBConfigsForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCBConfigsFromTTProgram(f, "Blackhole codegen");
}

ffi::Map<ffi::String, ffi::Any> GetCorePlanForCodegen(const tvm::tir::PrimFunc& f) {
  return tt_program_projection::GetCorePlanFromTTProgram(f, "Blackhole codegen");
}

std::string GetCoreTypeForCodegen(const tvm::tir::PrimFunc& f) {
  auto program = tt_program_projection::RequireTTProgram(f, "Blackhole codegen");
  if (!program->kernels.empty()) {
    return program->kernels[0]->core_type;
  }
  return "";
}

bool HasRuntimeArgsForCodegen(const tvm::tir::PrimFunc& f) {
  if (!GetRuntimeArgsForCodegen(f).empty()) {
    return true;
  }
  return !AggregateSegmentRuntimeArgsForCodegen(f).empty();
}

}  // namespace

CodeGenBlackhole::CodeGenBlackhole()
    : headers_emitted_(false),
      core_type_(CoreType::kBRISC),  // Default to BRISC for TT-Sim compatibility
      need_dataflow_api_h_(false),
      need_compute_api_h_(false) {}

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
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_kind_.clear();
  runtime_arg_vars_by_name_.clear();
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_id_by_requirement_name_.clear();
  cb_num_pages_by_requirement_name_.clear();
  logical_grid_x_ = 1;
  logical_grid_y_ = 1;
  linearization_ = "row_major";
}

std::string CodeGenBlackhole::GetKernelCode() const {
  // Return the kernel code with TT-Metal headers but without TVM-specific headers
  // decl_stream now contains TT-Metal headers (dataflow_api.h, etc.)
  // stream contains the actual kernel implementation
  std::ostringstream kernel_code;
  kernel_code << decl_stream.str();
  kernel_code << stream.str();
  return kernel_code.str();
}

void CodeGenBlackhole::AddFunction(const tvm::GlobalVar &gvar,
                                   const tvm::tir::PrimFunc &f) {
  ValidateNoUnsupportedFragmentRequirementsForCodegen(f);
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

    // Detect core type from function attributes (IR-driven, not function name)
    std::string core_type_str = GetCoreTypeForCodegen(f);
    if (core_type_str == "brisc") {
      core_type_ = CoreType::kBRISC;
    } else if (core_type_str == "ncrisc") {
      core_type_ = CoreType::kNCRISC;
    } else if (core_type_str == "trisc") {
      core_type_ = CoreType::kTRISC;
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
        decl_stream << "#include \"api/compute/tile_move_copy.h\"\n";
        decl_stream << "#include \"api/compute/matmul.h\"\n";
        decl_stream << "#include \"hostdevcommon/kernel_structs.h\"\n";
        decl_stream << "using half = _Float16;\n";
        decl_stream << "static constexpr float inff = std::numeric_limits<float>::infinity();\n";
        decl_stream << "ALWI float tilelang_fast_exp2f(float x) {\n";
        decl_stream << "  if (x <= -126.0f) { return 0.0f; }\n";
        decl_stream << "  if (x >= 126.0f) { x = 126.0f; }\n";
        decl_stream << "  int ipart = static_cast<int>(x);\n";
        decl_stream << "  if (static_cast<float>(ipart) > x) { --ipart; }\n";
        decl_stream << "  const float fpart = x - static_cast<float>(ipart);\n";
        decl_stream << "  const float poly = 1.0f + fpart * (0.69314718f + fpart * (0.24022651f + fpart * (0.05550411f + fpart * 0.00961813f)));\n";
        decl_stream << "  union { uint32_t i; float f; } exponent_bits{static_cast<uint32_t>(ipart + 127) << 23};\n";
        decl_stream << "  return exponent_bits.f * poly;\n";
        decl_stream << "}\n";
        decl_stream << "template <typename T>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_fill_fragment(T* dst, uint32_t num_elements, T value) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = value; }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_cast_fragment_slice(DstT* dst, const SrcT* src, uint32_t dst_offset, uint32_t src_offset, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[dst_offset + i] = static_cast<DstT>(src[src_offset + i]); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename SrcT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_scalar_max(DstT* dst, const SrcT* src, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = (static_cast<DstT>(src[i]) > dst[i]) ? static_cast<DstT>(src[i]) : dst[i]; }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename SrcT, typename DstT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_reduce_row_sum(const SrcT* src, DstT* dst, uint32_t num_elements, bool clear) {\n";
        decl_stream << "  if (clear) { dst[0] = static_cast<DstT>(0); }\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[0] = static_cast<DstT>(dst[0] + static_cast<DstT>(src[i])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename SrcT, typename DstT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_reduce_row_max(const SrcT* src, DstT* dst, uint32_t num_elements, bool clear) {\n";
        decl_stream << "  if (num_elements == 0) { return; }\n";
        decl_stream << "  DstT value = clear ? static_cast<DstT>(src[0]) : dst[0];\n";
        decl_stream << "  for (uint32_t i = clear ? 1u : 0u; i < num_elements; ++i) { const DstT src_value = static_cast<DstT>(src[i]); value = (src_value > value) ? src_value : value; }\n";
        decl_stream << "  dst[0] = value;\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename ScalarT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_mul_row_bcast(DstT* dst, const ScalarT* scalar, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = static_cast<DstT>(dst[i] * static_cast<DstT>(scalar[0])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename ScalarT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_div_row_bcast(DstT* dst, const ScalarT* scalar, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = static_cast<DstT>(dst[i] / static_cast<DstT>(scalar[0])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename LhsT, typename RhsT, typename AddT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_scalar_fma(DstT* dst, const LhsT* lhs, const RhsT* rhs, const AddT* add, uint32_t num_elements) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = static_cast<DstT>(static_cast<float>(lhs[i]) * static_cast<float>(rhs[i]) + static_cast<float>(add[i])); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename ScalarT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_exp2_row_bcast_affine(DstT* dst, const ScalarT* scalar, uint32_t num_elements, float dst_scale, float scalar_scale) {\n";
        decl_stream << "  for (uint32_t i = 0; i < num_elements; ++i) { const float expr = static_cast<float>(dst[i]) * dst_scale + static_cast<float>(scalar[0]) * scalar_scale; dst[i] = static_cast<DstT>(tilelang_fast_exp2f(expr)); }\n";
        decl_stream << "}\n";
        decl_stream << "template <typename DstT, typename LhsT, typename RhsT>\n";
        decl_stream << "__attribute__((noinline, noclone)) void tilelang_scalar_exp2_affine(DstT* dst, const LhsT* lhs, const RhsT* rhs, float lhs_scale, float rhs_scale) {\n";
        decl_stream << "  dst[0] = static_cast<DstT>(tilelang_fast_exp2f(static_cast<float>(lhs[0]) * lhs_scale + static_cast<float>(rhs[0]) * rhs_scale));\n";
        decl_stream << "}\n";
        decl_stream << "ALWI uint32_t tilelang_get_cb_write_ptr_bytes(uint32_t cb_id) {\n";
        decl_stream << "  uint32_t address = 0;\n";
        decl_stream << "  PACK({ address = get_local_cb_interface(cb_id).fifo_wr_ptr << 4; "
                       "mailbox_write(ckernel::ThreadId::MathThreadId, address); "
                       "mailbox_write(ckernel::ThreadId::UnpackThreadId, address); })\n";
        decl_stream << "  MATH(address = mailbox_read(ckernel::ThreadId::PackThreadId);)\n";
        decl_stream << "  UNPACK(address = mailbox_read(ckernel::ThreadId::PackThreadId);)\n";
        decl_stream << "  return address;\n";
        decl_stream << "}\n";
        break;
      default:
        decl_stream << "// DataMovement kernel API (default)\n";
        decl_stream << "#include \"api/dataflow/dataflow_api.h\"\n";
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
      // Float scalar argument - passed as bits in uint32_t
      stream << "  // Argument " << arg_idx << ": " << param_name << " ("
             << dtype << ")\n";
      stream << "  uint32_t " << param_name << "_bits = get_arg_val<uint32_t>("
             << arg_idx++ << ");\n";
      stream << "  " << dtype << " " << param_name
             << " = *reinterpret_cast<" << dtype << "*>(&" << param_name
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

void CodeGenBlackhole::EmitRuntimeArgLoads(const tvm::tir::PrimFunc &f) {
  buffer_runtime_arg_map_.clear();
  buffer_runtime_arg_map_by_name_.clear();
  runtime_arg_vars_by_kind_.clear();
  runtime_arg_vars_by_name_.clear();
  cb_page_size_by_id_.clear();
  cb_num_pages_by_id_.clear();
  cb_id_by_requirement_name_.clear();
  cb_num_pages_by_requirement_name_.clear();

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
      if (auto v = cb_info.Get("cb_id")) {
        cb_id = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("page_size")) {
        page_size = Downcast<tvm::Integer>(v.value()).IntValue();
      }
      if (auto v = cb_info.Get("num_pages")) {
        num_pages = Downcast<tvm::Integer>(v.value()).IntValue();
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
          }
        }
      }
    }
  }

  ffi::Array<ffi::Any> runtime_args = GetRuntimeArgsForCodegen(f);
  if (runtime_args.empty()) {
    runtime_args = AggregateSegmentRuntimeArgsForCodegen(f);
  }
  ICHECK(!runtime_args.empty())
      << "Blackhole codegen requires TTProgram ABI runtime args";

  std::unordered_map<std::string, const tvm::tir::VarNode *> buffer_vars_by_name;
  std::vector<std::string> ordered_handle_buffer_names;
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      buffer_vars_by_name[param->name_hint] = param.get();
      ordered_handle_buffer_names.push_back(param->name_hint);
    }
  }
  for (const auto &kv : f->buffer_map) {
    const auto &buffer = kv.second;
    buffer_vars_by_name[buffer->name] = buffer->data.get();
    if (std::find(ordered_handle_buffer_names.begin(), ordered_handle_buffer_names.end(),
                  buffer->name) == ordered_handle_buffer_names.end()) {
      ordered_handle_buffer_names.push_back(buffer->name);
    }
  }
  // Packed Blackhole entrypoints can arrive after MakePackedAPI, where the
  // public function params are no longer the original A/B handles and
  // buffer_map may be empty.  Recover the runtime-backed buffer vars from the
  // actual TIR body so builtins like read_tile_to_cb(A, ...) still bind.
  tir::PostOrderVisit(f->body, [&](const tvm::runtime::ObjectRef &node) {
    if (const auto *store = node.as<tvm::tir::BufferStoreNode>()) {
      buffer_vars_by_name[store->buffer->name] = store->buffer->data.get();
      if (const auto *load = store->value.as<tvm::tir::BufferLoadNode>()) {
        buffer_vars_by_name[load->buffer->name] = load->buffer->data.get();
      }
      return;
    }
    if (const auto *load = node.as<tvm::tir::BufferLoadNode>()) {
      buffer_vars_by_name[load->buffer->name] = load->buffer->data.get();
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
      }
      return;
    }
    if (op_name == "tl.blackhole.read_page_to_cb") {
      if (const auto *buffer_var = call->args[0].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
      }
      return;
    }
    if (op_name == "tl.blackhole.write_tile_from_cb") {
      if (const auto *buffer_var = call->args[1].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
      }
      return;
    }
    if (op_name == "tl.blackhole.write_page_from_cb") {
      if (const auto *buffer_var = call->args[1].as<tvm::tir::VarNode>()) {
        buffer_vars_by_name[buffer_var->name_hint] = buffer_var;
      }
    }
  });

  int arg_idx = 0;
  size_t next_input_buffer = 0;
  size_t next_output_buffer = ordered_handle_buffer_names.empty()
                                  ? 0
                                  : ordered_handle_buffer_names.size() - 1;
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

    std::optional<std::string> bound_buffer_name;
    if (auto v = arg_info.Get("buffer")) {
      bound_buffer_name = std::string(Downcast<tvm::ffi::String>(v.value()));
    } else if ((arg_kind == "input_buffer_addr32" || arg_kind == "input_buffer_addr") &&
               next_input_buffer < ordered_handle_buffer_names.size()) {
      bound_buffer_name = ordered_handle_buffer_names[next_input_buffer++];
    } else if ((arg_kind == "output_buffer_addr32" || arg_kind == "output_buffer_addr") &&
               !ordered_handle_buffer_names.empty() &&
               next_output_buffer < ordered_handle_buffer_names.size()) {
      bound_buffer_name = ordered_handle_buffer_names[next_output_buffer];
      if (next_output_buffer > 0) {
        --next_output_buffer;
      }
    }

    if (bound_buffer_name.has_value()) {
      std::vector<std::string> candidate_names{bound_buffer_name.value()};
      candidate_names.push_back(bound_buffer_name.value() + "_handle");
      for (const auto& candidate_name : candidate_names) {
        auto it = buffer_vars_by_name.find(candidate_name);
        if (it != buffer_vars_by_name.end()) {
          buffer_runtime_arg_map_[it->second] = arg_name;
          buffer_runtime_arg_map_by_name_[candidate_name] = arg_name;
        }
      }
      buffer_runtime_arg_map_by_name_[bound_buffer_name.value()] = arg_name;
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

  auto lookup_kind = [&](const char* kind) -> std::optional<std::string> {
    if (!kind) {
      return std::nullopt;
    }
    auto it = runtime_arg_vars_by_kind_.find(kind);
    if (it == runtime_arg_vars_by_kind_.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  if (auto preferred = lookup_kind(preferred_kind)) {
    return *preferred;
  }
  if (preferred_kind) {
    const std::string preferred32 = std::string(preferred_kind) + "32";
    if (auto preferred = lookup_kind(preferred32.c_str())) {
      return *preferred;
    }
  }

  auto out32 = runtime_arg_vars_by_kind_.find("output_buffer_addr32");
  auto out64 = runtime_arg_vars_by_kind_.find("output_buffer_addr");
  const bool has_input_addr32 = runtime_arg_vars_by_kind_.count("input_buffer_addr32");
  const bool has_input_addr64 = runtime_arg_vars_by_kind_.count("input_buffer_addr");
  if (!has_input_addr32 && !has_input_addr64) {
    if (out32 != runtime_arg_vars_by_kind_.end()) {
      return out32->second;
    }
    if (out64 != runtime_arg_vars_by_kind_.end()) {
      return out64->second;
    }
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

  if (runtime_managed_storage || (scope == "blackhole.acc" && !compute_local_fragment_storage)) {
    // Blackhole shared / CB allocations are runtime/device-managed
    // resources, not C arrays inside the generated kernel body.  The
    // blackhole.acc scope is compute-local and only materializes inside TRISC
    // kernels as L1-backed buffers addressed through synchronized CB pointers.
    this->PrintStmt(op->body);
    return;
  }

  ICHECK(!tvm::tir::is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (compute_local_fragment_storage) {
    auto cb_it = cb_id_by_requirement_name_.find(op->buffer_var->name_hint);
    ICHECK(cb_it != cb_id_by_requirement_name_.end())
        << "Missing CB binding for blackhole.acc buffer " << op->buffer_var->name_hint;
    const int cb_id = cb_it->second;
    const int num_pages = cb_num_pages_by_requirement_name_.count(op->buffer_var->name_hint)
                              ? cb_num_pages_by_requirement_name_.at(op->buffer_var->name_hint)
                              : GetCBNumPages(cb_id);

    std::ostringstream dtype_os;
    PrintType(op->dtype, dtype_os);

    PrintIndent();
    stream << "cb_reserve_back(" << cb_id << ", " << num_pages << ");\n";
    PrintIndent();
    stream << dtype_os.str() << "* " << vid << " = reinterpret_cast<" << dtype_os.str()
           << "*>(tilelang_get_cb_write_ptr_bytes(" << cb_id << "));\n";

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
  std::optional<std::string> work_id_var;
  auto work_id_it = runtime_arg_vars_by_kind_.find("work_linear_id");
  if (work_id_it != runtime_arg_vars_by_kind_.end()) {
    work_id_var = work_id_it->second;
  } else if ((work_id_it = runtime_arg_vars_by_kind_.find("current_work_linear_id")) !=
             runtime_arg_vars_by_kind_.end()) {
    work_id_var = work_id_it->second;
  } else {
    const bool requests_copy_work_descriptor =
        runtime_arg_vars_by_kind_.count("a_tile_start_id") ||
        runtime_arg_vars_by_kind_.count("output_tile_start_id");
    if (requests_copy_work_descriptor) {
      ICHECK(false)
          << "Blackhole blockIdx reconstruction requires explicit work_linear_id for the richer "
             "copy work schema; copy fallback without work_linear_id is unsupported";
    }
  }
  const bool has_runtime_work_id =
      work_id_var.has_value() && linearization_ == "row_major" && logical_grid_x_ > 0;

  // Map CUDA-style thread indices to Blackhole concepts
  // For staged single-core execution, logical block indices are reconstructed
  // from the execution-plan runtime arg instead of being constantized.
  if (thread_tag == "blockIdx.x") {
    if (has_runtime_work_id) {
      var_idmap_[iv->var.get()] =
          "(" + work_id_var.value() + " % " + std::to_string(logical_grid_x_) + ")";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_x */";
    }
  } else if (thread_tag == "blockIdx.y") {
    if (has_runtime_work_id) {
      var_idmap_[iv->var.get()] =
          "(" + work_id_var.value() + " / " + std::to_string(logical_grid_x_) + ")";
    } else {
      var_idmap_[iv->var.get()] = "0 /* core_y */";
    }
  } else if (thread_tag == "blockIdx.z") {
    var_idmap_[iv->var.get()] = "0 /* core_z */";
  } else if (thread_tag == "threadIdx.x") {
    // For Blackhole, threadIdx.x could map to worker threads within a core
    // For now, use the variable name directly
    var_idmap_[iv->var.get()] = iv->var->name_hint;
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
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
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
  } else if (builtin_name == "fill_fragment") {
    PrintFillFragment(op, os);
    return true;
  } else if (builtin_name == "write_local_slice_to_cb") {
    PrintWriteLocalSliceToCB(op, os);
    return true;
  } else if (builtin_name == "scalar_max") {
    PrintScalarMax(op, os);
    return true;
  } else if (builtin_name == "cast_fragment_slice") {
    PrintCastFragmentSlice(op, os);
    return true;
  } else if (builtin_name == "reduce_row") {
    PrintReduceRow(op, os);
    return true;
  } else if (builtin_name == "mul_row_bcast") {
    PrintMulRowBcast(op, os);
    return true;
  } else if (builtin_name == "mul_grouped_row_bcast") {
    PrintMulGroupedRowBcast(op, os);
    return true;
  } else if (builtin_name == "div_row_bcast") {
    PrintDivRowBcast(op, os);
    return true;
  } else if (builtin_name == "div_grouped_row_bcast") {
    PrintDivGroupedRowBcast(op, os);
    return true;
  } else if (builtin_name == "scalar_fma") {
    PrintScalarFma(op, os);
    return true;
  } else if (builtin_name == "exp2_row_bcast_affine") {
    PrintExp2RowBcastAffine(op, os);
    return true;
  } else if (builtin_name == "exp2_grouped_row_bcast_affine") {
    PrintExp2GroupedRowBcastAffine(op, os);
    return true;
  } else if (builtin_name == "scalar_exp2_affine") {
    PrintScalarExp2Affine(op, os);
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
  os << "cb_push_back(";
  PrintResolvedCBId(op->args[0], os);
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
  os << "cb_pop_front(";
  PrintResolvedCBId(op->args[0], os);
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
  os << "pack_tile(";
  PrintExpr(op->args[0], os);  // src_tile_index
  os << ", ";
  PrintResolvedCBId(op->args[1], os);
  os << ")";
}

void CodeGenBlackhole::PrintFillFragment(const tvm::tir::CallNode* op,
                                         std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  ICHECK(dst_var) << "tl.blackhole.fill_fragment expects a direct destination handle var";
  auto dtype_it = handle_data_type_.find(dst_var);
  ICHECK(dtype_it != handle_data_type_.end())
      << "Missing handle dtype for tl.blackhole.fill_fragment destination";

  std::ostringstream dtype_os;
  PrintType(dtype_it->second, dtype_os);

  os << "{ " << dtype_os.str() << "* dst = reinterpret_cast<" << dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[1], os);
  os << "; const " << dtype_os.str() << " value = static_cast<" << dtype_os.str() << ">(";
  PrintExpr(op->args[2], os);
  os << "); tilelang_fill_fragment(dst, num_elements, value); }";
}

void CodeGenBlackhole::PrintWriteLocalSliceToCB(const tvm::tir::CallNode* op,
                                                std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  ICHECK(src_var) << "tl.blackhole.write_local_slice_to_cb expects a direct source handle var";
  auto src_dtype_it = handle_data_type_.find(src_var);
  ICHECK(src_dtype_it != handle_data_type_.end())
      << "Missing source handle dtype for tl.blackhole.write_local_slice_to_cb";

  std::ostringstream src_dtype_os;
  PrintType(src_dtype_it->second, src_dtype_os);

  const int cb_id = ResolveCBId(op->args[1]);
  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const uint32_t dst_offset_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t num_elements = ";
  PrintExpr(op->args[3], os);
  os << "; PACK({ " << src_dtype_os.str() << "* dst = reinterpret_cast<" << src_dtype_os.str()
     << "*>((get_local_cb_interface(" << cb_id << ").fifo_wr_ptr << 4) + dst_offset_elements * sizeof("
     << src_dtype_os.str() << ")); "
     << "for (uint32_t i = 0; i < num_elements; ++i) { dst[i] = src[i]; } }); }";
}

void CodeGenBlackhole::PrintScalarMax(const tvm::tir::CallNode* op,
                                      std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.scalar_max expects direct handle vars";

  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto src_dtype_it = handle_data_type_.find(src_var);
  ICHECK(dst_dtype_it != handle_data_type_.end())
      << "Missing destination handle dtype for tl.blackhole.scalar_max";
  ICHECK(src_dtype_it != handle_data_type_.end())
      << "Missing source handle dtype for tl.blackhole.scalar_max";
  ICHECK(dst_dtype_it->second == src_dtype_it->second)
      << "tl.blackhole.scalar_max expects matching source/destination dtypes";

  std::ostringstream dtype_os;
  PrintType(dst_dtype_it->second, dtype_os);

  os << "{ " << dtype_os.str() << "* dst = reinterpret_cast<" << dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << dtype_os.str() << "* src = reinterpret_cast<const " << dtype_os.str()
     << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  if (op->args.size() >= 3) {
    PrintExpr(op->args[2], os);
  } else {
    os << "1";
  }
  os << "; tilelang_scalar_max(dst, src, num_elements); }";
}

void CodeGenBlackhole::PrintCastFragmentSlice(const tvm::tir::CallNode* op,
                                              std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* src_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && src_var)
      << "tl.blackhole.cast_fragment_slice expects direct source/destination handle vars";

  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto src_dtype_it = handle_data_type_.find(src_var);
  ICHECK(dst_dtype_it != handle_data_type_.end())
      << "Missing destination handle dtype for tl.blackhole.cast_fragment_slice";
  ICHECK(src_dtype_it != handle_data_type_.end())
      << "Missing source handle dtype for tl.blackhole.cast_fragment_slice";

  std::ostringstream dst_dtype_os;
  std::ostringstream src_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(src_dtype_it->second, src_dtype_os);

  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
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
  os << "; tilelang_cast_fragment_slice(dst, src, dst_offset, src_offset, num_elements); }";
}

void CodeGenBlackhole::PrintReduceRow(const tvm::tir::CallNode* op,
                                      std::ostream& os) {
  const auto* src_var = AsHandleVar(op->args[0]);
  const auto* dst_var = AsHandleVar(op->args[1]);
  const bool grouped = op->args.size() >= 6;
  const auto* reduce_kind = op->args[grouped ? 4 : 3].as<tvm::tir::StringImmNode>();
  const auto* clear = op->args[grouped ? 5 : 4].as<tvm::tir::IntImmNode>();
  ICHECK(src_var && dst_var && reduce_kind && clear)
      << "tl.blackhole.reduce_row expects direct handle vars and constant reduce metadata";

  auto src_dtype_it = handle_data_type_.find(src_var);
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  ICHECK(src_dtype_it != handle_data_type_.end() && dst_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.reduce_row";

  std::ostringstream src_dtype_os;
  std::ostringstream dst_dtype_os;
  PrintType(src_dtype_it->second, src_dtype_os);
  PrintType(dst_dtype_it->second, dst_dtype_os);

  os << "{ const " << src_dtype_os.str() << "* src = reinterpret_cast<const "
     << src_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); ";
  if (grouped) {
    os << "const uint32_t num_rows = ";
    PrintExpr(op->args[2], os);
    os << "; const uint32_t row_width = ";
    PrintExpr(op->args[3], os);
    os << "; for (uint32_t row = 0; row < num_rows; ++row) { ";
    if (reduce_kind->value == "sum") {
      os << dst_dtype_os.str() << " value = "
         << (clear->value ? "static_cast<" + dst_dtype_os.str() + ">(0)"
                          : "dst[row]")
         << "; for (uint32_t i = 0; i < row_width; ++i) { const uint32_t idx = row * row_width + i; "
         << "value = static_cast<" << dst_dtype_os.str() << ">(value + static_cast<"
         << dst_dtype_os.str() << ">(src[idx])); } dst[row] = value; }";
    } else if (reduce_kind->value == "max") {
      os << "if (row_width > 0) { " << dst_dtype_os.str() << " value = "
         << (clear->value ? "static_cast<" + dst_dtype_os.str() + ">(src[row * row_width])"
                          : "dst[row]")
         << "; for (uint32_t i = " << (clear->value ? "1" : "0")
         << "; i < row_width; ++i) { const uint32_t idx = row * row_width + i; const "
         << dst_dtype_os.str() << " src_value = static_cast<" << dst_dtype_os.str()
         << ">(src[idx]); value = (src_value > value) ? src_value : value; } dst[row] = value; } }";
    } else {
      ICHECK(false) << "Unsupported tl.blackhole.reduce_row kind: " << reduce_kind->value;
    }
  } else {
    os << "const uint32_t num_elements = ";
    PrintExpr(op->args[2], os);
    os << "; const bool clear = " << (clear->value ? "true" : "false") << "; ";
    if (reduce_kind->value == "sum") {
      os << "tilelang_reduce_row_sum(src, dst, num_elements, clear);";
    } else if (reduce_kind->value == "max") {
      os << "tilelang_reduce_row_max(src, dst, num_elements, clear);";
    } else {
      ICHECK(false) << "Unsupported tl.blackhole.reduce_row kind: " << reduce_kind->value;
    }
  }
  os << " }";
}

void CodeGenBlackhole::PrintMulRowBcast(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var) << "tl.blackhole.mul_row_bcast expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.mul_row_bcast";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);
  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_mul_row_bcast(dst, scalar, num_elements); }";
}

void CodeGenBlackhole::PrintMulGroupedRowBcast(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var)
      << "tl.blackhole.mul_grouped_row_bcast expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.mul_grouped_row_bcast";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);
  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[3], os);
  os << "; for (uint32_t i = 0; i < num_elements; ++i) { const uint32_t row = i / row_width; "
     << "dst[i] = static_cast<" << dst_dtype_os.str() << ">(dst[i] * static_cast<"
     << dst_dtype_os.str() << ">(scalar[row])); } }";
}

void CodeGenBlackhole::PrintDivRowBcast(const tvm::tir::CallNode* op,
                                        std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var) << "tl.blackhole.div_row_bcast expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.div_row_bcast";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);
  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; tilelang_div_row_bcast(dst, scalar, num_elements); }";
}

void CodeGenBlackhole::PrintDivGroupedRowBcast(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var)
      << "tl.blackhole.div_grouped_row_bcast expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.div_grouped_row_bcast";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);
  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[3], os);
  os << "; for (uint32_t i = 0; i < num_elements; ++i) { const uint32_t row = i / row_width; "
     << "dst[i] = static_cast<" << dst_dtype_os.str() << ">(dst[i] / static_cast<"
     << dst_dtype_os.str() << ">(scalar[row])); } }";
}

void CodeGenBlackhole::PrintScalarFma(const tvm::tir::CallNode* op,
                                      std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* lhs_var = AsHandleVar(op->args[1]);
  const auto* rhs_var = AsHandleVar(op->args[2]);
  const auto* add_var = AsHandleVar(op->args[3]);
  ICHECK(dst_var && lhs_var && rhs_var && add_var)
      << "tl.blackhole.scalar_fma expects direct handle vars";

  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto lhs_dtype_it = handle_data_type_.find(lhs_var);
  auto rhs_dtype_it = handle_data_type_.find(rhs_var);
  auto add_dtype_it = handle_data_type_.find(add_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && lhs_dtype_it != handle_data_type_.end() &&
         rhs_dtype_it != handle_data_type_.end() && add_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.scalar_fma";

  std::ostringstream dst_dtype_os;
  std::ostringstream lhs_dtype_os;
  std::ostringstream rhs_dtype_os;
  std::ostringstream add_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(lhs_dtype_it->second, lhs_dtype_os);
  PrintType(rhs_dtype_it->second, rhs_dtype_os);
  PrintType(add_dtype_it->second, add_dtype_os);

  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << lhs_dtype_os.str() << "* lhs = reinterpret_cast<const "
     << lhs_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const " << rhs_dtype_os.str() << "* rhs = reinterpret_cast<const "
     << rhs_dtype_os.str() << "*>(";
  PrintExpr(op->args[2], os);
  os << "); const " << add_dtype_os.str() << "* add = reinterpret_cast<const "
     << add_dtype_os.str() << "*>(";
  PrintExpr(op->args[3], os);
  os << "); const uint32_t num_elements = ";
  if (op->args.size() >= 5) {
    PrintExpr(op->args[4], os);
  } else {
    os << "1";
  }
  os << "; tilelang_scalar_fma(dst, lhs, rhs, add, num_elements); }";
}

void CodeGenBlackhole::PrintExp2RowBcastAffine(const tvm::tir::CallNode* op,
                                               std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var)
      << "tl.blackhole.exp2_row_bcast_affine expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.exp2_row_bcast_affine";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);

  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const float dst_scale = ";
  PrintExpr(op->args[3], os);
  os << "; const float scalar_scale = ";
  PrintExpr(op->args[4], os);
  os << "; tilelang_exp2_row_bcast_affine(dst, scalar, num_elements, dst_scale, scalar_scale); }";
}

void CodeGenBlackhole::PrintExp2GroupedRowBcastAffine(const tvm::tir::CallNode* op,
                                                      std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* scalar_var = AsHandleVar(op->args[1]);
  ICHECK(dst_var && scalar_var)
      << "tl.blackhole.exp2_grouped_row_bcast_affine expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto scalar_dtype_it = handle_data_type_.find(scalar_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && scalar_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.exp2_grouped_row_bcast_affine";

  std::ostringstream dst_dtype_os;
  std::ostringstream scalar_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(scalar_dtype_it->second, scalar_dtype_os);

  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << scalar_dtype_os.str() << "* scalar = reinterpret_cast<const "
     << scalar_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const uint32_t num_elements = ";
  PrintExpr(op->args[2], os);
  os << "; const uint32_t row_width = ";
  PrintExpr(op->args[3], os);
  os << "; const float dst_scale = ";
  PrintExpr(op->args[4], os);
  os << "; const float scalar_scale = ";
  PrintExpr(op->args[5], os);
  os << "; for (uint32_t i = 0; i < num_elements; ++i) { const uint32_t row = i / row_width; "
     << "const float expr = static_cast<float>(dst[i]) * dst_scale + "
     << "static_cast<float>(scalar[row]) * scalar_scale; "
     << "dst[i] = static_cast<" << dst_dtype_os.str() << ">(tilelang_fast_exp2f(expr)); } }";
}

void CodeGenBlackhole::PrintScalarExp2Affine(const tvm::tir::CallNode* op,
                                             std::ostream& os) {
  const auto* dst_var = AsHandleVar(op->args[0]);
  const auto* lhs_var = AsHandleVar(op->args[1]);
  const auto* rhs_var = AsHandleVar(op->args[2]);
  ICHECK(dst_var && lhs_var && rhs_var)
      << "tl.blackhole.scalar_exp2_affine expects direct handle vars";
  auto dst_dtype_it = handle_data_type_.find(dst_var);
  auto lhs_dtype_it = handle_data_type_.find(lhs_var);
  auto rhs_dtype_it = handle_data_type_.find(rhs_var);
  ICHECK(dst_dtype_it != handle_data_type_.end() && lhs_dtype_it != handle_data_type_.end() &&
         rhs_dtype_it != handle_data_type_.end())
      << "Missing handle dtypes for tl.blackhole.scalar_exp2_affine";

  std::ostringstream dst_dtype_os;
  std::ostringstream lhs_dtype_os;
  std::ostringstream rhs_dtype_os;
  PrintType(dst_dtype_it->second, dst_dtype_os);
  PrintType(lhs_dtype_it->second, lhs_dtype_os);
  PrintType(rhs_dtype_it->second, rhs_dtype_os);

  os << "{ " << dst_dtype_os.str() << "* dst = reinterpret_cast<" << dst_dtype_os.str() << "*>(";
  PrintExpr(op->args[0], os);
  os << "); const " << lhs_dtype_os.str() << "* lhs = reinterpret_cast<const "
     << lhs_dtype_os.str() << "*>(";
  PrintExpr(op->args[1], os);
  os << "); const " << rhs_dtype_os.str() << "* rhs = reinterpret_cast<const "
     << rhs_dtype_os.str() << "*>(";
  PrintExpr(op->args[2], os);
  os << "); const float lhs_scale = ";
  PrintExpr(op->args[3], os);
  os << "; const float rhs_scale = ";
  PrintExpr(op->args[4], os);
  os << "; tilelang_scalar_exp2_affine(dst, lhs, rhs, lhs_scale, rhs_scale); }";
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
