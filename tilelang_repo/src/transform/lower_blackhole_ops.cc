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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

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
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_matmul_tiles;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_tile_regs_commit;
using tir::builtin::blackhole_tile_regs_wait;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_pack_tile;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_tile_regs_acquire;
using tir::builtin::blackhole_tile_regs_release;
using tir::builtin::blackhole_noc_async_read;
using tir::builtin::blackhole_noc_async_write;
using tir::builtin::blackhole_noc_async_read_barrier;
using tir::builtin::blackhole_noc_async_write_barrier;
using tir::builtin::blackhole_read_tile_to_cb;
using tir::builtin::blackhole_write_tile_from_cb;
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

static PrimExpr ScalarizeVectorizedIndex(const PrimExpr& index) {
  if (const auto* ramp = index.as<tir::RampNode>()) {
    return ramp->base;
  }
  return index;
}

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

// Helper to get storage scope from buffer
static std::string GetStorageScope(const Buffer& buffer) {
  // Use the scope() method which returns ffi::String
  ffi::String scope = buffer.scope();
  if (scope.length() > 0) {
    return std::string(scope);
  }
  return "";
}

LowerBlackholeOps::LowerBlackholeOps() : next_cb_id_(0) {
  // Initialize CB allocation tracking
}

PrimFunc LowerBlackholeOps::Transform(const PrimFunc& func) {
  current_func_ = func;
  buffer_to_cb_.clear();
  cb_requirements_.clear();
  next_input_cb_ = 0;
  next_output_cb_ = 16;
  next_intermediate_cb_ = 32;
  next_cb_id_ = 0;
  saw_copy_op_ = false;
  needs_copy_runtime_args_ = false;
  copy_input_buffer_name_.clear();
  copy_output_buffer_name_.clear();
  copy_input_shape_.clear();
  copy_output_shape_.clear();
  copy_intermediate_shape_.clear();
  gemm_a_buffer_name_.clear();
  gemm_b_buffer_name_.clear();
  gemm_c_buffer_name_.clear();
  gemm_m_ = 0;
  gemm_n_ = 0;
  gemm_k_ = 0;

  // Transform the function body
  Stmt body = VisitStmt(func->body);

  // Create new function with transformed body
  PrimFunc new_func = func;
  new_func.CopyOnWrite()->body = body;

  // Store CB requirements in function attributes for PlanBlackholeCB
  StoreCBRequirements(new_func);
  StoreRuntimeArgs(new_func);
  StoreSegmentPlan(new_func);

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

// Allocate a CB ID for a buffer
int LowerBlackholeOps::AllocateCBId(const Buffer& buffer, CBType type) {
  // Check if this buffer already has a CB assigned
  auto it = buffer_to_cb_.find(buffer);
  if (it != buffer_to_cb_.end()) {
    return it->second;
  }

  // Allocate new CB ID based on type
  int cb_id;
  switch (type) {
    case CBType::kInput:
      cb_id = next_input_cb_++;
      break;
    case CBType::kOutput:
      cb_id = next_output_cb_++;
      break;
    default:
      cb_id = next_intermediate_cb_++;
      break;
  }

  buffer_to_cb_[buffer] = cb_id;

  // Record CB requirement
  CBRequirement req;
  req.name = buffer->name;
  req.type = type;
  req.lifetime_begin = static_cast<int>(cb_requirements_.size());
  req.lifetime_end = req.lifetime_begin;

  // Calculate page size from buffer shape
  int64_t total_elements = 1;
  for (const auto& shape_dim : buffer->shape) {
    if (const auto* int_imm = shape_dim.as<IntImmNode>()) {
      total_elements *= int_imm->value;
    }
  }
  const int total_bytes = static_cast<int>(total_elements * buffer->dtype.bytes());
  req.page_size = EstimateCopyPageSize(buffer);
  req.num_pages = std::max(
      2, req.page_size > 0 ? (total_bytes + req.page_size - 1) / req.page_size : 2);

  // Determine data format
  if (buffer->dtype.is_float()) {
    if (buffer->dtype.bits() == 16) {
      req.data_format = "Float16";
    } else if (buffer->dtype.bits() == 32) {
      req.data_format = "Float32";
    } else if (buffer->dtype.bits() == 8) {
      req.data_format = "Bfp8";
    }
  } else if (buffer->dtype.is_int()) {
    if (buffer->dtype.bits() == 32) {
      req.data_format = "Int32";
    } else if (buffer->dtype.bits() == 16) {
      req.data_format = "Int16";
    }
  }

  cb_requirements_.push_back(req);

  return cb_id;
}

int LowerBlackholeOps::EstimateCopyPageSize(const Buffer& buffer) const {
  int64_t total_elements = 1;
  bool all_static = true;
  for (const auto& shape_dim : buffer->shape) {
    if (const auto* int_imm = shape_dim.as<IntImmNode>()) {
      total_elements *= int_imm->value;
    } else {
      all_static = false;
      break;
    }
  }

  if (!all_static || total_elements <= 0) {
    return 2048;
  }

  const int64_t dtype_bytes = buffer->dtype.bytes();
  const int64_t total_bytes = total_elements * dtype_bytes;
  const int64_t default_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * dtype_bytes;
  return static_cast<int>(std::max<int64_t>(dtype_bytes, std::min(total_bytes, default_tile_bytes)));
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
  for (const auto& req : cb_requirements_) {
    Map<String, Any> req_map;
    req_map.Set("name", String(req.name));
    req_map.Set("type", String(req.type == CBType::kInput ? "input" :
                               req.type == CBType::kOutput ? "output" : "intermediate"));
    req_map.Set("page_size", Integer(req.page_size));
    req_map.Set("num_pages", Integer(req.num_pages));
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

  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  Array<Any> runtime_args;
  auto push_arg = [&](const std::string& name, const char* kind, const char* dtype,
                      const std::string& buffer_name = "") {
    Map<String, Any> arg_map;
    arg_map.Set("name", String(name));
    arg_map.Set("kind", String(kind));
    arg_map.Set("dtype", String(dtype));
    if (!buffer_name.empty()) {
      arg_map.Set("buffer", String(buffer_name));
    }
    runtime_args.push_back(arg_map);
  };

  const std::string input_arg_name =
      copy_input_buffer_name_.empty() ? "input_addr" : copy_input_buffer_name_ + "_addr";
  const std::string output_arg_name =
      copy_output_buffer_name_.empty() ? "output_addr" : copy_output_buffer_name_ + "_addr";

  push_arg(input_arg_name, "input_buffer_addr32", "uint32", copy_input_buffer_name_);
  push_arg(output_arg_name, "output_buffer_addr32", "uint32", copy_output_buffer_name_);
  push_arg("current_work_linear_id", "current_work_linear_id", "uint32");
  push_arg("tile_count", "tile_count", "uint32");
  push_arg("scratch_l1_addr", "scratch_l1_buffer_addr32", "uint32");

  attrs.Set("blackhole.runtime_args", runtime_args);
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

// Detect matmul operation using Op comparison
bool LowerBlackholeOps::IsMatmulCall(const CallNode* op) const {
  if (!op->op->IsInstance<OpNode>()) return false;

  Op call_op = Downcast<Op>(op->op);
  return call_op->name == "tl.tileop.gemm_py";
}

std::string LowerBlackholeOps::DataTypeToDataFormat(DataType dtype) {
  if (dtype.is_bfloat16()) return "Float16_b";
  if (dtype.is_float16()) return "Float16";
  if (dtype.is_float() && dtype.bits() == 32) return "Float32";
  if (dtype.is_float() && dtype.bits() == 8) return "Bfp8";
  if (dtype.is_int() && dtype.bits() == 32) return "Int32";
  if (dtype.is_int() && dtype.bits() == 16) return "Int16";
  return "Float16_b";  // safe fallback
}

void LowerBlackholeOps::ExtractGemmInfo(const CallNode* op) {
  // tl.tileop.gemm_py args layout (from gemm_op.py _gemm_impl):
  //   [0]=A_region, [1]=B_region, [2]=C_region,
  //   [3]=transA, [4]=transB, [5]=M, [6]=N, [7]=K, ...
  const auto& args = op->args;
  ICHECK_GE(args.size(), 8U) << "tl.tileop.gemm_py expects at least 8 args";

  tir::BufferRegion a_region = NormalizeToBufferRegion(args[0]);
  tir::BufferRegion b_region = NormalizeToBufferRegion(args[1]);
  tir::BufferRegion c_region = NormalizeToBufferRegion(args[2]);

  gemm_a_buffer_name_ = std::string(a_region->buffer->name);
  gemm_b_buffer_name_ = std::string(b_region->buffer->name);
  gemm_c_buffer_name_ = std::string(c_region->buffer->name);
  gemm_ab_dtype_ = a_region->buffer->dtype;
  gemm_c_dtype_ = c_region->buffer->dtype;

  if (const auto* imm = args[5].as<IntImmNode>()) gemm_m_ = static_cast<int>(imm->value);
  if (const auto* imm = args[6].as<IntImmNode>()) gemm_n_ = static_cast<int>(imm->value);
  if (const auto* imm = args[7].as<IntImmNode>()) gemm_k_ = static_cast<int>(imm->value);

  // Register CBs: A→CB0 (kInput), B→CB1 (kInput), C→CB16 (kOutput).
  // PlanBlackholeCB assigns IDs sequentially: kInput starts at 0, kOutput at 16.
  // Pushing in order A, B (kInput), then C (kOutput) produces exactly CB0, CB1, CB16.
  const int ab_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * gemm_ab_dtype_.bytes();
  const int c_tile_bytes = kBlackholeTileRows * kBlackholeTileCols * gemm_c_dtype_.bytes();

  CBRequirement req_a;
  req_a.name = gemm_a_buffer_name_;
  req_a.type = CBType::kInput;
  req_a.page_size = ab_tile_bytes;
  req_a.num_pages = 2;  // double-buffered for pipeline overlap
  req_a.data_format = DataTypeToDataFormat(gemm_ab_dtype_);
  req_a.lifetime_begin = 0;
  req_a.lifetime_end = 0;
  cb_requirements_.push_back(req_a);

  CBRequirement req_b = req_a;
  req_b.name = gemm_b_buffer_name_;
  cb_requirements_.push_back(req_b);

  CBRequirement req_c;
  req_c.name = gemm_c_buffer_name_;
  req_c.type = CBType::kOutput;
  req_c.page_size = c_tile_bytes;
  req_c.num_pages = 1;
  req_c.data_format = DataTypeToDataFormat(gemm_c_dtype_);
  req_c.lifetime_begin = 0;
  req_c.lifetime_end = 0;
  cb_requirements_.push_back(req_c);
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

  // Helper to check if scope indicates CB (shared memory)
  auto isCBScope = [](const std::string& scope) {
    return scope == "shared" || scope == "shared.dyn" || scope.find("shared") == 0;
  };

  // Helper to check if scope indicates DRAM (global memory)
  auto isDRAMScope = [](const std::string& scope) {
    return scope.empty() || scope == "global";
  };

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
  if (subst_map.empty()) {
    return expr;
  }
  Analyzer analyzer;
  return analyzer.Simplify(tir::Substitute(expr, subst_map));
}

PrimExpr LowerBlackholeOps::InferCopyTileIndex(const BufferStoreNode* op,
                                               const Var& loop_var) const {
  const auto* load = op->value.as<BufferLoadNode>();
  ICHECK(load) << "InferCopyTileIndex requires BufferLoad copy source";

  CopyDirection direction = GetCopyDirection(op);
  const Buffer& global_buffer =
      direction == CopyDirection::kDramToCB ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      direction == CopyDirection::kDramToCB ? load->indices : op->indices;

  Analyzer analyzer;
  PrimExpr base_row;
  PrimExpr base_col;
  int64_t cols_value = 0;

  if (global_indices.size() >= 2U && global_buffer->shape.size() >= 2U) {
    base_row = ZeroThreadAndLoopVars(global_indices[0], loop_var);
    base_col = ZeroThreadAndLoopVars(global_indices[1], loop_var);
    const auto* cols_imm = global_buffer->shape[1].as<IntImmNode>();
    ICHECK(cols_imm) << "Blackhole staged copy currently expects static tile width";
    cols_value = cols_imm->value;
  } else if (global_indices.size() == 1U) {
    const Array<Integer>& annotated_shape =
        direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
    ICHECK_GE(annotated_shape.size(), 2U)
        << "Blackhole staged copy requires rank-2 shape metadata after FlattenBuffer";
    cols_value = annotated_shape[1]->value;
    PrimExpr linear_index = ScalarizeVectorizedIndex(global_indices[0]);
    PrimExpr row_index = analyzer.Simplify(tir::FloorDiv(linear_index, IntImm32(cols_value)));
    PrimExpr col_index = analyzer.Simplify(tir::FloorMod(linear_index, IntImm32(cols_value)));
    base_row = ZeroThreadAndLoopVars(row_index, loop_var);
    base_col = ZeroThreadAndLoopVars(col_index, loop_var);
  } else {
    LOG(FATAL) << "Blackhole staged copy currently expects rank-2 tiled regions";
  }

  PrimExpr tile_row = analyzer.Simplify(tir::FloorDiv(base_row, IntImm32(kBlackholeTileRows)));
  PrimExpr tile_col = analyzer.Simplify(tir::FloorDiv(base_col, IntImm32(kBlackholeTileCols)));

  ICHECK_EQ(cols_value % kBlackholeTileCols, 0)
      << "Blackhole staged copy currently expects 32-wide tile alignment";
  PrimExpr tiles_per_row = IntImm32(static_cast<int>(cols_value / kBlackholeTileCols));
  return analyzer.Simplify(tile_row * tiles_per_row + tile_col);
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

  Analyzer analyzer;
  PrimExpr base_row;
  PrimExpr base_col;
  int64_t cols_value = 0;

  if (global_indices.size() >= 2U && global_buffer->shape.size() >= 2U) {
    base_row = ZeroThreadAndLoopVars(global_indices[0], loop_vars_to_zero);
    base_col = ZeroThreadAndLoopVars(global_indices[1], loop_vars_to_zero);
    const auto* cols_imm = global_buffer->shape[1].as<IntImmNode>();
    ICHECK(cols_imm) << "Blackhole staged copy currently expects static tile width";
    cols_value = cols_imm->value;
  } else if (global_indices.size() == 1U) {
    const Array<Integer>& annotated_shape =
        direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
    ICHECK_GE(annotated_shape.size(), 2U)
        << "Blackhole staged copy requires rank-2 shape metadata after FlattenBuffer";
    cols_value = annotated_shape[1]->value;
    PrimExpr linear_index = ScalarizeVectorizedIndex(global_indices[0]);
    PrimExpr row_index = analyzer.Simplify(tir::FloorDiv(linear_index, IntImm32(cols_value)));
    PrimExpr col_index = analyzer.Simplify(tir::FloorMod(linear_index, IntImm32(cols_value)));
    base_row = ZeroThreadAndLoopVars(row_index, loop_vars_to_zero);
    base_col = ZeroThreadAndLoopVars(col_index, loop_vars_to_zero);
  } else {
    LOG(FATAL) << "Blackhole staged copy currently expects rank-2 tiled regions";
  }

  PrimExpr tile_row =
      analyzer.Simplify(tir::FloorDiv(base_row, IntImm32(kBlackholeTileRows)));
  PrimExpr tile_col =
      analyzer.Simplify(tir::FloorDiv(base_col, IntImm32(kBlackholeTileCols)));

  ICHECK_EQ(cols_value % kBlackholeTileCols, 0)
      << "Blackhole staged copy currently expects 32-wide tile alignment";
  PrimExpr tiles_per_row = IntImm32(static_cast<int>(cols_value / kBlackholeTileCols));
  return analyzer.Simplify(tile_row * tiles_per_row + tile_col);
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
    nested_loop_vars->push_back(loop->loop_var);
    const BufferStoreNode* store = FindNestedCopyStore(loop->body, nested_loop_vars);
    if (!store) {
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
    loop_stack->push_back(loop->loop_var);
    CollectNestedCopyStores(loop->body, loop_stack, matches);
    loop_stack->pop_back();
  }
}

void LowerBlackholeOps::RecordStagedCopyBufferBinding(const BufferStoreNode* op,
                                                      CopyDirection direction) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) {
    return;
  }
  needs_copy_runtime_args_ = true;
  if (direction == CopyDirection::kDramToCB) {
    copy_input_buffer_name_ = load->buffer->name;
  } else if (direction == CopyDirection::kCBToDram) {
    copy_output_buffer_name_ = op->buffer->name;
  }
}

void LowerBlackholeOps::RecordDramToDramCopy(const BufferStoreNode* op) {
  const auto* load = op->value.as<BufferLoadNode>();
  if (!load) return;

  auto ensure_requirement = [&](const Buffer& buffer, CBType type) {
    auto it = buffer_to_cb_.find(buffer);
    if (it != buffer_to_cb_.end()) {
      return;
    }

    switch (type) {
      case CBType::kInput:
        buffer_to_cb_[buffer] = next_input_cb_++;
        break;
      case CBType::kOutput:
        buffer_to_cb_[buffer] = next_output_cb_++;
        break;
      default:
        buffer_to_cb_[buffer] = next_intermediate_cb_++;
        break;
    }

    CBRequirement req;
    req.name = buffer->name;
    req.type = type;
    req.lifetime_begin = static_cast<int>(cb_requirements_.size());
    req.lifetime_end = req.lifetime_begin;
    req.page_size = EstimateCopyPageSize(buffer);
    req.num_pages = 1;
    if (buffer->dtype.is_float()) {
      req.data_format = buffer->dtype.bits() == 16 ? "Float16_b" : "Float32";
    } else if (buffer->dtype.is_uint()) {
      req.data_format = buffer->dtype.bits() == 16 ? "UInt16" : "UInt32";
    } else if (buffer->dtype.is_int()) {
      req.data_format = buffer->dtype.bits() == 16 ? "UInt16" : "UInt32";
    } else {
      req.data_format = "Float16_b";
    }
    cb_requirements_.push_back(req);
  };

  ensure_requirement(load->buffer, CBType::kInput);
  ensure_requirement(op->buffer, CBType::kOutput);
  needs_copy_runtime_args_ = true;
  copy_input_buffer_name_ = load->buffer->name;
  copy_output_buffer_name_ = op->buffer->name;
}

Stmt LowerBlackholeOps::GenerateMatmulSequence(const CallNode* op) {
  // CB IDs for GEMM: A=CB0, B=CB1, C=CB16 (matched by PlanBlackholeCB allocation order).
  constexpr int in0_id = 0;
  constexpr int in1_id = 1;
  constexpr int out_id = 16;
  // num_k_tiles: how many 32-element K-tiles to accumulate
  const int num_k_tiles = (gemm_k_ > 0) ? (gemm_k_ / kBlackholeTileCols) : 1;

  std::vector<Stmt> stmts;

  // 1. Initialize MM engine
  stmts.push_back(MakeBlackholeCall(
      blackhole_mm_init(),
      {IntImm32(in0_id), IntImm32(in1_id), IntImm32(out_id)}));

  // 2. Acquire tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_acquire(), {}));

  // 3. Generate K-tile loop (statically unrolled for fixed-shape GEMM)
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    // Wait for input tiles
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(),
        {IntImm32(in0_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_wait_front(),
        {IntImm32(in1_id), IntImm32(1)}));

    // Perform matmul
    stmts.push_back(MakeBlackholeCall(
        blackhole_matmul_tiles(),
        {IntImm32(in0_id), IntImm32(in1_id),
         IntImm32(0), IntImm32(0), IntImm32(0)}));

    // Pop input tiles
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(in0_id), IntImm32(1)}));
    stmts.push_back(MakeBlackholeCall(
        blackhole_cb_pop_front(),
        {IntImm32(in1_id), IntImm32(1)}));
  }

  // 4-5. Commit and wait
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_commit(), {}));
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_wait(), {}));

  // 6-8. Pack and push output
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_reserve_back(),
      {IntImm32(out_id), IntImm32(1)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_pack_tile(),
      {IntImm32(0), IntImm32(out_id)}));
  stmts.push_back(MakeBlackholeCall(
      blackhole_cb_push_back(),
      {IntImm32(out_id), IntImm32(1)}));

  // 9. Release tile registers
  stmts.push_back(MakeBlackholeCall(blackhole_tile_regs_release(), {}));

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

      const int src_cb_id = buffer_to_cb_.at(load->buffer);
      const int dst_cb_id = buffer_to_cb_.at(op->buffer);
      const int tile_bytes = EstimateCopyPageSize(load->buffer);

      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(src_cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {load->buffer->data, IntImm32(0), IntImm32(src_cb_id), IntImm32(tile_bytes), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(src_cb_id), IntImm32(1)}));

      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(src_cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_tile_from_cb(),
          {IntImm32(src_cb_id), op->buffer->data, IntImm32(0), IntImm32(tile_bytes), IntImm32(0)}));
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
      int src_cb_id = AllocateCBId(load->buffer, CBType::kIntermediate);
      int dst_cb_id = AllocateCBId(op->buffer, CBType::kIntermediate);

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
  switch (direction) {
    case CopyDirection::kDramToCB: {
      int cb_id = AllocateCBId(op->buffer, CBType::kIntermediate);
      int tile_bytes = EstimateCopyPageSize(op->buffer);
      RecordStagedCopyBufferBinding(op, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {load->buffer->data, tile_index, IntImm32(cb_id), IntImm32(tile_bytes), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      return SeqStmt::Flatten(stmts);
    }
    case CopyDirection::kCBToDram: {
      int cb_id = AllocateCBId(load->buffer, CBType::kIntermediate);
      int tile_bytes = EstimateCopyPageSize(load->buffer);
      RecordStagedCopyBufferBinding(op, direction);
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_tile_from_cb(),
          {IntImm32(cb_id), op->buffer->data, tile_index, IntImm32(tile_bytes), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      return SeqStmt::Flatten(stmts);
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

  const Buffer& shared_buffer =
      direction == CopyDirection::kDramToCB ? op->buffer : load->buffer;
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  if (shared_buffer->shape.size() >= 2U) {
    const auto* rows_imm = shared_buffer->shape[0].as<IntImmNode>();
    const auto* cols_imm = shared_buffer->shape[1].as<IntImmNode>();
    ICHECK(rows_imm && cols_imm)
        << "Blackhole staged copy currently expects static shared tile shapes";
    shared_rows = rows_imm->value;
    shared_cols = cols_imm->value;
  } else {
    ICHECK_GE(copy_intermediate_shape_.size(), 2U)
        << "Blackhole staged copy currently expects rank-2 shared tiles";
    shared_rows = copy_intermediate_shape_[0]->value;
    shared_cols = copy_intermediate_shape_[1]->value;
  }
  ICHECK_EQ(shared_rows % kBlackholeTileRows, 0)
      << "Blackhole staged copy currently expects shared tile height aligned to 32";
  ICHECK_EQ(shared_cols % kBlackholeTileCols, 0)
      << "Blackhole staged copy currently expects shared tile width aligned to 32";

  const int subtile_rows = static_cast<int>(shared_rows / kBlackholeTileRows);
  const int subtile_cols = static_cast<int>(shared_cols / kBlackholeTileCols);
  const int tile_bytes = kBlackholeTileRows * kBlackholeTileCols * shared_buffer->dtype.bytes();

  std::vector<Stmt> stmts;
  Analyzer analyzer;
  auto make_tile_index = [&](int subtile_row, int subtile_col) -> PrimExpr {
    PrimExpr tile_index = base_tile_index;
    if (subtile_row != 0) {
      const Array<Integer>& global_shape =
          direction == CopyDirection::kDramToCB ? copy_input_shape_ : copy_output_shape_;
      int64_t global_cols = 0;
      if ((direction == CopyDirection::kDramToCB ? load->buffer : op->buffer)->shape.size() >= 2U) {
        const auto* global_cols_imm =
            (direction == CopyDirection::kDramToCB ? load->buffer : op->buffer)->shape[1]
                .as<IntImmNode>();
        ICHECK(global_cols_imm)
            << "Blackhole staged copy currently expects static global buffer width";
        global_cols = global_cols_imm->value;
      } else {
        ICHECK_GE(global_shape.size(), 2U)
            << "Blackhole staged copy requires rank-2 global shape metadata after FlattenBuffer";
        global_cols = global_shape[1]->value;
      }
      ICHECK_EQ(global_cols % kBlackholeTileCols, 0)
          << "Blackhole staged copy currently expects global width aligned to 32";
      int tiles_per_row = static_cast<int>(global_cols / kBlackholeTileCols);
      tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_row * tiles_per_row));
    }
    if (subtile_col != 0) {
      tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_col));
    }
    return tile_index;
  };

  if (direction == CopyDirection::kDramToCB) {
    int cb_id = AllocateCBId(op->buffer, CBType::kIntermediate);
    RecordStagedCopyBufferBinding(op, direction);
    for (int subtile_row = 0; subtile_row < subtile_rows; ++subtile_row) {
      for (int subtile_col = 0; subtile_col < subtile_cols; ++subtile_col) {
        PrimExpr tile_index = make_tile_index(subtile_row, subtile_col);
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_read_tile_to_cb(),
            {load->buffer->data, tile_index, IntImm32(cb_id), IntImm32(tile_bytes), IntImm32(0)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      }
    }
    return SeqStmt::Flatten(stmts);
  }

  if (direction == CopyDirection::kCBToDram) {
    int cb_id = AllocateCBId(load->buffer, CBType::kIntermediate);
    RecordStagedCopyBufferBinding(op, direction);
    for (int subtile_row = 0; subtile_row < subtile_rows; ++subtile_row) {
      for (int subtile_col = 0; subtile_col < subtile_cols; ++subtile_col) {
        PrimExpr tile_index = make_tile_index(subtile_row, subtile_col);
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_write_tile_from_cb(),
            {IntImm32(cb_id), op->buffer->data, tile_index, IntImm32(tile_bytes), IntImm32(0)}));
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_pop_front(), {IntImm32(cb_id), IntImm32(1)}));
      }
    }
    return SeqStmt::Flatten(stmts);
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
  if (shared_buffer->shape.size() >= 2U) {
    const auto* rows_imm = shared_buffer->shape[0].as<IntImmNode>();
    const auto* cols_imm = shared_buffer->shape[1].as<IntImmNode>();
    ICHECK(rows_imm && cols_imm)
        << "Blackhole staged copy currently expects static shared tile shapes";
    shared_rows = rows_imm->value;
    shared_cols = cols_imm->value;
  } else {
    ICHECK_GE(copy_intermediate_shape_.size(), 2U)
        << "Blackhole staged copy currently expects rank-2 shared tiles";
    shared_rows = copy_intermediate_shape_[0]->value;
    shared_cols = copy_intermediate_shape_[1]->value;
  }
  ICHECK_EQ(shared_rows % kBlackholeTileRows, 0)
      << "Blackhole staged copy currently expects shared tile height aligned to 32";
  ICHECK_EQ(shared_cols % kBlackholeTileCols, 0)
      << "Blackhole staged copy currently expects shared tile width aligned to 32";

  int64_t global_cols = 0;
  if (dram_load->buffer->shape.size() >= 2U) {
    const auto* global_cols_imm = dram_load->buffer->shape[1].as<IntImmNode>();
    ICHECK(global_cols_imm)
        << "Blackhole staged copy currently expects static global buffer width";
    global_cols = global_cols_imm->value;
  } else {
    ICHECK_GE(copy_input_shape_.size(), 2U)
        << "Blackhole staged copy requires rank-2 global shape metadata after FlattenBuffer";
    global_cols = copy_input_shape_[1]->value;
  }
  ICHECK_EQ(global_cols % kBlackholeTileCols, 0)
      << "Blackhole staged copy currently expects global width aligned to 32";
  const int tiles_per_row = static_cast<int>(global_cols / kBlackholeTileCols);
  const int subtile_rows = static_cast<int>(shared_rows / kBlackholeTileRows);
  const int subtile_cols = static_cast<int>(shared_cols / kBlackholeTileCols);
  const int tile_bytes = kBlackholeTileRows * kBlackholeTileCols * shared_buffer->dtype.bytes();
  const int cb_id = AllocateCBId(shared_buffer, CBType::kIntermediate);
  RecordStagedCopyBufferBinding(dram_to_cb, CopyDirection::kDramToCB);
  RecordStagedCopyBufferBinding(cb_to_dram, CopyDirection::kCBToDram);

  Analyzer analyzer;
  std::vector<Stmt> stmts;
  for (int subtile_row = 0; subtile_row < subtile_rows; ++subtile_row) {
    for (int subtile_col = 0; subtile_col < subtile_cols; ++subtile_col) {
      PrimExpr tile_index = base_tile_index;
      if (subtile_row != 0) {
        tile_index =
            analyzer.Simplify(tile_index + IntImm32(subtile_row * tiles_per_row));
      }
      if (subtile_col != 0) {
        tile_index = analyzer.Simplify(tile_index + IntImm32(subtile_col));
      }
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_read_tile_to_cb(),
          {dram_load->buffer->data, tile_index, IntImm32(cb_id), IntImm32(tile_bytes), IntImm32(0)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_wait_front(), {IntImm32(cb_id), IntImm32(1)}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_write_tile_from_cb(),
          {IntImm32(cb_id), cb_to_dram->buffer->data, tile_index, IntImm32(tile_bytes), IntImm32(0)}));
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

// Parse a colon-separated string into fields
Stmt LowerBlackholeOps::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    const std::string thread_tag = iv->thread_tag;
    const bool zero_thread_var = thread_tag.rfind("threadIdx.", 0) == 0;
    if (zero_thread_var) {
      thread_index_vars_.insert(iv->var.get());
    }
    Stmt body = VisitStmt(op->body);
    if (zero_thread_var) {
      thread_index_vars_.erase(iv->var.get());
    }
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt LowerBlackholeOps::VisitStmt_(const ForNode* op) {
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

    const std::string kind = get_string("kind");
    const std::string direction = get_string("direction");

    if (kind == "fused_staged_copy") {
      copy_input_buffer_name_ = get_string("src_buffer");
      copy_output_buffer_name_ = get_string("dst_buffer");
      copy_input_shape_ = get_shape("src_shape");
      copy_output_shape_ = get_shape("dst_shape");
      copy_intermediate_shape_ = get_shape("mid_shape");
    } else if (direction == "dram_to_cb") {
      copy_input_buffer_name_ = get_string("src_buffer");
      copy_input_shape_ = get_shape("src_shape");
      copy_intermediate_shape_ = get_shape("mid_shape");
    } else if (direction == "cb_to_dram") {
      copy_output_buffer_name_ = get_string("dst_buffer");
      copy_output_shape_ = get_shape("dst_shape");
      copy_intermediate_shape_ = get_shape("mid_shape");
    } else if (direction == "dram_to_dram") {
      copy_input_buffer_name_ = get_string("src_buffer");
      copy_output_buffer_name_ = get_string("dst_buffer");
      copy_input_shape_ = get_shape("src_shape");
      copy_output_shape_ = get_shape("dst_shape");
    }

    needs_copy_runtime_args_ = true;
    saw_copy_op_ = true;
  }

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
  }

  std::vector<Var> nested_loop_vars;
  if (const auto* nested_store = FindNestedCopyStore(op->body, &nested_loop_vars)) {
    CopyDirection direction = GetCopyDirection(nested_store);
    if (direction == CopyDirection::kDramToCB || direction == CopyDirection::kCBToDram) {
      saw_copy_op_ = true;
      std::vector<Var> loop_vars_to_zero{op->loop_var};
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
        PrimExpr tile_index = InferCopyTileIndex(store, op->loop_var);
        return GenerateCopySequence(store, tile_index);
      }
    }
  }
  return StmtExprMutator::VisitStmt_(op);
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
  if (IsCopyOperation(op)) {
    saw_copy_op_ = true;
    return GenerateCopySequence(op);
  }
  // Return original statement without recursion
  return GetRef<Stmt>(op);
}

// Modern TVM pass registration using CreatePrimFuncPass
tir::transform::Pass LowerBlackholeOpsPass() {
  auto fpass = [](PrimFunc func, IRModule m, tir::transform::PassContext ctx) -> PrimFunc {
    return LowerBlackholeOps().Transform(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.LowerBlackholeOps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.LowerBlackholeOps", LowerBlackholeOpsPass);
}

}  // namespace tl
}  // namespace tvm
