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
 * \file lower_blackhole_transport.cc
 * \brief Staged copy and transport source emission for Blackhole lowering.
 */

#include "lower_blackhole_ops.h"

#include "common/blackhole_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include "runtime/thread_storage_scope.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "../tir/builtin_blackhole.h"

namespace tvm {
namespace tl {

using tir::AttrStmt;
using tir::AttrStmtNode;
using tir::Buffer;
using tir::BufferLoadNode;
using tir::BufferStoreNode;
using tir::Call;
using tir::Evaluate;
using tir::For;
using tir::ForNode;
using tir::FloorDiv;
using tir::FloorMod;
using tir::IfThenElseNode;
using tir::PrimFunc;
using tir::SeqStmt;
using tir::SeqStmtNode;
using tir::Stmt;
using tir::StringImm;
using tir::Var;
using tir::builtin::blackhole_cb_pop_front;
using tir::builtin::blackhole_cb_push_back;
using tir::builtin::blackhole_cb_reserve_back;
using tir::builtin::blackhole_cb_wait_front;
using tir::builtin::blackhole_noc_async_read;
using tir::builtin::blackhole_noc_async_read_barrier;
using tir::builtin::blackhole_noc_async_write;
using tir::builtin::blackhole_noc_async_write_barrier;
using tir::builtin::blackhole_pack_untilize_slice;
using tir::builtin::blackhole_pack_untilize_tile;
using tir::builtin::blackhole_read_page_to_cb;
using tir::builtin::blackhole_read_bcast_cols_to_cb;
using tir::builtin::blackhole_read_tile_to_cb;
using tir::builtin::blackhole_untilize_cb_front_tile;
using tir::builtin::blackhole_write_page_from_cb;
using tir::builtin::blackhole_write_tile_from_cb;
using tvm::DataType;
using tvm::Integer;
using tvm::IntImm;
using tvm::Range;
using tvm::arith::Analyzer;
using tvm::ffi::Array;
using tvm::ffi::GetRef;
using tvm::ffi::Map;

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

bool IsBlackholeAccumulatorLikeScope(const std::string& scope) {
  return scope == "blackhole.acc" || scope == "local.fragment" ||
         scope.rfind("local", 0) == 0;
}

PrimExpr ScalarizeVectorizedIndex(const PrimExpr& index) {
  if (const auto* ramp = index.as<tir::RampNode>()) {
    return ramp->base;
  }
  return index;
}

std::optional<std::vector<int64_t>> ExtractStaticShape(const Array<PrimExpr>& shape) {
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

int64_t ComputeStaticElementCount(const std::vector<int64_t>& shape) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    total_elements *= dim;
  }
  return total_elements;
}

int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  if (value <= 0) {
    return 1;
  }
  return static_cast<int>((value + divisor - 1) / divisor);
}

Stmt WrapSegmentStmtIfNeeded(const std::string& current_segment_kind,
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
  const bool scalar_element_page =
      page_bytes == 1 || page_bytes == 2 || page_bytes == 4 || page_bytes == 8;
  ICHECK(scalar_element_page || page_bytes % 64 == 0)
      << "Blackhole staged stick copy direct-path boundary requires either a scalar "
         "element page or a 64B-aligned transport page size, but got "
      << page_bytes << " bytes";
}

static bool IsDramToDeviceCopyDirection(CopyDirection direction) {
  return direction == CopyDirection::kDramToCB ||
         direction == CopyDirection::kDramToLocal;
}

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

static std::optional<int64_t> ResolveStaticRank1ExtentFromBufferOrMetadata(
    const Buffer& buffer,
    const Array<Integer>& fallback_shape,
    const char* static_shape_message) {
  if (buffer->shape.size() == 1U) {
    const auto* extent_imm = buffer->shape[0].as<IntImmNode>();
    ICHECK(extent_imm) << static_shape_message;
    return extent_imm->value;
  }
  if (buffer->shape.size() > 1U) {
    return std::nullopt;
  }
  if (fallback_shape.size() == 1U) {
    return fallback_shape[0]->value;
  }
  return std::nullopt;
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
        << "Blackhole staged copy currently expects shared tile width aligned to 32"
        << " for buffer " << BufferIdentityName(shared_buffer)
        << " with shared shape [" << shared_rows << ", " << shared_cols << "]";
    ICHECK_EQ(global_cols % kBlackholeTileCols, 0)
        << "Blackhole staged copy currently expects global width aligned to 32"
        << " for buffer " << BufferIdentityName(shared_buffer)
        << " with shared shape [" << shared_rows << ", " << shared_cols
        << "] and global shape [" << global_rows << ", " << global_cols << "]";
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

static bool UseStagedCopyPageTransportForShape(int64_t shared_rows,
                                               int64_t shared_cols) {
  return shared_rows > 0 && shared_rows % kBlackholeTileRows == 0 &&
         shared_cols > 0 && shared_cols % kBlackholeTileCols != 0;
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
  if (shared_buffer->shape.size() < 2U && fallback_shape.size() >= 2U) {
    return {fallback_shape[fallback_shape.size() - 2]->value,
            fallback_shape[fallback_shape.size() - 1]->value};
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

}  // namespace

tvm::ffi::Optional<Buffer> PlanTTKernelABI::FindSingleTileComputeDirectCopyTarget(
    const Buffer& source) const {
  if (!source.defined()) {
    return tvm::ffi::Optional<Buffer>();
  }
  const std::vector<std::string> source_identities = CollectBufferFlowIdentities(source);
  tvm::ffi::Optional<Buffer> selected;
  for (const auto& [target_identity, source_identity] : direct_copy_source_by_buffer_identity_) {
    if (target_identity.empty() || source_identity.empty()) {
      continue;
    }
    if (std::find(source_identities.begin(), source_identities.end(), source_identity) ==
        source_identities.end()) {
      continue;
    }
    if (tile_compute_input_buffers_.count(target_identity) == 0U) {
      continue;
    }
    auto target_it = buffer_by_identity_.find(target_identity);
    if (target_it == buffer_by_identity_.end() || !target_it->second.defined()) {
      continue;
    }
    const Buffer& target = target_it->second;
    if (!IsBlackholeAccumulatorLikeScope(GetStorageScope(target))) {
      continue;
    }
    if (selected.defined() && !SameBufferIdentity(selected.value(), target)) {
      return tvm::ffi::Optional<Buffer>();
    }
    selected = target;
  }
  return selected;
}

Buffer PlanTTKernelABI::SelectCBProducerBufferForDramToCB(const Buffer& source) const {
  tvm::ffi::Optional<Buffer> target = FindSingleTileComputeDirectCopyTarget(source);
  return target.defined() ? target.value() : source;
}

std::optional<std::pair<int64_t, int64_t>>
PlanTTKernelABI::InferStagedCopySharedShapeFromTransportCoverage(
    const BufferStoreNode* op, const std::vector<Var>& loop_vars_to_zero) const {
  const auto* load = GetCopyLoad(op);
  if (!load) {
    return std::nullopt;
  }

  const CopyDirection direction = GetCopyDirection(op);
  if (!IsDramToDeviceCopyDirection(direction) && direction != CopyDirection::kCBToDram) {
    return std::nullopt;
  }

  const Buffer& global_buffer =
      IsDramToDeviceCopyDirection(direction) ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      IsDramToDeviceCopyDirection(direction) ? load->indices : op->indices;
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
  const auto* load = GetCopyLoad(op);
  if (!load) {
    return {};
  }
  const CopyDirection direction = GetCopyDirection(op);
  const Buffer& shared_buffer =
      IsDramToDeviceCopyDirection(direction) ? op->buffer : load->buffer;
  if (direction == CopyDirection::kCBToDram) {
    ExactTiledCBValue live_value;
    if (TryCreateExactOutputLiveTiledCBValue(load->buffer, &live_value) ||
        TryCreateLiveExactTiledCBValue(load->buffer, &live_value)) {
      if (live_value.num_elements > 0 && live_value.row_width > 0 &&
          live_value.num_elements % live_value.row_width == 0) {
        Array<Integer> live_shape;
        live_shape.push_back(Integer(live_value.num_elements / live_value.row_width));
        live_shape.push_back(Integer(live_value.row_width));
        return live_shape;
      }
    }
  }
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

bool PlanTTKernelABI::UseStagedCopyPageTransport(const Buffer& shared_buffer) const {
  if (shared_buffer->shape.size() < 2U) {
    return false;
  }
  const auto* rows_imm = shared_buffer->shape[0].as<IntImmNode>();
  const auto* cols_imm = shared_buffer->shape[1].as<IntImmNode>();
  if (!rows_imm || !cols_imm) {
    return false;
  }
  return UseStagedCopyPageTransportForShape(rows_imm->value, cols_imm->value);
}

namespace {

bool IsZeroFillValue(const PrimExpr& expr) {
  if (tir::is_zero(expr)) {
    return true;
  }
  if (const auto* float_imm = expr.as<FloatImmNode>()) {
    return float_imm->value == 0.0;
  }
  if (const auto* cast = expr.as<tir::CastNode>()) {
    return IsZeroFillValue(cast->value);
  }
  return false;
}

bool IsIfThenElseCall(const tir::CallNode* call) {
  if (call == nullptr || call->args.size() != 3U) {
    return false;
  }
  if (call->op.same_as(tir::builtin::if_then_else())) {
    return true;
  }
  if (const auto* op = call->op.as<OpNode>()) {
    return op->name == "tir.if_then_else";
  }
  return false;
}

const BufferLoadNode* SelectGuardedCopyLoad(const PrimExpr& true_value,
                                            const PrimExpr& false_value) {
  if (const auto* true_load = true_value.as<BufferLoadNode>()) {
    if (IsZeroFillValue(false_value)) {
      return true_load;
    }
  }
  if (const auto* false_load = false_value.as<BufferLoadNode>()) {
    if (IsZeroFillValue(true_value)) {
      return false_load;
    }
  }
  return nullptr;
}

}  // namespace

const BufferLoadNode* PlanTTKernelABI::GetCopyLoad(
    const BufferStoreNode* op) const {
  if (op == nullptr) {
    return nullptr;
  }
  if (const auto* load = op->value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto* select = op->value.as<tir::SelectNode>()) {
    return SelectGuardedCopyLoad(select->true_value, select->false_value);
  }
  if (const auto* call = op->value.as<tir::CallNode>()) {
    if (IsIfThenElseCall(call)) {
      return SelectGuardedCopyLoad(call->args[1], call->args[2]);
    }
  }
  return nullptr;
}

bool PlanTTKernelABI::IsCopyOperation(const BufferStoreNode* op) const {
  if (const auto* load = GetCopyLoad(op)) {
    if (op->buffer.same_as(load->buffer)) {
      return false;
    }
    return GetCopyDirection(op) != CopyDirection::kUnknown;
  }
  return false;
}

CopyDirection PlanTTKernelABI::GetCopyDirection(const BufferStoreNode* op) const {
  const auto* load = GetCopyLoad(op);
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

  // DRAM -> local/accumulator via typed reader CB materialization.
  if (isDRAMScope(src_scope) && isAccumulatorLikeScope(dst_scope)) {
    return CopyDirection::kDramToLocal;
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

  // CB -> local/accumulator (shared/CB -> fragment/local materialization)
  if (isCBScope(src_scope) && isAccumulatorLikeScope(dst_scope)) {
    return CopyDirection::kCBToLocal;
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
  const auto* load = GetCopyLoad(op);
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
      IsDramToDeviceCopyDirection(direction) ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      IsDramToDeviceCopyDirection(direction) ? load->indices : op->indices;
  const Buffer& shared_buffer =
      IsDramToDeviceCopyDirection(direction) ? op->buffer : load->buffer;
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
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);
  const bool use_page_transport =
      UseStagedCopyPageTransportForShape(shared_rows, shared_cols);
  const StagedCopyTransportGeometry geometry = BuildStagedCopyTransportGeometry(
      shared_buffer, shared_rows, shared_cols, global_info.global_rows, global_info.global_cols,
      use_page_transport);
  return LinearizeStagedCopyTransportIndex(
      &analyzer, global_info.base_row, global_info.base_col, global_info.outer_slice_index,
      geometry);
}

PrimExpr PlanTTKernelABI::InferStagedCopyBaseTileIndex(
    const BufferStoreNode* op, const std::vector<Var>& loop_vars_to_zero) const {
  const auto* load = GetCopyLoad(op);
  ICHECK(load) << "InferStagedCopyBaseTileIndex requires BufferLoad copy source";

  CopyDirection direction = GetCopyDirection(op);
  const Buffer& global_buffer =
      IsDramToDeviceCopyDirection(direction) ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      IsDramToDeviceCopyDirection(direction) ? load->indices : op->indices;
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
      IsDramToDeviceCopyDirection(direction) ? op->buffer : load->buffer;
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
      UseStagedCopyPageTransportForShape(shared_rows, shared_cols));
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
  if (const auto* decl_buffer = stmt.as<tir::DeclBufferNode>()) {
    return FindNestedCopyStore(decl_buffer->body, nested_loop_vars);
  }
  if (const auto* if_then_else = stmt.as<IfThenElseNode>()) {
    if (const BufferStoreNode* store = FindNestedCopyStore(if_then_else->then_case,
                                                           nested_loop_vars)) {
      return store;
    }
    if (if_then_else->else_case.defined()) {
      return FindNestedCopyStore(if_then_else->else_case.value(), nested_loop_vars);
    }
    return nullptr;
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
  if (const auto* decl_buffer = stmt.as<tir::DeclBufferNode>()) {
    CollectNestedCopyStores(decl_buffer->body, loop_stack, matches);
    return;
  }
  if (const auto* if_then_else = stmt.as<IfThenElseNode>()) {
    CollectNestedCopyStores(if_then_else->then_case, loop_stack, matches);
    if (if_then_else->else_case.defined()) {
      CollectNestedCopyStores(if_then_else->else_case.value(), loop_stack, matches);
    }
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

void PlanTTKernelABI::RecordStagedCopyBufferBinding(const BufferStoreNode* op,
                                                      CopyDirection direction) {
  const auto* load = GetCopyLoad(op);
  if (!load) {
    return;
  }
  auto append_unique = [](std::vector<std::string>* buffers,
                          const std::string& buffer_name) {
    if (buffer_name.empty()) {
      return;
    }
    if (std::find(buffers->begin(), buffers->end(), buffer_name) ==
        buffers->end()) {
      buffers->push_back(buffer_name);
    }
  };
  needs_copy_runtime_args_ = true;
  if (IsDramToDeviceCopyDirection(direction)) {
    copy_input_buffer_ = load->buffer;
    copy_input_buffer_name_ = BufferIdentityName(load->buffer);
    append_unique(&copy_input_buffer_names_, copy_input_buffer_name_);
    copy_input_shape_ = GetEncodedCurrentBufferShape(load->buffer);
    copy_intermediate_shape_ = GetEncodedCurrentBufferShape(op->buffer);
    host_buffer_by_compute_operand_buffer_[BufferIdentityName(op->buffer)] =
        BufferIdentityName(load->buffer);
  } else if (direction == CopyDirection::kCBToDram) {
    copy_output_buffer_ = op->buffer;
    copy_output_buffer_name_ = BufferIdentityName(op->buffer);
    append_unique(&copy_output_buffer_names_, copy_output_buffer_name_);
    copy_output_shape_ = GetEncodedCurrentBufferShape(op->buffer);
    copy_intermediate_shape_ = GetEncodedCurrentBufferShape(load->buffer);
    host_buffer_by_compute_operand_buffer_[BufferIdentityName(load->buffer)] =
        BufferIdentityName(op->buffer);
  }
}

void PlanTTKernelABI::RecordDramToDramCopy(const BufferStoreNode* op) {
  const auto* load = GetCopyLoad(op);
  if (!load) return;
  auto append_unique = [](std::vector<std::string>* buffers,
                          const std::string& buffer_name) {
    if (buffer_name.empty()) {
      return;
    }
    if (std::find(buffers->begin(), buffers->end(), buffer_name) ==
        buffers->end()) {
      buffers->push_back(buffer_name);
    }
  };

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
  append_unique(&copy_input_buffer_names_, copy_input_buffer_name_);
  append_unique(&copy_output_buffer_names_, copy_output_buffer_name_);
  copy_input_shape_ = GetEncodedCurrentBufferShape(load->buffer);
  copy_output_shape_ = GetEncodedCurrentBufferShape(op->buffer);
}

Stmt PlanTTKernelABI::GenerateCopySequence(const BufferStoreNode* op) {
  return GenerateCopySequence(op, std::vector<Var>{});
}

Stmt PlanTTKernelABI::GenerateCopySequence(const BufferStoreNode* op,
                                           const std::vector<Var>& loop_vars_to_zero) {
  CopyDirection direction = GetCopyDirection(op);

  ICHECK(direction != CopyDirection::kUnknown)
      << "PlanTTKernelABI copy lowering requires an explicit copy-direction classification";

  const auto* load = GetCopyLoad(op);
  if (!load) return StmtExprMutator::VisitStmt_(op);

  std::vector<Stmt> stmts;

  switch (direction) {
    case CopyDirection::kDramToCB: {
      const Array<Integer> global_shape = GetEncodedCurrentBufferShape(load->buffer);
      const Array<Integer> shared_shape = GetEncodedCurrentBufferShape(op->buffer);
      const auto global_rank1 = ResolveStaticRank1ExtentFromBufferOrMetadata(
          load->buffer, global_shape,
          "Blackhole rank-1 staged copy requires static global shape");
      const auto shared_rank1 = ResolveStaticRank1ExtentFromBufferOrMetadata(
          op->buffer, shared_shape,
          "Blackhole rank-1 staged copy requires static shared shape");
      if (global_rank1.has_value() && shared_rank1.has_value()) {
        const std::string segment_kind = ResolveAccessorSegmentKind(direction);
        const bool segmented_gemm = !gemm_a_buffer_name_.empty() && segment_kind == "reader";
        const int cb_id = AllocateRequirementIndex(
            op->buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
        const int page_bytes = static_cast<int>(
            std::max<int64_t>(1, shared_rank1.value()) * op->buffer->dtype.bytes());
        const bool bcast_cols_source = IsBroadcastColsSourceBuffer(op->buffer);
        const int cb_page_bytes =
            bcast_cols_source
                ? kBlackholeTileRows * kBlackholeTileCols *
                      ExactTiledCBStorageDType(op->buffer->dtype).bytes()
                : page_bytes;
        SetRequirementPageLayout(cb_id, cb_page_bytes, 1);
        RecordStagedCopyBufferBinding(op, direction);
        const int accessor_slot = GetReadAccessorSlot(segment_kind, load->buffer, direction);
        PrimExpr page_index = IntImm32(0);
        if (load->indices.size() == 1U && shared_rank1.value() > 0) {
          Analyzer analyzer;
          page_index = analyzer.Simplify(
              FloorDiv(ZeroThreadAndLoopVars(ScalarizeVectorizedIndex(load->indices[0]),
                                             loop_vars_to_zero),
                       IntImm32(static_cast<int>(shared_rank1.value()))));
        }
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_reserve_back(), {IntImm32(cb_id), IntImm32(1)}));
        if (bcast_cols_source) {
          stmts.push_back(MakeBlackholeCall(
              blackhole_read_bcast_cols_to_cb(),
              {load->buffer->data, page_index, IntImm32(cb_id), IntImm32(page_bytes),
               IntImm32(accessor_slot), IntImm32(static_cast<int>(shared_rank1.value()))}));
        } else {
          stmts.push_back(MakeBlackholeCall(
              blackhole_read_page_to_cb(), {load->buffer->data, page_index, IntImm32(cb_id),
                                            IntImm32(page_bytes), IntImm32(accessor_slot),
                                            IntImm32(0)}));
        }
        stmts.push_back(MakeBlackholeCall(blackhole_noc_async_read_barrier(), {}));
        RegisterAccessor(segment_kind, load->buffer, accessor_slot, 2, 0, 0, 2,
                         page_bytes, {0}, false, "page_indexed");
        stmts.push_back(MakeBlackholeCall(
            blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
        RecordTiledCBLiveFormAliases(op->buffer, cb_id);
        return WrapSegmentStmtIfNeeded(current_segment_kind_, segment_kind,
                                       SeqStmt::Flatten(stmts));
      }
      // Staged DRAM -> shared copies should be collapsed at loop granularity.
      return GetRef<Stmt>(op);
    }

    case CopyDirection::kDramToLocal: {
      const PrimExpr base_tile_index =
          InferStagedCopyBaseTileIndex(op, loop_vars_to_zero);
      return GenerateStagedCopyLoopSequence(op, base_tile_index,
                                            loop_vars_to_zero);
    }

    case CopyDirection::kDramToDram: {
      // Stage 2 transition path: reconnect pure copy to builtin-driven TIR first.
      // Execution may still temporarily rely on the minimal runtime emitter, but
      // the copy semantics should now exist explicitly in the lowered TIR body.
      RecordDramToDramCopy(op);

      const auto* load = GetCopyLoad(op);
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
      const Array<Integer> global_shape = GetEncodedCurrentBufferShape(op->buffer);
      const Array<Integer> shared_shape = GetEncodedCurrentBufferShape(load->buffer);
      const auto global_rank1 = ResolveStaticRank1ExtentFromBufferOrMetadata(
          op->buffer, global_shape,
          "Blackhole rank-1 staged copy requires static global shape");
      const auto shared_rank1 = ResolveStaticRank1ExtentFromBufferOrMetadata(
          load->buffer, shared_shape,
          "Blackhole rank-1 staged copy requires static shared shape");
      if (global_rank1.has_value() && shared_rank1.has_value()) {
        const std::string segment_kind = ResolveAccessorSegmentKind(direction);
        const bool segmented_gemm = !gemm_a_buffer_name_.empty() && segment_kind == "writer";
        const bool accumulator_like_src =
            GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
            runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
                runtime::StorageRank::kBlackholeAccumulator;
        ExactTiledCBValue live_output;
        const bool has_live_output =
            TryCreateExactOutputLiveTiledCBValue(load->buffer, &live_output) ||
            TryCreateLiveExactTiledCBValue(load->buffer, &live_output);
        if (has_live_output) {
          MarkExactTiledCBValueConsumedByTransport(live_output);
        }
        const int cb_id =
            has_live_output
                ? live_output.cb_id
                : AllocateRequirementIndex(
                      load->buffer,
                      (segmented_gemm && accumulator_like_src) ? CBType::kOutput
                                                               : CBType::kIntermediate);
        const int page_bytes = static_cast<int>(
            std::max<int64_t>(1, shared_rank1.value()) * load->buffer->dtype.bytes());
        if (!has_live_output) {
          SetRequirementPageLayout(cb_id, page_bytes, 1);
        }
        RecordStagedCopyBufferBinding(op, direction);
        const int accessor_slot = GetWriteAccessorSlot(segment_kind, op->buffer, direction);
        PrimExpr logical_index = IntImm32(0);
        if (op->indices.size() == 1U) {
          Analyzer analyzer;
          logical_index = analyzer.Simplify(ScalarizeVectorizedIndex(op->indices[0]));
        }
        PrimExpr page_index = IntImm32(0);
        if (op->indices.size() == 1U && shared_rank1.value() > 0) {
          Analyzer analyzer;
          page_index = analyzer.Simplify(
              FloorDiv(ZeroThreadAndLoopVars(ScalarizeVectorizedIndex(op->indices[0]),
                                             std::vector<Var>{}),
                       IntImm32(static_cast<int>(shared_rank1.value()))));
        }
        PrimExpr l1_offset = IntImm32(0);
        int write_page_bytes = page_bytes;
        if (has_live_output && global_rank1.value() > 1 && live_output.num_tiles == 1 &&
            live_output.num_elements < global_rank1.value()) {
          live_output.num_elements = global_rank1.value();
        }
        const bool live_rank1_vector_output =
            has_live_output && global_rank1.value() > 1 && live_output.num_tiles == 1 &&
            live_output.num_elements == global_rank1.value();
        if (live_rank1_vector_output) {
          Analyzer analyzer;
          constexpr int kFaceRows = 16;
          constexpr int kFaceCols = 16;
          const PrimExpr row = logical_index;
          page_index = analyzer.Simplify(row);
          const PrimExpr tiled_element_offset = analyzer.Simplify(
              FloorDiv(row, IntImm32(kFaceRows)) *
                  IntImm32(kFaceRows * kBlackholeTileCols) +
              FloorMod(row, IntImm32(kFaceRows)) * IntImm32(kFaceCols));
          l1_offset =
              analyzer.Simplify(tiled_element_offset *
                                IntImm32(static_cast<int>(load->buffer->dtype.bytes())));
          write_page_bytes = static_cast<int>(op->buffer->dtype.bytes());
        }
        auto make_wait = [&]() {
          return MakeBlackholeCall(blackhole_cb_wait_front(),
                                   {IntImm32(cb_id), IntImm32(1)});
        };
        auto make_pop = [&]() {
          return MakeBlackholeCall(blackhole_cb_pop_front(),
                                   {IntImm32(cb_id), IntImm32(1)});
        };
        if (live_rank1_vector_output && !active_serial_loop_vars_.empty()) {
          stmts.push_back(tir::IfThenElse(tir::EQ(page_index, IntImm32(0)), make_wait()));
        } else {
          stmts.push_back(make_wait());
        }
        stmts.push_back(MakeBlackholeCall(
            blackhole_write_page_from_cb(), {IntImm32(cb_id), op->buffer->data, page_index,
                                             IntImm32(write_page_bytes), IntImm32(accessor_slot),
                                             l1_offset}));
        if (live_rank1_vector_output && !active_serial_loop_vars_.empty()) {
          std::vector<Stmt> final_sync;
          final_sync.push_back(MakeBlackholeCall(blackhole_noc_async_write_barrier(), {}));
          final_sync.push_back(make_pop());
          stmts.push_back(tir::IfThenElse(
              tir::EQ(page_index, IntImm32(static_cast<int>(global_rank1.value() - 1))),
              SeqStmt::Flatten(final_sync)));
        } else {
          stmts.push_back(MakeBlackholeCall(blackhole_noc_async_write_barrier(), {}));
        }
        RegisterAccessor(segment_kind, op->buffer, accessor_slot, 2, 0, 0, 2,
                         write_page_bytes, {0}, false, "page_indexed");
        if (!(live_rank1_vector_output && !active_serial_loop_vars_.empty())) {
          stmts.push_back(make_pop());
        }
        return WrapSegmentStmtIfNeeded(current_segment_kind_, segment_kind,
                                       SeqStmt::Flatten(stmts));
      }
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

    case CopyDirection::kCBToLocal: {
      ExactTiledCBValue target_live_value;
      if (TryCreateExactOutputLiveTiledCBValue(op->buffer, &target_live_value) ||
          TryCreateLiveExactTiledCBValue(op->buffer, &target_live_value)) {
        return Evaluate(IntImm32(0));
      }
      const int src_cb_id = AllocateRequirementIndex(load->buffer, CBType::kIntermediate);
      auto& req = cb_requirements_.at(src_cb_id);
      req.lifetime_end = std::max(req.lifetime_end, next_requirement_index_);
      ExactTiledCBValue live_value;
      const FutureBufferUses future_uses =
          ClassifyFutureBufferUses(op->buffer, current_lowering_order_index_);
      const FutureBufferUses same_or_future_uses =
          current_lowering_order_index_ >= 0
              ? ClassifyFutureBufferUses(op->buffer, current_lowering_order_index_ - 1)
              : future_uses;
      const FutureBufferUses load_future_uses =
          ClassifyFutureBufferUses(load->buffer, current_lowering_order_index_);
      const FutureBufferUses load_same_or_future_uses =
          current_lowering_order_index_ >= 0
              ? ClassifyFutureBufferUses(load->buffer, current_lowering_order_index_ - 1)
              : load_future_uses;
      const bool has_live_value = TryCreateLiveExactTiledCBValue(load->buffer, &live_value);
      if (!has_live_value) {
        live_value.buffer = load->buffer;
        live_value.cb_id = src_cb_id;
        live_value.borrowed_live = true;
        PopulateExactTiledCBValueShape(load->buffer, &live_value);
      }
      bool has_tile_compute_input_use = false;
      for (const std::string& identity : CollectBufferFlowIdentities(op->buffer)) {
        if (tile_compute_input_buffers_.count(identity) != 0U) {
          has_tile_compute_input_use = true;
          break;
        }
      }
      if (future_uses.has_compute_consume || future_uses.has_transport_consume ||
          same_or_future_uses.has_transport_consume ||
          load_future_uses.has_transport_consume ||
          load_same_or_future_uses.has_transport_consume ||
          has_tile_compute_input_use ||
          IsBroadcastColsSourceBuffer(load->buffer) ||
          IsBroadcastColsSourceBuffer(op->buffer)) {
        RecordTiledCBLiveFormAliases(op->buffer, live_value.cb_id);
        return Evaluate(IntImm32(0));
      }
      return MaterializeExactTiledCBToLocalBuffer(op->buffer, live_value,
                                                  /*pop_front=*/true);
    }

    default:
      return StmtExprMutator::VisitStmt_(op);
  }

  return SeqStmt::Flatten(stmts);
}

Stmt PlanTTKernelABI::GenerateCopySequence(const BufferStoreNode* op,
                                             const PrimExpr& tile_index) {
  CopyDirection direction = GetCopyDirection(op);
  const auto* load = GetCopyLoad(op);
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
      const Buffer cb_producer_buffer = SelectCBProducerBufferForDramToCB(op->buffer);
      int cb_id = AllocateRequirementIndex(
          cb_producer_buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
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
      RecordTiledCBLiveFormAliases(cb_producer_buffer, cb_id);
      if (!SameBufferIdentity(cb_producer_buffer, op->buffer)) {
        RecordTiledCBLiveFormAliases(op->buffer, cb_id);
      }
      return maybe_wrap_segment_stmt(segment_kind, SeqStmt::Flatten(stmts));
    }
    case CopyDirection::kDramToLocal: {
      return GenerateStagedCopyLoopSequence(op, tile_index, std::vector<Var>{});
    }
    case CopyDirection::kCBToDram: {
      const std::string segment_kind = ResolveAccessorSegmentKind(direction);
      const bool segmented_gemm = !gemm_a_buffer_name_.empty() && segment_kind == "writer";
      const bool accumulator_like_src =
          GetStorageScope(load->buffer).rfind("local", 0) == 0 ||
          runtime::StorageScope::Create(GetStorageScope(load->buffer)).rank ==
              runtime::StorageRank::kBlackholeAccumulator;
      ExactTiledCBValue live_output;
      const bool has_live_output =
          TryCreateExactOutputLiveTiledCBValue(load->buffer, &live_output) ||
          TryCreateLiveExactTiledCBValue(load->buffer, &live_output);
      if (has_live_output) {
        MarkExactTiledCBValueConsumedByTransport(live_output);
      }
      int cb_id = has_live_output
                      ? live_output.cb_id
                      : AllocateRequirementIndex(
                            load->buffer,
                            (segmented_gemm && accumulator_like_src) ? CBType::kOutput
                                                                     : CBType::kIntermediate);
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
  const auto* load = GetCopyLoad(op);
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
      IsDramToDeviceCopyDirection(direction) ? op->buffer : load->buffer;
  int64_t shared_rows = 0;
  int64_t shared_cols = 0;
  const auto logical_shared_shape = GetLogicalMatrixShape(shared_buffer);
  const Array<Integer> shared_shape =
      GetEncodedCurrentStagedCopySharedShape(op, loop_vars_to_zero);
  std::tie(shared_rows, shared_cols) = ResolveStagedCopySharedShape(
      shared_buffer, shared_shape, logical_shared_shape, segmented_gemm, transpose_b_reader,
      accumulator_like_src, gemm_m_, gemm_n_, gemm_k_);

  const bool use_page_transport =
      UseStagedCopyPageTransportForShape(shared_rows, shared_cols);
  const Buffer& global_buffer =
      IsDramToDeviceCopyDirection(direction) ? load->buffer : op->buffer;
  const Array<PrimExpr>& global_indices =
      IsDramToDeviceCopyDirection(direction) ? load->indices : op->indices;
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

  if (IsDramToDeviceCopyDirection(direction)) {
    const bool materialize_to_local = direction == CopyDirection::kDramToLocal;
    const Buffer cb_producer_buffer =
        materialize_to_local ? op->buffer : SelectCBProducerBufferForDramToCB(op->buffer);
    int cb_id = AllocateRequirementIndex(
        cb_producer_buffer, segmented_gemm ? CBType::kInput : CBType::kIntermediate);
    RecordStagedCopyBufferBinding(op, direction);
    const int accessor_slot = GetReadAccessorSlot(segment_kind, load->buffer, direction);
    if (use_page_transport) {
      ICHECK(!materialize_to_local)
          << "Blackhole DRAM-to-local materialization currently admits tiled CB pages; "
             "page-indexed stick materialization must lower through an explicit local layout "
             "contract";
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
                         transpose_b_reader, "page_indexed");
      }
      stmts.push_back(MakeBlackholeCall(
          blackhole_noc_async_read_barrier(), {}));
      stmts.push_back(MakeBlackholeCall(
          blackhole_cb_push_back(), {IntImm32(cb_id), IntImm32(1)}));
      RecordTiledCBLiveFormAliases(cb_producer_buffer, cb_id);
      if (!SameBufferIdentity(cb_producer_buffer, op->buffer)) {
        RecordTiledCBLiveFormAliases(op->buffer, cb_id);
      }
      return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
    }
    const int total_subtiles = geometry.subtile_rows * geometry.subtile_cols;
    int tile_emit_count = total_subtiles;
    if (materialize_to_local) {
      ICHECK_EQ(tile_emit_count, total_subtiles);
      SetRequirementPageLayout(cb_id, geometry.tile_bytes, total_subtiles);
      auto& req = cb_requirements_.at(cb_id);
      req.publish_pages_per_event = std::max(req.publish_pages_per_event, total_subtiles);
      req.consume_pages_per_event = std::max(req.consume_pages_per_event, total_subtiles);
    }
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
    if (materialize_to_local) {
      ExactTiledCBValue live_value;
      live_value.buffer = op->buffer;
      live_value.cb_id = cb_id;
      live_value.borrowed_live = true;
      PopulateExactTiledCBValueShape(op->buffer, &live_value);
      live_value.num_tiles = std::max(live_value.num_tiles, total_subtiles);
      live_value.num_elements =
          std::max<int64_t>(live_value.num_elements,
                            static_cast<int64_t>(total_subtiles) *
                                kBlackholeTileRows * kBlackholeTileCols);
      Stmt reader_stmt = maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
      Stmt materialize_stmt =
          MaterializeExactTiledCBToLocalBuffer(op->buffer, live_value,
                                               /*pop_front=*/true);
      materialize_stmt =
          WrapSegmentStmtIfNeeded(current_segment_kind_, "compute", materialize_stmt);
      ClearTiledCBLiveFormAliases(op->buffer);
      InvalidateLastFragmentFillValue(op->buffer);
      Array<Stmt> joined{reader_stmt, materialize_stmt};
      return SeqStmt::Flatten(joined);
    }
    RecordTiledCBLiveFormAliases(cb_producer_buffer, cb_id);
    if (!SameBufferIdentity(cb_producer_buffer, op->buffer)) {
      RecordTiledCBLiveFormAliases(op->buffer, cb_id);
    }
    return maybe_wrap_segment_stmt(SeqStmt::Flatten(stmts));
  }

  if (direction == CopyDirection::kCBToDram) {
    ExactTiledCBValue live_output;
    const bool has_live_output =
        TryCreateExactOutputLiveTiledCBValue(load->buffer, &live_output) ||
        TryCreateLiveExactTiledCBValue(load->buffer, &live_output);
    if (has_live_output) {
      MarkExactTiledCBValueConsumedByTransport(live_output);
    }
    int cb_id = has_live_output
                    ? live_output.cb_id
                    : AllocateRequirementIndex(
                          load->buffer,
                          (segmented_gemm && accumulator_like_src) ? CBType::kOutput
                                                                   : CBType::kIntermediate);
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
                         accessor_slot, 2, 0, 0, 2, geometry.page_bytes, host_axis_order,
                         false, "page_indexed");
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
  const auto* dram_load = GetCopyLoad(dram_to_cb);
  const auto* cb_load = GetCopyLoad(cb_to_dram);
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
  const bool use_page_transport =
      UseStagedCopyPageTransportForShape(shared_rows, shared_cols);
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
                       geometry.page_bytes, {}, false, "page_indexed");
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
                       geometry.page_bytes, {}, false, "page_indexed");
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

}  // namespace tl
}  // namespace tvm
