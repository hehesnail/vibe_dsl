/*!
 * \file build_spatial_plan.cc
 * \brief Build Task 1 SpatialPlan directly from normalized TIR.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/blackhole_utils.h"
#include "common/buffer_tile_bridge_spec_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_access_region.h"
#include "common/spatial_dependence_graph.h"
#include "common/spatial_plan.h"

namespace tvm {
namespace tl {

using tvm::Bool;
using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

struct LocalValueFlow {
  std::string source;
  std::string target;
  bool accepts_distributed_slice{false};
};

struct StatementAccessSummary {
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::unordered_map<std::string, std::string> scope_by_buffer;
  std::vector<LocalValueFlow> local_value_flows;
  bool has_compute_consume{false};
};

struct ClosureCandidateInfo {
  std::string name;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::vector<LocalValueFlow> local_value_flows;
  ExecutionClosure closure;
};

struct MemoryConfigIntentInfo {
  std::string source = "derived_default";
  std::string dsl_origin = "global_buffer_default";
  std::string memory_space_class = "DRAM";
  std::string strategy_class = "interleaved";
  Array<Integer> shard_grid_shape;
  Array<Integer> shard_shape;
  std::string shard_orientation = "row_major";
  bool allow_reshard = true;
  bool hard_requirement = false;
};

void AppendUnique(std::vector<std::string>* values, const std::string& value) {
  ICHECK(values != nullptr);
  if (value.empty()) {
    return;
  }
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

void AppendUniqueValueFlow(std::vector<LocalValueFlow>* flows, LocalValueFlow flow) {
  ICHECK(flows != nullptr);
  if (flow.source.empty() || flow.target.empty() || flow.source == flow.target) {
    return;
  }
  for (LocalValueFlow& existing : *flows) {
    if (existing.source == flow.source && existing.target == flow.target) {
      existing.accepts_distributed_slice =
          existing.accepts_distributed_slice || flow.accepts_distributed_slice;
      return;
    }
  }
  flows->push_back(std::move(flow));
}

Array<Integer> ToIntegerArray(std::initializer_list<int64_t> values) {
  Array<Integer> result;
  for (int64_t value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

Array<Integer> ToIntegerArray(const std::vector<int>& values) {
  Array<Integer> result;
  for (int value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

Array<Integer> ToIntegerArray(const std::vector<int64_t>& values) {
  Array<Integer> result;
  for (int64_t value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

Array<PrimExpr> ToPrimExprArray(const std::vector<int64_t>& values) {
  Array<PrimExpr> result;
  for (int64_t value : values) {
    result.push_back(IntImm(DataType::Int(64), value));
  }
  return result;
}

Array<String> ToStringArraySorted(const std::unordered_set<std::string>& values) {
  std::vector<std::string> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  return ToStringArray(sorted);
}

std::string GetBufferScope(const tir::Buffer& buffer) {
  return buffer.defined() ? buffer.scope() : "";
}

tir::Stmt UnwrapStructuralStmt(tir::Stmt stmt) {
  tir::Stmt current = std::move(stmt);
  while (true) {
    if (const auto* attr = current.as<tir::AttrStmtNode>()) {
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
  return current;
}

std::vector<tir::Stmt> CollectTopLevelExecutableStmts(const tir::PrimFunc& func) {
  std::vector<tir::Stmt> stmts;
  tir::Stmt body = UnwrapStructuralStmt(func->body);
  if (const auto* seq = body.as<tir::SeqStmtNode>()) {
    for (const tir::Stmt& stmt : seq->seq) {
      stmts.push_back(stmt);
    }
  } else if (body.defined()) {
    stmts.push_back(body);
  }
  return stmts;
}

bool IsStorageSyncStmt(const tir::Stmt& stmt) {
  tir::Stmt current = UnwrapStructuralStmt(stmt);
  const auto* evaluate = current.as<tir::EvaluateNode>();
  if (!evaluate) {
    return false;
  }
  const auto* call = evaluate->value.as<tir::CallNode>();
  return call != nullptr && call->op.same_as(tir::builtin::tvm_storage_sync());
}

class ExprBufferReadCollector : public tir::StmtExprVisitor {
 public:
  std::vector<tir::Buffer> Collect(const PrimExpr& expr) {
    buffers_.clear();
    buffer_names_.clear();
    VisitExpr(expr);
    return buffers_;
  }

 private:
  void VisitExpr_(const tir::BufferLoadNode* op) final {
    const std::string name = BufferIdentityName(op->buffer);
    if (!name.empty() && buffer_names_.insert(name).second) {
      buffers_.push_back(op->buffer);
    }
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<tir::Buffer> buffers_;
  std::unordered_set<std::string> buffer_names_;
};

class CastDetector : public tir::StmtExprVisitor {
 public:
  bool ContainsCast(const PrimExpr& expr) {
    contains_cast_ = false;
    VisitExpr(expr);
    return contains_cast_;
  }

 private:
  void VisitExpr_(const tir::CastNode* op) final {
    contains_cast_ = true;
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  bool contains_cast_{false};
};

class StatementAccessAnalyzer : public tir::StmtExprVisitor {
 public:
  StatementAccessSummary Analyze(const tir::Stmt& stmt) {
    summary_ = StatementAccessSummary();
    VisitStmt(stmt);
    return summary_;
  }

 private:
  void RecordRead(const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty()) {
      return;
    }
    AppendUnique(&summary_.reads, name);
    summary_.scope_by_buffer.emplace(name, GetBufferScope(buffer));
  }

  void RecordWrite(const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty()) {
      return;
    }
    AppendUnique(&summary_.writes, name);
    summary_.scope_by_buffer.emplace(name, GetBufferScope(buffer));
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    RecordRead(op->buffer);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    RecordLocalValueFlows(op);
    RecordWrite(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::CallNode* op) final {
    if (op->op.same_as(tir::builtin::tvm_storage_sync())) {
      return;
    }
    if (op->op.same_as(RegionOp::Get())) {
      const RegionOp region(op->args);
      const int access_mask = region->GetAccessMask();
      if ((access_mask & 0x1) != 0) {
        RecordRead(region->GetBuffer());
      }
      if ((access_mask & 0x2) != 0) {
        RecordWrite(region->GetBuffer());
      }
      return;
    }
    TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<tir::Call>(op));
    if (tile_op.defined()) {
      for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
        if (access.kind == DataflowAccessKind::kComputeConsume) {
          summary_.has_compute_consume = true;
          RecordRead(access.buffer);
        } else if (access.kind == DataflowAccessKind::kComputeProduce) {
          RecordWrite(access.buffer);
        }
      }
    }
    RecordBlackholeTileComputeValueFlows(op);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void RecordBlackholeTileComputeValueFlows(const tir::CallNode* op) {
    if (!op || !op->op->IsInstance<OpNode>()) {
      return;
    }
    const Op call_op = Downcast<Op>(op->op);
    if (call_op->name != blackhole_tile_compute_schema::kOpName ||
        op->args.size() < 3U) {
      return;
    }
    const auto* operation = op->args[0].as<StringImmNode>();
    if (!operation) {
      return;
    }
    const std::string op_name = operation->value;
    const bool is_value_transfer =
        op_name == blackhole_tile_compute_schema::kCopyTile ||
        op_name == blackhole_tile_compute_schema::kTypecastTile;
    if (!is_value_transfer || !IsBufferLikeExpr(op->args[1]) ||
        !IsBufferLikeExpr(op->args[2])) {
      return;
    }
    const tir::Buffer src = NormalizeToBufferRegion(op->args[1])->buffer;
    const tir::Buffer dst = NormalizeToBufferRegion(op->args[2])->buffer;
    const std::string source_name = BufferIdentityName(src);
    const std::string target_name = BufferIdentityName(dst);
    const std::string source_scope = GetBufferScope(src);
    const std::string target_scope = GetBufferScope(dst);
    if (source_name.empty() || target_name.empty() || source_name == target_name ||
        source_scope.empty() || target_scope.empty() || source_scope == "global" ||
        target_scope == "global") {
      return;
    }
    AppendUniqueValueFlow(
        &summary_.local_value_flows,
        LocalValueFlow{source_name, target_name,
                       op_name == blackhole_tile_compute_schema::kTypecastTile});
  }

  void RecordLocalValueFlows(const tir::BufferStoreNode* op) {
    const std::string target_name = BufferIdentityName(op->buffer);
    const std::string target_scope = GetBufferScope(op->buffer);
    if (target_name.empty() || target_scope.empty() || target_scope == "global") {
      return;
    }

    ExprBufferReadCollector read_collector;
    CastDetector cast_detector;
    const bool accepts_distributed_slice = cast_detector.ContainsCast(op->value);
    for (const tir::Buffer& source : read_collector.Collect(op->value)) {
      const std::string source_name = BufferIdentityName(source);
      const std::string source_scope = GetBufferScope(source);
      if (source_name.empty() || source_name == target_name || source_scope.empty() ||
          source_scope == "global") {
        continue;
      }
      AppendUniqueValueFlow(&summary_.local_value_flows,
                            LocalValueFlow{source_name, target_name, accepts_distributed_slice});
    }
  }

  StatementAccessSummary summary_;
};

bool HasGlobalRead(const StatementAccessSummary& summary) {
  for (const std::string& read : summary.reads) {
    auto it = summary.scope_by_buffer.find(read);
    if (it != summary.scope_by_buffer.end() && it->second == "global") {
      return true;
    }
  }
  return false;
}

bool HasGlobalWrite(const StatementAccessSummary& summary) {
  for (const std::string& write : summary.writes) {
    auto it = summary.scope_by_buffer.find(write);
    if (it != summary.scope_by_buffer.end() && it->second == "global") {
      return true;
    }
  }
  return false;
}

bool HasLocalWrite(const StatementAccessSummary& summary) {
  for (const std::string& write : summary.writes) {
    auto it = summary.scope_by_buffer.find(write);
    if (it != summary.scope_by_buffer.end() && it->second != "global") {
      return true;
    }
  }
  return false;
}

std::string DeriveExecutionRole(const StatementAccessSummary& summary) {
  if (summary.has_compute_consume) {
    return "compute";
  }
  const bool reads_global = HasGlobalRead(summary);
  const bool writes_global = HasGlobalWrite(summary);
  const bool writes_local = HasLocalWrite(summary);
  if (reads_global && writes_local && !writes_global) {
    return "ingress";
  }
  if (writes_global) {
    return "egress";
  }
  if (writes_local) {
    return "dataflow";
  }
  return "dataflow";
}

std::vector<std::string> DeriveClosureTraits(const StatementAccessSummary& summary) {
  std::vector<std::string> traits;
  for (const std::string& write : summary.writes) {
    if (std::find(summary.reads.begin(), summary.reads.end(), write) != summary.reads.end()) {
      AppendUnique(&traits, "carry_obligation");
    }
  }
  if (summary.has_compute_consume) {
    AppendUnique(&traits, "locality_obligation");
  }
  return traits;
}

ExecutionClosure BuildExecutionClosure(int stmt_index, const StatementAccessSummary& summary) {
  const std::string name = "closure_" + std::to_string(stmt_index);
  return ExecutionClosure(
      String(name), String("normalized_tir_top_level_stmt"), String(DeriveExecutionRole(summary)),
      ToIntegerArray({stmt_index}), ToStringArray(summary.reads), ToStringArray(summary.writes),
      MakeTraits({"statement_boundary"}), ToStringArray(DeriveClosureTraits(summary)),
      MakeAnchors("execution_closure", name));
}

ValidatedHintSet BuildEmptyValidatedHintSet(const std::string& member_func) {
  return ValidatedHintSet(Array<String>{}, Array<String>{}, Map<String, Any>{},
                          MakeAnchors("validated_hint_set", member_func));
}

std::vector<ClosureCandidateInfo> AnalyzeClosureCandidates(const tir::PrimFunc& func) {
  std::vector<ClosureCandidateInfo> candidates;
  StatementAccessAnalyzer analyzer;
  const std::vector<tir::Stmt> top_level_stmts = CollectTopLevelExecutableStmts(func);
  for (int stmt_index = 0; stmt_index < static_cast<int>(top_level_stmts.size()); ++stmt_index) {
    const tir::Stmt& stmt = top_level_stmts[stmt_index];
    if (IsStorageSyncStmt(stmt)) {
      continue;
    }
    StatementAccessSummary summary = analyzer.Analyze(stmt);
    if (summary.reads.empty() && summary.writes.empty() && !summary.has_compute_consume) {
      continue;
    }
    ClosureCandidateInfo info;
    info.closure = BuildExecutionClosure(stmt_index, summary);
    info.name = str(info.closure->name);
    info.reads = std::move(summary.reads);
    info.writes = std::move(summary.writes);
    info.local_value_flows = std::move(summary.local_value_flows);
    candidates.push_back(std::move(info));
  }
  return candidates;
}

Array<String> ExecutionUnitNamesForIndices(const std::vector<int>& unit_indices,
                                           const Array<ExecutionUnit>& execution_units) {
  Array<String> unit_names;
  for (int unit_index : unit_indices) {
    unit_names.push_back(execution_units[unit_index]->name);
  }
  return unit_names;
}

Array<String> DataflowEdgeNamesForIndices(const std::vector<int>& edge_indices,
                                          const Array<DataflowEdge>& dataflow_edges) {
  Array<String> edge_names;
  for (int edge_index : edge_indices) {
    edge_names.push_back(dataflow_edges[edge_index]->name);
  }
  return edge_names;
}

std::vector<int> ComputeExecutionUnitPhases(const Array<ClosureBoundary>& boundaries,
                                            int unit_count) {
  std::vector<std::vector<int>> preds(unit_count);
  for (const ClosureBoundary& boundary : boundaries) {
    if (boundary->source_closure_index < 0 || boundary->target_closure_index < 0 ||
        boundary->source_closure_index >= unit_count ||
        boundary->target_closure_index >= unit_count ||
        boundary->source_closure_index == boundary->target_closure_index) {
      continue;
    }
    preds[boundary->target_closure_index].push_back(boundary->source_closure_index);
  }
  std::vector<int> phases(unit_count, 0);
  for (int unit_index = 0; unit_index < unit_count; ++unit_index) {
    for (int pred_index : preds[unit_index]) {
      phases[unit_index] = std::max(phases[unit_index], phases[pred_index] + 1);
    }
  }
  return phases;
}

Array<ExecutionUnit> BuildExecutionUnits(const Array<ExecutionClosure>& closures) {
  Array<ExecutionUnit> execution_units;
  for (const ExecutionClosure& closure : closures) {
    execution_units.push_back(
        ExecutionUnit(closure->name, closure->closure_basis, closure->execution_role,
                      closure->stmt_indices, closure->read_buffers, closure->write_buffers,
                      closure->traits, MakeAnchors("execution_unit", str(closure->name))));
  }
  return execution_units;
}

Array<DataflowEdge> BuildDataflowEdges(const Array<ClosureBoundary>& boundaries,
                                       const std::vector<int>& unit_phases) {
  Array<DataflowEdge> dataflow_edges;
  for (const ClosureBoundary& boundary : boundaries) {
    bool crosses_phase = false;
    if (boundary->source_closure_index >= 0 && boundary->target_closure_index >= 0 &&
        boundary->source_closure_index < static_cast<int64_t>(unit_phases.size()) &&
        boundary->target_closure_index < static_cast<int64_t>(unit_phases.size()) &&
        boundary->source_closure_index != boundary->target_closure_index) {
      crosses_phase = unit_phases[boundary->source_closure_index] !=
                      unit_phases[boundary->target_closure_index];
    }
    dataflow_edges.push_back(DataflowEdge(
        boundary->name, boundary->kind, boundary->source_closure, boundary->target_closure,
        boundary->source_closure_index, boundary->target_closure_index, boundary->subject,
        crosses_phase, boundary->traits, MakeAnchors("dataflow_edge", str(boundary->name))));
  }
  return dataflow_edges;
}

std::vector<SpatialLocalValueFlowEvidence> BuildLocalValueFlowEvidence(
    const std::vector<ClosureCandidateInfo>& candidates) {
  std::vector<SpatialLocalValueFlowEvidence> result;
  for (int unit_index = 0; unit_index < static_cast<int>(candidates.size()); ++unit_index) {
    const ClosureCandidateInfo& candidate = candidates[unit_index];
    for (const LocalValueFlow& flow : candidate.local_value_flows) {
      result.push_back(SpatialLocalValueFlowEvidence{flow.source, flow.target, unit_index,
                                                     flow.accepts_distributed_slice});
    }
  }
  return result;
}

Array<DataflowEdge> ConcatDataflowEdges(const Array<DataflowEdge>& lhs,
                                        const Array<DataflowEdge>& rhs) {
  Array<DataflowEdge> result;
  result.reserve(lhs.size() + rhs.size());
  for (const DataflowEdge& edge : lhs) {
    result.push_back(edge);
  }
  for (const DataflowEdge& edge : rhs) {
    result.push_back(edge);
  }
  return result;
}

class BufferScopeCollector : public tir::StmtExprVisitor {
 public:
  std::unordered_map<std::string, std::string> Collect(const tir::PrimFunc& func) {
    scope_by_buffer_.clear();
    for (const auto& [_, buffer] : func->buffer_map) {
      Record(buffer);
    }
    VisitStmt(func->body);
    return scope_by_buffer_;
  }

 private:
  void Record(const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty() || scope_by_buffer_.count(name)) {
      return;
    }
    scope_by_buffer_.emplace(name, std::string(buffer.scope()));
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      Record(buffer);
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<std::string, std::string> scope_by_buffer_;
};

struct BufferMetadata {
  std::string scope;
  std::vector<int64_t> shape;
  std::string dtype;
};

struct LogicalTileLayoutInfo {
  Array<PrimExpr> logical_shape;
  Array<PrimExpr> local_shape;
  PrimExpr thread_extent;
  PrimExpr replicate_extent;
  Array<PrimExpr> inverse_logical_index_vars;
  Array<PrimExpr> inverse_logical_index_exprs;
};

class LogicalTileLayoutCollector : public tir::StmtExprVisitor {
 public:
  std::unordered_map<std::string, LogicalTileLayoutInfo> Collect(const tir::PrimFunc& func) {
    layout_by_buffer_.clear();
    VisitStmt(func->body);
    return layout_by_buffer_;
  }

 private:
  void Record(const tir::Buffer& buffer, const Layout& layout) {
    const std::string scope = buffer.scope();
    if (scope != "local" && scope != "local.fragment" && scope != "blackhole.acc") {
      return;
    }
    auto maybe_spec = TryBuildBufferTileBridgeSpec(buffer, layout);
    if (!maybe_spec) {
      return;
    }
    const Map<String, Any>& spec = maybe_spec.value();
    auto buffer_it = spec.find(String(schema_key::kBuffer));
    if (buffer_it == spec.end()) {
      return;
    }
    const std::string buffer_name = Downcast<String>((*buffer_it).second);
    if (buffer_name.empty() || layout_by_buffer_.count(buffer_name)) {
      return;
    }
    LogicalTileLayoutInfo info;
    if (auto value = spec.Get(String(schema_key::kShape))) {
      info.logical_shape = Downcast<Array<PrimExpr>>(value.value());
    }
    if (auto value = spec.Get(String(schema_key::kLocalShape))) {
      info.local_shape = Downcast<Array<PrimExpr>>(value.value());
    }
    if (auto value = spec.Get(String(schema_key::kThreadExtent))) {
      info.thread_extent = Downcast<PrimExpr>(value.value());
    }
    if (auto value = spec.Get(String(schema_key::kReplicateExtent))) {
      info.replicate_extent = Downcast<PrimExpr>(value.value());
    }
    if (auto value = spec.Get(String(schema_key::kInverseLogicalIndexVars))) {
      info.inverse_logical_index_vars = Downcast<Array<PrimExpr>>(value.value());
    }
    if (auto value = spec.Get(String(schema_key::kInverseLogicalIndexExprs))) {
      info.inverse_logical_index_exprs = Downcast<Array<PrimExpr>>(value.value());
    }
    layout_by_buffer_.emplace(buffer_name, std::move(info));
  }

  void RecordExplicitReduceOutput(const tir::CallNode* op) {
    if (!op || !op->op->IsInstance<OpNode>()) {
      return;
    }
    const Op call_op = Downcast<Op>(op->op);
    if (call_op->name != "tl.tileop.reduce" || op->args.size() < 4U ||
        !IsBufferLikeExpr(op->args[0]) || !IsBufferLikeExpr(op->args[1])) {
      return;
    }
    const tir::BufferRegion src_region = NormalizeToBufferRegion(op->args[0]);
    const tir::BufferRegion dst_region = NormalizeToBufferRegion(op->args[1]);
    const std::string src_name = BufferIdentityName(src_region->buffer);
    const std::string dst_name = BufferIdentityName(dst_region->buffer);
    if (src_name.empty() || dst_name.empty() || layout_by_buffer_.count(dst_name)) {
      return;
    }
    auto source_layout_it = layout_by_buffer_.find(src_name);
    if (source_layout_it == layout_by_buffer_.end() ||
        source_layout_it->second.logical_shape.empty()) {
      return;
    }

    LogicalTileLayoutInfo info;
    for (const Range& range : dst_region->region) {
      info.logical_shape.push_back(range->extent);
    }
    if (info.logical_shape.empty()) {
      return;
    }
    info.local_shape.push_back(IntImm(DataType::Int(32), 1));
    info.thread_extent = source_layout_it->second.thread_extent;
    info.replicate_extent = source_layout_it->second.replicate_extent;
    info.inverse_logical_index_vars =
        source_layout_it->second.inverse_logical_index_vars;

    int64_t reduced_dim = -1;
    if (const auto* dim_imm = op->args[3].as<IntImmNode>()) {
      reduced_dim = dim_imm->value;
    }
    if (reduced_dim < 0) {
      reduced_dim += static_cast<int64_t>(
          source_layout_it->second.logical_shape.size());
    }
    const Array<PrimExpr>& source_inverse_exprs =
        source_layout_it->second.inverse_logical_index_exprs;
    for (int64_t i = 0; i < static_cast<int64_t>(source_inverse_exprs.size()); ++i) {
      if (i == reduced_dim) {
        continue;
      }
      info.inverse_logical_index_exprs.push_back(source_inverse_exprs[i]);
    }
    layout_by_buffer_.emplace(dst_name, std::move(info));
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    if (op->annotations.count(attr::kLayoutMap)) {
      if (auto layout_map_any = op->annotations.Get(attr::kLayoutMap)) {
        auto layout_map = layout_map_any->as<Map<tir::Buffer, Layout>>();
        if (layout_map && layout_map.value().defined()) {
          for (const auto& [buffer, layout] : layout_map.value()) {
            Record(buffer, layout);
          }
        }
      }
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::CallNode* op) final {
    RecordExplicitReduceOutput(op);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<std::string, LogicalTileLayoutInfo> layout_by_buffer_;
};

std::optional<std::vector<int64_t>> ExtractStaticShape(const Array<PrimExpr>& shape) {
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (const PrimExpr& dim : shape) {
    const auto* imm = dim.as<IntImmNode>();
    if (imm == nullptr) {
      return std::nullopt;
    }
    result.push_back(imm->value);
  }
  return result;
}

class BufferMetadataCollector : public tir::StmtExprVisitor {
 public:
  std::unordered_map<std::string, BufferMetadata> Collect(const tir::PrimFunc& func) {
    metadata_by_buffer_.clear();
    for (const auto& [_, buffer] : func->buffer_map) {
      Record(buffer);
    }
    VisitStmt(func->body);
    return metadata_by_buffer_;
  }

 private:
  void Record(const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty() || metadata_by_buffer_.count(name)) {
      return;
    }
    BufferMetadata metadata;
    metadata.scope = GetBufferScope(buffer);
    metadata.dtype = tvm::runtime::DLDataTypeToString(buffer->dtype);
    if (auto shape = ExtractStaticShape(buffer->shape)) {
      metadata.shape = std::move(shape.value());
    }
    metadata_by_buffer_.emplace(name, std::move(metadata));
  }

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    Record(op->buffer);
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      Record(buffer);
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<std::string, BufferMetadata> metadata_by_buffer_;
};

constexpr const char* kMemoryConfigMapAttr = "tl.memory_config_map";

String GetMapString(const Map<String, Any>& item, const char* key,
                    const char* default_value = "") {
  if (auto value = item.Get(key)) {
    return Downcast<String>(value.value());
  }
  return String(default_value);
}

bool GetMapBool(const Map<String, Any>& item, const char* key, bool default_value) {
  if (auto value = item.Get(key)) {
    return Downcast<Bool>(value.value());
  }
  return default_value;
}

Array<Integer> GetMapIntegerArray(const Map<String, Any>& item, const char* key) {
  Array<Integer> values;
  if (auto value = item.Get(key)) {
    for (const Any& element : Downcast<Array<Any>>(value.value())) {
      values.push_back(Downcast<Integer>(element));
    }
  }
  return values;
}

Map<String, Any> GetMapField(const Map<String, Any>& item, const char* key) {
  if (auto value = item.Get(key)) {
    return value.value().as<Map<String, Any>>().value_or(Map<String, Any>());
  }
  return Map<String, Any>();
}

std::string MemorySpaceClassFromBufferType(const std::string& buffer_type) {
  if (buffer_type == "dram") {
    return "DRAM";
  }
  if (buffer_type == "l1") {
    return "L1";
  }
  return buffer_type;
}

Array<Integer> PartitionedDimsForStrategy(const std::string& strategy_class,
                                          int64_t logical_rank) {
  Array<Integer> dims;
  if (strategy_class == "height_sharded" && logical_rank >= 1) {
    dims.push_back(Integer(0));
  } else if (strategy_class == "width_sharded" && logical_rank >= 2) {
    dims.push_back(Integer(1));
  } else if (strategy_class == "block_sharded") {
    for (int64_t i = 0; i < std::min<int64_t>(logical_rank, 2); ++i) {
      dims.push_back(Integer(i));
    }
  } else if (strategy_class == "nd_sharded") {
    for (int64_t i = 0; i < logical_rank; ++i) {
      dims.push_back(Integer(i));
    }
  }
  return dims;
}

Array<Integer> ReplicatedDimsForStrategy(const std::string& strategy_class,
                                         int64_t logical_rank) {
  const Array<Integer> partitioned = PartitionedDimsForStrategy(strategy_class, logical_rank);
  std::unordered_set<int64_t> partitioned_set;
  for (const Integer& dim : partitioned) {
    partitioned_set.insert(dim.IntValue());
  }
  Array<Integer> dims;
  for (int64_t i = 0; i < logical_rank; ++i) {
    if (partitioned_set.count(i) == 0U) {
      dims.push_back(Integer(i));
    }
  }
  return dims;
}

class MemoryConfigAnnotationCollector : public tir::StmtExprVisitor {
 public:
  std::unordered_map<std::string, MemoryConfigIntentInfo> Collect(const tir::PrimFunc& func) {
    intents_by_buffer_.clear();
    if (auto config_map = func->GetAttr<Map<tir::Var, Any>>(kMemoryConfigMapAttr)) {
      RecordMemoryConfigMap(config_map.value(), /*allow_existing=*/false);
    }
    VisitStmt(func->body);
    return intents_by_buffer_;
  }

 private:
  static MemoryConfigIntentInfo DecodeMemoryConfig(const Map<String, Any>& config) {
    MemoryConfigIntentInfo info;
    const std::string memory_layout = str(GetMapString(config, "memory_layout", "interleaved"));
    const std::string buffer_type = str(GetMapString(config, "buffer_type", "dram"));
    info.source = "user";
    info.dsl_origin = "memory_config_map";
    info.memory_space_class = MemorySpaceClassFromBufferType(buffer_type);
    info.strategy_class = memory_layout;
    info.allow_reshard = GetMapBool(config, "allow_reshard", true);
    info.hard_requirement = !info.allow_reshard;
    const Map<String, Any> shard = GetMapField(config, "shard");
    if (shard.defined()) {
      info.shard_grid_shape = GetMapIntegerArray(shard, "grid_shape");
      info.shard_shape = GetMapIntegerArray(shard, "shape");
      info.shard_orientation = str(GetMapString(shard, "orientation", "row_major"));
    }
    return info;
  }

  void RecordMemoryConfigMap(const Map<tir::Var, Any>& config_map, bool allow_existing) {
    for (const auto& [buffer_var, config_any] : config_map) {
      const std::string buffer_name = buffer_var->name_hint;
      ICHECK(!buffer_name.empty())
          << "tl.memory_config_map key requires a named buffer data var";
      if (allow_existing && intents_by_buffer_.count(buffer_name) != 0U) {
        continue;
      }
      ICHECK(intents_by_buffer_.count(buffer_name) == 0U)
          << "duplicate tl.memory_config_map entry for buffer " << buffer_name;
      const Map<String, Any> config =
          config_any.as<Map<String, Any>>().value_or(Map<String, Any>());
      ICHECK(config.defined())
          << "tl.memory_config_map entry for " << buffer_name
          << " must be a MemoryConfig attr map";
      intents_by_buffer_.emplace(buffer_name, DecodeMemoryConfig(config));
    }
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    if (op->annotations.count(kMemoryConfigMapAttr)) {
      if (auto config_map_any = op->annotations.Get(kMemoryConfigMapAttr)) {
        auto config_map = config_map_any->as<Map<tir::Var, Any>>();
        ICHECK(config_map && config_map.value().defined())
            << "tl.memory_config_map must map buffer data vars to MemoryConfig attrs";
        RecordMemoryConfigMap(config_map.value(), /*allow_existing=*/true);
      }
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<std::string, MemoryConfigIntentInfo> intents_by_buffer_;
};

Array<PrimExpr> LogicalShapeForIntent(
    const LayoutSpec& layout,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  if (!layout->logical_shape.empty()) {
    return layout->logical_shape;
  }
  auto metadata_it = metadata_by_buffer.find(str(layout->subject));
  if (metadata_it != metadata_by_buffer.end()) {
    return ToPrimExprArray(metadata_it->second.shape);
  }
  return Array<PrimExpr>();
}

TensorPlacementIntent MakeTensorPlacementIntent(
    const LayoutSpec& layout, MemoryConfigIntentInfo info,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  const Array<PrimExpr> logical_shape = LogicalShapeForIntent(layout, metadata_by_buffer);
  const int64_t logical_rank = static_cast<int64_t>(logical_shape.size());
  const std::string subject = str(layout->subject);
  return TensorPlacementIntent(
      String("tensor_placement_" + subject), layout->subject, String(""),
      String(info.source), String(info.dsl_origin),
      info.source == "user" ? String("memory_config_" + subject) : String(""),
      logical_rank, logical_shape,
      PartitionedDimsForStrategy(info.strategy_class, logical_rank),
      ReplicatedDimsForStrategy(info.strategy_class, logical_rank),
      Array<String>{}, String(info.memory_space_class), String(info.strategy_class),
      info.shard_grid_shape, info.shard_shape, String(info.shard_orientation),
      info.allow_reshard, info.hard_requirement, MakeAnchors("tensor_placement", subject));
}

Array<TensorPlacementIntent> BuildTensorPlacementIntents(
    const tir::PrimFunc& func, const Array<LayoutSpec>& layout_specs,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  MemoryConfigAnnotationCollector collector;
  std::unordered_map<std::string, MemoryConfigIntentInfo> user_intents = collector.Collect(func);
  Array<TensorPlacementIntent> intents;
  std::unordered_set<std::string> emitted;
  for (const LayoutSpec& layout : layout_specs) {
    const std::string subject = str(layout->subject);
    auto user_it = user_intents.find(subject);
    if (user_it != user_intents.end()) {
      intents.push_back(MakeTensorPlacementIntent(layout, user_it->second, metadata_by_buffer));
      emitted.insert(subject);
      continue;
    }
    auto metadata_it = metadata_by_buffer.find(subject);
    if (metadata_it != metadata_by_buffer.end() && metadata_it->second.scope == "global") {
      intents.push_back(
          MakeTensorPlacementIntent(layout, MemoryConfigIntentInfo(), metadata_by_buffer));
      emitted.insert(subject);
    }
  }
  for (const auto& [subject, _] : user_intents) {
    ICHECK(emitted.count(subject) != 0U)
        << "tl.memory_config_map references unknown buffer " << subject;
  }
  return intents;
}

std::string DeriveDistributionKind(const std::string& scope) {
  if (scope == "global") {
    return "global_visible";
  }
  if (scope.rfind("shared", 0) == 0) {
    return "shared_visible";
  }
  if (scope.empty()) {
    return "logical_only";
  }
  return "local_visible";
}

Array<LayoutSpec> BuildLayoutSpecs(const tir::PrimFunc& func,
                                   const Array<ExecutionUnit>& execution_units) {
  struct LayoutInfo {
    std::string scope;
    std::vector<int> unit_indices;
    std::unordered_set<std::string> unit_names;
  };

  BufferScopeCollector collector;
  std::unordered_map<std::string, std::string> scope_by_buffer = collector.Collect(func);
  LogicalTileLayoutCollector tile_layout_collector;
  std::unordered_map<std::string, LogicalTileLayoutInfo> tile_layout_by_buffer =
      tile_layout_collector.Collect(func);
  std::unordered_map<std::string, LayoutInfo> layout_info_by_subject;

  for (int unit_index = 0; unit_index < static_cast<int>(execution_units.size()); ++unit_index) {
    const ExecutionUnit& unit = execution_units[unit_index];
    auto record_subject = [&](const String& subject) {
      const std::string key = str(subject);
      if (key.empty()) {
        return;
      }
      LayoutInfo& info = layout_info_by_subject[key];
      if (info.scope.empty()) {
        auto scope_it = scope_by_buffer.find(key);
        if (scope_it != scope_by_buffer.end()) {
          info.scope = scope_it->second;
        }
      }
      if (std::find(info.unit_indices.begin(), info.unit_indices.end(), unit_index) ==
          info.unit_indices.end()) {
        info.unit_indices.push_back(unit_index);
      }
      info.unit_names.insert(str(unit->name));
    };
    for (const String& subject : unit->read_buffers) {
      record_subject(subject);
    }
    for (const String& subject : unit->write_buffers) {
      record_subject(subject);
    }
  }

  std::vector<std::string> subjects;
  subjects.reserve(layout_info_by_subject.size());
  for (const auto& [subject, _] : layout_info_by_subject) {
    subjects.push_back(subject);
  }
  std::sort(subjects.begin(), subjects.end());

  Array<LayoutSpec> layout_specs;
  for (const std::string& subject : subjects) {
    LayoutInfo& info = layout_info_by_subject[subject];
    std::sort(info.unit_indices.begin(), info.unit_indices.end());
    auto tile_layout_it = tile_layout_by_buffer.find(subject);
    LogicalTileLayoutInfo tile_layout =
        tile_layout_it == tile_layout_by_buffer.end() ? LogicalTileLayoutInfo()
                                                      : tile_layout_it->second;
    layout_specs.push_back(LayoutSpec(
        String("layout_" + subject), String(subject), String(info.scope),
        String(DeriveDistributionKind(info.scope)),
        ExecutionUnitNamesForIndices(info.unit_indices, execution_units),
        ToIntegerArray(info.unit_indices), Array<String>{}, tile_layout.logical_shape,
        tile_layout.local_shape, tile_layout.thread_extent, tile_layout.replicate_extent,
        tile_layout.inverse_logical_index_vars, tile_layout.inverse_logical_index_exprs,
        MakeAnchors("layout_spec", subject)));
  }
  return layout_specs;
}

LogicalTileLayoutInfo LogicalTileLayoutInfoFromLayoutSpec(const LayoutSpec& layout_spec) {
  LogicalTileLayoutInfo info;
  info.logical_shape = layout_spec->logical_shape;
  info.local_shape = layout_spec->local_shape;
  info.thread_extent = layout_spec->thread_extent;
  info.replicate_extent = layout_spec->replicate_extent;
  info.inverse_logical_index_vars = layout_spec->inverse_logical_index_vars;
  info.inverse_logical_index_exprs = layout_spec->inverse_logical_index_exprs;
  return info;
}

Array<LayoutSpec> MergePriorTypedLayoutSpecs(const tir::PrimFunc& func,
                                             const Array<LayoutSpec>& layout_specs) {
  auto prior_plan = func->GetAttr<SpatialPlan>(attr::kTLSpatialPlan);
  if (!prior_plan) {
    return layout_specs;
  }
  std::unordered_map<std::string, LayoutSpec> prior_layout_by_subject;
  for (const LayoutSpec& prior_layout : prior_plan.value()->layout_specs) {
    if (!prior_layout->logical_shape.empty()) {
      prior_layout_by_subject.emplace(str(prior_layout->subject), prior_layout);
    }
  }
  if (prior_layout_by_subject.empty()) {
    return layout_specs;
  }

  Array<LayoutSpec> merged_layout_specs;
  for (const LayoutSpec& layout : layout_specs) {
    LogicalTileLayoutInfo tile_layout = LogicalTileLayoutInfoFromLayoutSpec(layout);
    if (tile_layout.logical_shape.empty()) {
      auto prior_it = prior_layout_by_subject.find(str(layout->subject));
      if (prior_it != prior_layout_by_subject.end()) {
        tile_layout = LogicalTileLayoutInfoFromLayoutSpec(prior_it->second);
      }
    }
    merged_layout_specs.push_back(LayoutSpec(
        layout->name, layout->subject, layout->scope, layout->distribution_kind,
        layout->unit_names, layout->unit_indices, layout->virtual_device_axes,
        tile_layout.logical_shape, tile_layout.local_shape, tile_layout.thread_extent,
        tile_layout.replicate_extent, tile_layout.inverse_logical_index_vars,
        tile_layout.inverse_logical_index_exprs, layout->anchors));
  }
  return merged_layout_specs;
}

Array<LayoutSpec> PropagateLocalValueFlowLayoutSpecs(
    const Array<LayoutSpec>& layout_specs,
    const std::vector<ClosureCandidateInfo>& candidates) {
  std::unordered_map<std::string, LogicalTileLayoutInfo> layout_by_subject;
  for (const LayoutSpec& layout : layout_specs) {
    layout_by_subject.emplace(str(layout->subject), LogicalTileLayoutInfoFromLayoutSpec(layout));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const ClosureCandidateInfo& candidate : candidates) {
      for (const LocalValueFlow& flow : candidate.local_value_flows) {
        auto source_it = layout_by_subject.find(flow.source);
        auto target_it = layout_by_subject.find(flow.target);
        if (source_it == layout_by_subject.end() || target_it == layout_by_subject.end()) {
          continue;
        }
        if (source_it->second.logical_shape.empty() ||
            !target_it->second.logical_shape.empty()) {
          continue;
        }
        target_it->second = source_it->second;
        changed = true;
      }
    }
  }

  Array<LayoutSpec> propagated;
  for (const LayoutSpec& layout : layout_specs) {
    LogicalTileLayoutInfo tile_layout = LogicalTileLayoutInfoFromLayoutSpec(layout);
    auto propagated_it = layout_by_subject.find(str(layout->subject));
    if (tile_layout.logical_shape.empty() && propagated_it != layout_by_subject.end()) {
      tile_layout = propagated_it->second;
    }
    propagated.push_back(LayoutSpec(
        layout->name, layout->subject, layout->scope, layout->distribution_kind,
        layout->unit_names, layout->unit_indices, layout->virtual_device_axes,
        tile_layout.logical_shape, tile_layout.local_shape, tile_layout.thread_extent,
        tile_layout.replicate_extent, tile_layout.inverse_logical_index_vars,
        tile_layout.inverse_logical_index_exprs, layout->anchors));
  }
  return propagated;
}

std::string DeriveAccessRegionValueKind(const std::string& scope) {
  if (scope == "global") {
    return "tensor";
  }
  if (scope == "blackhole.acc") {
    return "accumulator";
  }
  if (scope == "local" || scope == "local.fragment") {
    return "fragment";
  }
  if (scope.rfind("shared", 0) == 0 || scope.rfind("blackhole.cb", 0) == 0) {
    return "tile";
  }
  return "tensor";
}

Array<PrimExpr> MakeZeroBounds(size_t rank) {
  Array<PrimExpr> bounds;
  for (size_t i = 0; i < rank; ++i) {
    bounds.push_back(IntImm(DataType::Int(64), 0));
  }
  return bounds;
}

Array<PrimExpr> MakeUnitStrides(size_t rank) {
  Array<PrimExpr> strides;
  for (size_t i = 0; i < rank; ++i) {
    strides.push_back(IntImm(DataType::Int(64), 1));
  }
  return strides;
}

Array<AccessRegion> BuildAccessRegions(
    const Array<ExecutionUnit>& execution_units, const Array<LayoutSpec>& layout_specs,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  std::unordered_map<std::string, LayoutSpec> layout_by_subject;
  for (const LayoutSpec& layout : layout_specs) {
    layout_by_subject.emplace(str(layout->subject), layout);
  }

  auto extents_for_subject = [&](const std::string& subject) -> Array<PrimExpr> {
    auto layout_it = layout_by_subject.find(subject);
    if (layout_it != layout_by_subject.end() && !layout_it->second->logical_shape.empty()) {
      return layout_it->second->logical_shape;
    }
    auto metadata_it = metadata_by_buffer.find(subject);
    if (metadata_it != metadata_by_buffer.end() && !metadata_it->second.shape.empty()) {
      return ToPrimExprArray(metadata_it->second.shape);
    }
    return Array<PrimExpr>{};
  };

  auto value_kind_for_subject = [&](const std::string& subject) -> std::string {
    auto layout_it = layout_by_subject.find(subject);
    if (layout_it != layout_by_subject.end()) {
      return DeriveAccessRegionValueKind(str(layout_it->second->scope));
    }
    auto metadata_it = metadata_by_buffer.find(subject);
    return metadata_it == metadata_by_buffer.end()
               ? "tensor"
               : DeriveAccessRegionValueKind(metadata_it->second.scope);
  };

  Array<AccessRegion> regions;
  for (int unit_index = 0; unit_index < execution_units.size(); ++unit_index) {
    const ExecutionUnit& unit = execution_units[unit_index];
    auto emit_region = [&](const String& subject, const char* access_kind) {
      const std::string subject_name = str(subject);
      if (subject_name.empty()) {
        return;
      }
      const Array<PrimExpr> extents = extents_for_subject(subject_name);
      const int64_t rank = static_cast<int64_t>(extents.size());
      const std::string name = "access_" + str(unit->name) + "_" + access_kind + "_" +
                               subject_name;
      regions.push_back(AccessRegion(
          String(name), String(subject_name), unit->name, unit_index, String(access_kind),
          String(value_kind_for_subject(subject_name)), rank, Array<String>{}, Array<PrimExpr>{},
          MakeZeroBounds(extents.size()), extents, MakeUnitStrides(extents.size()),
          String(rank == 0 ? "scalar" : "full"), String("unconditional"),
          MakeAnchors("access_region", name)));
    };
    for (const String& subject : unit->read_buffers) {
      emit_region(subject, "read");
    }
    for (const String& subject : unit->write_buffers) {
      emit_region(subject, "write");
    }
  }
  return regions;
}

std::string DeriveLiveValueRole(const BufferMetadata* metadata) {
  if (metadata == nullptr) {
    return "consumer_input";
  }
  if (metadata->scope == "global") {
    return "consumer_input";
  }
  return "fragment";
}

const BufferMetadata* FindBufferMetadata(
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer,
    const std::string& subject) {
  auto it = metadata_by_buffer.find(subject);
  if (it == metadata_by_buffer.end()) {
    return nullptr;
  }
  return &it->second;
}

std::string LiveValueKey(int64_t producer_unit_index, const std::string& subject) {
  return std::to_string(producer_unit_index) + "|" + subject;
}

bool AccessRegionReads(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "read" || access_kind == "read_write" || access_kind == "reduce_read";
}

bool AccessRegionWrites(const AccessRegion& region) {
  const std::string access_kind = str(region->access_kind);
  return access_kind == "write" || access_kind == "read_write" || access_kind == "reduce_write";
}

int64_t FindAccessRegionIndex(const Array<AccessRegion>& access_regions, int64_t unit_index,
                              const std::string& subject, bool require_write) {
  for (int64_t region_index = 0; region_index < static_cast<int64_t>(access_regions.size());
       ++region_index) {
    const AccessRegion& region = access_regions[region_index];
    if (region->unit_index != unit_index || str(region->subject) != subject) {
      continue;
    }
    if (require_write ? AccessRegionWrites(region) : AccessRegionReads(region)) {
      return region_index;
    }
  }
  return -1;
}

std::unordered_map<std::string, int64_t> BuildLiveValueIndexByProducerSubject(
    const Array<LiveValue>& live_values) {
  std::unordered_map<std::string, int64_t> index_by_key;
  for (int64_t live_value_index = 0; live_value_index < static_cast<int64_t>(live_values.size());
       ++live_value_index) {
    const LiveValue& value = live_values[live_value_index];
    index_by_key.emplace(
        LiveValueKey(value->producer_unit_index, static_cast<std::string>(value->subject)),
        live_value_index);
  }
  return index_by_key;
}

bool HasTrait(const Array<String>& traits, const std::string& trait) {
  for (const String& existing : traits) {
    if (str(existing) == trait) {
      return true;
    }
  }
  return false;
}

bool IsAccessRegionCompatibleForSlice(const Array<AccessRegion>& access_regions,
                                      int64_t producer_access_region_index,
                                      int64_t consumer_access_region_index) {
  if (producer_access_region_index < 0 || consumer_access_region_index < 0 ||
      producer_access_region_index >= static_cast<int64_t>(access_regions.size()) ||
      consumer_access_region_index >= static_cast<int64_t>(access_regions.size())) {
    return false;
  }
  return IsSliceCompatible(access_regions[producer_access_region_index],
                           access_regions[consumer_access_region_index]);
}

std::string DeriveLiveValueDefinitionKind(const DataflowEdge& edge) {
  const std::string edge_kind = str(edge->kind);
  if (edge_kind == "carry" || edge_kind == "join" || edge_kind == "reduction") {
    return "phi";
  }
  return "compute_write";
}

std::string DeriveLiveValueUseKind(const DataflowEdge& edge) {
  if (str(edge->kind) == "materialize") {
    return "materialization_consume";
  }
  return "compute_consume";
}

Array<LiveValue> BuildLiveValues(
    const Array<DataflowEdge>& dataflow_edges, const Array<ExecutionUnit>& execution_units,
    const Array<AccessRegion>& access_regions,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  Array<LiveValue> live_values;
  std::unordered_set<std::string> emitted;
  std::unordered_map<std::string, int64_t> next_version_by_subject;

  auto emit_live_value = [&](const std::string& name_seed, const std::string& subject,
                             const std::string& producer_unit, int64_t producer_unit_index,
                             std::string definition_kind, int64_t defining_access_region_index,
                             int64_t defining_event_index, std::vector<std::string> traits) {
    const std::string key = LiveValueKey(producer_unit_index, subject);
    if (!emitted.insert(key).second) {
      return;
    }
    const BufferMetadata* metadata = FindBufferMetadata(metadata_by_buffer, subject);
    const int64_t version_index = next_version_by_subject[subject]++;
    live_values.push_back(LiveValue(
        String("live_" + name_seed), String(subject), String(producer_unit), producer_unit_index,
        version_index, String(definition_kind), defining_access_region_index,
        defining_event_index, String(DeriveLiveValueRole(metadata)),
        metadata == nullptr ? Array<Integer>{} : ToIntegerArray(metadata->shape),
        metadata == nullptr ? String("unknown") : String(metadata->dtype), ToStringArray(traits),
        MakeAnchors("live_value", name_seed)));
  };

  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    if (str(edge->kind) == "materialize") {
      continue;
    }
    std::vector<std::string> traits;
    AppendUnique(&traits, str(edge->kind));
    if (edge->crosses_phase) {
      AppendUnique(&traits, "cross_phase");
    }
    const std::string subject = str(edge->subject);
    const int64_t defining_access_region_index =
        FindAccessRegionIndex(access_regions, edge->producer_unit_index, subject,
                              /*require_write=*/true);
    emit_live_value(str(edge->name), str(edge->subject), str(edge->producer_unit),
                    edge->producer_unit_index, DeriveLiveValueDefinitionKind(edge),
                    defining_access_region_index, edge_index, std::move(traits));
  }

  for (int64_t unit_index = 0; unit_index < static_cast<int64_t>(execution_units.size());
       ++unit_index) {
    const ExecutionUnit& unit = execution_units[unit_index];
    for (const String& subject : unit->write_buffers) {
      const std::string subject_name = str(subject);
      const BufferMetadata* metadata = FindBufferMetadata(metadata_by_buffer, subject_name);
      if (metadata == nullptr || metadata->scope == "global") {
        continue;
      }
      const int64_t defining_access_region_index =
          FindAccessRegionIndex(access_regions, unit_index, subject_name, /*require_write=*/true);
      emit_live_value("write_" + subject_name + "_" + std::to_string(unit_index), subject_name,
                      str(unit->name), unit_index, "compute_write", defining_access_region_index,
                      /*defining_event_index=*/-1, {"write_value"});
    }
  }
  return live_values;
}

int64_t ResolveSourceLiveValueIndexForEdge(
    int64_t edge_index, const Array<DataflowEdge>& dataflow_edges,
    const std::unordered_map<std::string, int64_t>& live_value_index_by_key) {
  const DataflowEdge& edge = dataflow_edges[edge_index];
  const std::string subject = str(edge->subject);
  if (str(edge->kind) == "materialize") {
    for (int64_t prior_index = edge_index - 1; prior_index >= 0; --prior_index) {
      const DataflowEdge& prior_edge = dataflow_edges[prior_index];
      if (str(prior_edge->subject) != subject ||
          prior_edge->consumer_unit_index != edge->producer_unit_index) {
        continue;
      }
      const std::string reaching_key = LiveValueKey(prior_edge->producer_unit_index, subject);
      auto reaching_it = live_value_index_by_key.find(reaching_key);
      if (reaching_it != live_value_index_by_key.end()) {
        return reaching_it->second;
      }
    }
  }
  const std::string live_value_key = LiveValueKey(edge->producer_unit_index, subject);
  auto live_value_index_it = live_value_index_by_key.find(live_value_key);
  if (live_value_index_it == live_value_index_by_key.end()) {
    return -1;
  }
  return live_value_index_it->second;
}

Array<LiveValueEdge> BuildLiveValueEdges(const Array<DataflowEdge>& dataflow_edges,
                                         const Array<LiveValue>& live_values,
                                         const Array<AccessRegion>& access_regions,
                                         const std::unordered_map<std::string, std::string>&
                                             target_subject_by_edge) {
  Array<LiveValueEdge> live_value_edges;
  const std::unordered_map<std::string, int64_t> live_value_index_by_key =
      BuildLiveValueIndexByProducerSubject(live_values);
  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    const int64_t live_value_index =
        ResolveSourceLiveValueIndexForEdge(edge_index, dataflow_edges, live_value_index_by_key);
    ICHECK_GE(live_value_index, 0)
        << "BuildLiveValueEdges requires producer live value for DataflowEdge " << edge->name;
    const LiveValue& live_value = live_values[live_value_index];
    const auto target_subject_it =
        target_subject_by_edge.find(static_cast<std::string>(edge->name));
    int64_t target_version_index = live_value->version_index;
    if (target_subject_it != target_subject_by_edge.end()) {
      const std::string target_key =
          LiveValueKey(edge->consumer_unit_index, target_subject_it->second);
      auto target_live_value_it = live_value_index_by_key.find(target_key);
      ICHECK(target_live_value_it != live_value_index_by_key.end())
          << "BuildLiveValueEdges requires target live value for DataflowEdge " << edge->name
          << " target " << target_subject_it->second;
      target_version_index = live_values[target_live_value_it->second]->version_index;
    }
    const int64_t consumer_access_region_index =
        FindAccessRegionIndex(access_regions, edge->consumer_unit_index, str(edge->subject),
                              /*require_write=*/false);
    const bool accepts_distributed_slice =
        HasTrait(edge->traits, "distributed_slice_consumer") &&
        IsAccessRegionCompatibleForSlice(access_regions, live_value->defining_access_region_index,
                                         consumer_access_region_index);
    live_value_edges.push_back(LiveValueEdge(
        String("live_edge_" + str(edge->name)), live_value->name, live_value_index, edge->name,
        edge_index, edge->producer_unit, edge->consumer_unit, edge->producer_unit_index,
        edge->consumer_unit_index, edge->kind, String(DeriveLiveValueUseKind(edge)),
        consumer_access_region_index, live_value->version_index, target_version_index,
        !accepts_distributed_slice, accepts_distributed_slice,
        MakeAnchors("live_value_edge", str(edge->name))));
  }
  return live_value_edges;
}

Array<MaterializationBoundary> BuildMaterializationBoundaries(
    const Array<DataflowEdge>& dataflow_edges, const Array<LiveValue>& live_values,
    const Array<LiveValueEdge>& live_value_edges,
    const std::unordered_map<std::string, std::string>& target_subject_by_edge,
    const Array<DependenceComponent>& dependence_components) {
  Array<MaterializationBoundary> materialization_boundaries;
  ICHECK_EQ(dataflow_edges.size(), live_value_edges.size())
      << "BuildMaterializationBoundaries requires "
         "dataflow_edges/live_value_edges alignment";
  const std::unordered_map<std::string, int64_t> live_value_index_by_key =
      BuildLiveValueIndexByProducerSubject(live_values);
  std::unordered_set<int64_t> recurrent_edge_indices;
  for (const DependenceComponent& component : dependence_components) {
    const std::string component_kind = str(component->component_kind);
    if (component_kind != "carry_cycle" && component_kind != "reduction_cycle" &&
        component_kind != "recurrence") {
      continue;
    }
    for (const Integer& edge_index_value : component->edge_indices) {
      recurrent_edge_indices.insert(edge_index_value->value);
    }
  }
  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    const LiveValueEdge& live_value_edge = live_value_edges[edge_index];
    const LiveValue& source_live_value = live_values[live_value_edge->source_live_value_index];
    const auto target_subject_it =
        target_subject_by_edge.find(static_cast<std::string>(edge->name));
    const std::string target_subject = target_subject_it == target_subject_by_edge.end()
                                           ? static_cast<std::string>(source_live_value->subject)
                                           : target_subject_it->second;
    int64_t target_live_value_index = live_value_edge->source_live_value_index;
    if (target_subject_it != target_subject_by_edge.end()) {
      const std::string target_key = LiveValueKey(edge->consumer_unit_index, target_subject);
      auto target_live_value_it = live_value_index_by_key.find(target_key);
      ICHECK(target_live_value_it != live_value_index_by_key.end())
          << "BuildMaterializationBoundaries requires target live value for DataflowEdge "
          << edge->name << " target " << target_subject;
      target_live_value_index = target_live_value_it->second;
    }
    const LiveValue& target_live_value = live_values[target_live_value_index];
    const bool crosses_phase = edge->crosses_phase;
    const bool distributed_slice = live_value_edge->accepts_distributed_slice &&
                                   !live_value_edge->requires_full_logical_value;
    const std::string event_lifetime_kind = recurrent_edge_indices.count(edge_index)
                                                ? "loop_carried"
                                                : (crosses_phase ? "multi_event" : "single_event");
    materialization_boundaries.push_back(MaterializationBoundary(
        String("materialization_" + str(edge->name)), source_live_value->name,
        live_value_edge->source_live_value_index, target_live_value->name, target_live_value_index,
        live_value_edge->name, edge_index, String(crosses_phase ? "next_phase" : "same_unit"),
        String(distributed_slice ? "distributed_slice" : "full_logical_value"),
        String(crosses_phase ? "cross_phase" : "same_phase"),
        source_live_value->defining_access_region_index,
        target_live_value->defining_access_region_index, String(event_lifetime_kind),
        /*min_publish_pages=*/1, /*max_consume_pages=*/1,
        MakeAnchors("materialization_boundary", str(edge->name))));
  }
  return materialization_boundaries;
}

Array<PhasePlan> BuildPhasePlans(const Array<ExecutionUnit>& execution_units,
                                 const Array<DataflowEdge>& dataflow_edges,
                                 const std::vector<int>& unit_phases) {
  struct PhaseInfo {
    std::unordered_set<std::string> unit_names;
    std::vector<int> unit_indices;
    std::unordered_set<std::string> ingress_edge_names;
    std::vector<int> ingress_edge_indices;
    std::unordered_set<std::string> egress_edge_names;
    std::vector<int> egress_edge_indices;
    std::unordered_set<std::string> boundary_subjects;
  };

  std::unordered_map<int, PhaseInfo> phases;
  for (int unit_index = 0; unit_index < static_cast<int>(execution_units.size()); ++unit_index) {
    const int phase_index =
        unit_index < static_cast<int>(unit_phases.size()) ? unit_phases[unit_index] : 0;
    PhaseInfo& phase = phases[phase_index];
    phase.unit_names.insert(str(execution_units[unit_index]->name));
    phase.unit_indices.push_back(unit_index);
  }

  for (int edge_index = 0; edge_index < static_cast<int>(dataflow_edges.size()); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    if (edge->producer_unit_index < 0 || edge->consumer_unit_index < 0 ||
        edge->producer_unit_index >= static_cast<int64_t>(unit_phases.size()) ||
        edge->consumer_unit_index >= static_cast<int64_t>(unit_phases.size()) ||
        !edge->crosses_phase) {
      continue;
    }
    const int producer_phase = unit_phases[edge->producer_unit_index];
    const int consumer_phase = unit_phases[edge->consumer_unit_index];
    PhaseInfo& producer_info = phases[producer_phase];
    producer_info.egress_edge_names.insert(str(edge->name));
    producer_info.egress_edge_indices.push_back(edge_index);
    producer_info.boundary_subjects.insert(str(edge->subject));

    PhaseInfo& consumer_info = phases[consumer_phase];
    consumer_info.ingress_edge_names.insert(str(edge->name));
    consumer_info.ingress_edge_indices.push_back(edge_index);
    consumer_info.boundary_subjects.insert(str(edge->subject));
  }

  std::vector<int> phase_indices;
  phase_indices.reserve(phases.size());
  for (const auto& [phase_index, _] : phases) {
    phase_indices.push_back(phase_index);
  }
  std::sort(phase_indices.begin(), phase_indices.end());

  Array<PhasePlan> phase_plans;
  for (int phase_index : phase_indices) {
    PhaseInfo& phase = phases[phase_index];
    std::sort(phase.unit_indices.begin(), phase.unit_indices.end());
    std::sort(phase.ingress_edge_indices.begin(), phase.ingress_edge_indices.end());
    std::sort(phase.egress_edge_indices.begin(), phase.egress_edge_indices.end());
    phase_plans.push_back(PhasePlan(
        String("phase_" + std::to_string(phase_index)), phase_index,
        ExecutionUnitNamesForIndices(phase.unit_indices, execution_units),
        ToIntegerArray(phase.unit_indices),
        DataflowEdgeNamesForIndices(phase.ingress_edge_indices, dataflow_edges),
        ToIntegerArray(phase.ingress_edge_indices),
        DataflowEdgeNamesForIndices(phase.egress_edge_indices, dataflow_edges),
        ToIntegerArray(phase.egress_edge_indices), ToStringArraySorted(phase.boundary_subjects),
        MakeAnchors("phase_plan", std::to_string(phase_index))));
  }
  return phase_plans;
}

SpatialPlan BuildSpatialPlanForFunc(const std::string& member_func, const tir::PrimFunc& func) {
  const std::vector<ClosureCandidateInfo> candidates = AnalyzeClosureCandidates(func);
  Array<ExecutionClosure> closures;
  for (const ClosureCandidateInfo& candidate : candidates) {
    closures.push_back(candidate.closure);
  }
  const ValidatedHintSet validated_hints = BuildEmptyValidatedHintSet(member_func);
  const Array<ExecutionUnit> execution_units = BuildExecutionUnits(closures);
  const Array<LayoutSpec> layout_specs =
      PropagateLocalValueFlowLayoutSpecs(
          MergePriorTypedLayoutSpecs(func, BuildLayoutSpecs(func, execution_units)),
          candidates);
  BufferMetadataCollector metadata_collector;
  const std::unordered_map<std::string, BufferMetadata> metadata_by_buffer =
      metadata_collector.Collect(func);
  const Array<TensorPlacementIntent> tensor_placement_intents =
      BuildTensorPlacementIntents(func, layout_specs, metadata_by_buffer);
  const Array<AccessRegion> access_regions =
      BuildAccessRegions(execution_units, layout_specs, metadata_by_buffer);
  const Array<ClosureBoundary> boundaries =
      BuildClosureBoundariesFromAccessRegions(execution_units, access_regions);
  const std::vector<int> unit_phases =
      ComputeExecutionUnitPhases(boundaries, static_cast<int>(execution_units.size()));
  const Array<DataflowEdge> closure_dataflow_edges = BuildDataflowEdges(boundaries, unit_phases);
  const SpatialLocalValueDependenceEdges local_value_flow_edges =
      BuildLocalValueDependenceEdges(execution_units, BuildLocalValueFlowEvidence(candidates));
  const Array<DataflowEdge> dataflow_edges =
      ConcatDataflowEdges(closure_dataflow_edges, local_value_flow_edges.dataflow_edges);
  const Array<DependenceComponent> dependence_components =
      BuildDependenceComponents(execution_units, dataflow_edges);
  const Array<PhasePlan> phase_plans =
      BuildPhasePlans(execution_units, dataflow_edges, unit_phases);
  const Array<LiveValue> live_values =
      BuildLiveValues(dataflow_edges, execution_units, access_regions, metadata_by_buffer);
  const Array<LiveValueEdge> live_value_edges =
      BuildLiveValueEdges(dataflow_edges, live_values, access_regions,
                          local_value_flow_edges.target_subject_by_edge);
  const Array<MaterializationBoundary> materialization_boundaries =
      BuildMaterializationBoundaries(dataflow_edges, live_values, live_value_edges,
                                     local_value_flow_edges.target_subject_by_edge,
                                     dependence_components);

  return SpatialPlan(String(member_func), execution_units, access_regions, dataflow_edges,
                     dependence_components, layout_specs, phase_plans, live_values,
                     live_value_edges, materialization_boundaries, tensor_placement_intents,
                     validated_hints, closures, boundaries,
                     MakeAnchors("spatial_plan", member_func));
}

}  // namespace

tvm::transform::Pass BuildSpatialPlan() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }

      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const SpatialPlan plan = BuildSpatialPlanForFunc(member_func, func.value());

      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs =
          updated_func->attrs.defined() ? updated_func->attrs->dict : Map<String, Any>();
      attrs.Set(attr::kTLSpatialPlan, plan);
      attrs.Set(attr::kTLSpatialPlanValidated, Bool(false));
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tl.transform.BuildSpatialPlan", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.BuildSpatialPlan", BuildSpatialPlan);
}

}  // namespace tl
}  // namespace tvm
