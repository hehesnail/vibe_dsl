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
#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
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

struct StatementAccessSummary {
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::unordered_map<std::string, std::string> scope_by_buffer;
  bool has_compute_consume{false};
};

struct ClosureCandidateInfo {
  std::string name;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  ExecutionClosure closure;
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
        }
      }
    }
    tir::StmtExprVisitor::VisitExpr_(op);
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
    candidates.push_back(std::move(info));
  }
  return candidates;
}

Array<ClosureBoundary> BuildBoundaryCandidates(const std::vector<ClosureCandidateInfo>& closures) {
  Array<ClosureBoundary> boundaries;
  std::unordered_set<std::string> emitted;
  std::unordered_map<std::string, std::vector<int>> producer_indices_by_subject;

  for (int closure_index = 0; closure_index < static_cast<int>(closures.size()); ++closure_index) {
    const ClosureCandidateInfo& closure = closures[closure_index];
    const std::unordered_set<std::string> read_set(closure.reads.begin(), closure.reads.end());

    for (const std::string& subject : closure.writes) {
      if (read_set.count(subject)) {
        const std::string key = "carry|" + subject + "|" + std::to_string(closure_index) + "|" +
                                std::to_string(closure_index);
        if (emitted.insert(key).second) {
          boundaries.push_back(ClosureBoundary(
              String("carry_" + subject + "_" + std::to_string(closure_index)), String("carry"),
              String(closure.name), String(closure.name), closure_index, closure_index,
              String(subject), ToStringArray({"self_edge"}), MakeAnchors("closure_boundary", key)));
        }
      }
    }

    for (const std::string& subject : closure.reads) {
      auto it = producer_indices_by_subject.find(subject);
      if (it == producer_indices_by_subject.end() || it->second.empty()) {
        continue;
      }
      const std::vector<int>& producers = it->second;
      const bool is_join = producers.size() > 1;
      const std::string boundary_kind = is_join ? "join" : "flow";
      for (int producer_index : producers) {
        const std::string key = boundary_kind + "|" + subject + "|" +
                                std::to_string(producer_index) + "|" +
                                std::to_string(closure_index);
        if (!emitted.insert(key).second) {
          continue;
        }
        boundaries.push_back(ClosureBoundary(
            String(boundary_kind + "_" + subject + "_" + std::to_string(producer_index) + "_" +
                   std::to_string(closure_index)),
            String(boundary_kind), String(closures[producer_index].name), String(closure.name),
            producer_index, closure_index, String(subject),
            is_join ? MakeTraits({"multi_producer"}) : Array<String>{},
            MakeAnchors("closure_boundary", key)));
      }
    }

    for (const std::string& subject : closure.writes) {
      producer_indices_by_subject[subject].push_back(closure_index);
    }
  }

  return boundaries;
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
    layout_specs.push_back(LayoutSpec(
        String("layout_" + subject), String(subject), String(info.scope),
        String(DeriveDistributionKind(info.scope)),
        ExecutionUnitNamesForIndices(info.unit_indices, execution_units),
        ToIntegerArray(info.unit_indices), Array<String>{}, MakeAnchors("layout_spec", subject)));
  }
  return layout_specs;
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
    const String& subject) {
  auto it = metadata_by_buffer.find(str(subject));
  if (it == metadata_by_buffer.end()) {
    return nullptr;
  }
  return &it->second;
}

Array<LiveValue> BuildLiveValues(
    const Array<DataflowEdge>& dataflow_edges,
    const std::unordered_map<std::string, BufferMetadata>& metadata_by_buffer) {
  Array<LiveValue> live_values;
  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    const BufferMetadata* metadata = FindBufferMetadata(metadata_by_buffer, edge->subject);
    std::vector<std::string> traits;
    AppendUnique(&traits, str(edge->kind));
    if (edge->crosses_phase) {
      AppendUnique(&traits, "cross_phase");
    }
    live_values.push_back(
        LiveValue(String("live_" + str(edge->name)), edge->subject, edge->producer_unit,
                  edge->producer_unit_index, String(DeriveLiveValueRole(metadata)),
                  metadata == nullptr ? Array<Integer>{} : ToIntegerArray(metadata->shape),
                  metadata == nullptr ? String("unknown") : String(metadata->dtype),
                  ToStringArray(traits), MakeAnchors("live_value", str(edge->name))));
  }
  return live_values;
}

Array<LiveValueEdge> BuildLiveValueEdges(const Array<DataflowEdge>& dataflow_edges,
                                         const Array<LiveValue>& live_values) {
  Array<LiveValueEdge> live_value_edges;
  ICHECK_EQ(dataflow_edges.size(), live_values.size())
      << "BuildLiveValueEdges requires dataflow_edges/live_values alignment";
  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    const LiveValue& live_value = live_values[edge_index];
    live_value_edges.push_back(
        LiveValueEdge(String("live_edge_" + str(edge->name)), live_value->name, edge_index,
                      edge->name, edge_index, edge->producer_unit, edge->consumer_unit,
                      edge->producer_unit_index, edge->consumer_unit_index, edge->kind, true, false,
                      MakeAnchors("live_value_edge", str(edge->name))));
  }
  return live_value_edges;
}

Array<MaterializationBoundary> BuildMaterializationBoundaries(
    const Array<DataflowEdge>& dataflow_edges, const Array<LiveValue>& live_values,
    const Array<LiveValueEdge>& live_value_edges) {
  Array<MaterializationBoundary> materialization_boundaries;
  ICHECK_EQ(dataflow_edges.size(), live_values.size())
      << "BuildMaterializationBoundaries requires dataflow_edges/live_values "
         "alignment";
  ICHECK_EQ(dataflow_edges.size(), live_value_edges.size())
      << "BuildMaterializationBoundaries requires "
         "dataflow_edges/live_value_edges alignment";
  for (int edge_index = 0; edge_index < dataflow_edges.size(); ++edge_index) {
    const DataflowEdge& edge = dataflow_edges[edge_index];
    const LiveValue& live_value = live_values[edge_index];
    const LiveValueEdge& live_value_edge = live_value_edges[edge_index];
    const bool crosses_phase = edge->crosses_phase;
    materialization_boundaries.push_back(MaterializationBoundary(
        String("materialization_" + str(edge->name)), live_value->name, edge_index,
        live_value_edge->name, edge_index, String(crosses_phase ? "next_phase" : "same_unit"),
        String("full_logical_value"), String(crosses_phase ? "cross_phase" : "same_phase"),
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
  const Array<ClosureBoundary> boundaries = BuildBoundaryCandidates(candidates);
  const ValidatedHintSet validated_hints = BuildEmptyValidatedHintSet(member_func);
  const Array<ExecutionUnit> execution_units = BuildExecutionUnits(closures);
  const std::vector<int> unit_phases =
      ComputeExecutionUnitPhases(boundaries, static_cast<int>(execution_units.size()));
  const Array<DataflowEdge> dataflow_edges = BuildDataflowEdges(boundaries, unit_phases);
  const Array<LayoutSpec> layout_specs = BuildLayoutSpecs(func, execution_units);
  const Array<PhasePlan> phase_plans =
      BuildPhasePlans(execution_units, dataflow_edges, unit_phases);
  BufferMetadataCollector metadata_collector;
  const std::unordered_map<std::string, BufferMetadata> metadata_by_buffer =
      metadata_collector.Collect(func);
  const Array<LiveValue> live_values = BuildLiveValues(dataflow_edges, metadata_by_buffer);
  const Array<LiveValueEdge> live_value_edges = BuildLiveValueEdges(dataflow_edges, live_values);
  const Array<MaterializationBoundary> materialization_boundaries =
      BuildMaterializationBoundaries(dataflow_edges, live_values, live_value_edges);

  return SpatialPlan(String(member_func), execution_units, dataflow_edges, layout_specs,
                     phase_plans, live_values, live_value_edges, materialization_boundaries,
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
