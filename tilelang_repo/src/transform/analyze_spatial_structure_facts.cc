/*!
 * \file analyze_spatial_structure_facts.cc
 * \brief Analyze normalized TIR into minimal Task 1 spatial structure facts.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_plan.h"

namespace tvm {
namespace tl {

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
  bool has_gemm{false};
};

struct ClosureCandidateInfo {
  int stmt_index{-1};
  std::string name;
  std::string execution_role;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::vector<std::string> traits;
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

std::string GetBufferScope(const tir::Buffer& buffer) {
  return buffer.defined() ? buffer.scope() : "";
}

std::string ExtractBufferIdentityFromExpr(const PrimExpr& expr) {
  if (const auto* load = expr.as<tir::BufferLoadNode>()) {
    return BufferIdentityName(load->buffer);
  }
  if (const auto* call = expr.as<tir::CallNode>()) {
    if (!call->args.empty()) {
      if (const auto* load = call->args[0].as<tir::BufferLoadNode>()) {
        return BufferIdentityName(load->buffer);
      }
    }
  }
  return "";
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
    if (op->op->IsInstance<OpNode>()) {
      const std::string op_name = Downcast<Op>(op->op)->name;
      if (op_name == "tl.tileop.gemm_py" && op->args.size() >= 3) {
        summary_.has_gemm = true;
        AppendUnique(&summary_.reads, ExtractBufferIdentityFromExpr(op->args[0]));
        AppendUnique(&summary_.reads, ExtractBufferIdentityFromExpr(op->args[1]));
        AppendUnique(&summary_.writes, ExtractBufferIdentityFromExpr(op->args[2]));
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
  if (summary.has_gemm) {
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
  if (summary.has_gemm) {
    AppendUnique(&traits, "locality_obligation");
  }
  return traits;
}

ExecutionClosure BuildExecutionClosure(int stmt_index, const StatementAccessSummary& summary) {
  const std::string name = "closure_" + std::to_string(stmt_index);
  return ExecutionClosure(
      String(name), String("normalized_tir_top_level_stmt"),
      String(DeriveExecutionRole(summary)), ToIntegerArray({stmt_index}),
      ToStringArray(summary.reads), ToStringArray(summary.writes),
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
    if (summary.reads.empty() && summary.writes.empty() && !summary.has_gemm) {
      continue;
    }
    ClosureCandidateInfo info;
    info.stmt_index = stmt_index;
    info.closure = BuildExecutionClosure(stmt_index, summary);
    info.name = str(info.closure->name);
    info.execution_role = str(info.closure->execution_role);
    info.reads = std::move(summary.reads);
    info.writes = std::move(summary.writes);
    info.traits = DeriveClosureTraits(summary);
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
    const std::unordered_set<std::string> write_set(closure.writes.begin(), closure.writes.end());

    for (const std::string& subject : closure.writes) {
      if (read_set.count(subject)) {
        const std::string key =
            "carry|" + subject + "|" + std::to_string(closure_index) + "|" +
            std::to_string(closure_index);
        if (emitted.insert(key).second) {
          boundaries.push_back(ClosureBoundary(
              String("carry_" + subject + "_" + std::to_string(closure_index)), String("carry"),
              String(closure.name), String(closure.name), closure_index, closure_index,
              String(subject), ToStringArray({"self_edge"}),
              MakeAnchors("closure_boundary", key)));
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

SpatialStructureFacts BuildSpatialStructureFacts(const std::string& member_func,
                                                 const tir::PrimFunc& func) {
  const std::vector<ClosureCandidateInfo> candidates = AnalyzeClosureCandidates(func);
  Array<ExecutionClosure> closures;
  for (const ClosureCandidateInfo& candidate : candidates) {
    closures.push_back(candidate.closure);
  }
  return SpatialStructureFacts(String(member_func), closures, BuildBoundaryCandidates(candidates),
                               BuildEmptyValidatedHintSet(member_func),
                               MakeAnchors("spatial_structure_facts", member_func));
}

}  // namespace

tvm::transform::Pass AnalyzeSpatialStructureFacts() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }

      const std::string member_func = GetMemberFuncName(gvar, func.value());
      const SpatialStructureFacts facts = BuildSpatialStructureFacts(member_func, func.value());
      tir::PrimFunc updated_func = func.value();
      Map<String, Any> attrs = updated_func->attrs.defined() ? updated_func->attrs->dict
                                                             : Map<String, Any>();
      attrs.Set(attr::kTLSpatialStructureFacts, facts);
      updated_func.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated_func);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.transform.AnalyzeSpatialStructureFacts", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSpatialStructureFacts",
                        AnalyzeSpatialStructureFacts);
}

}  // namespace tl
}  // namespace tvm
