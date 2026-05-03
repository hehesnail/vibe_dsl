/*!
 * \file blackhole_lowering_requirements.cc
 * \brief Derive typed Blackhole lowering support facts directly from SpatialPlan and current TIR.
 */

#include "blackhole_lowering_requirements.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "blackhole_utils.h"
#include "buffer_tile_bridge_spec_utils.h"
#include "companion_base.h"

namespace tvm {
namespace tl {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;

namespace {

struct LoweringSupportFactsAnalysis {
  std::unordered_set<std::string> recurrence_subjects;
  BlackholeLoweringSupportFacts facts;
};

template <typename T>
void PushBackUnique(std::vector<T>* values, const T& value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

std::optional<std::vector<int64_t>> ExtractStaticShape(const Array<PrimExpr>& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (const PrimExpr& dim : shape) {
    const auto* imm = dim.as<tir::IntImmNode>();
    if (!imm) {
      return std::nullopt;
    }
    dims.push_back(imm->value);
  }
  return dims;
}

Array<Any> CollectLogicalTileLayoutSpecsFromBody(const tir::Stmt& body) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    Array<Any> Collect(const tir::Stmt& stmt) {
      specs_.clear();
      seen_.clear();
      VisitStmt(stmt);
      return specs_;
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
      auto scope_it = spec.find(String(schema_key::kScope));
      if (buffer_it == spec.end() || scope_it == spec.end()) {
        return;
      }
      const std::string key = str(Downcast<String>((*buffer_it).second)) + "|" +
                              str(Downcast<String>((*scope_it).second));
      if (!key.empty() && seen_.insert(key).second) {
        specs_.push_back(spec);
      }
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

    Array<Any> specs_;
    std::unordered_set<std::string> seen_;
  };
  return Collector().Collect(body);
}

std::unordered_map<std::string, std::vector<int64_t>> BuildLogicalBufferShapes(
    const tir::PrimFunc& func) {
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  auto remember = [&](const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (name.empty()) {
      return;
    }
    auto static_shape = ExtractStaticShape(buffer->shape);
    if (static_shape) {
      shapes[name] = static_shape.value();
    }
  };
  for (const auto& [_, buffer] : func->buffer_map) {
    remember(buffer);
  }
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    if (const auto* block = node.as<tir::BlockNode>()) {
      for (const tir::Buffer& buffer : block->alloc_buffers) {
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
    }
  });
  for (const Any& spec_any : CollectLogicalTileLayoutSpecsFromBody(func->body)) {
    Map<String, Any> spec = Downcast<Map<String, Any>>(spec_any);
    auto buffer_it = spec.find(String(schema_key::kBuffer));
    auto shape_it = spec.find(String(schema_key::kShape));
    if (buffer_it == spec.end() || shape_it == spec.end()) {
      continue;
    }
    std::vector<int64_t> shape;
    for (const Any& dim_any : Downcast<Array<Any>>((*shape_it).second)) {
      if (auto dim = dim_any.try_cast<Integer>()) {
        shape.push_back(dim.value()->value);
      } else {
        shape.clear();
        break;
      }
    }
    if (!shape.empty()) {
      shapes[Downcast<String>((*buffer_it).second)] = shape;
    }
  }
  return shapes;
}

int64_t GetLogicalElementCount(
    const std::string& buffer_name,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes) {
  auto it = logical_buffer_shapes.find(buffer_name);
  if (it == logical_buffer_shapes.end() || it->second.empty()) {
    return -1;
  }
  int64_t count = 1;
  for (int64_t dim : it->second) {
    if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

int64_t GetLogicalRowWidth(
    const tir::Buffer& buffer,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes) {
  const std::string name = BufferIdentityName(buffer);
  auto it = logical_buffer_shapes.find(name);
  if (it != logical_buffer_shapes.end() && !it->second.empty()) {
    return it->second.back();
  }
  auto shape = ExtractStaticShape(buffer->shape);
  if (shape && !shape.value().empty()) {
    return shape.value().back();
  }
  return -1;
}

int64_t GetLogicalRowWidth(
    const std::string& buffer_name,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes) {
  auto it = logical_buffer_shapes.find(buffer_name);
  if (it == logical_buffer_shapes.end() || it->second.empty()) {
    return -1;
  }
  return it->second.back();
}

bool IsTrackedStateScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

bool IsCBScope(const std::string& scope) {
  if (scope.rfind("shared", 0) == 0) {
    return true;
  }
  auto parsed = runtime::StorageScope::Create(scope);
  return parsed.rank == runtime::StorageRank::kBlackholeCB;
}

bool IsTrackedBufferFlowScope(const std::string& scope) {
  return IsTrackedStateScope(scope) || IsCBScope(scope);
}

bool IsDRAMScope(const std::string& scope) { return scope.empty() || scope == "global"; }

bool IsAccumulatorLikeScope(const std::string& scope) {
  if (scope.rfind("local", 0) == 0) {
    return true;
  }
  auto parsed = runtime::StorageScope::Create(scope);
  return parsed.rank == runtime::StorageRank::kBlackholeAccumulator;
}

bool IsBufferMaterializationCandidate(const tir::CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
  return tile_op.defined() && tile_op->GetBufferMaterializationInfo().has_value();
}

std::optional<BlackholeBufferMaterializationFact> TryBuildBufferMaterializationFact(
    const tir::CallNode* call, tir::Buffer* target_buffer_out) {
  if (!call) {
    return std::nullopt;
  }
  TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
  if (!tile_op.defined()) {
    return std::nullopt;
  }
  auto info = tile_op->GetBufferMaterializationInfo();
  if (!info.has_value()) {
    return std::nullopt;
  }
  const tir::Buffer& target = info->target_buffer;
  const std::string target_buffer_name = BufferIdentityName(target);
  const std::string scope = target.scope();
  if (target_buffer_name.empty() || !IsTrackedStateScope(scope)) {
    return std::nullopt;
  }
  if (target_buffer_out != nullptr) {
    *target_buffer_out = target;
  }
  BlackholeBufferMaterializationFact fact;
  fact.kind = buffer_materialization::kIntermediateAccumulatorMerge;
  fact.target_buffer = target_buffer_name;
  fact.scope = scope;
  fact.materialization_kind = str(info->materialization_kind);
  fact.bridge_kind = str(info->bridge_kind);
  fact.value_role = str(info->value_role);
  fact.merge_kind = str(info->merge_kind);
  fact.execution_protocol = str(info->execution_protocol);
  fact.result_live_form = str(info->result_live_form);
  return fact;
}

bool FlowFactHasEventKind(const BlackholeBufferFlowFact& flow_fact,
                          BlackholeBufferFlowEventKind kind) {
  for (const BlackholeBufferFlowEvent& event : flow_fact.events) {
    if (event.kind == kind) {
      return true;
    }
  }
  return false;
}

bool FlowFactHasSameOrderConsumeAndWrite(const BlackholeBufferFlowFact& flow_fact) {
  for (const BlackholeBufferFlowEvent& lhs : flow_fact.events) {
    if (lhs.kind != BlackholeBufferFlowEventKind::kComputeConsume) {
      continue;
    }
    for (const BlackholeBufferFlowEvent& rhs : flow_fact.events) {
      if (rhs.kind == BlackholeBufferFlowEventKind::kWrite &&
          rhs.order_index == lhs.order_index) {
        return true;
      }
    }
  }
  return false;
}

bool ExprIsLiteralZero(const PrimExpr& expr) {
  arith::Analyzer analyzer;
  const PrimExpr simplified = analyzer.Simplify(expr);
  if (const auto* imm = simplified.as<tir::IntImmNode>()) {
    return imm->value == 0;
  }
  if (const auto* imm = simplified.as<tir::FloatImmNode>()) {
    return imm->value == 0.0;
  }
  if (const auto* cast = simplified.as<tir::CastNode>()) {
    return ExprIsLiteralZero(cast->value);
  }
  return false;
}

bool StmtWritesBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer);
const tir::ForNode* AsUnwrappedFor(const tir::Stmt& stmt);
bool MatchDirectFragmentCastTarget(const tir::ForNode* op, tir::Buffer* src_buffer,
                                   tir::Buffer* dst_buffer);

bool StmtWritesOnlyZeroToBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer) {
  const std::string identity = BufferIdentityName(buffer);
  bool saw_write = false;
  bool only_zero_writes = true;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    const auto* store = node.as<tir::BufferStoreNode>();
    if (!store || BufferIdentityName(store->buffer) != identity) {
      return;
    }
    saw_write = true;
    if (!ExprIsLiteralZero(store->value)) {
      only_zero_writes = false;
    }
  });
  return saw_write && only_zero_writes;
}

bool BufferHasLiveInStateBeforeOrderIndex(const tir::Buffer& buffer,
                                          const std::vector<tir::Stmt>& ordered_stmts,
                                          const std::unordered_set<std::string>& recurrence_subjects,
                                          int order_index) {
  const std::string buffer_name = BufferIdentityName(buffer);
  if (buffer_name.empty()) {
    return true;
  }
  (void)recurrence_subjects;
  // SpatialPlan carry/self-edge truth only records structural read/write relationships
  // inside the normalized Tile TIR. It is not evidence that a fragment buffer already
  // holds live accumulator state before this materialization candidate executes.
  for (int i = 0; i < order_index; ++i) {
    if (!StmtWritesBuffer(ordered_stmts[i], buffer)) {
      continue;
    }
    if (!StmtWritesOnlyZeroToBuffer(ordered_stmts[i], buffer)) {
      return true;
    }
  }
  return false;
}

bool BufferFeedsDirectFragmentCastConsumerAfterOrderIndex(
    const tir::Buffer& buffer, const std::vector<tir::Stmt>& ordered_stmts, int order_index) {
  for (int i = order_index + 1; i < static_cast<int>(ordered_stmts.size()); ++i) {
    tir::Buffer src_buffer;
    tir::Buffer dst_buffer;
    const auto* loop = AsUnwrappedFor(ordered_stmts[i]);
    if (loop && MatchDirectFragmentCastTarget(loop, &src_buffer, &dst_buffer) &&
        SameBufferIdentity(src_buffer, buffer)) {
      return true;
    }
  }
  return false;
}

bool ShouldKeepBufferMaterializationFact(
    const BlackholeBufferMaterializationFact& fact, const tir::Buffer& target_buffer,
    const std::vector<tir::Stmt>& ordered_stmts,
    const std::unordered_set<std::string>& recurrence_subjects, int order_index) {
  if (fact.kind.empty() || !target_buffer.defined()) {
    return true;
  }
  if (fact.kind != buffer_materialization::kIntermediateAccumulatorMerge) {
    return true;
  }
  if (BufferFeedsDirectFragmentCastConsumerAfterOrderIndex(target_buffer, ordered_stmts,
                                                           order_index)) {
    return true;
  }
  return BufferHasLiveInStateBeforeOrderIndex(target_buffer, ordered_stmts, recurrence_subjects,
                                              order_index);
}

std::optional<BlackholeBufferMaterializationFact> TryBuildRepublishBufferMaterializationFact(
    const BlackholeBufferFlowFact& flow_fact) {
  if (flow_fact.buffer.empty() || !IsTrackedStateScope(flow_fact.scope) ||
      flow_fact.flow_class != CBFlowClass::kRepublish ||
      !FlowFactHasEventKind(flow_fact, BlackholeBufferFlowEventKind::kComputeConsume) ||
      FlowFactHasSameOrderConsumeAndWrite(flow_fact)) {
    return std::nullopt;
  }
  BlackholeBufferMaterializationFact fact;
  fact.kind = buffer_materialization::kRepublishedLogicalTile;
  fact.target_buffer = flow_fact.buffer;
  fact.scope = flow_fact.scope;
  fact.materialization_kind = buffer_materialization::kRepublishedBuffer;
  fact.bridge_kind = buffer_materialization::kTileNFacesMaterialization;
  fact.value_role = buffer_materialization::kConsumerInput;
  fact.merge_kind = buffer_materialization::kDirectWrite;
  fact.execution_protocol = buffer_materialization::kTiledCBRepublish;
  fact.result_live_form = buffer_live_form::kTiledCB;
  return fact;
}

BlackholeBufferMaterializationFact MakeRepublishedLogicalTileMaterializationFact(
    const std::string& buffer_name, const std::string& scope,
    const std::string& source_buffer = std::string(), int64_t logical_row_width = -1,
    int64_t logical_element_count = -1) {
  BlackholeBufferMaterializationFact fact;
  fact.kind = buffer_materialization::kRepublishedLogicalTile;
  fact.target_buffer = buffer_name;
  fact.scope = scope;
  fact.materialization_kind = buffer_materialization::kRepublishedBuffer;
  fact.bridge_kind = buffer_materialization::kTileNFacesMaterialization;
  fact.value_role = buffer_materialization::kConsumerInput;
  fact.merge_kind = buffer_materialization::kDirectWrite;
  fact.execution_protocol = buffer_materialization::kTiledCBRepublish;
  fact.result_live_form = buffer_live_form::kTiledCB;
  fact.source_buffer = source_buffer;
  fact.logical_row_width = logical_row_width;
  fact.logical_element_count = logical_element_count;
  return fact;
}

bool IsVectorLocalFragmentBuffer(const tir::Buffer& buffer) {
  const std::string scope = buffer.scope();
  return IsTrackedStateScope(scope) && buffer->shape.size() == 1 && !buffer->shape.empty() &&
         !tir::is_one(buffer->shape[0]);
}

const tir::ForNode* AsUnwrappedFor(const tir::Stmt& stmt) {
  tir::Stmt current = stmt;
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
  return current.as<tir::ForNode>();
}

const tir::BufferStoreNode* AsUnwrappedBufferStore(const tir::Stmt& stmt) {
  tir::Stmt current = stmt;
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
  return current.as<tir::BufferStoreNode>();
}

bool ExprUsesVar(const PrimExpr& expr, const tir::Var& var) {
  if (!var.defined()) {
    return false;
  }
  bool uses_var = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (const auto* candidate = node.as<tir::VarNode>()) {
      uses_var = uses_var || candidate == var.get();
    }
  });
  return uses_var;
}

bool MatchDirectFragmentCastTarget(const tir::ForNode* op, tir::Buffer* src_buffer,
                                   tir::Buffer* dst_buffer) {
  if (!op || !src_buffer || !dst_buffer) {
    return false;
  }
  const auto* store = AsUnwrappedBufferStore(op->body);
  const tir::ForNode* inner_loop = nullptr;
  const auto* inner_store = store;
  PrimExpr linear_index = op->loop_var;
  if (!store) {
    inner_loop = AsUnwrappedFor(op->body);
    inner_store = inner_loop ? AsUnwrappedBufferStore(inner_loop->body) : nullptr;
    if (!inner_loop || !inner_store || inner_store->indices.size() != 1) {
      return false;
    }
    linear_index = op->loop_var * inner_loop->extent + inner_loop->loop_var;
  } else if (store->indices.size() != 1) {
    return false;
  }
  if (!inner_store || !IsVectorLocalFragmentBuffer(inner_store->buffer)) {
    return false;
  }
  const auto* cast = inner_store->value.as<tir::CastNode>();
  const auto* load = cast ? cast->value.as<tir::BufferLoadNode>() : nullptr;
  if (!cast || !load || load->indices.size() != 1 || !IsVectorLocalFragmentBuffer(load->buffer) ||
      SameBufferIdentity(inner_store->buffer, load->buffer)) {
    return false;
  }
  arith::Analyzer analyzer;
  PrimExpr dst_offset = analyzer.Simplify(inner_store->indices[0] - linear_index);
  PrimExpr src_offset = analyzer.Simplify(load->indices[0] - linear_index);
  if (ExprUsesVar(dst_offset, op->loop_var) || ExprUsesVar(src_offset, op->loop_var)) {
    return false;
  }
  if (inner_loop &&
      (ExprUsesVar(dst_offset, inner_loop->loop_var) ||
       ExprUsesVar(src_offset, inner_loop->loop_var))) {
    return false;
  }
  *src_buffer = load->buffer;
  *dst_buffer = inner_store->buffer;
  return true;
}

bool MatchExplicitFragmentCastTarget(const tir::CallNode* op, tir::Buffer* src_buffer,
                                     tir::Buffer* dst_buffer) {
  if (!op || !src_buffer || !dst_buffer || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(op->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName || op->args.size() < 4U) {
    return false;
  }
  const auto* operation = op->args[0].as<StringImmNode>();
  if (!operation || operation->value != blackhole_tile_compute_schema::kTypecastTile ||
      !IsBufferLikeExpr(op->args[1]) || !IsBufferLikeExpr(op->args[2])) {
    return false;
  }
  *src_buffer = NormalizeToBufferRegion(op->args[1])->buffer;
  *dst_buffer = NormalizeToBufferRegion(op->args[2])->buffer;
  return src_buffer->defined() && dst_buffer->defined() &&
         !SameBufferIdentity(*src_buffer, *dst_buffer);
}

std::vector<tir::Stmt> CollectMaterializationOrderedStmts(const tir::Stmt& root) {
  class OrderedLeafStmtCollector : public tir::StmtVisitor {
   public:
    explicit OrderedLeafStmtCollector(std::vector<tir::Stmt>* ordered_stmts)
        : ordered_stmts_(ordered_stmts) {}

    void Collect(const tir::Stmt& stmt) {
      if (stmt.defined()) {
        VisitStmt(stmt);
      }
    }

    void VisitStmt_(const tir::SeqStmtNode* op) final {
      for (const tir::Stmt& child : op->seq) {
        VisitStmt(child);
      }
    }

    void VisitStmt_(const tir::BlockRealizeNode* op) final { VisitStmt(op->block); }

    void VisitStmt_(const tir::BlockNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tir::AttrStmtNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tir::AllocateNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tir::DeclBufferNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tir::LetStmtNode* op) final { VisitStmt(op->body); }

    void VisitStmt_(const tir::ForNode* op) final {
      ordered_stmts_->push_back(GetRef<tir::Stmt>(op));
    }

    void VisitStmt_(const tir::BufferStoreNode* op) final {
      ordered_stmts_->push_back(GetRef<tir::Stmt>(op));
    }

    void VisitStmt_(const tir::EvaluateNode* op) final {
      ordered_stmts_->push_back(GetRef<tir::Stmt>(op));
    }

   private:
    std::vector<tir::Stmt>* ordered_stmts_;
  };

  std::vector<tir::Stmt> ordered_stmts;
  OrderedLeafStmtCollector collector(&ordered_stmts);
  collector.Collect(root);
  return ordered_stmts;
}

bool IsCopyOperation(const tir::BufferStoreNode* store) {
  const auto* load = store ? store->value.as<tir::BufferLoadNode>() : nullptr;
  return load != nullptr && !SameBufferIdentity(store->buffer, load->buffer);
}

bool IsTransportConsumerDirection(const tir::BufferStoreNode* store) {
  const auto* load = store ? store->value.as<tir::BufferLoadNode>() : nullptr;
  if (!load) {
    return false;
  }
  const std::string dst_scope = store->buffer.scope();
  const std::string src_scope = load->buffer.scope();
  return (IsCBScope(src_scope) && IsDRAMScope(dst_scope)) ||
         (IsCBScope(src_scope) && IsCBScope(dst_scope)) ||
         (IsAccumulatorLikeScope(src_scope) && IsDRAMScope(dst_scope)) ||
         (IsAccumulatorLikeScope(src_scope) && IsCBScope(dst_scope));
}

bool StmtWritesBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer) {
  const std::string identity = BufferIdentityName(buffer);
  bool writes = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      writes = writes || BufferIdentityName(store->buffer) == identity;
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
    if (!tile_op.defined()) {
      return;
    }
    for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
      if (access.kind == DataflowAccessKind::kComputeProduce &&
          BufferIdentityName(access.buffer) == identity) {
        writes = true;
        return;
      }
    }
  });
  return writes;
}

bool StmtReadsBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer) {
  const std::string identity = BufferIdentityName(buffer);
  bool reads = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      reads = reads || BufferIdentityName(load->buffer) == identity;
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
    if (!tile_op.defined()) {
      return;
    }
    for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
      if (access.kind == DataflowAccessKind::kComputeConsume &&
          BufferIdentityName(access.buffer) == identity) {
        reads = true;
        return;
      }
    }
  });
  return reads;
}

bool StmtReferencesBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer) {
  const std::string identity = BufferIdentityName(buffer);
  bool referenced = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      referenced = referenced || BufferIdentityName(store->buffer) == identity;
      return;
    }
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      referenced = referenced || BufferIdentityName(load->buffer) == identity;
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call) {
      return;
    }
    for (const PrimExpr& arg : call->args) {
      if (!IsBufferLikeExpr(arg)) {
        continue;
      }
      referenced =
          referenced || BufferIdentityName(NormalizeToBufferRegion(arg)->buffer) == identity;
    }
  });
  return referenced;
}

bool StmtConsumesBufferViaTransport(const tir::Stmt& stmt, const tir::Buffer& buffer) {
  const std::string identity = BufferIdentityName(buffer);
  bool consumed = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    const auto* store = node.as<tir::BufferStoreNode>();
    if (!IsCopyOperation(store) || !IsTransportConsumerDirection(store)) {
      return;
    }
    const auto* load = store->value.as<tir::BufferLoadNode>();
    consumed = consumed || (load && BufferIdentityName(load->buffer) == identity);
  });
  return consumed;
}

std::unordered_set<std::string> CollectComputeConsumedBuffers(const tir::Stmt& stmt) {
  std::unordered_set<std::string> buffers;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    const auto* call = node.as<tir::CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
    if (!tile_op.defined()) {
      return;
    }
    for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
      if (access.kind == DataflowAccessKind::kComputeConsume) {
        const std::string name = BufferIdentityName(access.buffer);
        if (!name.empty()) {
          buffers.insert(name);
        }
      }
    }
  });
  return buffers;
}

CBFlowClass DeriveBufferFlowClass(const std::vector<BlackholeBufferFlowEvent>& events) {
  int first_consume = std::numeric_limits<int>::max();
  bool has_consume = false;
  for (const BlackholeBufferFlowEvent& event : events) {
    if (event.kind == BlackholeBufferFlowEventKind::kComputeConsume ||
        event.kind == BlackholeBufferFlowEventKind::kTransportConsume) {
      has_consume = true;
      first_consume = std::min(first_consume, event.order_index);
    }
  }
  if (!has_consume) {
    return CBFlowClass::kState;
  }
  for (const BlackholeBufferFlowEvent& event : events) {
    if (event.kind == BlackholeBufferFlowEventKind::kWrite && event.order_index > first_consume) {
      return CBFlowClass::kRepublish;
    }
  }
  return CBFlowClass::kStream;
}

bool IsStructuralBufferFlowWrapper(const tir::Stmt& stmt) {
  return stmt.as<tir::SeqStmtNode>() || stmt.as<tir::ForNode>() ||
         stmt.as<tir::AttrStmtNode>() || stmt.as<tir::DeclBufferNode>() ||
         stmt.as<tir::AllocateNode>() || stmt.as<tir::LetStmtNode>() ||
         stmt.as<tir::IfThenElseNode>();
}

std::vector<BlackholeBufferFlowFact> CollectBufferFlowFactsFromBody(const tir::Stmt& body) {
  std::unordered_map<std::string, tir::Buffer> tracked_buffers;
  const std::vector<tir::Stmt> ordered_stmts = CollectExecutionOrderedStmts(body);
  for (const tir::Stmt& stmt : ordered_stmts) {
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      auto remember = [&](const tir::Buffer& buffer) {
        if (!buffer.defined()) {
          return;
        }
        const std::string scope = buffer.scope();
        if (!IsTrackedBufferFlowScope(scope)) {
          return;
        }
        const std::string name = BufferIdentityName(buffer);
        if (!name.empty()) {
          tracked_buffers.emplace(name, buffer);
        }
      };
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
      TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
      if (tile_op.defined()) {
        for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
          remember(access.buffer);
        }
      }
      for (const PrimExpr& arg : call->args) {
        if (IsBufferLikeExpr(arg)) {
          remember(NormalizeToBufferRegion(arg)->buffer);
        }
      }
    });
  }

  std::unordered_map<std::string, BlackholeBufferFlowFact> facts_by_buffer;
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    const tir::Stmt& stmt = ordered_stmts[order_index];
    if (IsStructuralBufferFlowWrapper(stmt)) {
      continue;
    }
    const auto compute_consumed = CollectComputeConsumedBuffers(stmt);
    for (const auto& [buffer_name, buffer] : tracked_buffers) {
      if (!StmtReferencesBuffer(stmt, buffer)) {
        continue;
      }
      BlackholeBufferFlowFact& fact = facts_by_buffer[buffer_name];
      fact.buffer = buffer_name;
      fact.scope = buffer.scope();
      const bool compute_consume = compute_consumed.count(buffer_name);
      const bool transport_consume = StmtConsumesBufferViaTransport(stmt, buffer);
      const bool reads = StmtReadsBuffer(stmt, buffer);
      const bool writes = StmtWritesBuffer(stmt, buffer);
      auto append = [&](BlackholeBufferFlowEventKind kind) {
        fact.events.push_back(BlackholeBufferFlowEvent{order_index, kind});
      };
      if (compute_consume) {
        append(BlackholeBufferFlowEventKind::kComputeConsume);
      }
      if (!compute_consume && reads && !transport_consume) {
        append(BlackholeBufferFlowEventKind::kReference);
      }
      if (transport_consume) {
        append(BlackholeBufferFlowEventKind::kTransportConsume);
      }
      if (writes) {
        append(BlackholeBufferFlowEventKind::kWrite);
      }
      if (!compute_consume && !reads && !transport_consume && !writes) {
        append(BlackholeBufferFlowEventKind::kReference);
      }
    }
  }

  std::vector<BlackholeBufferFlowFact> facts;
  for (auto& [_, fact] : facts_by_buffer) {
    if (fact.events.empty()) {
      continue;
    }
    fact.flow_class = DeriveBufferFlowClass(fact.events);
    fact.publish_pages_per_event = 1;
    fact.consume_pages_per_event = 1;
    facts.push_back(std::move(fact));
  }
  return facts;
}

void AppendUniqueBufferMaterializationFactsFromFlowFacts(
    const std::vector<BlackholeBufferFlowFact>& flow_facts,
    std::vector<BlackholeBufferMaterializationFact>* materialization_facts) {
  std::unordered_set<std::string> seen;
  for (const BlackholeBufferMaterializationFact& fact : *materialization_facts) {
    seen.insert(fact.target_buffer + "|" + fact.scope);
  }
  for (const BlackholeBufferFlowFact& flow_fact : flow_facts) {
    auto maybe_fact = TryBuildRepublishBufferMaterializationFact(flow_fact);
    if (!maybe_fact) {
      continue;
    }
    const std::string key = maybe_fact->target_buffer + "|" + maybe_fact->scope;
    if (seen.insert(key).second) {
      materialization_facts->push_back(maybe_fact.value());
    }
  }
}

void AppendUniqueCastDrivenBufferMaterializationFactsFromBody(
    const tir::Stmt& body, const std::vector<BlackholeBufferFlowFact>& flow_facts,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes,
    std::vector<BlackholeBufferMaterializationFact>* materialization_facts) {
  std::unordered_map<std::string, BlackholeBufferFlowFact> flow_by_buffer;
  for (const BlackholeBufferFlowFact& flow : flow_facts) {
    if (!flow.buffer.empty()) {
      flow_by_buffer.emplace(flow.buffer, flow);
    }
  }
  auto upsert = [&](const BlackholeBufferMaterializationFact& fact) {
    const std::string key = fact.target_buffer + "|" + fact.scope;
    for (BlackholeBufferMaterializationFact& existing : *materialization_facts) {
      const std::string existing_key = existing.target_buffer + "|" + existing.scope;
      if (existing_key == key) {
        existing = fact;
        return;
      }
    }
    materialization_facts->push_back(fact);
  };

  auto maybe_append_cast_fact = [&](const tir::Buffer& src_buffer,
                                    const tir::Buffer& dst_buffer) {
    const std::string src_name = BufferIdentityName(src_buffer);
    const std::string dst_name = BufferIdentityName(dst_buffer);
    auto flow_it = flow_by_buffer.find(dst_name);
    if (src_name.empty() || dst_name.empty() || flow_it == flow_by_buffer.end()) {
      return;
    }
    const BlackholeBufferFlowFact& flow_fact = flow_it->second;
    if (!IsTrackedStateScope(flow_fact.scope) ||
        (!FlowFactHasEventKind(flow_fact, BlackholeBufferFlowEventKind::kComputeConsume) &&
         !FlowFactHasEventKind(flow_fact, BlackholeBufferFlowEventKind::kTransportConsume))) {
      return;
    }
    auto lookup_row_width = [&](const tir::Buffer& buffer) {
      return GetLogicalRowWidth(buffer, logical_buffer_shapes);
    };
    auto lookup_element_count = [&](const std::string& buffer_name) {
      return GetLogicalElementCount(buffer_name, logical_buffer_shapes);
    };
    int64_t logical_row_width = lookup_row_width(src_buffer);
    if (logical_row_width <= 0) {
      logical_row_width = lookup_row_width(dst_buffer);
    }
    const int64_t dst_logical_element_count = lookup_element_count(dst_name);
    const int64_t src_logical_element_count = lookup_element_count(src_name);
    int64_t logical_element_count = std::max(dst_logical_element_count, src_logical_element_count);
    if (logical_element_count <= 0) {
      logical_element_count =
          dst_logical_element_count > 0 ? dst_logical_element_count : src_logical_element_count;
    }
    upsert(MakeRepublishedLogicalTileMaterializationFact(
        dst_name, flow_fact.scope, src_name, logical_row_width, logical_element_count));
  };

  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    tir::Buffer src_buffer;
    tir::Buffer dst_buffer;
    if (const auto* loop = node.as<tir::ForNode>()) {
      if (MatchDirectFragmentCastTarget(loop, &src_buffer, &dst_buffer)) {
        maybe_append_cast_fact(src_buffer, dst_buffer);
      }
      return;
    }
    if (const auto* call = node.as<tir::CallNode>()) {
      if (MatchExplicitFragmentCastTarget(call, &src_buffer, &dst_buffer)) {
        maybe_append_cast_fact(src_buffer, dst_buffer);
      }
    }
  });
}

std::unordered_map<std::string, std::string> CollectBufferScopesFromBody(const tir::Stmt& body) {
  std::unordered_map<std::string, std::string> scopes;
  auto remember = [&](const tir::Buffer& buffer) {
    const std::string name = BufferIdentityName(buffer);
    if (!name.empty() && scopes.find(name) == scopes.end()) {
      scopes.emplace(name, std::string(buffer.scope()));
    }
  };
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    if (const auto* store = node.as<tir::BufferStoreNode>()) {
      remember(store->buffer);
      return;
    }
    if (const auto* load = node.as<tir::BufferLoadNode>()) {
      remember(load->buffer);
      return;
    }
    if (const auto* decl = node.as<tir::DeclBufferNode>()) {
      remember(decl->buffer);
      return;
    }
    if (const auto* block = node.as<tir::BlockNode>()) {
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        remember(buffer);
      }
      return;
    }
    const auto* call = node.as<tir::CallNode>();
    if (!call) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
    if (tile_op.defined()) {
      for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
        remember(access.buffer);
      }
    }
    for (const PrimExpr& arg : call->args) {
      if (IsBufferLikeExpr(arg)) {
        remember(NormalizeToBufferRegion(arg)->buffer);
      }
    }
  });
  return scopes;
}

void AppendUniqueSpatialPlanMaterializationFacts(
    const SpatialPlan& plan, const std::unordered_map<std::string, std::string>& scope_by_buffer,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes,
    std::vector<BlackholeBufferMaterializationFact>* materialization_facts) {
  std::unordered_map<std::string, std::string> subject_by_live_value;
  for (const LiveValue& live_value : plan->live_values) {
    const std::string name = str(live_value->name);
    const std::string subject = str(live_value->subject);
    if (!name.empty() && !subject.empty()) {
      subject_by_live_value.emplace(name, subject);
    }
  }
  auto upsert = [&](const BlackholeBufferMaterializationFact& fact) {
    const std::string key = fact.target_buffer + "|" + fact.scope;
    for (BlackholeBufferMaterializationFact& existing : *materialization_facts) {
      const std::string existing_key = existing.target_buffer + "|" + existing.scope;
      if (existing_key == key) {
        existing = fact;
        return;
      }
    }
    materialization_facts->push_back(fact);
  };
  for (int64_t boundary_index = 0;
       boundary_index < static_cast<int64_t>(plan->materialization_boundaries.size());
       ++boundary_index) {
    const MaterializationBoundary& boundary =
        plan->materialization_boundaries[static_cast<size_t>(boundary_index)];
    auto source_it = subject_by_live_value.find(str(boundary->source_live_value));
    auto target_it = subject_by_live_value.find(str(boundary->target_live_value));
    if (source_it == subject_by_live_value.end() ||
        target_it == subject_by_live_value.end() ||
        source_it->second == target_it->second) {
      continue;
    }
    auto scope_it = scope_by_buffer.find(target_it->second);
    if (scope_it == scope_by_buffer.end() || !IsTrackedStateScope(scope_it->second)) {
      continue;
    }
    int64_t logical_row_width = GetLogicalRowWidth(source_it->second, logical_buffer_shapes);
    if (logical_row_width <= 0) {
      logical_row_width = GetLogicalRowWidth(target_it->second, logical_buffer_shapes);
    }
    int64_t logical_element_count =
        std::max(GetLogicalElementCount(source_it->second, logical_buffer_shapes),
                 GetLogicalElementCount(target_it->second, logical_buffer_shapes));
    BlackholeBufferMaterializationFact fact = MakeRepublishedLogicalTileMaterializationFact(
        target_it->second, scope_it->second, source_it->second, logical_row_width,
        logical_element_count);
    fact.spatial_materialization_boundary = str(boundary->name);
    fact.spatial_materialization_boundary_index = boundary_index;
    fact.spatial_live_value_edge = str(boundary->live_value_edge);
    fact.spatial_live_value_edge_index = boundary->live_value_edge_index;
    fact.source_live_value = str(boundary->source_live_value);
    fact.source_live_value_index = boundary->source_live_value_index;
    fact.target_live_value = str(boundary->target_live_value);
    fact.target_live_value_index = boundary->target_live_value_index;
    upsert(fact);
  }
}

std::vector<BlackholeBufferMaterializationFact> CollectBufferMaterializationFactsFromBody(
    const tir::PrimFunc& func, const std::unordered_set<std::string>& recurrence_subjects) {
  std::vector<BlackholeBufferMaterializationFact> facts;
  std::unordered_set<std::string> seen;
  const std::vector<tir::Stmt> ordered_stmts = CollectMaterializationOrderedStmts(func->body);
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    tir::PostOrderVisit(ordered_stmts[order_index], [&](const ObjectRef& node) {
      const auto* call = node.as<tir::CallNode>();
      if (!IsBufferMaterializationCandidate(call)) {
        return;
      }
      tir::Buffer target_buffer;
      auto maybe_fact = TryBuildBufferMaterializationFact(call, &target_buffer);
      if (!maybe_fact || !ShouldKeepBufferMaterializationFact(
                             maybe_fact.value(), target_buffer, ordered_stmts,
                             recurrence_subjects, order_index)) {
        return;
      }
      const std::string key = maybe_fact->target_buffer + "|" + maybe_fact->scope;
      if (seen.insert(key).second) {
        facts.push_back(maybe_fact.value());
      }
    });
  }
  return facts;
}

BlackholeLoweringSupportFacts CollectBlackholeLoweringSupportFactsImpl(const tir::PrimFunc& func,
                                                                       const SpatialPlan& plan) {
  LoweringSupportFactsAnalysis analysis;
  for (const DataflowEdge& edge : plan->dataflow_edges) {
    if (str(edge->kind) != "carry") {
      continue;
    }
    const std::string subject = str(edge->subject);
    if (!subject.empty()) {
      analysis.recurrence_subjects.insert(subject);
    }
  }
  std::vector<BlackholeBufferFlowFact> flow_facts = CollectBufferFlowFactsFromBody(func->body);
  std::vector<BlackholeBufferMaterializationFact> materialization_facts =
      CollectBufferMaterializationFactsFromBody(func, analysis.recurrence_subjects);
  const std::unordered_map<std::string, std::vector<int64_t>> logical_buffer_shapes =
      BuildLogicalBufferShapes(func);
  AppendUniqueBufferMaterializationFactsFromFlowFacts(flow_facts, &materialization_facts);
  AppendUniqueCastDrivenBufferMaterializationFactsFromBody(
      func->body, flow_facts, logical_buffer_shapes, &materialization_facts);
  AppendUniqueSpatialPlanMaterializationFacts(
      plan, CollectBufferScopesFromBody(func->body), logical_buffer_shapes,
      &materialization_facts);
  if (!materialization_facts.empty()) {
    analysis.facts.buffer_materialization_facts = std::move(materialization_facts);
  }
  if (!flow_facts.empty()) {
    analysis.facts.buffer_flow_facts = std::move(flow_facts);
  }
  return analysis.facts;
}

}  // namespace

BlackholeLoweringSupportFacts CollectBlackholeLoweringSupportFacts(const tir::PrimFunc& func,
                                                                   const SpatialPlan& plan) {
  return CollectBlackholeLoweringSupportFactsImpl(func, plan);
}

}  // namespace tl
}  // namespace tvm
