/*!
 * \file blackhole_lowering_requirements.cc
 * \brief Derive Blackhole lowering requirements directly from SpatialPlan and analysis attrs.
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

static constexpr const char* kLogicalComputeRegionsAttr = "blackhole.logical_compute_regions";

struct LoweringSupportFacts {
  std::unordered_set<std::string> selection_subjects;
  std::unordered_set<std::string> recurrence_subjects;
  std::unordered_set<std::string> reduction_subjects;
  Map<String, Any> compute_support_payload;
  Array<Any> buffer_tile_bridge_specs;
  Array<Any> buffer_materialization_contracts;
  Array<Any> buffer_flow_contracts;
};

template <typename T>
void PushBackUnique(std::vector<T>* values, const T& value) {
  if (std::find(values->begin(), values->end(), value) == values->end()) {
    values->push_back(value);
  }
}

void PushBackUnique(Array<Any>* values, std::unordered_set<std::string>* seen, const Any& value,
                    const std::string& key) {
  if (!key.empty() && seen->insert(key).second) {
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
  if (auto maybe_regions = func->GetAttr<Array<Any>>(kLogicalComputeRegionsAttr)) {
    for (const Any& region_any : maybe_regions.value()) {
      Map<String, Any> region = Downcast<Map<String, Any>>(region_any);
      auto region_buffers_it = region.find(String(manifest_key::kRegionBuffers));
      if (region_buffers_it == region.end()) {
        continue;
      }
      for (const Any& region_buffer_any : Downcast<Array<Any>>((*region_buffers_it).second)) {
        Map<String, Any> region_buffer = Downcast<Map<String, Any>>(region_buffer_any);
        auto name_it = region_buffer.find(String(schema_key::kName));
        auto buffer_it = region_buffer.find(String(schema_key::kBuffer));
        if (name_it == region_buffer.end() || buffer_it == region_buffer.end()) {
          continue;
        }
        auto buffer = (*buffer_it).second.try_cast<tir::Buffer>();
        if (!buffer.has_value()) {
          continue;
        }
        auto static_shape = ExtractStaticShape(buffer.value()->shape);
        if (!static_shape) {
          continue;
        }
        shapes[Downcast<String>((*name_it).second)] = static_shape.value();
      }
      auto bridge_specs_it = region.find(String(schema_key::kBufferTileBridgeSpecs));
      if (bridge_specs_it == region.end()) {
        continue;
      }
      for (const Any& spec_any : Downcast<Array<Any>>((*bridge_specs_it).second)) {
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

Optional<Map<String, Any>> TryBuildBufferMaterializationContract(const tir::CallNode* call,
                                                                tir::Buffer* target_buffer_out) {
  if (!call) {
    return Optional<Map<String, Any>>();
  }
  TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
  if (!tile_op.defined()) {
    return Optional<Map<String, Any>>();
  }
  auto info = tile_op->GetBufferMaterializationInfo();
  if (!info.has_value()) {
    return Optional<Map<String, Any>>();
  }
  const tir::Buffer& target = info->target_buffer;
  const std::string target_buffer_name = BufferIdentityName(target);
  const std::string scope = target.scope();
  if (target_buffer_name.empty() || !IsTrackedStateScope(scope)) {
    return Optional<Map<String, Any>>();
  }
  if (target_buffer_out != nullptr) {
    *target_buffer_out = target;
  }
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(buffer_materialization::kIntermediateAccumulatorMerge));
  contract.Set(String(schema_key::kTargetBuffer), String(target_buffer_name));
  contract.Set(String(schema_key::kScope), String(scope));
  contract.Set(String(schema_key::kMaterializationKind), info->materialization_kind);
  contract.Set(String(schema_key::kBridgeKind), info->bridge_kind);
  contract.Set(String(schema_key::kValueRole), info->value_role);
  contract.Set(String(schema_key::kMergeKind), info->merge_kind);
  contract.Set(String(schema_key::kExecutionProtocol), info->execution_protocol);
  contract.Set(String(schema_key::kResultLiveForm), info->result_live_form);
  return contract;
}

bool FlowContractHasEventKind(const Map<String, Any>& flow_contract, const char* kind) {
  auto events_it = flow_contract.find(String(schema_key::kEvents));
  if (events_it == flow_contract.end()) {
    return false;
  }
  for (const Any& event_any : Downcast<Array<Any>>((*events_it).second)) {
    Map<String, Any> event = Downcast<Map<String, Any>>(event_any);
    auto kind_it = event.find(String(schema_key::kKind));
    if (kind_it != event.end() && Downcast<String>((*kind_it).second) == kind) {
      return true;
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
  if (recurrence_subjects.count(buffer_name) != 0U) {
    return true;
  }
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

bool ShouldKeepBufferMaterializationContract(
    const Map<String, Any>& contract, const tir::Buffer& target_buffer,
    const std::vector<tir::Stmt>& ordered_stmts,
    const std::unordered_set<std::string>& recurrence_subjects, int order_index) {
  auto kind_it = contract.find(String(schema_key::kKind));
  if (kind_it == contract.end() || !target_buffer.defined()) {
    return true;
  }
  if (Downcast<String>((*kind_it).second) !=
      buffer_materialization::kIntermediateAccumulatorMerge) {
    return true;
  }
  if (BufferFeedsDirectFragmentCastConsumerAfterOrderIndex(target_buffer, ordered_stmts,
                                                           order_index)) {
    return true;
  }
  return BufferHasLiveInStateBeforeOrderIndex(target_buffer, ordered_stmts, recurrence_subjects,
                                              order_index);
}

Optional<Map<String, Any>> TryBuildRepublishBufferMaterializationContract(
    const Map<String, Any>& flow_contract) {
  auto buffer_it = flow_contract.find(String(schema_key::kBuffer));
  auto scope_it = flow_contract.find(String(schema_key::kScope));
  auto flow_class_it = flow_contract.find(String(schema_key::kFlowClass));
  auto granule_kind_it = flow_contract.find(String(schema_key::kGranuleKind));
  if (buffer_it == flow_contract.end() || scope_it == flow_contract.end() ||
      flow_class_it == flow_contract.end() || granule_kind_it == flow_contract.end()) {
    return Optional<Map<String, Any>>();
  }
  const std::string buffer_name = Downcast<String>((*buffer_it).second);
  const std::string scope = Downcast<String>((*scope_it).second);
  const std::string flow_class = Downcast<String>((*flow_class_it).second);
  const std::string granule_kind = Downcast<String>((*granule_kind_it).second);
  if (buffer_name.empty() || !IsTrackedStateScope(scope) ||
      flow_class != buffer_flow::kRepublish ||
      granule_kind != buffer_flow::kLogicalTile ||
      !FlowContractHasEventKind(flow_contract, buffer_flow::kComputeConsume)) {
    return Optional<Map<String, Any>>();
  }
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(buffer_materialization::kRepublishedLogicalTile));
  contract.Set(String(schema_key::kTargetBuffer), String(buffer_name));
  contract.Set(String(schema_key::kScope), String(scope));
  contract.Set(String(schema_key::kMaterializationKind),
               String(buffer_materialization::kRepublishedBuffer));
  contract.Set(String(schema_key::kBridgeKind),
               String(buffer_materialization::kTileNFacesMaterialization));
  contract.Set(String(schema_key::kValueRole),
               String(buffer_materialization::kConsumerInput));
  contract.Set(String(schema_key::kMergeKind),
               String(buffer_materialization::kDirectWrite));
  contract.Set(String(schema_key::kExecutionProtocol),
               String(buffer_materialization::kTiledCBRepublish));
  contract.Set(String(schema_key::kResultLiveForm), String(buffer_live_form::kTiledCB));
  return contract;
}

Map<String, Any> MakeRepublishedLogicalTileMaterializationContract(
    const std::string& buffer_name, const std::string& scope,
    const std::string& source_buffer = std::string(), int64_t logical_row_width = -1,
    int64_t logical_element_count = -1) {
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(buffer_materialization::kRepublishedLogicalTile));
  contract.Set(String(schema_key::kTargetBuffer), String(buffer_name));
  contract.Set(String(schema_key::kScope), String(scope));
  contract.Set(String(schema_key::kMaterializationKind),
               String(buffer_materialization::kRepublishedBuffer));
  contract.Set(String(schema_key::kBridgeKind),
               String(buffer_materialization::kTileNFacesMaterialization));
  contract.Set(String(schema_key::kValueRole),
               String(buffer_materialization::kConsumerInput));
  contract.Set(String(schema_key::kMergeKind),
               String(buffer_materialization::kDirectWrite));
  contract.Set(String(schema_key::kExecutionProtocol),
               String(buffer_materialization::kTiledCBRepublish));
  contract.Set(String(schema_key::kResultLiveForm), String(buffer_live_form::kTiledCB));
  if (!source_buffer.empty()) {
    contract.Set(String(schema_key::kSourceBuffer), String(source_buffer));
  }
  if (logical_row_width > 0) {
    contract.Set(String(schema_key::kLogicalRowWidth), Integer(logical_row_width));
  }
  if (logical_element_count > 0) {
    contract.Set(String(schema_key::kLogicalElementCount), Integer(logical_element_count));
  }
  return contract;
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
      if (BufferIdentityName(access.buffer) == identity) {
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

enum class BufferFlowEventKind {
  kWrite,
  kComputeConsume,
  kTransportConsume,
  kReference,
};

struct BufferFlowEvent {
  int order_index = -1;
  BufferFlowEventKind kind = BufferFlowEventKind::kReference;
};

struct BufferFlowContract {
  std::string buffer_name;
  std::string scope;
  std::string flow_class = buffer_flow::kState;
  int publish_granule = 1;
  int consume_granule = 1;
  std::vector<BufferFlowEvent> events;
};

std::string BufferFlowEventKindToString(BufferFlowEventKind kind) {
  switch (kind) {
    case BufferFlowEventKind::kWrite:
      return buffer_flow::kWrite;
    case BufferFlowEventKind::kComputeConsume:
      return buffer_flow::kComputeConsume;
    case BufferFlowEventKind::kTransportConsume:
      return buffer_flow::kTransportConsume;
    default:
      return buffer_flow::kReference;
  }
}

std::string DeriveBufferFlowClassLabel(const std::vector<BufferFlowEvent>& events) {
  int first_consume = std::numeric_limits<int>::max();
  bool has_consume = false;
  for (const BufferFlowEvent& event : events) {
    if (event.kind == BufferFlowEventKind::kComputeConsume ||
        event.kind == BufferFlowEventKind::kTransportConsume) {
      has_consume = true;
      first_consume = std::min(first_consume, event.order_index);
    }
  }
  if (!has_consume) {
    return buffer_flow::kState;
  }
  for (const BufferFlowEvent& event : events) {
    if (event.kind == BufferFlowEventKind::kWrite && event.order_index > first_consume) {
      return buffer_flow::kRepublish;
    }
  }
  return buffer_flow::kStream;
}

Array<Any> CollectBufferFlowContractsFromBody(const tir::Stmt& body) {
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

  std::unordered_map<std::string, BufferFlowContract> contracts_by_buffer;
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    const tir::Stmt& stmt = ordered_stmts[order_index];
    const auto compute_consumed = CollectComputeConsumedBuffers(stmt);
    for (const auto& [buffer_name, buffer] : tracked_buffers) {
      if (!StmtReferencesBuffer(stmt, buffer)) {
        continue;
      }
      BufferFlowContract& contract = contracts_by_buffer[buffer_name];
      contract.buffer_name = buffer_name;
      contract.scope = buffer.scope();
      const bool compute_consume = compute_consumed.count(buffer_name);
      const bool transport_consume = StmtConsumesBufferViaTransport(stmt, buffer);
      const bool reads = StmtReadsBuffer(stmt, buffer);
      const bool writes = StmtWritesBuffer(stmt, buffer);
      auto append = [&](BufferFlowEventKind kind) {
        contract.events.push_back(BufferFlowEvent{order_index, kind});
      };
      if (compute_consume) {
        append(BufferFlowEventKind::kComputeConsume);
      }
      if (!compute_consume && reads && !transport_consume) {
        append(BufferFlowEventKind::kReference);
      }
      if (transport_consume) {
        append(BufferFlowEventKind::kTransportConsume);
      }
      if (writes) {
        append(BufferFlowEventKind::kWrite);
      }
      if (!compute_consume && !reads && !transport_consume && !writes) {
        append(BufferFlowEventKind::kReference);
      }
    }
  }

  Array<Any> contracts;
  for (auto& [buffer_name, contract] : contracts_by_buffer) {
    if (contract.events.empty()) {
      continue;
    }
    contract.flow_class = DeriveBufferFlowClassLabel(contract.events);
    Map<String, Any> encoded;
    encoded.Set(String(schema_key::kBuffer), String(buffer_name));
    encoded.Set(String(schema_key::kScope), String(contract.scope));
    encoded.Set(String(schema_key::kFlowClass), String(contract.flow_class));
    encoded.Set(String(schema_key::kGranuleKind), String(buffer_flow::kLogicalTile));
    encoded.Set(String(schema_key::kPublishGranule), Integer(contract.publish_granule));
    encoded.Set(String(schema_key::kConsumeGranule), Integer(contract.consume_granule));
    Array<Any> events;
    for (const BufferFlowEvent& event : contract.events) {
      Map<String, Any> encoded_event;
      encoded_event.Set(String(schema_key::kKind),
                        String(BufferFlowEventKindToString(event.kind)));
      encoded_event.Set(String(schema_key::kOrderIndex), Integer(event.order_index));
      events.push_back(encoded_event);
    }
    encoded.Set(String(schema_key::kEvents), events);
    contracts.push_back(encoded);
  }
  return contracts;
}

void AppendUniqueBufferMaterializationContractsFromFlowContracts(
    const Array<Any>& flow_contracts, Array<Any>* materialization_contracts) {
  std::unordered_set<std::string> seen;
  for (const Any& contract_any : *materialization_contracts) {
    Map<String, Any> contract = Downcast<Map<String, Any>>(contract_any);
    seen.insert(Downcast<String>(contract.at(String(schema_key::kTargetBuffer))) + "|" +
                Downcast<String>(contract.at(String(schema_key::kScope))));
  }
  for (const Any& flow_contract_any : flow_contracts) {
    Map<String, Any> flow_contract = Downcast<Map<String, Any>>(flow_contract_any);
    auto maybe_contract = TryBuildRepublishBufferMaterializationContract(flow_contract);
    if (!maybe_contract) {
      continue;
    }
    const std::string key =
        Downcast<String>(maybe_contract.value().at(String(schema_key::kTargetBuffer))) + "|" +
        Downcast<String>(maybe_contract.value().at(String(schema_key::kScope)));
    if (seen.insert(key).second) {
      materialization_contracts->push_back(maybe_contract.value());
    }
  }
}

void AppendUniqueCastDrivenBufferMaterializationContractsFromBody(
    const tir::Stmt& body, const Array<Any>& flow_contracts,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes,
    Array<Any>* materialization_contracts) {
  std::unordered_map<std::string, Map<String, Any>> flow_by_buffer;
  for (const Any& flow_any : flow_contracts) {
    Map<String, Any> flow = Downcast<Map<String, Any>>(flow_any);
    auto buffer_it = flow.find(String(schema_key::kBuffer));
    if (buffer_it != flow.end()) {
      flow_by_buffer.emplace(Downcast<String>((*buffer_it).second), flow);
    }
  }
  auto upsert = [&](const Map<String, Any>& contract) {
    const std::string key = Downcast<String>(contract.at(String(schema_key::kTargetBuffer))) + "|" +
                            Downcast<String>(contract.at(String(schema_key::kScope)));
    for (int i = 0; i < materialization_contracts->size(); ++i) {
      Map<String, Any> existing = Downcast<Map<String, Any>>((*materialization_contracts)[i]);
      const std::string existing_key =
          Downcast<String>(existing.at(String(schema_key::kTargetBuffer))) + "|" +
          Downcast<String>(existing.at(String(schema_key::kScope)));
      if (existing_key == key) {
        materialization_contracts->Set(i, contract);
        return;
      }
    }
    materialization_contracts->push_back(contract);
  };

  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* loop = node.as<tir::ForNode>();
    if (!loop) {
      return;
    }
    tir::Buffer src_buffer;
    tir::Buffer dst_buffer;
    if (!MatchDirectFragmentCastTarget(loop, &src_buffer, &dst_buffer)) {
      return;
    }
    const std::string src_name = BufferIdentityName(src_buffer);
    const std::string dst_name = BufferIdentityName(dst_buffer);
    auto flow_it = flow_by_buffer.find(dst_name);
    if (src_name.empty() || dst_name.empty() || flow_it == flow_by_buffer.end()) {
      return;
    }
    const Map<String, Any>& flow_contract = flow_it->second;
    const std::string scope = Downcast<String>(flow_contract.at(String(schema_key::kScope)));
    const std::string granule_kind =
        Downcast<String>(flow_contract.at(String(schema_key::kGranuleKind)));
    if (!IsTrackedStateScope(scope) || granule_kind != buffer_flow::kLogicalTile ||
        (!FlowContractHasEventKind(flow_contract, buffer_flow::kComputeConsume) &&
         !FlowContractHasEventKind(flow_contract, buffer_flow::kTransportConsume))) {
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
    upsert(MakeRepublishedLogicalTileMaterializationContract(
        dst_name, scope, src_name, logical_row_width, logical_element_count));
  });
}

Array<Any> CollectBufferMaterializationContractsFromBody(
    const tir::PrimFunc& func, const std::unordered_set<std::string>& recurrence_subjects) {
  Array<Any> contracts;
  std::unordered_set<std::string> seen;
  const std::vector<tir::Stmt> ordered_stmts = CollectMaterializationOrderedStmts(func->body);
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    tir::PostOrderVisit(ordered_stmts[order_index], [&](const ObjectRef& node) {
      const auto* call = node.as<tir::CallNode>();
      if (!IsBufferMaterializationCandidate(call)) {
        return;
      }
      tir::Buffer target_buffer;
      auto maybe_contract = TryBuildBufferMaterializationContract(call, &target_buffer);
      if (!maybe_contract || !ShouldKeepBufferMaterializationContract(
                                 maybe_contract.value(), target_buffer, ordered_stmts,
                                 recurrence_subjects, order_index)) {
        return;
      }
      const std::string key =
          Downcast<String>(maybe_contract.value().at(String(schema_key::kTargetBuffer))) + "|" +
          Downcast<String>(maybe_contract.value().at(String(schema_key::kScope)));
      if (seen.insert(key).second) {
        contracts.push_back(maybe_contract.value());
      }
    });
  }
  return contracts;
}

Array<Any> GetComputeRegions(const tir::PrimFunc& func) {
  return func->GetAttr<Array<Any>>("blackhole.compute_regions").value_or(Array<Any>());
}

void SetArrayRequirementIfMissing(Map<String, Any>* lowering_requirements,
                                  const String& key, const Array<Any>& values) {
  if (!values.empty() && !lowering_requirements->count(key)) {
    lowering_requirements->Set(key, values);
  }
}

std::string KeyForAnyMap(const Map<String, Any>& map, const char* primary_key,
                         const char* secondary_key = nullptr) {
  if (auto value = map.Get(String(primary_key))) {
    if (auto maybe_string = value.value().try_cast<String>()) {
      return maybe_string.value();
    }
  }
  if (secondary_key != nullptr) {
    if (auto value = map.Get(String(secondary_key))) {
      if (auto maybe_string = value.value().try_cast<String>()) {
        return maybe_string.value();
      }
    }
  }
  return "";
}

LoweringSupportFacts AnalyzeLoweringSupportFacts(const tir::PrimFunc& func) {
  LoweringSupportFacts facts;
  Array<Any> compute_op_kinds;
  Array<Any> pointwise_ops;
  Array<Any> reduction_targets;
  Array<Any> broadcast_sources;
  Array<Any> loop_carried_state;
  Array<Any> buffer_tile_bridge_specs;
  std::unordered_set<std::string> seen_compute_ops;
  std::unordered_set<std::string> seen_pointwise_ops;
  std::unordered_set<std::string> seen_reduction_targets;
  std::unordered_set<std::string> seen_broadcasts;
  std::unordered_set<std::string> seen_loop_carried;
  std::unordered_set<std::string> seen_buffer_tile_bridge_specs;

  for (const Any& region_any : GetComputeRegions(func)) {
    Map<String, Any> region = Downcast<Map<String, Any>>(region_any);
    if (auto maybe_ops = region.Get(String("ops"))) {
      for (const Any& op_any : Downcast<Array<Any>>(maybe_ops.value())) {
        const std::string name = Downcast<String>(op_any);
        PushBackUnique(&compute_op_kinds, &seen_compute_ops, String(name), name);
      }
    }
    if (auto maybe_ops = region.Get(String("pointwise_ops"))) {
      for (const Any& op_any : Downcast<Array<Any>>(maybe_ops.value())) {
        const std::string name = Downcast<String>(op_any);
        PushBackUnique(&pointwise_ops, &seen_pointwise_ops, String(name), name);
      }
    }
    if (auto maybe_targets = region.Get(String(manifest_key::kReductions))) {
      for (const Any& target_any : Downcast<Array<Any>>(maybe_targets.value())) {
        Map<String, Any> target = Downcast<Map<String, Any>>(target_any);
        const std::string name = KeyForAnyMap(target, schema_key::kTarget);
        if (!name.empty()) {
          facts.reduction_subjects.insert(name);
          PushBackUnique(&reduction_targets, &seen_reduction_targets, target, name);
        }
      }
    }
    if (auto maybe_sources = region.Get(String(manifest_key::kBroadcasts))) {
      for (const Any& source_any : Downcast<Array<Any>>(maybe_sources.value())) {
        Map<String, Any> source = Downcast<Map<String, Any>>(source_any);
        const std::string name = KeyForAnyMap(source, schema_key::kSource);
        PushBackUnique(&broadcast_sources, &seen_broadcasts, source, name);
      }
    }
    if (auto maybe_targets = region.Get(String(manifest_key::kSelectionTargets))) {
      for (const Any& target_any : Downcast<Array<Any>>(maybe_targets.value())) {
        Map<String, Any> target = Downcast<Map<String, Any>>(target_any);
        const std::string name = KeyForAnyMap(target, schema_key::kName);
        if (!name.empty()) {
          facts.selection_subjects.insert(name);
        }
      }
    }
    if (auto maybe_state = region.Get(String(manifest_key::kLoopCarriedState))) {
      for (const Any& state_any : Downcast<Array<Any>>(maybe_state.value())) {
        Map<String, Any> state = Downcast<Map<String, Any>>(state_any);
        const std::string name = KeyForAnyMap(state, schema_key::kName);
        if (!name.empty()) {
          facts.recurrence_subjects.insert(name);
          PushBackUnique(&loop_carried_state, &seen_loop_carried, state, name);
        }
      }
    }
    if (auto maybe_specs = region.Get(String(schema_key::kBufferTileBridgeSpecs))) {
      for (const Any& spec_any : Downcast<Array<Any>>(maybe_specs.value())) {
        Map<String, Any> spec = Downcast<Map<String, Any>>(spec_any);
        const std::string key =
            KeyForAnyMap(spec, schema_key::kBuffer) + "|" +
            KeyForAnyMap(spec, schema_key::kScope);
        PushBackUnique(&buffer_tile_bridge_specs, &seen_buffer_tile_bridge_specs, spec, key);
      }
    }
    if (auto maybe_recurrence = region.Get(String(manifest_key::kRecurrenceEdges))) {
      for (const Any& edge_any : Downcast<Array<Any>>(maybe_recurrence.value())) {
        Map<String, Any> edge = Downcast<Map<String, Any>>(edge_any);
        const std::string target = KeyForAnyMap(edge, schema_key::kTarget);
        if (!target.empty()) {
          facts.recurrence_subjects.insert(target);
        }
      }
    }
  }
  if (auto maybe_regions = func->GetAttr<Array<Any>>(kLogicalComputeRegionsAttr)) {
    for (const Any& region_any : maybe_regions.value()) {
      Map<String, Any> region = Downcast<Map<String, Any>>(region_any);
      if (auto maybe_specs = region.Get(String(schema_key::kBufferTileBridgeSpecs))) {
        for (const Any& spec_any : Downcast<Array<Any>>(maybe_specs.value())) {
          Map<String, Any> spec = Downcast<Map<String, Any>>(spec_any);
          const std::string key =
              KeyForAnyMap(spec, schema_key::kBuffer) + "|" +
              KeyForAnyMap(spec, schema_key::kScope);
          PushBackUnique(&buffer_tile_bridge_specs, &seen_buffer_tile_bridge_specs, spec, key);
        }
      }
    }
  }

  Array<Any> flow_contracts = CollectBufferFlowContractsFromBody(func->body);
  Array<Any> materialization_contracts =
      CollectBufferMaterializationContractsFromBody(func, facts.recurrence_subjects);
  AppendUniqueBufferMaterializationContractsFromFlowContracts(flow_contracts,
                                                              &materialization_contracts);
  AppendUniqueCastDrivenBufferMaterializationContractsFromBody(
      func->body, flow_contracts, BuildLogicalBufferShapes(func), &materialization_contracts);

  if (!compute_op_kinds.empty()) {
    facts.compute_support_payload.Set(String(schema_key::kComputeOpKinds), compute_op_kinds);
  }
  if (!pointwise_ops.empty()) {
    facts.compute_support_payload.Set(String(schema_key::kPointwiseOpKinds), pointwise_ops);
  }
  if (!reduction_targets.empty()) {
    facts.compute_support_payload.Set(String(schema_key::kReductionTargets),
                                      reduction_targets);
  }
  if (!broadcast_sources.empty()) {
    facts.compute_support_payload.Set(String(schema_key::kBroadcastSources),
                                      broadcast_sources);
  }
  if (!loop_carried_state.empty()) {
    facts.compute_support_payload.Set(String(schema_key::kLoopCarriedState), loop_carried_state);
  }
  if (!buffer_tile_bridge_specs.empty()) {
    facts.buffer_tile_bridge_specs = buffer_tile_bridge_specs;
  }
  if (!materialization_contracts.empty()) {
    facts.buffer_materialization_contracts = materialization_contracts;
  }
  if (!flow_contracts.empty()) {
    facts.buffer_flow_contracts = flow_contracts;
  }
  return facts;
}

std::vector<int> ComputeClosurePhases(const SpatialPlan& plan) {
  const int n = static_cast<int>(plan->closures.size());
  std::vector<std::vector<int>> preds(n);
  for (const ClosureBoundary& boundary : plan->boundaries) {
    if (boundary->source_closure_index < 0 || boundary->target_closure_index < 0 ||
        boundary->source_closure_index == boundary->target_closure_index) {
      continue;
    }
    preds[boundary->target_closure_index].push_back(boundary->source_closure_index);
  }
  std::vector<int> phase(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int pred : preds[i]) {
      phase[i] = std::max(phase[i], phase[pred] + 1);
    }
  }
  return phase;
}

void CollectWorkDecompositionFacts(const tir::PrimFunc& func, Map<String, Any>* requirements) {
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  if (!work) {
    return;
  }
  if (auto axes = work.value().Get(String("axes"))) {
    requirements->Set(String("work_axes"), Downcast<Array<Any>>(axes.value()));
  }
  if (auto derived = work.value().Get(String("derived_index_exprs"))) {
    const int count = static_cast<int>(Downcast<Array<Any>>(derived.value()).size());
    if (count > 0) {
      requirements->Set(String("derived_index_expr_count"), Integer(count));
    }
  }
  if (auto bounds = work.value().Get(String("work_dependent_loop_bounds"))) {
    const int count = static_cast<int>(Downcast<Array<Any>>(bounds.value()).size());
    if (count > 0) {
      requirements->Set(String("work_dependent_loop_bound_count"), Integer(count));
    }
  }
}

void CollectPhaseFacts(const SpatialPlan& plan, Map<String, Any>* requirements) {
  if (plan->closures.empty()) {
    return;
  }
  const std::vector<int> phases = ComputeClosurePhases(plan);
  std::unordered_set<int> unique_phases;
  Array<Any> phase_boundary_buffers;
  std::unordered_set<std::string> seen_boundary_subjects;
  for (int phase : phases) {
    unique_phases.insert(phase);
  }
  for (const ClosureBoundary& boundary : plan->boundaries) {
    if (boundary->source_closure_index < 0 || boundary->target_closure_index < 0) {
      continue;
    }
    if (phases[boundary->source_closure_index] == phases[boundary->target_closure_index]) {
      continue;
    }
    const std::string subject = boundary->subject;
    if (!subject.empty() && seen_boundary_subjects.insert(subject).second) {
      phase_boundary_buffers.push_back(String(subject));
    }
  }
  if (!unique_phases.empty()) {
    requirements->Set(String("spatial_phase_count"), Integer(static_cast<int>(unique_phases.size())));
  }
  if (!plan->boundaries.empty()) {
    requirements->Set(String("spatial_channel_count"),
                      Integer(static_cast<int>(plan->boundaries.size())));
  }
  if (!phase_boundary_buffers.empty()) {
    requirements->Set(String("spatial_phase_boundary_buffers"), phase_boundary_buffers);
  }
}

void CollectPipelineStageFacts(const tir::PrimFunc& func, Map<String, Any>* requirements) {
  auto stages = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
  if (!stages) {
    return;
  }
  Array<Any> stage_counts;
  Array<Any> loop_vars;
  std::unordered_set<std::string> seen_loop_vars;
  for (const Any& stage_any : stages.value()) {
    auto stage = stage_any.as<Map<String, Any>>();
    if (!stage.has_value()) {
      continue;
    }
    if (auto num_stages = stage.value().Get(String(schema_key::kNumStages))) {
      stage_counts.push_back(Downcast<Integer>(num_stages.value()));
    }
    if (auto loop_var = stage.value().Get(String(schema_key::kLoopVar))) {
      const std::string loop_var_name = Downcast<String>(loop_var.value());
      if (!loop_var_name.empty() && seen_loop_vars.insert(loop_var_name).second) {
        loop_vars.push_back(loop_var.value());
      }
    }
  }
  if (!stage_counts.empty()) {
    requirements->Set(String("pipeline_stage_counts"), stage_counts);
  }
  if (!loop_vars.empty()) {
    requirements->Set(String("pipeline_loop_vars"), loop_vars);
  }
}

}  // namespace

Map<String, Any> BuildBlackholeLoweringRequirements(const tir::PrimFunc& func,
                                                    const SpatialPlan& plan) {
  Map<String, Any> lowering_requirements;
  CollectWorkDecompositionFacts(func, &lowering_requirements);
  CollectPhaseFacts(plan, &lowering_requirements);
  CollectPipelineStageFacts(func, &lowering_requirements);

  LoweringSupportFacts lowering_support_facts = AnalyzeLoweringSupportFacts(func);
  if (!lowering_support_facts.compute_support_payload.empty()) {
    if (auto compute_ops =
            lowering_support_facts.compute_support_payload.Get(String(schema_key::kComputeOpKinds))) {
      SetArrayRequirementIfMissing(&lowering_requirements, String(schema_key::kComputeOpKinds),
                                   Downcast<Array<Any>>(compute_ops.value()));
    }
    if (auto reduction_targets = lowering_support_facts.compute_support_payload.Get(
            String(schema_key::kReductionTargets))) {
      SetArrayRequirementIfMissing(&lowering_requirements,
                                   String(schema_key::kReductionTargets),
                                   Downcast<Array<Any>>(reduction_targets.value()));
    }
    if (auto broadcast_sources = lowering_support_facts.compute_support_payload.Get(
            String(schema_key::kBroadcastSources))) {
      SetArrayRequirementIfMissing(&lowering_requirements,
                                   String(schema_key::kBroadcastSources),
                                   Downcast<Array<Any>>(broadcast_sources.value()));
    }
    if (auto pointwise_ops =
            lowering_support_facts.compute_support_payload.Get(String(schema_key::kPointwiseOpKinds))) {
      SetArrayRequirementIfMissing(&lowering_requirements, String(schema_key::kPointwiseOpKinds),
                                   Downcast<Array<Any>>(pointwise_ops.value()));
    }
    if (auto loop_carried =
            lowering_support_facts.compute_support_payload.Get(String(schema_key::kLoopCarriedState))) {
      SetArrayRequirementIfMissing(&lowering_requirements, String(schema_key::kLoopCarriedState),
                                   Downcast<Array<Any>>(loop_carried.value()));
    }
  }
  if (!lowering_support_facts.buffer_materialization_contracts.empty()) {
    lowering_requirements.Set(String(schema_key::kBufferMaterializationContracts),
                              lowering_support_facts.buffer_materialization_contracts);
  }
  if (!lowering_support_facts.buffer_tile_bridge_specs.empty()) {
    lowering_requirements.Set(String(schema_key::kBufferTileBridgeSpecs),
                              lowering_support_facts.buffer_tile_bridge_specs);
  }
  if (!lowering_support_facts.buffer_flow_contracts.empty()) {
    lowering_requirements.Set(String(schema_key::kBufferFlowContracts),
                              lowering_support_facts.buffer_flow_contracts);
  }
  return lowering_requirements;
}

}  // namespace tl
}  // namespace tvm
