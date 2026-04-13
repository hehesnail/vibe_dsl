/*!
 * \file spatial_program_builder.cc
 * \brief Materialize SpatialExecutionPlan directly from TIR + SpatialPlan facts.
 */

#include "spatial_program_builder.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
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

#include "../op/utils.h"
#include "common/blackhole_utils.h"
#include "common/spatial_analysis.h"
#include "common/spatial_vocab.h"
#include "runtime/thread_storage_scope.h"

namespace tvm {
namespace tl {

using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;
namespace sp = tvm::tl::spatial;

namespace {

struct DomainContract {
  std::string transform_kind = "identity";
  std::string partition_family = "regular";
  sp::SpatialLayoutKind layout_kind = sp::SpatialLayoutKind::kRegular;
  sp::SpatialPartitionKind partition_kind = sp::SpatialPartitionKind::kReplicated;
};

struct TaskRecord {
  std::string name;
  std::string kind;
  std::string phase_name;
  std::string execution_role;
  std::string formation_basis;
  std::vector<std::string> traits;
};

struct ChannelRecord {
  std::string name;
  sp::SpatialChannelKind kind = sp::SpatialChannelKind::kPointToPoint;
  sp::SpatialChannelPayloadKind payload_kind = sp::SpatialChannelPayloadKind::kTensor;
  sp::SpatialChannelDeliveryKind delivery_kind = sp::SpatialChannelDeliveryKind::kOrdered;
  int source_task_index = -1;
  int target_task_index = -1;
  std::string source_task_name;
  std::string target_task_name;
  std::string subject;
  std::vector<std::string> traits;
};

struct FragmentFacts {
  std::unordered_set<std::string> selection_subjects;
  std::unordered_set<std::string> recurrence_subjects;
  std::unordered_set<std::string> reduction_subjects;
  Map<String, Any> fragment_payload;
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
  if (it != logical_buffer_shapes.end() && it->second.size() >= 2U) {
    return it->second.back();
  }
  auto shape = ExtractStaticShape(buffer->shape);
  if (shape && shape.value().size() >= 2U) {
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

bool IsFragmentMaterializationCandidate(const tir::CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
  return tile_op.defined() && tile_op->GetFragmentMaterializationInfo().has_value();
}

Optional<Map<String, Any>> TryBuildFragmentMaterializationContract(const tir::CallNode* call) {
  if (!call) {
    return Optional<Map<String, Any>>();
  }
  TileOperator tile_op = ParseOperator(GetRef<tir::Call>(call));
  if (!tile_op.defined()) {
    return Optional<Map<String, Any>>();
  }
  auto info = tile_op->GetFragmentMaterializationInfo();
  if (!info.has_value()) {
    return Optional<Map<String, Any>>();
  }
  const tir::Buffer& target = info->target_buffer;
  const std::string target_buffer = BufferIdentityName(target);
  const std::string scope = target.scope();
  if (target_buffer.empty() || !IsTrackedStateScope(scope)) {
    return Optional<Map<String, Any>>();
  }
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(fragment_materialization::kIntermediateFragmentMerge));
  contract.Set(String(schema_key::kTargetBuffer), String(target_buffer));
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

Optional<Map<String, Any>> TryBuildRepublishFragmentMaterializationContract(
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
      flow_class != fragment_flow::kRepublish ||
      granule_kind != fragment_flow::kLogicalTile ||
      (!FlowContractHasEventKind(flow_contract, fragment_flow::kComputeConsume) &&
       !FlowContractHasEventKind(flow_contract, fragment_flow::kTransportConsume))) {
    return Optional<Map<String, Any>>();
  }
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(fragment_materialization::kRepublishedLogicalTile));
  contract.Set(String(schema_key::kTargetBuffer), String(buffer_name));
  contract.Set(String(schema_key::kScope), String(scope));
  contract.Set(String(schema_key::kMaterializationKind),
               String(fragment_materialization::kRepublishedBuffer));
  contract.Set(String(schema_key::kBridgeKind),
               String(fragment_materialization::kTileNFacesMaterialization));
  contract.Set(String(schema_key::kValueRole),
               String(fragment_materialization::kConsumerInput));
  contract.Set(String(schema_key::kMergeKind),
               String(fragment_materialization::kDirectWrite));
  contract.Set(String(schema_key::kExecutionProtocol),
               String(fragment_materialization::kTiledCBRepublish));
  contract.Set(String(schema_key::kResultLiveForm), String(fragment_live_form::kTiledCB));
  return contract;
}

Map<String, Any> MakeRepublishedLogicalTileMaterializationContract(
    const std::string& buffer_name, const std::string& scope,
    const std::string& source_buffer = std::string(), int64_t logical_row_width = -1,
    int64_t logical_element_count = -1) {
  Map<String, Any> contract;
  contract.Set(String(schema_key::kKind),
               String(fragment_materialization::kRepublishedLogicalTile));
  contract.Set(String(schema_key::kTargetBuffer), String(buffer_name));
  contract.Set(String(schema_key::kScope), String(scope));
  contract.Set(String(schema_key::kMaterializationKind),
               String(fragment_materialization::kRepublishedBuffer));
  contract.Set(String(schema_key::kBridgeKind),
               String(fragment_materialization::kTileNFacesMaterialization));
  contract.Set(String(schema_key::kValueRole),
               String(fragment_materialization::kConsumerInput));
  contract.Set(String(schema_key::kMergeKind),
               String(fragment_materialization::kDirectWrite));
  contract.Set(String(schema_key::kExecutionProtocol),
               String(fragment_materialization::kTiledCBRepublish));
  contract.Set(String(schema_key::kResultLiveForm), String(fragment_live_form::kTiledCB));
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

enum class FragmentBufferFlowEventKind {
  kWrite,
  kComputeConsume,
  kTransportConsume,
  kReference,
};

struct FragmentBufferFlowEvent {
  int order_index = -1;
  FragmentBufferFlowEventKind kind = FragmentBufferFlowEventKind::kReference;
};

struct FragmentBufferFlowContract {
  std::string buffer_name;
  std::string scope;
  std::string flow_class = fragment_flow::kState;
  int publish_granule = 1;
  int consume_granule = 1;
  std::vector<FragmentBufferFlowEvent> events;
};

std::string FragmentBufferFlowEventKindToString(FragmentBufferFlowEventKind kind) {
  switch (kind) {
    case FragmentBufferFlowEventKind::kWrite:
      return fragment_flow::kWrite;
    case FragmentBufferFlowEventKind::kComputeConsume:
      return fragment_flow::kComputeConsume;
    case FragmentBufferFlowEventKind::kTransportConsume:
      return fragment_flow::kTransportConsume;
    default:
      return fragment_flow::kReference;
  }
}

std::string DeriveFragmentFlowClassLabel(const std::vector<FragmentBufferFlowEvent>& events) {
  int first_consume = std::numeric_limits<int>::max();
  bool has_consume = false;
  for (const FragmentBufferFlowEvent& event : events) {
    if (event.kind == FragmentBufferFlowEventKind::kComputeConsume ||
        event.kind == FragmentBufferFlowEventKind::kTransportConsume) {
      has_consume = true;
      first_consume = std::min(first_consume, event.order_index);
    }
  }
  if (!has_consume) {
    return fragment_flow::kState;
  }
  for (const FragmentBufferFlowEvent& event : events) {
    if (event.kind == FragmentBufferFlowEventKind::kWrite &&
        event.order_index > first_consume) {
      return fragment_flow::kRepublish;
    }
  }
  return fragment_flow::kStream;
}

Array<Any> CollectFragmentBufferFlowContractsFromBody(const tir::Stmt& body) {
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

  std::unordered_map<std::string, FragmentBufferFlowContract> contracts_by_buffer;
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    const tir::Stmt& stmt = ordered_stmts[order_index];
    const auto compute_consumed = CollectComputeConsumedBuffers(stmt);
    for (const auto& [buffer_name, buffer] : tracked_buffers) {
      if (!StmtReferencesBuffer(stmt, buffer)) {
        continue;
      }
      FragmentBufferFlowContract& contract = contracts_by_buffer[buffer_name];
      contract.buffer_name = buffer_name;
      contract.scope = buffer.scope();
      const bool compute_consume = compute_consumed.count(buffer_name);
      const bool transport_consume = StmtConsumesBufferViaTransport(stmt, buffer);
      const bool reads = StmtReadsBuffer(stmt, buffer);
      const bool writes = StmtWritesBuffer(stmt, buffer);
      auto append = [&](FragmentBufferFlowEventKind kind) {
        contract.events.push_back(FragmentBufferFlowEvent{order_index, kind});
      };
      if (compute_consume) {
        append(FragmentBufferFlowEventKind::kComputeConsume);
      }
      if (!compute_consume && reads && !transport_consume) {
        append(FragmentBufferFlowEventKind::kReference);
      }
      if (transport_consume) {
        append(FragmentBufferFlowEventKind::kTransportConsume);
      }
      if (writes) {
        append(FragmentBufferFlowEventKind::kWrite);
      }
      if (!compute_consume && !reads && !transport_consume && !writes) {
        append(FragmentBufferFlowEventKind::kReference);
      }
    }
  }

  Array<Any> contracts;
  for (auto& [buffer_name, contract] : contracts_by_buffer) {
    if (contract.events.empty()) {
      continue;
    }
    contract.flow_class = DeriveFragmentFlowClassLabel(contract.events);
    Map<String, Any> encoded;
    encoded.Set(String(schema_key::kBuffer), String(buffer_name));
    encoded.Set(String(schema_key::kScope), String(contract.scope));
    encoded.Set(String(schema_key::kFlowClass), String(contract.flow_class));
    encoded.Set(String(schema_key::kGranuleKind), String(fragment_flow::kLogicalTile));
    encoded.Set(String(schema_key::kPublishGranule), Integer(contract.publish_granule));
    encoded.Set(String(schema_key::kConsumeGranule), Integer(contract.consume_granule));
    Array<Any> events;
    for (const FragmentBufferFlowEvent& event : contract.events) {
      Map<String, Any> encoded_event;
      encoded_event.Set(String(schema_key::kKind),
                        String(FragmentBufferFlowEventKindToString(event.kind)));
      encoded_event.Set(String(schema_key::kOrderIndex), Integer(event.order_index));
      events.push_back(encoded_event);
    }
    encoded.Set(String(schema_key::kEvents), events);
    contracts.push_back(encoded);
  }
  return contracts;
}

void AppendUniqueFragmentMaterializationContractsFromFlowContracts(
    const Array<Any>& flow_contracts, Array<Any>* materialization_contracts) {
  std::unordered_set<std::string> seen;
  for (const Any& contract_any : *materialization_contracts) {
    Map<String, Any> contract = Downcast<Map<String, Any>>(contract_any);
    seen.insert(Downcast<String>(contract.at(String(schema_key::kTargetBuffer))) + "|" +
                Downcast<String>(contract.at(String(schema_key::kScope))));
  }
  for (const Any& flow_contract_any : flow_contracts) {
    Map<String, Any> flow_contract = Downcast<Map<String, Any>>(flow_contract_any);
    auto maybe_contract = TryBuildRepublishFragmentMaterializationContract(flow_contract);
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

void AppendUniqueCastDrivenFragmentMaterializationContractsFromBody(
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
    if (!IsTrackedStateScope(scope) || granule_kind != fragment_flow::kLogicalTile ||
        (!FlowContractHasEventKind(flow_contract, fragment_flow::kComputeConsume) &&
         !FlowContractHasEventKind(flow_contract, fragment_flow::kTransportConsume))) {
      return;
    }
    int64_t logical_row_width = GetLogicalRowWidth(src_buffer, logical_buffer_shapes);
    if (logical_row_width <= 0) {
      logical_row_width = GetLogicalRowWidth(dst_buffer, logical_buffer_shapes);
    }
    int64_t logical_element_count = GetLogicalElementCount(dst_name, logical_buffer_shapes);
    if (logical_element_count <= 0) {
      logical_element_count = GetLogicalElementCount(src_name, logical_buffer_shapes);
    }
    upsert(MakeRepublishedLogicalTileMaterializationContract(
        dst_name, scope, src_name, logical_row_width, logical_element_count));
  });
}

Array<Any> CollectFragmentMaterializationContractsFromBody(const tir::Stmt& body) {
  Array<Any> contracts;
  std::unordered_set<std::string> seen;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<tir::CallNode>();
    if (!IsFragmentMaterializationCandidate(call)) {
      return;
    }
    auto maybe_contract = TryBuildFragmentMaterializationContract(call);
    if (!maybe_contract) {
      return;
    }
    const std::string key =
        Downcast<String>(maybe_contract.value().at(String(schema_key::kTargetBuffer))) + "|" +
        Downcast<String>(maybe_contract.value().at(String(schema_key::kScope)));
    if (seen.insert(key).second) {
      contracts.push_back(maybe_contract.value());
    }
  });
  return contracts;
}

Array<String> GetAxesFromWorkDecomposition(const tir::PrimFunc& func) {
  Array<String> axes;
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  if (!work) {
    return axes;
  }
  auto maybe_axes = work.value().Get(String("axes"));
  if (!maybe_axes) {
    return axes;
  }
  for (const Any& axis_any : Downcast<Array<Any>>(maybe_axes.value())) {
    axes.push_back(Downcast<String>(axis_any));
  }
  return axes;
}

bool HasNonEmptyArrayField(const Map<String, Any>& payload, const char* key) {
  auto maybe_value = payload.Get(String(key));
  return maybe_value && !Downcast<Array<Any>>(maybe_value.value()).empty();
}

bool WorkDecompositionHasDerivedIndices(const tir::PrimFunc& func) {
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  return work && HasNonEmptyArrayField(work.value(), "derived_index_exprs");
}

bool WorkDecompositionHasWorkDependentBounds(const tir::PrimFunc& func) {
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  return work && HasNonEmptyArrayField(work.value(), "work_dependent_loop_bounds");
}

Array<Any> GetPipelineStages(const tir::PrimFunc& func) {
  return func->GetAttr<Array<Any>>("blackhole.pipeline_stages").value_or(Array<Any>());
}

Array<Any> GetFragmentRegions(const tir::PrimFunc& func) {
  return func->GetAttr<Array<Any>>("blackhole.fragment_regions").value_or(Array<Any>());
}

std::string GetCopySemanticsField(const Map<String, Any>& ann, const char* key,
                                  const std::string& default_value = "") {
  if (auto value = ann.Get(String(key))) {
    if (auto maybe_string = value.value().try_cast<String>()) {
      return maybe_string.value();
    }
  }
  return default_value;
}

std::vector<std::string> CollectSegmentKindsFromBody(const tir::Stmt& body) {
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

bool FragmentFlowContractHasEventKind(const Map<String, Any>& flow_contract, const char* kind) {
  return FlowContractHasEventKind(flow_contract, kind);
}

Map<String, Any> EmptyPayload() { return Map<String, Any>(); }

Array<Any> ToIntegerAnyArray(const std::vector<int>& values) {
  Array<Any> result;
  for (int value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

const char* SelectChannelKind(const SpatialCapabilityModel& capability_model,
                              sp::SpatialChannelKind channel_kind) {
  const char* kind = sp::ToString(channel_kind);
  ICHECK(ContainsKind(capability_model->supported_flow_kinds, kind));
  return kind;
}

const char* SelectChannelPayloadKind(const SpatialCapabilityModel& capability_model,
                                     sp::SpatialChannelPayloadKind payload_kind) {
  const char* kind = sp::ToString(payload_kind);
  ICHECK(ContainsKind(capability_model->supported_payload_kinds, kind));
  return kind;
}

const char* SelectChannelDeliveryKind(const SpatialCapabilityModel& capability_model,
                                      sp::SpatialChannelDeliveryKind delivery_kind) {
  const char* kind = sp::ToString(delivery_kind);
  ICHECK(ContainsKind(capability_model->supported_delivery_kinds, kind));
  return kind;
}

const char* SelectLayoutKind(const SpatialCapabilityModel& capability_model,
                             const DomainContract& contract) {
  const char* kind = sp::ToString(contract.layout_kind);
  ICHECK(ContainsKind(capability_model->supported_layout_kinds, kind));
  return kind;
}

const char* SelectPartitionKind(const SpatialCapabilityModel& capability_model,
                                const DomainContract& contract) {
  const char* kind = sp::ToString(contract.partition_kind);
  ICHECK(ContainsKind(capability_model->supported_partition_kinds, kind));
  return kind;
}

const char* NeutralPlacementAffinityForExecutionRole(const std::string& execution_role) {
  if (execution_role == "ingress" || execution_role == "tile_ingress") {
    return "ingress";
  }
  if (execution_role == "egress" || execution_role == "tile_egress") {
    return "egress";
  }
  return "compute";
}

DomainContract DeriveDomainContract(const tir::PrimFunc& func, const Array<String>& axes,
                                    const FragmentFacts& fragment_facts) {
  const bool has_derived = WorkDecompositionHasDerivedIndices(func);
  const bool has_bounds = WorkDecompositionHasWorkDependentBounds(func);
  const bool has_selection = !fragment_facts.selection_subjects.empty();
  const bool multi_axis = axes.size() > 1;
  DomainContract contract;
  if (has_derived) {
    contract.transform_kind = has_bounds ? "paged" : "derived";
    contract.partition_family = has_bounds ? "paged" : "derived";
    contract.layout_kind = sp::SpatialLayoutKind::kIndexed;
    contract.partition_kind = sp::SpatialPartitionKind::kIndexed;
    return contract;
  }
  if (has_selection) {
    contract.transform_kind = "filtered";
    contract.partition_family = "filtered";
    contract.partition_kind = sp::SpatialPartitionKind::kFiltered;
    return contract;
  }
  contract.partition_kind =
      multi_axis ? sp::SpatialPartitionKind::kBlocked : sp::SpatialPartitionKind::kReplicated;
  return contract;
}

Map<String, Any> BuildDomainPayload(int domain_index, const DomainContract& contract) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kDomainIndex), Integer(domain_index));
  payload.Set(String(schema_key::kDomainTransformKind), String(contract.transform_kind));
  return payload;
}

Map<String, Any> BuildWorkPartitionPayload(const tir::PrimFunc& func, int domain_index,
                                           const DomainContract& contract) {
  Map<String, Any> payload = BuildDomainPayload(domain_index, contract);
  payload.Set(String(schema_key::kPartitionFamily), String(contract.partition_family));
  auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition");
  if (work) {
    auto maybe_loop_bounds = work.value().Get(String("work_dependent_loop_bounds"));
    if (maybe_loop_bounds) {
      payload.Set(String(schema_key::kWorkDependentLoopBounds),
                  Downcast<Array<Any>>(maybe_loop_bounds.value()));
    }
  }
  return payload;
}

Map<String, Any> BuildTaskPayload(int phase_index, const TaskRecord& record) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kPhaseIndex), Integer(phase_index));
  payload.Set(String(schema_key::kExecutionRole), String(record.execution_role));
  payload.Set(String(schema_key::kFormationBasis), String(record.formation_basis));
  return payload;
}

Map<String, Any> BuildChannelPayload(const ChannelRecord& record, const char* payload_kind,
                                     const char* delivery_kind) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kSourceTaskIndex), Integer(record.source_task_index));
  payload.Set(String(schema_key::kTargetTaskIndex), Integer(record.target_task_index));
  payload.Set(String(schema_key::kPayloadKind), String(payload_kind));
  payload.Set(String(schema_key::kDeliveryKind), String(delivery_kind));
  return payload;
}

Map<String, Any> BuildPlacementPayload(int task_index, const char* affinity_kind) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTaskIndex), Integer(task_index));
  payload.Set(String(schema_key::kAffinityKind), String(affinity_kind));
  payload.Set(String(schema_key::kObligationKind), String("execution"));
  payload.Set(String(schema_key::kPlacementDomain), String("logical_worker_grid"));
  return payload;
}

Map<String, Any> BuildSyncEdgePayload(int source_task_index, int target_task_index,
                                      const std::string& ordering_kind,
                                      const std::string& materialization_kind) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kSourceTaskIndex), Integer(source_task_index));
  payload.Set(String(schema_key::kTargetTaskIndex), Integer(target_task_index));
  payload.Set(String(schema_key::kOrderingKind), String(ordering_kind));
  payload.Set(String(schema_key::kMaterializationKind), String(materialization_kind));
  return payload;
}

Map<String, Any> BuildProgramPhasePayload(int phase_index, const std::vector<int>& task_indices,
                                          const std::vector<int>& channel_indices,
                                          const std::string& closure_basis) {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kPhaseIndex), Integer(phase_index));
  payload.Set(String(schema_key::kTaskIndices), ToIntegerAnyArray(task_indices));
  payload.Set(String(schema_key::kChannelIndices), ToIntegerAnyArray(channel_indices));
  payload.Set(String(schema_key::kClosureBasis), String(closure_basis));
  return payload;
}

Map<String, Any> BuildMemberFuncTargetPayload() {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTargetKind), String(spatial_contract::kMemberFuncTarget));
  return payload;
}

Map<String, Any> BuildBufferTargetPayload() {
  Map<String, Any> payload;
  payload.Set(String(schema_key::kTargetKind), String(spatial_contract::kBufferTarget));
  return payload;
}

Task MakeSpatialTask(const TaskRecord& record, int phase_index) {
  return Task(String(record.name), String(record.kind), String(record.phase_name), Array<String>{},
              ToStringArray(record.traits), BuildTaskPayload(phase_index, record),
              MakeAnchors("spatial_task", record.name));
}

Placement MakeExecutionPlacement(const TaskRecord& record, int task_index,
                                 const std::string& member_func) {
  const char* affinity = NeutralPlacementAffinityForExecutionRole(record.execution_role);
  return Placement(String("place_" + record.name),
                   String(sp::ToString(sp::SpatialPlacementKind::kExecution)),
                   String(record.name), String(member_func), MakeTraits({"phase_b"}),
                   BuildPlacementPayload(task_index, affinity),
                   MakeAnchors("spatial_placement", record.name));
}

Channel MakeSpatialChannel(const ChannelRecord& record, const SpatialCapabilityModel& capability_model) {
  const char* channel_kind = SelectChannelKind(capability_model, record.kind);
  const char* payload_kind = SelectChannelPayloadKind(capability_model, record.payload_kind);
  const char* delivery_kind = SelectChannelDeliveryKind(capability_model, record.delivery_kind);
  Array<String> traits = ToStringArray(record.traits);
  return Channel(String(record.name), String(channel_kind), String(record.source_task_name),
                 String(record.target_task_name), String(record.subject), std::move(traits),
                 BuildChannelPayload(record, payload_kind, delivery_kind),
                 MakeAnchors("spatial_channel", record.name));
}

ProgramPhase MakeProgramPhaseNode(const std::string& phase_name,
                                  const std::vector<std::string>& task_names,
                                  const std::vector<std::string>& channel_names,
                                  int phase_index, const std::vector<int>& task_indices,
                                  const std::vector<int>& channel_indices,
                                  const std::string& closure_basis) {
  return ProgramPhase(String(phase_name), ToStringArray(task_names), ToStringArray(channel_names),
                      MakeTraits({"phase_b"}),
                      BuildProgramPhasePayload(phase_index, task_indices, channel_indices,
                                               closure_basis),
                      MakeAnchors("spatial_phase", phase_name));
}

SyncEdge MakeCompletionSyncEdge(const std::string& name, const std::string& source,
                                const std::string& target, int source_task_index,
                                int target_task_index, const std::string& ordering_kind,
                                const std::string& materialization_kind) {
  return SyncEdge(String(name), String(sp::ToString(sp::SpatialSyncKind::kCompletion)),
                  String(source), String(target),
                  MakeTraits({"phase_boundary", "graph_ordered"}),
                  BuildSyncEdgePayload(source_task_index, target_task_index, ordering_kind,
                                       materialization_kind),
                  MakeAnchors("spatial_sync", name));
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

FragmentFacts AnalyzeFragmentFacts(const tir::PrimFunc& func) {
  FragmentFacts facts;
  Array<Any> fragment_ops;
  Array<Any> pointwise_ops;
  Array<Any> row_reduction_targets;
  Array<Any> row_broadcast_sources;
  Array<Any> fragment_loop_carried_state;
  Array<Any> fragment_layout_contracts;
  std::unordered_set<std::string> seen_fragment_ops;
  std::unordered_set<std::string> seen_pointwise_ops;
  std::unordered_set<std::string> seen_reduction_targets;
  std::unordered_set<std::string> seen_row_broadcasts;
  std::unordered_set<std::string> seen_loop_carried;
  std::unordered_set<std::string> seen_layout_contracts;

  for (const Any& region_any : GetFragmentRegions(func)) {
    Map<String, Any> region = Downcast<Map<String, Any>>(region_any);
    if (auto maybe_ops = region.Get(String("ops"))) {
      for (const Any& op_any : Downcast<Array<Any>>(maybe_ops.value())) {
        const std::string name = Downcast<String>(op_any);
        PushBackUnique(&fragment_ops, &seen_fragment_ops, String(name), name);
      }
    }
    if (auto maybe_ops = region.Get(String("pointwise_ops"))) {
      for (const Any& op_any : Downcast<Array<Any>>(maybe_ops.value())) {
        const std::string name = Downcast<String>(op_any);
        PushBackUnique(&pointwise_ops, &seen_pointwise_ops, String(name), name);
      }
    }
    if (auto maybe_targets = region.Get(String(manifest_key::kRowReductions))) {
      for (const Any& target_any : Downcast<Array<Any>>(maybe_targets.value())) {
        Map<String, Any> target = Downcast<Map<String, Any>>(target_any);
        const std::string name = KeyForAnyMap(target, schema_key::kTarget);
        if (!name.empty()) {
          facts.reduction_subjects.insert(name);
          PushBackUnique(&row_reduction_targets, &seen_reduction_targets, target, name);
        }
      }
    }
    if (auto maybe_sources = region.Get(String("row_broadcasts"))) {
      for (const Any& source_any : Downcast<Array<Any>>(maybe_sources.value())) {
        Map<String, Any> source = Downcast<Map<String, Any>>(source_any);
        const std::string name = KeyForAnyMap(source, schema_key::kSource);
        PushBackUnique(&row_broadcast_sources, &seen_row_broadcasts, source, name);
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
          PushBackUnique(&fragment_loop_carried_state, &seen_loop_carried, state, name);
        }
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
    if (auto maybe_layouts = region.Get(String(schema_key::kFragmentLayoutContracts))) {
      for (const Any& layout_any : Downcast<Array<Any>>(maybe_layouts.value())) {
        Map<String, Any> layout = Downcast<Map<String, Any>>(layout_any);
        const std::string buffer = KeyForAnyMap(layout, schema_key::kBuffer);
        PushBackUnique(&fragment_layout_contracts, &seen_layout_contracts, layout, buffer);
      }
    }
  }

  Array<Any> flow_contracts = CollectFragmentBufferFlowContractsFromBody(func->body);
  Array<Any> materialization_contracts = CollectFragmentMaterializationContractsFromBody(func->body);
  AppendUniqueFragmentMaterializationContractsFromFlowContracts(flow_contracts,
                                                                &materialization_contracts);
  AppendUniqueCastDrivenFragmentMaterializationContractsFromBody(
      func->body, flow_contracts, BuildLogicalBufferShapes(func), &materialization_contracts);

  if (!fragment_ops.empty()) {
    facts.fragment_payload.Set(String(schema_key::kFragmentOpKinds), fragment_ops);
  }
  if (!pointwise_ops.empty()) {
    facts.fragment_payload.Set(String(schema_key::kPointwiseOpKinds), pointwise_ops);
  }
  if (!row_reduction_targets.empty()) {
    facts.fragment_payload.Set(String(schema_key::kRowReductionTargets), row_reduction_targets);
  }
  if (!row_broadcast_sources.empty()) {
    facts.fragment_payload.Set(String(schema_key::kRowBroadcastSources), row_broadcast_sources);
  }
  if (!fragment_loop_carried_state.empty()) {
    facts.fragment_payload.Set(String(schema_key::kFragmentLoopCarriedState),
                               fragment_loop_carried_state);
  }
  if (!fragment_layout_contracts.empty()) {
    facts.fragment_payload.Set(String(schema_key::kFragmentLayoutContracts),
                               fragment_layout_contracts);
  }
  if (!materialization_contracts.empty()) {
    facts.fragment_payload.Set(String(schema_key::kFragmentMaterializationContracts),
                               materialization_contracts);
  }
  if (!flow_contracts.empty()) {
    facts.fragment_payload.Set(String(schema_key::kFragmentBufferFlowContracts), flow_contracts);
  }
  return facts;
}

bool HasFragmentContract(const FragmentFacts& fragment_facts) {
  auto maybe_fragment_ops = fragment_facts.fragment_payload.Get(String(schema_key::kFragmentOpKinds));
  return maybe_fragment_ops && !Downcast<Array<Any>>(maybe_fragment_ops.value()).empty();
}

std::vector<std::string> DeriveFragmentFastPathSegmentKinds(const tir::PrimFunc& func,
                                                            const FragmentFacts& fragment_facts) {
  std::vector<std::string> kinds = CollectSegmentKindsFromBody(func->body);
  if (std::find(kinds.begin(), kinds.end(), "compute") != kinds.end()) {
    return kinds;
  }
  if (!HasFragmentContract(fragment_facts)) {
    return {};
  }
  bool has_reader = false;
  bool has_writer = false;
  tir::PostOrderVisit(func->body, [&](const ObjectRef& node) {
    const auto* loop = node.as<tir::ForNode>();
    if (!loop) {
      return;
    }
    auto ann = loop->annotations.Get(String("blackhole.copy_semantics"));
    if (!ann) {
      return;
    }
    Map<String, Any> ann_map = ann.value().as<Map<String, Any>>().value_or(Map<String, Any>());
    const std::string direction = GetCopySemanticsField(ann_map, schema_key::kDirection);
    const std::string kind = GetCopySemanticsField(ann_map, schema_key::kKind);
    has_reader = has_reader || direction == "dram_to_cb" || kind == "fused_staged_copy";
    has_writer = has_writer || direction == "cb_to_dram";
  });
  if (auto maybe_flows =
          fragment_facts.fragment_payload.Get(String(schema_key::kFragmentBufferFlowContracts))) {
    for (const Any& flow_any : Downcast<Array<Any>>(maybe_flows.value())) {
      Map<String, Any> flow = Downcast<Map<String, Any>>(flow_any);
      const std::string scope = KeyForAnyMap(flow, schema_key::kScope);
      if (!has_reader && !IsTrackedStateScope(scope) &&
          FragmentFlowContractHasEventKind(flow, fragment_flow::kComputeConsume)) {
        has_reader = true;
      }
      if (!has_writer &&
          FragmentFlowContractHasEventKind(flow, fragment_flow::kTransportConsume)) {
        has_writer = true;
      }
    }
  }
  std::vector<std::string> result;
  if (has_reader) result.push_back("reader");
  result.push_back("compute");
  if (has_writer) result.push_back("writer");
  return result;
}

bool HasSimpleSegmentKinds(const tir::PrimFunc& func, const std::vector<std::string>& expected) {
  return CollectSegmentKindsFromBody(func->body) == expected;
}

bool IsSimpleCopyFastPath(const SpatialPlan& plan, const tir::PrimFunc& func,
                          const FragmentFacts& fragment_facts) {
  if (HasFragmentContract(fragment_facts) || HasSimpleSegmentKinds(func, {"reader", "compute", "writer"})) {
    return false;
  }
  return plan->closures.size() == 2 && plan->boundaries.size() <= 2;
}

bool IsSimpleGemmFastPath(const SpatialPlan& plan, const tir::PrimFunc& func) {
  if (!HasSimpleSegmentKinds(func, {"reader", "compute", "writer"})) {
    return false;
  }
  if (plan->closures.size() != 4 || plan->boundaries.size() != 4) {
    return false;
  }
  int ingress_count = 0;
  int compute_count = 0;
  int egress_count = 0;
  for (const ExecutionClosure& closure : plan->closures) {
    const std::string role = closure->execution_role;
    ingress_count += role == "ingress";
    compute_count += role == "compute";
    egress_count += role == "egress";
  }
  return ingress_count == 2 && compute_count == 1 && egress_count == 1;
}

bool IsSimpleFragmentComputeFastPath(const SpatialPlan& plan, const tir::PrimFunc& func,
                                     const FragmentFacts& fragment_facts) {
  if (IsSimpleGemmFastPath(plan, func)) {
    return false;
  }
  if (plan->closures.size() > 4 || plan->boundaries.size() > 6) {
    return false;
  }
  return !DeriveFragmentFastPathSegmentKinds(func, fragment_facts).empty();
}

void BuildCommonSpatialScaffolding(const std::string& member_func, const tir::PrimFunc& func,
                                   const SpatialCapabilityModel& capability_model,
                                   const FragmentFacts& fragment_facts,
                                   Array<SpatialLayout>* layouts,
                                   Array<WorkPartition>* work_partitions) {
  const Array<String> axes = GetAxesFromWorkDecomposition(func);
  const DomainContract contract = DeriveDomainContract(func, axes, fragment_facts);
  layouts->push_back(SpatialLayout(
      String("layout_" + member_func), String(SelectLayoutKind(capability_model, contract)),
      String(member_func), axes, MakeTraits({"phase_b"}),
      BuildDomainPayload(0, contract), MakeAnchors("spatial_layout", member_func)));
  work_partitions->push_back(WorkPartition(
      String("partition_" + member_func), String(SelectPartitionKind(capability_model, contract)),
      String(member_func), axes, MakeTraits({"phase_b"}),
      BuildWorkPartitionPayload(func, 0, contract),
      MakeAnchors("spatial_partition", member_func)));
}

void AppendPipelineResourceIntent(const std::string& member_func, const tir::PrimFunc& func,
                                  Array<ResourceIntent>* resource_intents) {
  Array<Any> stages = GetPipelineStages(func);
  if (stages.empty()) {
    return;
  }
  Map<String, Any> payload = BuildMemberFuncTargetPayload();
  payload.Set(String(schema_key::kPipelineStages), stages);
  resource_intents->push_back(ResourceIntent(
      String("pipeline_contract_" + member_func),
      String(sp::ToString(sp::SpatialResourceIntentKind::kSynchronizationSupport)),
      String(member_func), MakeTraits({"phase_b", "pipeline_contract"}), payload,
      MakeAnchors("spatial_resource_intent", "pipeline_contract_" + member_func)));
}

void AppendFragmentResourceIntent(const std::string& member_func,
                                  const FragmentFacts& fragment_facts,
                                  Array<ResourceIntent>* resource_intents) {
  if (fragment_facts.fragment_payload.empty()) {
    return;
  }
  Map<String, Any> payload = fragment_facts.fragment_payload;
  payload.Set(String(schema_key::kTargetKind), String(spatial_contract::kMemberFuncTarget));
  resource_intents->push_back(ResourceIntent(
      String("fragment_contract_" + member_func),
      String(sp::ToString(sp::SpatialResourceIntentKind::kLoweringSupport)),
      String(member_func), MakeTraits({"phase_b", "fragment_contract"}), payload,
      MakeAnchors("spatial_resource_intent", "fragment_contract_" + member_func)));
}

void AppendPhaseBoundaryResourceIntents(const std::vector<ChannelRecord>& channel_records,
                                        Array<ResourceIntent>* resource_intents) {
  std::unordered_set<std::string> seen_subjects;
  for (const ChannelRecord& record : channel_records) {
    if (record.delivery_kind != sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized ||
        record.subject.empty() || !seen_subjects.insert(record.subject).second) {
      continue;
    }
    resource_intents->push_back(ResourceIntent(
        String("phase_boundary_" + record.subject),
        String(sp::ToString(sp::SpatialResourceIntentKind::kPhaseBoundaryMaterialization)),
        String(record.subject), MakeTraits({"phase_boundary"}), BuildBufferTargetPayload(),
        MakeAnchors("spatial_resource_intent", "phase_boundary_" + record.subject)));
  }
}

SpatialExecutionPlan BuildCopyFastPath(const std::string& member_func, const tir::PrimFunc& func,
                                       const SpatialCapabilityModel& capability_model,
                                       const FragmentFacts& fragment_facts) {
  TaskRecord record{"copy", sp::ToString(sp::SpatialTaskKind::kTransfer), "phase0_copy",
                    "transfer_copy", "fast_path|segment=copy|boundary=tensor_transfer",
                    {"fast_path", "copy"}};
  Array<Task> tasks{MakeSpatialTask(record, 0)};
  ChannelRecord channel_record{"copy_tensor", sp::SpatialChannelKind::kPointToPoint,
                               sp::SpatialChannelPayloadKind::kTensor,
                               sp::SpatialChannelDeliveryKind::kBufferedAsync, 0, 0, "copy",
                               "copy", "tensor_transfer", {"fast_path", "copy"}};
  Array<Channel> channels{MakeSpatialChannel(channel_record, capability_model)};
  Array<ProgramPhase> phases{MakeProgramPhaseNode("phase0_copy", {"copy"}, {"copy_tensor"}, 0, {0},
                                                  {0}, "single_phase_fast_path|closure_copy")};
  Array<Placement> placements{MakeExecutionPlacement(record, 0, member_func)};
  Array<SyncEdge> sync_edges;
  Array<ResourceIntent> resource_intents;
  AppendFragmentResourceIntent(member_func, fragment_facts, &resource_intents);
  AppendPipelineResourceIntent(member_func, func, &resource_intents);
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  BuildCommonSpatialScaffolding(member_func, func, capability_model, fragment_facts, &layouts,
                                &work_partitions);
  return SpatialExecutionPlan(String(member_func), phases, tasks, channels, placements, sync_edges,
                              resource_intents, MakeAnchors("spatial_execution_plan", member_func));
}

SpatialExecutionPlan BuildSegmentFastPath(const std::string& member_func, const tir::PrimFunc& func,
                                          const SpatialCapabilityModel& capability_model,
                                          const FragmentFacts& fragment_facts,
                                          const std::vector<std::string>& segment_kinds,
                                          const std::vector<std::string>& task_traits,
                                          const std::string& phase_name,
                                          const std::string& compute_role) {
  Array<Task> tasks;
  Array<Placement> placements;
  std::vector<TaskRecord> task_records;
  for (const std::string& segment_name : segment_kinds) {
    const std::string execution_role =
        segment_name == "reader" ? "tile_ingress"
        : segment_name == "writer" ? "tile_egress"
                                   : compute_role;
    const std::string formation_basis =
        "fast_path|segment=" + segment_name + "|boundary=" +
        (segment_name == "reader" ? "tensor_transfer"
         : segment_name == "writer" ? "completion_handoff"
                                    : "fragment_dataflow");
    TaskRecord record{segment_name,
                      segment_name == "compute" ? sp::ToString(sp::SpatialTaskKind::kCompute)
                                                : sp::ToString(sp::SpatialTaskKind::kTransfer),
                      phase_name,
                      execution_role,
                      formation_basis,
                      task_traits};
    task_records.push_back(record);
    tasks.push_back(MakeSpatialTask(record, 0));
    placements.push_back(MakeExecutionPlacement(record, task_records.size() - 1, member_func));
  }

  Array<Channel> channels;
  Array<SyncEdge> sync_edges;
  std::vector<std::string> channel_names;
  for (size_t i = 0; i + 1 < task_records.size(); ++i) {
    ChannelRecord record{task_records[i].name + "_to_" + task_records[i + 1].name,
                         sp::SpatialChannelKind::kPointToPoint,
                         sp::SpatialChannelPayloadKind::kTensor,
                         task_records[i].name == "compute"
                             ? sp::SpatialChannelDeliveryKind::kCompletionVisible
                             : sp::SpatialChannelDeliveryKind::kBufferedAsync,
                         static_cast<int>(i),
                         static_cast<int>(i + 1),
                         task_records[i].name,
                         task_records[i + 1].name,
                         task_records[i].name + "_boundary",
                         task_traits};
    channel_names.push_back(record.name);
    channels.push_back(MakeSpatialChannel(record, capability_model));
    sync_edges.push_back(MakeCompletionSyncEdge(
        record.name + "_sync", record.source_task_name, record.target_task_name,
        record.source_task_index, record.target_task_index,
        DeriveOrderingKindForChannel(record.kind, record.delivery_kind),
        DeriveMaterializationKindForChannel(record.kind, record.delivery_kind)));
  }
  Array<ProgramPhase> phases{MakeProgramPhaseNode(
      phase_name, [&]() {
        std::vector<std::string> names;
        for (const TaskRecord& record : task_records) names.push_back(record.name);
        return names;
      }(), channel_names, 0, [&]() {
        std::vector<int> indices;
        for (int i = 0; i < static_cast<int>(task_records.size()); ++i) indices.push_back(i);
        return indices;
      }(), [&]() {
        std::vector<int> indices;
        for (int i = 0; i < static_cast<int>(channel_names.size()); ++i) indices.push_back(i);
        return indices;
      }(), "segment_graph_closure|single_phase_fast_path")};
  Array<ResourceIntent> resource_intents;
  AppendFragmentResourceIntent(member_func, fragment_facts, &resource_intents);
  AppendPipelineResourceIntent(member_func, func, &resource_intents);
  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  BuildCommonSpatialScaffolding(member_func, func, capability_model, fragment_facts, &layouts,
                                &work_partitions);
  return SpatialExecutionPlan(String(member_func), phases, tasks, channels, placements, sync_edges,
                              resource_intents, MakeAnchors("spatial_execution_plan", member_func));
}

std::vector<int> ComputeTaskPhases(const SpatialPlan& plan) {
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

std::unordered_map<std::string, int> CountSubjectConsumers(const SpatialPlan& plan) {
  std::unordered_map<std::string, std::unordered_set<int64_t>> consumers;
  for (const ClosureBoundary& boundary : plan->boundaries) {
    if (boundary->target_closure_index >= 0) {
      consumers[boundary->subject].insert(boundary->target_closure_index);
    }
  }
  std::unordered_map<std::string, int> counts;
  for (const auto& [subject, targets] : consumers) {
    counts[subject] = static_cast<int>(targets.size());
  }
  return counts;
}

std::string BuildPhaseClosureBasis(const std::string& phase_name, int task_count,
                                   const std::vector<std::string>& ordering_kinds) {
  std::string basis = "spatial_boundary_graph|phase=" + phase_name +
                      "|task_count=" + std::to_string(task_count) + "|ordering_basis=";
  for (int i = 0; i < static_cast<int>(ordering_kinds.size()); ++i) {
    if (i != 0) basis += ",";
    basis += ordering_kinds[i];
  }
  return basis;
}

SpatialExecutionPlan BuildGenericSpatialProgram(const std::string& member_func, const SpatialPlan& plan,
                                                const tir::PrimFunc& func,
                                                const SpatialCapabilityModel& capability_model,
                                                const FragmentFacts& fragment_facts) {
  const std::vector<int> task_phase = ComputeTaskPhases(plan);
  const auto subject_consumer_count = CountSubjectConsumers(plan);

  std::vector<TaskRecord> task_records;
  task_records.reserve(plan->closures.size());
  for (int i = 0; i < plan->closures.size(); ++i) {
    const ExecutionClosure& closure = plan->closures[i];
    std::vector<std::string> traits{"phase_b"};
    for (const String& trait : closure->traits) {
      PushBackUnique(&traits, static_cast<std::string>(trait));
    }
    for (const ClosureBoundary& boundary : plan->boundaries) {
      if (boundary->source_closure_index != i && boundary->target_closure_index != i) {
        continue;
      }
      const std::string subject = boundary->subject;
      if (fragment_facts.selection_subjects.count(subject)) {
        PushBackUnique(&traits, std::string("select"));
      }
      if (str(boundary->kind) == "carry" || fragment_facts.recurrence_subjects.count(subject)) {
        PushBackUnique(&traits, std::string("recurrence"));
      }
      if (str(boundary->kind) == "join" || fragment_facts.reduction_subjects.count(subject)) {
        PushBackUnique(&traits, std::string("reduce"));
      }
    }
    std::string task_kind = sp::ToString(sp::SpatialTaskKind::kTransfer);
    if (closure->execution_role == "compute") {
      task_kind = HasTrait(ToStringArray(traits), "reduce")
                      ? sp::ToString(sp::SpatialTaskKind::kCollective)
                      : HasTrait(ToStringArray(traits), "select") ||
                                HasTrait(ToStringArray(traits), "recurrence")
                            ? sp::ToString(sp::SpatialTaskKind::kControl)
                            : sp::ToString(sp::SpatialTaskKind::kCompute);
    }
    task_records.push_back(TaskRecord{
        static_cast<std::string>(closure->name),
        task_kind,
        "phase_" + std::to_string(task_phase[i]),
        static_cast<std::string>(closure->execution_role),
        "spatial_plan_closure|basis=" + static_cast<std::string>(closure->closure_basis),
        std::move(traits),
    });
  }

  std::vector<ChannelRecord> channel_records;
  channel_records.reserve(plan->boundaries.size());
  for (const ClosureBoundary& boundary : plan->boundaries) {
    if (boundary->source_closure_index < 0 || boundary->target_closure_index < 0) {
      continue;
    }
    const std::string subject = boundary->subject;
    sp::SpatialChannelKind kind = sp::SpatialChannelKind::kPointToPoint;
    if (str(boundary->kind) == "carry" || fragment_facts.recurrence_subjects.count(subject)) {
      kind = sp::SpatialChannelKind::kCarry;
    } else if (str(boundary->kind) == "join" || fragment_facts.reduction_subjects.count(subject)) {
      kind = sp::SpatialChannelKind::kReduceMerge;
    } else if (fragment_facts.selection_subjects.count(subject)) {
      kind = sp::SpatialChannelKind::kGather;
    } else if (subject_consumer_count.count(subject) && subject_consumer_count.at(subject) > 1) {
      kind = sp::SpatialChannelKind::kBroadcast;
    }
    const bool cross_phase = task_phase[boundary->source_closure_index] !=
                             task_phase[boundary->target_closure_index];
    sp::SpatialChannelDeliveryKind delivery_kind =
        cross_phase ? sp::SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized
                    : kind == sp::SpatialChannelKind::kReduceMerge
                          ? sp::SpatialChannelDeliveryKind::kCompletionVisible
                          : sp::SpatialChannelDeliveryKind::kOrdered;
    channel_records.push_back(ChannelRecord{
        static_cast<std::string>(boundary->name),
        kind,
        kind == sp::SpatialChannelKind::kGather || kind == sp::SpatialChannelKind::kScatter
            ? sp::SpatialChannelPayloadKind::kIndex
            : sp::SpatialChannelPayloadKind::kTensor,
        delivery_kind,
        static_cast<int>(boundary->source_closure_index),
        static_cast<int>(boundary->target_closure_index),
        task_records[boundary->source_closure_index].name,
        task_records[boundary->target_closure_index].name,
        subject,
        cross_phase ? std::vector<std::string>{"phase_b", "phase_boundary"}
                    : std::vector<std::string>{"phase_b"},
    });
  }

  Array<Task> tasks;
  Array<Placement> placements;
  for (int i = 0; i < static_cast<int>(task_records.size()); ++i) {
    tasks.push_back(MakeSpatialTask(task_records[i], task_phase[i]));
    placements.push_back(MakeExecutionPlacement(task_records[i], i, member_func));
  }

  Array<Channel> channels;
  for (const ChannelRecord& record : channel_records) {
    channels.push_back(MakeSpatialChannel(record, capability_model));
  }

  std::unordered_map<int, std::vector<int>> task_indices_by_phase;
  std::unordered_map<int, std::vector<int>> channel_indices_by_phase;
  std::unordered_map<int, std::vector<std::string>> task_names_by_phase;
  std::unordered_map<int, std::vector<std::string>> channel_names_by_phase;
  for (int i = 0; i < static_cast<int>(task_records.size()); ++i) {
    task_indices_by_phase[task_phase[i]].push_back(i);
    task_names_by_phase[task_phase[i]].push_back(task_records[i].name);
  }
  for (int i = 0; i < static_cast<int>(channel_records.size()); ++i) {
    channel_indices_by_phase[task_phase[channel_records[i].target_task_index]].push_back(i);
    channel_names_by_phase[task_phase[channel_records[i].target_task_index]].push_back(
        channel_records[i].name);
  }

  Array<ProgramPhase> phases;
  for (const auto& [phase_index, task_indices] : task_indices_by_phase) {
    std::vector<std::string> ordering_kinds;
    for (const ChannelRecord& record : channel_records) {
      if (task_phase[record.source_task_index] == phase_index ||
          task_phase[record.target_task_index] == phase_index) {
        ordering_kinds.push_back(
            DeriveOrderingKindForChannel(record.kind, record.delivery_kind));
      }
    }
    std::sort(ordering_kinds.begin(), ordering_kinds.end());
    ordering_kinds.erase(std::unique(ordering_kinds.begin(), ordering_kinds.end()),
                         ordering_kinds.end());
    const std::string phase_name = "phase_" + std::to_string(phase_index);
    phases.push_back(MakeProgramPhaseNode(
        phase_name, task_names_by_phase[phase_index], channel_names_by_phase[phase_index],
        phase_index, task_indices, channel_indices_by_phase[phase_index],
        BuildPhaseClosureBasis(phase_name, task_indices.size(), ordering_kinds)));
  }

  Array<SyncEdge> sync_edges;
  std::unordered_set<std::string> seen_sync_keys;
  for (const ChannelRecord& record : channel_records) {
    if (record.source_task_index == record.target_task_index) {
      continue;
    }
    const std::string ordering_kind =
        DeriveOrderingKindForChannel(record.kind, record.delivery_kind);
    const std::string materialization_kind =
        DeriveMaterializationKindForChannel(record.kind, record.delivery_kind);
    const std::string key = std::to_string(record.source_task_index) + "->" +
                            std::to_string(record.target_task_index) + "|" + ordering_kind + "|" +
                            materialization_kind;
    if (!seen_sync_keys.insert(key).second) {
      continue;
    }
    sync_edges.push_back(MakeCompletionSyncEdge(
        "sync_" + record.name, record.source_task_name, record.target_task_name,
        record.source_task_index, record.target_task_index, ordering_kind,
        materialization_kind));
  }

  Array<ResourceIntent> resource_intents;
  AppendFragmentResourceIntent(member_func, fragment_facts, &resource_intents);
  AppendPipelineResourceIntent(member_func, func, &resource_intents);
  AppendPhaseBoundaryResourceIntents(channel_records, &resource_intents);

  Array<SpatialLayout> layouts;
  Array<WorkPartition> work_partitions;
  BuildCommonSpatialScaffolding(member_func, func, capability_model, fragment_facts, &layouts,
                                &work_partitions);
  return SpatialExecutionPlan(String(member_func), phases, tasks, channels, placements, sync_edges,
                              resource_intents, MakeAnchors("spatial_execution_plan", member_func));
}

}  // namespace

TVM_DLL SpatialExecutionPlan BuildSpatialExecutionPlanForFunc(
    const std::string& member_func, const SpatialPlan& plan, const tir::PrimFunc& func,
    const SpatialCapabilityModel& capability_model) {
  const FragmentFacts fragment_facts = AnalyzeFragmentFacts(func);
  if (IsSimpleGemmFastPath(plan, func)) {
    return BuildSegmentFastPath(member_func, func, capability_model, fragment_facts,
                                {"reader", "compute", "writer"}, {"fast_path", "gemm"},
                                "phase0_gemm", "gemm_compute");
  }
  if (IsSimpleFragmentComputeFastPath(plan, func, fragment_facts)) {
    return BuildSegmentFastPath(member_func, func, capability_model, fragment_facts,
                                DeriveFragmentFastPathSegmentKinds(func, fragment_facts),
                                {"fast_path", "fragment_compute"}, "phase0_fragment",
                                "fragment_compute");
  }
  if (IsSimpleCopyFastPath(plan, func, fragment_facts)) {
    return BuildCopyFastPath(member_func, func, capability_model, fragment_facts);
  }
  return BuildGenericSpatialProgram(member_func, plan, func, capability_model, fragment_facts);
}

}  // namespace tl
}  // namespace tvm
