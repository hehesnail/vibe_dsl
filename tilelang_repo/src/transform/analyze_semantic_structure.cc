/*
 * \file analyze_semantic_structure.cc
 * \brief Build a minimal semantic-structure summary from existing Blackhole analysis attrs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/utils.h"
#include "common/blackhole_utils.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"
#include "common/semantic_witness_payloads.h"
#include "runtime/thread_storage_scope.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::Integer;
using namespace tvm::tl::semantic;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

std::unordered_map<std::string, std::vector<int64_t>> BuildLogicalBufferShapes(
    const tir::PrimFunc& func) {
  std::unordered_map<std::string, std::vector<int64_t>> logical_buffer_shapes;
  auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest);
  auto manifest_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifestSeeds);
  if (!manifest.has_value() && !manifest_seeds.has_value()) {
    return logical_buffer_shapes;
  }

  std::unordered_map<std::string, std::vector<int64_t>> canonical_shapes;
  auto register_shape = [&](const std::string& name, const std::vector<int64_t>& shape) {
    if (name.empty() || shape.empty()) {
      return;
    }
    logical_buffer_shapes[name] = shape;
    canonical_shapes[name] = shape;
  };

  auto ingest_buffer_descriptors = [&](const Optional<Map<String, Any>>& maybe_payload) {
    if (!maybe_payload.has_value()) {
      return;
    }
    auto buffers_it = maybe_payload.value().find(manifest_key::kBuffers);
    if (buffers_it == maybe_payload.value().end()) {
      return;
    }
    for (const Any& buffer_any : Downcast<Array<Any>>((*buffers_it).second)) {
      auto descriptor = Downcast<Map<String, Any>>(buffer_any);
      auto name = descriptor.Get(String(schema_key::kName));
      auto shape = descriptor.Get(String(schema_key::kShape));
      if (!name.has_value() || !shape.has_value()) {
        continue;
      }
      auto static_shape = ExtractStaticShape(Downcast<Array<PrimExpr>>(shape.value()));
      if (!static_shape.has_value()) {
        continue;
      }
      register_shape(static_cast<std::string>(Downcast<String>(name.value())),
                     static_shape.value());
    }
  };
  ingest_buffer_descriptors(manifest);
  ingest_buffer_descriptors(manifest_seeds);

  if (!manifest.has_value()) {
    return logical_buffer_shapes;
  }
  auto structural_regions_it = manifest.value().find(manifest_key::kStructuralRegions);
  if (structural_regions_it == manifest.value().end()) {
    return logical_buffer_shapes;
  }

  auto infer_shape_from_names = [&](const Array<Any>& names) -> std::vector<int64_t> {
    for (const Any& name_any : names) {
      const std::string name = static_cast<std::string>(Downcast<String>(name_any));
      auto it = canonical_shapes.find(name);
      if (it != canonical_shapes.end()) {
        return it->second;
      }
    }
    return {};
  };

  auto infer_shape_from_buffers = [&](const Array<Any>& buffers) -> std::vector<int64_t> {
    for (const Any& buffer_any : buffers) {
      auto buffer = buffer_any.try_cast<Buffer>();
      if (!buffer.has_value()) {
        continue;
      }
      const std::string identity = BufferIdentityName(buffer.value());
      auto it = logical_buffer_shapes.find(identity);
      if (it != logical_buffer_shapes.end()) {
        return it->second;
      }
    }
    return {};
  };

  const Array<Any> structural_regions = Downcast<Array<Any>>((*structural_regions_it).second);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const Any& region_any : structural_regions) {
      auto region = Downcast<Map<String, Any>>(region_any);
      auto update_sources_it = region.find(manifest_key::kUpdateSources);
      if (update_sources_it == region.end()) {
        continue;
      }
      for (const Any& update_any : Downcast<Array<Any>>((*update_sources_it).second)) {
        auto update = Downcast<Map<String, Any>>(update_any);
        auto target_it = update.find(schema_key::kTarget);
        if (target_it == update.end()) {
          continue;
        }
        const std::string target_name =
            static_cast<std::string>(Downcast<String>((*target_it).second));
        if (target_name.empty() || canonical_shapes.count(target_name)) {
          continue;
        }
        std::vector<int64_t> inferred_shape;
        auto sources_it = update.find(schema_key::kSources);
        if (sources_it != update.end()) {
          inferred_shape = infer_shape_from_names(Downcast<Array<Any>>((*sources_it).second));
        }
        if (inferred_shape.empty()) {
          auto source_states_it = update.find(schema_key::kSourceStates);
          if (source_states_it != update.end()) {
            inferred_shape =
                infer_shape_from_names(Downcast<Array<Any>>((*source_states_it).second));
          }
        }
        if (inferred_shape.empty()) {
          auto source_buffers_it = update.find(schema_key::kSourceBuffers);
          if (source_buffers_it != update.end()) {
            inferred_shape =
                infer_shape_from_buffers(Downcast<Array<Any>>((*source_buffers_it).second));
          }
        }
        if (!inferred_shape.empty()) {
          canonical_shapes[target_name] = inferred_shape;
          logical_buffer_shapes[target_name] = inferred_shape;
          changed = true;
        }
      }
    }
  }

  for (const Any& region_any : structural_regions) {
    auto region = Downcast<Map<String, Any>>(region_any);
    auto fragment_buffers_it = region.find(manifest_key::kFragmentBuffers);
    if (fragment_buffers_it == region.end()) {
      continue;
    }
    for (const Any& fragment_any : Downcast<Array<Any>>((*fragment_buffers_it).second)) {
      auto fragment = Downcast<Map<String, Any>>(fragment_any);
      auto name_it = fragment.find(schema_key::kName);
      auto buffer_it = fragment.find(schema_key::kBuffer);
      if (name_it == fragment.end() || buffer_it == fragment.end()) {
        continue;
      }
      const std::string canonical_name =
          static_cast<std::string>(Downcast<String>((*name_it).second));
      auto shape_it = canonical_shapes.find(canonical_name);
      auto buffer = (*buffer_it).second.try_cast<Buffer>();
      if (shape_it == canonical_shapes.end() || !buffer.has_value()) {
        continue;
      }
      logical_buffer_shapes[BufferIdentityName(buffer.value())] = shape_it->second;
    }
  }

  return logical_buffer_shapes;
}

int64_t GetLogicalElementCount(const std::string& buffer_name,
                              const std::unordered_map<std::string, std::vector<int64_t>>&
                                  logical_buffer_shapes) {
  auto it = logical_buffer_shapes.find(buffer_name);
  if (it == logical_buffer_shapes.end() || it->second.empty()) {
    return -1;
  }
  int64_t logical_element_count = 1;
  for (int64_t dim : it->second) {
    if (dim <= 0) {
      return -1;
    }
    if (logical_element_count > std::numeric_limits<int64_t>::max() / dim) {
      return -1;
    }
    logical_element_count *= dim;
  }
  return logical_element_count;
}

int64_t GetLogicalRowWidth(
    const Buffer& buffer,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes) {
  const std::string buffer_identity = BufferIdentityName(buffer);
  auto logical_it = logical_buffer_shapes.find(buffer_identity);
  if (logical_it != logical_buffer_shapes.end() && logical_it->second.size() >= 2U) {
    return logical_it->second.back();
  }
  auto static_shape = ExtractStaticShape(buffer->shape);
  if (static_shape.has_value() && static_shape.value().size() >= 2U) {
    return static_shape.value().back();
  }
  return -1;
}

std::string ResolveStateNameFromMap(const Map<String, Any>& entry, const char* name_key,
                                    const char* buffer_key) {
  if (auto it = entry.find(buffer_key); it != entry.end()) {
    auto buffer = (*it).second.try_cast<tir::Buffer>();
    if (buffer.has_value()) {
      const std::string resolved = BufferIdentityName(buffer.value());
      if (!resolved.empty()) {
        return resolved;
      }
    }
  }
  if (auto it = entry.find(name_key); it != entry.end()) {
    return (*it).second.cast<String>();
  }
  return "";
}

std::string ResolveStateName(const Any& value) {
  if (auto string_value = value.try_cast<String>(); string_value.has_value()) {
    return string_value.value();
  }
  auto map_value = value.try_cast<Map<String, Any>>();
  if (!map_value.has_value()) {
    return "";
  }
  return ResolveStateNameFromMap(map_value.value(), schema_key::kName, schema_key::kBuffer);
}

Array<Any> ResolveStateArray(const Map<String, Any>& entry, const char* state_key,
                             const char* buffer_key) {
  Array<Any> resolved;
  if (auto it = entry.find(buffer_key); it != entry.end()) {
    for (const Any& buffer_any : tvm::Downcast<Array<Any>>((*it).second)) {
      auto buffer = buffer_any.try_cast<tir::Buffer>();
      if (!buffer.has_value()) {
        continue;
      }
      const std::string name = BufferIdentityName(buffer.value());
      if (!name.empty()) {
        resolved.push_back(String(name));
      }
    }
    if (!resolved.empty()) {
      return resolved;
    }
  }
  if (auto it = entry.find(state_key); it != entry.end()) {
    for (const Any& state_any : tvm::Downcast<Array<Any>>((*it).second)) {
      const std::string name = ResolveStateName(state_any);
      if (!name.empty()) {
        resolved.push_back(String(name));
      }
    }
  }
  return resolved;
}

void AppendUniqueResolvedString(Array<Any>* values, std::unordered_set<std::string>* seen,
                                const std::string& value) {
  if (!value.empty() && seen->insert(value).second) {
    values->push_back(String(value));
  }
}

void CollectUniqueStringField(Array<Any>* values, std::unordered_set<std::string>* seen,
                              const Map<String, Any>& region, const char* field_name,
                              const char* nested_field_name = nullptr) {
  auto it = region.find(field_name);
  if (it == region.end()) {
    return;
  }
  for (const Any& item : tvm::Downcast<Array<Any>>((*it).second)) {
    if (nested_field_name == nullptr) {
      AppendUniqueResolvedString(values, seen, ResolveStateName(item));
      continue;
    }
    auto nested = item.try_cast<Map<String, Any>>();
    if (!nested.has_value()) {
      continue;
    }
    auto field_it = nested.value().find(nested_field_name);
    if (field_it == nested.value().end()) {
      continue;
    }
    AppendUniqueResolvedString(values, seen, ResolveStateName((*field_it).second));
  }
}

void CollectUniqueNestedMapField(Array<Any>* values, std::unordered_set<std::string>* seen,
                                 const Map<String, Any>& region, const char* field_name,
                                 const char* nested_field_name) {
  auto it = region.find(field_name);
  if (it == region.end()) {
    return;
  }
  for (const Any& item : tvm::Downcast<Array<Any>>((*it).second)) {
    auto nested = item.try_cast<Map<String, Any>>();
    if (!nested.has_value()) {
      continue;
    }
    auto field_it = nested.value().find(nested_field_name);
    if (field_it == nested.value().end()) {
      continue;
    }
    const std::string identity = ResolveStateName((*field_it).second);
    if (identity.empty() || !seen->insert(identity).second) {
      continue;
    }
    values->push_back(nested.value());
  }
}

bool IsTrackedStateScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

bool IsFragmentMaterializationCandidate(const CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  TileOperator tile_op = ParseOperator(GetRef<Call>(call));
  if (!tile_op.defined()) {
    return false;
  }
  return tile_op->GetFragmentMaterializationInfo().has_value();
}

Optional<Map<String, Any>> TryBuildFragmentMaterializationContract(const CallNode* call) {
  if (call == nullptr) {
    return Optional<Map<String, Any>>();
  }
  TileOperator tile_op = ParseOperator(GetRef<Call>(call));
  if (!tile_op.defined()) {
    return Optional<Map<String, Any>>();
  }
  auto materialization_info = tile_op->GetFragmentMaterializationInfo();
  if (!materialization_info.has_value()) {
    return Optional<Map<String, Any>>();
  }
  const Buffer& target = materialization_info->target_buffer;
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
  contract.Set(String(schema_key::kMaterializationKind),
               materialization_info->materialization_kind);
  contract.Set(String(schema_key::kBridgeKind), materialization_info->bridge_kind);
  contract.Set(String(schema_key::kValueRole), materialization_info->value_role);
  contract.Set(String(schema_key::kMergeKind), materialization_info->merge_kind);
  contract.Set(String(schema_key::kExecutionProtocol), materialization_info->execution_protocol);
  contract.Set(String(schema_key::kResultLiveForm), materialization_info->result_live_form);
  return contract;
}

Optional<Map<String, Any>> TryBuildRepublishFragmentMaterializationContract(
    const Map<String, Any>& flow_contract) {
  auto buffer_it = flow_contract.find(String(schema_key::kBuffer));
  auto scope_it = flow_contract.find(String(schema_key::kScope));
  auto flow_class_it = flow_contract.find(String(schema_key::kFlowClass));
  auto granule_kind_it = flow_contract.find(String(schema_key::kGranuleKind));
  auto events_it = flow_contract.find(String(schema_key::kEvents));
  if (buffer_it == flow_contract.end() || scope_it == flow_contract.end() ||
      flow_class_it == flow_contract.end() || granule_kind_it == flow_contract.end() ||
      events_it == flow_contract.end()) {
    return Optional<Map<String, Any>>();
  }

  const std::string buffer_name = Downcast<String>((*buffer_it).second);
  const std::string scope = Downcast<String>((*scope_it).second);
  const std::string flow_class = Downcast<String>((*flow_class_it).second);
  const std::string granule_kind = Downcast<String>((*granule_kind_it).second);
  if (buffer_name.empty() || !IsTrackedStateScope(scope) ||
      flow_class != fragment_flow::kRepublish ||
      granule_kind != fragment_flow::kLogicalTile) {
    return Optional<Map<String, Any>>();
  }

  bool has_compute_consume = false;
  bool has_transport_consume = false;
  for (const Any& event_any : Downcast<Array<Any>>((*events_it).second)) {
    Map<String, Any> event = Downcast<Map<String, Any>>(event_any);
    auto kind_it = event.find(String(schema_key::kKind));
    if (kind_it == event.end()) {
      continue;
    }
    const std::string kind = Downcast<String>((*kind_it).second);
    if (kind == fragment_flow::kComputeConsume) {
      has_compute_consume = true;
    } else if (kind == fragment_flow::kTransportConsume) {
      has_transport_consume = true;
    }
  }
  if (!has_compute_consume && !has_transport_consume) {
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
  contract.Set(String(schema_key::kResultLiveForm),
               String(fragment_live_form::kTiledCB));
  return contract;
}

Map<String, Any> MakeRepublishedLogicalTileMaterializationContract(const std::string& buffer_name,
                                                                  const std::string& scope,
                                                                  const std::string& source_buffer =
                                                                      std::string(),
                                                                  int64_t logical_row_width = -1,
                                                                  int64_t logical_element_count =
                                                                      -1) {
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
  contract.Set(String(schema_key::kResultLiveForm),
               String(fragment_live_form::kTiledCB));
  if (!source_buffer.empty()) {
    contract.Set(String(schema_key::kSourceBuffer), String(source_buffer));
  }
  if (logical_row_width > 0) {
    contract.Set(String(schema_key::kLogicalRowWidth),
                 Integer(static_cast<int>(logical_row_width)));
  }
  if (logical_element_count > 0) {
    contract.Set(String(schema_key::kLogicalElementCount),
                 Integer(static_cast<int64_t>(logical_element_count)));
  }
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
    if (kind_it == event.end()) {
      continue;
    }
    if (Downcast<String>((*kind_it).second) == kind) {
      return true;
    }
  }
  return false;
}

bool IsVectorLocalFragmentBuffer(const Buffer& buffer) {
  const std::string scope = buffer.scope();
  return IsTrackedStateScope(scope) && buffer->shape.size() == 1 && !buffer->shape.empty() &&
         !tir::is_one(buffer->shape[0]);
}

const ForNode* AsUnwrappedFor(const tir::Stmt& stmt) {
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

const BufferStoreNode* AsUnwrappedBufferStore(const tir::Stmt& stmt) {
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
  return current.as<BufferStoreNode>();
}

bool ExprUsesVar(const PrimExpr& expr, const tir::Var& var) {
  if (!var.defined()) {
    return false;
  }
  bool uses_var = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (uses_var) {
      return;
    }
    if (const auto* candidate = node.as<tir::VarNode>()) {
      uses_var = candidate == var.get();
    }
  });
  return uses_var;
}

bool MatchDirectFragmentCastTarget(const ForNode* op, Buffer* src_buffer, Buffer* dst_buffer) {
  if (!op || !src_buffer || !dst_buffer) {
    return false;
  }

  const auto* store = AsUnwrappedBufferStore(op->body);
  const ForNode* inner_loop = nullptr;
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
  const auto* cast = inner_store->value.as<CastNode>();
  const auto* load = cast ? cast->value.as<BufferLoadNode>() : nullptr;
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

std::unordered_set<std::string> CollectGroupedRowsFragmentBuffers(
    const Array<Any>& fragment_layout_contracts) {
  std::unordered_set<std::string> grouped_rows_buffers;
  for (const Any& contract_any : fragment_layout_contracts) {
    Map<String, Any> contract = Downcast<Map<String, Any>>(contract_any);
    auto buffer_it = contract.find(String(schema_key::kBuffer));
    auto distribution_it = contract.find(String(schema_key::kDistributionKind));
    if (buffer_it == contract.end() || distribution_it == contract.end()) {
      continue;
    }
    if (Downcast<String>((*distribution_it).second) != fragment_layout::kGroupedRows) {
      continue;
    }
    grouped_rows_buffers.insert(Downcast<String>((*buffer_it).second));
  }
  return grouped_rows_buffers;
}

void AppendUniqueCastDrivenFragmentMaterializationContractsFromBody(
    const tir::Stmt& body, const Array<Any>& flow_contracts,
    const std::unordered_map<std::string, std::vector<int64_t>>& logical_buffer_shapes,
    Array<Any>* materialization_contracts) {
  auto upsert_materialization_contract = [&](const Map<String, Any>& contract) {
    auto target_it = contract.find(String(schema_key::kTargetBuffer));
    auto scope_it = contract.find(String(schema_key::kScope));
    if (target_it == contract.end() || scope_it == contract.end()) {
      return;
    }
    const std::string key = Downcast<String>((*target_it).second) + "|" +
                            Downcast<String>((*scope_it).second);
    for (int i = 0; i < materialization_contracts->size(); ++i) {
      Map<String, Any> existing = Downcast<Map<String, Any>>((*materialization_contracts)[i]);
      auto existing_target_it = existing.find(String(schema_key::kTargetBuffer));
      auto existing_scope_it = existing.find(String(schema_key::kScope));
      if (existing_target_it == existing.end() || existing_scope_it == existing.end()) {
        continue;
      }
      const std::string existing_key =
          Downcast<String>((*existing_target_it).second) + "|" +
          Downcast<String>((*existing_scope_it).second);
      if (existing_key == key) {
        (*materialization_contracts).Set(i, contract);
        return;
      }
    }
    materialization_contracts->push_back(contract);
  };

  std::unordered_map<std::string, Map<String, Any>> flow_contracts_by_buffer;
  for (const Any& flow_contract_any : flow_contracts) {
    Map<String, Any> flow_contract = Downcast<Map<String, Any>>(flow_contract_any);
    auto buffer_it = flow_contract.find(String(schema_key::kBuffer));
    if (buffer_it == flow_contract.end()) {
      continue;
    }
    flow_contracts_by_buffer.emplace(Downcast<String>((*buffer_it).second), flow_contract);
  }

  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* loop = node.as<ForNode>();
    if (!loop) {
      return;
    }
    Buffer src_buffer;
    Buffer dst_buffer;
    if (!MatchDirectFragmentCastTarget(loop, &src_buffer, &dst_buffer)) {
      return;
    }
    const std::string src_name = BufferIdentityName(src_buffer);
    const std::string dst_name = BufferIdentityName(dst_buffer);
    if (src_name.empty() || dst_name.empty()) {
      return;
    }
    auto flow_it = flow_contracts_by_buffer.find(dst_name);
    if (flow_it == flow_contracts_by_buffer.end()) {
      return;
    }
    const Map<String, Any>& flow_contract = flow_it->second;
    auto scope_it = flow_contract.find(String(schema_key::kScope));
    auto granule_it = flow_contract.find(String(schema_key::kGranuleKind));
    if (scope_it == flow_contract.end() || granule_it == flow_contract.end()) {
      return;
    }
    const std::string scope = Downcast<String>((*scope_it).second);
    const std::string granule_kind = Downcast<String>((*granule_it).second);
    if (!IsTrackedStateScope(scope) || granule_kind != fragment_flow::kLogicalTile ||
        (!FlowContractHasEventKind(flow_contract, fragment_flow::kComputeConsume) &&
         !FlowContractHasEventKind(flow_contract, fragment_flow::kTransportConsume))) {
      return;
    }
    int64_t logical_row_width = GetLogicalRowWidth(src_buffer, logical_buffer_shapes);
    if (logical_row_width <= 0) {
      logical_row_width = GetLogicalRowWidth(dst_buffer, logical_buffer_shapes);
    }
    if (logical_row_width <= 0) {
      if (const auto* inner_loop = AsUnwrappedFor(loop->body)) {
        if (const auto* extent_imm = inner_loop->extent.as<IntImmNode>()) {
          logical_row_width = extent_imm->value;
        }
      }
    }
    int64_t logical_element_count = GetLogicalElementCount(dst_name, logical_buffer_shapes);
    if (logical_element_count <= 0) {
      logical_element_count = GetLogicalElementCount(src_name, logical_buffer_shapes);
    }
    upsert_materialization_contract(MakeRepublishedLogicalTileMaterializationContract(
        dst_name, scope, src_name, logical_row_width, logical_element_count));
  });
}

Array<Any> CollectFragmentMaterializationContractsFromBody(const tir::Stmt& body) {
  Array<Any> contracts;
  std::unordered_set<std::string> seen;
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (call == nullptr || !IsFragmentMaterializationCandidate(call)) {
      return;
    }
    auto contract = TryBuildFragmentMaterializationContract(call);
    if (!contract.has_value()) {
      return;
    }
    const std::string target_buffer =
        Downcast<String>(contract.value().at(String(schema_key::kTargetBuffer)));
    const std::string scope = Downcast<String>(contract.value().at(String(schema_key::kScope)));
    const std::string key = target_buffer + "|" + scope;
    if (!seen.insert(key).second) {
      return;
    }
    contracts.push_back(contract.value());
  });
  return contracts;
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
    case FragmentBufferFlowEventKind::kReference:
    default:
      return fragment_flow::kReference;
  }
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

bool IsCopyOperation(const BufferStoreNode* store) {
  const auto* load = store ? store->value.as<BufferLoadNode>() : nullptr;
  return load != nullptr && !SameBufferIdentity(store->buffer, load->buffer);
}

bool IsTransportConsumerDirection(const BufferStoreNode* store) {
  const auto* load = store ? store->value.as<BufferLoadNode>() : nullptr;
  if (load == nullptr) {
    return false;
  }
  const std::string dst_scope = load ? str(store->buffer.scope()) : std::string();
  const std::string src_scope = str(load->buffer.scope());
  return (IsCBScope(src_scope) && IsDRAMScope(dst_scope)) ||
         (IsCBScope(src_scope) && IsCBScope(dst_scope)) ||
         (IsAccumulatorLikeScope(src_scope) && IsDRAMScope(dst_scope)) ||
         (IsAccumulatorLikeScope(src_scope) && IsCBScope(dst_scope));
}

bool StmtWritesBuffer(const tir::Stmt& stmt, const Buffer& buffer) {
  const std::string target_identity = BufferIdentityName(buffer);
  bool writes = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (writes) {
      return;
    }
    const auto* store = node.as<BufferStoreNode>();
    writes = store != nullptr && BufferIdentityName(store->buffer) == target_identity;
  });
  return writes;
}

bool StmtReferencesBuffer(const tir::Stmt& stmt, const Buffer& buffer) {
  const std::string target_identity = BufferIdentityName(buffer);
  bool referenced = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (referenced) {
      return;
    }
    if (const auto* store = node.as<BufferStoreNode>()) {
      referenced = BufferIdentityName(store->buffer) == target_identity;
      return;
    }
    if (const auto* load = node.as<BufferLoadNode>()) {
      referenced = BufferIdentityName(load->buffer) == target_identity;
      return;
    }
    const auto* call = node.as<CallNode>();
    if (call == nullptr) {
      return;
    }
    for (const PrimExpr& arg : call->args) {
      if (!IsBufferLikeExpr(arg)) {
        continue;
      }
      tir::BufferRegion region = NormalizeToBufferRegion(arg);
      if (BufferIdentityName(region->buffer) == target_identity) {
        referenced = true;
        return;
      }
    }
  });
  return referenced;
}

bool StmtReadsBuffer(const tir::Stmt& stmt, const Buffer& buffer) {
  const std::string target_identity = BufferIdentityName(buffer);
  bool reads = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (reads) {
      return;
    }
    if (const auto* load = node.as<BufferLoadNode>()) {
      reads = BufferIdentityName(load->buffer) == target_identity;
      return;
    }
    const auto* call = node.as<CallNode>();
    if (call == nullptr || !call->op->IsInstance<OpNode>()) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    if (!tile_op.defined()) {
      return;
    }
    for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
      if (BufferIdentityName(access.buffer) == target_identity) {
        reads = true;
        return;
      }
    }
  });
  return reads;
}

bool StmtConsumesBufferViaTransport(const tir::Stmt& stmt, const Buffer& buffer) {
  const std::string target_identity = BufferIdentityName(buffer);
  bool consumed = false;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    if (consumed) {
      return;
    }
    const auto* store = node.as<BufferStoreNode>();
    if (!IsCopyOperation(store) || !IsTransportConsumerDirection(store)) {
      return;
    }
    const auto* load = store->value.as<BufferLoadNode>();
    consumed = load != nullptr && BufferIdentityName(load->buffer) == target_identity;
  });
  return consumed;
}

std::unordered_set<std::string> CollectComputeConsumedBuffers(const tir::Stmt& stmt) {
  std::unordered_set<std::string> consumed_buffers;
  tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
    const auto* call = node.as<CallNode>();
    if (call == nullptr || !call->op->IsInstance<OpNode>()) {
      return;
    }
    TileOperator tile_op = ParseOperator(GetRef<Call>(call));
    if (!tile_op.defined()) {
      return;
    }
    for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
      if (access.kind != DataflowAccessKind::kComputeConsume) {
        continue;
      }
      const std::string name = BufferIdentityName(access.buffer);
      if (!name.empty()) {
        consumed_buffers.insert(name);
      }
    }
  });
  return consumed_buffers;
}

std::string DeriveFragmentFlowClassLabel(const std::vector<FragmentBufferFlowEvent>& events) {
  int first_consume_order = std::numeric_limits<int>::max();
  bool has_consume = false;
  for (const FragmentBufferFlowEvent& event : events) {
    if (event.kind == FragmentBufferFlowEventKind::kComputeConsume ||
        event.kind == FragmentBufferFlowEventKind::kTransportConsume) {
      has_consume = true;
      first_consume_order = std::min(first_consume_order, event.order_index);
    }
  }
  if (!has_consume) {
    return fragment_flow::kState;
  }
  for (const FragmentBufferFlowEvent& event : events) {
    if (event.kind == FragmentBufferFlowEventKind::kWrite &&
        event.order_index > first_consume_order) {
      return fragment_flow::kRepublish;
    }
  }
  return fragment_flow::kStream;
}

Array<Any> CollectFragmentBufferFlowContractsFromBody(const tir::Stmt& body) {
  std::unordered_map<std::string, Buffer> tracked_buffers;
  const std::vector<tir::Stmt> ordered_stmts = CollectExecutionOrderedStmts(body);
  for (const tir::Stmt& stmt : ordered_stmts) {
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      auto remember_buffer = [&](const Buffer& buffer) {
        if (!buffer.defined()) {
          return;
        }
        const std::string scope = str(buffer.scope());
        if (!IsTrackedBufferFlowScope(scope)) {
          return;
        }
        const std::string name = BufferIdentityName(buffer);
        if (!name.empty()) {
          tracked_buffers.emplace(name, buffer);
        }
      };
      if (const auto* store = node.as<BufferStoreNode>()) {
        remember_buffer(store->buffer);
        return;
      }
      if (const auto* load = node.as<BufferLoadNode>()) {
        remember_buffer(load->buffer);
        return;
      }
      const auto* call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      TileOperator tile_op = ParseOperator(GetRef<Call>(call));
      if (tile_op.defined()) {
        for (const DataflowAccessInfo& access : tile_op->GetDataflowAccessInfo()) {
          remember_buffer(access.buffer);
        }
      }
      for (const PrimExpr& arg : call->args) {
        if (!IsBufferLikeExpr(arg)) {
          continue;
        }
        remember_buffer(NormalizeToBufferRegion(arg)->buffer);
      }
    });
  }

  std::unordered_map<std::string, FragmentBufferFlowContract> contracts_by_buffer;
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    const tir::Stmt& stmt = ordered_stmts[order_index];
    const std::unordered_set<std::string> compute_consumed_buffers =
        CollectComputeConsumedBuffers(stmt);
    for (const auto& [buffer_name, buffer] : tracked_buffers) {
      if (!StmtReferencesBuffer(stmt, buffer)) {
        continue;
      }
      FragmentBufferFlowContract& contract = contracts_by_buffer[buffer_name];
      contract.buffer_name = buffer_name;
      contract.scope = str(buffer.scope());
      const bool compute_consumed = compute_consumed_buffers.count(buffer_name);
      const bool transport_consumed = StmtConsumesBufferViaTransport(stmt, buffer);
      const bool reads_buffer = StmtReadsBuffer(stmt, buffer);
      const bool writes_buffer = StmtWritesBuffer(stmt, buffer);
      auto append_event = [&](FragmentBufferFlowEventKind kind) {
        FragmentBufferFlowEvent event;
        event.order_index = order_index;
        event.kind = kind;
        contract.events.push_back(event);
      };
      if (compute_consumed) {
        append_event(FragmentBufferFlowEventKind::kComputeConsume);
      }
      if (!compute_consumed && reads_buffer && !transport_consumed) {
        append_event(FragmentBufferFlowEventKind::kReference);
      }
      if (transport_consumed) {
        append_event(FragmentBufferFlowEventKind::kTransportConsume);
      }
      if (writes_buffer) {
        append_event(FragmentBufferFlowEventKind::kWrite);
      }
      if (!compute_consumed && !reads_buffer && !transport_consumed && !writes_buffer) {
        append_event(FragmentBufferFlowEventKind::kReference);
      }
    }
  }

  Array<Any> contracts;
  for (auto& [buffer_name, contract] : contracts_by_buffer) {
    if (contract.events.empty()) {
      continue;
    }
    contract.flow_class = DeriveFragmentFlowClassLabel(contract.events);
    Map<String, Any> encoded_contract;
    encoded_contract.Set(String(schema_key::kBuffer), String(buffer_name));
    encoded_contract.Set(String(schema_key::kScope), String(contract.scope));
    encoded_contract.Set(String(schema_key::kFlowClass), String(contract.flow_class));
    encoded_contract.Set(String(schema_key::kGranuleKind), String(fragment_flow::kLogicalTile));
    encoded_contract.Set(String(schema_key::kPublishGranule), Integer(contract.publish_granule));
    encoded_contract.Set(String(schema_key::kConsumeGranule), Integer(contract.consume_granule));
    Array<Any> events;
    for (const FragmentBufferFlowEvent& event : contract.events) {
      Map<String, Any> encoded_event;
      encoded_event.Set(String(schema_key::kKind),
                        String(FragmentBufferFlowEventKindToString(event.kind)));
      encoded_event.Set(String(schema_key::kOrderIndex), Integer(event.order_index));
      events.push_back(encoded_event);
    }
    encoded_contract.Set(String(schema_key::kEvents), events);
    contracts.push_back(encoded_contract);
  }
  return contracts;
}

void AppendUniqueFragmentMaterializationContractsFromFlowContracts(
    const Array<Any>& flow_contracts, Array<Any>* materialization_contracts) {
  std::unordered_set<std::string> seen;
  for (const Any& contract_any : *materialization_contracts) {
    Map<String, Any> contract = Downcast<Map<String, Any>>(contract_any);
    auto target_it = contract.find(String(schema_key::kTargetBuffer));
    auto scope_it = contract.find(String(schema_key::kScope));
    if (target_it == contract.end() || scope_it == contract.end()) {
      continue;
    }
    seen.insert(Downcast<String>((*target_it).second) + "|" +
                Downcast<String>((*scope_it).second));
  }

  for (const Any& flow_contract_any : flow_contracts) {
    Map<String, Any> flow_contract = Downcast<Map<String, Any>>(flow_contract_any);
    auto maybe_contract = TryBuildRepublishFragmentMaterializationContract(flow_contract);
    if (!maybe_contract.has_value()) {
      continue;
    }
    const std::string key =
        Downcast<String>(maybe_contract.value().at(String(schema_key::kTargetBuffer))) + "|" +
        Downcast<String>(maybe_contract.value().at(String(schema_key::kScope)));
    if (!seen.insert(key).second) {
      continue;
    }
    materialization_contracts->push_back(maybe_contract.value());
  }
}

class LocalBufferCollector : public tir::StmtExprVisitor {
 public:
  void VisitStmt_(const tir::BlockNode* op) final {
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      Register(buffer);
    }
    tir::StmtExprVisitor::VisitStmt_(op);
  }

  Array<Any> Encode() const {
    Array<Any> states;
    for (const auto& name : order_) {
      const auto& entry = entries_.at(name);
      Map<String, Any> state;
      state.Set(schema_key::kName, String(name));
      state.Set("role", String(ToString(StateRole::kTransient)));
      state.Set(schema_key::kScope, String(entry.scope));
      states.push_back(state);
    }
    return states;
  }

  bool HasIntegerDType(const std::string& name) const {
    auto it = entries_.find(name);
    return it != entries_.end() && it->second.is_integer;
  }

 private:
  struct StateEntry {
    std::string scope;
    bool is_integer{false};
  };

  void Register(const tir::Buffer& buffer) {
    const std::string scope = buffer.scope();
    if (!IsTrackedStateScope(scope)) {
      return;
    }
    const std::string name = BufferIdentityName(buffer);
    if (name.empty()) {
      return;
    }
    if (entries_.count(name)) {
      return;
    }
    entries_.emplace(name, StateEntry{scope, buffer->dtype.is_int() || buffer->dtype.is_uint()});
    order_.push_back(name);
  }

  std::unordered_map<std::string, StateEntry> entries_;
  std::vector<std::string> order_;
};

void PushStringUnique(Array<Any>* arr, std::unordered_set<std::string>* seen,
                      const std::string& value) {
  if (seen->insert(value).second) {
    arr->push_back(String(value));
  }
}

SemanticWitness MakeWitness(const std::string& subject_kind, const std::string& subject_anchor_id,
                            const std::string& fact_axis, Map<String, Any> fact_value,
                            Array<String> related_anchor_ids,
                            Array<String> evidence_sources) {
  return SemanticWitness(String(subject_kind), String(subject_anchor_id), String(fact_axis),
                         std::move(fact_value), std::move(related_anchor_ids),
                         std::move(evidence_sources),
                         String("analyze_semantic_structure"));
}

Array<String> EvidenceSourceArray(const std::string& source) {
  return Array<String>{String(source)};
}

std::string LookupEvidenceSource(
    const std::unordered_map<std::string, std::string>& evidence_sources,
    const std::string& key, const char* fallback) {
  auto it = evidence_sources.find(key);
  return it != evidence_sources.end() ? it->second : std::string(fallback);
}

// ---------------------------------------------------------------------------
// EvidenceAccumulator — collects merged evidence from manifest + fragment_regions
// ---------------------------------------------------------------------------

struct EvidenceAccumulator {
  Array<Any> states;
  std::unordered_map<std::string, int> state_index;
  std::unordered_set<std::string> reduction_targets;
  std::unordered_set<std::string> arg_reduce_targets;
  std::unordered_set<std::string> integer_states;
  std::unordered_set<std::string> loop_carried_states;
  std::unordered_set<std::string> selection_targets;
  std::unordered_map<std::string, Array<Any>> update_sources_by_target;
  std::unordered_map<std::string, std::string> paired_value_state_by_selection_target;
  std::unordered_set<std::string> paired_selection_companions;
  std::unordered_map<std::string, Array<Any>> recurrence_edges_by_target;

  // Reduction evidence from manifest and/or fragment_regions.
  struct ReductionEntry {
    std::string target;
    std::string kind;           // "max" / "sum"
    std::string evidence_source;
  };
  std::vector<ReductionEntry> reductions;
  std::unordered_set<std::string> seen_reduction_targets;

  // Evidence-source tracking per fact.
  std::unordered_map<std::string, std::string> arg_reduce_target_evidence_sources;
  std::unordered_map<std::string, std::string> loop_carried_state_evidence_sources;
  std::unordered_map<std::string, std::string> selection_target_evidence_sources;
  std::unordered_map<std::string, std::string> update_source_evidence_sources;
  std::unordered_map<std::string, std::string> selection_pair_evidence_sources;
  std::unordered_map<std::string, std::string> recurrence_edge_evidence_sources;
  std::unordered_map<std::string, std::string> reduction_evidence_sources;

  void RegisterState(const std::string& name, const std::string& role,
                     const std::string& scope) {
    auto it = state_index.find(name);
    if (it != state_index.end()) {
      auto entry = tvm::Downcast<Map<String, Any>>(states[it->second]);
      entry.Set("role", String(role));
      if (!scope.empty()) {
        entry.Set(schema_key::kScope, String(scope));
      }
      states.Set(it->second, entry);
      return;
    }
    Map<String, Any> entry;
    entry.Set(schema_key::kName, String(name));
    entry.Set("role", String(role));
    entry.Set(schema_key::kScope, String(scope));
    state_index.emplace(name, static_cast<int>(states.size()));
    states.push_back(entry);
  }

  void RegisterStringFact(std::unordered_set<std::string>* values,
                          std::unordered_map<std::string, std::string>* evidence_sources,
                          const std::string& value, const std::string& evidence_source) {
    if (values->insert(value).second) {
      evidence_sources->emplace(value, evidence_source);
    }
  }

  void RegisterArrayFact(std::unordered_map<std::string, Array<Any>>* values,
                         std::unordered_map<std::string, std::string>* evidence_sources,
                         const std::string& key, const Array<Any>& value,
                         const std::string& evidence_source) {
    if (!values->count(key)) {
      values->emplace(key, value);
      evidence_sources->emplace(key, evidence_source);
    }
  }

  void RegisterSelectionPair(const std::string& companion_target,
                             const std::string& value_target,
                             const std::string& evidence_source) {
    if (!paired_value_state_by_selection_target.count(companion_target)) {
      paired_value_state_by_selection_target[companion_target] = value_target;
      paired_selection_companions.insert(companion_target);
      selection_pair_evidence_sources[companion_target] = evidence_source;
    }
  }

  void RegisterReduction(const std::string& target, const std::string& kind,
                         const std::string& evidence_source) {
    if (seen_reduction_targets.insert(target).second) {
      reductions.push_back({target, kind, evidence_source});
      reduction_evidence_sources[target] = evidence_source;
    }
  }

  // Ingest a structural region from either manifest or fragment_regions.
  void IngestStructuralRegion(const Map<String, Any>& region, bool from_manifest,
                              const LocalBufferCollector& buffer_collector) {
    const std::string source_tag = from_manifest ? "semantic_manifest" : "fragment_regions";
    const std::string selection_target_source =
        from_manifest ? "semantic_manifest" : "selection_targets";
    const std::string update_source_source =
        from_manifest ? "semantic_manifest" : "update_sources";
    const std::string arg_reduce_source =
        from_manifest ? "semantic_manifest" : "fragment_regions";
    const std::string selection_pair_source =
        from_manifest ? "semantic_manifest" : "selection_pairs";
    const std::string recurrence_edge_source =
        from_manifest ? "semantic_manifest" : "recurrence_edges";
    const std::string loop_carried_source =
        from_manifest ? "semantic_manifest" : "loop_carried_state";

    if (region.count(manifest_key::kFragmentBuffers)) {
      for (const Any& buffer_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kFragmentBuffers])) {
        auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
        const std::string name =
            ResolveStateNameFromMap(buffer, schema_key::kName, schema_key::kBuffer);
        RegisterState(name, ToString(StateRole::kTransient),
                      buffer[schema_key::kScope].cast<String>());
        bool is_integer = buffer_collector.HasIntegerDType(name);
        if (auto it = buffer.find(schema_key::kIsInteger); it != buffer.end()) {
          is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
        }
        if (is_integer) {
          integer_states.insert(name);
        }
      }
    }
    if (region.count(manifest_key::kLoopCarriedState)) {
      for (const Any& carried_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kLoopCarriedState])) {
        auto carried = tvm::Downcast<Map<String, Any>>(carried_any);
        const std::string name =
            ResolveStateNameFromMap(carried, schema_key::kName, schema_key::kBuffer);
        RegisterStringFact(&loop_carried_states, &loop_carried_state_evidence_sources, name,
                           loop_carried_source);
        RegisterState(name, ToString(StateRole::kCarry), "");
      }
    }
    if (region.count(manifest_key::kSelectionTargets)) {
      for (const Any& target_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kSelectionTargets])) {
        RegisterStringFact(&selection_targets, &selection_target_evidence_sources,
                           ResolveStateName(target_any), selection_target_source);
      }
    }
    if (region.count(manifest_key::kUpdateSources)) {
      for (const Any& source_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kUpdateSources])) {
        auto source_map = tvm::Downcast<Map<String, Any>>(source_any);
        RegisterArrayFact(&update_sources_by_target, &update_source_evidence_sources,
                          ResolveStateNameFromMap(source_map, schema_key::kTarget,
                                                  schema_key::kTargetBuffer),
                          ResolveStateArray(source_map, schema_key::kSources,
                                            schema_key::kSourceBuffers),
                          update_source_source);
      }
    }
    if (region.count(manifest_key::kArgReduceTargets)) {
      for (const Any& target_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kArgReduceTargets])) {
        RegisterStringFact(&arg_reduce_targets, &arg_reduce_target_evidence_sources,
                           ResolveStateName(target_any), arg_reduce_source);
      }
    }
    if (region.count(manifest_key::kSelectionPairs)) {
      for (const Any& pair_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kSelectionPairs])) {
        auto pair_map = tvm::Downcast<Map<String, Any>>(pair_any);
        RegisterSelectionPair(
            ResolveStateNameFromMap(pair_map, schema_key::kCompanionTarget,
                                    schema_key::kCompanionBuffer),
            ResolveStateNameFromMap(pair_map, schema_key::kValueTarget,
                                    schema_key::kValueBuffer),
                              selection_pair_source);
      }
    }
    if (region.count(manifest_key::kRecurrenceEdges)) {
      for (const Any& edge_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kRecurrenceEdges])) {
        auto edge_map = tvm::Downcast<Map<String, Any>>(edge_any);
        RegisterArrayFact(&recurrence_edges_by_target, &recurrence_edge_evidence_sources,
                          ResolveStateNameFromMap(edge_map, schema_key::kTarget,
                                                  schema_key::kTargetBuffer),
                          ResolveStateArray(edge_map, schema_key::kSourceStates,
                                            schema_key::kSourceBuffers),
                          recurrence_edge_source);
        RegisterStringFact(&loop_carried_states, &loop_carried_state_evidence_sources,
                           ResolveStateNameFromMap(edge_map, schema_key::kTarget,
                                                   schema_key::kTargetBuffer),
                           loop_carried_source);
        RegisterState(ResolveStateNameFromMap(edge_map, schema_key::kTarget,
                                              schema_key::kTargetBuffer),
                      ToString(StateRole::kCarry), "");
      }
    }
    // row_reductions — manifest-first: only ingest if not already registered.
    if (region.count(manifest_key::kRowReductions)) {
      for (const Any& reduction_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kRowReductions])) {
        auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
        auto kind = reduction.Get(String(schema_key::kKind));
        if (!kind.has_value()) {
          continue;
        }
        RegisterReduction(ResolveStateNameFromMap(reduction, schema_key::kTarget,
                                                  schema_key::kTargetBuffer),
                          kind.value().cast<String>(),
                          source_tag);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Phase 1: Collect domain skeleton
// ---------------------------------------------------------------------------

void CollectDomainSkeleton(const tir::PrimFunc& func,
                           Array<Any>* domain_axes,
                           Array<Any>* domain_traits,
                           std::unordered_set<std::string>* seen_traits) {
  if (auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
    if (auto axes = work.value().find("axes"); axes != work.value().end()) {
      *domain_axes = tvm::Downcast<Array<Any>>((*axes).second);
    }
    if (auto bounds = work.value().find("work_dependent_loop_bounds");
        bounds != work.value().end() &&
        !tvm::Downcast<Array<Any>>((*bounds).second).empty()) {
      PushStringUnique(domain_traits, seen_traits, "work_dependent_bounds");
    }
    if (auto derived = work.value().find("derived_index_exprs");
        derived != work.value().end() &&
        !tvm::Downcast<Array<Any>>((*derived).second).empty()) {
      PushStringUnique(domain_traits, seen_traits, "derived_indices");
    }
  }
  if (auto pipeline = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
      pipeline && !pipeline.value().empty()) {
    PushStringUnique(domain_traits, seen_traits, "pipeline");
  }
}

// ---------------------------------------------------------------------------
// Phase 2: Ingest evidence from manifest and fragment_regions
// ---------------------------------------------------------------------------

void IngestAllEvidence(const tir::PrimFunc& func, EvidenceAccumulator* acc,
                       const LocalBufferCollector& buffer_collector) {
  // Manifest-first: ingest structural regions from manifest before fragment_regions.
  if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
    if (auto structural_it = manifest.value().find(manifest_key::kStructuralRegions);
        structural_it != manifest.value().end()) {
      for (const Any& region_any : tvm::Downcast<Array<Any>>((*structural_it).second)) {
        acc->IngestStructuralRegion(
            tvm::Downcast<Map<String, Any>>(region_any), true, buffer_collector);
      }
    }
  }

  // Fragment regions: fallback for evidence not already present in manifest.
  if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
    for (const Any& region_any : regions.value()) {
      auto region = tvm::Downcast<Map<String, Any>>(region_any);
      // Register fragment buffers for state tracking.
      for (const Any& buffer_any :
           tvm::Downcast<Array<Any>>(region[manifest_key::kFragmentBuffers])) {
        auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
        const std::string name =
            ResolveStateNameFromMap(buffer, schema_key::kName, schema_key::kBuffer);
        acc->RegisterState(name, ToString(StateRole::kTransient),
                           buffer[schema_key::kScope].cast<String>());
        bool is_integer = buffer_collector.HasIntegerDType(name);
        if (auto it = buffer.find(schema_key::kIsInteger); it != buffer.end()) {
          is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
        }
        if (is_integer) {
          acc->integer_states.insert(name);
        }
      }
      // Structural evidence (manifest-first dedup handled inside IngestStructuralRegion).
      acc->IngestStructuralRegion(region, false, buffer_collector);
    }
  } else {
    // No fragment regions at all — fall back to buffer collector.
    for (const Any& state_any : buffer_collector.Encode()) {
      auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
      const std::string name = state_map[schema_key::kName].cast<String>();
      acc->RegisterState(name, state_map["role"].cast<String>(),
                         state_map[schema_key::kScope].cast<String>());
      if (buffer_collector.HasIntegerDType(name)) {
        acc->integer_states.insert(name);
      }
    }
  }

  // Apply reduction evidence to state roles.
  for (const auto& red : acc->reductions) {
    acc->reduction_targets.insert(red.target);
    if (buffer_collector.HasIntegerDType(red.target)) {
      acc->integer_states.insert(red.target);
    }
    // A reduction target is index_state only if it actually carries index information:
    // either it has integer dtype, or it is an integer arg-reduce target.  Non-integer
    // arg_reduce_targets (e.g. the value component of an arg-reduce pair) remain
    // reduction_accumulator — they participate in arg-reduce but don't carry indices.
    const bool is_index = acc->integer_states.count(red.target) ||
                          (acc->arg_reduce_targets.count(red.target) &&
                           buffer_collector.HasIntegerDType(red.target));
    const std::string role = is_index ? ToString(StateRole::kIndexState)
                                      : ToString(StateRole::kReductionAccumulator);
    acc->RegisterState(red.target, role, "");
  }

  // Refine roles: carry and selection.
  for (const std::string& carried : acc->loop_carried_states) {
    if (!acc->reduction_targets.count(carried)) {
      acc->RegisterState(carried, ToString(StateRole::kCarry), "");
    }
  }
  for (const std::string& name : acc->selection_targets) {
    if (acc->paired_selection_companions.count(name)) {
      acc->RegisterState(name, ToString(StateRole::kIndexState), "");
    } else if (!acc->integer_states.count(name)) {
      acc->RegisterState(name, ToString(StateRole::kSelectionState), "");
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 3: Emit witnesses from accumulated evidence
// ---------------------------------------------------------------------------

void EmitStateRoleWitnesses(const EvidenceAccumulator& acc,
                            Array<SemanticWitness>* witnesses) {
  for (const Any& state_any : acc.states) {
    auto state_map = tvm::Downcast<Map<String, Any>>(state_any);
    auto role = ParseStateRole(state_map["role"].cast<String>());
    ICHECK(role) << "AnalyzeSemanticStructure encountered unsupported state role "
                 << state_map["role"].cast<String>();
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kState),
                                     state_map[schema_key::kName].cast<String>(),
                                     ToString(WitnessFactAxis::kRole),
                                     MakeStateRolePayload(*role), Array<String>{},
                                     Array<String>{String("states")}));
  }
  for (const std::string& target : acc.arg_reduce_targets) {
    if (!acc.integer_states.count(target)) {
      continue;
    }
    const std::string evidence_source =
        LookupEvidenceSource(acc.arg_reduce_target_evidence_sources, target, "fragment_regions");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), target,
                                     ToString(WitnessFactAxis::kDerivesIndexFrom),
                                     MakeEmptyPayload(), Array<String>{},
                                     EvidenceSourceArray(evidence_source)));
  }
}

void EmitReductionUpdates(const EvidenceAccumulator& acc,
                          Array<Any>* updates,
                          Array<SemanticWitness>* witnesses) {
  for (const auto& red : acc.reductions) {
    Map<String, Any> entry;
    const std::string update_name = std::string("reduce_") + red.target;
    entry.Set("name", String(update_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kReduce)));
    entry.Set("target_state", String(red.target));
    entry.Set("reduce_kind", String(red.kind));
    if (auto it = acc.update_sources_by_target.find(red.target);
        it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    updates->push_back(entry);

    const std::string evidence_source =
        LookupEvidenceSource(acc.reduction_evidence_sources, red.target, "row_reductions");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kReduce),
                                     Array<String>{},
                                     EvidenceSourceArray(evidence_source)));
    if (auto it = acc.update_sources_by_target.find(red.target);
        it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, red.target, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
  }
}

void EmitSelectionUpdates(const EvidenceAccumulator& acc,
                          Array<Any>* updates,
                          Array<SemanticWitness>* witnesses) {
  for (const std::string& state_name : acc.selection_targets) {
    Map<String, Any> entry;
    entry.Set("name", String(std::string("select_") + state_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kSelect)));
    entry.Set("target_state", String(state_name));
    entry.Set("traits", Array<Any>{String("selected"), String("indexed")});
    if (auto it = acc.update_sources_by_target.find(state_name);
        it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    if (auto it = acc.paired_value_state_by_selection_target.find(state_name);
        it != acc.paired_value_state_by_selection_target.end()) {
      Array<Any> bindings;
      Map<String, Any> binding;
      binding.Set("kind", String(ToString(BindingKind::kPairedValueState)));
      binding.Set("symbol", String("state"));
      binding.Set("value_repr", String(it->second));
      bindings.push_back(binding);
      entry.Set("bindings", bindings);
    }
    updates->push_back(entry);
    const std::string update_name = std::string("select_") + state_name;
    const std::string selection_target_source =
        LookupEvidenceSource(acc.selection_target_evidence_sources, state_name,
                             "selection_targets");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kSelect),
                                     Array<String>{},
                                     EvidenceSourceArray(selection_target_source)));
    if (auto it = acc.update_sources_by_target.find(state_name);
        it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, state_name, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
    if (auto it = acc.paired_value_state_by_selection_target.find(state_name);
        it != acc.paired_value_state_by_selection_target.end()) {
      const std::string selection_pair_source =
          LookupEvidenceSource(acc.selection_pair_evidence_sources, state_name,
                               "selection_pairs");
      witnesses->push_back(
          MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                      ToString(WitnessFactAxis::kCompanion),
                      MakeRelationBindingPayload(BindingKind::kPairedValueState),
                      Array<String>{String(it->second)},
                      EvidenceSourceArray(selection_pair_source)));
    }
  }
}

void EmitRecurrenceUpdates(const EvidenceAccumulator& acc,
                           Array<Any>* updates,
                           Array<SemanticWitness>* witnesses) {
  for (const std::string& state_name : acc.loop_carried_states) {
    Map<String, Any> entry;
    entry.Set("name", String(std::string("recur_") + state_name));
    entry.Set("kind", String(ToString(UpdateLawKind::kRecurrence)));
    entry.Set("target_state", String(state_name));
    entry.Set("traits", Array<Any>{String("carried"), String("staged")});
    if (auto it = acc.recurrence_edges_by_target.find(state_name);
        it != acc.recurrence_edges_by_target.end()) {
      entry.Set("source_states", it->second);
      Array<Any> bindings;
      for (const Any& source_any : it->second) {
        Map<String, Any> binding;
        binding.Set("kind", String(ToString(BindingKind::kRecurrenceSourceState)));
        binding.Set("symbol", String("state"));
        binding.Set("value_repr", tvm::Downcast<String>(source_any));
        bindings.push_back(binding);
      }
      entry.Set("bindings", bindings);
    } else if (auto it = acc.update_sources_by_target.find(state_name);
               it != acc.update_sources_by_target.end()) {
      entry.Set("source_states", it->second);
    }
    updates->push_back(entry);
    const std::string update_name = std::string("recur_") + state_name;
    const std::string loop_carried_source =
        LookupEvidenceSource(acc.loop_carried_state_evidence_sources, state_name,
                             "loop_carried_state");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kLawFamily),
                                     MakeUpdateLawFamilyPayload(UpdateLawKind::kRecurrence),
                                     Array<String>{},
                                     EvidenceSourceArray(loop_carried_source)));
    Map<String, Any> ordering_payload;
    ordering_payload.Set("ordering", String("ordered"));
    const std::string recurrence_edge_source =
        LookupEvidenceSource(acc.recurrence_edge_evidence_sources, state_name,
                             "recurrence_edges");
    witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                     ToString(WitnessFactAxis::kOrdering),
                                     std::move(ordering_payload), Array<String>{},
                                     EvidenceSourceArray(recurrence_edge_source)));
    if (auto it = acc.recurrence_edges_by_target.find(state_name);
        it != acc.recurrence_edges_by_target.end()) {
      Array<String> related_sources;
      for (const Any& source_any : it->second) {
        related_sources.push_back(tvm::Downcast<String>(source_any));
      }
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kRelation), update_name,
                                       ToString(WitnessFactAxis::kCarriedFrom),
                                       MakeRelationBindingPayload(
                                           BindingKind::kRecurrenceSourceState),
                                       std::move(related_sources),
                                       EvidenceSourceArray(recurrence_edge_source)));
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(recurrence_edge_source)));
    } else if (auto it = acc.update_sources_by_target.find(state_name);
               it != acc.update_sources_by_target.end()) {
      const std::string update_source =
          LookupEvidenceSource(acc.update_source_evidence_sources, state_name, "update_sources");
      witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), update_name,
                                       ToString(WitnessFactAxis::kSourceSet),
                                       MakeUpdateSourceSetPayload(it->second),
                                       Array<String>{},
                                       EvidenceSourceArray(update_source)));
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 4: Collect seeds and supplements from manifest
// ---------------------------------------------------------------------------

void CollectSeedsAndSupplements(const tir::PrimFunc& func,
                                Array<Any>* seeds,
                                Array<Any>* supplements,
                                Array<SemanticWitness>* witnesses) {
  std::unordered_set<std::string> seen_seed_markers;
  if (auto semantic_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticSeeds)) {
    if (auto capture = semantic_seeds.value().find("capture_kinds");
        capture != semantic_seeds.value().end()) {
      for (const Any& seed_any : tvm::Downcast<Array<Any>>((*capture).second)) {
        PushStringUnique(seeds, &seen_seed_markers, tvm::Downcast<String>(seed_any));
      }
    }
  }
  if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
    PushStringUnique(seeds, &seen_seed_markers, "explicit_op_manifest");

    Array<Any> manifest_op_kinds;
    std::unordered_set<std::string> seen_manifest_op_kinds;
    if (auto op_it = manifest.value().find(manifest_key::kOperations);
        op_it != manifest.value().end()) {
      for (const Any& op_any : tvm::Downcast<Array<Any>>((*op_it).second)) {
        auto op_map = tvm::Downcast<Map<String, Any>>(op_any);
        PushStringUnique(&manifest_op_kinds, &seen_manifest_op_kinds,
                         op_map["kind"].cast<String>());
      }
    }

    int ordered_region_count = 0;
    if (auto region_it = manifest.value().find(manifest_key::kOrderedRegions);
        region_it != manifest.value().end()) {
      for (const Any& region_any : tvm::Downcast<Array<Any>>((*region_it).second)) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        ++ordered_region_count;
        witnesses->push_back(MakeWitness(ToString(WitnessSubjectKind::kBoundary),
                                         region["anchor"].cast<String>(),
                                         ToString(WitnessFactAxis::kOrderedRegion),
                                         MakeEmptyPayload(), Array<String>{},
                                         Array<String>{String("semantic_manifest")}));
      }
    }

    Map<String, Any> supplement_payload;
    supplement_payload.Set("source", String("semantic_manifest"));
    supplement_payload.Set("operation_kinds", manifest_op_kinds);
    supplement_payload.Set("ordered_region_count", Integer(ordered_region_count));
    Map<String, Any> supplement;
    supplement.Set("kind", String(ToString(SupplementKind::kSemanticBoundary)));
    supplement.Set("payload", supplement_payload);
    supplements->push_back(supplement);
  }
  if (auto fragment_regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
    const auto logical_buffer_shapes = BuildLogicalBufferShapes(func);
    Array<Any> fragment_op_kinds;
    std::unordered_set<std::string> seen_fragment_ops;
    Array<Any> row_reduction_targets;
    std::unordered_set<std::string> seen_row_reduction_targets;
    Array<Any> row_broadcast_sources;
    std::unordered_set<std::string> seen_row_broadcast_sources;
    Array<Any> pointwise_op_kinds;
    std::unordered_set<std::string> seen_pointwise_ops;
    Array<Any> fragment_loop_carried_state;
    std::unordered_set<std::string> seen_loop_carried_state;
    Array<Any> fragment_layout_contracts;
    std::unordered_set<std::string> seen_fragment_layout_contracts;
    for (const Any& region_any : fragment_regions.value()) {
      auto region = tvm::Downcast<Map<String, Any>>(region_any);
      CollectUniqueStringField(&fragment_op_kinds, &seen_fragment_ops, region, "ops");
      CollectUniqueStringField(&row_reduction_targets, &seen_row_reduction_targets, region,
                               manifest_key::kRowReductions, schema_key::kTarget);
      CollectUniqueStringField(&row_broadcast_sources, &seen_row_broadcast_sources, region,
                               "row_broadcasts", schema_key::kSource);
      CollectUniqueStringField(&pointwise_op_kinds, &seen_pointwise_ops, region,
                               "pointwise_ops");
      CollectUniqueStringField(&fragment_loop_carried_state, &seen_loop_carried_state, region,
                               manifest_key::kLoopCarriedState, schema_key::kName);
      CollectUniqueNestedMapField(&fragment_layout_contracts,
                                  &seen_fragment_layout_contracts, region,
                                  schema_key::kFragmentLayoutContracts,
                                  schema_key::kBuffer);
    }
    Array<Any> fragment_materialization_contracts =
        CollectFragmentMaterializationContractsFromBody(func->body);
    Array<Any> fragment_buffer_flow_contracts =
        CollectFragmentBufferFlowContractsFromBody(func->body);
    AppendUniqueFragmentMaterializationContractsFromFlowContracts(
        fragment_buffer_flow_contracts, &fragment_materialization_contracts);
    AppendUniqueCastDrivenFragmentMaterializationContractsFromBody(
        func->body, fragment_buffer_flow_contracts, logical_buffer_shapes,
        &fragment_materialization_contracts);
    if (!fragment_op_kinds.empty()) {
      PushStringUnique(seeds, &seen_seed_markers, "fragment_region_analysis");
      Map<String, Any> supplement_payload;
      supplement_payload.Set(String(schema_key::kSource), String("blackhole.fragment_regions"));
      supplement_payload.Set(String(schema_key::kFragmentOpKinds), fragment_op_kinds);
      if (!row_reduction_targets.empty()) {
        supplement_payload.Set(String(schema_key::kRowReductionTargets), row_reduction_targets);
      }
      if (!row_broadcast_sources.empty()) {
        supplement_payload.Set(String(schema_key::kRowBroadcastSources), row_broadcast_sources);
      }
      if (!pointwise_op_kinds.empty()) {
        supplement_payload.Set(String(schema_key::kPointwiseOpKinds), pointwise_op_kinds);
      }
      if (!fragment_loop_carried_state.empty()) {
        supplement_payload.Set(String(schema_key::kFragmentLoopCarriedState),
                               fragment_loop_carried_state);
      }
      if (!fragment_layout_contracts.empty()) {
        supplement_payload.Set(String(schema_key::kFragmentLayoutContracts),
                               fragment_layout_contracts);
      }
      if (!fragment_materialization_contracts.empty()) {
        supplement_payload.Set(String(schema_key::kFragmentMaterializationContracts),
                               fragment_materialization_contracts);
      }
      if (!fragment_buffer_flow_contracts.empty()) {
        supplement_payload.Set(String(schema_key::kFragmentBufferFlowContracts),
                               fragment_buffer_flow_contracts);
      }
      Map<String, Any> supplement;
      supplement.Set("kind", String(ToString(SupplementKind::kFragmentLoweringStructure)));
      supplement.Set("payload", supplement_payload);
      supplements->push_back(supplement);
    }
  }
  if (auto pipeline_stages = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
      pipeline_stages && !pipeline_stages.value().empty()) {
    PushStringUnique(seeds, &seen_seed_markers, "pipeline_stage_analysis");
    Map<String, Any> supplement_payload;
    supplement_payload.Set(String(schema_key::kSource), String("blackhole.pipeline_stages"));
    supplement_payload.Set(String(schema_key::kPipelineStages), pipeline_stages.value());
    Map<String, Any> supplement;
    supplement.Set("kind", String(ToString(SupplementKind::kPipelineStructure)));
    supplement.Set("payload", supplement_payload);
    supplements->push_back(supplement);
  }
  if (auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
    if (auto bounds = work.value().Get(String(schema_key::kWorkDependentLoopBounds))) {
      Array<Any> loop_bounds = tvm::Downcast<Array<Any>>(bounds.value());
      if (!loop_bounds.empty()) {
        PushStringUnique(seeds, &seen_seed_markers, "work_decomposition_analysis");
        Map<String, Any> supplement_payload;
        supplement_payload.Set(String(schema_key::kSource), String("blackhole.work_decomposition"));
        supplement_payload.Set(String(schema_key::kWorkDependentLoopBounds), loop_bounds);
        Map<String, Any> supplement;
        supplement.Set("kind", String(ToString(SupplementKind::kWorkDecompositionStructure)));
        supplement.Set("payload", supplement_payload);
        supplements->push_back(supplement);
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Pass entry point
// ---------------------------------------------------------------------------

tir::transform::Pass AnalyzeSemanticStructure() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }

    // Phase 1: domain skeleton.
    Array<Any> domain_axes;
    Array<Any> domain_traits;
    std::unordered_set<std::string> seen_traits;
    CollectDomainSkeleton(func, &domain_axes, &domain_traits, &seen_traits);

    // Phase 2: evidence ingestion.
    LocalBufferCollector buffer_collector;
    buffer_collector(func->body);
    EvidenceAccumulator acc;
    IngestAllEvidence(func, &acc, buffer_collector);

    // Phase 3: emit witnesses and updates.
    Array<SemanticWitness> witnesses;
    EmitStateRoleWitnesses(acc, &witnesses);

    Array<Any> updates;
    {
      Map<String, Any> entry;
      entry.Set("name", String("root_map"));
      entry.Set("kind", String(ToString(UpdateLawKind::kMap)));
      String root_target("");
      if (acc.states.size() == 1) {
        root_target =
            tvm::Downcast<Map<String, Any>>(acc.states[0])[schema_key::kName].cast<String>();
      }
      entry.Set("target_state", root_target);
      updates.push_back(entry);
      witnesses.push_back(MakeWitness(ToString(WitnessSubjectKind::kUpdate), "root_map",
                                      ToString(WitnessFactAxis::kLawFamily),
                                      MakeUpdateLawFamilyPayload(UpdateLawKind::kMap),
                                      Array<String>{},
                                      Array<String>{String("semantic_structure")}));
    }
    EmitReductionUpdates(acc, &updates, &witnesses);
    if (!acc.selection_targets.empty()) {
      EmitSelectionUpdates(acc, &updates, &witnesses);
    }
    if (!acc.loop_carried_states.empty()) {
      EmitRecurrenceUpdates(acc, &updates, &witnesses);
    }

    // Phase 4: seeds and supplements.
    Array<Any> seeds;
    Array<Any> supplements;
    CollectSeedsAndSupplements(func, &seeds, &supplements, &witnesses);

    // Assemble structure.
    Map<String, Any> structure;
    structure.Set("domain_name", String("device_program"));
    structure.Set("domain_axes", domain_axes);
    structure.Set("domain_traits", domain_traits);
    structure.Set("states", acc.states);
    structure.Set("updates", updates);
    structure.Set("seeds", seeds);
    structure.Set("supplements", supplements);

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticStructure, structure);
    attrs.Set(attr::kTLSemanticWitnesses, witnesses);
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.AnalyzeSemanticStructure", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnalyzeSemanticStructure", AnalyzeSemanticStructure);
}

}  // namespace tl
}  // namespace tvm
