/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file project_semantic_seeds.cc
 * \brief Project lightweight pre-lift semantic seeds, explicit-op manifest evidence, and
 * companion invalidation.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>
#include <unordered_set>
#include <vector>

#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm_py.h"
#include "../op/operator.h"
#include "../op/reduce.h"
#include "common/blackhole_utils.h"
#include "common/fragment_region_analysis.h"
#include "common/semantic_program.h"
#include "common/semantic_vocab.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;
using tvm::Integer;
using namespace tvm::tl::semantic;

namespace {

class SeedCollector : public tir::StmtVisitor {
 public:
  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget && op->node.as<Target>()) {
      ++device_region_count_;
    }
    if (op->attr_key == tir::attr::thread_extent) {
      saw_launch_threads_ = true;
    }
    tir::StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const tir::ForNode* op) final {
    if (op->annotations.defined()) {
      if (op->annotations.find("num_stages") != op->annotations.end() ||
          op->annotations.find("software_pipeline_stage") != op->annotations.end()) {
        has_pipeline_stage_skeleton_ = true;
      }
    }
    tir::StmtVisitor::VisitStmt_(op);
  }

  int device_region_count() const {
    return device_region_count_ == 0 && saw_launch_threads_ ? 1 : device_region_count_;
  }
  bool has_pipeline_stage_skeleton() const { return has_pipeline_stage_skeleton_; }

 private:
  int device_region_count_{0};
  bool has_pipeline_stage_skeleton_{false};
  bool saw_launch_threads_{false};
};

ffi::Array<ffi::String> PlannedKernelNames(const ffi::String& root_symbol, int region_count) {
  ffi::Array<ffi::String> names;
  if (region_count <= 0) {
    return names;
  }
  std::string base = std::string(root_symbol) + "_kernel";
  names.push_back(base);
  for (int i = 1; i < region_count; ++i) {
    names.push_back(base + "_" + std::to_string(i));
  }
  return names;
}

Map<String, Any> MakeSeedPayload(const tir::PrimFunc& func, const ffi::String& root_symbol) {
  SeedCollector collector;
  collector(func->body);

  Map<String, Any> seeds;
  ffi::Array<ffi::Any> device_kernel_regions;
  for (const auto& name : PlannedKernelNames(root_symbol, collector.device_region_count())) {
    device_kernel_regions.push_back(name);
  }
  ffi::Array<ffi::Any> capture_kinds;
  if (collector.device_region_count() > 0) {
    capture_kinds.push_back(ffi::String("device_program_membership"));
  }
  if (collector.has_pipeline_stage_skeleton()) {
    capture_kinds.push_back(ffi::String("pipeline_stage_skeleton"));
  }
  seeds.Set("device_kernel_regions", device_kernel_regions);
  seeds.Set("capture_kinds", capture_kinds);
  return seeds;
}

const char* ReduceTypeToString(const ReduceType& type) {
  if (type->isSum()) return "sum";
  if (type->isAbsSum()) return "abssum";
  if (type->isMax()) return "max";
  if (type->isMin()) return "min";
  if (type->isAbsMax()) return "absmax";
  if (type->isBitAnd()) return "bitand";
  if (type->isBitOr()) return "bitor";
  if (type->isBitXor()) return "bitxor";
  return "unknown";
}

Map<String, Any> EncodeManifestAnchor(const String& anchor, const String& kind,
                                      const String& capture_stage, int ordinal) {
  Map<String, Any> encoded;
  encoded.Set(schema_key::kAnchor, anchor);
  encoded.Set(schema_key::kKind, kind);
  encoded.Set(schema_key::kCaptureStage, capture_stage);
  encoded.Set("ordinal", Integer(ordinal));
  return encoded;
}

Map<String, Any> EncodeBufferDescriptor(const tir::Buffer& buffer) {
  Map<String, Any> descriptor;
  descriptor.Set(schema_key::kBuffer, buffer);
  descriptor.Set(schema_key::kName, String(BufferIdentityName(buffer)));
  descriptor.Set(schema_key::kScope, String(buffer.scope()));
  descriptor.Set(schema_key::kDType, buffer->dtype);
  descriptor.Set(schema_key::kShape, buffer->shape);
  return descriptor;
}

Map<String, Any> MergeManifest(const Map<String, Any>& base, const Map<String, Any>& extra) {
  if (base.empty()) {
    return extra;
  }
  if (extra.empty()) {
    return base;
  }

  Map<String, Any> merged = base;
  ffi::Array<Any> buffers;
  ffi::Array<Any> operations;
  ffi::Array<Any> ordered_regions;
  ffi::Array<Any> anchors;
  ffi::Array<Any> structural_regions;
  std::unordered_set<std::string> seen_buffer_names;

  auto append_buffers = [&](const Map<String, Any>& manifest) {
    if (auto it = manifest.find(manifest_key::kBuffers); it != manifest.end()) {
      for (const Any& buffer_any : tvm::Downcast<ffi::Array<Any>>((*it).second)) {
        auto descriptor = tvm::Downcast<Map<String, Any>>(buffer_any);
        const std::string name = descriptor[schema_key::kName].cast<String>();
        if (seen_buffer_names.insert(name).second) {
          buffers.push_back(descriptor);
        }
      }
    }
  };
  auto append_array = [](ffi::Array<Any>* dst, const Map<String, Any>& manifest, const char* key) {
    if (auto it = manifest.find(String(key)); it != manifest.end()) {
      for (const Any& item : tvm::Downcast<ffi::Array<Any>>((*it).second)) {
        dst->push_back(item);
      }
    }
  };

  append_buffers(base);
  append_buffers(extra);
  append_array(&operations, base, manifest_key::kOperations);
  append_array(&operations, extra, manifest_key::kOperations);
  append_array(&ordered_regions, base, manifest_key::kOrderedRegions);
  append_array(&ordered_regions, extra, manifest_key::kOrderedRegions);
  append_array(&anchors, base, manifest_key::kAnchors);
  append_array(&anchors, extra, manifest_key::kAnchors);
  append_array(&structural_regions, base, manifest_key::kStructuralRegions);
  append_array(&structural_regions, extra, manifest_key::kStructuralRegions);

  merged.Set(manifest_key::kBuffers, buffers);
  merged.Set(manifest_key::kOperations, operations);
  merged.Set(manifest_key::kOrderedRegions, ordered_regions);
  merged.Set(manifest_key::kAnchors, anchors);
  if (!structural_regions.empty()) {
    merged.Set(manifest_key::kStructuralRegions, structural_regions);
  }
  return merged;
}

Map<String, Any> EncodeStructuralManifestRegion(const Map<String, Any>& region,
                                                const String& capture_stage) {
  Map<String, Any> encoded;
  encoded.Set(schema_key::kCaptureStage, capture_stage);

  auto copy_field = [&](const char* key) {
    if (auto it = region.find(String(key)); it != region.end()) {
      encoded.Set(String(key), (*it).second);
    }
  };

  copy_field(manifest_key::kFragmentBuffers);
  copy_field(manifest_key::kSelectionTargets);
  copy_field(manifest_key::kSelectionPairs);
  copy_field(manifest_key::kArgReduceTargets);
  copy_field(manifest_key::kUpdateSources);
  copy_field(manifest_key::kLoopCarriedState);
  copy_field(manifest_key::kRecurrenceEdges);
  copy_field(manifest_key::kRowReductions);
  return encoded;
}

Map<String, Any> CollectStructuralManifestEvidence(const tir::PrimFunc& func,
                                                   const String& capture_stage) {
  Map<String, Any> encoded_regions = AnalyzeBlackholeFragmentRegionEvidence(func);
  if (encoded_regions.empty()) {
    encoded_regions = Map<String, Any>();
  }

  ffi::Array<Any> structural_regions;
  if (auto regions_it = encoded_regions.find("regions"); regions_it != encoded_regions.end()) {
    for (const Any& region_any : tvm::Downcast<ffi::Array<Any>>((*regions_it).second)) {
      auto region = tvm::Downcast<Map<String, Any>>(region_any);
      auto structural_region = EncodeStructuralManifestRegion(region, capture_stage);
      if (structural_region.size() > 1) {
        structural_regions.push_back(structural_region);
      }
    }
  }

  if (auto manifest = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifest)) {
    if (auto ops_it = manifest.value().find(String(manifest_key::kOperations));
        ops_it != manifest.value().end()) {
      ffi::Array<Any> row_reductions;
      std::unordered_set<std::string> seen_targets;
      for (const Any& op_any : tvm::Downcast<ffi::Array<Any>>((*ops_it).second)) {
        auto op = tvm::Downcast<Map<String, Any>>(op_any);
        auto kind = op.Get(String(schema_key::kKind));
        auto payload = op.Get(String(schema_key::kPayload));
        if (!kind.has_value() || !payload.has_value() ||
            tvm::Downcast<String>(kind.value()) != "reduce") {
          continue;
        }
        auto payload_map = tvm::Downcast<Map<String, Any>>(payload.value());
        auto dst = payload_map.Get(String("dst"));
        auto reduce_kind = payload_map.Get(String("reduce_kind"));
        if (!dst.has_value() || !reduce_kind.has_value()) {
          continue;
        }
        tir::BufferRegion dst_region = tvm::Downcast<tir::BufferRegion>(dst.value());
        const std::string target_name = BufferIdentityName(dst_region->buffer);
        if (!seen_targets.insert(target_name).second) {
          continue;
        }
        Map<String, Any> entry;
        entry.Set(schema_key::kTarget, String(target_name));
        entry.Set(schema_key::kTargetBuffer, dst_region->buffer);
        entry.Set(schema_key::kKind, tvm::Downcast<String>(reduce_kind.value()));
        row_reductions.push_back(entry);
      }
      if (!row_reductions.empty()) {
        Map<String, Any> structural_region;
        structural_region.Set(schema_key::kCaptureStage, capture_stage);
        structural_region.Set(manifest_key::kRowReductions, row_reductions);
        structural_regions.push_back(structural_region);
      }
    }
  }

  if (structural_regions.empty()) {
    return {};
  }

  Map<String, Any> manifest;
  manifest.Set(manifest_key::kStructuralRegions, structural_regions);
  return manifest;
}

class SemanticManifestCollector : public tir::StmtVisitor {
 public:
  SemanticManifestCollector(String capture_stage, bool collect_early_ops, bool collect_late_ops)
      : capture_stage_(std::move(capture_stage)),
        region_anchor_(String(std::string("ordered_region:") +
                              static_cast<std::string>(capture_stage_) + ":0")),
        collect_early_ops_(collect_early_ops),
        collect_late_ops_(collect_late_ops) {}

  Map<String, Any> Encode() const {
    if (operations_.empty()) {
      return {};
    }

    ffi::Array<Any> ordered_regions;
    {
      Map<String, Any> region;
      region.Set(schema_key::kAnchor, region_anchor_);
      region.Set(schema_key::kCaptureStage, capture_stage_);
      region.Set(schema_key::kOperations, ordered_region_operations_);
      ordered_regions.push_back(region);
    }

    ffi::Array<Any> anchors;
    anchors.push_back(EncodeManifestAnchor(region_anchor_, String("ordered_region"), capture_stage_, 0));
    for (const Any& anchor_any : anchors_) {
      anchors.push_back(anchor_any);
    }

    Map<String, Any> manifest;
    manifest.Set(manifest_key::kBuffers, buffers_);
    manifest.Set(manifest_key::kOperations, operations_);
    manifest.Set(manifest_key::kOrderedRegions, ordered_regions);
    manifest.Set(manifest_key::kAnchors, anchors);
    return manifest;
  }

 private:
  void VisitStmt_(const tir::EvaluateNode* op) final {
    if (const auto* call = op->value.as<tir::CallNode>()) {
      TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<tir::Call>(call));
      if (tile_op.defined()) {
        MaybeEmit(tile_op);
      }
    }
    tir::StmtVisitor::VisitStmt_(op);
  }

  void AppendBufferUnique(const tir::Buffer& buffer) {
    const std::string name = buffer->name;
    if (buffer_names_.insert(name).second) {
      buffers_.push_back(EncodeBufferDescriptor(buffer));
    }
  }

  void EmitOperation(const char* kind, Map<String, Any> payload,
                     const std::vector<tir::Buffer>& buffers) {
    const int ordinal = static_cast<int>(ordered_region_operations_.size());
    String anchor =
        String(std::string("operation:") + static_cast<std::string>(capture_stage_) + ":" +
               std::to_string(ordinal));
    ffi::Array<Any> buffer_names;
    for (const tir::Buffer& buffer : buffers) {
      AppendBufferUnique(buffer);
      buffer_names.push_back(String(buffer->name));
    }

    Map<String, Any> op;
    op.Set(schema_key::kAnchor, anchor);
    op.Set(schema_key::kKind, String(kind));
    op.Set(schema_key::kCaptureStage, capture_stage_);
    op.Set(schema_key::kOrderedRegion, region_anchor_);
    op.Set(schema_key::kBuffers, buffer_names);
    op.Set(schema_key::kPayload, payload);
    operations_.push_back(op);
    ordered_region_operations_.push_back(anchor);
    anchors_.push_back(EncodeManifestAnchor(anchor, String("operation"), capture_stage_, ordinal));
  }

  void MaybeEmit(const TileOperator& tile_op) {
    if (collect_early_ops_) {
      if (const auto* copy = tile_op.as<CopyNode>()) {
        Map<String, Any> payload;
        payload.Set("src", tir::BufferRegion(copy->src, copy->src_range));
        payload.Set("dst", tir::BufferRegion(copy->dst, copy->dst_range));
        EmitOperation("copy", payload, {copy->src, copy->dst});
        return;
      }
      if (const auto* fill = tile_op.as<FillNode>()) {
        Map<String, Any> payload;
        payload.Set("dst", tir::BufferRegion(fill->dst, fill->region));
        payload.Set("value", fill->value);
        EmitOperation("fill", payload, {fill->dst});
        return;
      }
      if (const auto* reduce = tile_op.as<ReduceOpNode>()) {
        Map<String, Any> payload;
        payload.Set("src", reduce->srcRegion_);
        payload.Set("dst", reduce->dstRegion_);
        payload.Set("reduce_kind", String(ReduceTypeToString(reduce->type)));
        payload.Set("dim", Integer(reduce->dim));
        payload.Set("clear", Integer(reduce->clear ? 1 : 0));
        EmitOperation("reduce", payload, {reduce->src, reduce->dst});
        return;
      }
      if (const auto* cumsum = tile_op.as<CumSumOpNode>()) {
        Map<String, Any> payload;
        payload.Set("src", cumsum->srcRegion_);
        payload.Set("dst", cumsum->dstRegion_);
        payload.Set("dim", Integer(cumsum->dim));
        payload.Set("reverse", Integer(cumsum->reverse ? 1 : 0));
        EmitOperation("cumsum", payload, {cumsum->src, cumsum->dst});
        return;
      }
    }

    if (collect_late_ops_) {
      if (const auto* gemm = tile_op.as<GemmPyNode>()) {
        Map<String, Any> payload;
        payload.Set("a", gemm->aRegion_);
        payload.Set("b", gemm->bRegion_);
        payload.Set("c", gemm->cRegion_);
        payload.Set("transpose_a", Integer(gemm->transA_ ? 1 : 0));
        payload.Set("transpose_b", Integer(gemm->transB_ ? 1 : 0));
        payload.Set("m", Integer(gemm->m_));
        payload.Set("n", Integer(gemm->n_));
        payload.Set("k", Integer(gemm->k_));
        EmitOperation("gemm_py", payload, {gemm->a_, gemm->b_, gemm->c_});
      }
    }
  }

  String capture_stage_;
  String region_anchor_;
  bool collect_early_ops_{false};
  bool collect_late_ops_{false};
  ffi::Array<Any> buffers_;
  ffi::Array<Any> operations_;
  ffi::Array<Any> ordered_region_operations_;
  ffi::Array<Any> anchors_;
  std::unordered_set<std::string> buffer_names_;
};

tir::PrimFunc StripCompanionAttrs(const tir::PrimFunc& func, const ffi::String& reason) {
  tir::PrimFunc updated = func;
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticStructure);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticWitnesses);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticProgram);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSpatialDomainPlan);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSpatialExecutionPlan);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSpatialProgram);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLTTProgram);
  updated = WithAttrs(
      std::move(updated),
      {{attr::kTLCompanionInvalidationReason, reason},
       {attr::kTLSemanticHardFreeze,
        Map<String, Any>{{"state", ffi::String("invalidated")},
                         {"contract_mode", ffi::String(ToString(ContractMode::kInvalidate))},
                         {"unsafe_mutation_policy", ffi::String("require_relift")},
                         {"invalidation_reason", reason}}}});
  return updated;
}

}  // namespace

tir::transform::Pass CollectSemanticManifestSeeds() {
  auto fpass = [](tir::PrimFunc func, IRModule, tvm::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    SemanticManifestCollector collector(String("early_capture"), true, false);
    collector(func->body);
    auto manifest = collector.Encode();
    if (manifest.empty()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticManifestSeeds, manifest);
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.CollectSemanticManifestSeeds", {});
}

tir::transform::Pass ProjectSemanticSeeds() {
  auto fpass = [](tir::PrimFunc func, IRModule, tvm::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto root_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or("main");
    auto seeds = MakeSeedPayload(func, root_symbol);

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticSeeds, seeds);
    attrs.Set(attr::kTLSemanticHardFreeze,
              Map<String, Any>{{"state", ffi::String("pre_lift_seeded")},
                               {"unsafe_mutation_policy",
                                ffi::String("invalidate_companion_programs")}});
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.ProjectSemanticSeeds", {});
}

tir::transform::Pass ProjectSemanticManifest() {
  auto fpass = [](tir::PrimFunc func, IRModule, tvm::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    auto manifest_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticManifestSeeds);
    if (!manifest_seeds || manifest_seeds.value().empty()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticManifest, manifest_seeds.value());
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.ProjectSemanticManifest", {});
}

tir::transform::Pass AugmentSemanticManifest() {
  auto fpass = [](tir::PrimFunc func, IRModule, tvm::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }

    SemanticManifestCollector collector(String("late_augment"), false, true);
    collector(func->body);
    auto augment = collector.Encode();
    augment =
        MergeManifest(augment, CollectStructuralManifestEvidence(func, String("late_augment")));
    if (augment.empty()) {
      return func;
    }

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    auto manifest = attrs.find(attr::kTLSemanticManifest);
    if (manifest != attrs.end()) {
      attrs.Set(attr::kTLSemanticManifest,
                MergeManifest(tvm::Downcast<Map<String, Any>>((*manifest).second), augment));
    } else {
      attrs.Set(attr::kTLSemanticManifest, augment);
    }
    tir::PrimFunc updated = func;
    updated.CopyOnWrite()->attrs = DictAttrs(attrs);
    return updated;
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.AugmentSemanticManifest", {});
}

tir::transform::Pass InvalidateBlackholeCompanionPrograms(ffi::String reason) {
  auto fpass = [reason](tir::PrimFunc func, IRModule,
                        tvm::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }
    return StripCompanionAttrs(func, reason);
  };
  return tir::transform::CreatePrimFuncPass(
      fpass, 0, "tl.transform.InvalidateBlackholeCompanionPrograms", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CollectSemanticManifestSeeds", CollectSemanticManifestSeeds);
  refl::GlobalDef().def("tl.transform.ProjectSemanticSeeds", ProjectSemanticSeeds);
  refl::GlobalDef().def("tl.transform.ProjectSemanticManifest", ProjectSemanticManifest);
  refl::GlobalDef().def("tl.transform.AugmentSemanticManifest", AugmentSemanticManifest);
  refl::GlobalDef().def("tl.transform.InvalidateBlackholeCompanionPrograms",
                        InvalidateBlackholeCompanionPrograms);
}

}  // namespace tl
}  // namespace tvm
