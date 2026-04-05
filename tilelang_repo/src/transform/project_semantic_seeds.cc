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
 * \brief Project lightweight pre-lift semantic seeds and explicit companion invalidation.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <string>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Map;
using tvm::ffi::String;

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

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

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

tir::PrimFunc StripCompanionAttrs(const tir::PrimFunc& func, const ffi::String& reason) {
  tir::PrimFunc updated = func;
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticStructure);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticWitnesses);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSemanticProgram);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLSpatialProgram);
  updated = tvm::WithoutAttr(std::move(updated), attr::kTLTTProgram);
  updated = WithAttrs(
      std::move(updated),
      {{attr::kTLCompanionInvalidationReason, reason},
       {attr::kTLSemanticHardFreeze,
        Map<String, Any>{{"state", ffi::String("invalidated")},
                         {"contract_mode", ffi::String("invalidate")},
                         {"unsafe_mutation_policy", ffi::String("require_relift")},
                         {"invalidation_reason", reason}}}});
  return updated;
}

}  // namespace

transform::Pass ProjectSemanticSeeds() {
  auto fpass = [](tir::PrimFunc func, IRModule, transform::PassContext) -> tir::PrimFunc {
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

transform::Pass InvalidateBlackholeCompanionPrograms(ffi::String reason) {
  auto fpass = [reason](tir::PrimFunc func, IRModule, transform::PassContext) -> tir::PrimFunc {
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
  refl::GlobalDef().def("tl.transform.ProjectSemanticSeeds", ProjectSemanticSeeds);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InvalidateBlackholeCompanionPrograms",
                        InvalidateBlackholeCompanionPrograms);
}

}  // namespace tl
}  // namespace tvm
