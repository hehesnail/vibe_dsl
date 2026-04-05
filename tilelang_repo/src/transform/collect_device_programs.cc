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
 * \file collect_device_programs.cc
 * \brief Collect module-scope device-program registry before SplitHostDevice.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

TLDeviceProgramInfo::TLDeviceProgramInfo(ffi::String root_symbol,
                                         ffi::Array<ffi::String> member_funcs) {
  auto n = ffi::make_object<TLDeviceProgramInfoNode>();
  n->root_symbol = std::move(root_symbol);
  n->member_funcs = std::move(member_funcs);
  data_ = std::move(n);
}

namespace {

class DeviceRegionCounter : public tir::StmtVisitor {
 public:
  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget && op->node.as<Target>()) {
      ++count_;
    }
    if (op->attr_key == tir::attr::thread_extent) {
      saw_launch_threads_ = true;
    }
    tir::StmtVisitor::VisitStmt_(op);
  }

  int count() const { return count_ == 0 && saw_launch_threads_ ? 1 : count_; }

 private:
  int count_{0};
  bool saw_launch_threads_{false};
};

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

ffi::Array<ffi::String> PlanKernelNames(const IRModule& mod, const tir::PrimFunc& func,
                                        int region_count, const GlobalVar& gvar) {
  ffi::Array<ffi::String> names;
  if (region_count <= 0) {
    return names;
  }
  GlobalVarSupply supply(mod);
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  std::string name_prefix = global_symbol.value_or(gvar->name_hint);
  std::string kernel_name = name_prefix + "_kernel";
  for (int i = 0; i < region_count; ++i) {
    names.push_back(supply->FreshGlobal(kernel_name, false)->name_hint);
  }
  return names;
}

}  // namespace

transform::Pass CollectDevicePrograms() {
  auto pass_func = [](IRModule mod, transform::PassContext ctx) {
    ffi::Array<GlobalInfo> programs;
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }

      DeviceRegionCounter counter;
      counter(func.value()->body);
      if (counter.count() == 0) {
        continue;
      }

      ffi::Array<ffi::String> member_funcs =
          PlanKernelNames(mod, func.value(), counter.count(), gvar);
      programs.push_back(TLDeviceProgramInfo(gvar->name_hint, member_funcs));
    }

    mod = mod->ShallowCopy();
    mod->UpdateGlobalInfo(attr::kTLDevicePrograms, programs);
    return mod;
  };
  return transform::CreateModulePass(pass_func, 0, "tl.transform.CollectDevicePrograms", {});
}

TVM_FFI_STATIC_INIT_BLOCK() { TLDeviceProgramInfoNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TLDeviceProgramInfo", [](ffi::String root_symbol,
                                                     ffi::Array<ffi::String> member_funcs) {
    return TLDeviceProgramInfo(std::move(root_symbol), std::move(member_funcs));
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CollectDevicePrograms", CollectDevicePrograms);
}

}  // namespace tl
}  // namespace tvm
