/*
 * \file analyze_semantic_structure.cc
 * \brief Build a minimal semantic-structure summary from existing Blackhole analysis attrs.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/semantic_program.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

std::string CanonicalBufferName(const std::string& name) {
  size_t pos = name.size();
  while (pos > 0 && std::isdigit(static_cast<unsigned char>(name[pos - 1]))) {
    --pos;
  }
  if (pos > 0 && pos < name.size() && name[pos - 1] == '_') {
    return name.substr(0, pos - 1);
  }
  return name;
}

bool IsTrackedStateScope(const std::string& scope) {
  return scope == "local" || scope == "local.fragment" || scope == "blackhole.acc";
}

class LocalStateCollector : public tir::StmtExprVisitor {
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
      state.Set("name", String(name));
      state.Set("role", String(entry.role));
      state.Set("scope", String(entry.scope));
      states.push_back(state);
    }
    return states;
  }

 private:
  struct StateEntry {
    std::string role;
    std::string scope;
  };

  void Register(const tir::Buffer& buffer) {
    const std::string scope = buffer.scope();
    if (!IsTrackedStateScope(scope)) {
      return;
    }
    const std::string name = CanonicalBufferName(buffer->name);
    if (entries_.count(name)) {
      return;
    }
    entries_.emplace(name, StateEntry{"compute_state", scope});
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

}  // namespace

tir::transform::Pass AnalyzeSemanticStructure() {
  auto fpass = [](tir::PrimFunc func, IRModule, tir::transform::PassContext) -> tir::PrimFunc {
    if (!IsBlackholePrimFunc(func)) {
      return func;
    }

    Map<String, Any> structure;
    Array<Any> domain_axes;
    Array<Any> domain_traits;
    std::unordered_set<std::string> seen_traits;

    if (auto work = func->GetAttr<Map<String, Any>>("blackhole.work_decomposition")) {
      if (auto axes = work.value().find("axes"); axes != work.value().end()) {
        domain_axes = tvm::Downcast<Array<Any>>((*axes).second);
      }
      if (auto bounds = work.value().find("work_dependent_loop_bounds");
          bounds != work.value().end() &&
          !tvm::Downcast<Array<Any>>((*bounds).second).empty()) {
        PushStringUnique(&domain_traits, &seen_traits, "work_dependent_bounds");
      }
      if (auto derived = work.value().find("derived_index_exprs");
          derived != work.value().end() &&
          !tvm::Downcast<Array<Any>>((*derived).second).empty()) {
        PushStringUnique(&domain_traits, &seen_traits, "derived_indices");
      }
    }

    if (auto pipeline = func->GetAttr<Array<Any>>("blackhole.pipeline_stages");
        pipeline && !pipeline.value().empty()) {
      PushStringUnique(&domain_traits, &seen_traits, "pipeline");
    }

    Array<Any> states;
    std::unordered_map<std::string, int> state_index;
    auto register_state = [&states, &state_index](const std::string& name, const std::string& role,
                                                  const std::string& scope) {
      auto it = state_index.find(name);
      if (it != state_index.end()) {
        auto entry = tvm::Downcast<Map<String, Any>>(states[it->second]);
        entry.Set("role", String(role));
        if (!scope.empty()) {
          entry.Set("scope", String(scope));
        }
        states.Set(it->second, entry);
        return;
      }
      Map<String, Any> entry;
      entry.Set("name", String(name));
      entry.Set("role", String(role));
      entry.Set("scope", String(scope));
      state_index.emplace(name, static_cast<int>(states.size()));
      states.push_back(entry);
    };

    if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
      for (const Any& region_any : regions.value()) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        for (const Any& buffer_any : tvm::Downcast<Array<Any>>(region["fragment_buffers"])) {
          auto buffer = tvm::Downcast<Map<String, Any>>(buffer_any);
          register_state(buffer["name"].cast<String>(), "compute_state",
                         buffer["scope"].cast<String>());
        }
        for (const Any& carried_any : tvm::Downcast<Array<Any>>(region["loop_carried_state"])) {
          auto carried = tvm::Downcast<Map<String, Any>>(carried_any);
          register_state(carried["name"].cast<String>(), "carry_state", "");
        }
      }
    } else {
      LocalStateCollector collector;
      collector(func->body);
      states = collector.Encode();
    }

    Array<Any> updates;
    {
      Map<String, Any> entry;
      entry.Set("name", String("root_map"));
      entry.Set("kind", String("map"));
      entry.Set("target_state", String(states.empty() ? "" : tvm::Downcast<Map<String, Any>>(states[0])["name"].cast<String>()));
      updates.push_back(entry);
    }
    if (auto regions = func->GetAttr<Array<Any>>("blackhole.fragment_regions")) {
      for (const Any& region_any : regions.value()) {
        auto region = tvm::Downcast<Map<String, Any>>(region_any);
        for (const Any& reduction_any : tvm::Downcast<Array<Any>>(region["row_reductions"])) {
          auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
          Map<String, Any> entry;
          entry.Set("name", String(std::string("reduce_") + reduction["target"].cast<String>()));
          entry.Set("kind", String("reduce"));
          entry.Set("target_state", reduction["target"].cast<String>());
          entry.Set("reduce_kind", reduction["kind"].cast<String>());
          updates.push_back(entry);
        }
      }
    }

    Array<Any> seeds;
    if (auto semantic_seeds = func->GetAttr<Map<String, Any>>(attr::kTLSemanticSeeds)) {
      if (auto capture = semantic_seeds.value().find("capture_kinds");
          capture != semantic_seeds.value().end()) {
        seeds = tvm::Downcast<Array<Any>>((*capture).second);
      }
    }

    structure.Set("domain_name", String("device_program"));
    structure.Set("domain_axes", domain_axes);
    structure.Set("domain_traits", domain_traits);
    structure.Set("states", states);
    structure.Set("updates", updates);
    structure.Set("seeds", seeds);

    Map<String, Any> attrs = func->attrs.defined() ? func->attrs->dict : Map<String, Any>();
    attrs.Set(attr::kTLSemanticStructure, structure);
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
