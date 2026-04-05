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
      state.Set("name", String(name));
      state.Set("role", String("transient"));
      state.Set("scope", String(entry.scope));
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
    const std::string name = CanonicalBufferName(buffer->name);
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

    LocalBufferCollector buffer_collector;
    buffer_collector(func->body);

    Array<Any> states;
    std::unordered_map<std::string, int> state_index;
    std::unordered_set<std::string> reduction_targets;
    std::unordered_set<std::string> integer_states;
    std::unordered_set<std::string> loop_carried_states;
    std::unordered_set<std::string> selection_targets;
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
          const std::string name = buffer["name"].cast<String>();
          register_state(name, "transient", buffer["scope"].cast<String>());
          bool is_integer = buffer_collector.HasIntegerDType(name);
          if (auto it = buffer.find("is_integer"); it != buffer.end()) {
            is_integer = static_cast<bool>(tvm::Downcast<Integer>((*it).second)->value);
          }
          if (is_integer) {
            integer_states.insert(name);
          }
        }
        for (const Any& carried_any : tvm::Downcast<Array<Any>>(region["loop_carried_state"])) {
          auto carried = tvm::Downcast<Map<String, Any>>(carried_any);
          const std::string name = carried["name"].cast<String>();
          loop_carried_states.insert(name);
          register_state(name, "carry", "");
        }
        if (region.count("selection_targets")) {
          for (const Any& target_any : tvm::Downcast<Array<Any>>(region["selection_targets"])) {
            selection_targets.insert(tvm::Downcast<String>(target_any));
          }
        }
        for (const Any& reduction_any : tvm::Downcast<Array<Any>>(region["row_reductions"])) {
          auto reduction = tvm::Downcast<Map<String, Any>>(reduction_any);
          const std::string target = reduction["target"].cast<String>();
          reduction_targets.insert(target);
          if (buffer_collector.HasIntegerDType(target)) {
            integer_states.insert(target);
          }
          const std::string role =
              integer_states.count(target) ? "index_state" : "reduction_accumulator";
          register_state(target, role, "");
        }
        for (const std::string& carried : loop_carried_states) {
          if (!reduction_targets.count(carried)) {
            register_state(carried, "carry", "");
          }
        }
        for (const std::string& name : selection_targets) {
          if (!integer_states.count(name)) {
            register_state(name, "selection_state", "");
          }
        }
      }
    } else {
      states = buffer_collector.Encode();
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
        if (!selection_targets.empty()) {
          for (const std::string& state_name : selection_targets) {
            Map<String, Any> entry;
            entry.Set("name", String(std::string("select_") + state_name));
            entry.Set("kind", String("select"));
            entry.Set("target_state", String(state_name));
            entry.Set("traits", Array<Any>{String("selected"), String("indexed")});
            updates.push_back(entry);
          }
        }
        if (!loop_carried_states.empty()) {
          for (const std::string& state_name : loop_carried_states) {
            Map<String, Any> entry;
            entry.Set("name", String(std::string("recur_") + state_name));
            entry.Set("kind", String("recurrence"));
            entry.Set("target_state", String(state_name));
            entry.Set("traits", Array<Any>{String("carried"), String("staged")});
            updates.push_back(entry);
          }
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
    structure.Set("supplements", Array<Any>{
                                     Map<String, Any>{{"kind", String("analysis_sources")},
                                                      {"payload",
                                                       Map<String, Any>{{"has_fragment_regions",
                                                                         Bool(func->GetAttr<Array<Any>>("blackhole.fragment_regions").has_value())},
                                                                        {"has_pipeline_stages",
                                                                         Bool(func->GetAttr<Array<Any>>("blackhole.pipeline_stages").has_value())}}}}});

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
