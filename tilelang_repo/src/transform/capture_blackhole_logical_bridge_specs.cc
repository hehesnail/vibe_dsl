/*!
 * \file capture_blackhole_logical_bridge_specs.cc
 * \brief Capture narrow logical bridge specs directly from current Blackhole TIR.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>
#include <unordered_set>

#include "common/buffer_tile_bridge_spec_utils.h"
#include "common/companion_base.h"

namespace tvm {
namespace tl {

using tvm::DictAttrs;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;

namespace {

class LogicalBridgeSpecCollector final : public tir::StmtExprVisitor {
 public:
  Array<Any> Collect(const tir::PrimFunc& func) {
    specs_.clear();
    seen_keys_.clear();
    VisitStmt(func->body);
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
    if (!key.empty() && seen_keys_.insert(key).second) {
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
  std::unordered_set<std::string> seen_keys_;
};

tvm::transform::Pass CaptureBlackholeLogicalBridgeSpecs() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext) {
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));
    for (const auto& [gvar, base_func] : mod->functions) {
      auto func = base_func.as<tir::PrimFunc>();
      if (!func || !IsBlackholePrimFunc(func.value())) {
        continue;
      }

      LogicalBridgeSpecCollector collector;
      Array<Any> specs = collector.Collect(func.value());
      if (specs.empty()) {
        continue;
      }

      tir::PrimFunc updated = func.value();
      Map<String, Any> attrs =
          updated->attrs.defined() ? updated->attrs->dict : Map<String, Any>();
      attrs.Set(attr::kTLBlackholeLogicalBufferTileBridgeSpecs, specs);
      updated.CopyOnWrite()->attrs = DictAttrs(attrs);
      updates->Add(gvar, updated);
    }
    mod->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(
      pass_func, 0, "tl.transform.CaptureBlackholeLogicalBridgeSpecs", {});
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CaptureBlackholeLogicalBridgeSpecs",
                        CaptureBlackholeLogicalBridgeSpecs);
}

}  // namespace tl
}  // namespace tvm
