/*
 * Hoist selected root/user block annotations to PrimFunc attributes.
 *
 * `LowerOpaqueBlock` drops non-pragma block annotations. Frontend annotations
 * that are durable planning inputs must be captured before that point.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {
using namespace tvm::tir;

namespace {
constexpr const char *kMemoryConfigMap = "tl.memory_config_map";
}

class NonRestrictCollector : public StmtVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  Array<Var> Result() const {
    Array<Var> out;
    out.reserve(collected_.size());
    for (const Var &v : collected_)
      out.push_back(v);
    return out;
  }

private:
  static std::string NormalizeName(const std::string &s) {
    if (s.size() >= 8 && s.rfind("_handle") == s.size() - 7) {
      return s.substr(0, s.size() - 7);
    }
    return s;
  }

  void MaybeInsert(const Var &v) {
    if (!v.defined())
      return;
    const VarNode *p = v.get();
    if (seen_ptr_.count(p))
      return;
    // Also dedup by normalized name to be robust w.r.t recreated Vars
    std::string norm = NormalizeName(v->name_hint);
    if (seen_name_.count(norm))
      return;
    seen_ptr_.insert(p);
    seen_name_.insert(std::move(norm));
    collected_.push_back(v);
  }

  void VisitStmt_(const BlockNode *op) final {
    auto it = op->annotations.find(attr::kNonRestrictParams);
    if (it != op->annotations.end()) {
      if (const auto *arr = (*it).second.as<ffi::ArrayObj>()) {
        // Downcast directly to Array<Var> for convenience
        Array<Var> vars = tvm::Downcast<Array<Var>>((*it).second);
        for (const Var &v : vars) {
          MaybeInsert(v);
        }
      }
    }
    // Recurse into child statements
    StmtVisitor::VisitStmt_(op);
  }

  std::vector<Var> collected_;
  std::unordered_set<const VarNode *> seen_ptr_;
  std::unordered_set<std::string> seen_name_;
};

class MemoryConfigMapCollector : public StmtVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  Map<Var, ffi::Any> Result() const { return collected_; }

private:
  void VisitStmt_(const BlockNode *op) final {
    auto it = op->annotations.find(kMemoryConfigMap);
    if (it != op->annotations.end()) {
      Map<Var, ffi::Any> memory_config_map =
          tvm::Downcast<Map<Var, ffi::Any>>((*it).second);
      for (const auto &kv : memory_config_map) {
        collected_.Set(kv.first, kv.second);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  Map<Var, ffi::Any> collected_;
};

static PrimFunc HoistNonRestrictParams(PrimFunc f) {
  if (!f.defined())
    return f;

  NonRestrictCollector collector;
  collector.Collect(f->body);
  Array<Var> from_blocks = collector.Result();

  // Merge with any existing PrimFunc-level attribute if present
  if (auto opt_existing = f->GetAttr<Array<Var>>(attr::kNonRestrictParams)) {
    for (const Var &v : opt_existing.value()) {
      // Reuse the collector's dedup logic by temporarily constructing a new
      // collector Alternatively, do a small inline dedup mirroring MaybeInsert
      // Here we inline a simplified pointer-based dedup plus name-based
      // fallback
      bool exists = false;
      for (const Var &cur : from_blocks) {
        if (cur.get() == v.get() || cur->name_hint == v->name_hint) {
          exists = true;
          break;
        }
      }
      if (!exists)
        from_blocks.push_back(v);
    }
  }

  MemoryConfigMapCollector memory_config_collector;
  memory_config_collector.Collect(f->body);
  Map<Var, ffi::Any> memory_config_map = memory_config_collector.Result();
  if (auto existing = f->GetAttr<Map<Var, ffi::Any>>(kMemoryConfigMap)) {
    for (const auto &kv : existing.value()) {
      memory_config_map.Set(kv.first, kv.second);
    }
  }

  PrimFunc updated = std::move(f);
  if (!from_blocks.empty()) {
    updated = WithAttr(std::move(updated), attr::kNonRestrictParams,
                       std::move(from_blocks));
  }
  if (!memory_config_map.empty()) {
    updated = WithAttr(std::move(updated), kMemoryConfigMap,
                       std::move(memory_config_map));
  }
  return updated;
}

namespace transform {

tvm::transform::Pass HoistNonRestrictParams() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    return tvm::tl::HoistNonRestrictParams(std::move(f));
  };
  return tvm::tir::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.HoistNonRestrictParams", {});
}

} // namespace transform

} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistNonRestrictParams",
                        tvm::tl::transform::HoistNonRestrictParams);
}
