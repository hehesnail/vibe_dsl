/*!
 * \file blackhole_utils.h
 * \brief Shared utilities for Blackhole transform passes.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

/*! \brief Convert ffi::String to std::string without static_cast noise. */
inline std::string str(const ffi::String& s) { return static_cast<std::string>(s); }

inline bool IsBlackholePrimFunc(const tir::PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "blackhole";
}

inline const tir::VarNode* BufferDataIdentity(const tir::Buffer& buffer) {
  return buffer.defined() && buffer->data.defined() ? buffer->data.get() : nullptr;
}

inline bool SameBufferIdentity(const tir::Buffer& lhs, const tir::Buffer& rhs) {
  return lhs.same_as(rhs) ||
         (BufferDataIdentity(lhs) != nullptr && BufferDataIdentity(lhs) == BufferDataIdentity(rhs));
}

inline std::string BufferIdentityName(const tir::Buffer& buffer) {
  if (!buffer.defined()) {
    return "";
  }
  if (buffer->data.defined() && !std::string(buffer->data->name_hint).empty()) {
    return buffer->data->name_hint;
  }
  if (!std::string(buffer->name).empty()) {
    return buffer->name;
  }
  return "";
}

inline std::vector<tir::Stmt> CollectExecutionOrderedStmts(const tir::Stmt& root) {
  class OrderedStmtCollector : public tir::StmtVisitor {
   public:
    explicit OrderedStmtCollector(std::vector<tir::Stmt>* ordered_stmts)
        : ordered_stmts_(ordered_stmts) {}

    void Collect(const tir::Stmt& root) {
      if (!root.defined()) {
        return;
      }
      if (!root->IsInstance<tir::SeqStmtNode>()) {
        ordered_stmts_->push_back(root);
      }
      VisitStmt(root);
    }

    void VisitStmt_(const tir::SeqStmtNode* op) final {
      for (const tir::Stmt& child : op->seq) {
        ordered_stmts_->push_back(child);
        VisitStmt(child);
      }
    }

   private:
    std::vector<tir::Stmt>* ordered_stmts_;
  };

  std::vector<tir::Stmt> ordered_stmts;
  OrderedStmtCollector collector(&ordered_stmts);
  collector.Collect(root);
  return ordered_stmts;
}

inline std::unordered_map<const Object*, int> BuildExecutionOrderIndexByStmtNode(
    const tir::Stmt& root) {
  std::unordered_map<const Object*, int> order_by_stmt_node;
  const std::vector<tir::Stmt> ordered_stmts = CollectExecutionOrderedStmts(root);
  for (int order_index = 0; order_index < static_cast<int>(ordered_stmts.size()); ++order_index) {
    order_by_stmt_node.emplace(ordered_stmts[order_index].get(), order_index);
  }
  return order_by_stmt_node;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BLACKHOLE_UTILS_H_
