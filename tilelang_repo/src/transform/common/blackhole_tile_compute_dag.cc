/*!
 * \file blackhole_tile_compute_dag.cc
 * \brief Pass-local Blackhole tile compute DAG diagnostics.
 */

#include "blackhole_tile_compute_dag.h"

#include "blackhole_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>

#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using tir::CallNode;
using tir::StringImmNode;

struct TileComputeDAGEdge {
  int64_t id{-1};
  int64_t producer_node{-1};
  int64_t consumer_node{-1};
  std::string value_role;
  std::string value_repr;
};

struct TileComputeDAGNode {
  int64_t id{-1};
  std::string op_kind;
  std::string op_name;
  std::string side_effect_class;
  std::string token_input;
  std::string token_output;
};

struct TileComputeDAGDiagnostic {
  std::vector<TileComputeDAGNode> nodes;
  std::vector<TileComputeDAGEdge> edges;
};

const CallNode* UnwrapEvaluateCall(const tir::Stmt& stmt) {
  const auto* eval = stmt.as<tir::EvaluateNode>();
  if (eval == nullptr) {
    return nullptr;
  }
  return eval->value.as<CallNode>();
}

std::string ExprDisplay(const PrimExpr& expr) {
  std::ostringstream os;
  os << expr;
  return os.str();
}

std::string SideEffectClassForOperation(const std::string& op_name) {
  if (op_name == "pack_tile") {
    return "pack";
  }
  if (op_name == "fill_tile" || op_name == "matmul_tiles") {
    return "dst";
  }
  if (op_name == "copy_tile" || op_name == "typecast_tile") {
    return "fragment";
  }
  return "tile_regs";
}

void AddOperandEdge(TileComputeDAGDiagnostic* dag, int64_t node_id,
                    const std::string& role, const PrimExpr& value) {
  ICHECK(dag != nullptr);
  dag->edges.push_back(TileComputeDAGEdge{
      static_cast<int64_t>(dag->edges.size()),
      -1,
      node_id,
      role,
      ExprDisplay(value),
  });
}

void AddNode(TileComputeDAGDiagnostic* dag, const std::string& op_kind,
             const std::string& op_name, const std::vector<std::pair<std::string, PrimExpr>>& edges) {
  ICHECK(dag != nullptr);
  const int64_t node_id = static_cast<int64_t>(dag->nodes.size());
  dag->nodes.push_back(TileComputeDAGNode{
      node_id,
      op_kind,
      op_name,
      SideEffectClassForOperation(op_name),
      "token_" + std::to_string(node_id),
      "token_" + std::to_string(node_id + 1),
  });
  for (const auto& [role, value] : edges) {
    AddOperandEdge(dag, node_id, role, value);
  }
}

bool TryAddBlackholeComputeNode(TileComputeDAGDiagnostic* dag, const CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>() || call->args.empty()) {
    return false;
  }
  const Op call_op = Downcast<Op>(call->op);
  if (call_op->name != blackhole_tile_compute_schema::kOpName) {
    return false;
  }
  const auto* op_name = call->args[0].as<StringImmNode>();
  if (op_name == nullptr) {
    return false;
  }
  const std::string operation = op_name->value;
  std::vector<std::pair<std::string, PrimExpr>> edges;
  auto add_if_present = [&](const std::string& role, size_t index) {
    if (index < call->args.size()) {
      edges.push_back({role, call->args[index]});
    }
  };
  if (operation == blackhole_tile_compute_schema::kFillTile) {
    add_if_present("output", 1);
  } else if (operation == blackhole_tile_compute_schema::kCopyTile ||
             operation == blackhole_tile_compute_schema::kTypecastTile) {
    add_if_present("input", 1);
    add_if_present("output", 2);
  } else if (operation == blackhole_tile_compute_schema::kBinaryMaxTile ||
             operation == blackhole_tile_compute_schema::kAddTiles ||
             operation == blackhole_tile_compute_schema::kMulTiles) {
    add_if_present("lhs", 1);
    add_if_present("rhs", 2);
    add_if_present("output", 1);
  } else if (operation == blackhole_tile_compute_schema::kMulTilesBcastCols) {
    add_if_present("lhs", 2);
    add_if_present("rhs", 3);
    add_if_present("output", 2);
  } else if (operation == blackhole_tile_compute_schema::kExp2Tile) {
    add_if_present("output", 2);
    add_if_present("lhs", 3);
    add_if_present("rhs", 4);
  } else {
    for (size_t i = 1; i < call->args.size(); ++i) {
      add_if_present("operand" + std::to_string(i), i);
    }
  }
  AddNode(dag, "blackhole_leaf_op", operation, edges);
  return true;
}

bool TryAddReduceNode(TileComputeDAGDiagnostic* dag, const CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(call->op);
  if (call_op->name != "tl.tileop.reduce" || call->args.size() < 2U) {
    return false;
  }
  AddNode(dag, "generic_tile_op", "reduce_tile",
          {{"input", call->args[0]}, {"output", call->args[1]}});
  return true;
}

bool TryAddGemmNode(TileComputeDAGDiagnostic* dag, const CallNode* call) {
  if (call == nullptr || !call->op->IsInstance<OpNode>()) {
    return false;
  }
  const Op call_op = Downcast<Op>(call->op);
  if (call_op->name != "tl.tileop.gemm_py" || call->args.size() < 3U) {
    return false;
  }
  AddNode(dag, "generic_tile_op", "matmul_tiles",
          {{"a", call->args[0]}, {"b", call->args[1]}, {"c", call->args[2]}});
  return true;
}

ffi::Map<ffi::String, ffi::Any> EncodeNode(const TileComputeDAGNode& node) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("id"), Integer(node.id));
  encoded.Set(ffi::String("op_kind"), ffi::String(node.op_kind));
  encoded.Set(ffi::String("op_name"), ffi::String(node.op_name));
  encoded.Set(ffi::String("side_effect_class"), ffi::String(node.side_effect_class));
  encoded.Set(ffi::String("token_input"), ffi::String(node.token_input));
  encoded.Set(ffi::String("token_output"), ffi::String(node.token_output));
  return encoded;
}

ffi::Map<ffi::String, ffi::Any> EncodeEdge(const TileComputeDAGEdge& edge) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("id"), Integer(edge.id));
  encoded.Set(ffi::String("producer_node"), Integer(edge.producer_node));
  encoded.Set(ffi::String("consumer_node"), Integer(edge.consumer_node));
  encoded.Set(ffi::String("value_role"), ffi::String(edge.value_role));
  encoded.Set(ffi::String("value_repr"), ffi::String(edge.value_repr));
  return encoded;
}

}  // namespace

ffi::Map<ffi::String, ffi::Any> BuildBlackholeTileComputeDAGDiagnostic(
    const tir::PrimFunc& func) {
  TileComputeDAGDiagnostic dag;
  for (const tir::Stmt& stmt : CollectExecutionOrderedStmts(func->body)) {
    const CallNode* call = UnwrapEvaluateCall(stmt);
    if (TryAddBlackholeComputeNode(&dag, call) ||
        TryAddReduceNode(&dag, call) ||
        TryAddGemmNode(&dag, call)) {
      continue;
    }
  }
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> nodes;
  for (const TileComputeDAGNode& node : dag.nodes) {
    nodes.push_back(EncodeNode(node));
  }
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> edges;
  for (const TileComputeDAGEdge& edge : dag.edges) {
    edges.push_back(EncodeEdge(edge));
  }
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("nodes"), nodes);
  encoded.Set(ffi::String("edges"), edges);
  encoded.Set(ffi::String("source"), ffi::String("normalized_tile_tir"));
  return encoded;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.BuildBlackholeTileComputeDAGDiagnostic",
                        BuildBlackholeTileComputeDAGDiagnostic);
}

}  // namespace tl
}  // namespace tvm
