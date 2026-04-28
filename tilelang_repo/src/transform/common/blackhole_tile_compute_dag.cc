/*!
 * \file blackhole_tile_compute_dag.cc
 * \brief Pass-local Blackhole tile compute DAG diagnostics.
 */

#include "blackhole_tile_compute_dag.h"

#include "../../op/utils.h"
#include "blackhole_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using tir::CallNode;
using tir::StringImmNode;

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

std::string ExprValueKey(const PrimExpr& expr) {
  if (IsBufferLikeExpr(expr)) {
    const tir::BufferRegion region = NormalizeToBufferRegion(expr);
    return "buffer:" + BufferIdentityName(region->buffer);
  }
  return "expr:" + ExprDisplay(expr);
}

const Object* ExprValueIdentity(const PrimExpr& expr) {
  if (!IsBufferLikeExpr(expr)) {
    return nullptr;
  }
  const tir::BufferRegion region = NormalizeToBufferRegion(expr);
  return BufferDataIdentity(region->buffer);
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

bool IsOutputRole(const std::string& role) {
  return role == "output" || role == "c";
}

void AddOperandEdge(BlackholeTileComputeDAG* dag, int64_t node_id,
                    const std::string& role, const PrimExpr& value) {
  ICHECK(dag != nullptr);
  dag->edges.push_back(BlackholeTileComputeDAGEdge{
      static_cast<int64_t>(dag->edges.size()),
      -1,
      node_id,
      role,
      ExprDisplay(value),
      ExprValueKey(value),
      ExprValueIdentity(value),
      false,
  });
}

void AddNode(BlackholeTileComputeDAG* dag, const std::string& op_kind,
             const std::string& op_name, const std::vector<std::pair<std::string, PrimExpr>>& edges) {
  ICHECK(dag != nullptr);
  const int64_t node_id = static_cast<int64_t>(dag->nodes.size());
  dag->nodes.push_back(BlackholeTileComputeDAGNode{
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

bool TryAddBlackholeComputeNode(BlackholeTileComputeDAG* dag, const CallNode* call) {
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

bool TryAddReduceNode(BlackholeTileComputeDAG* dag, const CallNode* call) {
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

bool TryAddGemmNode(BlackholeTileComputeDAG* dag, const CallNode* call) {
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

void ConnectProducerEdges(BlackholeTileComputeDAG* dag) {
  ICHECK(dag != nullptr);
  std::vector<std::vector<int64_t>> edge_indices_by_node(dag->nodes.size());
  for (int64_t edge_index = 0; edge_index < static_cast<int64_t>(dag->edges.size());
       ++edge_index) {
    const int64_t consumer = dag->edges[edge_index].consumer_node;
    if (consumer >= 0 && consumer < static_cast<int64_t>(edge_indices_by_node.size())) {
      edge_indices_by_node[consumer].push_back(edge_index);
    }
  }

  std::unordered_map<const Object*, int64_t> latest_producer_by_identity;
  std::unordered_map<std::string, int64_t> latest_producer_by_value;
  for (const BlackholeTileComputeDAGNode& node : dag->nodes) {
    for (int64_t edge_index : edge_indices_by_node[node.id]) {
      BlackholeTileComputeDAGEdge& edge = dag->edges[edge_index];
      if (IsOutputRole(edge.value_role)) {
        continue;
      }
      if (edge.value_identity != nullptr) {
        auto producer_it = latest_producer_by_identity.find(edge.value_identity);
        if (producer_it != latest_producer_by_identity.end()) {
          edge.producer_node = producer_it->second;
        }
      } else {
        auto producer_it = latest_producer_by_value.find(edge.value_key);
        if (producer_it != latest_producer_by_value.end()) {
          edge.producer_node = producer_it->second;
        }
      }
    }
    for (int64_t edge_index : edge_indices_by_node[node.id]) {
      const BlackholeTileComputeDAGEdge& edge = dag->edges[edge_index];
      if (IsOutputRole(edge.value_role)) {
        if (edge.value_identity != nullptr) {
          latest_producer_by_identity[edge.value_identity] = node.id;
        } else {
          latest_producer_by_value[edge.value_key] = node.id;
        }
      }
    }
  }
}

ffi::Map<ffi::String, ffi::Any> EncodeNode(const BlackholeTileComputeDAGNode& node) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("id"), Integer(node.id));
  encoded.Set(ffi::String("op_kind"), ffi::String(node.op_kind));
  encoded.Set(ffi::String("op_name"), ffi::String(node.op_name));
  encoded.Set(ffi::String("side_effect_class"), ffi::String(node.side_effect_class));
  encoded.Set(ffi::String("token_input"), ffi::String(node.token_input));
  encoded.Set(ffi::String("token_output"), ffi::String(node.token_output));
  return encoded;
}

ffi::Map<ffi::String, ffi::Any> EncodeEdge(const BlackholeTileComputeDAGEdge& edge) {
  ffi::Map<ffi::String, ffi::Any> encoded;
  encoded.Set(ffi::String("id"), Integer(edge.id));
  encoded.Set(ffi::String("producer_node"), Integer(edge.producer_node));
  encoded.Set(ffi::String("consumer_node"), Integer(edge.consumer_node));
  encoded.Set(ffi::String("value_role"), ffi::String(edge.value_role));
  encoded.Set(ffi::String("value_repr"), ffi::String(edge.value_repr));
  encoded.Set(ffi::String("value_key"), ffi::String(edge.value_key));
  encoded.Set(ffi::String("requires_materialization"), Bool(edge.requires_materialization));
  return encoded;
}

}  // namespace

BlackholeTileComputeDAG BuildBlackholeTileComputeDAG(const tir::PrimFunc& func) {
  BlackholeTileComputeDAG dag;
  for (const tir::Stmt& stmt : CollectExecutionOrderedStmts(func->body)) {
    const CallNode* call = UnwrapEvaluateCall(stmt);
    if (TryAddBlackholeComputeNode(&dag, call) ||
        TryAddReduceNode(&dag, call) ||
        TryAddGemmNode(&dag, call)) {
      continue;
    }
  }
  ConnectProducerEdges(&dag);
  return dag;
}

ffi::Map<ffi::String, ffi::Any> BuildBlackholeTileComputeDAGDiagnostic(
    const tir::PrimFunc& func) {
  const BlackholeTileComputeDAG dag = BuildBlackholeTileComputeDAG(func);
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> nodes;
  for (const BlackholeTileComputeDAGNode& node : dag.nodes) {
    nodes.push_back(EncodeNode(node));
  }
  ffi::Array<ffi::Map<ffi::String, ffi::Any>> edges;
  for (const BlackholeTileComputeDAGEdge& edge : dag.edges) {
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
