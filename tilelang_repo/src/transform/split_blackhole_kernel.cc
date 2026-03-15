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
 * \file split_blackhole_kernel.cc
 * \brief Split unified PrimFunc into Reader/Compute/Writer kernels
 */

#include "split_blackhole_kernel.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

// Main entry point
SplitResult SplitBlackholeKernel::Transform(const PrimFunc& func) {
  SplitResult result;

  // Generate Reader kernel (data movement: DRAM -> CB)
  result.reader_func = GenerateReaderKernel(func);

  // Generate Compute kernel (computation: CB -> CB)
  result.compute_func = GenerateComputeKernel(func);

  // Generate Writer kernel (data movement: CB -> DRAM)
  result.writer_func = GenerateWriterKernel(func);

  return result;
}

// Analyze data flow to identify read/compute/write regions
SplitBlackholeKernel::DataFlowAnalysis SplitBlackholeKernel::AnalyzeDataFlow(
    const PrimFunc& func) {
  DataFlowAnalysis analysis;
  analysis.has_compute = false;

  class DataFlowVisitor : public StmtExprVisitor {
   public:
    DataFlowAnalysis* analysis;

    void VisitStmt_(const BufferLoadNode* op) final {
      // Track buffer reads
      if (!IsAlreadyTracked(op->buffer)) {
        analysis->input_buffers.push_back(op->buffer);
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode* op) final {
      // Track buffer writes
      if (!IsAlreadyTracked(op->buffer)) {
        analysis->output_buffers.push_back(op->buffer);
      }

      // Check for compute patterns (e.g., matrix multiply, add)
      if (ContainsComputePattern(op->value)) {
        analysis->has_compute = true;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

   private:
    bool IsAlreadyTracked(const Buffer& buf) {
      for (const auto& b : analysis->input_buffers) {
        if (b.same_as(buf)) return true;
      }
      for (const auto& b : analysis->output_buffers) {
        if (b.same_as(buf)) return true;
      }
      return false;
    }

    bool ContainsComputePattern(const PrimExpr& expr) {
      // Check for multiplication and addition patterns
      if (expr.as<AddNode>() || expr.as<MulNode>()) {
        return true;
      }
      if (auto* call = expr.as<CallNode>()) {
        // Check for compute-related builtins
        std::string name = call->op.as<OpNode>()->name;
        if (name.find("mma") != std::string::npos ||
            name.find("dot") != std::string::npos ||
            name.find("reduce") != std::string::npos) {
          return true;
        }
      }
      return false;
    }
  };

  DataFlowVisitor visitor;
  visitor.analysis = &analysis;
  visitor(func->body);

  return analysis;
}

// Generate Reader kernel from original function
PrimFunc SplitBlackholeKernel::GenerateReaderKernel(const PrimFunc& func) {
  // Analyze data flow
  DataFlowAnalysis analysis = AnalyzeDataFlow(func);

  // Create a new function for reader kernel
  std::string reader_name = std::string(func->attrs.GetAttr<String>("global_symbol").value_or(String("kernel")))
                            + "_reader";

  // For now, create a simplified version that handles input data movement
  // In full implementation, this would extract only DRAM->CB statements

  // Clone the original function with modifications
  PrimFunc reader_func = func;

  // Update attributes to mark as reader kernel
  Map<String, ObjectRef> attrs = reader_func->attrs;
  attrs.Set("tl_kernel_type", String("reader"));
  attrs.Set("global_symbol", String(reader_name));
  reader_func.CopyOnWrite()->attrs = attrs;

  return reader_func;
}

// Generate Compute kernel from original function
PrimFunc SplitBlackholeKernel::GenerateComputeKernel(const PrimFunc& func) {
  // Analyze data flow
  DataFlowAnalysis analysis = AnalyzeDataFlow(func);

  // If no compute pattern found, return undefined
  if (!analysis.has_compute) {
    return PrimFunc();
  }

  // Create a new function for compute kernel
  std::string compute_name = std::string(func->attrs.GetAttr<String>("global_symbol").value_or(String("kernel")))
                             + "_compute";

  // Clone the original function with modifications
  PrimFunc compute_func = func;

  // Update attributes to mark as compute kernel
  Map<String, ObjectRef> attrs = compute_func->attrs;
  attrs.Set("tl_kernel_type", String("compute"));
  attrs.Set("global_symbol", String(compute_name));
  compute_func.CopyOnWrite()->attrs = attrs;

  return compute_func;
}

// Generate Writer kernel from original function
PrimFunc SplitBlackholeKernel::GenerateWriterKernel(const PrimFunc& func) {
  // Analyze data flow
  DataFlowAnalysis analysis = AnalyzeDataFlow(func);

  // Create a new function for writer kernel
  std::string writer_name = std::string(func->attrs.GetAttr<String>("global_symbol").value_or(String("kernel")))
                            + "_writer";

  // Clone the original function with modifications
  PrimFunc writer_func = func;

  // Update attributes to mark as writer kernel
  Map<String, ObjectRef> attrs = writer_func->attrs;
  attrs.Set("tl_kernel_type", String("writer"));
  attrs.Set("global_symbol", String(writer_name));
  writer_func.CopyOnWrite()->attrs = attrs;

  return writer_func;
}

// Check if a function contains CB synchronization of given type
bool SplitBlackholeKernel::ContainsCBSync(const PrimFunc& func,
                                          const std::string& sync_type) {
  class CBSyncFinder : public StmtExprVisitor {
   public:
    std::string sync_type;
    bool found = false;

    void VisitExpr_(const CallNode* op) final {
      if (auto* op_node = op->op.as<OpNode>()) {
        std::string name = op_node->name;
        if (name.find(sync_type) != std::string::npos) {
          found = true;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };

  CBSyncFinder finder;
  finder.sync_type = sync_type;
  finder(func->body);

  return finder.found;
}

// Pass registration
class SplitBlackholeKernelPassNode : public transform::PassNode {
 public:
  // Entry point
  IRModule operator()(IRModule mod, const transform::PassContext& pass_ctx) const final {
    IRModule new_mod = mod;

    for (const auto& [gvar, func] : mod->functions) {
      if (auto* prim_func = func.as<PrimFuncNode>()) {
        SplitResult result = SplitBlackholeKernel().Transform(GetRef<PrimFunc>(prim_func));

        // Add split kernels to module
        if (result.HasReader()) {
          GlobalVar reader_gvar(result.reader_func->attrs.GetAttr<String>("global_symbol").value());
          new_mod.CopyOnWrite()->Add(reader_gvar, result.reader_func);
        }
        if (result.HasCompute()) {
          GlobalVar compute_gvar(result.compute_func->attrs.GetAttr<String>("global_symbol").value());
          new_mod.CopyOnWrite()->Add(compute_gvar, result.compute_func);
        }
        if (result.HasWriter()) {
          GlobalVar writer_gvar(result.writer_func->attrs.GetAttr<String>("global_symbol").value());
          new_mod.CopyOnWrite()->Add(writer_gvar, result.writer_func);
        }

        // Mark original function as split
        PrimFunc updated_func = GetRef<PrimFunc>(prim_func);
        Map<String, ObjectRef> attrs = updated_func->attrs;
        attrs.Set("tl_blackhole_split", Bool(true));
        updated_func.CopyOnWrite()->attrs = attrs;
        new_mod.CopyOnWrite()->Add(gvar, updated_func);
      }
    }
    return new_mod;
  }

  TVM_OBJECT_ENABLE(SplitBlackholeKernelPassNode, transform::PassNode);
};

tvm::tir::transform::Pass SplitBlackholeKernelPass() {
  return tvm::make_object<SplitBlackholeKernelPassNode>();
}

TVM_REGISTER_GLOBAL("tl.transform.SplitBlackholeKernel")
    .set_body_typed(SplitBlackholeKernelPass);

}  // namespace tl
}  // namespace tvm
