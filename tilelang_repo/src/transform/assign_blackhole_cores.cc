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
 * \file assign_blackhole_cores.cc
 * \brief Assign T.Kernel grid work items to Blackhole 14x10 Tensix cores
 */

#include "assign_blackhole_cores.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>

namespace tvm {
namespace tl {

using tir::PrimFunc;
using tir::PrimFuncNode;
using tir::StmtExprVisitor;
using tvm::Integer;
using tvm::ObjectRef;
using tvm::DictAttrs;
using tir::AttrStmtNode;
using tir::IterVar;
using tir::as_const_int;
using tvm::DictAttrs;
using tvm::DictAttrsNode;
using tvm::Integer;

// Main entry point
PrimFunc AssignBlackholeCores::Transform(const PrimFunc& func) {
  // Analyze grid dimensions
  assignment_ = AnalyzeGrid(func);

  // Calculate work distribution
  CalculateWorkDistribution(assignment_);

  // Create a mutable copy of the function
  PrimFunc new_func = func;

  // Store assignment in function attributes
  StoreAssignment(new_func, assignment_);

  return new_func;
}

// Analyze T.Kernel grid dimensions from the function
CoreAssignment AssignBlackholeCores::AnalyzeGrid(const PrimFunc& func) {
  CoreAssignment assignment;

  // Look for thread extent attributes that define grid dimensions
  class GridAnalyzer : public StmtExprVisitor {
   public:
    int grid_x = 1;
    int grid_y = 1;
    bool found_grid = false;

    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        IterVar iv = Downcast<IterVar>(op->node);
        // Check for blockIdx.x and blockIdx.y patterns
        std::string name = iv->var->name_hint;
        if (name.find("blockIdx") != std::string::npos ||
            name.find("bx") != std::string::npos ||
            name.find("by") != std::string::npos) {
          auto extent = as_const_int(iv->dom->extent);
          if (extent) {
            if (name.find('x') != std::string::npos) {
              grid_x = static_cast<int>(*extent);
              found_grid = true;
            } else if (name.find('y') != std::string::npos) {
              grid_y = static_cast<int>(*extent);
              found_grid = true;
            }
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  };

  GridAnalyzer analyzer;
  analyzer(func->body);

  if (analyzer.found_grid) {
    assignment.grid_x = analyzer.grid_x;
    assignment.grid_y = analyzer.grid_y;
  }

  // Default grid if not found (for simple kernels)
  if (assignment.grid_x <= 0) assignment.grid_x = 1;
  if (assignment.grid_y <= 0) assignment.grid_y = 1;

  assignment.core_grid_x = kBlackholeGridX;
  assignment.core_grid_y = kBlackholeGridY;

  return assignment;
}

// Calculate work distribution across cores
void AssignBlackholeCores::CalculateWorkDistribution(CoreAssignment& assignment) {
  const int total_work = std::max(1, assignment.grid_x * assignment.grid_y);

  // Stage 2 keeps Blackhole execution on a single physical core and models the
  // logical grid explicitly in core_plan/work_packets. Multi-core distribution
  // is deferred to Stage 3.
  assignment.work_per_core = total_work;
  assignment.cores_needed = 1;
}

// Calculate runtime args for a specific core
RuntimeArgs AssignBlackholeCores::GetRuntimeArgs(int core_idx) const {
  RuntimeArgs args;

  if (core_idx != 0) {
    args.work_offset_linear = 0;
    args.work_count = 0;
    return args;
  }

  args.work_offset_linear = 0;
  args.work_count = std::max(1, assignment_.grid_x * assignment_.grid_y);
  return args;
}

// Get physical core coordinate for a logical core index
CoreCoord AssignBlackholeCores::GetCoreCoord(int core_idx) const {
  CoreCoord coord;

  // Map to logical 14x10 grid
  int x_in_grid = core_idx % kBlackholeGridX;
  int y_in_grid = core_idx / kBlackholeGridX;

  // Map to physical coordinates (avoiding x=8,9 which are DRAM/ARC/Eth)
  // Physical x: 1-7, 10-16
  if (x_in_grid < 7) {
    coord.x = x_in_grid + 1;  // 1-7
  } else {
    coord.x = x_in_grid + 3;  // 10-16 (skip 8,9)
  }

  // Physical y: 2-11
  coord.y = y_in_grid + 2;

  return coord;
}

// Check if a core coordinate is valid
bool AssignBlackholeCores::IsValidCoreCoord(const CoreCoord& coord) const {
  // Valid x: 1-7, 10-16 (avoid 8,9 which are DRAM/ARC/Eth)
  bool valid_x = (coord.x >= 1 && coord.x <= 7) ||
                 (coord.x >= 10 && coord.x <= 16);

  // Valid y: 2-11
  bool valid_y = (coord.y >= 2 && coord.y <= 11);

  return valid_x && valid_y;
}

// Store assignment in function attributes
void AssignBlackholeCores::StoreAssignment(PrimFunc& func,
                                           const CoreAssignment& assignment) {
  // Get existing attributes (if any)
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  // Store core assignment values (merge with existing)
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> core_plan;
  core_plan.Set("logical_grid_x", Integer(assignment.grid_x));
  core_plan.Set("logical_grid_y", Integer(assignment.grid_y));
  core_plan.Set("linearization", ffi::String("row_major"));

  tvm::ffi::Array<tvm::ffi::Any> physical_cores;
  tvm::ffi::Array<tvm::ffi::Any> work_packets;

  for (int core_idx = 0; core_idx < assignment.cores_needed; ++core_idx) {
    const CoreCoord coord = GetCoreCoord(core_idx);
    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> core_info;
    core_info.Set("core_x", Integer(coord.x));
    core_info.Set("core_y", Integer(coord.y));
    physical_cores.push_back(core_info);

    const RuntimeArgs runtime_args = GetRuntimeArgs(core_idx);
    tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> packet_info;
    packet_info.Set("core_x", Integer(coord.x));
    packet_info.Set("core_y", Integer(coord.y));
    packet_info.Set("work_offset", Integer(runtime_args.work_offset_linear));
    packet_info.Set("work_count", Integer(runtime_args.work_count));
    work_packets.push_back(packet_info);
  }

  core_plan.Set("physical_cores", physical_cores);
  core_plan.Set("work_packets", work_packets);

  attrs.Set("blackhole.grid_shape", Integer(assignment.grid_x * assignment.grid_y));
  attrs.Set("blackhole.grid_x", Integer(assignment.grid_x));
  attrs.Set("blackhole.grid_y", Integer(assignment.grid_y));
  attrs.Set("blackhole.cores_needed", Integer(assignment.cores_needed));
  attrs.Set("blackhole.work_per_core", Integer(assignment.work_per_core));
  attrs.Set("blackhole.core_plan", core_plan);

  // Update function attributes
  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

// Modern TVM pass registration
tir::transform::Pass AssignBlackholeCoresPass() {
  auto fpass = [](PrimFunc func, IRModule m, tir::transform::PassContext ctx) -> PrimFunc {
    return AssignBlackholeCores().Transform(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.AssignBlackholeCores", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.AssignBlackholeCores", AssignBlackholeCoresPass);
}

}  // namespace tl
}  // namespace tvm
