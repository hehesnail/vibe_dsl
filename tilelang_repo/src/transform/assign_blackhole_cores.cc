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
 * \brief Assign T.Kernel grid work items to Blackhole logical worker cores
 */

#include "assign_blackhole_cores.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

#include <tvm/ir/attrs.h>
#include <tvm/ir/module.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <optional>

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
using tvm::ffi::Array;
using tvm::ffi::String;

namespace {

int PositiveOrDefault(int64_t value, int fallback) {
  return value > 0 ? static_cast<int>(value) : fallback;
}

}  // namespace

// Main entry point
PrimFunc PlanTTCoreGroups::Transform(const PrimFunc& func,
                                     std::optional<TTHardwareModel> hardware_model) {
  // Analyze grid dimensions
  assignment_ = AnalyzeGrid(func);
  ApplyHardwareModel(assignment_, hardware_model);

  // Calculate work distribution
  CalculateWorkDistribution(assignment_);

  // Create a mutable copy of the function
  PrimFunc new_func = func;

  // Store assignment in function attributes
  StoreAssignment(new_func, assignment_);

  return new_func;
}

// Analyze T.Kernel grid dimensions from the function
CoreAssignment PlanTTCoreGroups::AnalyzeGrid(const PrimFunc& func) {
  CoreAssignment assignment;

  // Look for thread extent attributes that define grid dimensions
  class GridAnalyzer : public StmtExprVisitor {
   public:
    int grid_x = 1;
    int grid_y = 1;
    int grid_z = 1;
    bool found_grid = false;

    void VisitStmt_(const AttrStmtNode* op) final {
      if (op->attr_key == tir::attr::thread_extent) {
        IterVar iv = Downcast<IterVar>(op->node);
        std::string name = iv->var->name_hint;
        auto extent = as_const_int(iv->dom->extent);
        if (!extent) {
          StmtExprVisitor::VisitStmt_(op);
          return;
        }
        if (name == "blockIdx.x" || name == "bx") {
          grid_x = static_cast<int>(*extent);
          found_grid = true;
        } else if (name == "blockIdx.y" || name == "by") {
          grid_y = static_cast<int>(*extent);
          found_grid = true;
        } else if (name == "blockIdx.z" || name == "bz") {
          grid_z = static_cast<int>(*extent);
          found_grid = true;
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
    assignment.grid_z = analyzer.grid_z;
  }

  // Default grid if not found (for simple kernels)
  if (assignment.grid_x <= 0) assignment.grid_x = 1;
  if (assignment.grid_y <= 0) assignment.grid_y = 1;
  if (assignment.grid_z <= 0) assignment.grid_z = 1;

  return assignment;
}

void PlanTTCoreGroups::ApplyHardwareModel(
    CoreAssignment& assignment,
    const std::optional<TTHardwareModel>& hardware_model) {
  if (hardware_model) {
    const TTHardwareModel& model = hardware_model.value();
    assignment.core_grid_x =
        PositiveOrDefault(model->logical_worker_grid_x, kBlackholeGridX);
    assignment.core_grid_y =
        PositiveOrDefault(model->logical_worker_grid_y, kBlackholeGridY);
    const int grid_capacity =
        std::max(1, assignment.core_grid_x * assignment.core_grid_y);
    assignment.available_worker_cores =
        std::min(grid_capacity,
                 PositiveOrDefault(model->functional_worker_count, grid_capacity));
    return;
  }

  assignment.core_grid_x = kBlackholeGridX;
  assignment.core_grid_y = kBlackholeGridY;
  assignment.available_worker_cores = kTotalCores;
}

// Calculate work distribution across cores
void PlanTTCoreGroups::CalculateWorkDistribution(CoreAssignment& assignment) {
  const int total_work =
      std::max(1, assignment.grid_x * assignment.grid_y * assignment.grid_z);
  const int available_cores =
      std::max(1, std::min(assignment.available_worker_cores,
                           assignment.core_grid_x * assignment.core_grid_y));

  assignment.cores_needed = std::min(total_work, available_cores);
  const int base_work = total_work / assignment.cores_needed;
  const int remainder = total_work % assignment.cores_needed;
  assignment.work_per_core = base_work + (remainder > 0 ? 1 : 0);
}

// Calculate runtime args for a specific core
RuntimeArgs PlanTTCoreGroups::GetRuntimeArgs(int core_idx) const {
  RuntimeArgs args;

  const int total_work =
      std::max(1, assignment_.grid_x * assignment_.grid_y * assignment_.grid_z);
  const int cores_needed = std::max(1, assignment_.cores_needed);
  if (core_idx < 0 || core_idx >= cores_needed) {
    args.work_offset_linear = 0;
    args.work_count = 0;
    return args;
  }

  const int base_work = total_work / cores_needed;
  const int remainder = total_work % cores_needed;
  const int work_offset = core_idx * base_work + std::min(core_idx, remainder);
  args.work_offset_linear = work_offset;
  args.work_count = base_work + (core_idx < remainder ? 1 : 0);
  return args;
}

// Get logical worker core coordinate for a logical core index
CoreCoord PlanTTCoreGroups::GetCoreCoord(int core_idx) const {
  CoreCoord coord;

  const int core_grid_x = std::max(1, assignment_.core_grid_x);
  coord.x = core_idx % core_grid_x;
  coord.y = core_idx / core_grid_x;

  return coord;
}

// Check if a core coordinate is valid
bool PlanTTCoreGroups::IsValidCoreCoord(const CoreCoord& coord) const {
  bool valid_x = coord.x >= 0 && coord.x < assignment_.core_grid_x;
  bool valid_y = coord.y >= 0 && coord.y < assignment_.core_grid_y;

  return valid_x && valid_y;
}

// Store assignment in function attributes
void PlanTTCoreGroups::StoreAssignment(PrimFunc& func,
                                           const CoreAssignment& assignment) {
  (void)func;
  (void)assignment;
}

}  // namespace tl
}  // namespace tvm
