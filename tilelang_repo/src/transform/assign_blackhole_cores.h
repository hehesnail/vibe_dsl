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
 * \file assign_blackhole_cores.h
 * \brief Assign T.Kernel grid work items to Blackhole 14x10 Tensix cores
 */

#ifndef TVM_TL_ASSIGN_BLACKHOLE_CORES_H_
#define TVM_TL_ASSIGN_BLACKHOLE_CORES_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief Core coordinate on Blackhole physical grid
 * Physical x: 1-7, 10-16 (avoid x=8,9 which are DRAM/ARC/Eth)
 * Physical y: 2-11
 */
struct CoreCoord {
  int x, y;
};

/*!
 * \brief Core assignment configuration for a kernel
 */
struct CoreAssignment {
  int grid_x, grid_y;           // T.Kernel grid dimensions
  int core_grid_x, core_grid_y; // Blackhole core grid (14, 10)
  int work_per_core;            // Work items per core
  int cores_needed;             // Total cores needed

  CoreAssignment()
      : grid_x(1), grid_y(1), core_grid_x(14), core_grid_y(10),
        work_per_core(1), cores_needed(1) {}
};

/*!
 * \brief Runtime arguments for each core
 */
struct RuntimeArgs {
  int work_offset_x, work_offset_y;  // Start work item for this core
  int work_count_x, work_count_y;    // Number of work items for this core

  RuntimeArgs() : work_offset_x(0), work_offset_y(0),
                  work_count_x(1), work_count_y(1) {}
};

/*!
 * \brief AssignBlackholeCores Pass
 *
 * This pass analyzes T.Kernel grid dimensions and assigns work items
 * to Blackhole's 14x10 Tensix core grid.
 */
class AssignBlackholeCores : public tvm::tir::StmtExprMutator {
 public:
  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  /*! \brief Get core assignment result */
  CoreAssignment GetCoreAssignment() const { return assignment_; }

  /*! \brief Calculate runtime args for a specific core */
  RuntimeArgs GetRuntimeArgs(int core_idx) const;

  /*! \brief Get physical core coordinate for a logical core index */
  CoreCoord GetCoreCoord(int core_idx) const;

  /*! \brief Check if a core coordinate is valid */
  bool IsValidCoreCoord(const CoreCoord& coord) const;

  /*! \brief Blackhole grid constants */
  static constexpr int kBlackholeGridX = 14;
  static constexpr int kBlackholeGridY = 10;
  static constexpr int kTotalCores = 140;

 private:
  /*! \brief Analyze T.Kernel grid dimensions from the function */
  CoreAssignment AnalyzeGrid(const tvm::tir::PrimFunc& func);

  /*! \brief Calculate work distribution across cores */
  void CalculateWorkDistribution(CoreAssignment& assignment);

  /*! \brief Store assignment in function attributes */
  void StoreAssignment(tvm::tir::PrimFunc& func, const CoreAssignment& assignment);

  CoreAssignment assignment_;
};

/*!
 * \brief Create the AssignBlackholeCores pass
 * \return The pass function
 */
tir::transform::Pass AssignBlackholeCoresPass();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASSIGN_BLACKHOLE_CORES_H_
