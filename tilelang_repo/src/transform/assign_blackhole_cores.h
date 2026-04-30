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
 * \brief Assign T.Kernel grid work items to Blackhole logical worker cores
 */

#ifndef TVM_TL_ASSIGN_BLACKHOLE_CORES_H_
#define TVM_TL_ASSIGN_BLACKHOLE_CORES_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>

#include "common/tt_hardware_model.h"

namespace tvm {
namespace tl {

/*!
 * \brief Core coordinate on Blackhole logical worker grid
 * The coordinate bounds come from TTHardwareModel.
 */
struct CoreCoord {
  int x, y;
};

/*!
 * \brief Core assignment configuration for a kernel
 */
struct CoreAssignment {
  int grid_x, grid_y;           // T.Kernel grid dimensions
  int core_grid_x, core_grid_y; // Blackhole logical worker grid
  int available_worker_cores;   // Usable logical worker cores
  int work_per_core;            // Work items per core
  int cores_needed;             // Total cores needed

  CoreAssignment()
      : grid_x(1), grid_y(1), core_grid_x(11), core_grid_y(10),
        available_worker_cores(110), work_per_core(1), cores_needed(1) {}
};

/*!
 * \brief Runtime arguments for each core
 */
struct RuntimeArgs {
  int work_offset_linear;  // Row-major logical block offset assigned to this core
  int work_count;          // Number of logical blocks assigned to this core

  RuntimeArgs() : work_offset_linear(0), work_count(1) {}
};

/*!
 * \brief PlanTTCoreGroups Pass
 *
 * This pass analyzes T.Kernel grid dimensions and assigns work items
 * to the target hardware model's logical worker core grid.
 */
class PlanTTCoreGroups : public tvm::tir::StmtExprMutator {
 public:
  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(
      const tvm::tir::PrimFunc& func,
      std::optional<TTHardwareModel> hardware_model = std::nullopt);

  /*! \brief Get core assignment result */
  CoreAssignment GetCoreAssignment() const { return assignment_; }

  /*! \brief Calculate runtime args for a specific core */
  RuntimeArgs GetRuntimeArgs(int core_idx) const;

  /*! \brief Get logical worker core coordinate for a logical core index */
  CoreCoord GetCoreCoord(int core_idx) const;

  /*! \brief Check if a core coordinate is valid */
  bool IsValidCoreCoord(const CoreCoord& coord) const;

  /*! \brief Conservative fallback grid constants */
  static constexpr int kBlackholeGridX = 11;
  static constexpr int kBlackholeGridY = 10;
  static constexpr int kTotalCores = 110;

 private:
  /*! \brief Analyze T.Kernel grid dimensions from the function */
  CoreAssignment AnalyzeGrid(const tvm::tir::PrimFunc& func);

  /*! \brief Calculate work distribution across cores */
  void CalculateWorkDistribution(CoreAssignment& assignment);

  /*! \brief Apply target hardware limits to a logical assignment. */
  void ApplyHardwareModel(CoreAssignment& assignment,
                          const std::optional<TTHardwareModel>& hardware_model);

  /*! \brief Store assignment in function attributes */
  void StoreAssignment(tvm::tir::PrimFunc& func, const CoreAssignment& assignment);

  CoreAssignment assignment_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASSIGN_BLACKHOLE_CORES_H_
