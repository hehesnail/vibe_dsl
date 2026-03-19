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
 * \file plan_blackhole_cb.h
 * \brief Plan Circular Buffer (CB) allocation for Blackhole backend
 *
 * MVP Implementation (Phase 1):
 * - Read CB requirements from function attributes (written by LowerBlackholeOps)
 * - Validate constraints (CB count <= 64, total L1 <= 1.5MB)
 * - Assign CB IDs following TT-Metal convention: 0-15 input, 16-31 output
 * - Store CB configuration in function attributes
 */

#ifndef TVM_TL_PLAN_BLACKHOLE_CB_H_
#define TVM_TL_PLAN_BLACKHOLE_CB_H_

#include "blackhole_cb_common.h"

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief CB configuration result (output from planning)
 */
struct CBConfig {
  std::string name;        // Buffer name
  std::string role;        // input/output/intermediate
  int cb_id;               // Assigned CB identifier (0-63)
  int page_size;           // Size of each page in bytes
  int num_pages;           // Number of pages
  int total_size;          // Total size = num_pages * page_size
  std::string data_format; // Data format string
  int lifetime_begin;      // First requirement slot covered by this memory object
  int lifetime_end;        // Last requirement slot covered by this memory object
  std::vector<int> requirement_indices;         // Requirement indices merged into this memory object
  std::vector<std::string> requirement_names;  // Requirement names merged into this memory object

  CBConfig()
      : role("intermediate"),
        cb_id(0),
        page_size(2048),
        num_pages(2),
        total_size(4096),
        data_format("Float16"),
        lifetime_begin(0),
        lifetime_end(0) {}
};

/*!
 * \brief PlanBlackholeCB Pass
 *
 * This pass analyzes CB requirements and plans CB allocation
 * respecting Blackhole constraints:
 * - Maximum 64 CBs (CB 0-63)
 * - Maximum 1.5MB L1 memory per core
 *
 * CB ID allocation convention (TT-Metal compatible):
 * - CB 0-15: Input buffers (Reader -> Compute)
 * - CB 16-31: Output buffers (Compute -> Writer)
 * - CB 32-63: Intermediate / overflow
 */
class PlanBlackholeCB : public tvm::tir::StmtExprMutator {
 public:
  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  /*! \brief Get CB configurations (after Transform) */
  std::vector<CBConfig> GetCBConfigs() const { return cb_configs_; }

  /*! \brief Blackhole CB constraints */
  static constexpr int kMaxL1Size = 1572864;   // 1.5MB = 1,572,864 bytes
  static constexpr int kMaxCBCount = 64;       // CB 0-63

  // CB ID allocation ranges
  static constexpr int kInputCBStart = 0;
  static constexpr int kInputCBEnd = 15;
  static constexpr int kOutputCBStart = 16;
  static constexpr int kOutputCBEnd = 31;

 private:
  /*! \brief Get CB requirements from function attributes */
  std::vector<CBRequirement> GetCBRequirements(const tvm::tir::PrimFunc& func);

  /*! \brief Infer CB requirements from alloc_shared buffers */
  std::vector<CBRequirement> InferFromAllocShared(const tvm::tir::PrimFunc& func);

  /*! \brief Assign CB IDs to requirements */
  std::vector<CBConfig> AssignCBIds(const std::vector<CBRequirement>& requirements);

  /*! \brief Validate CB allocation constraints */
  bool Validate(const std::vector<CBConfig>& configs) const;

  /*! \brief Store CB configuration in function attributes */
  void StoreCBConfig(tvm::tir::PrimFunc& func, const std::vector<CBConfig>& configs);

  std::vector<CBConfig> cb_configs_;
};

/*!
 * \brief Create the PlanBlackholeCB pass
 * \return The pass function
 */
tir::transform::Pass PlanBlackholeCBPass();

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_PLAN_BLACKHOLE_CB_H_
