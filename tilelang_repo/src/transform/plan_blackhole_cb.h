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
 */

#ifndef TVM_TL_PLAN_BLACKHOLE_CB_H_
#define TVM_TL_PLAN_BLACKHOLE_CB_H_

#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/op.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief CB configuration for a shared buffer
 */
struct CBConfig {
  int cb_id;           // CB identifier (0-63)
  int num_pages;       // Number of pages (for double buffering)
  int page_size;       // Size of each page in bytes
  int total_size;      // Total size = num_pages * page_size
  DataType dtype;      // Data type of buffer elements

  CBConfig() : cb_id(0), num_pages(1), page_size(0), total_size(0),
               dtype(DataType::Float(32)) {}
};

/*!
 * \brief PlanBlackholeCB Pass
 *
 * This pass analyzes T.alloc_shared statements and plans CB allocation
 * respecting Blackhole constraints:
 * - Maximum 64 CBs (CB 0-63)
 * - Maximum 1.5MB L1 memory per core
 */
class PlanBlackholeCB : public StmtExprMutator {
 public:
  /*! \brief Main entry point */
  PrimFunc Transform(const PrimFunc& func);

  /*! \brief Get CB configurations */
  std::vector<CBConfig> GetCBConfigs() const { return cb_configs_; }

  /*! \brief Validate CB allocation constraints */
  bool Validate() const;

  /*! \brief Calculate page size for a tile */
  static int CalculatePageSize(int rows, int cols, int dtype_size);

  /*! \brief Blackhole CB constraints */
  static constexpr int kMaxCBSize = 1572864;   // 1.5MB
  static constexpr int kMaxCBCount = 64;       // CB 0-63

 private:
  /*! \brief Visitor to collect alloc_shared statements */
  class AllocSharedCollector : public StmtExprVisitor {
   public:
    std::vector<Buffer> shared_buffers;

   private:
    void VisitStmt_(const AllocateNode* op) final;
  };

  /*! \brief Analyze alloc_shared statements and create CB configs */
  std::vector<CBConfig> AnalyzeAllocShared(const PrimFunc& func);

  /*! \brief Assign CB IDs to configurations */
  void AssignCBIds(std::vector<CBConfig>& configs);

  /*! \brief Store CB config in function attributes */
  void StoreCBConfig(PrimFunc& func, const std::vector<CBConfig>& configs);

  std::vector<CBConfig> cb_configs_;
};

/*!
 * \brief Create the PlanBlackholeCB pass
 * \return The pass function
 */
tvm::tir::transform::Pass PlanBlackholeCBPass();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_PLAN_BLACKHOLE_CB_H_
