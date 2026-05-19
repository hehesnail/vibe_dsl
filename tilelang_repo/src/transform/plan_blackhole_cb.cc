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
 * \file plan_blackhole_cb.cc
 * \brief Plan Circular Buffer (CB) allocation for Blackhole backend
 *
 * MVP Implementation (Phase 1):
 * - Read staged CB requirements from TTProgram cb_plans
 * - Validate constraints from TTHardwareModel (default CB count <= 64, total L1 <= 1.5MB)
 * - Assign CB IDs following TT-Metal convention: 0-15 input, 16-31 output
 * - Rewrite placeholder requirement indices in the IR to final CB IDs
 */

#include "plan_blackhole_cb.h"
#include "common/companion_base.h"
#include "common/tt_hardware_model.h"
#include "common/tt_target_program.h"
#include "../tir/builtin_blackhole.h"

#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using tir::PrimFunc;
using tir::PrimFuncNode;
using tvm::DataType;
using tvm::Integer;
using tvm::DictAttrs;
using tvm::ffi::String;
using tvm::ffi::Map;
using tvm::ffi::Array;
using tvm::ffi::Any;

// Blackhole / TT-Metal compute API CB identifiers are architectural CB indices.
// Kernel APIs such as pack_tile and cb_wait_front operate on architectural CB IDs.
constexpr int kDefaultMaxCBs = 64;
constexpr int kDefaultMaxL1Size = 1572864;  // 1.5MB = 1,572,864 bytes

// CB ID allocation ranges
constexpr int kInputCBStart = 0;
constexpr int kInputCBEnd = 15;
constexpr int kOutputCBStart = 16;
constexpr int kOutputCBEnd = 31;

namespace {

struct CBRequirementUseInfo {
  bool referenced = false;
  int first_use = std::numeric_limits<int>::max();
  int last_use = -1;
};

struct CBRequirementEventInfo {
  int max_reserve_pages = 0;
  int max_push_pages = 0;
  int max_wait_pages = 0;
  int max_pop_pages = 0;
};

bool IsBlackholeOp(const tir::CallNode* op, const char* op_name) {
  if (op == nullptr || !op->op->IsInstance<OpNode>()) {
    return false;
  }
  return Downcast<Op>(op->op)->name == op_name;
}

std::string RoleForType(CBType type) {
  switch (type) {
    case CBType::kInput:
      return "input";
    case CBType::kOutput:
      return "output";
    case CBType::kIntermediate:
    default:
      return "intermediate";
  }
}

std::string CBFlowClassToString(CBFlowClass flow_class) {
  switch (flow_class) {
    case CBFlowClass::kStream:
      return "stream";
    case CBFlowClass::kRepublish:
      return "republish";
    case CBFlowClass::kState:
    default:
      return "state";
  }
}

CBFlowClass CBFlowClassFromString(const std::string& flow_class) {
  if (flow_class == "stream") {
    return CBFlowClass::kStream;
  }
  if (flow_class == "republish") {
    return CBFlowClass::kRepublish;
  }
  return CBFlowClass::kState;
}

bool IsExclusiveRequirement(int requirement_index,
                            const std::vector<bool>& exclusive_requirements) {
  return requirement_index >= 0 &&
         requirement_index < static_cast<int>(exclusive_requirements.size()) &&
         exclusive_requirements[requirement_index];
}

bool ConfigHasExclusiveRequirement(const CBConfig& config,
                                   const std::vector<bool>& exclusive_requirements) {
  return std::any_of(config.requirement_indices.begin(),
                     config.requirement_indices.end(),
                     [&](int requirement_index) {
                       return IsExclusiveRequirement(requirement_index,
                                                     exclusive_requirements);
                     });
}

bool IsCompatibleForReuse(const CBRequirement& req, int req_index,
                          const CBConfig& config,
                          const std::vector<bool>& exclusive_requirements) {
  if (IsExclusiveRequirement(req_index, exclusive_requirements) ||
      ConfigHasExclusiveRequirement(config, exclusive_requirements)) {
    return false;
  }
  if ((req.initial_reserve_pages > 0 && req.flow_class == CBFlowClass::kState) ||
      (config.initial_reserve_pages > 0 && config.flow_class == CBFlowClass::kState)) {
    return false;
  }
  if (config.role != RoleForType(req.type)) {
    return false;
  }
  if (config.name != req.name) {
    return false;
  }
  if (req.type == CBType::kInput) {
    return false;
  }
  if (config.page_size != req.page_size) {
    return false;
  }
  if (config.num_pages != req.num_pages) {
    return false;
  }
  if (config.initial_reserve_pages != req.initial_reserve_pages) {
    return false;
  }
  if (config.flow_class != req.flow_class) {
    return false;
  }
  if (config.publish_pages_per_event != req.publish_pages_per_event) {
    return false;
  }
  if (config.consume_pages_per_event != req.consume_pages_per_event) {
    return false;
  }
  if (config.data_format != req.data_format) {
    return false;
  }
  return req.lifetime_begin > config.lifetime_end;
}

std::vector<int> GetCBArgPositions(const std::string& op_name) {
  if (op_name == "tl.blackhole.cb_reserve_back" ||
      op_name == "tl.blackhole.cb_push_back" ||
      op_name == "tl.blackhole.cb_wait_front" ||
      op_name == "tl.blackhole.cb_pop_front" ||
      op_name == "tl.blackhole.copy_cb_page" ||
      op_name == "tl.blackhole.write_tile_from_cb" ||
      op_name == "tl.blackhole.write_page_from_cb" ||
      op_name == "tl.blackhole.zero_cb_page" ||
      op_name == "tl.blackhole.guard_mask_to_cb") {
    if (op_name == "tl.blackhole.copy_cb_page") {
      return {0, 1};
    }
    return {0};
  }
  if (op_name == "tl.blackhole.read_tile_to_cb" ||
      op_name == "tl.blackhole.read_page_to_cb" ||
      op_name == "tl.blackhole.read_bcast_cols_to_cb") {
    return {2};
  }
  if (op_name == "tl.blackhole.mm_init") {
    return {0, 1, 2};
  }
  if (op_name == "tl.blackhole.reconfig_data_format") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.mm_init_short") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.mm_init_short_with_dt") {
    return {0, 1, 2};
  }
  if (op_name == "tl.blackhole.matmul_tiles") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.pack_tile") {
    return {1};
  }
  if (op_name == "tl.blackhole.pack_reconfig_data_format") {
    return {0};
  }
  if (op_name == "tl.blackhole.copy_tile") {
    return {0};
  }
  if (op_name == "tl.blackhole.copy_tile_to_dst_init_short" ||
      op_name == "tl.blackhole.copy_tile_to_dst_init_short_with_dt") {
    return op_name == "tl.blackhole.copy_tile_to_dst_init_short_with_dt" ? std::vector<int>{0, 1}
                                                                          : std::vector<int>{0};
  }
  if (op_name == "tl.blackhole.binary_op_init_common") {
    return {0, 1, 2};
  }
  if (op_name == "tl.blackhole.unary_op_init_common") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.add_tiles_init" || op_name == "tl.blackhole.add_tiles") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.sub_tiles_init" || op_name == "tl.blackhole.sub_tiles") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.mul_tiles_init" || op_name == "tl.blackhole.mul_tiles") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.add_bcast_rows_init_short" ||
      op_name == "tl.blackhole.add_tiles_bcast_rows" ||
      op_name == "tl.blackhole.add_bcast_cols_init_short" ||
      op_name == "tl.blackhole.add_tiles_bcast_cols" ||
      op_name == "tl.blackhole.mul_bcast_rows_init_short" ||
      op_name == "tl.blackhole.mul_bcast_cols_init_short" ||
      op_name == "tl.blackhole.mul_tiles_bcast_rows" ||
      op_name == "tl.blackhole.mul_tiles_bcast_cols") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.reduce_init") {
    return {0, 1, 2};
  }
  if (op_name == "tl.blackhole.reduce_tile") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.pack_untilize_slice") {
    return {1};  // args: (src_handle, cb_id, dst_offset, num_elements, src_offset)
  }
  if (op_name == "tl.blackhole.pack_untilize_tile") {
    return {1};  // args: (src_handle, cb_id, dst_tile_index, src_offset)
  }
  if (op_name == "tl.blackhole.tilize_local_fragment_slice") {
    return {1};  // args: (src_handle, cb_id, dst_offset, num_elements, row_width, src_offset)
  }
  if (op_name == "tl.blackhole.tilize_cast_fragment_slice") {
    return {2};  // args: (dst_handle, src_handle, cb_id, dst_offset, src_offset, num_elements, row_width)
  }
  if (op_name == "tl.blackhole.untilize_cb_front_tile") {
    return {1};  // args: (dst_handle, src_cb_id, src_tile_index, dst_offset, num_elements)
  }
  if (op_name == "tl.blackhole.untilize_cb_front_tile_fragment") {
    return {1};  // args: (dst_handle, src_cb_id, src_tile_index, dst_offset)
  }
  if (op_name == "tl.blackhole.pack_fill_fragment_to_tiled_cb") {
    return {1};  // args: (dst_handle, cb_id, dst_offset, num_elements, row_width, value)
  }
  if (op_name == "tl.blackhole.add_fragment_from_cb_front") {
    return {1};  // args: (dst_handle, src_cb_id, num_elements)
  }
  return {};
}

bool HasNoCBArgs(const std::string& op_name) {
  return op_name == "tl.blackhole.cast_fragment_slice" ||
         op_name == "tl.blackhole.add_fragment" ||
         op_name == "tl.blackhole.fill_fragment" ||
         op_name == "tl.blackhole.tile_regs_acquire" ||
         op_name == "tl.blackhole.tile_regs_commit" ||
         op_name == "tl.blackhole.tile_regs_wait" ||
         op_name == "tl.blackhole.tile_regs_release" ||
         op_name == "tl.blackhole.binary_max_tile_init" ||
         op_name == "tl.blackhole.binary_max_tile" ||
         op_name == "tl.blackhole.div_binary_tile_init" ||
         op_name == "tl.blackhole.div_binary_tile" ||
         op_name == "tl.blackhole.exp_tile_init" ||
         op_name == "tl.blackhole.exp_tile" ||
         op_name == "tl.blackhole.exp2_tile_init" ||
         op_name == "tl.blackhole.exp2_tile" ||
         op_name == "tl.blackhole.recip_tile_init" ||
         op_name == "tl.blackhole.recip_tile" ||
         op_name == "tl.blackhole.reduce_uninit";
}

std::vector<CBRequirementUseInfo> CollectCBRequirementUseInfo(const tir::Stmt& body,
                                                              int requirement_count) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(int requirement_count)
        : use_info_(std::max(0, requirement_count)) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    void VisitExpr_(const tir::CallNode* op) final {
      if (op->op->IsInstance<OpNode>()) {
        const std::vector<int> positions = GetCBArgPositions(Downcast<Op>(op->op)->name);
        if (!positions.empty()) {
          const int position = next_position_++;
          for (int pos : positions) {
            ICHECK_LT(pos, static_cast<int>(op->args.size()));
            if (const auto* imm = op->args[pos].as<IntImmNode>()) {
              const int requirement_index = static_cast<int>(imm->value);
              ICHECK_GE(requirement_index, 0)
                  << "PlanTTCBAlloc expects non-negative requirement_index placeholders";
              ICHECK_LT(requirement_index, static_cast<int>(use_info_.size()))
                  << "PlanTTCBAlloc found requirement_index=" << requirement_index
                  << " outside staged requirement count=" << use_info_.size();
              auto& info = use_info_[requirement_index];
              info.referenced = true;
              info.first_use = std::min(info.first_use, position);
              info.last_use = std::max(info.last_use, position);
            }
          }
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    std::vector<CBRequirementUseInfo> Take() { return std::move(use_info_); }

   private:
    int next_position_ = 0;
    std::vector<CBRequirementUseInfo> use_info_;
  };

  Collector collector(requirement_count);
  collector.Collect(body);
  return collector.Take();
}

std::vector<bool> ReferencedRequirementMask(
    const std::vector<CBRequirementUseInfo>& use_info) {
  std::vector<bool> referenced;
  referenced.reserve(use_info.size());
  for (const auto& info : use_info) {
    referenced.push_back(info.referenced);
  }
  return referenced;
}

std::vector<CBRequirementEventInfo> CollectCBRequirementEventInfo(const tir::Stmt& body,
                                                                  int requirement_count) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(int requirement_count)
        : event_info_(std::max(0, requirement_count)) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    void VisitExpr_(const tir::CallNode* op) final {
      if (op->args.size() >= 2U) {
        const auto* cb_id = op->args[0].as<IntImmNode>();
        const auto* pages = op->args[1].as<IntImmNode>();
        if (cb_id != nullptr && pages != nullptr && cb_id->value >= 0 &&
            cb_id->value < static_cast<int64_t>(event_info_.size()) && pages->value > 0) {
          auto& info = event_info_[static_cast<size_t>(cb_id->value)];
          const int page_count = static_cast<int>(pages->value);
          if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
            info.max_reserve_pages = std::max(info.max_reserve_pages, page_count);
          } else if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
            info.max_push_pages = std::max(info.max_push_pages, page_count);
          } else if (IsBlackholeOp(op, "tl.blackhole.cb_wait_front")) {
            info.max_wait_pages = std::max(info.max_wait_pages, page_count);
          } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
            info.max_pop_pages = std::max(info.max_pop_pages, page_count);
          }
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    std::vector<CBRequirementEventInfo> Take() { return std::move(event_info_); }

   private:
    std::vector<CBRequirementEventInfo> event_info_;
  };

  Collector collector(requirement_count);
  collector.Collect(body);
  return collector.Take();
}

std::vector<bool> CollectExclusiveRequirementMask(const tir::Stmt& body,
                                                  int requirement_count) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(int requirement_count)
        : exclusive_(std::max(0, requirement_count), false) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    void VisitExpr_(const tir::CallNode* op) final {
      if (IsBlackholeOp(op, "tl.blackhole.guard_mask_to_cb") &&
          !op->args.empty()) {
        if (const auto* cb_id = op->args[0].as<IntImmNode>()) {
          if (cb_id->value >= 0 &&
              cb_id->value < static_cast<int64_t>(exclusive_.size())) {
            exclusive_[static_cast<size_t>(cb_id->value)] = true;
          }
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    std::vector<bool> Take() { return std::move(exclusive_); }

   private:
    std::vector<bool> exclusive_;
  };

  Collector collector(requirement_count);
  collector.Collect(body);
  return collector.Take();
}

std::vector<int> CollectOutstandingReservedPages(const tir::Stmt& body,
                                                 int requirement_count) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(int requirement_count)
        : outstanding_pages_(std::max(0, requirement_count), 0) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    void VisitExpr_(const tir::CallNode* op) final {
      int delta_sign = 0;
      if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
        delta_sign = 1;
      } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
        delta_sign = -1;
      }
      if (delta_sign != 0 && op->args.size() >= 2U) {
        const auto* cb_id = op->args[0].as<IntImmNode>();
        const auto* pages = op->args[1].as<IntImmNode>();
        if (cb_id != nullptr && pages != nullptr && cb_id->value >= 0 &&
            cb_id->value < static_cast<int64_t>(outstanding_pages_.size())) {
          outstanding_pages_[static_cast<size_t>(cb_id->value)] +=
              delta_sign * static_cast<int>(pages->value);
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    std::vector<int> Take() { return std::move(outstanding_pages_); }

   private:
    std::vector<int> outstanding_pages_;
  };

  Collector collector(requirement_count);
  collector.Collect(body);
  return collector.Take();
}

std::vector<int> CollectMaxPhysicalOutstandingPages(const tir::Stmt& body, int cb_count) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(int cb_count)
        : reserved_pages_(std::max(0, cb_count), 0),
          front_pages_(std::max(0, cb_count), 0),
          max_outstanding_pages_(std::max(0, cb_count), 0) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    void VisitExpr_(const tir::CallNode* op) final {
      if (op->args.size() < 2U) {
        tir::StmtExprVisitor::VisitExpr_(op);
        return;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      const auto* pages = op->args[1].as<IntImmNode>();
      if (cb_id != nullptr && pages != nullptr && cb_id->value >= 0 &&
          cb_id->value < static_cast<int64_t>(reserved_pages_.size()) && pages->value > 0) {
        const int id = static_cast<int>(cb_id->value);
        const int page_count = static_cast<int>(pages->value);
        if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
          reserved_pages_[id] += page_count;
        } else if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
          reserved_pages_[id] = std::max(0, reserved_pages_[id] - page_count);
          front_pages_[id] += page_count;
        } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
          front_pages_[id] = std::max(0, front_pages_[id] - page_count);
        }
        max_outstanding_pages_[id] =
            std::max(max_outstanding_pages_[id], reserved_pages_[id] + front_pages_[id]);
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

    std::vector<int> Take() { return std::move(max_outstanding_pages_); }

   private:
    std::vector<int> reserved_pages_;
    std::vector<int> front_pages_;
    std::vector<int> max_outstanding_pages_;
  };

  Collector collector(cb_count);
  collector.Collect(body);
  return collector.Take();
}

void ApplyMaxPhysicalOutstandingPages(const tir::Stmt& body, int cb_count,
                                      std::vector<CBConfig>* configs) {
  if (!body.defined() || configs == nullptr || cb_count <= 0) {
    return;
  }
  const std::vector<int> max_outstanding_pages =
      CollectMaxPhysicalOutstandingPages(body, cb_count);
  for (CBConfig& config : *configs) {
    if (config.cb_id >= 0 &&
        config.cb_id < static_cast<int>(max_outstanding_pages.size()) &&
        max_outstanding_pages[config.cb_id] > config.num_pages) {
      config.num_pages = max_outstanding_pages[config.cb_id];
      config.total_size = config.page_size * config.num_pages;
    }
  }
}

bool CanAutoManageStateFront(const CBRequirement& req) {
  return (req.type == CBType::kOutput || req.type == CBType::kIntermediate) &&
         req.flow_class == CBFlowClass::kState && req.initial_reserve_pages == 0;
}

bool CanAutoPopFrontBeforeProducer(const CBRequirement& req) {
  if (CanAutoManageStateFront(req)) {
    return true;
  }
  return req.type == CBType::kIntermediate &&
         req.flow_class == CBFlowClass::kStream &&
         req.initial_reserve_pages == 0 &&
         std::max(req.publish_pages_per_event, req.consume_pages_per_event) > 0;
}

bool IsDirectLocalStateConfig(const CBConfig& config) {
  return config.role == "intermediate" && config.flow_class == CBFlowClass::kState &&
         config.initial_reserve_pages == 0 && config.publish_pages_per_event == 0 &&
         config.consume_pages_per_event == 0;
}

tir::Stmt MakeCBPopFrontStmt(int requirement_index, int pages) {
  ICHECK_GT(pages, 0);
  return tir::Evaluate(tir::Call(
      DataType::Int(32), tir::builtin::blackhole_cb_pop_front(),
      {tvm::IntImm(DataType::Int(32), requirement_index),
       tvm::IntImm(DataType::Int(32), pages)}));
}

tir::Stmt MakeCBWaitFrontStmt(int requirement_index, int pages) {
  ICHECK_GT(pages, 0);
  return tir::Evaluate(tir::Call(
      DataType::Int(32), tir::builtin::blackhole_cb_wait_front(),
      {tvm::IntImm(DataType::Int(32), requirement_index),
       tvm::IntImm(DataType::Int(32), pages)}));
}

tir::Stmt InsertStateFrontPopsBeforeReReserve(
    const tir::Stmt& body,
    const std::vector<CBRequirement>& requirements) {
  class Inserter final : public tir::StmtExprMutator {
   public:
    explicit Inserter(const std::vector<CBRequirement>& requirements)
        : requirements_(requirements), available_front_pages_(requirements.size(), 0) {}

    using tir::StmtExprMutator::VisitStmt_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    tir::Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
      Array<tir::Stmt> rewritten;
      rewritten.reserve(op->seq.size());
      for (const tir::Stmt& child : op->seq) {
        if (const auto* eval = child.as<tir::EvaluateNode>()) {
          if (const auto* call = eval->value.as<tir::CallNode>()) {
            MaybeInsertPopBeforeStateProducer(call, &rewritten);
          }
        }
        tir::Stmt new_child = VisitStmt(child);
        rewritten.push_back(new_child);
        if (const auto* eval = new_child.as<tir::EvaluateNode>()) {
          if (const auto* call = eval->value.as<tir::CallNode>()) {
            RecordQueueMutation(call);
          }
        }
      }
      return tir::SeqStmt::Flatten(rewritten);
    }

   private:
    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(requirements_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    void MaybeInsertPopBeforeStateProducer(const tir::CallNode* op, Array<tir::Stmt>* stmts) {
      ICHECK(stmts != nullptr);
      const bool is_reserve = IsBlackholeOp(op, "tl.blackhole.cb_reserve_back");
      const bool is_pack_reconfig =
          IsBlackholeOp(op, "tl.blackhole.pack_reconfig_data_format");
      if (!is_reserve && !is_pack_reconfig) {
        return;
      }
      const int cb_id = StaticCBId(op);
      if (cb_id < 0 || !CanAutoPopFrontBeforeProducer(requirements_[cb_id]) ||
          available_front_pages_[cb_id] <= 0) {
        return;
      }
      const int event_pages =
          requirements_[cb_id].consume_pages_per_event > 0
              ? requirements_[cb_id].consume_pages_per_event
              : std::max(1, requirements_[cb_id].publish_pages_per_event);
      const int capacity_pages =
          requirements_[cb_id].flow_class == CBFlowClass::kStream
              ? std::max(1, event_pages)
              : std::max(1, requirements_[cb_id].num_pages);
      const int producer_pages =
          is_reserve ? std::max(1, StaticPages(op)) : event_pages;
      const int overflow_pages =
          std::max(0, available_front_pages_[cb_id] + producer_pages - capacity_pages);
      if (overflow_pages <= 0) {
        return;
      }
      const int pop_pages = std::min(available_front_pages_[cb_id],
                                     std::max(overflow_pages, event_pages));
      stmts->push_back(MakeCBPopFrontStmt(cb_id, pop_pages));
      available_front_pages_[cb_id] -= pop_pages;
    }

    void RecordQueueMutation(const tir::CallNode* op) {
      const int cb_id = StaticCBId(op);
      if (cb_id < 0 || !CanAutoPopFrontBeforeProducer(requirements_[cb_id])) {
        return;
      }
      const int pages = StaticPages(op);
      if (pages <= 0) {
        return;
      }
      if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
        available_front_pages_[cb_id] += pages;
      } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
        available_front_pages_[cb_id] =
            std::max(0, available_front_pages_[cb_id] - pages);
      }
    }

    const std::vector<CBRequirement>& requirements_;
    std::vector<int> available_front_pages_;
  };

  Inserter inserter(requirements);
  return inserter.Rewrite(body);
}

std::vector<int> ComputeLoopLocalBackedgePopPages(
    const tir::Stmt& body,
    const std::vector<CBRequirement>& requirements) {
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(const std::vector<CBRequirement>& requirements)
        : requirements_(requirements),
          pushed_pages_(requirements.size(), 0),
          popped_pages_(requirements.size(), 0),
          locally_available_pages_(requirements.size(), 0),
          waits_after_local_publish_(requirements.size(), 0) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& body) { VisitStmt(body); }

    std::vector<int> Take() const {
      std::vector<int> pop_pages(requirements_.size(), 0);
      for (size_t i = 0; i < requirements_.size(); ++i) {
        const CBRequirement& req = requirements_[i];
        if (req.type == CBType::kInput || req.flow_class == CBFlowClass::kState) {
          continue;
        }
        const int net_local_front = pushed_pages_[i] - popped_pages_[i];
        if (net_local_front <= 0 || waits_after_local_publish_[i] <= 0) {
          continue;
        }
        pop_pages[i] = net_local_front;
      }
      return pop_pages;
    }

    void VisitExpr_(const tir::CallNode* op) final {
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (cb_id >= 0 && pages > 0) {
        if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
          pushed_pages_[cb_id] += pages;
          locally_available_pages_[cb_id] += pages;
        } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
          popped_pages_[cb_id] += pages;
          locally_available_pages_[cb_id] =
              std::max(0, locally_available_pages_[cb_id] - pages);
        } else if (IsBlackholeOp(op, "tl.blackhole.cb_wait_front") &&
                   locally_available_pages_[cb_id] > 0) {
          waits_after_local_publish_[cb_id] =
              std::max(waits_after_local_publish_[cb_id], pages);
        }
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

   private:
    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(requirements_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    const std::vector<CBRequirement>& requirements_;
    std::vector<int> pushed_pages_;
    std::vector<int> popped_pages_;
    std::vector<int> locally_available_pages_;
    std::vector<int> waits_after_local_publish_;
  };

  Collector collector(requirements);
  collector.Collect(body);
  return collector.Take();
}

tir::Stmt InsertLoopLocalBackedgePops(
    const tir::Stmt& body,
    const std::vector<CBRequirement>& requirements) {
  class Inserter final : public tir::StmtExprMutator {
   public:
    explicit Inserter(const std::vector<CBRequirement>& requirements)
        : requirements_(requirements) {}

    using tir::StmtExprMutator::VisitStmt_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    tir::Stmt VisitStmt_(const tir::ForNode* op) final {
      tir::Stmt body = VisitStmt(op->body);
      body = AppendBackedgePops(body);
      return tir::For(op->loop_var, op->min, op->extent, op->kind, body,
                      op->thread_binding, op->annotations, op->step, op->span);
    }

   private:
    tir::Stmt AppendBackedgePops(const tir::Stmt& body) const {
      const std::vector<int> pop_pages =
          ComputeLoopLocalBackedgePopPages(body, requirements_);
      std::vector<tir::Stmt> suffix;
      suffix.push_back(body);
      for (size_t i = 0; i < pop_pages.size(); ++i) {
        if (pop_pages[i] > 0) {
          suffix.push_back(MakeCBPopFrontStmt(static_cast<int>(i), pop_pages[i]));
        }
      }
      if (suffix.size() == 1U) {
        return body;
      }
      return tir::SeqStmt::Flatten(suffix);
    }

    const std::vector<CBRequirement>& requirements_;
  };

  Inserter inserter(requirements);
  return inserter.Rewrite(body);
}

void ApplyIRUseIntervalsToRequirements(
    std::vector<CBRequirement>* requirements,
    const std::vector<CBRequirementUseInfo>& use_info) {
  ICHECK(requirements != nullptr);
  ICHECK_EQ(requirements->size(), use_info.size());
  for (size_t i = 0; i < requirements->size(); ++i) {
    auto& req = (*requirements)[i];
    const auto& info = use_info[i];
    if (info.referenced) {
      req.lifetime_begin = info.first_use;
      req.lifetime_end = info.last_use;
    } else {
      // Metadata-only requirements do not own an executable CB lifetime. Keep
      // their ordering stable without letting old transitive overlap widening
      // force them to consume unique architectural CB IDs.
      req.lifetime_begin = static_cast<int>(i);
      req.lifetime_end = static_cast<int>(i);
    }
  }
}

void ApplyIRCBEventsToRequirements(
    std::vector<CBRequirement>* requirements,
    const std::vector<CBRequirementEventInfo>& event_info) {
  ICHECK(requirements != nullptr);
  ICHECK_EQ(requirements->size(), event_info.size());
  for (size_t i = 0; i < requirements->size(); ++i) {
    CBRequirement& req = requirements->at(i);
    const CBRequirementEventInfo& info = event_info[i];
    const int produced_pages = std::max(info.max_reserve_pages, info.max_push_pages);
    const int consumed_pages = std::max(info.max_wait_pages, info.max_pop_pages);
    if (req.flow_class == CBFlowClass::kState &&
        req.publish_pages_per_event == 0 &&
        req.consume_pages_per_event == 0 &&
        produced_pages > 0 && consumed_pages > 0) {
      req.flow_class = CBFlowClass::kRepublish;
      req.publish_pages_per_event = produced_pages;
      req.consume_pages_per_event = consumed_pages;
    }
  }
}

void ApplyExactCBLiveIntervalsToRequirements(
    const TTProgram& program,
    std::vector<CBRequirement>* requirements) {
  ICHECK(requirements != nullptr);
  if (program->exact_cb_live_intervals.empty() ||
      program->exact_cb_allocations.empty()) {
    return;
  }

  struct IntervalRange {
    int begin_point;
    int end_point;
  };
  std::unordered_map<int64_t, IntervalRange> interval_by_virtual_value_index;
  for (const TTExactCBLiveInterval& interval :
       program->exact_cb_live_intervals) {
    ICHECK_GE(interval->virtual_value_index, 0)
        << "TTExactCBLiveInterval requires virtual_value_index before CB "
           "allocation";
    const int begin_point =
        static_cast<int>(std::max<int64_t>(0, interval->begin_point));
    const int end_point = static_cast<int>(
        std::max<int64_t>(interval->begin_point, interval->end_point));
    ICHECK(interval_by_virtual_value_index
               .emplace(interval->virtual_value_index,
                        IntervalRange{begin_point, end_point})
               .second)
        << "duplicate TTExactCBLiveInterval for virtual_value "
        << interval->virtual_value;
  }

  for (const TTExactCBAllocation& allocation : program->exact_cb_allocations) {
    ICHECK_GE(allocation->cb_plan_index, 0)
        << "TTExactCBAllocation requires requirement-index cb_plan_index "
           "before CB allocation";
    ICHECK_LT(allocation->cb_plan_index,
              static_cast<int64_t>(requirements->size()))
        << "TTExactCBAllocation cb_plan_index out of staged requirement "
           "range before CB allocation";
    auto interval_it =
        interval_by_virtual_value_index.find(allocation->virtual_value_index);
    ICHECK(interval_it != interval_by_virtual_value_index.end())
        << "TTExactCBAllocation requires matching TTExactCBLiveInterval before "
           "CB allocation";
    CBRequirement& req =
        requirements->at(static_cast<size_t>(allocation->cb_plan_index));
    req.lifetime_begin = std::min(req.lifetime_begin,
                                  interval_it->second.begin_point);
    req.lifetime_end = std::max(req.lifetime_end,
                                interval_it->second.end_point);
  }
}

std::vector<int> ComputeAutoPopPages(
    const std::vector<CBRequirement>& requirements,
    const std::vector<CBRequirementUseInfo>& use_info,
    const tir::Stmt& body) {
  ICHECK_EQ(requirements.size(), use_info.size());
  const std::vector<int> outstanding_pages =
      CollectOutstandingReservedPages(body, static_cast<int>(requirements.size()));
  std::vector<int> auto_pop_pages(requirements.size(), 0);
  for (size_t i = 0; i < requirements.size(); ++i) {
    const auto& req = requirements[i];
    if (!use_info[i].referenced || outstanding_pages[i] <= 0) {
      continue;
    }
    if (!CanAutoManageStateFront(req)) {
      continue;
    }
    auto_pop_pages[i] = outstanding_pages[i];
  }
  return auto_pop_pages;
}

tir::Stmt InsertAutoPopsAfterLastUse(const tir::Stmt& body,
                                     const std::vector<CBRequirementUseInfo>& use_info,
                                     const std::vector<int>& auto_pop_pages) {
  class Inserter final : public tir::StmtExprMutator {
   public:
    Inserter(const std::vector<CBRequirementUseInfo>& use_info,
             const std::vector<int>& auto_pop_pages)
        : use_info_(use_info), auto_pop_pages_(auto_pop_pages) {
      ICHECK_EQ(use_info_.size(), auto_pop_pages_.size());
    }

    using tir::StmtExprMutator::VisitExpr_;
    using tir::StmtExprMutator::VisitStmt_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    tir::Stmt VisitStmt_(const tir::ForNode* op) final {
      ++loop_depth_;
      tir::Stmt body = VisitStmt(op->body);
      --loop_depth_;
      return tir::For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
                      op->annotations, op->step, op->span);
    }

    tir::Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
      Array<tir::Stmt> rewritten;
      rewritten.reserve(op->seq.size());
      for (const tir::Stmt& child : op->seq) {
        tir::Stmt new_child = VisitStmt(child);
        rewritten.push_back(new_child);
        if (ShouldFlushAfterStmt(new_child)) {
          FlushPendingPops(&rewritten);
        }
      }
      FlushPendingPops(&rewritten);
      return tir::SeqStmt::Flatten(rewritten);
    }

    PrimExpr VisitExpr_(const tir::CallNode* op) final {
      if (op->op->IsInstance<OpNode>()) {
        const std::vector<int> positions = GetCBArgPositions(Downcast<Op>(op->op)->name);
        if (!positions.empty()) {
          const int position = next_position_++;
          for (int pos : positions) {
            ICHECK_LT(pos, static_cast<int>(op->args.size()));
            if (const auto* imm = op->args[pos].as<IntImmNode>()) {
              const int requirement_index = static_cast<int>(imm->value);
              if (requirement_index >= 0 &&
                  requirement_index < static_cast<int>(auto_pop_pages_.size()) &&
                  auto_pop_pages_[requirement_index] > 0 &&
                  use_info_[requirement_index].last_use == position &&
                  emitted_pops_.insert(requirement_index).second) {
                pending_pops_.push_back(requirement_index);
              }
            }
          }
        }
      }
      PrimExpr expr = tir::StmtExprMutator::VisitExpr_(op);
      const auto* rewritten = expr.as<tir::CallNode>();
      if (IsBlackholeOp(rewritten, "tl.blackhole.tile_regs_acquire")) {
        ++tile_section_depth_;
      } else if (IsBlackholeOp(rewritten, "tl.blackhole.tile_regs_release") &&
                 tile_section_depth_ > 0) {
        --tile_section_depth_;
      }
      return expr;
    }

   private:
    bool ShouldFlushAfterStmt(const tir::Stmt& stmt) const {
      if (pending_pops_.empty()) {
        return false;
      }
      if (const auto* eval = stmt.as<tir::EvaluateNode>()) {
        if (IsBlackholeOp(eval->value.as<tir::CallNode>(), "tl.blackhole.tile_regs_release")) {
          return true;
        }
      }
      return tile_section_depth_ == 0;
    }

    void FlushPendingPops(Array<tir::Stmt>* stmts) {
      ICHECK(stmts != nullptr);
      if (pending_pops_.empty() || tile_section_depth_ != 0 || loop_depth_ != 0) {
        return;
      }
      for (int requirement_index : pending_pops_) {
        const int pages = auto_pop_pages_[requirement_index];
        ICHECK_GT(pages, 0);
        stmts->push_back(MakeCBPopFrontStmt(requirement_index, pages));
      }
      pending_pops_.clear();
    }

    const std::vector<CBRequirementUseInfo>& use_info_;
    const std::vector<int>& auto_pop_pages_;
    int next_position_ = 0;
    int tile_section_depth_ = 0;
    int loop_depth_ = 0;
    std::vector<int> pending_pops_;
    std::unordered_set<int> emitted_pops_;
  };

  Inserter inserter(use_info, auto_pop_pages);
  return inserter.Rewrite(body);
}

tir::Stmt InsertPhysicalPopsBeforeBlockingReserve(
    const tir::Stmt& body, const std::vector<CBConfig>& configs) {
  int max_cb_id = -1;
  for (const CBConfig& config : configs) {
    max_cb_id = std::max(max_cb_id, config.cb_id);
  }
  std::vector<int> capacity_pages(std::max(0, max_cb_id + 1), 0);
  for (const CBConfig& config : configs) {
    if (config.cb_id >= 0 && config.role != "input") {
      capacity_pages[config.cb_id] = std::max(capacity_pages[config.cb_id],
                                              std::max(1, config.num_pages));
    }
  }

  class Inserter final : public tir::StmtExprMutator {
   public:
    explicit Inserter(std::vector<int> capacity_pages)
        : capacity_pages_(std::move(capacity_pages)),
          front_pages_(capacity_pages_.size(), 0),
          reserved_pages_(capacity_pages_.size(), 0) {}

    using tir::StmtExprMutator::VisitStmt_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    tir::Stmt VisitStmt_(const tir::SeqStmtNode* op) final {
      Array<tir::Stmt> rewritten;
      rewritten.reserve(op->seq.size());
      for (const tir::Stmt& child : op->seq) {
        tir::Stmt new_child = VisitStmt(child);
        if (const auto* eval = new_child.as<tir::EvaluateNode>()) {
          if (const auto* call = eval->value.as<tir::CallNode>()) {
            MaybeInsertPopBeforeReserve(call, &rewritten);
          }
        }
        rewritten.push_back(new_child);
        if (const auto* eval = new_child.as<tir::EvaluateNode>()) {
          if (const auto* call = eval->value.as<tir::CallNode>()) {
            RecordQueueMutation(call);
          }
        }
      }
      return tir::SeqStmt::Flatten(rewritten);
    }

   private:
    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(capacity_pages_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    void MaybeInsertPopBeforeReserve(const tir::CallNode* op, Array<tir::Stmt>* stmts) {
      ICHECK(stmts != nullptr);
      if (!IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
        return;
      }
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (cb_id < 0 || pages <= 0 || capacity_pages_[cb_id] <= 0) {
        return;
      }
      const int over_capacity =
          front_pages_[cb_id] + reserved_pages_[cb_id] + pages - capacity_pages_[cb_id];
      if (over_capacity <= 0 || front_pages_[cb_id] <= 0) {
        return;
      }
      const int pop_pages = std::min(front_pages_[cb_id], over_capacity);
      stmts->push_back(MakeCBPopFrontStmt(cb_id, pop_pages));
      front_pages_[cb_id] -= pop_pages;
    }

    void RecordQueueMutation(const tir::CallNode* op) {
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (cb_id < 0 || pages <= 0) {
        return;
      }
      if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
        reserved_pages_[cb_id] += pages;
      } else if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
        reserved_pages_[cb_id] = std::max(0, reserved_pages_[cb_id] - pages);
        front_pages_[cb_id] += pages;
      } else if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
        front_pages_[cb_id] = std::max(0, front_pages_[cb_id] - pages);
      }
    }

    std::vector<int> capacity_pages_;
    std::vector<int> front_pages_;
    std::vector<int> reserved_pages_;
  };

  Inserter inserter(std::move(capacity_pages));
  return inserter.Rewrite(body);
}

tir::Stmt ClampIntermediatePopsToAvailableFront(
    const tir::Stmt& body, const std::vector<CBConfig>& configs) {
  int max_cb_id = -1;
  for (const CBConfig& config : configs) {
    max_cb_id = std::max(max_cb_id, config.cb_id);
  }
  std::vector<bool> is_local_intermediate(std::max(0, max_cb_id + 1), false);
  for (const CBConfig& config : configs) {
    if (config.cb_id >= 0 && config.role == "intermediate") {
      is_local_intermediate[config.cb_id] = true;
    }
  }

  class Rewriter final : public tir::StmtExprMutator {
   public:
    explicit Rewriter(std::vector<bool> is_local_intermediate)
        : is_local_intermediate_(std::move(is_local_intermediate)),
          front_pages_(is_local_intermediate_.size(), 0),
          locally_pushed_pages_(is_local_intermediate_.size(), 0) {}

    using tir::StmtExprMutator::VisitExpr_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    PrimExpr VisitExpr_(const tir::CallNode* op) final {
      PrimExpr expr = tir::StmtExprMutator::VisitExpr_(op);
      const auto* rewritten = expr.as<tir::CallNode>();
      if (rewritten == nullptr || !rewritten->op->IsInstance<OpNode>()) {
        return expr;
      }
      const int cb_id = StaticCBId(rewritten);
      const int pages = StaticPages(rewritten);
      if (!IsTracked(cb_id) || pages <= 0) {
        return expr;
      }
      if (IsBlackholeOp(rewritten, "tl.blackhole.cb_push_back")) {
        front_pages_[cb_id] += pages;
        locally_pushed_pages_[cb_id] += pages;
        return expr;
      }
      if (!IsBlackholeOp(rewritten, "tl.blackhole.cb_pop_front")) {
        return expr;
      }
      int pop_pages = pages;
      if (locally_pushed_pages_[cb_id] > 0 && front_pages_[cb_id] > 0 &&
          pages > front_pages_[cb_id]) {
        pop_pages = front_pages_[cb_id];
      }
      front_pages_[cb_id] = std::max(0, front_pages_[cb_id] - pop_pages);
      if (pop_pages == pages) {
        return expr;
      }
      Array<PrimExpr> args = rewritten->args;
      args.Set(1, tvm::IntImm(args[1].dtype(), pop_pages));
      return tir::Call(rewritten->dtype, rewritten->op, args,
                       rewritten->annotations, rewritten->span);
    }

   private:
    bool IsTracked(int cb_id) const {
      return cb_id >= 0 && cb_id < static_cast<int>(is_local_intermediate_.size()) &&
             is_local_intermediate_[cb_id];
    }

    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(is_local_intermediate_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    std::vector<bool> is_local_intermediate_;
    std::vector<int> front_pages_;
    std::vector<int> locally_pushed_pages_;
  };

  Rewriter rewriter(std::move(is_local_intermediate));
  return rewriter.Rewrite(body);
}

struct CBQueueEvent {
  enum class Kind { kWait, kPush, kPop, kReserve };
  Kind kind;
  int logical_cb_id;
  int physical_cb_id;
  int pages;
};

class PhysicalCBQueueView {
 public:
  PhysicalCBQueueView(const std::vector<CBConfig>& configs,
                      const std::unordered_map<int, int>& cb_id_by_requirement_index)
      : cb_id_by_requirement_index_(cb_id_by_requirement_index) {
    int max_cb_id = -1;
    for (const CBConfig& config : configs) {
      max_cb_id = std::max(max_cb_id, config.cb_id);
    }
    is_local_cb_.assign(std::max(0, max_cb_id + 1), false);
    for (const CBConfig& config : configs) {
      if (config.cb_id >= 0 && config.cb_id < static_cast<int>(is_local_cb_.size()) &&
          config.role != "input") {
        is_local_cb_[config.cb_id] = true;
      }
    }
  }

  bool IsLocalCB(int physical_cb_id) const {
    return physical_cb_id >= 0 && physical_cb_id < static_cast<int>(is_local_cb_.size()) &&
           is_local_cb_[physical_cb_id];
  }

  int LocalCBCount() const { return static_cast<int>(is_local_cb_.size()); }

  int ResolvePhysicalCBId(int cb_id) const {
    auto it = cb_id_by_requirement_index_.find(cb_id);
    return it != cb_id_by_requirement_index_.end() ? it->second : cb_id;
  }

  int StaticCBId(const tir::CallNode* op) const {
    if (op == nullptr || op->args.empty()) {
      return -1;
    }
    const auto* cb_id = op->args[0].as<IntImmNode>();
    if (cb_id == nullptr || cb_id->value < 0) {
      return -1;
    }
    return static_cast<int>(cb_id->value);
  }

  int StaticPages(const tir::CallNode* op) const {
    if (op == nullptr || op->args.size() < 2U) {
      return 0;
    }
    const auto* pages = op->args[1].as<IntImmNode>();
    return pages != nullptr ? static_cast<int>(pages->value) : 0;
  }

  bool GetEvent(const tir::CallNode* op, CBQueueEvent* event) const {
    if (event == nullptr) {
      return false;
    }
    const int cb_id = StaticCBId(op);
    const int pages = StaticPages(op);
    if (cb_id < 0 || pages <= 0) {
      return false;
    }
    const int physical_cb_id = ResolvePhysicalCBId(cb_id);
    if (!IsLocalCB(physical_cb_id)) {
      return false;
    }
    if (IsBlackholeOp(op, "tl.blackhole.cb_wait_front")) {
      *event = {CBQueueEvent::Kind::kWait, cb_id, physical_cb_id, pages};
      return true;
    }
    if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
      *event = {CBQueueEvent::Kind::kPush, cb_id, physical_cb_id, pages};
      return true;
    }
    if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
      *event = {CBQueueEvent::Kind::kPop, cb_id, physical_cb_id, pages};
      return true;
    }
    if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
      *event = {CBQueueEvent::Kind::kReserve, cb_id, physical_cb_id, pages};
      return true;
    }
    return false;
  }

 private:
  std::unordered_map<int, int> cb_id_by_requirement_index_;
  std::vector<bool> is_local_cb_;
};

void CollectCBQueueEvents(const tir::Stmt& stmt, const PhysicalCBQueueView& queue_view,
                          std::vector<CBQueueEvent>* events) {
  ICHECK(events != nullptr);
  if (!stmt.defined()) {
    return;
  }
  class Collector final : public tir::StmtExprVisitor {
   public:
    Collector(const PhysicalCBQueueView& queue_view,
              std::vector<CBQueueEvent>* events)
        : queue_view_(queue_view), events_(events) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& stmt) { VisitStmt(stmt); }

    void VisitExpr_(const tir::CallNode* op) final {
      CBQueueEvent event;
      if (queue_view_.GetEvent(op, &event)) {
        events_->push_back(event);
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

   private:
    const PhysicalCBQueueView& queue_view_;
    std::vector<CBQueueEvent>* events_;
  };

  Collector(queue_view, events).Collect(stmt);
}

const char* TTKernelQueueEventKindName(const tir::CallNode* op) {
  if (IsBlackholeOp(op, "tl.blackhole.cb_wait_front")) {
    return "wait_front";
  }
  if (IsBlackholeOp(op, "tl.blackhole.cb_push_back")) {
    return "push_back";
  }
  if (IsBlackholeOp(op, "tl.blackhole.cb_pop_front")) {
    return "pop_front";
  }
  if (IsBlackholeOp(op, "tl.blackhole.cb_reserve_back")) {
    return "reserve_back";
  }
  return nullptr;
}

bool TryReadNonNegativeIntImm(const PrimExpr& expr, int64_t* value) {
  ICHECK(value != nullptr);
  const auto* imm = expr.as<IntImmNode>();
  if (imm == nullptr || imm->value < 0) {
    return false;
  }
  *value = imm->value;
  return true;
}

Array<TTKernelQueueEvent> CollectTTKernelQueueEventsFromBody(
    const tir::Stmt& body) {
  Array<TTKernelQueueEvent> events;
  if (!body.defined()) {
    return events;
  }
  class Collector final : public tir::StmtExprVisitor {
   public:
    explicit Collector(Array<TTKernelQueueEvent>* events)
        : events_(events) {}

    using tir::StmtExprVisitor::VisitExpr_;

    void Collect(const tir::Stmt& stmt) { VisitStmt(stmt); }

    void VisitExpr_(const tir::CallNode* op) final {
      const char* kind = TTKernelQueueEventKindName(op);
      if (kind != nullptr) {
        ICHECK_GE(op->args.size(), 2U)
            << "PlanTTCBAlloc typed CB queue event record requires "
               "cb_id/pages args";
        int64_t cb_id = -1;
        int64_t pages = 0;
        ICHECK(TryReadNonNegativeIntImm(op->args[0], &cb_id))
            << "PlanTTCBAlloc typed CB queue event record requires static "
               "cb_id";
        ICHECK(TryReadNonNegativeIntImm(op->args[1], &pages) && pages > 0)
            << "PlanTTCBAlloc typed CB queue event record requires positive "
               "static page count";
        events_->push_back(TTKernelQueueEvent(String(kind), cb_id, pages));
      }
      tir::StmtExprVisitor::VisitExpr_(op);
    }

   private:
    Array<TTKernelQueueEvent>* events_;
  };

  Collector(&events).Collect(body);
  return events;
}

tir::Stmt RetainLocalCBFrontForFutureWaits(
    const tir::Stmt& body, const std::vector<CBConfig>& configs,
    const std::unordered_map<int, int>& cb_id_by_requirement_index = {}) {
  const PhysicalCBQueueView queue_view(configs, cb_id_by_requirement_index);
  if (queue_view.LocalCBCount() == 0) {
    return body;
  }

  std::vector<CBQueueEvent> events;
  CollectCBQueueEvents(body, queue_view, &events);
  if (events.empty()) {
    return body;
  }

  auto max_future_wait_before_next_producer =
      [&](size_t next_event, int physical_cb_id) {
        int required_front = 0;
        for (size_t i = next_event; i < events.size(); ++i) {
          const CBQueueEvent& event = events[i];
          if (event.physical_cb_id != physical_cb_id) {
            continue;
          }
          if (event.kind == CBQueueEvent::Kind::kReserve ||
              event.kind == CBQueueEvent::Kind::kPush) {
            return required_front;
          }
          if (event.kind == CBQueueEvent::Kind::kWait) {
            required_front = std::max(required_front, event.pages);
          }
        }
        return required_front;
      };

  std::vector<int> adjusted_pop_pages(events.size(), -1);
  std::vector<int> inserted_pop_before_event_pages(events.size(), 0);
  std::vector<bool> can_retain_front(queue_view.LocalCBCount(), true);
  for (const CBConfig& config : configs) {
    if (config.cb_id >= 0 &&
        config.cb_id < static_cast<int>(can_retain_front.size()) &&
        (config.role == "input" || config.role == "output")) {
      can_retain_front[config.cb_id] = false;
    }
  }
  std::vector<int> front_pages(queue_view.LocalCBCount(), 0);
  for (size_t i = 0; i < events.size(); ++i) {
    const CBQueueEvent& event = events[i];
    const int cb_id = event.physical_cb_id;
    if (cb_id < 0 || cb_id >= static_cast<int>(front_pages.size()) ||
        event.pages <= 0) {
      continue;
    }
    if (event.kind == CBQueueEvent::Kind::kReserve) {
      if (!can_retain_front[cb_id]) {
        continue;
      }
      const int required_front =
          max_future_wait_before_next_producer(i + 1, cb_id);
      if (front_pages[cb_id] > 0 && required_front <= 0) {
        inserted_pop_before_event_pages[i] = front_pages[cb_id];
        front_pages[cb_id] = 0;
      }
      continue;
    }
    if (event.kind == CBQueueEvent::Kind::kPush) {
      front_pages[cb_id] += event.pages;
      continue;
    }
    if (event.kind != CBQueueEvent::Kind::kPop) {
      continue;
    }
    if (!can_retain_front[cb_id]) {
      front_pages[cb_id] = std::max(0, front_pages[cb_id] - event.pages);
      continue;
    }
    const int required_front =
        max_future_wait_before_next_producer(i + 1, cb_id);
    int pop_pages = event.pages;
    if (required_front > 0) {
      const int retained_pages = std::min(front_pages[cb_id], required_front);
      pop_pages = std::max(0, front_pages[cb_id] - retained_pages);
      pop_pages = std::min(pop_pages, event.pages);
    }
    adjusted_pop_pages[i] = pop_pages;
    front_pages[cb_id] = std::max(0, front_pages[cb_id] - pop_pages);
  }

  class Rewriter final : public tir::StmtExprMutator {
   public:
    Rewriter(PhysicalCBQueueView queue_view,
             const std::vector<int>& adjusted_pop_pages,
             const std::vector<int>& inserted_pop_before_event_pages)
        : queue_view_(std::move(queue_view)),
          adjusted_pop_pages_(adjusted_pop_pages),
          inserted_pop_before_event_pages_(inserted_pop_before_event_pages) {}

    using tir::StmtExprMutator::VisitStmt_;

    tir::Stmt Rewrite(const tir::Stmt& body) { return VisitStmt(body); }

    tir::Stmt VisitStmt_(const tir::EvaluateNode* op) final {
      const auto* call = op->value.as<tir::CallNode>();
      CBQueueEvent event;
      if (!queue_view_.GetEvent(call, &event)) {
        return tir::StmtExprMutator::VisitStmt_(op);
      }
      const size_t event_index = event_index_++;
      if (event.kind == CBQueueEvent::Kind::kReserve &&
          event_index < inserted_pop_before_event_pages_.size() &&
          inserted_pop_before_event_pages_[event_index] > 0) {
        Array<tir::Stmt> stmts;
        stmts.push_back(MakeCBWaitFrontStmt(
            event.logical_cb_id, inserted_pop_before_event_pages_[event_index]));
        stmts.push_back(MakeCBPopFrontStmt(
            event.logical_cb_id, inserted_pop_before_event_pages_[event_index]));
        stmts.push_back(tir::StmtExprMutator::VisitStmt_(op));
        return tir::SeqStmt::Flatten(stmts);
      }
      if (event.kind != CBQueueEvent::Kind::kPop ||
          event_index >= adjusted_pop_pages_.size() ||
          adjusted_pop_pages_[event_index] < 0) {
        return tir::StmtExprMutator::VisitStmt_(op);
      }
      const int pop_pages = adjusted_pop_pages_[event_index];
      if (pop_pages == event.pages) {
        return tir::StmtExprMutator::VisitStmt_(op);
      }
      if (pop_pages <= 0) {
        return tir::Evaluate(tvm::IntImm(DataType::Int(32), 0));
      }
      Array<PrimExpr> args = call->args;
      args.Set(1, tvm::IntImm(args[1].dtype(), pop_pages));
      return tir::Evaluate(tir::Call(call->dtype, call->op, args,
                                     call->annotations, call->span));
    }

   private:
    PhysicalCBQueueView queue_view_;
    const std::vector<int>& adjusted_pop_pages_;
    const std::vector<int>& inserted_pop_before_event_pages_;
    size_t event_index_ = 0;
  };

  Rewriter rewriter(queue_view, adjusted_pop_pages,
                    inserted_pop_before_event_pages);
  return rewriter.Rewrite(body);
}

tir::Stmt RewriteRetainedStreamInputMatmulTileOffsets(
    const tir::Stmt& body, const std::vector<CBConfig>& configs) {
  int max_cb_id = -1;
  for (const CBConfig& config : configs) {
    max_cb_id = std::max(max_cb_id, config.cb_id);
  }
  const std::vector<int> max_outstanding_pages =
      CollectMaxPhysicalOutstandingPages(body, max_cb_id + 1);
  struct StreamInputInfo {
    bool tracked = false;
    int capacity_pages = 0;
    int event_pages = 1;
  };
  std::vector<StreamInputInfo> stream_inputs(std::max(0, max_cb_id + 1));
  for (const CBConfig& config : configs) {
    if (config.cb_id < 0 || config.cb_id >= static_cast<int>(stream_inputs.size()) ||
        config.role != "input") {
      continue;
    }
    const int event_pages = std::max(1, config.consume_pages_per_event);
    int capacity_pages = config.num_pages;
    if (config.cb_id >= 0 &&
        config.cb_id < static_cast<int>(max_outstanding_pages.size())) {
      capacity_pages =
          std::max(capacity_pages, max_outstanding_pages[config.cb_id]);
    }
    if (capacity_pages <= event_pages &&
        config.requirement_indices.size() <= 1U) {
      continue;
    }
    StreamInputInfo& info = stream_inputs[config.cb_id];
    info.tracked = true;
    info.capacity_pages = std::max(info.capacity_pages, capacity_pages);
    info.event_pages = std::max(info.event_pages, event_pages);
  }

  class Rewriter final : public tir::StmtExprMutator {
   public:
    explicit Rewriter(std::vector<StreamInputInfo> stream_inputs)
        : stream_inputs_(std::move(stream_inputs)),
          next_tile_offset_(stream_inputs_.size(), 0),
          active_event_base_(stream_inputs_.size(), -1) {}

    PrimExpr VisitExpr_(const tir::CallNode* op) final {
      PrimExpr expr = tir::StmtExprMutator::VisitExpr_(op);
      const auto* rewritten = expr.as<tir::CallNode>();
      if (rewritten == nullptr || !rewritten->op->IsInstance<OpNode>()) {
        return expr;
      }
      if (IsBlackholeOp(rewritten, "tl.blackhole.cb_wait_front")) {
        return VisitWait(rewritten, expr);
      }
      if (IsBlackholeOp(rewritten, "tl.blackhole.cb_pop_front")) {
        return VisitPop(rewritten, expr);
      }
      return VisitTileReadWithOffsets(rewritten, expr);
    }

   private:
    int StaticCBId(const tir::CallNode* op) const {
      if (op == nullptr || op->args.empty()) {
        return -1;
      }
      const auto* cb_id = op->args[0].as<IntImmNode>();
      if (cb_id == nullptr || cb_id->value < 0 ||
          cb_id->value >= static_cast<int64_t>(stream_inputs_.size())) {
        return -1;
      }
      return static_cast<int>(cb_id->value);
    }

    int StaticPages(const tir::CallNode* op) const {
      if (op == nullptr || op->args.size() < 2U) {
        return 0;
      }
      const auto* pages = op->args[1].as<IntImmNode>();
      return pages != nullptr ? static_cast<int>(pages->value) : 0;
    }

    bool IsTracked(int cb_id) const {
      return cb_id >= 0 && cb_id < static_cast<int>(stream_inputs_.size()) &&
             stream_inputs_[cb_id].tracked;
    }

    PrimExpr VisitWait(const tir::CallNode* op, const PrimExpr& expr) {
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (!IsTracked(cb_id) || pages <= 0) {
        return expr;
      }
      const int base = next_tile_offset_[cb_id];
      const bool wait_uses_absolute_depth =
          pages > std::max(1, stream_inputs_[cb_id].event_pages);
      active_event_base_[cb_id] =
          wait_uses_absolute_depth && base > 0
              ? std::max(base, pages - std::max(1, stream_inputs_[cb_id].event_pages))
              : base;
      next_tile_offset_[cb_id] =
          std::min(stream_inputs_[cb_id].capacity_pages,
                   wait_uses_absolute_depth ? std::max(base, pages) : base + pages);
      if (wait_uses_absolute_depth || base <= 0) {
        return expr;
      }
      Array<PrimExpr> args = op->args;
      args.Set(1, tvm::IntImm(args[1].dtype(), base + pages));
      return tir::Call(op->dtype, op->op, args, op->annotations, op->span);
    }

    PrimExpr VisitTileReadWithOffsets(const tir::CallNode* op, const PrimExpr& expr) {
      if (op == nullptr || !op->op->IsInstance<OpNode>()) {
        return expr;
      }
      auto is_any = [&](std::initializer_list<const char*> names) {
        for (const char* name : names) {
          if (IsBlackholeOp(op, name)) {
            return true;
          }
        }
        return false;
      };

      Array<PrimExpr> args = op->args;
      bool changed = false;
      auto apply_input_offset = [&](size_t cb_arg_pos, size_t tile_arg_pos) {
        if (args.size() <= std::max(cb_arg_pos, tile_arg_pos)) {
          return;
        }
        const auto* cb_id = args[cb_arg_pos].as<IntImmNode>();
        const auto* tile = args[tile_arg_pos].as<IntImmNode>();
        if (cb_id == nullptr || tile == nullptr || !IsTracked(static_cast<int>(cb_id->value))) {
          return;
        }
        const int cb = static_cast<int>(cb_id->value);
        const int base = active_event_base_[cb];
        if (base <= 0) {
          return;
        }
        args.Set(tile_arg_pos,
                 tvm::IntImm(args[tile_arg_pos].dtype(), tile->value + base));
        changed = true;
      };

      if (IsBlackholeOp(op, "tl.blackhole.copy_tile")) {
        apply_input_offset(/*cb_arg_pos=*/0, /*tile_arg_pos=*/1);
      } else if (is_any({"tl.blackhole.matmul_tiles", "tl.blackhole.add_tiles",
                         "tl.blackhole.sub_tiles", "tl.blackhole.mul_tiles",
                         "tl.blackhole.add_tiles_bcast_rows",
                         "tl.blackhole.add_tiles_bcast_cols",
                         "tl.blackhole.mul_tiles_bcast_rows",
                         "tl.blackhole.mul_tiles_bcast_cols"})) {
        apply_input_offset(/*cb_arg_pos=*/0, /*tile_arg_pos=*/2);
        apply_input_offset(/*cb_arg_pos=*/1, /*tile_arg_pos=*/3);
      } else if (IsBlackholeOp(op, "tl.blackhole.reduce_tile")) {
        apply_input_offset(/*cb_arg_pos=*/0, /*tile_arg_pos=*/2);
        apply_input_offset(/*cb_arg_pos=*/1, /*tile_arg_pos=*/3);
      } else if (is_any({"tl.blackhole.untilize_cb_front_tile",
                         "tl.blackhole.untilize_cb_front_tile_fragment"})) {
        apply_input_offset(/*cb_arg_pos=*/1, /*tile_arg_pos=*/2);
      }

      if (!changed) {
        return expr;
      }
      return tir::Call(op->dtype, op->op, args, op->annotations, op->span);
    }

    PrimExpr VisitPop(const tir::CallNode* op, const PrimExpr& expr) {
      const int cb_id = StaticCBId(op);
      const int pages = StaticPages(op);
      if (!IsTracked(cb_id) || pages <= 0) {
        return expr;
      }
      const int pop_pages = std::max(pages, next_tile_offset_[cb_id]);
      next_tile_offset_[cb_id] = std::max(0, next_tile_offset_[cb_id] - pop_pages);
      active_event_base_[cb_id] = -1;
      if (pop_pages == pages) {
        return expr;
      }
      Array<PrimExpr> args = op->args;
      args.Set(1, tvm::IntImm(args[1].dtype(), pop_pages));
      return tir::Call(op->dtype, op->op, args, op->annotations, op->span);
    }

    std::vector<StreamInputInfo> stream_inputs_;
    std::vector<int> next_tile_offset_;
    std::vector<int> active_event_base_;
  };

  Rewriter rewriter(std::move(stream_inputs));
  return rewriter(body);
}

}  // namespace

// Main entry point
PrimFunc PlanTTCBAlloc::Transform(const PrimFunc& func) {
  max_cb_count_ = kDefaultMaxCBs;
  max_l1_size_ = kDefaultMaxL1Size;
  if (auto maybe_target = func->GetAttr<Target>(tvm::attr::kTarget)) {
    const TTHardwareModel hardware_model =
        BuildBlackholeTTHardwareModel(maybe_target.value());
    if (hardware_model->max_cb_count > 0) {
      max_cb_count_ = static_cast<int>(hardware_model->max_cb_count);
    }
    if (hardware_model->worker_l1_size > 0) {
      max_l1_size_ = static_cast<int>(hardware_model->worker_l1_size);
    }
  }

  // Get CB requirements from function attributes
  std::vector<CBRequirement> requirements = GetCBRequirements(func);

  // If no CB requirements found, return original function
  if (requirements.empty()) {
    return func;
  }

  // Assign CB IDs to requirements. The staged TTProgram cb_plans carry
  // requirement schema, but physical CB reuse must be derived from the current
  // lowered IR. Earlier lifetime widening is intentionally not used here: it is
  // transitive and can turn a sequence of pairwise overlaps into a false global
  // overlap set, causing generated kernels to exceed TT-Metal's 0..31 CB API
  // range for larger flash-attn shapes.
  tir::Stmt body_with_state_front_pops =
      InsertStateFrontPopsBeforeReReserve(func->body, requirements);
  tir::Stmt body_with_loop_local_pops =
      InsertLoopLocalBackedgePops(body_with_state_front_pops, requirements);
  const std::vector<CBRequirementUseInfo> initial_use_info =
      CollectCBRequirementUseInfo(body_with_loop_local_pops,
                                  static_cast<int>(requirements.size()));
  const std::vector<int> auto_pop_pages =
      ComputeAutoPopPages(requirements, initial_use_info, body_with_loop_local_pops);
  tir::Stmt body_with_auto_pops =
      InsertAutoPopsAfterLastUse(body_with_loop_local_pops, initial_use_info, auto_pop_pages);
  const std::vector<CBRequirementUseInfo> use_info =
      CollectCBRequirementUseInfo(body_with_auto_pops, static_cast<int>(requirements.size()));
  ApplyIRUseIntervalsToRequirements(&requirements, use_info);
  const std::vector<CBRequirementEventInfo> event_info =
      CollectCBRequirementEventInfo(body_with_auto_pops, static_cast<int>(requirements.size()));
  ApplyIRCBEventsToRequirements(&requirements, event_info);
  if (auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram)) {
    ApplyExactCBLiveIntervalsToRequirements(staged_program.value(), &requirements);
  }
  const std::vector<bool> referenced_requirements = ReferencedRequirementMask(use_info);
  const std::vector<bool> exclusive_requirements =
      CollectExclusiveRequirementMask(body_with_auto_pops,
                                      static_cast<int>(requirements.size()));
  std::vector<CBConfig> configs =
      AssignCBIds(requirements, referenced_requirements, exclusive_requirements);

  // Create mutable copy and store CB configuration
  PrimFunc new_func = func;
  std::unordered_map<int, int> cb_id_by_requirement_index;
  for (const auto& config : configs) {
    for (int requirement_index : config.requirement_indices) {
      cb_id_by_requirement_index[requirement_index] = config.cb_id;
    }
  }
  planned_requirements_ = requirements;
  cb_id_by_requirement_index_ = cb_id_by_requirement_index;
  tir::Stmt physical_cb_body =
      RewriteCBIdsInIR(body_with_auto_pops, cb_id_by_requirement_index);
  int max_physical_cb_id = -1;
  for (const CBConfig& config : configs) {
    max_physical_cb_id = std::max(max_physical_cb_id, config.cb_id);
  }
  physical_cb_body =
      RewriteRetainedStreamInputMatmulTileOffsets(physical_cb_body, configs);
  physical_cb_body =
      RetainLocalCBFrontForFutureWaits(physical_cb_body, configs);
  ApplyMaxPhysicalOutstandingPages(physical_cb_body, max_physical_cb_id + 1,
                                   &configs);
  if (auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram)) {
    for (const TTKernel& kernel : staged_program.value()->kernels) {
      if (!kernel->body.defined()) {
        continue;
      }
      tir::Stmt physical_kernel_body =
          RewriteCBIdsInIR(kernel->body, cb_id_by_requirement_index);
      physical_kernel_body =
          RewriteRetainedStreamInputMatmulTileOffsets(physical_kernel_body, configs);
      physical_kernel_body =
          RetainLocalCBFrontForFutureWaits(physical_kernel_body, configs);
      ApplyMaxPhysicalOutstandingPages(physical_kernel_body,
                                       max_physical_cb_id + 1, &configs);
    }
  }

  // Validate the allocation
  ICHECK(Validate(configs))
      << "PlanTTCBAlloc: CB allocation exceeds Blackhole per-core constraints";

  StoreCBConfig(new_func, configs);
  physical_cb_body = InsertPhysicalPopsBeforeBlockingReserve(physical_cb_body, configs);
  physical_cb_body = RetainLocalCBFrontForFutureWaits(physical_cb_body, configs);
  physical_cb_body = ClampIntermediatePopsToAvailableFront(physical_cb_body, configs);
  new_func.CopyOnWrite()->body = physical_cb_body;
  cb_configs_ = configs;

  // Post-condition: verify no blackhole builtin retains an unrewritten requirement_index.
  // This catches cases where a new builtin with a cb_id parameter was not registered in
  // GetCBArgPositions.
  if (!cb_id_by_requirement_index.empty()) {
    const int max_requirement_index = static_cast<int>(requirements.size()) - 1;
    tir::PostOrderVisit(new_func->body, [&](const ObjectRef& node) {
      if (const auto* call = node.as<tir::CallNode>()) {
        if (!call->op->IsInstance<OpNode>()) return;
        const std::string op_name = Downcast<Op>(call->op)->name;
        if (op_name.rfind("tl.blackhole.", 0) != 0) return;
        if (HasNoCBArgs(op_name)) return;
        // Skip builtins that are known to have no cb_id args
        if (GetCBArgPositions(op_name).empty()) {
          // Scan all IntImm args: if any value falls in [0, max_requirement_index] and is
          // also a key in cb_id_by_requirement_index with a DIFFERENT final cb_id, we have
          // an unrewritten cb_id.
          for (size_t i = 0; i < call->args.size(); ++i) {
            if (const auto* imm = call->args[i].as<IntImmNode>()) {
              int val = static_cast<int>(imm->value);
              auto it = cb_id_by_requirement_index.find(val);
              if (it != cb_id_by_requirement_index.end() && it->second != val) {
                LOG(WARNING) << "PlanTTCBAlloc post-condition: builtin " << op_name
                             << " arg[" << i << "]=" << val
                             << " looks like an unrewritten requirement_index"
                             << " (expected cb_id=" << it->second << ")."
                             << " Did you forget to register this builtin in"
                             << " GetCBArgPositions?";
              }
            }
          }
        }
      }
    });
  }

  return new_func;
}

// Get staged CB requirements from TTProgram.
std::vector<CBRequirement> PlanTTCBAlloc::GetCBRequirements(
    const PrimFunc& func) {
  std::vector<CBRequirement> requirements;

  auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  if (staged_program) {
    for (const TTCBPlan& staged_cb_plan : staged_program.value()->cb_plans) {
      ICHECK_EQ(static_cast<int>(staged_cb_plan->cb_id), static_cast<int>(requirements.size()))
          << "PlanTTCBAlloc requires staged TTProgram cb_plans to preserve dense requirement "
             "slot ordering";
      CBRequirement cb_req;
      cb_req.name = static_cast<std::string>(staged_cb_plan->name);
      cb_req.page_size = static_cast<int>(staged_cb_plan->page_size_bytes);
      cb_req.num_pages = static_cast<int>(staged_cb_plan->num_pages);
      cb_req.data_format = static_cast<std::string>(staged_cb_plan->data_format);
      cb_req.initial_reserve_pages = static_cast<int>(staged_cb_plan->initial_reserve_pages);
      cb_req.flow_class = CBFlowClassFromString(static_cast<std::string>(staged_cb_plan->flow_class));
      cb_req.publish_pages_per_event = static_cast<int>(staged_cb_plan->publish_pages_per_event);
      cb_req.consume_pages_per_event = static_cast<int>(staged_cb_plan->consume_pages_per_event);
      cb_req.lifetime_begin = static_cast<int>(staged_cb_plan->lifetime_begin);
      cb_req.lifetime_end = static_cast<int>(staged_cb_plan->lifetime_end);

      const std::string role = static_cast<std::string>(staged_cb_plan->resource_class);
      if (role == "input") cb_req.type = CBType::kInput;
      else if (role == "output") cb_req.type = CBType::kOutput;
      else cb_req.type = CBType::kIntermediate;
      if (cb_req.lifetime_end < cb_req.lifetime_begin) {
        std::swap(cb_req.lifetime_begin, cb_req.lifetime_end);
      }

      requirements.push_back(cb_req);
    }
  }

  ICHECK(!requirements.empty())
      << "PlanTTCBAlloc requires staged TTProgram cb_plans; "
         "alloc_shared inference is no longer part of the formal planner contract";

  return requirements;
}

// Assign CB IDs to requirements
std::vector<CBConfig> PlanTTCBAlloc::AssignCBIds(
    const std::vector<CBRequirement>& requirements,
    const std::vector<bool>& referenced_requirements,
    const std::vector<bool>& exclusive_requirements) {
  std::vector<CBConfig> configs;
  ICHECK_EQ(referenced_requirements.size(), requirements.size());
  ICHECK_EQ(exclusive_requirements.size(), requirements.size());

  int next_input_id = kInputCBStart;
  int next_compute_cb_id = kOutputCBStart;
  const int reserved_input_ids = std::min(
      kInputCBEnd - kInputCBStart + 1,
      static_cast<int>(std::count_if(requirements.begin(), requirements.end(), [](const auto& req) {
        return req.type == CBType::kInput;
      })));
  int next_low_spill_id = kInputCBStart + reserved_input_ids;
  int next_spill_id = kOutputCBEnd + 1;

  std::vector<size_t> allocation_order;
  allocation_order.reserve(requirements.size());
  for (CBType type : {CBType::kInput, CBType::kOutput, CBType::kIntermediate}) {
    for (size_t req_index = 0; req_index < requirements.size(); ++req_index) {
      if (referenced_requirements[req_index] && requirements[req_index].type == type) {
        allocation_order.push_back(req_index);
      }
    }
  }
  for (CBType type : {CBType::kInput, CBType::kOutput, CBType::kIntermediate}) {
    for (size_t req_index = 0; req_index < requirements.size(); ++req_index) {
      if (!referenced_requirements[req_index] && requirements[req_index].type == type) {
        allocation_order.push_back(req_index);
      }
    }
  }

  auto allocate_compute_cb_id = [&]() {
    if (next_compute_cb_id <= kOutputCBEnd) {
      return next_compute_cb_id++;
    }
    if (next_low_spill_id <= kInputCBEnd) {
      return next_low_spill_id++;
    }
    return next_spill_id++;
  };

  for (size_t req_index : allocation_order) {
    const auto& req = requirements[req_index];
    bool reused = false;
    for (auto& config : configs) {
      if (!IsCompatibleForReuse(req, static_cast<int>(req_index), config,
                                exclusive_requirements)) {
        continue;
      }
      config.lifetime_end = std::max(config.lifetime_end, req.lifetime_end);
      config.requirement_indices.push_back(static_cast<int>(req_index));
      reused = true;
      break;
    }
    if (reused) {
      continue;
    }

    CBConfig config;
    config.name = req.name;
    config.role = RoleForType(req.type);
    config.page_size = req.page_size;
    config.num_pages = req.num_pages;
    config.initial_reserve_pages = req.initial_reserve_pages;
    config.flow_class = req.flow_class;
    config.publish_pages_per_event = req.publish_pages_per_event;
    config.consume_pages_per_event = req.consume_pages_per_event;
    config.data_format = req.data_format;
    config.lifetime_begin = req.lifetime_begin;
    config.lifetime_end = req.lifetime_end;
    config.requirement_indices.push_back(static_cast<int>(req_index));

    // Assign CB ID based on type
    switch (req.type) {
      case CBType::kInput:
        if (next_input_id < kInputCBStart + reserved_input_ids &&
            next_input_id <= kInputCBEnd) {
          config.cb_id = next_input_id++;
        } else {
          config.cb_id = next_spill_id++;
        }
        break;
      case CBType::kOutput:
        config.cb_id = allocate_compute_cb_id();
        break;
      case CBType::kIntermediate:
      default:
        config.cb_id = allocate_compute_cb_id();
        break;
    }

    // Calculate total size
    config.total_size = config.page_size * config.num_pages;

    configs.push_back(config);
  }

  for (CBConfig& config : configs) {
    if (IsDirectLocalStateConfig(config)) {
      config.num_pages =
          std::max(config.num_pages, static_cast<int>(config.requirement_indices.size()));
      config.total_size = config.page_size * config.num_pages;
    }
    if (config.role == "input" && config.flow_class == CBFlowClass::kStream &&
        config.requirement_indices.size() > 1U) {
      int queued_producer_pages = 0;
      for (int requirement_index : config.requirement_indices) {
        ICHECK_GE(requirement_index, 0);
        ICHECK_LT(requirement_index, static_cast<int>(requirements.size()));
        const CBRequirement& req = requirements[requirement_index];
        queued_producer_pages +=
            std::max(1, req.publish_pages_per_event > 0 ? req.publish_pages_per_event
                                                        : req.num_pages);
      }
      config.num_pages = std::max(config.num_pages, queued_producer_pages);
      config.total_size = config.page_size * config.num_pages;
    }
  }

  return configs;
}

// Validate CB allocation constraints
bool PlanTTCBAlloc::Validate(const std::vector<CBConfig>& configs) const {
  // Check CB count
  if (configs.size() > static_cast<size_t>(max_cb_count_)) {
    LOG(ERROR) << "PlanTTCBAlloc: Too many CBs requested: " << configs.size()
               << " (max " << max_cb_count_ << ")";
    return false;
  }

  // Check total L1 usage
  int total_l1 = 0;
  for (const auto& config : configs) {
    total_l1 += config.total_size;
    if (config.cb_id < 0 || config.cb_id >= max_cb_count_) {
      LOG(ERROR) << "PlanTTCBAlloc: Assigned CB id " << config.cb_id
                 << " outside TT-Metal architectural range [0, " << (max_cb_count_ - 1)
                 << "] for " << config.name;
      return false;
    }
  }

  if (total_l1 > max_l1_size_) {
    LOG(ERROR) << "PlanTTCBAlloc: Total L1 usage exceeds limit: " << total_l1
               << " bytes (max " << max_l1_size_ << " bytes)";
    return false;
  }

  LOG(INFO) << "PlanTTCBAlloc: Allocated " << configs.size() << " CBs, "
            << "total L1 usage: " << total_l1 << " bytes ("
            << (total_l1 * 100 / max_l1_size_) << "% of worker L1 budget)";

  return true;
}

// Store CB configuration in function attributes
void PlanTTCBAlloc::StoreCBConfig(PrimFunc& func, const std::vector<CBConfig>& configs) {
  (void)func;
  (void)configs;
}

tvm::tir::Stmt PlanTTCBAlloc::RewriteCBIdsInIR(
    const tvm::tir::Stmt& body,
    const std::unordered_map<int, int>& cb_id_by_requirement_index) const {
  class CBIdRewriter : public tir::StmtExprMutator {
   public:
    explicit CBIdRewriter(const std::unordered_map<int, int>& mapping) : mapping_(mapping) {}

    PrimExpr VisitExpr_(const tir::CallNode* op) final {
      PrimExpr expr = tir::StmtExprMutator::VisitExpr_(op);
      const auto* rewritten = expr.as<tir::CallNode>();
      ICHECK(rewritten);
      if (!rewritten->op->IsInstance<OpNode>()) {
        return expr;
      }
      const std::vector<int> positions = GetCBArgPositions(Downcast<Op>(rewritten->op)->name);
      if (positions.empty()) {
        return expr;
      }

      Array<PrimExpr> args = rewritten->args;
      bool changed = false;
      for (int pos : positions) {
        ICHECK_LT(pos, static_cast<int>(args.size()));
        const auto* imm = args[pos].as<IntImmNode>();
        ICHECK(imm) << "PlanTTCBAlloc expects constant requirement_index before IR rewrite";
        auto it = mapping_.find(static_cast<int>(imm->value));
        ICHECK(it != mapping_.end())
            << "Missing final cb_id for requirement_index=" << imm->value;
        if (it->second != imm->value) {
          args.Set(pos, tvm::IntImm(args[pos].dtype(), it->second));
          changed = true;
        }
      }
      if (!changed) {
        return expr;
      }
      return tir::Call(rewritten->dtype, rewritten->op, args, rewritten->annotations,
                       rewritten->span);
    }

   private:
    const std::unordered_map<int, int>& mapping_;
  };

  return CBIdRewriter(cb_id_by_requirement_index)(body);
}

tvm::tir::Stmt PlanTTCBAlloc::RewriteBodyWithFinalCBAllocation(
    const tvm::tir::Stmt& body) const {
  if (!body.defined() || cb_configs_.empty()) {
    return body;
  }
  tir::Stmt rewritten = body;
  rewritten = RewriteRetainedStreamInputMatmulTileOffsets(rewritten, cb_configs_);
  rewritten = RetainLocalCBFrontForFutureWaits(rewritten, cb_configs_,
                                               cb_id_by_requirement_index_);
  return rewritten;
}

tvm::ffi::Array<TTKernel> PlanTTCBAlloc::RewriteKernelBodies(
    const tvm::ffi::Array<TTKernel>& kernels) {
  tvm::ffi::Array<TTKernel> rewritten;
  for (const TTKernel& kernel : kernels) {
    tir::Stmt rewritten_body = RewriteBodyWithFinalCBAllocation(kernel->body);
    Array<TTKernelQueueEvent> queue_events = kernel->queue_events;
    if (rewritten_body.defined()) {
      queue_events = CollectTTKernelQueueEventsFromBody(rewritten_body);
      int max_physical_cb_id = -1;
      for (const CBConfig& config : cb_configs_) {
        max_physical_cb_id = std::max(max_physical_cb_id, config.cb_id);
      }
      std::vector<int> front_pages(std::max(0, max_physical_cb_id + 1), 0);
      std::vector<int> reserved_pages(std::max(0, max_physical_cb_id + 1), 0);
      std::vector<int> max_outstanding_pages(std::max(0, max_physical_cb_id + 1), 0);
      for (const TTKernelQueueEvent& event : queue_events) {
        int cb_id = static_cast<int>(event->cb_id);
        auto remap_it = cb_id_by_requirement_index_.find(cb_id);
        if (remap_it != cb_id_by_requirement_index_.end()) {
          cb_id = remap_it->second;
        }
        const int pages = static_cast<int>(event->pages);
        if (cb_id < 0 || cb_id >= static_cast<int>(max_outstanding_pages.size()) ||
            pages <= 0) {
          continue;
        }
        const std::string kind = event->kind;
        if (kind == "reserve_back") {
          reserved_pages[cb_id] += pages;
        } else if (kind == "push_back") {
          reserved_pages[cb_id] = std::max(0, reserved_pages[cb_id] - pages);
          front_pages[cb_id] += pages;
        } else if (kind == "pop_front") {
          front_pages[cb_id] = std::max(0, front_pages[cb_id] - pages);
        }
        max_outstanding_pages[cb_id] =
            std::max(max_outstanding_pages[cb_id],
                     reserved_pages[cb_id] + front_pages[cb_id]);
      }
      for (CBConfig& config : cb_configs_) {
        if (config.cb_id >= 0 &&
            config.cb_id < static_cast<int>(max_outstanding_pages.size()) &&
            max_outstanding_pages[config.cb_id] > config.num_pages) {
          config.num_pages = max_outstanding_pages[config.cb_id];
          config.total_size = config.page_size * config.num_pages;
        }
      }
    }
    rewritten.push_back(TTKernel(kernel->name, kernel->kind, kernel->core_type,
                                 kernel->abi_plan_index, kernel->launch_spec,
                                 kernel->compute_config,
                                 kernel->per_work_arg_specs,
                                 rewritten_body, queue_events));
  }
  return rewritten;
}

}  // namespace tl
}  // namespace tvm
