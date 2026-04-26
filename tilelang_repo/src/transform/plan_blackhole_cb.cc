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
 * - Validate constraints (CB count <= 64, total L1 <= 1.5MB)
 * - Assign CB IDs following TT-Metal convention: 0-15 input, 16-31 output
 * - Rewrite placeholder requirement indices in the IR to final CB IDs
 */

#include "plan_blackhole_cb.h"
#include "common/companion_base.h"
#include "common/tt_target_program.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
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

// Blackhole hardware constraints
constexpr int kMaxCBs = 64;
constexpr int kMaxL1Size = 1572864;  // 1.5MB = 1,572,864 bytes

// CB ID allocation ranges
constexpr int kInputCBStart = 0;
constexpr int kInputCBEnd = 15;
constexpr int kOutputCBStart = 16;
constexpr int kOutputCBEnd = 31;

namespace {

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

bool IsCompatibleForReuse(const CBRequirement& req, const CBConfig& config) {
  if ((req.initial_reserve_pages > 0 && req.flow_class == CBFlowClass::kState) ||
      (config.initial_reserve_pages > 0 && config.flow_class == CBFlowClass::kState)) {
    return false;
  }
  if (config.role != RoleForType(req.type)) {
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
      op_name == "tl.blackhole.write_tile_from_cb" ||
      op_name == "tl.blackhole.write_page_from_cb") {
    return {0};
  }
  if (op_name == "tl.blackhole.read_tile_to_cb" ||
      op_name == "tl.blackhole.read_page_to_cb") {
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
  if (op_name == "tl.blackhole.add_tiles_init" || op_name == "tl.blackhole.add_tiles") {
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
  if (op_name == "tl.blackhole.write_local_slice_to_cb") {
    return {1};  // args: (src_handle, cb_id, dst_offset, num_elements)
  }
  if (op_name == "tl.blackhole.write_local_fragment_tile_to_cb") {
    return {1};  // args: (src_handle, cb_id, dst_tile_index, src_offset)
  }
  if (op_name == "tl.blackhole.write_local_fragment_slice_to_tiled_cb") {
    return {1};  // args: (src_handle, cb_id, dst_offset, num_elements, row_width, src_offset)
  }
  if (op_name == "tl.blackhole.cast_fragment_slice_to_tiled_cb") {
    return {2};  // args: (dst_handle, src_handle, cb_id, dst_offset, src_offset, num_elements, row_width)
  }
  if (op_name == "tl.blackhole.pack_fill_fragment_to_tiled_cb") {
    return {1};  // args: (dst_handle, cb_id, dst_offset, num_elements, row_width, value)
  }
  if (op_name == "tl.blackhole.read_cb_front_tile_to_local") {
    return {1};  // args: (dst_handle, src_cb_id, src_tile_index, dst_offset, num_elements)
  }
  if (op_name == "tl.blackhole.read_cb_front_tile_to_local_fragment") {
    return {1};  // args: (dst_handle, src_cb_id, src_tile_index, dst_offset)
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

std::vector<bool> CollectReferencedCBRequirements(const tir::Stmt& body, int requirement_count) {
  std::vector<bool> referenced(std::max(0, requirement_count), false);
  tir::PostOrderVisit(body, [&](const ObjectRef& node) {
    const auto* call = node.as<tir::CallNode>();
    if (!call || !call->op->IsInstance<OpNode>()) {
      return;
    }
    const std::vector<int> positions = GetCBArgPositions(Downcast<Op>(call->op)->name);
    for (int pos : positions) {
      ICHECK_LT(pos, static_cast<int>(call->args.size()));
      if (const auto* imm = call->args[pos].as<IntImmNode>()) {
        const int requirement_index = static_cast<int>(imm->value);
        ICHECK_GE(requirement_index, 0)
            << "PlanTTCBAlloc expects non-negative requirement_index placeholders";
        ICHECK_LT(requirement_index, requirement_count)
            << "PlanTTCBAlloc found requirement_index=" << requirement_index
            << " outside staged requirement count=" << requirement_count;
        referenced[requirement_index] = true;
      }
    }
  });
  return referenced;
}

}  // namespace

// Main entry point
PrimFunc PlanTTCBAlloc::Transform(const PrimFunc& func) {
  // Get CB requirements from function attributes
  std::vector<CBRequirement> requirements = GetCBRequirements(func);

  // If no CB requirements found, return original function
  if (requirements.empty()) {
    return func;
  }

  // Assign CB IDs to requirements. Builtins can only pack from CB IDs in the
  // architectural compute range, so requirements referenced by executable IR get
  // first claim on low IDs before stale metadata-only requirements spill.
  const std::vector<bool> referenced_requirements =
      CollectReferencedCBRequirements(func->body, static_cast<int>(requirements.size()));
  std::vector<CBConfig> configs = AssignCBIds(requirements, referenced_requirements);

  // Validate the allocation
  ICHECK(Validate(configs))
      << "PlanTTCBAlloc: CB allocation exceeds Blackhole per-core constraints";

  // Create mutable copy and store CB configuration
  PrimFunc new_func = func;
  StoreCBConfig(new_func, configs);
  std::unordered_map<int, int> cb_id_by_requirement_index;
  for (const auto& config : configs) {
    for (int requirement_index : config.requirement_indices) {
      cb_id_by_requirement_index[requirement_index] = config.cb_id;
    }
  }
  new_func.CopyOnWrite()->body = RewriteCBIdsInIR(new_func->body, cb_id_by_requirement_index);
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
    const std::vector<bool>& referenced_requirements) {
  std::vector<CBConfig> configs;
  ICHECK_EQ(referenced_requirements.size(), requirements.size());

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
  for (size_t req_index = 0; req_index < requirements.size(); ++req_index) {
    if (referenced_requirements[req_index]) {
      allocation_order.push_back(req_index);
    }
  }
  for (size_t req_index = 0; req_index < requirements.size(); ++req_index) {
    if (!referenced_requirements[req_index]) {
      allocation_order.push_back(req_index);
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
      if (!IsCompatibleForReuse(req, config)) {
        continue;
      }
      config.lifetime_end = std::max(config.lifetime_end, req.lifetime_end);
      config.requirement_indices.push_back(static_cast<int>(req_index));
      config.requirement_names.push_back(req.name);
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
    config.requirement_names.push_back(req.name);

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

  return configs;
}

// Validate CB allocation constraints
bool PlanTTCBAlloc::Validate(const std::vector<CBConfig>& configs) const {
  // Check CB count
  if (configs.size() > kMaxCBs) {
    LOG(ERROR) << "PlanTTCBAlloc: Too many CBs requested: " << configs.size()
               << " (max " << kMaxCBs << ")";
    return false;
  }

  // Check total L1 usage
  int total_l1 = 0;
  for (const auto& config : configs) {
    total_l1 += config.total_size;
  }

  if (total_l1 > kMaxL1Size) {
    LOG(ERROR) << "PlanTTCBAlloc: Total L1 usage exceeds limit: " << total_l1
               << " bytes (max " << kMaxL1Size << " bytes = 1.5MB)";
    return false;
  }

  LOG(INFO) << "PlanTTCBAlloc: Allocated " << configs.size() << " CBs, "
            << "total L1 usage: " << total_l1 << " bytes ("
            << (total_l1 * 100 / kMaxL1Size) << "% of 1.5MB)";

  return true;
}

// Store CB configuration in function attributes
void PlanTTCBAlloc::StoreCBConfig(PrimFunc& func, const std::vector<CBConfig>& configs) {
  (void)func;
  (void)configs;
}

tvm::tir::Stmt PlanTTCBAlloc::RewriteCBIdsInIR(
    const tvm::tir::Stmt& body, const std::unordered_map<int, int>& cb_id_by_requirement_index) {
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

}  // namespace tl
}  // namespace tvm
