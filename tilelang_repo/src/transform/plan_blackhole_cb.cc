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
 * - Read CB requirements from function attributes (written by LowerBlackholeOps)
 * - Validate constraints (CB count <= 64, total L1 <= 1.5MB)
 * - Assign CB IDs following TT-Metal convention: 0-15 input, 16-31 output
 * - Store CB configuration in function attributes
 */

#include "plan_blackhole_cb.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <sstream>
#include <string>

namespace tvm {
namespace tl {

using tir::PrimFunc;
using tir::PrimFuncNode;
using tvm::DataType;
using tvm::Integer;
using tvm::String;
using tvm::Map;
using tvm::ObjectRef;
using tvm::DictAttrs;

// Blackhole hardware constraints
constexpr int kMaxCBs = 64;
constexpr int kMaxL1Size = 1572864;  // 1.5MB = 1,572,864 bytes

// CB ID allocation ranges
constexpr int kInputCBStart = 0;
constexpr int kInputCBEnd = 15;
constexpr int kOutputCBStart = 16;
constexpr int kOutputCBEnd = 31;

// Main entry point
PrimFunc PlanBlackholeCB::Transform(const PrimFunc& func) {
  // Get CB requirements from function attributes
  std::vector<CBRequirement> requirements = GetCBRequirements(func);

  // If no CB requirements found, return original function
  if (requirements.empty()) {
    return func;
  }

  // Assign CB IDs to requirements
  std::vector<CBConfig> configs = AssignCBIds(requirements);

  // Validate the allocation
  if (!Validate(configs)) {
    LOG(WARNING) << "PlanBlackholeCB: CB allocation validation failed";
    // Continue anyway - let runtime handle it
  }

  // Create mutable copy and store CB configuration
  PrimFunc new_func = func;
  StoreCBConfig(new_func, configs);

  return new_func;
}

// Get CB requirements from function attributes
std::vector<PlanBlackholeCB::CBRequirement> PlanBlackholeCB::GetCBRequirements(
    const PrimFunc& func) {
  std::vector<CBRequirement> requirements;

  // Read from function attributes (set by LowerBlackholeOps)
  // Attribute format: "blackhole.cb_requirements" = [cb0_info, cb1_info, ...]
  if (auto cb_req_attr = func->GetAttr<Array<ObjectRef>>("blackhole.cb_requirements")) {
    Array<ObjectRef> cb_reqs = cb_req_attr.value();
    for (const auto& req : cb_reqs) {
      if (const auto* map_node = req.as<MapNode>()) {
        CBRequirement req;
        Map<String, ObjectRef> req_map = Downcast<Map<String, ObjectRef>>(req);

        if (auto name = req_map.Get("name")) {
          req.name = Downcast<String>(name).c_str();
        }
        if (auto cb_type = req_map.Get("type")) {
          String type_str = Downcast<String>(cb_type);
          if (type_str == "input") req.type = CBType::kInput;
          else if (type_str == "output") req.type = CBType::kOutput;
          else req.type = CBType::kIntermediate;
        }
        if (auto page_size = req_map.Get("page_size")) {
          req.page_size = Downcast<Integer>(page_size)->value;
        }
        if (auto num_pages = req_map.Get("num_pages")) {
          req.num_pages = Downcast<Integer>(num_pages)->value;
        }
        if (auto data_format = req_map.Get("data_format")) {
          req.data_format = Downcast<String>(data_format).c_str();
        }

        requirements.push_back(req);
      }
    }
  }

  // If no explicit requirements, infer from alloc_shared buffers
  if (requirements.empty()) {
    requirements = InferFromAllocShared(func);
  }

  return requirements;
}

// Infer CB requirements from alloc_shared buffers
std::vector<PlanBlackholeCB::CBRequirement> PlanBlackholeCB::InferFromAllocShared(
    const PrimFunc& func) {
  std::vector<CBRequirement> requirements;

  // Analyze function body for Allocate nodes with "shared" scope
  class AllocSharedAnalyzer : public tir::StmtVisitor {
   public:
    std::vector<CBRequirement> requirements;

    void VisitStmt_(const tir::AllocateNode* op) final {
      // Check if this is a shared memory allocation
      auto* ptr_type = op->buffer_var->type_annotation.as<tir::PointerTypeNode>();
      if (ptr_type && ptr_type->storage_scope == "shared") {
        CBRequirement req;
        req.name = op->buffer_var->name_hint;
        req.type = CBType::kIntermediate;  // Default to intermediate

        // Calculate size from allocation extent
        int64_t total_elements = 1;
        for (const auto& extent : op->extents) {
          if (const auto* int_imm = extent.as<IntImmNode>()) {
            total_elements *= int_imm->value;
          }
        }

        // Estimate page size (assuming tile-based allocation)
        // For FP16: 2 bytes per element, typical tile 32x32 = 2048 bytes
        int dtype_bytes = op->dtype.bytes();
        req.page_size = static_cast<int>(total_elements * dtype_bytes);
        req.num_pages = 2;  // Default double buffering
        req.data_format = (dtype_bytes == 2) ? "Float16" : "Float32";

        requirements.push_back(req);
      }
      StmtVisitor::VisitStmt_(op);
    }
  };

  AllocSharedAnalyzer analyzer;
  analyzer(func->body);

  return analyzer.requirements;
}

// Assign CB IDs to requirements
std::vector<PlanBlackholeCB::CBConfig> PlanBlackholeCB::AssignCBIds(
    const std::vector<CBRequirement>& requirements) {
  std::vector<CBConfig> configs;

  int next_input_id = kInputCBStart;
  int next_output_id = kOutputCBStart;
  int next_intermediate_id = kOutputCBEnd + 1;  // Start after output range

  for (const auto& req : requirements) {
    CBConfig config;
    config.name = req.name;
    config.page_size = req.page_size;
    config.num_pages = req.num_pages;
    config.data_format = req.data_format;

    // Assign CB ID based on type
    switch (req.type) {
      case CBType::kInput:
        if (next_input_id <= kInputCBEnd) {
          config.cb_id = next_input_id++;
        } else {
          config.cb_id = next_intermediate_id++;
        }
        break;
      case CBType::kOutput:
        if (next_output_id <= kOutputCBEnd) {
          config.cb_id = next_output_id++;
        } else {
          config.cb_id = next_intermediate_id++;
        }
        break;
      case CBType::kIntermediate:
      default:
        config.cb_id = next_intermediate_id++;
        break;
    }

    // Calculate total size
    config.total_size = config.page_size * config.num_pages;

    configs.push_back(config);
  }

  return configs;
}

// Validate CB allocation constraints
bool PlanBlackholeCB::Validate(const std::vector<CBConfig>& configs) const {
  // Check CB count
  if (configs.size() > kMaxCBs) {
    LOG(ERROR) << "PlanBlackholeCB: Too many CBs requested: " << configs.size()
               << " (max " << kMaxCBs << ")";
    return false;
  }

  // Check total L1 usage
  int total_l1 = 0;
  for (const auto& config : configs) {
    total_l1 += config.total_size;
  }

  if (total_l1 > kMaxL1Size) {
    LOG(ERROR) << "PlanBlackholeCB: Total L1 usage exceeds limit: " << total_l1
               << " bytes (max " << kMaxL1Size << " bytes = 1.5MB)";
    return false;
  }

  LOG(INFO) << "PlanBlackholeCB: Allocated " << configs.size() << " CBs, "
            << "total L1 usage: " << total_l1 << " bytes ("
            << (total_l1 * 100 / kMaxL1Size) << "% of 1.5MB)";

  return true;
}

// Store CB configuration in function attributes
void PlanBlackholeCB::StoreCBConfig(PrimFunc& func, const std::vector<CBConfig>& configs) {
  // Get existing attributes
  Map<String, ObjectRef> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  // Build CB configs array
  Array<ObjectRef> cb_configs;
  int total_l1 = 0;

  for (const auto& config : configs) {
    Map<String, ObjectRef> cb_attr;
    cb_attr.Set("cb_id", Integer(config.cb_id));
    cb_attr.Set("page_size", Integer(config.page_size));
    cb_attr.Set("num_pages", Integer(config.num_pages));
    cb_attr.Set("data_format", String(config.data_format));
    cb_attr.Set("name", String(config.name));

    cb_configs.push_back(cb_attr);
    total_l1 += config.total_size;
  }

  // Store in function attributes
  attrs.Set("blackhole.cb_configs", cb_configs);
  attrs.Set("blackhole.total_l1_bytes", Integer(total_l1));
  attrs.Set("blackhole.num_cbs", Integer(static_cast<int>(configs.size())));

  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

// Modern TVM pass registration
tir::transform::Pass PlanBlackholeCBPass() {
  auto fpass = [](PrimFunc func, IRModule m, tir::transform::PassContext ctx) -> PrimFunc {
    return PlanBlackholeCB().Transform(func);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "tl.transform.PlanBlackholeCB", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.PlanBlackholeCB", PlanBlackholeCBPass);
}

}  // namespace tl
}  // namespace tvm
