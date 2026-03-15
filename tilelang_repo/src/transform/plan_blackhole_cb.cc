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
 */

#include "plan_blackhole_cb.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

// Main entry point
PrimFunc PlanBlackholeCB::Transform(const PrimFunc& func) {
  // Analyze alloc_shared statements and create CB configs
  cb_configs_ = AnalyzeAllocShared(func);

  // Assign CB IDs
  AssignCBIds(cb_configs_);

  // Create a mutable copy of the function
  PrimFunc new_func = func;

  // Store CB config in function attributes
  StoreCBConfig(new_func, cb_configs_);

  return new_func;
}

// Visitor implementation for collecting alloc_shared statements
void PlanBlackholeCB::AllocSharedCollector::VisitStmt_(const AllocateNode* op) {
  // Check if this is a shared memory allocation
  auto storage_scope = op->buffer_var->type_annotation.as<PointerTypeNode>();
  if (storage_scope && storage_scope->storage_scope == "shared") {
    // Create a buffer to represent this allocation
    Buffer buffer = Buffer(op->buffer_var, op->dtype, op->extents,
                          {}, op->buffer_var->name_hint, 0, 0, BufferType::kDefault);
    shared_buffers.push_back(buffer);
  }
  StmtExprVisitor::VisitStmt_(op);
}

// Analyze alloc_shared statements and create CB configs
std::vector<CBConfig> PlanBlackholeCB::AnalyzeAllocShared(const PrimFunc& func) {
  std::vector<CBConfig> configs;

  // Collect all shared buffer allocations
  AllocSharedCollector collector;
  collector(func->body);

  // Create CB config for each shared buffer
  for (size_t i = 0; i < collector.shared_buffers.size(); ++i) {
    const Buffer& buffer = collector.shared_buffers[i];

    CBConfig config;
    config.cb_id = static_cast<int>(i);
    config.dtype = buffer->dtype;

    // Calculate page size from buffer dimensions
    // Assume 2D tile: [rows, cols]
    int rows = 1, cols = 1;
    if (buffer->shape.size() >= 2) {
      if (auto* rows_int = buffer->shape[0].as<IntImmNode>()) {
        rows = static_cast<int>(rows_int->value);
      }
      if (auto* cols_int = buffer->shape[1].as<IntImmNode>()) {
        cols = static_cast<int>(cols_int->value);
      }
    } else if (buffer->shape.size() == 1) {
      if (auto* cols_int = buffer->shape[0].as<IntImmNode>()) {
        cols = static_cast<int>(cols_int->value);
      }
    }

    // Page size = rows * cols * dtype_size
    int dtype_size = buffer->dtype.bytes();
    config.page_size = CalculatePageSize(rows, cols, dtype_size);

    // Default: double buffering (2 pages)
    config.num_pages = 2;
    config.total_size = config.page_size * config.num_pages;

    configs.push_back(config);
  }

  return configs;
}

// Calculate page size for a tile
int PlanBlackholeCB::CalculatePageSize(int rows, int cols, int dtype_size) {
  return rows * cols * dtype_size;
}

// Assign CB IDs to configurations
void PlanBlackholeCB::AssignCBIds(std::vector<CBConfig>& configs) {
  // IDs are already assigned sequentially during creation
  // This function can be used for renumbering or optimization
  for (size_t i = 0; i < configs.size(); ++i) {
    configs[i].cb_id = static_cast<int>(i);
  }
}

// Validate CB allocation constraints
bool PlanBlackholeCB::Validate() const {
  // Check CB count
  if (cb_configs_.size() > kMaxCBCount) {
    return false;
  }

  // Check total size
  int total_size = 0;
  for (const auto& config : cb_configs_) {
    total_size += config.total_size;
  }

  if (total_size > kMaxCBSize) {
    return false;
  }

  return true;
}

// Store CB config in function attributes
void PlanBlackholeCB::StoreCBConfig(PrimFunc& func,
                                    const std::vector<CBConfig>& configs) {
  Map<String, ObjectRef> attrs = func->attrs;

  // Create array of CB configs
  Array<Map<String, ObjectRef>> cb_config_array;
  for (const auto& config : configs) {
    Map<String, ObjectRef> cb_map;
    cb_map.Set("cb_id", Integer(config.cb_id));
    cb_map.Set("num_pages", Integer(config.num_pages));
    cb_map.Set("page_size", Integer(config.page_size));
    cb_map.Set("total_size", Integer(config.total_size));
    cb_map.Set("dtype", String(config.dtype.name()));
    cb_config_array.push_back(cb_map);
  }

  attrs.Set("tl_blackhole_cb_configs", cb_config_array);

  // Also store total size and count for quick access
  int total_size = 0;
  for (const auto& config : configs) {
    total_size += config.total_size;
  }
  attrs.Set("tl_blackhole_cb_total_size", Integer(total_size));
  attrs.Set("tl_blackhole_cb_count", Integer(static_cast<int>(configs.size())));

  func.CopyOnWrite()->attrs = attrs;
}

// Pass registration
class PlanBlackholeCBPassNode : public transform::PassNode {
 public:
  // Entry point
  IRModule operator()(IRModule mod, const transform::PassContext& pass_ctx) const final {
    for (const auto& [gvar, func] : mod->functions) {
      if (auto* prim_func = func.as<PrimFuncNode>()) {
        PrimFunc updated_func = PlanBlackholeCB().Transform(GetRef<PrimFunc>(prim_func));
        mod.CopyOnWrite()->Add(gvar, updated_func);
      }
    }
    return mod;
  }

  TVM_OBJECT_ENABLE(PlanBlackholeCBPassNode, transform::PassNode);
};

tvm::tir::transform::Pass PlanBlackholeCBPass() {
  return tvm::make_object<PlanBlackholeCBPassNode>();
}

TVM_REGISTER_GLOBAL("tl.transform.PlanBlackholeCB")
    .set_body_typed(PlanBlackholeCBPass);

}  // namespace tl
}  // namespace tvm
