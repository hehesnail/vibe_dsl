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
#include <unordered_map>

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

bool IsCompatibleForReuse(const CBRequirement& req, const CBConfig& config) {
  if (config.role != RoleForType(req.type)) {
    return false;
  }
  if (config.page_size != req.page_size) {
    return false;
  }
  if (config.num_pages != req.num_pages) {
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
  if (op_name == "tl.blackhole.matmul_tiles") {
    return {0, 1};
  }
  if (op_name == "tl.blackhole.pack_tile") {
    return {1};
  }
  if (op_name == "tl.blackhole.write_local_slice_to_cb") {
    return {1};  // args: (src_handle, cb_id, dst_offset, num_elements)
  }
  return {};
}

}  // namespace

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
  ICHECK(Validate(configs))
      << "PlanBlackholeCB: CB allocation exceeds Blackhole per-core constraints";

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
                LOG(WARNING) << "PlanBlackholeCB post-condition: builtin " << op_name
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

// Get CB requirements from function attributes
std::vector<CBRequirement> PlanBlackholeCB::GetCBRequirements(
    const PrimFunc& func) {
  std::vector<CBRequirement> requirements;

  // Read from function attributes (set by LowerBlackholeOps)
  // Attribute format: "blackhole.cb_requirements" = [cb0_info, cb1_info, ...]
  if (auto cb_req_attr = func->GetAttr<Array<Any>>("blackhole.cb_requirements")) {
    Array<Any> cb_reqs = cb_req_attr.value();
    int req_index = 0;
    for (const auto& req : cb_reqs) {
      // Try to downcast to Map - if it fails, req_map will be empty
      Map<String, Any> req_map = req.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (!req_map.empty()) {
        CBRequirement cb_req;
        cb_req.lifetime_begin = req_index;
        cb_req.lifetime_end = req_index;

        if (auto name = req_map.Get("name")) {
          cb_req.name = Downcast<String>(name.value()).c_str();
        }
        if (auto cb_type = req_map.Get("type")) {
          String type_str = Downcast<String>(cb_type.value());
          if (type_str == "input") cb_req.type = CBType::kInput;
          else if (type_str == "output") cb_req.type = CBType::kOutput;
          else cb_req.type = CBType::kIntermediate;
        }
        if (auto page_size = req_map.Get("page_size")) {
          cb_req.page_size = Downcast<Integer>(page_size.value())->value;
        }
        if (auto num_pages = req_map.Get("num_pages")) {
          cb_req.num_pages = Downcast<Integer>(num_pages.value())->value;
        }
        if (auto data_format = req_map.Get("data_format")) {
          cb_req.data_format = Downcast<String>(data_format.value()).c_str();
        }
        if (auto lifetime_begin = req_map.Get("lifetime_begin")) {
          cb_req.lifetime_begin = Downcast<Integer>(lifetime_begin.value())->value;
        }
        if (auto lifetime_end = req_map.Get("lifetime_end")) {
          cb_req.lifetime_end = Downcast<Integer>(lifetime_end.value())->value;
        }
        if (cb_req.lifetime_end < cb_req.lifetime_begin) {
          std::swap(cb_req.lifetime_begin, cb_req.lifetime_end);
        }

        requirements.push_back(cb_req);
      }
      ++req_index;
    }
  }

  ICHECK(!requirements.empty())
      << "PlanBlackholeCB requires explicit blackhole.cb_requirements; "
         "alloc_shared inference is no longer part of the formal planner contract";

  return requirements;
}

// Assign CB IDs to requirements
std::vector<CBConfig> PlanBlackholeCB::AssignCBIds(
    const std::vector<CBRequirement>& requirements) {
  std::vector<CBConfig> configs;

  int next_input_id = kInputCBStart;
  int next_compute_cb_id = kOutputCBStart;
  int next_spill_id = kOutputCBEnd + 1;

  for (size_t req_index = 0; req_index < requirements.size(); ++req_index) {
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
    config.data_format = req.data_format;
    config.lifetime_begin = req.lifetime_begin;
    config.lifetime_end = req.lifetime_end;
    config.requirement_indices.push_back(static_cast<int>(req_index));
    config.requirement_names.push_back(req.name);

    // Assign CB ID based on type
    switch (req.type) {
      case CBType::kInput:
        if (next_input_id <= kInputCBEnd) {
          config.cb_id = next_input_id++;
        } else {
          config.cb_id = next_spill_id++;
        }
        break;
      case CBType::kOutput:
        if (next_compute_cb_id <= kOutputCBEnd) {
          config.cb_id = next_compute_cb_id++;
        } else {
          config.cb_id = next_spill_id++;
        }
        break;
      case CBType::kIntermediate:
      default:
        if (next_compute_cb_id <= kOutputCBEnd) {
          config.cb_id = next_compute_cb_id++;
        } else {
          config.cb_id = next_spill_id++;
        }
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
  Map<String, Any> attrs;
  if (func->attrs.defined()) {
    attrs = func->attrs->dict;
  }

  // Build CB configs array
  Array<Any> cb_configs;
  Array<Any> cb_bindings;
  int total_l1 = 0;

  for (size_t config_index = 0; config_index < configs.size(); ++config_index) {
    const auto& config = configs[config_index];
    Map<String, Any> cb_attr;
    cb_attr.Set("cb_id", Integer(config.cb_id));
    cb_attr.Set("page_size", Integer(config.page_size));
    cb_attr.Set("num_pages", Integer(config.num_pages));
    cb_attr.Set("total_size_bytes", Integer(config.total_size));
    cb_attr.Set("data_format", String(config.data_format));
    cb_attr.Set("name", String(config.name));
    cb_attr.Set("role", String(config.role));
    cb_attr.Set("lifetime_begin", Integer(config.lifetime_begin));
    cb_attr.Set("lifetime_end", Integer(config.lifetime_end));
    Array<Any> requirement_names;
    for (const auto& req_name : config.requirement_names) {
      requirement_names.push_back(String(req_name));
    }
    cb_attr.Set("requirement_names", requirement_names);

    cb_configs.push_back(cb_attr);
    total_l1 += config.total_size;

    for (size_t binding_index = 0; binding_index < config.requirement_names.size(); ++binding_index) {
      Map<String, Any> binding_attr;
      binding_attr.Set("requirement_index", Integer(config.requirement_indices[binding_index]));
      binding_attr.Set("requirement_name", String(config.requirement_names[binding_index]));
      binding_attr.Set("cb_id", Integer(config.cb_id));
      binding_attr.Set("cb_config_index", Integer(static_cast<int>(config_index)));
      binding_attr.Set("memory_object_name", String(config.name));
      cb_bindings.push_back(binding_attr);
    }
  }

  // Store in function attributes
  attrs.Set("blackhole.cb_configs", cb_configs);
  attrs.Set("blackhole.cb_bindings", cb_bindings);
  attrs.Set("blackhole.total_l1_bytes", Integer(total_l1));
  attrs.Set("blackhole.num_cbs", Integer(static_cast<int>(configs.size())));

  func.CopyOnWrite()->attrs = DictAttrs(attrs);
}

tvm::tir::Stmt PlanBlackholeCB::RewriteCBIdsInIR(
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
        ICHECK(imm) << "PlanBlackholeCB expects constant requirement_index before IR rewrite";
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
