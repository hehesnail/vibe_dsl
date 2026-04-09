/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file codegen_blackhole.h
 * \brief Generate TT-Metal code for Blackhole backend.
 */
#ifndef TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
#define TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "codegen_c_host.h"

namespace tvm {
namespace tl {

/*!
 * \brief CodeGen for Blackhole (TT-Metal) backend.
 *
 * Generates C++ code for Tenstorrent Blackhole architecture.
 * Supports BRISC, TRISC, and NCRISC core types.
 */
class CodeGenBlackhole : public CodeGenCHost {
 public:
  CodeGenBlackhole();

  // Note: Parent class Init is not virtual, so we just shadow it
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            std::string target_str,
            const std::unordered_set<std::string> &devices);

  // Get only the kernel code (without TVM headers)
  std::string GetKernelCode() const;

  void AddFunction(const tvm::GlobalVar &gvar,
                   const tvm::tir::PrimFunc &f) override;

  // Generate generic kernel_main entry point (IR-driven, no hardcoded paths)
  void GenerateGenericKernelMain(const tvm::tir::PrimFunc &f,
                                 const std::string &func_name);

  // Override visitor to handle TT-Metal builtin calls
  void VisitExpr_(const tvm::tir::CallNode *op,
                  std::ostream &os) override;

  // Override to handle FloorDiv/FloorMod (not implemented in base CodeGenC)
  void VisitExpr_(const tvm::tir::FloorDivNode *op,
                  std::ostream &os) override;
  void VisitExpr_(const tvm::tir::FloorModNode *op,
                  std::ostream &os) override;

  // Override AttrStmt visitor to handle CUDA-specific attributes
  void VisitStmt_(const tvm::tir::AttrStmtNode *op) override;

  // Override EvaluateNode to handle TT-Metal builtin calls as statements
  void VisitStmt_(const tvm::tir::EvaluateNode *op) override;

  // Skip C-array allocation for CB-backed shared buffers.
  void VisitStmt_(const tvm::tir::AllocateNode *op) override;

  // Override storage scope printing for Blackhole memory types
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) override;

  // Override thread index binding for Blackhole
  void BindThreadIndex(const tvm::tir::IterVar &iv) override;

  // Note: PrintFuncPrefix and PrintType are final in parent class,
  // so we don't override them here. Blackhole-specific handling
  // is done through visitor and AddFunction.

  // Blackhole core type enumeration
  enum class CoreType {
    kBRISC,   // Broadcast RISC - control core
    kTRISC,   // Tensix RISC - compute core
    kNCRISC,  // NOC RISC - data movement core
    kUnknown
  };

  // Set the core type for code generation
  void SetCoreType(CoreType core_type) { core_type_ = core_type; }

  // Get the core type
  CoreType GetCoreType() const { return core_type_; }

  // Note: Visitor methods (VisitExpr_, VisitStmt_) are marked 'final' in the
  // CodeGenCHost/CodeGenC parent classes and cannot be overridden further.
  // Blackhole-specific IR handling should be performed via preprocessing passes
  // that transform the TIR before it reaches the CodeGen stage.

 protected:
  // Print TT-Metal specific function attributes
  void PrintKernelAttributes();

  // Handle TT-Metal builtin calls (from VisitExpr_)
  bool HandleBlackholeBuiltin(const tvm::tir::CallNode *op, std::ostream &os);

  // Print CB (Circular Buffer) operations
  void PrintCBReserveBack(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBPushBack(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBWaitFront(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCBPopFront(const tvm::tir::CallNode *op, std::ostream &os);

  // Print NOC operations
  void PrintNOCAsyncRead(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintNOCAsyncWrite(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintNOCReadBarrier(std::ostream &os);
  void PrintNOCWriteBarrier(std::ostream &os);
  void PrintReadTileToCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintReadPageToCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintWriteTileFromCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintWritePageFromCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintGetSemaphore(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintRuntimeArgU32(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintSemaphoreWait(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintSemaphoreSet(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintSemaphoreIncRemote(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintSemaphoreSetRemote(const tvm::tir::CallNode *op, std::ostream &os);

  // Print compute operations
  void PrintMMInit(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintReconfigDataFormat(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMMInitShort(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMMInitShortWithDT(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMatmulTiles(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintTileRegsAcquire(std::ostream &os);
  void PrintTileRegsCommit(std::ostream &os);
  void PrintTileRegsWait(std::ostream &os);
  void PrintTileRegsRelease(std::ostream &os);
  void PrintPackTile(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintPackReconfigDataFormat(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCopyTileToDstInitShort(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCopyTileToDstInitShortWithDT(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCopyTileFromCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintFillFragment(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintAddFragment(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintAddFragmentFromCBFront(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintWriteLocalSliceToCB(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintScalarMax(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintCastFragmentSlice(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintReduceRow(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMulRowBcast(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintMulGroupedRowBcast(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintDivRowBcast(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintDivGroupedRowBcast(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintScalarFma(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintExp2RowBcastAffine(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintExp2GroupedRowBcastAffine(const tvm::tir::CallNode *op, std::ostream &os);
  void PrintScalarExp2Affine(const tvm::tir::CallNode *op, std::ostream &os);

  // Legacy: Print CB operations (old interface)
  void PrintCBDeclare(const std::string &name, tvm::DataType dtype,
                      int num_pages, int page_size);
  void PrintCBWaitFront(const std::string &name, int num_tiles);
  void PrintCBPopFront(const std::string &name, int num_tiles);
  void PrintCBReserveBack(const std::string &name, int num_tiles);
  void PrintCBPushBack(const std::string &name, int num_tiles);

  // Legacy: Print NOC operations (old interface)
  void PrintNOCRead(const std::string &src_addr, const std::string &dst_addr,
                    int size);
  void PrintNOCWrite(const std::string &src_addr, const std::string &dst_addr,
                     int size);
  void PrintNOCWait();

  void EmitRuntimeArgLoads(const tvm::tir::PrimFunc &f);
  void LoadCorePlan(const tvm::tir::PrimFunc &f);
  std::string GetRuntimeArgVarByKind(const std::string &kind) const;
  std::string GetRuntimeArgVarForBuffer(const tvm::PrimExpr &buffer_expr,
                                        const char* preferred_kind = nullptr) const;
  int ResolveCBId(const tvm::PrimExpr &expr) const;
  void PrintResolvedCBId(const tvm::PrimExpr &expr, std::ostream &os) const;
  int GetCBPageSize(int cb_id) const;
  int GetCBNumPages(int cb_id) const;
  std::string GetCBHeadVar(int cb_id) const;
  std::string GetCBTailVar(int cb_id) const;
  void RegisterActiveCBWritePtrBinding(int cb_id, const std::string& var_name,
                                       const std::string& type_name);
  void UnregisterActiveCBWritePtrBinding(int cb_id, const std::string& var_name);
  void EmitActiveCBWritePtrRefreshes(int cb_id);
  void MaybeEmitMathWaypoint(std::ostream& os, const char* code);
  void MaybeEmitPackWaypoint(std::ostream& os, const char* code);
  void MaybeEmitUnpackWaypoint(std::ostream& os, const char* code);
  std::string GetCBRequirementName(int cb_id) const;

 private:
  struct ActiveCBWritePtrBinding {
    std::string var_name;
    std::string type_name;
  };

  struct PerWorkArgSpecBinding {
    std::string arg_identity;
    std::string value_kind;
    uint32_t constant_value{0};
  };

  // Per-instance header emission flag (replaces static variable)
  bool headers_emitted_{false};

  // Current core type being generated (from IR attrs, not function name)
  CoreType core_type_{CoreType::kBRISC};  // Default to BRISC for TT-Sim compatibility

  // TT-Metal specific state
  bool need_tt_metal_h_{false};
  bool need_dataflow_api_h_{false};
  bool need_compute_api_h_{false};
  bool emit_debug_waypoints_{false};

  // Whether to emit kernel entry point wrapper
  bool emit_kernel_wrapper_{true};

  // Track declared CBs
  std::unordered_set<std::string> declared_cbs_;
  std::unordered_map<const tvm::tir::VarNode *, std::string> buffer_runtime_arg_map_;
  std::unordered_map<std::string, std::string> buffer_runtime_arg_map_by_name_;
  std::unordered_map<std::string, std::string> runtime_arg_vars_by_kind_;
  std::unordered_map<std::string, std::string> runtime_arg_vars_by_name_;
  std::unordered_map<std::string, PerWorkArgSpecBinding> per_work_arg_bindings_by_kind_;
  std::unordered_map<int, int> cb_page_size_by_id_;
  std::unordered_map<int, int> cb_num_pages_by_id_;
  std::unordered_map<std::string, int> cb_id_by_requirement_name_;
  std::unordered_map<int, std::string> cb_requirement_name_by_id_;
  std::unordered_map<std::string, int> cb_num_pages_by_requirement_name_;
  std::unordered_map<std::string, int> cb_initial_reserve_pages_by_requirement_name_;
  std::unordered_map<int, std::vector<ActiveCBWritePtrBinding>> active_cb_write_ptr_bindings_;
  int logical_grid_x_{1};
  int logical_grid_y_{1};
  std::string linearization_{"row_major"};

  // Default L1 memory alignment
  static constexpr int kL1Alignment = 16;
};

} // namespace tl
} // namespace tvm

#endif // TL_TARGET_SOURCE_CODEGEN_BLACKHOLE_H_
