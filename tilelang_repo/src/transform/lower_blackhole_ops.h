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
 * \file lower_blackhole_ops.h
 * \brief Lower TileLang high-level ops to TT-Metal builtins for Blackhole backend
 */

#ifndef TVM_TL_LOWER_BLACKHOLE_OPS_H_
#define TVM_TL_LOWER_BLACKHOLE_OPS_H_

#include "blackhole_cb_common.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief Copy direction classification
 */
enum class CopyDirection {
  kDramToCB,     // DRAM -> CB (Reader)
  kCBToDram,     // CB -> DRAM (Writer)
  kCBToCB,       // CB -> CB (local copy)
  kLocalToCB,    // local/accumulator -> CB (fragment/local staging write)
  kDramToDram,   // DRAM -> DRAM (Stage 2 copy pass integration path)
  kUnknown
};

/*!
 * \brief LowerBlackholeOps Pass
 *
 * This pass transforms TileLang high-level operations:
 * - T.copy(A, B) -> CB reserve + NOC read/write + push/pop sequence
 * - T.gemm(A, B, C) -> MM init + tile_regs_acquire + matmul_tiles + pack_tile
 * - T.clear(C) -> tile_regs_acquire (zero DST)
 *
 * Also records CB requirements in function attributes for PlanBlackholeCB.
 */
class LowerBlackholeOps : public tvm::tir::StmtExprMutator {
 public:
  LowerBlackholeOps();

  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

 private:
  enum class BufferFlowEventKind {
    kWrite,
    kComputeConsume,
    kTransportConsume,
    kReference,
  };

  struct BufferFlowEvent {
    int order_index = -1;
    BufferFlowEventKind kind = BufferFlowEventKind::kReference;
  };

  struct BufferFlowContract {
    CBFlowClass flow_class = CBFlowClass::kState;
    int publish_pages_per_event = 0;
    int consume_pages_per_event = 0;
    std::vector<BufferFlowEvent> events;
  };

  struct AccessorDescriptor {
    std::string segment_kind;
    tvm::tir::Buffer buffer;
    std::string buffer_name;
    int compile_time_arg_offset = 0;
    int compile_time_arg_count = 2;
    int common_runtime_arg_offset = 0;
    int common_runtime_arg_count = 0;
    int args_config_bits = 1;
    int transport_page_size_bytes = 0;
    std::string layout = "interleaved";
    std::string memory_space = "dram";
    std::vector<int64_t> host_axis_order;
    bool transpose_2d = false;
  };

  struct NestedCopyMatch {
    const tvm::tir::BufferStoreNode* store = nullptr;
    std::vector<tvm::tir::Var> loop_vars;
    CopyDirection direction = CopyDirection::kUnknown;
  };

  struct RowReductionMatch {
    tvm::tir::Buffer src;
    tvm::tir::Buffer dst;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr row_width;
    std::string kind;
    bool grouped = false;
    bool clear = true;
  };

  struct RowBroadcastMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer scalar;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr row_width;
    bool grouped = false;
    std::string kind;
  };

  struct ScalarFmaMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer lhs;
    tvm::tir::Buffer rhs;
    tvm::tir::Buffer add;
    tvm::PrimExpr num_elements;
  };

  struct Exp2RowBroadcastAffineMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer scalar;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr row_width;
    bool grouped = false;
    tvm::PrimExpr dst_scale;
    tvm::PrimExpr scalar_scale;
  };

  struct ScalarExp2AffineMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer lhs;
    tvm::tir::Buffer rhs;
    tvm::PrimExpr lhs_scale;
    tvm::PrimExpr rhs_scale;
  };

  struct FragmentFillMatch {
    tvm::tir::Buffer dst;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr value;
  };

  struct ScalarMaxMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::PrimExpr num_elements;
  };

  struct FragmentCastMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::PrimExpr dst_offset;
    tvm::PrimExpr src_offset;
    tvm::PrimExpr num_elements;
  };

  struct ScalarFragmentCopyMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::PrimExpr num_elements;
  };

  struct LocalToCBSliceMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::PrimExpr dst_offset_elements;
    tvm::PrimExpr num_elements;
    tvm::tir::Stmt lowered_loop_body;
    bool wrap_src_allocation = false;
  };

  /*! \brief CB configuration from function attributes */
  struct CBConfig {
    int in0_id = 0;
    int in1_id = 1;
    int out_id = 16;
    int num_k_tiles = 1;
  };

  /*! \brief Get CB configuration from function attributes */
  CBConfig GetCBConfig() const;

  /*! \brief Register a CB requirement for a buffer and return its requirement_index.
   *
   * The returned index is the position of this requirement in cb_requirements_.
   * It is used as the cb_id placeholder in the IR body.  PlanBlackholeCB will
   * replace all requirement_index placeholders with the final hardware cb_id.
   */
  int AllocateRequirementIndex(const tvm::tir::Buffer& buffer, CBType type);

  /*! \brief Store CB requirements in function attributes */
  void StoreCBRequirements(tvm::tir::PrimFunc& func);

  /*! \brief Store runtime argument schema inferred during lowering */
  void StoreRuntimeArgs(tvm::tir::PrimFunc& func);

  /*! \brief Store minimal segment/kernel plan inferred during lowering */
  void StoreSegmentPlan(tvm::tir::PrimFunc& func);

  /*! \brief Store minimal GEMM contract metadata for runtime layout handling */
  void StoreGemmContract(tvm::tir::PrimFunc& func);

  /*! \brief Store per-segment accessor descriptors for dataflow kernels */
  void StoreAccessorDescriptors(tvm::tir::PrimFunc& func);

  /*! \brief Encode current lowering-time accessor descriptors as TIR attrs */
  tvm::ffi::Array<tvm::ffi::Any> EncodeAccessorDescriptors(const std::string& segment_kind) const;

  /*! \brief Encode empty or richer common-runtime args for a segment */
  tvm::ffi::Array<tvm::ffi::Any> EncodeCommonRuntimeArgs(const std::string& segment_kind) const;

  /*! \brief Load logical buffer shapes from the semantic manifest when present. */
  void LoadLogicalBufferShapes(const tvm::tir::PrimFunc& func);

  /*! \brief Return manifest-backed logical shape for a buffer when available. */
  std::vector<int64_t> GetLogicalBufferShape(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return static logical element count, falling back to the lowered buffer shape. */
  int64_t GetLogicalBufferElementCount(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the logical 32x32 tile/page count for a buffer. */
  int GetLogicalBufferTileCount(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the manifest-backed 1-D logical vector length for a row/state buffer. */
  int64_t GetLogicalVectorLength(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the manifest-backed logical matrix shape for a fragment buffer. */
  std::pair<int64_t, int64_t> GetLogicalMatrixShape(const tvm::tir::Buffer& buffer) const;

  /*! \brief Detect matmul call using Op comparison (not string matching) */
  bool IsMatmulCall(const tvm::tir::CallNode* op) const;

  /*! \brief Extract GEMM buffer names and dimensions from a tl.tileop.gemm_py call */
  void ExtractGemmInfo(const tvm::tir::CallNode* op);

  /*! \brief Convert a DataType to TT-Metal data format string */
  static std::string DataTypeToDataFormat(tvm::DataType dtype);

  /*! \brief Detect clear operation using Op comparison */
  bool IsClearOperation(const tvm::tir::CallNode* op) const;

  /*! \brief Detect copy operation by buffer pattern */
  bool IsCopyOperation(const tvm::tir::BufferStoreNode* op) const;

  /*! \brief Determine copy direction using buffer scopes */
  CopyDirection GetCopyDirection(const tvm::tir::BufferStoreNode* op) const;

  /*! \brief Infer copy tile index from a post-LowerTileOp staged copy loop */
  tvm::PrimExpr InferCopyTileIndex(const tvm::tir::BufferStoreNode* op,
                                   const tvm::tir::Var& loop_var) const;

  /*! \brief Infer the base hardware-tile index for a staged copy loop nest */
  tvm::PrimExpr InferStagedCopyBaseTileIndex(
      const tvm::tir::BufferStoreNode* op,
      const std::vector<tvm::tir::Var>& loop_vars_to_zero) const;

  /*! \brief Zero out thread/local element vars when extracting tile base indices */
  tvm::PrimExpr ZeroThreadAndLoopVars(const tvm::PrimExpr& expr,
                                      const tvm::tir::Var& loop_var) const;

  /*! \brief Zero out a set of loop vars and thread vars when extracting tile bases */
  tvm::PrimExpr ZeroThreadAndLoopVars(const tvm::PrimExpr& expr,
                                      const std::vector<tvm::tir::Var>& loop_vars) const;

  /*! \brief Whether an expression uses transport-local vars (thread or supplied loop vars). */
  bool ExprUsesTransportVar(const tvm::PrimExpr& expr,
                            const std::vector<tvm::tir::Var>& loop_vars) const;

  /*! \brief Pick the active thread var that indexes logical matrix rows, when unique. */
  tvm::tir::Var SelectLogicalRowThreadVar(int64_t logical_rows) const;

  /*! \brief Select the 2-D transport axes for a possibly higher-rank global buffer view. */
  std::pair<int, int> SelectStagedCopyTransportAxes(
      const tvm::ffi::Array<tvm::PrimExpr>& global_indices,
      const std::vector<tvm::tir::Var>& loop_vars) const;

  /*! \brief Build explicit host-axis order matching staged-copy transport linearization. */
  std::vector<int64_t> BuildStagedCopyHostAxisOrder(
      const tvm::ffi::Array<tvm::PrimExpr>& global_indices,
      const tvm::ffi::Array<tvm::Integer>& global_shape,
      int row_axis,
      int col_axis) const;

  /*! \brief Find a staged copy BufferStore inside a loop nest and collect nested loop vars */
  const tvm::tir::BufferStoreNode* FindNestedCopyStore(
      const tvm::tir::Stmt& stmt,
      std::vector<tvm::tir::Var>* nested_loop_vars) const;

  /*! \brief Collect staged copy stores reachable under a loop/body statement */
  void CollectNestedCopyStores(const tvm::tir::Stmt& stmt,
                               std::vector<tvm::tir::Var>* loop_stack,
                               std::vector<NestedCopyMatch>* matches) const;

  /*! \brief Record runtime schema for staged copy global input/output buffers */
  void RecordStagedCopyBufferBinding(const tvm::tir::BufferStoreNode* op,
                                     CopyDirection direction);

  /*! \brief Record Stage 2 copy requirements for a DRAM -> DRAM copy */
  void RecordDramToDramCopy(const tvm::tir::BufferStoreNode* op);

  /*! \brief Register a segment-local interleaved accessor descriptor */
  void RegisterAccessor(const std::string& segment_kind,
                        const tvm::tir::Buffer& buffer,
                        int compile_time_arg_offset,
                        int compile_time_arg_count,
                        int common_runtime_arg_offset,
                        int common_runtime_arg_count,
                        int args_config_bits,
                        int transport_page_size_bytes = 0,
                        std::vector<int64_t> host_axis_order = {},
                        bool transpose_2d = false);

  std::string ResolveAccessorSegmentKind(CopyDirection direction) const;

  /*! \brief Return compile-time accessor slot for a reader/source buffer */
  int GetReadAccessorSlot(const std::string& segment_kind, const tvm::tir::Buffer& buffer,
                          CopyDirection direction);

  /*! \brief Return compile-time accessor slot for a writer/destination buffer */
  int GetWriteAccessorSlot(const std::string& segment_kind, const tvm::tir::Buffer& buffer,
                           CopyDirection direction);

  int GetOrAllocateSegmentAccessorSlot(std::unordered_map<std::string, int>* slot_map,
                                       const std::string& segment_kind,
                                       const tvm::tir::Buffer& buffer);

  /*! \brief Estimate a copy tile page size for a buffer */
  int EstimateCopyPageSize(const tvm::tir::Buffer& buffer) const;

  /*! \brief Override CB requirement page sizing after a more specific contract is known. */
  void SetRequirementPageLayout(int requirement_index, int page_size, int num_pages);

  /*! \brief Determine whether a staged copy should use page/stick transport. */
  bool UseStagedCopyPageTransport(const tvm::tir::Buffer& shared_buffer) const;

  /*! \brief Generate matmul builtin sequence */
  tvm::tir::Stmt GenerateMatmulSequence(const tvm::tir::CallNode* op,
                                        bool retain_in0 = false,
                                        bool retain_in1 = false,
                                        bool publish_out = true,
                                        bool reacquire_in0 = false,
                                        bool reacquire_in1 = false);
  tvm::tir::Stmt GenerateMatmulSequenceForOutputRequirement(int out_req_index,
                                                            bool retain_in0,
                                                            bool retain_in1,
                                                            bool reserve_out,
                                                            bool publish_out,
                                                            bool reacquire_in0,
                                                            bool reacquire_in1);
  tvm::tir::Buffer CreateClearAccumPartialsBuffer(const tvm::tir::Buffer& buffer);
  bool ClearAccumReloadNeedsDataFormatReconfig() const;
  tvm::tir::Stmt GenerateMatmulSequenceWithPartialReload(int out_req_index,
                                                         int partials_cb_id,
                                                         bool retain_in0,
                                                         bool retain_in1,
                                                         bool reserve_out,
                                                         bool publish_out,
                                                         bool reacquire_in0,
                                                         bool reacquire_in1);
  tvm::tir::Stmt GenerateAccumulatingMatmulSequence(const tvm::tir::CallNode* op,
                                                    bool retain_in0,
                                                    bool retain_in1,
                                                    bool reacquire_in0,
                                                    bool reacquire_in1);
  tvm::tir::Stmt GenerateAddFragmentSequence(const tvm::tir::Buffer& dst,
                                             const tvm::tir::Buffer& src,
                                             const tvm::PrimExpr& num_elements);
  tvm::tir::Stmt GenerateAddFragmentFromCBFrontSequence(const tvm::tir::Buffer& dst,
                                                        int src_cb_id,
                                                        const tvm::PrimExpr& num_elements,
                                                        const tvm::tir::Buffer& src_buffer);

  /*! \brief Generate copy builtin sequence (DRAM->CB, CB->DRAM, CB->CB) */
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op);

  /*! \brief Generate staged copy builtin sequence for a collapsed loop */
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op,
                                      const tvm::PrimExpr& tile_index);

  /*! \brief Generate staged copy builtin sequence for a collapsed loop nest */
  tvm::tir::Stmt GenerateStagedCopyLoopSequence(const tvm::tir::BufferStoreNode* op,
                                                const tvm::PrimExpr& base_tile_index);

  /*! \brief Generate fused staged copy sequence for a read-then-write tile loop */
  tvm::tir::Stmt GenerateFusedStagedCopySequence(const tvm::tir::BufferStoreNode* dram_to_cb,
                                                 const tvm::tir::BufferStoreNode* cb_to_dram,
                                                 const tvm::PrimExpr& base_tile_index);

  /*! \brief Generate clear builtin sequence */
  tvm::tir::Stmt GenerateClearSequence(const tvm::tir::CallNode* op);

  /*! \brief Detect and lower canonical scalar row-reduction loops on local fragment buffers. */
  bool MatchDirectRowReduction(const tvm::tir::ForNode* op, RowReductionMatch* match) const;
  bool MatchAllocatedRowReduction(const tvm::tir::AllocateNode* op, RowReductionMatch* match) const;
  bool MatchGroupedRowReduction(const tvm::tir::ForNode* op, RowReductionMatch* match) const;
  tvm::tir::Stmt GenerateRowReductionSequence(const RowReductionMatch& match);
  bool MatchDirectRowBroadcast(const tvm::tir::ForNode* op, RowBroadcastMatch* match) const;
  tvm::tir::Stmt GenerateRowBroadcastSequence(const RowBroadcastMatch& match);
  bool MatchScalarFmaStore(const tvm::tir::BufferStoreNode* op, ScalarFmaMatch* match) const;
  bool MatchGroupedScalarFmaLoop(const tvm::tir::ForNode* op, ScalarFmaMatch* match) const;
  tvm::tir::Stmt GenerateScalarFmaSequence(const ScalarFmaMatch& match);
  bool MatchExp2RowBroadcastAffine(const tvm::tir::ForNode* op,
                                   Exp2RowBroadcastAffineMatch* match) const;
  tvm::tir::Stmt GenerateExp2RowBroadcastAffineSequence(
      const Exp2RowBroadcastAffineMatch& match);
  bool MatchScalarExp2AffineStore(const tvm::tir::BufferStoreNode* op,
                                  ScalarExp2AffineMatch* match) const;
  bool MatchGroupedScalarExp2AffineLoop(const tvm::tir::ForNode* op,
                                        ScalarExp2AffineMatch* match) const;
  tvm::tir::Stmt GenerateScalarExp2AffineSequence(const ScalarExp2AffineMatch& match);
  bool MatchDirectFragmentFill(const tvm::tir::ForNode* op, FragmentFillMatch* match) const;
  bool MatchScalarFragmentFillStore(const tvm::tir::BufferStoreNode* op, FragmentFillMatch* match) const;
  tvm::tir::Stmt GenerateFragmentFillSequence(const FragmentFillMatch& match);
  bool MatchScalarMaxStore(const tvm::tir::BufferStoreNode* op, ScalarMaxMatch* match) const;
  bool MatchGroupedScalarMaxLoop(const tvm::tir::ForNode* op, ScalarMaxMatch* match) const;
  tvm::tir::Stmt GenerateScalarMaxSequence(const ScalarMaxMatch& match);
  bool MatchDirectFragmentCast(const tvm::tir::ForNode* op, FragmentCastMatch* match) const;
  tvm::tir::Stmt GenerateFragmentCastSequence(const FragmentCastMatch& match,
                                              bool publish_cb = false);
  bool MatchScalarFragmentCopyStore(const tvm::tir::BufferStoreNode* op,
                                    ScalarFragmentCopyMatch* match) const;
  bool MatchGroupedScalarFragmentCopyLoop(const tvm::tir::ForNode* op,
                                          ScalarFragmentCopyMatch* match) const;
  tvm::tir::Stmt GenerateScalarFragmentCopySequence(const ScalarFragmentCopyMatch& match);
  void LoadBufferFlowContracts(const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>& lowering_requirements);
  bool ShouldRetainComputeInputBuffer(const tvm::tir::Buffer& buffer,
                                      int current_order_index) const;
  bool ShouldReacquireComputeInputBuffer(const tvm::tir::Buffer& buffer,
                                         int current_order_index) const;
  bool ShouldPublishBufferResult(const tvm::tir::Buffer& buffer,
                                 int current_order_index) const;
  bool MatchDirectLocalToCBSliceLoop(const tvm::tir::ForNode* op, LocalToCBSliceMatch* match) const;
  tvm::tir::Stmt GenerateLocalToCBSliceLoopSequence(const tvm::tir::ForNode* op,
                                                    const LocalToCBSliceMatch& match);
  void ActivateCurrentComputeContractPayload();
  void RecordComputeEpilogueOp(tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> op_payload);

  // StmtExprMutator overrides
  tvm::tir::Stmt VisitStmt_(const tvm::tir::AttrStmtNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::DeclBufferNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::AllocateNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::SeqStmtNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::ForNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::EvaluateNode* op) override;
  tvm::tir::Stmt VisitStmt_(const tvm::tir::BufferStoreNode* op) override;

  tvm::tir::PrimFunc current_func_;
  // Maps buffer object → requirement_index in cb_requirements_
  std::map<tvm::tir::Buffer, int, std::less<>> buffer_to_req_;
  // Secondary lookup by backing storage identity for aliased Buffer objects.
  std::unordered_map<const tvm::tir::VarNode*, int> buffer_data_to_req_index_;
  // Tertiary lookup by logical buffer identity for cases like C_local where
  // the same physical resource can appear as distinct Buffer/Var objects.
  std::unordered_map<std::string, int> buffer_identity_to_req_index_;
  std::vector<CBRequirement> cb_requirements_;
  bool saw_copy_op_ = false;
  bool needs_copy_runtime_args_ = false;
  tvm::tir::Buffer copy_input_buffer_;
  tvm::tir::Buffer copy_output_buffer_;
  std::string copy_input_buffer_name_;
  std::string copy_output_buffer_name_;

  // GEMM info populated by ExtractGemmInfo (pre-scan)
  tvm::tir::Buffer gemm_a_buffer_;
  tvm::tir::Buffer gemm_b_buffer_;
  tvm::tir::Buffer gemm_c_buffer_;
  std::string gemm_a_buffer_name_;
  std::string gemm_b_buffer_name_;
  std::string gemm_c_buffer_name_;
  std::string gemm_c_scope_;
  bool gemm_has_mbarrier_ = false;
  tvm::tir::Buffer gemm_mbarrier_buffer_;
  std::string gemm_mbarrier_buffer_name_;
  std::string gemm_mbarrier_scope_;
  std::vector<std::string> gemm_mbarrier_index_exprs_;
  int gemm_a_req_index_ = -1;  // requirement_index for GEMM input A
  int gemm_b_req_index_ = -1;  // requirement_index for GEMM input B
  int gemm_c_req_index_ = -1;  // requirement_index for GEMM output C
  int gemm_m_ = 0;
  int gemm_n_ = 0;
  int gemm_k_ = 0;
  std::unordered_set<std::string> gemm_contract_signatures_;
  std::unordered_map<std::string, int> compute_contract_payload_index_by_signature_;
  std::vector<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>> multi_gemm_contract_payloads_;
  std::vector<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>> multi_compute_contract_payloads_;
  std::vector<std::vector<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>> compute_epilogue_payloads_;
  std::vector<std::unordered_set<std::string>> compute_contract_known_buffers_;
  std::vector<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>> compute_epilogue_payloads_flat_;
  std::unordered_map<std::string, tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>
      fragment_materialization_contracts_by_target_buffer_;
  int active_compute_contract_payload_index_ = -1;
  std::unordered_map<std::string, int> gemm_input_buffer_num_tiles_;
  bool gemm_transpose_a_ = false;
  bool gemm_transpose_b_ = false;
  int gemm_policy_type_ = 0;
  bool gemm_clear_accum_ = false;
  int gemm_k_pack_ = 1;
  int gemm_wg_wait_ = 0;
  bool gemm_dst_full_sync_en_ = false;
  bool gemm_bfp8_pack_precise_ = false;
  std::vector<std::pair<std::string, std::string>> gemm_defines_;
  std::vector<std::pair<std::string, uint32_t>> gemm_named_compile_args_;
  tvm::DataType gemm_a_dtype_;
  tvm::DataType gemm_b_dtype_;
  tvm::DataType gemm_c_dtype_;
  tvm::ffi::Array<tvm::Integer> copy_input_shape_;
  tvm::ffi::Array<tvm::Integer> copy_output_shape_;
  tvm::ffi::Array<tvm::Integer> copy_intermediate_shape_;
  std::unordered_set<const tvm::tir::VarNode*> thread_index_vars_;
  std::unordered_set<std::string> thread_index_var_names_;
  std::unordered_map<const tvm::tir::VarNode*, int64_t> thread_index_var_static_extents_;
  std::unordered_set<const tvm::tir::VarNode*> block_index_vars_;
  std::unordered_set<std::string> block_index_var_names_;
  std::vector<AccessorDescriptor> accessor_descriptors_;
  std::string current_segment_kind_;
  std::unordered_map<std::string, int> read_accessor_slots_;
  std::unordered_map<std::string, int> write_accessor_slots_;
  std::unordered_map<std::string, int> cb_consumed_fragment_pages_by_buffer_identity_;
  std::unordered_map<std::string, int> cb_consumed_fragment_use_count_by_buffer_identity_;
  std::unordered_map<std::string, BufferFlowContract> buffer_flow_contracts_;
  std::unordered_map<std::string, std::vector<int64_t>> logical_buffer_shapes_;
  std::unordered_map<const Object*, int> stmt_order_index_by_node_;

  // Requirement index counter (sequential, 0-based)
  int next_requirement_index_ = 0;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LOWER_BLACKHOLE_OPS_H_
