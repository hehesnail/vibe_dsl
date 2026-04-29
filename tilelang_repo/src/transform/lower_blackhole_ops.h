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
#include "common/blackhole_lowering_requirements.h"
#include "common/blackhole_tile_compute_covering.h"
#include "common/tt_target_program.h"

#include <tvm/ir/op.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

constexpr const char* kTLBlackholeTTMetalBuiltinSelection =
    "tl.blackhole_tt_metal_builtin_selection";

tvm::transform::Pass SelectBlackholeTTMetalBuiltins();

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

struct GemmComputeOpFact {
  std::string a_buffer;
  std::string b_buffer;
  std::string c_buffer;
  int m = 0;
  int n = 0;
  int k = 0;
  bool transpose_a = false;
  bool transpose_b = false;
  int policy_type = 0;
  bool clear_accum = false;
  int k_pack = 1;
  int wg_wait = 0;
  bool dst_full_sync_en = false;
  bool bfp8_pack_precise = false;
  std::vector<std::pair<std::string, std::string>> defines;
  std::vector<std::pair<std::string, uint32_t>> named_compile_args;
  std::string mbarrier_buffer;
  std::string mbarrier_scope;
  std::vector<std::string> mbarrier_index_exprs;
  tvm::DataType a_dtype;
  tvm::DataType b_dtype;
  tvm::DataType c_dtype;
};

struct TileComputeDAGLoweringDecision {
  int64_t node_id{-1};
  std::string operation_name;
  BlackholeTileComputeCoveringDecision covering;
  int64_t fanout_use_count{0};
  std::string fanout_policy;
};

/*!
 * \brief PlanTTKernelABI Pass
 *
 * This pass transforms TileLang high-level operations:
 * - T.copy(A, B) -> CB reserve + NOC read/write + push/pop sequence
 * - T.gemm(A, B, C) -> MM init + tile_regs_acquire + matmul_tiles + pack_tile
 * - T.clear(C) -> tile_regs_acquire (zero DST)
 *
 * Also records CB requirements in function attributes for PlanTTCBAlloc.
 */
class PlanTTKernelABI : public tvm::tir::StmtExprMutator {
 public:
 PlanTTKernelABI();

  /*! \brief Main entry point */
  tvm::tir::PrimFunc Transform(const tvm::tir::PrimFunc& func);

  /*! \brief Rewrite compute-side high-level TIR idioms into selected TT-Metal builtins. */
  tvm::tir::PrimFunc SelectComputeBuiltins(const tvm::tir::PrimFunc& func);

  /*! \brief Get TT kernels synthesized during Transform. */
  tvm::ffi::Array<TTKernel> GetTTKernels() const { return tt_kernels_; }

  /*! \brief Get TT ABI plans synthesized during Transform. */
  tvm::ffi::Array<TTABIPlan> GetTTABIPlans() const { return tt_abi_plans_; }

  /*! \brief Get TT compute op plans synthesized during Transform. */
  tvm::ffi::Array<TTComputeOpPlan> GetTTComputeOpPlans() const { return tt_compute_op_plans_; }

  /*! \brief Get TT live-form plans synthesized during Transform. */
  tvm::ffi::Array<TTLiveFormPlan> GetTTLiveFormPlans() const { return tt_live_form_plans_; }

  /*! \brief Get TT materialization plans synthesized during Transform. */
  tvm::ffi::Array<TTMaterializationPlan> GetTTMaterializationPlans() const {
    return tt_materialization_plans_;
  }

  /*! \brief Get TT consumer binding plans synthesized during Transform. */
  tvm::ffi::Array<TTConsumerBindingPlan> GetTTConsumerBindingPlans() const {
    return tt_consumer_binding_plans_;
  }

  /*! \brief Get staged CB plans synthesized during selection/lowering.
   *
   * The staged plans already own the CB requirement contract. Before PlanTTCBAlloc
   * finalizes hardware bindings, `cb_id` carries the dense requirement slot
   * referenced by the lowered IR.
   */
  tvm::ffi::Array<TTCBPlan> GetStagedCBPlans() const;

 private:
  struct FutureBufferUses {
    bool has_compute_consume = false;
    bool has_transport_consume = false;
    bool has_reference = false;
  };

  struct ExactTiledCBValue {
    tvm::tir::Buffer buffer;
    int cb_id = -1;
    int num_tiles = 0;
    int64_t num_elements = 0;
    int64_t row_width = 0;
    bool borrowed_live = false;
  };

  struct ComputeOperandPlanSeed {
    std::string role;
    tvm::tir::Buffer buffer;
    std::string transform_kind;
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
    bool accumulate_existing = false;
  };

  struct FragmentCastMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::PrimExpr dst_offset;
    tvm::PrimExpr src_offset;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr row_width;
  };

  using TileComputeSourceEmitterFn =
      tvm::tir::Stmt (PlanTTKernelABI::*)(
          const tvm::tir::CallNode* op,
          const BlackholeTileComputeCoveringDecision& covering);

  struct TileComputeSourceEmitterHook {
    BlackholeTileComputeSourceEmitterKind kind;
    TileComputeSourceEmitterFn emit;
  };

  struct LocalToCBSliceMatch {
    tvm::tir::Buffer dst;
    tvm::tir::Buffer src;
    tvm::tir::Buffer cast_src;
    tvm::PrimExpr dst_offset_elements;
    tvm::PrimExpr num_elements;
    tvm::PrimExpr row_width;
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

  struct SpatialLiveValueRef {
    std::string name;
    int64_t index = -1;
  };

  struct SpatialMaterializationBoundaryRef {
    std::string name;
    int64_t index = -1;
    std::string source_live_value;
    int64_t source_live_value_index = -1;
    std::string source_subject;
    std::string target_live_value;
    int64_t target_live_value_index = -1;
    std::string target_subject;
    std::string live_value_edge;
    int64_t live_value_edge_index = -1;
    std::string logical_coverage;
    std::string event_lifetime_kind;
    int64_t min_publish_pages = 0;
    int64_t max_consume_pages = 0;
  };

  /*! \brief Get CB configuration from function attributes */
  CBConfig GetCBConfig() const;

  /*! \brief Register a CB requirement for a buffer and return its requirement_index.
   *
   * The returned index is the position of this requirement in cb_requirements_.
   * It is used as the cb_id placeholder in the IR body.  PlanTTCBAlloc will
   * replace all requirement_index placeholders with the final hardware cb_id.
   */
  int AllocateRequirementIndex(const tvm::tir::Buffer& buffer, CBType type);

  /*! \brief Load staged CB plans from TTProgram and preserve requirement indices. */
  void LoadSeededCBRequirements(const tvm::tir::PrimFunc& func);

  /*! \brief Load staged exact compute op plans from TTProgram. */
  void LoadSeededComputeOpPlans(const tvm::tir::PrimFunc& func);

  /*! \brief Store minimal segment/kernel plan inferred during lowering */
  void StoreSegmentPlan(tvm::tir::PrimFunc& func);

  /*! \brief Store per-segment accessor descriptors for dataflow kernels */
  void StoreAccessorDescriptors(tvm::tir::PrimFunc& func);

  /*! \brief Reject unresolved compute builtin legality before TTProgram construction. */
  void RejectUnsupportedComputeOps(const std::vector<std::string>& unsupported_ops);

  /*! \brief Encode current lowering-time accessor descriptors as TIR attrs */
  tvm::ffi::Array<tvm::ffi::Any> EncodeAccessorDescriptors(const std::string& segment_kind) const;

  /*! \brief Encode empty or richer common-runtime args for a segment */
  tvm::ffi::Array<tvm::ffi::Any> EncodeCommonRuntimeArgs(const std::string& segment_kind) const;

  /*! \brief Load logical buffer shapes from current IR and typed SpatialPlan fields. */
  void LoadLogicalBufferShapes(const tvm::tir::PrimFunc& func,
                               const BlackholeLoweringSupportFacts& lowering_support_facts,
                               const SpatialPlan& spatial_plan);

  /*! \brief Return manifest-backed logical shape for a buffer when available. */
  std::vector<int64_t> GetLogicalBufferShape(const tvm::tir::Buffer& buffer) const;

  /*! \brief Encode the current IR-visible buffer shape, preferring logical shape metadata. */
  tvm::ffi::Array<tvm::Integer> GetEncodedCurrentBufferShape(
      const tvm::tir::Buffer& buffer) const;

  /*! \brief Return static logical element count, falling back to the lowered buffer shape. */
  int64_t GetLogicalBufferElementCount(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the logical 32x32 tile/page count for a buffer. */
  int GetLogicalBufferTileCount(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the manifest-backed 1-D logical vector length for a row/state buffer. */
  int64_t GetLogicalVectorLength(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return the manifest-backed logical matrix shape for a compute-region buffer. */
  std::pair<int64_t, int64_t> GetLogicalMatrixShape(const tvm::tir::Buffer& buffer) const;

  /*! \brief Return true when the logical value is exactly one full hardware tile. */
  bool IsSingleFullTileLogicalMatrix(const tvm::tir::Buffer& buffer) const;

  /*! \brief Recover staged-copy shared shape from the current copy op when the buffer is flat. */
  tvm::ffi::Array<tvm::Integer> GetEncodedCurrentStagedCopySharedShape(
      const tvm::tir::BufferStoreNode* op,
      const std::vector<tvm::tir::Var>& loop_vars_to_zero) const;

  /*! \brief Infer staged-copy shared matrix coverage from transport-variable global access. */
  std::optional<std::pair<int64_t, int64_t>> InferStagedCopySharedShapeFromTransportCoverage(
      const tvm::tir::BufferStoreNode* op,
      const std::vector<tvm::tir::Var>& loop_vars_to_zero) const;

  /*! \brief Load logical tile layout facts from typed SpatialPlan LayoutSpec fields. */
  void LoadLogicalTileLayoutSpecs(const SpatialPlan& spatial_plan);

  /*! \brief Return logical tile layout facts for a buffer, or nullptr if absent. */
  const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>* FindLogicalTileLayoutSpec(
      const tvm::tir::Buffer& buffer) const;

  /*! \brief Load first-class SpatialPlan live-value references for TT physical plans. */
  void LoadSpatialLiveValueBoundaries(const SpatialPlan& plan);

  /*! \brief Return the SpatialPlan materialization boundary for a boundary index. */
  const SpatialMaterializationBoundaryRef* FindSpatialMaterializationBoundaryRef(
      int64_t materialization_boundary_index) const;

  /*! \brief Load compute-region buffer to physical accumulator bindings from compute regions. */
  void LoadPhysicalComputeBufferBindings(const tvm::tir::PrimFunc& func);

  /*! \brief Resolve the physical accumulator/backing buffer for a compute-region buffer. */
  tvm::tir::Buffer ResolvePhysicalComputeBuffer(const tvm::tir::Buffer& buffer) const;

  /*! \brief Detect matmul call using Op comparison (not string matching) */
  bool IsMatmulCall(const tvm::tir::CallNode* op) const;

  /*! \brief Detect a matmul whose explicit M/N output extent is one full hardware tile. */
  bool IsSingleFullTileMatmulOutput(const tvm::tir::CallNode* op) const;

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
  tvm::tir::Stmt MaybeWrapComputeSegment(const tvm::tir::Stmt& stmt) const;

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

  /*! \brief Override CB requirement page sizing after a more specific fact is known. */
  void SetRequirementPageLayout(int requirement_index, int page_size, int num_pages);

  /*! \brief Mark two CB requirements as overlapping so planner cannot reuse one CB for both. */
  void MarkRequirementLifetimeOverlap(int lhs_requirement_index, int rhs_requirement_index);

  /*! \brief Determine whether a staged copy should use page/stick transport. */
  bool UseStagedCopyPageTransport(const tvm::tir::Buffer& shared_buffer) const;

  /*! \brief Generate matmul builtin sequence */
  tvm::tir::Stmt LowerMatmulCallWithFlowAnalysis(const tvm::tir::CallNode* op,
                                                 int current_order_index,
                                                 const FragmentCastMatch* post_merge_cast = nullptr,
                                                 int post_merge_cast_order_index = -1,
                                                 bool* consumed_post_merge_cast = nullptr);
  tvm::tir::Stmt GenerateMatmulSequence(const tvm::tir::CallNode* op,
                                        bool retain_in0 = false,
                                        bool retain_in1 = false,
                                        bool publish_out = true,
                                        bool publish_transport_out = true,
                                        bool preserve_out_local_state = false,
                                        bool reacquire_in0 = false,
                                        bool reacquire_in1 = false,
                                        const FragmentCastMatch* post_merge_cast = nullptr,
                                        int post_merge_cast_order_index = -1);
  tvm::tir::Stmt GenerateMatmulSequenceForOutputRequirement(int out_req_index,
                                                            bool retain_in0,
                                                            bool retain_in1,
                                                            bool reserve_out,
                                                            bool publish_out,
                                                            bool reacquire_in0,
                                                            bool reacquire_in1);
  tvm::tir::Buffer CreateClearAccumPartialsBuffer(const tvm::tir::Buffer& buffer);
  tvm::tir::Buffer CreateFragmentMergeReloadBuffer(const tvm::tir::Buffer& buffer);
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
                                                    bool publish_transport_out,
                                                    bool preserve_out_local_state,
                                                    bool reacquire_in0,
                                                    bool reacquire_in1,
                                                    const FragmentCastMatch* post_merge_cast,
                                                    int post_merge_cast_order_index,
                                                    bool merge_with_zero_reload);
  bool CanPublishPostMergeCastWithPackTile(const FragmentCastMatch& match,
                                           int cast_order_index) const;
  bool HasZeroFragmentFillFact(const tvm::tir::Buffer& buffer) const;
  int PreparePostMergeCastPublishCB(const FragmentCastMatch& match, int num_c_tiles);
  tvm::tir::Stmt GenerateAddFragmentSequence(const tvm::tir::Buffer& dst,
                                             const tvm::tir::Buffer& src,
                                             const tvm::PrimExpr& num_elements);
  tvm::tir::Stmt GenerateAddFragmentFromCBFrontSequence(const tvm::tir::Buffer& dst,
                                                        int src_cb_id,
                                                        const tvm::PrimExpr& num_elements,
                                                        const tvm::tir::Buffer& src_buffer);
  tvm::tir::Stmt GenerateMergeFragmentTilesSequence(const tvm::tir::Buffer& dst,
                                                    int partials_cb_id,
                                                    const tvm::tir::Buffer& partials_buffer,
                                                    int reload_cb_id,
                                                    const tvm::tir::Buffer& reload_buffer,
                                                    int live_form_cb_id,
                                                    const tvm::tir::Buffer& live_form_buffer,
                                                    const tvm::PrimExpr& num_elements,
                                                    int num_c_tiles,
                                                    bool materialize_live_form_to_local_state,
                                                    int publish_cb_id,
                                                    int materialized_cast_cb_id = -1,
                                                    bool merge_with_zero_reload = false,
                                                    int live_reload_cb_id = -1,
                                                    const tvm::tir::Buffer& live_reload_buffer =
                                                        tvm::tir::Buffer());
  tvm::tir::Buffer CreateEphemeralBufferLike(const tvm::tir::Buffer& buffer,
                                             const std::string& suffix) const;
  tvm::tir::Buffer CreateConstantTileBuffer(tvm::DataType dtype, const std::string& suffix) const;
  tvm::DataType ExactTiledCBStorageDType(tvm::DataType dtype) const;
  int PrepareExactTiledCBRequirement(const tvm::tir::Buffer& buffer);
  tvm::tir::Stmt FillLocalTileBuffer(const tvm::tir::Buffer& buffer,
                                     const tvm::PrimExpr& value);
  void PopulateExactTiledCBValueShape(const tvm::tir::Buffer& buffer,
                                      ExactTiledCBValue* value) const;
  bool TryCreateLiveExactTiledCBValue(const tvm::tir::Buffer& buffer,
                                      ExactTiledCBValue* value) const;
  bool TryCreateExactOutputLiveTiledCBValue(const tvm::tir::Buffer& buffer,
                                            ExactTiledCBValue* value) const;
  bool TryCreateSelectedSourceLiveExactTiledCBValue(const tvm::tir::Buffer& buffer,
                                                    ExactTiledCBValue* value);
  ExactTiledCBValue CreateExactInputCBValue(const tvm::tir::Buffer& src,
                                            const std::string& suffix);
  ExactTiledCBValue CreateRowReductionInputCBValue(const tvm::tir::Buffer& src);
  bool TryGetLastFragmentFillValue(const tvm::tir::Buffer& buffer,
                                   tvm::PrimExpr* value) const;
  tvm::tir::Stmt PublishConstantToExactTiledCB(const tvm::tir::Buffer& buffer,
                                              const tvm::PrimExpr& fill_value,
                                              const ExactTiledCBValue& cb_value);
  tvm::tir::Stmt PublishExactInputToTiledCB(const tvm::tir::Buffer& src,
                                           ExactTiledCBValue* cb_value);
  void RecordExactOutputLiveForm(const tvm::tir::Buffer& dst,
                                 const ExactTiledCBValue& cb_value);
  void MarkExactCBValuesOverlap(std::initializer_list<int> cb_ids);
  tvm::tir::Stmt PublishLocalBufferToExactTiledCB(const tvm::tir::Buffer& src,
                                                  const ExactTiledCBValue& cb_value);
  tvm::tir::Stmt MaterializeExactTiledCBToLocalBuffer(const tvm::tir::Buffer& dst,
                                                      const ExactTiledCBValue& cb_value,
                                                      bool pop_front = true);
  tvm::tir::Stmt AttachExactOutputLiveFormMarker(const tvm::tir::Buffer& dst,
                                                 const ExactTiledCBValue& cb_value,
                                                 const tvm::tir::Stmt& body) const;
  ExactTiledCBValue CreatePublishedExactTiledCBValue(const tvm::tir::Buffer& src,
                                                     const std::string& suffix);
  ExactTiledCBValue CreateEmptyExactTiledCBValue(const tvm::tir::Buffer& like_buffer,
                                                 const std::string& suffix);
  ExactTiledCBValue CreateConstantExactTiledCBValue(tvm::DataType dtype,
                                                    const std::string& suffix);
  ExactTiledCBValue CreateReduceScalerExactTiledCBValue();

  /*! \brief Generate copy builtin sequence (DRAM->CB, CB->DRAM, CB->CB) */
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op);

  /*! \brief Generate staged copy builtin sequence for a collapsed loop */
  tvm::tir::Stmt GenerateCopySequence(const tvm::tir::BufferStoreNode* op,
                                      const tvm::PrimExpr& tile_index);

  /*! \brief Generate staged copy builtin sequence for a collapsed loop nest */
  tvm::tir::Stmt GenerateStagedCopyLoopSequence(const tvm::tir::BufferStoreNode* op,
                                                const tvm::PrimExpr& base_tile_index,
                                                const std::vector<tvm::tir::Var>& loop_vars_to_zero);

  /*! \brief Generate fused staged copy sequence for a read-then-write tile loop */
  tvm::tir::Stmt GenerateFusedStagedCopySequence(const tvm::tir::BufferStoreNode* dram_to_cb,
                                                 const tvm::tir::BufferStoreNode* cb_to_dram,
                                                 const tvm::PrimExpr& base_tile_index,
                                                 const std::vector<tvm::tir::Var>& loop_vars_to_zero);

  /*! \brief Generate clear builtin sequence */
  tvm::tir::Stmt GenerateClearSequence(const tvm::tir::CallNode* op);

  /*! \brief Detect and lower explicit preserved tile reductions. */
  bool MatchExplicitTileReduce(const tvm::tir::CallNode* op, RowReductionMatch* match) const;
  bool MatchExplicitTileTypecast(const tvm::tir::CallNode* op,
                                 FragmentCastMatch* match) const;
  tvm::tir::Stmt LowerExplicitTileComputeCall(const tvm::tir::CallNode* op);
  void LoadTileComputeDAGLoweringPlan(const tvm::tir::PrimFunc& func);
  BlackholeTileComputeCoveringDecision ConsumeTileComputeDAGLoweringDecision(
      const std::string& operation_name);
  int64_t CurrentTileComputeDAGNodeId() const;
  tvm::ffi::String CurrentTileComputeDAGSourceEmitter() const;
  tvm::ffi::String CurrentTileComputeDAGMaterializationPolicy() const;
  int64_t CurrentTileComputeDAGFanoutUseCount() const;
  tvm::ffi::String CurrentTileComputeDAGFanoutPolicy() const;
  tvm::tir::Stmt EmitCoveredBlackholeTileCompute(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  static const std::vector<TileComputeSourceEmitterHook>&
  GetTileComputeSourceEmitterHooks();
  const TileComputeSourceEmitterHook* FindTileComputeSourceEmitterHook(
      BlackholeTileComputeSourceEmitterKind source_emitter) const;
  tvm::tir::Buffer GetBlackholeTileComputeBufferArg(
      const tvm::tir::CallNode* op,
      BlackholeTileComputeOperandRole role,
      const BlackholeTileComputeCoveringDecision& covering) const;
  tvm::PrimExpr GetBlackholeTileComputePrimArg(
      const tvm::tir::CallNode* op,
      size_t index,
      const BlackholeTileComputeCoveringDecision& covering) const;
  tvm::tir::Stmt EmitFillFragmentTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitCopyTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitTypecastTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitBinaryMaxTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitAddTilesComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitMulTilesComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitMulTilesBcastColsComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitAddTilesBcastColsComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitExp2TileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitRecipTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt EmitReduceTileComputeSource(
      const tvm::tir::CallNode* op,
      const BlackholeTileComputeCoveringDecision& covering);
  tvm::tir::Stmt GenerateRowReductionSequence(const RowReductionMatch& match);
  tvm::tir::Stmt GenerateFillTileSequence(const tvm::tir::Buffer& dst,
                                          const tvm::PrimExpr& value,
                                          const tvm::PrimExpr& num_elements);
  tvm::tir::Stmt GenerateCopyTileSequence(const tvm::tir::Buffer& src,
                                          const tvm::tir::Buffer& dst,
                                          const tvm::PrimExpr& num_elements);
  tvm::tir::Stmt GenerateBinaryMaxTileSequence(const tvm::tir::Buffer& dst,
                                               const tvm::tir::Buffer& rhs);
  tvm::tir::Stmt GenerateBinaryTileSequence(const tvm::tir::Buffer& dst,
                                            const tvm::tir::Buffer& rhs,
                                            const std::string& operation_name,
                                            const tvm::Op& init_op,
                                            const tvm::Op& tile_op);
  tvm::tir::Stmt GenerateBroadcastColsBinaryTileSequence(
      const tvm::tir::Buffer& dst,
      const tvm::tir::Buffer& rhs,
      const std::string& operation_name,
      const tvm::Op& init_op,
      const tvm::Op& tile_op,
      const tvm::PrimExpr& num_elements,
      const tvm::PrimExpr& row_width);
  tvm::tir::Stmt GenerateUnaryTileSequence(const tvm::tir::Buffer& input,
                                           const tvm::tir::Buffer& output,
                                           const std::string& operation_name,
                                           const tvm::Op& init_op,
                                           const tvm::Op& tile_op);
  bool MatchDirectFragmentCast(const tvm::tir::ForNode* op, FragmentCastMatch* match) const;
  tvm::tir::Stmt GenerateFragmentCastSequence(const FragmentCastMatch& match,
                                              bool publish_cb = false,
                                              int current_order_index = -1);
  void LoadBufferFlowFacts(const BlackholeLoweringSupportFacts& lowering_support_facts);
  std::vector<std::string> CollectBufferFlowIdentities(const tvm::tir::Buffer& buffer) const;
  bool HasInterveningBufferWrite(const tvm::tir::Buffer& buffer,
                                 int live_order_index,
                                 int current_order_index) const;
  int ResolveCurrentBufferTransferOrder(const tvm::tir::Buffer& src,
                                        const tvm::tir::Buffer& dst,
                                        int lower_bound_order_index) const;
  FutureBufferUses ClassifyFutureBufferUses(const tvm::tir::Buffer& buffer,
                                            int current_order_index) const;
  FutureBufferUses ClassifyFutureLiveCBReadsBeforeNextWrite(
      const tvm::tir::Buffer& buffer,
      int current_order_index) const;
  const BlackholeBufferMaterializationFact* FindBufferMaterializationFact(
      const tvm::tir::Buffer& buffer) const;
  bool BufferUsesTiledCBLiveForm(const tvm::tir::Buffer& buffer) const;
  void ValidatePublishedBufferSourceEdge(const tvm::tir::Buffer& src,
                                         const tvm::tir::Buffer& dst) const;
  void AppendPublishedBufferSourceMaterialization(const tvm::tir::Buffer& src,
                                                  int current_order_index,
                                                  std::vector<tvm::tir::Stmt>* prefix,
                                                  std::vector<tvm::tir::Stmt>* suffix);
  void RecordFragmentCastMaterializationPlans(
      const FragmentCastMatch& match,
      const BlackholeBufferMaterializationFact& fact,
      int cb_requirement_index, const tvm::PrimExpr& num_elements_expr,
      const std::string& publication_protocol);
  void RecordTiledCBLiveFormAliases(const tvm::tir::Buffer& buffer, int cb_id);
  void ClearTiledCBLiveFormAliases(const tvm::tir::Buffer& buffer);
  void ClearTiledCBLiveFormIdentity(const std::string& identity);
  void InvalidateLastFragmentFillValue(const tvm::tir::Buffer& buffer);
  void ClearSelectedSourceLiveProducer(const tvm::tir::Buffer& buffer);
  void RecordSelectedSourceLiveProducer(const tvm::tir::Buffer& buffer);
  bool HasSelectedSourceLiveProducer(const tvm::tir::Buffer& buffer) const;
  void FinalizeConsumerBindingABIIndices();
  void FinalizeMaterializationPlanHostBuffers();
  bool ShouldRetainComputeInputBuffer(const tvm::tir::Buffer& buffer,
                                      int current_order_index) const;
  bool ShouldReacquireComputeInputBuffer(const tvm::tir::Buffer& buffer,
                                         int current_order_index) const;
  bool ShouldPublishBufferResult(const tvm::tir::Buffer& buffer,
                                 int current_order_index) const;
  bool MatchDirectLocalToCBSliceLoop(const tvm::tir::ForNode* op, LocalToCBSliceMatch* match) const;
  tvm::tir::Stmt GenerateLocalToCBSliceLoopSequence(const tvm::tir::ForNode* op,
                                                    const LocalToCBSliceMatch& match);
  std::string ResolveHostBufferForComputeOperand(const tvm::tir::Buffer& buffer) const;
  std::string ComputeKernelNameForCurrentPlan() const;
  void RecordExactComputeOpPlan(const std::string& kind,
                                const std::string& operation_name,
                                const std::vector<ComputeOperandPlanSeed>& operands);

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
  bool requires_compute_segment_ = false;
  bool select_compute_builtins_only_ = false;
  tvm::tir::Buffer copy_input_buffer_;
  tvm::tir::Buffer copy_output_buffer_;
  std::string copy_input_buffer_name_;
  std::string copy_output_buffer_name_;
  std::unordered_map<std::string, std::string> host_buffer_by_compute_operand_buffer_;

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
  std::unordered_set<std::string> compute_op_signatures_;
  std::unordered_map<std::string, int> gemm_compute_op_fact_index_by_signature_;
  std::vector<GemmComputeOpFact> gemm_compute_op_facts_;
  std::vector<std::unordered_set<std::string>> gemm_compute_op_known_buffers_;
  std::unordered_map<std::string, tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>
      logical_tile_layout_specs_by_buffer_;
  std::unordered_map<std::string, BlackholeBufferMaterializationFact>
      buffer_materialization_facts_by_target_buffer_;
  std::unordered_map<const tvm::tir::VarNode*, tvm::tir::Buffer> compute_physical_buffers_by_data_;
  std::unordered_map<std::string, tvm::tir::Buffer> compute_physical_buffers_by_identity_;
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
  std::unordered_map<const tvm::tir::VarNode*, int64_t> loop_var_static_extents_;
  std::unordered_set<const tvm::tir::VarNode*> block_index_vars_;
  std::unordered_set<std::string> block_index_var_names_;
  std::vector<AccessorDescriptor> accessor_descriptors_;
  std::string current_segment_kind_;
  std::unordered_map<std::string, int> read_accessor_slots_;
  std::unordered_map<std::string, int> write_accessor_slots_;
  std::unordered_map<std::string, int> cb_consumed_compute_input_pages_by_buffer_identity_;
  std::unordered_map<std::string, int> cb_consumed_compute_input_use_count_by_buffer_identity_;
  std::unordered_map<std::string, BlackholeBufferFlowFact> buffer_flow_facts_;
  std::unordered_map<std::string, int> buffer_live_form_cb_by_buffer_identity_;
  std::unordered_map<std::string, int> buffer_live_form_order_by_buffer_identity_;
  std::unordered_map<std::string, int> exact_output_live_form_cb_by_buffer_identity_;
  std::unordered_map<std::string, int> exact_output_live_form_order_by_buffer_identity_;
  std::unordered_set<std::string> selected_source_live_producer_buffers_;
  std::unordered_set<std::string> seeded_cb_requirement_names_;
  std::vector<SpatialMaterializationBoundaryRef> spatial_materialization_boundaries_;
  std::unordered_map<int64_t, size_t> spatial_materialization_boundary_position_by_index_;
  std::unordered_map<std::string, tvm::PrimExpr> last_fragment_fill_value_by_buffer_identity_;
  std::unordered_map<const tvm::tir::VarNode*, tvm::PrimExpr> last_fragment_fill_value_by_data_;
  std::unordered_map<std::string, std::vector<int64_t>> logical_buffer_shapes_;
  std::unordered_map<const Object*, int> stmt_order_index_by_node_;
  int current_lowering_order_index_ = -1;
  tvm::ffi::Array<tvm::ffi::Any> segment_plan_;
  tvm::ffi::Array<TTKernel> tt_kernels_;
  tvm::ffi::Array<TTABIPlan> tt_abi_plans_;
  tvm::ffi::Array<TTComputeOpPlan> tt_compute_op_plans_;
  tvm::ffi::Array<TTLiveFormPlan> tt_live_form_plans_;
  tvm::ffi::Array<TTMaterializationPlan> tt_materialization_plans_;
  tvm::ffi::Array<TTConsumerBindingPlan> tt_consumer_binding_plans_;
  std::vector<TileComputeDAGLoweringDecision> tile_compute_dag_lowering_decisions_;
  std::vector<bool> tile_compute_dag_lowering_decision_consumed_;
  std::optional<TileComputeDAGLoweringDecision> active_tile_compute_dag_lowering_decision_;

  // Requirement index counter (sequential, 0-based)
  int next_requirement_index_ = 0;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LOWER_BLACKHOLE_OPS_H_
