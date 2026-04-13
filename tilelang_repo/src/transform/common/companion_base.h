/*!
 * \file companion_base.h
 * \brief Shared companion attr/schema keys and neutral companion primitives.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_COMPANION_BASE_H_
#define TVM_TL_TRANSFORM_COMMON_COMPANION_BASE_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {

namespace attr {
constexpr const char* kTLDevicePrograms = "tl.device_programs";
constexpr const char* kTLSpatialStructureFacts = "tl.spatial_structure_facts";
constexpr const char* kTLSpatialPlan = "tl.spatial_plan";
constexpr const char* kTLSpatialDomainPlan = "tl.spatial_domain_plan";
constexpr const char* kTLSpatialExecutionPlan = "tl.spatial_execution_plan";
constexpr const char* kTLSpatialProgram = "tl.spatial_program";
constexpr const char* kTLTTProgram = "tl.tt_program";
constexpr const char* kTLTTSemaphorePlans = "tl.tt_semaphore_plans";
constexpr const char* kTLSpatialCapabilityModel = "tl.spatial_capability_model";
constexpr const char* kTLTTHardwareModel = "tl.tt_hardware_model";
}  // namespace attr

namespace manifest_key {
constexpr const char* kBuffers = "buffers";
constexpr const char* kOperations = "operations";
constexpr const char* kOrderedRegions = "ordered_regions";
constexpr const char* kAnchors = "anchors";
constexpr const char* kStructuralRegions = "structural_regions";
constexpr const char* kFragmentBuffers = "fragment_buffers";
constexpr const char* kSelectionTargets = "selection_targets";
constexpr const char* kSelectionPairs = "selection_pairs";
constexpr const char* kArgReduceTargets = "arg_reduce_targets";
constexpr const char* kUpdateSources = "update_sources";
constexpr const char* kLoopCarriedState = "loop_carried_state";
constexpr const char* kRecurrenceEdges = "recurrence_edges";
constexpr const char* kRowReductions = "row_reductions";
}  // namespace manifest_key

namespace schema_key {
constexpr const char* kAffinityKind = "affinity_kind";
constexpr const char* kAnchor = "anchor";
constexpr const char* kBuffer = "buffer";
constexpr const char* kBuffers = "buffers";
constexpr const char* kCaptureStage = "capture_stage";
constexpr const char* kChannelIndices = "channel_indices";
constexpr const char* kCompanionBuffer = "companion_buffer";
constexpr const char* kCompanionTarget = "companion_target";
constexpr const char* kClosureBasis = "closure_basis";
constexpr const char* kDeliveryKind = "delivery_kind";
constexpr const char* kDirection = "direction";
constexpr const char* kDomainTransformKind = "domain_transform_kind";
constexpr const char* kDomainIndex = "domain_index";
constexpr const char* kDType = "dtype";
constexpr const char* kExecutionRole = "execution_role";
constexpr const char* kExecutionProtocol = "execution_protocol";
constexpr const char* kFormationBasis = "formation_basis";
constexpr const char* kFragmentMaterializationContracts = "fragment_materialization_contracts";
constexpr const char* kFragmentMaterializationContract = "fragment_materialization_contract";
constexpr const char* kFragmentLayoutContracts = "fragment_layout_contracts";
constexpr const char* kFragmentLayoutContract = "fragment_layout_contract";
constexpr const char* kFragmentBufferFlowContracts = "fragment_buffer_flow_contracts";
constexpr const char* kFragmentLoopCarriedState = "fragment_loop_carried_state";
constexpr const char* kFragmentOpKinds = "fragment_op_kinds";
constexpr const char* kDistributionKind = "distribution_kind";
constexpr const char* kFlowClass = "flow_class";
constexpr const char* kBridgeKind = "bridge_kind";
constexpr const char* kGranuleKind = "granule_kind";
constexpr const char* kPublishGranule = "publish_granule";
constexpr const char* kConsumeGranule = "consume_granule";
constexpr const char* kEvents = "events";
constexpr const char* kOrderIndex = "order_index";
constexpr const char* kIsInteger = "is_integer";
constexpr const char* kKind = "kind";
constexpr const char* kMaterializationKind = "materialization_kind";
constexpr const char* kMergeKind = "merge_kind";
constexpr const char* kMidBuffer = "mid_buffer";
constexpr const char* kMidBufferRef = "mid_buffer_ref";
constexpr const char* kMidShape = "mid_shape";
constexpr const char* kName = "name";
constexpr const char* kNumStages = "num_stages";
constexpr const char* kObligationKind = "obligation_kind";
constexpr const char* kOrderingKind = "ordering_kind";
constexpr const char* kOperations = "operations";
constexpr const char* kOrderedRegion = "ordered_region";
constexpr const char* kPayload = "payload";
constexpr const char* kPayloadKind = "payload_kind";
constexpr const char* kPartitionFamily = "partition_family";
constexpr const char* kPlacementDomain = "placement_domain";
constexpr const char* kPipelineStages = "pipeline_stages";
constexpr const char* kPhaseIndex = "phase_index";
constexpr const char* kPointwiseOpKinds = "pointwise_op_kinds";
constexpr const char* kRowBroadcastSources = "row_broadcast_sources";
constexpr const char* kRowReductionTargets = "row_reduction_targets";
constexpr const char* kResultLiveForm = "result_live_form";
constexpr const char* kScope = "scope";
constexpr const char* kShape = "shape";
constexpr const char* kSource = "source";
constexpr const char* kSourceBuffer = "source_buffer";
constexpr const char* kSourceBuffers = "source_buffers";
constexpr const char* kLogicalRowWidth = "logical_row_width";
constexpr const char* kLogicalElementCount = "logical_element_count";
constexpr const char* kLocalShape = "local_shape";
constexpr const char* kSourceDomainIndex = "source_domain_index";
constexpr const char* kSourceStates = "source_states";
constexpr const char* kSourceTaskIndex = "source_task_index";
constexpr const char* kSourceVersion = "source_version";
constexpr const char* kSources = "sources";
constexpr const char* kSrcBuffer = "src_buffer";
constexpr const char* kSrcBufferRef = "src_buffer_ref";
constexpr const char* kSrcShape = "src_shape";
constexpr const char* kStorageTopologyKind = "storage_topology_kind";
constexpr const char* kThreadExtent = "thread_extent";
constexpr const char* kReplicateExtent = "replicate_extent";
constexpr const char* kInverseLogicalIndexVars = "inverse_logical_index_vars";
constexpr const char* kInverseLogicalIndexExprs = "inverse_logical_index_exprs";
constexpr const char* kStateIndex = "state_index";
constexpr const char* kStageLocalBuffers = "stage_local_buffers";
constexpr const char* kTarget = "target";
constexpr const char* kTargetBuffer = "target_buffer";
constexpr const char* kTargetDomainIndex = "target_domain_index";
constexpr const char* kTargetIndex = "target_index";
constexpr const char* kTargetKind = "target_kind";
constexpr const char* kTargetTaskIndex = "target_task_index";
constexpr const char* kTargetVersion = "target_version";
constexpr const char* kTaskIndex = "task_index";
constexpr const char* kTaskIndices = "task_indices";
constexpr const char* kValue = "value";
constexpr const char* kValueRole = "value_role";
constexpr const char* kValueBuffer = "value_buffer";
constexpr const char* kValueTarget = "value_target";
constexpr const char* kDstBuffer = "dst_buffer";
constexpr const char* kDstBufferRef = "dst_buffer_ref";
constexpr const char* kDstShape = "dst_shape";
constexpr const char* kLoopVar = "loop_var";
constexpr const char* kLoopCarriedState = "loop_carried_state";
constexpr const char* kWorkDependentLoopBounds = "work_dependent_loop_bounds";
}  // namespace schema_key

namespace fragment_flow {
constexpr const char* kState = "state";
constexpr const char* kStream = "stream";
constexpr const char* kRepublish = "republish";

constexpr const char* kLogicalTile = "logical_tile";

constexpr const char* kWrite = "write";
constexpr const char* kComputeConsume = "compute_consume";
constexpr const char* kTransportConsume = "transport_consume";
constexpr const char* kReference = "reference";
}  // namespace fragment_flow

namespace fragment_materialization {
constexpr const char* kIntermediateFragmentMerge = "intermediate_fragment_merge";
constexpr const char* kRepublishedLogicalTile = "republished_logical_tile";

constexpr const char* kIntermediateBuffer = "intermediate_buffer";
constexpr const char* kRepublishedBuffer = "republished_buffer";

constexpr const char* kTileNFacesMaterialization = "tile_nfaces_materialization";

constexpr const char* kFragmentDelta = "fragment_delta";
constexpr const char* kConsumerInput = "consumer_input";

constexpr const char* kFragmentAdd = "fragment_add";
constexpr const char* kDirectWrite = "direct_write";

constexpr const char* kDstCbBinaryPack = "dst_cb_binary_pack";
constexpr const char* kTiledCBRepublish = "tiled_cb_republish";
}  // namespace fragment_materialization

namespace fragment_live_form {
constexpr const char* kTiledCB = "tiled_cb";
constexpr const char* kLocalFragment = "local_fragment";
}  // namespace fragment_live_form

namespace fragment_layout {
constexpr const char* kLinear = "linear";
constexpr const char* kGroupedRows = "grouped_rows";
constexpr const char* kRowState = "row_state";
constexpr const char* kThreadDistributed = "thread_distributed";
}  // namespace fragment_layout

namespace spatial_contract {
constexpr const char* kTaskTarget = "task";
constexpr const char* kMemberFuncTarget = "member_func";
constexpr const char* kBufferTarget = "buffer";
}  // namespace spatial_contract

class TIRAnchorNode : public Object {
 public:
  ffi::String kind;
  ffi::String value_repr;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRAnchorNode>()
        .def_ro("kind", &TIRAnchorNode::kind)
        .def_ro("value_repr", &TIRAnchorNode::value_repr);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TIRAnchor", TIRAnchorNode, Object);
};

class TIRAnchor : public ObjectRef {
 public:
  TVM_DLL TIRAnchor(ffi::String kind, ffi::String value_repr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TIRAnchor, ObjectRef, TIRAnchorNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_COMPANION_BASE_H_
