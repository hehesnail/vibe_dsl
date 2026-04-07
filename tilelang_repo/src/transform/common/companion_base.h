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
constexpr const char* kTLSemanticSeeds = "tl.semantic_seeds";
constexpr const char* kTLSemanticManifestSeeds = "tl.semantic_manifest_seeds";
constexpr const char* kTLSemanticManifest = "tl.semantic_manifest";
constexpr const char* kTLSemanticHardFreeze = "tl.semantic_hard_freeze";
constexpr const char* kTLSemanticStructure = "tl.semantic_structure";
constexpr const char* kTLSemanticWitnesses = "tl.semantic_witnesses";
constexpr const char* kTLSemanticProgram = "tl.semantic_program";
constexpr const char* kTLSpatialDomainPlan = "tl.spatial_domain_plan";
constexpr const char* kTLSpatialExecutionPlan = "tl.spatial_execution_plan";
constexpr const char* kTLSpatialProgram = "tl.spatial_program";
constexpr const char* kTLTTProgram = "tl.tt_program";
constexpr const char* kTLSpatialCapabilityModel = "tl.spatial_capability_model";
constexpr const char* kTLTTHardwareModel = "tl.tt_hardware_model";
constexpr const char* kTLCompanionInvalidationReason = "tl.companion_invalidation_reason";
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
constexpr const char* kFormationBasis = "formation_basis";
constexpr const char* kFragmentLoopCarriedState = "fragment_loop_carried_state";
constexpr const char* kFragmentOpKinds = "fragment_op_kinds";
constexpr const char* kIsInteger = "is_integer";
constexpr const char* kKind = "kind";
constexpr const char* kMaterializationKind = "materialization_kind";
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
constexpr const char* kScope = "scope";
constexpr const char* kShape = "shape";
constexpr const char* kSource = "source";
constexpr const char* kSourceBuffers = "source_buffers";
constexpr const char* kSourceDomainIndex = "source_domain_index";
constexpr const char* kSourceStates = "source_states";
constexpr const char* kSourceTaskIndex = "source_task_index";
constexpr const char* kSourceVersion = "source_version";
constexpr const char* kSources = "sources";
constexpr const char* kSrcBuffer = "src_buffer";
constexpr const char* kSrcBufferRef = "src_buffer_ref";
constexpr const char* kSrcShape = "src_shape";
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
constexpr const char* kValueBuffer = "value_buffer";
constexpr const char* kValueTarget = "value_target";
constexpr const char* kDstBuffer = "dst_buffer";
constexpr const char* kDstBufferRef = "dst_buffer_ref";
constexpr const char* kDstShape = "dst_shape";
constexpr const char* kLoopVar = "loop_var";
constexpr const char* kLoopCarriedState = "loop_carried_state";
constexpr const char* kWorkDependentLoopBounds = "work_dependent_loop_bounds";
}  // namespace schema_key

namespace spatial_contract {
constexpr const char* kSemanticStateTarget = "semantic_state";
constexpr const char* kTaskTarget = "task";
constexpr const char* kMemberFuncTarget = "member_func";
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
