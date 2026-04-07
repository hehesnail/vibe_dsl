/*!
 * \file spatial_vocab.h
 * \brief Closed spatial vocabulary for Phase B contracts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_VOCAB_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_VOCAB_H_

#include <tvm/runtime/base.h>

#include <optional>
#include <string>

namespace tvm {
namespace tl {
namespace spatial {

enum class SpatialTaskKind {
  kTransfer,
  kCompute,
  kCollective,
  kControl,
};

enum class SpatialChannelKind {
  kPointToPoint,
  kBroadcast,
  kGather,
  kScatter,
  kReduceMerge,
  kCarry,
};

enum class SpatialChannelPayloadKind {
  kTensor,
  kStateVersion,
  kIndex,
  kPredicate,
  kToken,
};

enum class SpatialChannelDeliveryKind {
  kOrdered,
  kCompletionVisible,
  kBufferedAsync,
  kPhaseBoundaryMaterialized,
};

enum class SpatialLayoutKind {
  kRegular,
  kPacked,
  kIndexed,
};

enum class SpatialPartitionKind {
  kReplicated,
  kBlocked,
  kIndexed,
  kFiltered,
  kGrouped,
  kRouted,
  kPaged,
  kChunked,
};

enum class SpatialPlacementKind {
  kExecution,
  kCommunication,
  kPhaseBoundary,
};

enum class SpatialSyncKind {
  kDependency,
  kBarrier,
  kCompletion,
};

enum class SpatialResourceIntentKind {
  kBuffer,
  kStateResidency,
  kSynchronizationSupport,
  kPhaseBoundaryMaterialization,
  kLoweringSupport,
};

TVM_DLL std::optional<SpatialTaskKind> ParseSpatialTaskKind(const std::string& value);
TVM_DLL std::optional<SpatialChannelKind> ParseSpatialChannelKind(const std::string& value);
TVM_DLL std::optional<SpatialChannelPayloadKind> ParseSpatialChannelPayloadKind(
    const std::string& value);
TVM_DLL std::optional<SpatialChannelDeliveryKind> ParseSpatialChannelDeliveryKind(
    const std::string& value);
TVM_DLL std::optional<SpatialLayoutKind> ParseSpatialLayoutKind(const std::string& value);
TVM_DLL std::optional<SpatialPartitionKind> ParseSpatialPartitionKind(const std::string& value);
TVM_DLL std::optional<SpatialPlacementKind> ParseSpatialPlacementKind(const std::string& value);
TVM_DLL std::optional<SpatialSyncKind> ParseSpatialSyncKind(const std::string& value);
TVM_DLL std::optional<SpatialResourceIntentKind> ParseSpatialResourceIntentKind(
    const std::string& value);

TVM_DLL const char* ToString(SpatialTaskKind kind);
TVM_DLL const char* ToString(SpatialChannelKind kind);
TVM_DLL const char* ToString(SpatialChannelPayloadKind kind);
TVM_DLL const char* ToString(SpatialChannelDeliveryKind kind);
TVM_DLL const char* ToString(SpatialLayoutKind kind);
TVM_DLL const char* ToString(SpatialPartitionKind kind);
TVM_DLL const char* ToString(SpatialPlacementKind kind);
TVM_DLL const char* ToString(SpatialSyncKind kind);
TVM_DLL const char* ToString(SpatialResourceIntentKind kind);

}  // namespace spatial
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_VOCAB_H_
