/*!
 * \file spatial_vocab.cc
 * \brief Closed spatial vocabulary parse/print helpers.
 */

#include "spatial_vocab.h"

#include <tvm/runtime/logging.h>

namespace tvm {
namespace tl {
namespace spatial {

std::optional<SpatialTaskKind> ParseSpatialTaskKind(const std::string& value) {
  if (value == "transfer") return SpatialTaskKind::kTransfer;
  if (value == "compute") return SpatialTaskKind::kCompute;
  if (value == "collective") return SpatialTaskKind::kCollective;
  if (value == "control") return SpatialTaskKind::kControl;
  return std::nullopt;
}

std::optional<SpatialChannelKind> ParseSpatialChannelKind(const std::string& value) {
  if (value == "point_to_point") return SpatialChannelKind::kPointToPoint;
  if (value == "broadcast") return SpatialChannelKind::kBroadcast;
  if (value == "gather") return SpatialChannelKind::kGather;
  if (value == "scatter") return SpatialChannelKind::kScatter;
  if (value == "reduce_merge") return SpatialChannelKind::kReduceMerge;
  if (value == "carry") return SpatialChannelKind::kCarry;
  return std::nullopt;
}

std::optional<SpatialChannelPayloadKind> ParseSpatialChannelPayloadKind(const std::string& value) {
  if (value == "tensor") return SpatialChannelPayloadKind::kTensor;
  if (value == "state_version") return SpatialChannelPayloadKind::kStateVersion;
  if (value == "index") return SpatialChannelPayloadKind::kIndex;
  if (value == "predicate") return SpatialChannelPayloadKind::kPredicate;
  if (value == "token") return SpatialChannelPayloadKind::kToken;
  return std::nullopt;
}

std::optional<SpatialChannelDeliveryKind> ParseSpatialChannelDeliveryKind(
    const std::string& value) {
  if (value == "ordered") return SpatialChannelDeliveryKind::kOrdered;
  if (value == "completion_visible") return SpatialChannelDeliveryKind::kCompletionVisible;
  if (value == "buffered_async") return SpatialChannelDeliveryKind::kBufferedAsync;
  if (value == "phase_boundary_materialized") {
    return SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized;
  }
  return std::nullopt;
}

std::optional<SpatialLayoutKind> ParseSpatialLayoutKind(const std::string& value) {
  if (value == "regular") return SpatialLayoutKind::kRegular;
  if (value == "packed") return SpatialLayoutKind::kPacked;
  if (value == "indexed") return SpatialLayoutKind::kIndexed;
  return std::nullopt;
}

std::optional<SpatialPartitionKind> ParseSpatialPartitionKind(const std::string& value) {
  if (value == "replicated") return SpatialPartitionKind::kReplicated;
  if (value == "blocked") return SpatialPartitionKind::kBlocked;
  if (value == "indexed") return SpatialPartitionKind::kIndexed;
  if (value == "filtered") return SpatialPartitionKind::kFiltered;
  if (value == "grouped") return SpatialPartitionKind::kGrouped;
  if (value == "routed") return SpatialPartitionKind::kRouted;
  if (value == "paged") return SpatialPartitionKind::kPaged;
  if (value == "chunked") return SpatialPartitionKind::kChunked;
  return std::nullopt;
}

std::optional<SpatialPlacementKind> ParseSpatialPlacementKind(const std::string& value) {
  if (value == "execution") return SpatialPlacementKind::kExecution;
  if (value == "communication") return SpatialPlacementKind::kCommunication;
  if (value == "phase_boundary") return SpatialPlacementKind::kPhaseBoundary;
  return std::nullopt;
}

std::optional<SpatialSyncKind> ParseSpatialSyncKind(const std::string& value) {
  if (value == "dependency") return SpatialSyncKind::kDependency;
  if (value == "barrier") return SpatialSyncKind::kBarrier;
  if (value == "completion") return SpatialSyncKind::kCompletion;
  return std::nullopt;
}

std::optional<SpatialResourceIntentKind> ParseSpatialResourceIntentKind(
    const std::string& value) {
  if (value == "buffer") return SpatialResourceIntentKind::kBuffer;
  if (value == "state_residency") return SpatialResourceIntentKind::kStateResidency;
  if (value == "synchronization_support") {
    return SpatialResourceIntentKind::kSynchronizationSupport;
  }
  if (value == "phase_boundary_materialization") {
    return SpatialResourceIntentKind::kPhaseBoundaryMaterialization;
  }
  if (value == "lowering_support") return SpatialResourceIntentKind::kLoweringSupport;
  return std::nullopt;
}

const char* ToString(SpatialTaskKind kind) {
  switch (kind) {
    case SpatialTaskKind::kTransfer:
      return "transfer";
    case SpatialTaskKind::kCompute:
      return "compute";
    case SpatialTaskKind::kCollective:
      return "collective";
    case SpatialTaskKind::kControl:
      return "control";
  }
  LOG(FATAL) << "Unknown SpatialTaskKind";
  return "unknown";
}

const char* ToString(SpatialChannelKind kind) {
  switch (kind) {
    case SpatialChannelKind::kPointToPoint:
      return "point_to_point";
    case SpatialChannelKind::kBroadcast:
      return "broadcast";
    case SpatialChannelKind::kGather:
      return "gather";
    case SpatialChannelKind::kScatter:
      return "scatter";
    case SpatialChannelKind::kReduceMerge:
      return "reduce_merge";
    case SpatialChannelKind::kCarry:
      return "carry";
  }
  LOG(FATAL) << "Unknown SpatialChannelKind";
  return "unknown";
}

const char* ToString(SpatialChannelPayloadKind kind) {
  switch (kind) {
    case SpatialChannelPayloadKind::kTensor:
      return "tensor";
    case SpatialChannelPayloadKind::kStateVersion:
      return "state_version";
    case SpatialChannelPayloadKind::kIndex:
      return "index";
    case SpatialChannelPayloadKind::kPredicate:
      return "predicate";
    case SpatialChannelPayloadKind::kToken:
      return "token";
  }
  LOG(FATAL) << "Unknown SpatialChannelPayloadKind";
  return "unknown";
}

const char* ToString(SpatialChannelDeliveryKind kind) {
  switch (kind) {
    case SpatialChannelDeliveryKind::kOrdered:
      return "ordered";
    case SpatialChannelDeliveryKind::kCompletionVisible:
      return "completion_visible";
    case SpatialChannelDeliveryKind::kBufferedAsync:
      return "buffered_async";
    case SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized:
      return "phase_boundary_materialized";
  }
  LOG(FATAL) << "Unknown SpatialChannelDeliveryKind";
  return "unknown";
}

const char* ToString(SpatialLayoutKind kind) {
  switch (kind) {
    case SpatialLayoutKind::kRegular:
      return "regular";
    case SpatialLayoutKind::kPacked:
      return "packed";
    case SpatialLayoutKind::kIndexed:
      return "indexed";
  }
  LOG(FATAL) << "Unknown SpatialLayoutKind";
  return "unknown";
}

const char* ToString(SpatialPartitionKind kind) {
  switch (kind) {
    case SpatialPartitionKind::kReplicated:
      return "replicated";
    case SpatialPartitionKind::kBlocked:
      return "blocked";
    case SpatialPartitionKind::kIndexed:
      return "indexed";
    case SpatialPartitionKind::kFiltered:
      return "filtered";
    case SpatialPartitionKind::kGrouped:
      return "grouped";
    case SpatialPartitionKind::kRouted:
      return "routed";
    case SpatialPartitionKind::kPaged:
      return "paged";
    case SpatialPartitionKind::kChunked:
      return "chunked";
  }
  LOG(FATAL) << "Unknown SpatialPartitionKind";
  return "unknown";
}

const char* ToString(SpatialPlacementKind kind) {
  switch (kind) {
    case SpatialPlacementKind::kExecution:
      return "execution";
    case SpatialPlacementKind::kCommunication:
      return "communication";
    case SpatialPlacementKind::kPhaseBoundary:
      return "phase_boundary";
  }
  LOG(FATAL) << "Unknown SpatialPlacementKind";
  return "unknown";
}

const char* ToString(SpatialSyncKind kind) {
  switch (kind) {
    case SpatialSyncKind::kDependency:
      return "dependency";
    case SpatialSyncKind::kBarrier:
      return "barrier";
    case SpatialSyncKind::kCompletion:
      return "completion";
  }
  LOG(FATAL) << "Unknown SpatialSyncKind";
  return "unknown";
}

const char* ToString(SpatialResourceIntentKind kind) {
  switch (kind) {
    case SpatialResourceIntentKind::kBuffer:
      return "buffer";
    case SpatialResourceIntentKind::kStateResidency:
      return "state_residency";
    case SpatialResourceIntentKind::kSynchronizationSupport:
      return "synchronization_support";
    case SpatialResourceIntentKind::kPhaseBoundaryMaterialization:
      return "phase_boundary_materialization";
    case SpatialResourceIntentKind::kLoweringSupport:
      return "lowering_support";
  }
  LOG(FATAL) << "Unknown SpatialResourceIntentKind";
  return "unknown";
}

}  // namespace spatial
}  // namespace tl
}  // namespace tvm
