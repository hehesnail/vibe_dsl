/*!
 * \file spatial_analysis.h
 * \brief Shared semantic-to-spatial analysis helpers and contracts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_
#define TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_

#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "semantic_program.h"
#include "semantic_vocab.h"
#include "spatial_program.h"
#include "spatial_vocab.h"

namespace tvm {
namespace tl {

using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::Integer;
using namespace tvm::tl::semantic;
namespace sp = tvm::tl::spatial;

struct ProducerVersionEdge {
  std::string producer_update;
  std::string produced_version;
};

struct DomainRealizationContract {
  std::string domain_transform_kind = "identity";
  std::string partition_family = "regular";
  sp::SpatialLayoutKind layout_kind = sp::SpatialLayoutKind::kRegular;
  sp::SpatialPartitionKind partition_kind = sp::SpatialPartitionKind::kReplicated;
};

std::optional<int64_t> GetPayloadIndex(const Map<String, Any>& payload, const char* key);
std::optional<std::string> GetPayloadString(const Map<String, Any>& payload, const char* key);
std::optional<std::vector<int64_t>> GetPayloadIndices(const Map<String, Any>& payload,
                                                      const char* key);

Array<TIRAnchor> MakeAnchors(const std::string& kind, const std::string& value);
std::string GetMemberFuncName(const GlobalVar& gvar, const tir::PrimFunc& func);
bool ContainsKind(const Array<String>& supported_kinds, const std::string& expected);

Array<String> ToStringArray(const std::vector<std::string>& values);
Array<String> MakeTraits(std::initializer_list<const char*> values);
bool HasTrait(const Array<String>& traits, const char* trait);

bool SameStringArray(const Array<String>& lhs, const Array<String>& rhs);
bool SameIntegerAnyArray(const Array<Any>& lhs, const Array<Any>& rhs);

std::unordered_map<std::string, std::optional<StateRole>> BuildStateRoleByName(
    const SemanticProgram& program);
bool HasSupplementPayload(const SemanticProgram& program, SupplementKind kind,
                          const char* payload_key);
bool AccessMapTouchesDomain(const AccessMap& access_map, const Domain& domain);
bool UpdateTouchesDomain(const Update& update, const Domain& domain);
bool DomainHasStateRole(
    const SemanticProgram& program, const Domain& domain,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name,
    StateRole role);
bool DomainHasAccessTrait(const SemanticProgram& program, const Domain& domain, const char* trait);

std::optional<Array<Any>> GetPipelineStagesFromSupplements(const SemanticProgram& program);
std::optional<Array<Any>> GetWorkDependentLoopBoundsFromSupplements(const SemanticProgram& program);
std::optional<Map<String, Any>> GetFragmentLoweringPayloadFromSupplements(
    const SemanticProgram& program);

DomainRealizationContract DeriveDomainRealizationContract(
    const SemanticProgram& program, const Domain& domain,
    const std::unordered_map<std::string, std::optional<StateRole>>& state_role_by_name);
std::vector<DomainRealizationContract> DeriveDomainRealizationContracts(
    const SemanticProgram& program);

std::unordered_map<std::string, std::string> BuildUniqueUpdateResultVersionByUpdate(
    const Array<StateDef>& state_defs);
std::unordered_map<std::string, StateJoin> BuildStateJoinByOutputVersion(
    const Array<StateJoin>& state_joins);
std::unordered_map<std::string, int> BuildDistinctConsumerCountByVersion(
    const Array<StateUse>& state_uses);
std::unordered_set<std::string> CollectKnownUpdateNames(const SemanticProgram& program);
std::unordered_map<std::string, std::vector<ProducerVersionEdge>> BuildVersionProducerEdges(
    const SemanticProgram& program);
std::unordered_map<std::string, std::vector<ProducerVersionEdge>> BuildVersionProducerEdges(
    const SemanticProgram& program, const std::unordered_set<std::string>& allowed_updates);

std::string DeriveOrderingKindForChannel(sp::SpatialChannelKind channel_kind,
                                         sp::SpatialChannelDeliveryKind delivery_kind);
std::string DeriveMaterializationKindForChannel(sp::SpatialChannelKind channel_kind,
                                                sp::SpatialChannelDeliveryKind delivery_kind);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SPATIAL_ANALYSIS_H_
