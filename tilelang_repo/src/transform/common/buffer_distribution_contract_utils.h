/*!
 * \file buffer_distribution_contract_utils.h
 * \brief Shared helpers for typed buffer-distribution contracts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BUFFER_DISTRIBUTION_CONTRACT_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BUFFER_DISTRIBUTION_CONTRACT_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>

#include <string>

#include "../../layout/layout.h"
#include "blackhole_utils.h"
#include "companion_base.h"

namespace tvm {
namespace tl {

inline tvm::ffi::Array<tvm::ffi::Any> EncodeBufferDistributionShape(
    const tvm::ffi::Array<tvm::PrimExpr>& shape) {
  tvm::ffi::Array<tvm::ffi::Any> encoded;
  for (const tvm::PrimExpr& dim : shape) {
    if (const auto* int_imm = dim.as<tvm::tir::IntImmNode>()) {
      encoded.push_back(tvm::Integer(int_imm->value));
    } else {
      encoded.push_back(dim);
    }
  }
  return encoded;
}

inline tvm::ffi::Array<tvm::ffi::Any> EncodeLogicalDistributionShape(
    const tvm::ffi::Array<tvm::PrimExpr>& shape,
    const std::string& distribution_kind) {
  if (distribution_kind == buffer_distribution_kind::kGroupedRows && !shape.empty()) {
    tvm::ffi::Array<tvm::PrimExpr> row_shape;
    row_shape.push_back(shape[shape.size() - 1]);
    return EncodeBufferDistributionShape(row_shape);
  }
  if (distribution_kind == buffer_distribution_kind::kRowState) {
    tvm::ffi::Array<tvm::PrimExpr> row_state_shape;
    row_state_shape.push_back(tvm::IntImm(tvm::DataType::Int(32), 1));
    return EncodeBufferDistributionShape(row_state_shape);
  }
  return EncodeBufferDistributionShape(shape);
}

inline bool MatchRowStateDistribution(const Fragment& fragment) {
  return fragment->InputShape().size() == 1 && fragment->GetForwardIndex().size() == 1 &&
         StructuralEqual()(fragment->GetForwardIndex()[0],
                           tvm::IntImm(DataType::Int(32), 0)) &&
         StructuralEqual()(fragment->GetForwardThread(), InputPlaceholder(0));
}

inline bool MatchGroupedRowsDistribution(const Fragment& fragment) {
  return fragment->InputShape().size() == 2 && fragment->GetForwardIndex().size() == 1 &&
         StructuralEqual()(fragment->GetForwardIndex()[0], InputPlaceholder(1)) &&
         StructuralEqual()(fragment->GetForwardThread(), InputPlaceholder(0));
}

inline tvm::ffi::Optional<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>
TryBuildBufferDistributionContract(const tvm::tir::Buffer& buffer, const Layout& layout,
                                   const std::string& scope_override = std::string()) {
  auto fragment = layout.as<Fragment>();
  if (!fragment.has_value()) {
    return tvm::ffi::Optional<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>();
  }
  const std::string buffer_name = BufferIdentityName(buffer);
  const std::string scope =
      scope_override.empty() ? static_cast<std::string>(buffer.scope()) : scope_override;
  if (buffer_name.empty() || scope.empty()) {
    return tvm::ffi::Optional<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>();
  }

  std::string distribution_kind;
  if (MatchGroupedRowsDistribution(fragment.value())) {
    distribution_kind = buffer_distribution_kind::kGroupedRows;
  } else if (MatchRowStateDistribution(fragment.value())) {
    distribution_kind = buffer_distribution_kind::kRowState;
  } else {
    distribution_kind = buffer_distribution_kind::kThreadDistributed;
  }

  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> contract;
  contract.Set(tvm::ffi::String(schema_key::kBuffer), tvm::ffi::String(buffer_name));
  contract.Set(tvm::ffi::String(schema_key::kScope), tvm::ffi::String(scope));
  contract.Set(tvm::ffi::String(schema_key::kShape),
               EncodeLogicalDistributionShape(fragment.value()->InputShape(), distribution_kind));
  contract.Set(tvm::ffi::String(schema_key::kLocalShape),
               EncodeBufferDistributionShape(fragment.value()->OutputShape()));
  contract.Set(tvm::ffi::String(schema_key::kDistributionKind),
               tvm::ffi::String(distribution_kind));
  contract.Set(tvm::ffi::String(schema_key::kStorageTopologyKind),
               tvm::ffi::String(buffer_topology_kind::kLinear));
  contract.Set(tvm::ffi::String(schema_key::kThreadExtent), fragment.value()->ThreadExtent());
  contract.Set(tvm::ffi::String(schema_key::kReplicateExtent), fragment.value()->ReplicateExtent());
  Layout inverse = fragment.value()->Inverse();
  contract.Set(tvm::ffi::String(schema_key::kInverseLogicalIndexVars), inverse->GetForwardVars());
  contract.Set(tvm::ffi::String(schema_key::kInverseLogicalIndexExprs), inverse->GetForwardIndex());
  return contract;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BUFFER_DISTRIBUTION_CONTRACT_UTILS_H_
