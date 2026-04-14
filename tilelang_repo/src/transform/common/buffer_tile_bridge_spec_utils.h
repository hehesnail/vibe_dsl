/*!
 * \file buffer_tile_bridge_spec_utils.h
 * \brief Shared helpers for tiled-CB bridge specs derived from fragment layouts.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_BUFFER_TILE_BRIDGE_SPEC_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BUFFER_TILE_BRIDGE_SPEC_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>

#include <string>

#include "../../layout/layout.h"
#include "blackhole_utils.h"
#include "companion_base.h"

namespace tvm {
namespace tl {

inline tvm::ffi::Array<tvm::ffi::Any> EncodeBridgeShape(
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

inline tvm::ffi::Optional<tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>>
TryBuildBufferTileBridgeSpec(const tvm::tir::Buffer& buffer, const Layout& layout,
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

  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> spec;
  spec.Set(tvm::ffi::String(schema_key::kBuffer), tvm::ffi::String(buffer_name));
  spec.Set(tvm::ffi::String(schema_key::kScope), tvm::ffi::String(scope));
  spec.Set(tvm::ffi::String(schema_key::kShape),
           EncodeBridgeShape(fragment.value()->InputShape()));
  spec.Set(tvm::ffi::String(schema_key::kLocalShape),
           EncodeBridgeShape(fragment.value()->OutputShape()));
  spec.Set(tvm::ffi::String(schema_key::kThreadExtent), fragment.value()->ThreadExtent());
  spec.Set(tvm::ffi::String(schema_key::kReplicateExtent), fragment.value()->ReplicateExtent());
  Layout inverse = fragment.value()->Inverse();
  spec.Set(tvm::ffi::String(schema_key::kInverseLogicalIndexVars), inverse->GetForwardVars());
  spec.Set(tvm::ffi::String(schema_key::kInverseLogicalIndexExprs), inverse->GetForwardIndex());
  return spec;
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_BUFFER_TILE_BRIDGE_SPEC_UTILS_H_
