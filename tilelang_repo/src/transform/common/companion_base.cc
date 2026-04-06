/*!
 * \file companion_base.cc
 * \brief Shared companion attr/schema keys and neutral companion primitives.
 */

#include "companion_base.h"

namespace tvm {
namespace tl {

TIRAnchor::TIRAnchor(ffi::String kind, ffi::String value_repr) {
  auto n = ffi::make_object<TIRAnchorNode>();
  n->kind = std::move(kind);
  n->value_repr = std::move(value_repr);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { TIRAnchorNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TIRAnchor",
                        [](ffi::String kind, ffi::String value_repr) {
                          return TIRAnchor(std::move(kind), std::move(value_repr));
                        });
}

}  // namespace tl
}  // namespace tvm
