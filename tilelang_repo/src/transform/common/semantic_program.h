/*!
 * \file semantic_program.h
 * \brief Stage 4 companion IR guardrail constants and lightweight registry objects.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_
#define TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_info.h>

namespace tvm {
namespace tl {

namespace attr {
constexpr const char* kTLDevicePrograms = "tl.device_programs";
constexpr const char* kTLSemanticSeeds = "tl.semantic_seeds";
constexpr const char* kTLSemanticHardFreeze = "tl.semantic_hard_freeze";
constexpr const char* kTLSemanticProgram = "tl.semantic_program";
constexpr const char* kTLSpatialProgram = "tl.spatial_program";
constexpr const char* kTLTTProgram = "tl.tt_program";
constexpr const char* kTLCompanionInvalidationReason = "tl.companion_invalidation_reason";
}  // namespace attr

class TLDeviceProgramInfoNode : public GlobalInfoNode {
 public:
  ffi::String root_symbol;
  ffi::Array<ffi::String> member_funcs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TLDeviceProgramInfoNode>()
        .def_ro("root_symbol", &TLDeviceProgramInfoNode::root_symbol)
        .def_ro("member_funcs", &TLDeviceProgramInfoNode::member_funcs);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TLDeviceProgramInfo", TLDeviceProgramInfoNode,
                                    GlobalInfoNode);
};

class TLDeviceProgramInfo : public GlobalInfo {
 public:
  TVM_DLL TLDeviceProgramInfo(ffi::String root_symbol, ffi::Array<ffi::String> member_funcs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TLDeviceProgramInfo, GlobalInfo,
                                             TLDeviceProgramInfoNode);
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_SEMANTIC_PROGRAM_H_
