/*!
 * \file tt_hardware_model.h
 * \brief Minimal TT hardware snapshot intake for Blackhole lowering.
 */

#ifndef TVM_TL_TRANSFORM_COMMON_TT_HARDWARE_MODEL_H_
#define TVM_TL_TRANSFORM_COMMON_TT_HARDWARE_MODEL_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include "companion_base.h"

namespace tvm {
namespace tl {

class TTHardwareModelNode : public GlobalInfoNode {
 public:
  ffi::String arch_name;
  ffi::String descriptor_path;
  int64_t logical_worker_grid_x = 0;
  int64_t logical_worker_grid_y = 0;
  int64_t functional_worker_count = 0;
  int64_t router_only_count = 0;
  int64_t dram_view_count = 0;
  int64_t worker_l1_size = 0;
  int64_t dram_view_size = 0;
  int64_t max_cb_count = 0;
  int64_t l1_allocation_alignment_bytes = 0;
  bool noc_translation_id_enabled = false;
  int64_t unpacker_version = 0;
  int64_t packer_version = 0;
  int64_t overlay_version = 0;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TTHardwareModelNode>()
        .def_ro("arch_name", &TTHardwareModelNode::arch_name)
        .def_ro("descriptor_path", &TTHardwareModelNode::descriptor_path)
        .def_ro("logical_worker_grid_x", &TTHardwareModelNode::logical_worker_grid_x)
        .def_ro("logical_worker_grid_y", &TTHardwareModelNode::logical_worker_grid_y)
        .def_ro("functional_worker_count", &TTHardwareModelNode::functional_worker_count)
        .def_ro("router_only_count", &TTHardwareModelNode::router_only_count)
        .def_ro("dram_view_count", &TTHardwareModelNode::dram_view_count)
        .def_ro("worker_l1_size", &TTHardwareModelNode::worker_l1_size)
        .def_ro("dram_view_size", &TTHardwareModelNode::dram_view_size)
        .def_ro("max_cb_count", &TTHardwareModelNode::max_cb_count)
        .def_ro("l1_allocation_alignment_bytes",
                &TTHardwareModelNode::l1_allocation_alignment_bytes)
        .def_ro("noc_translation_id_enabled",
                &TTHardwareModelNode::noc_translation_id_enabled)
        .def_ro("unpacker_version", &TTHardwareModelNode::unpacker_version)
        .def_ro("packer_version", &TTHardwareModelNode::packer_version)
        .def_ro("overlay_version", &TTHardwareModelNode::overlay_version);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.TTHardwareModel", TTHardwareModelNode, GlobalInfoNode);
};

class TTHardwareModel : public GlobalInfo {
 public:
  TVM_DLL TTHardwareModel(ffi::String arch_name, ffi::String descriptor_path,
                          int64_t logical_worker_grid_x, int64_t logical_worker_grid_y,
                          int64_t functional_worker_count, int64_t router_only_count,
                          int64_t dram_view_count, int64_t worker_l1_size,
                          int64_t dram_view_size, int64_t max_cb_count,
                          int64_t l1_allocation_alignment_bytes,
                          bool noc_translation_id_enabled,
                          int64_t unpacker_version, int64_t packer_version,
                          int64_t overlay_version);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TTHardwareModel, GlobalInfo, TTHardwareModelNode);
};

TVM_DLL TTHardwareModel BuildBlackholeTTHardwareModel(const Target& target);
TVM_DLL std::optional<TTHardwareModel> GetModuleTTHardwareModel(const IRModule& mod);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_TT_HARDWARE_MODEL_H_
