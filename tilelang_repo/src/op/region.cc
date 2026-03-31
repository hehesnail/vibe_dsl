/*!
 * \file tl/op/region.cc
 * \brief Define region operator (bridge to carry BufferRegion via Call args).
 *
 * Notes:
 * - BufferLoad/Ramp cannot represent a general PrimExpr as a vector lane
 *   count. Dynamic extents like (H1 - H0) cannot be encoded as
 *   Ramp(lanes = H1 - H0), and lowering BufferRegion to BufferLoad loses the
 *   explicit extent information.
 * - tl.region carries both mins and extents in Call args and lets the backend
 *   reconstruct a BufferRegion faithfully.
 */

#include "region.h"
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {
using namespace tir;

RegionOp::RegionOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  size_t n = args.size();
  size_t provided_ndim = n - 2;
  auto load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  size_t load_ndim = load->indices.size();
  ICHECK(provided_ndim <= load_ndim)
      << "RegionOp expects at most one extent per load axis: load->indices.size() = "
      << load->indices << " provided_ndim = " << provided_ndim;
  Array<Range> ranges;
  size_t leading_singleton_axes = load_ndim - provided_ndim;

  // Keep unmatched leading indices as singleton axes. This lets tl.region
  // carry staged/shared views like (stage, row, col) while only providing
  // extents for the trailing tile axes.
  for (size_t i = 0; i < leading_singleton_axes; ++i) {
    ranges.push_back(Range::FromMinExtent(load->indices[i], 1));
  }

  // Rebuild the trailing region axes from mins (BufferLoad indices) and the
  // provided extents.
  for (size_t i = 0; i < provided_ndim; i++) {
    PrimExpr index = load->indices[leading_singleton_axes + i];
    PrimExpr extent = args[2 + i];
    if (const auto *ramp = index.as<RampNode>()) {
      const auto *stride_imm = ramp->stride.as<IntImmNode>();
      ICHECK(stride_imm && stride_imm->value == 1)
          << "RegionOp expects stride-1 Ramp for index";
      if (const auto *lanes_imm = ramp->lanes.as<IntImmNode>()) {
        if (const auto *ext_imm = extent.as<IntImmNode>()) {
          ICHECK_EQ(lanes_imm->value, ext_imm->value)
              << "Ramp lanes and provided extent must match";
        }
      }
      ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
    } else {
      ranges.push_back(Range::FromMinExtent(index, extent));
    }
  }
  ObjectPtr<RegionOpNode> node = tvm::ffi::make_object<RegionOpNode>();
  node->buffer_ = load->buffer;
  node->access_mask_ = static_cast<int>(*as_const_int(args[1]));
  node->ranges_ = ranges;
  data_ = std::move(node);
}

TileOperator RegionOpNode::Clone() const {
  auto op = tvm::ffi::make_object<RegionOpNode>(*this);
  return RegionOp(op);
}

bool RegionOpNode::IsFullRegion() const {
  for (size_t i = 0; i < ranges_.size(); i++) {
    if (!is_zero(ranges_[i]->min))
      return false;
    if (!StructuralEqual()(ranges_[i]->extent, buffer_->shape[i]))
      return false;
  }
  return true;
}

Stmt RegionOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return Evaluate(0);
}

LayoutMap RegionOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  return {};
}

TIR_REGISTER_TL_TILE_OP(RegionOp, region)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TVM_FFI_STATIC_INIT_BLOCK() { RegionOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
