/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file builtin_blackhole.h
 * \brief Blackhole (TT-Metal) specific builtin functions.
 */
#ifndef TL_TIR_BUILTIN_BLACKHOLE_H_
#define TL_TIR_BUILTIN_BLACKHOLE_H_

#include <tvm/ir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tir {
namespace builtin {

// TT-Metal Circular Buffer Operations

/*!
 * \brief Reserve space in circular buffer (back)
 * \param cb_id Circular buffer ID
 * \param num_tiles Number of tiles to reserve
 */
TVM_DLL const Op& blackhole_cb_reserve_back();

/*!
 * \brief Push tiles to circular buffer (back)
 * \param cb_id Circular buffer ID
 * \param num_tiles Number of tiles to push
 */
TVM_DLL const Op& blackhole_cb_push_back();

/*!
 * \brief Wait for tiles in circular buffer (front)
 * \param cb_id Circular buffer ID
 * \param num_tiles Number of tiles to wait for
 */
TVM_DLL const Op& blackhole_cb_wait_front();

/*!
 * \brief Pop tiles from circular buffer (front)
 * \param cb_id Circular buffer ID
 * \param num_tiles Number of tiles to pop
 */
TVM_DLL const Op& blackhole_cb_pop_front();

// TT-Metal NOC Operations

/*!
 * \brief Async read from DRAM to L1 via NOC
 * \param src_addr Source address in DRAM
 * \param dst_addr Destination address in L1
 * \param size Size in bytes
 */
TVM_DLL const Op& blackhole_noc_async_read();

/*!
 * \brief Async write from L1 to DRAM via NOC
 * \param src_addr Source address in L1
 * \param dst_addr Destination address in DRAM
 * \param size Size in bytes
 */
TVM_DLL const Op& blackhole_noc_async_write();

/*!
 * \brief Wait for NOC async read to complete
 */
TVM_DLL const Op& blackhole_noc_async_read_barrier();

/*!
 * \brief Wait for NOC async write to complete
 */
TVM_DLL const Op& blackhole_noc_async_write_barrier();

/*!
 * \brief Read one tile from a backing buffer into a CB/L1 staging area.
 * \param buffer Backing buffer handle
 * \param tile_index Logical tile index in the source buffer
 * \param cb_id Destination circular buffer ID
 * \param tile_bytes Tile size in bytes
 * \param accessor_slot Compile-time accessor slot for later TT-Metal mapping
 */
TVM_DLL const Op& blackhole_read_tile_to_cb();

/*!
 * \brief Read one contiguous page/stick from a backing buffer into a CB/L1 staging area.
 * \param buffer Backing buffer handle
 * \param page_id Logical page index in the source buffer
 * \param cb_id Destination circular buffer ID
 * \param page_bytes Page size in bytes
 * \param accessor_slot Compile-time accessor slot for later TT-Metal mapping
 * \param cb_offset_bytes Byte offset within the current CB page
 */
TVM_DLL const Op& blackhole_read_page_to_cb();

/*!
 * \brief Write one tile from a CB/L1 staging area back to a backing buffer.
 * \param cb_id Source circular buffer ID
 * \param buffer Backing buffer handle
 * \param tile_index Logical tile index in the destination buffer
 * \param tile_bytes Tile size in bytes
 * \param accessor_slot Compile-time accessor slot for later TT-Metal mapping
 */
TVM_DLL const Op& blackhole_write_tile_from_cb();

/*!
 * \brief Write one contiguous page/stick from a CB/L1 staging area back to a backing buffer.
 * \param cb_id Source circular buffer ID
 * \param buffer Backing buffer handle
 * \param page_id Logical page index in the destination buffer
 * \param page_bytes Page size in bytes
 * \param accessor_slot Compile-time accessor slot for later TT-Metal mapping
 * \param cb_offset_bytes Byte offset within the current CB page
 */
TVM_DLL const Op& blackhole_write_page_from_cb();

// TT-Metal Semaphore Operations

/*!
 * \brief Return the local L1 address of a program-local semaphore.
 * \param semaphore_id Program-local semaphore id.
 */
TVM_DLL const Op& blackhole_get_semaphore();
TVM_DLL const Op& blackhole_runtime_arg_u32();

/*!
 * \brief Wait until a local semaphore reaches the requested value.
 * \param semaphore_addr Local L1 semaphore address.
 * \param value Target value.
 */
TVM_DLL const Op& blackhole_semaphore_wait();

/*!
 * \brief Set a local semaphore to a specific value.
 * \param semaphore_addr Local L1 semaphore address.
 * \param value Value to store.
 */
TVM_DLL const Op& blackhole_semaphore_set();

/*!
 * \brief Atomically increment a remote worker semaphore.
 * \param remote_l1_addr Destination core's local semaphore address/offset.
 * \param remote_core_x Destination worker core x coordinate.
 * \param remote_core_y Destination worker core y coordinate.
 * \param value Increment amount.
 */
TVM_DLL const Op& blackhole_semaphore_inc_remote();

/*!
 * \brief Set a remote worker semaphore by sending a local L1 32-bit value to the
 *        destination core's semaphore address.
 * \param src_local_l1_addr Local L1 address containing the value to forward.
 * \param remote_core_x Destination worker core x coordinate.
 * \param remote_core_y Destination worker core y coordinate.
 * \param remote_l1_addr Destination core's local semaphore address/offset.
 */
TVM_DLL const Op& blackhole_semaphore_set_remote();

// TT-Metal Compute Operations

/*!
 * \brief Initialize matrix multiplication engine
 * \param in0_cb_id Input CB 0 (A matrix)
 * \param in1_cb_id Input CB 1 (B matrix)
 * \param out_cb_id Output CB (C matrix)
 */
TVM_DLL const Op& blackhole_mm_init();

/*!
 * \brief Perform tile-wise matrix multiplication
 * \param in0_cb_id Input CB 0 (A matrix)
 * \param in1_cb_id Input CB 1 (B matrix)
 * \param in0_tile_index Tile index in CB 0
 * \param in1_tile_index Tile index in CB 1
 * \param dst_tile_index Destination tile index
 */
TVM_DLL const Op& blackhole_matmul_tiles();

/*!
 * \brief Acquire destination registers for compute
 */
TVM_DLL const Op& blackhole_tile_regs_acquire();

/*!
 * \brief Commit compute results to destination registers
 */
TVM_DLL const Op& blackhole_tile_regs_commit();

/*!
 * \brief Wait for compute results in destination registers
 */
TVM_DLL const Op& blackhole_tile_regs_wait();

/*!
 * \brief Release destination registers
 */
TVM_DLL const Op& blackhole_tile_regs_release();

/*!
 * \brief Pack tile from destination register to CB
 * \param src_tile_index Source tile index in DST
 * \param dst_cb_id Destination CB
 */
TVM_DLL const Op& blackhole_pack_tile();

/*!
 * \brief Copy a contiguous local fragment slice into the currently reserved CB write window.
 * \param src_buffer Source local fragment buffer handle
 * \param dst_cb_id Destination CB id
 * \param dst_offset_elements Element offset from the beginning of the reserved CB write window
 * \param num_elements Number of contiguous elements to copy
 */
TVM_DLL const Op& blackhole_write_local_slice_to_cb();

/*!
 * \brief Reduce a contiguous 1-D local fragment row into a scalar local fragment target.
 * \param src_buffer Source local fragment buffer handle
 * \param dst_buffer Destination scalar local fragment buffer handle
 * \param num_elements Number of contiguous elements to reduce
 * \param reduce_kind Reduction kind string ("sum" / "max")
 * \param clear Whether to clear destination before reduction
 */
TVM_DLL const Op& blackhole_reduce_row();

/*!
 * \brief Multiply a contiguous 1-D local fragment row by a scalar local fragment source.
 * \param dst_buffer Destination/source vector local fragment buffer handle
 * \param scalar_buffer Source scalar local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 */
TVM_DLL const Op& blackhole_mul_row_bcast();
TVM_DLL const Op& blackhole_mul_grouped_row_bcast();

/*!
 * \brief Divide a contiguous 1-D local fragment row by a scalar local fragment source.
 * \param dst_buffer Destination/source vector local fragment buffer handle
 * \param scalar_buffer Source scalar local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 */
TVM_DLL const Op& blackhole_div_row_bcast();
TVM_DLL const Op& blackhole_div_grouped_row_bcast();

/*!
 * \brief Fused scalar fragment update: dst = lhs * rhs + addend.
 * \param dst_buffer Destination scalar local fragment buffer handle
 * \param lhs_buffer Left multiplicand scalar local fragment buffer handle
 * \param rhs_buffer Right multiplicand scalar local fragment buffer handle
 * \param add_buffer Addend scalar local fragment buffer handle
 */
TVM_DLL const Op& blackhole_scalar_fma();

/*!
 * \brief Fused vector row-broadcast update:
 *        dst[i] = exp2(dst[i] * dst_scale + scalar * scalar_scale).
 * \param dst_buffer Destination/source vector local fragment buffer handle
 * \param scalar_buffer Source scalar local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 * \param dst_scale Scale applied to the vector source term
 * \param scalar_scale Scale applied to the scalar broadcast term
 */
TVM_DLL const Op& blackhole_exp2_row_bcast_affine();
TVM_DLL const Op& blackhole_exp2_grouped_row_bcast_affine();

/*!
 * \brief Fused scalar fragment update:
 *        dst = exp2(lhs * lhs_scale + rhs * rhs_scale).
 * \param dst_buffer Destination scalar local fragment buffer handle
 * \param lhs_buffer Left scalar local fragment buffer handle
 * \param rhs_buffer Right scalar local fragment buffer handle
 * \param lhs_scale Scale applied to lhs
 * \param rhs_scale Scale applied to rhs
 */
TVM_DLL const Op& blackhole_scalar_exp2_affine();

/*!
 * \brief Fill a local fragment buffer with a scalar literal value.
 * \param dst_buffer Destination local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 * \param value Scalar literal fill value
 */
TVM_DLL const Op& blackhole_fill_fragment();

/*!
 * \brief Update a scalar fragment buffer in-place with max(dst, src).
 * \param dst_buffer Destination scalar local fragment buffer handle
 * \param src_buffer Source scalar local fragment buffer handle
 */
TVM_DLL const Op& blackhole_scalar_max();

/*!
 * \brief Cast a contiguous slice from one local fragment buffer into another.
 * \param dst_buffer Destination local fragment buffer handle
 * \param src_buffer Source local fragment buffer handle
 * \param dst_offset Destination element offset
 * \param src_offset Source element offset
 * \param num_elements Number of contiguous elements to cast
 */
TVM_DLL const Op& blackhole_cast_fragment_slice();

}  // namespace builtin
}  // namespace tir
}  // namespace tvm

#endif  // TL_TIR_BUILTIN_BLACKHOLE_H_
