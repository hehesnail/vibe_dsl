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
 * \brief Read one rank-1 vector page into the first column of a tiled CB page.
 * \param buffer Backing buffer handle
 * \param page_id Logical page id in the source buffer
 * \param cb_id Destination circular buffer ID
 * \param page_bytes Source page size in bytes
 * \param accessor_slot Compile-time accessor slot for later TT-Metal mapping
 * \param vector_len Number of vector elements to place into the bcast tile
 */
TVM_DLL const Op& blackhole_read_bcast_cols_to_cb();

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
 * \brief Reconfigure SrcA/SrcB data formats for a TT-Metal compute op.
 * \param in0_cb_id New SrcA CB id
 * \param in1_cb_id New SrcB CB id
 */
TVM_DLL const Op& blackhole_reconfig_data_format();

/*!
 * \brief Reconfigure the compute engine back to matmul mode after a DST reload.
 * \param in0_cb_id Input CB 0 (A matrix)
 * \param in1_cb_id Input CB 1 (B matrix)
 */
TVM_DLL const Op& blackhole_mm_init_short();

/*!
 * \brief Reconfigure the compute engine back to matmul mode after a mixed-dtype DST reload.
 * \param in0_cb_id Input CB 0 (A matrix)
 * \param in1_cb_id Input CB 1 (B matrix)
 * \param old_srca_cb_id Previous SrcA CB used during the reload step
 */
TVM_DLL const Op& blackhole_mm_init_short_with_dt();

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
 * \param dst_tile_index Optional destination tile index within the currently
 *        reserved CB write window. When omitted, TT-Metal advances the CB
 *        write-tile cursor implicitly.
 */
TVM_DLL const Op& blackhole_pack_tile();

/*!
 * \brief Reconfigure packer data format for the destination CB.
 * \param dst_cb_id Destination CB id
 */
TVM_DLL const Op& blackhole_pack_reconfig_data_format();

/*!
 * \brief Initialize TT-Metal tile reload from a CB into DST registers.
 * \param src_cb_id Source CB whose front tiles will be copied into DST
 */
TVM_DLL const Op& blackhole_copy_tile_to_dst_init_short();

/*!
 * \brief Initialize TT-Metal tile reload from a CB into DST registers with SrcA dtype reconfiguration.
 * \param old_srca_cb_id Previously configured SrcA CB
 * \param src_cb_id Source CB whose front tiles will be copied into DST
 */
TVM_DLL const Op& blackhole_copy_tile_to_dst_init_short_with_dt();

/*!
 * \brief Copy a single tile from a CB into a DST register slot.
 * \param src_cb_id Source CB
 * \param src_tile_index Tile index within the CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_copy_tile();

/*!
 * \brief Initialize TT-Metal common binary-op unpack/math/pack state.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param out_cb_id Output CB id
 */
TVM_DLL const Op& blackhole_binary_op_init_common();

/*!
 * \brief Initialize TT-Metal common unary-op/SFPU unpack/math/pack state.
 * \param input_cb_id Input CB id
 * \param out_cb_id Output CB id
 */
TVM_DLL const Op& blackhole_unary_op_init_common();

/*!
 * \brief Initialize TT-Metal elementwise add for two CB-backed tile streams.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_add_tiles_init();

/*!
 * \brief Add one tile from two CBs into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Tile index in the left-hand-side CB front window
 * \param rhs_tile_index Tile index in the right-hand-side CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_add_tiles();

/*!
 * \brief Initialize TT-Metal add broadcast-rows sequence.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_add_bcast_rows_init_short();

/*!
 * \brief Initialize TT-Metal add broadcast-cols sequence.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_add_bcast_cols_init_short();

/*!
 * \brief Add CB tiles with broadcast-rows layout into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Left-hand-side tile index in the CB front window
 * \param rhs_tile_index Right-hand-side tile index in the CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_add_tiles_bcast_rows();

/*!
 * \brief Add CB tiles with broadcast-cols layout into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Left-hand-side tile index in the CB front window
 * \param rhs_tile_index Right-hand-side tile index in the CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_add_tiles_bcast_cols();

/*!
 * \brief Initialize TT-Metal elementwise mul for two CB-backed tile streams.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_mul_tiles_init();

/*!
 * \brief Multiply one tile from two CBs into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Tile index in the left-hand-side CB front window
 * \param rhs_tile_index Tile index in the right-hand-side CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_mul_tiles();

/*!
 * \brief Initialize TT-Metal mul broadcast-rows sequence.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_mul_bcast_rows_init_short();

/*!
 * \brief Initialize TT-Metal mul broadcast-cols sequence.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 */
TVM_DLL const Op& blackhole_mul_bcast_cols_init_short();

/*!
 * \brief Multiply CB tiles with broadcast-rows layout into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Left-hand-side tile index in the CB front window
 * \param rhs_tile_index Right-hand-side tile index in the CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_mul_tiles_bcast_rows();

/*!
 * \brief Multiply CB tiles with broadcast-cols layout into a DST register slot.
 * \param lhs_cb_id Left-hand-side CB id
 * \param rhs_cb_id Right-hand-side CB id
 * \param lhs_tile_index Left-hand-side tile index in the CB front window
 * \param rhs_tile_index Right-hand-side tile index in the CB front window
 * \param dst_tile_index Destination tile index within DST registers
 */
TVM_DLL const Op& blackhole_mul_tiles_bcast_cols();

/*!
 * \brief Initialize exact TT-Metal reduction state.
 * \param src_cb_id Source CB id
 * \param scaler_cb_id Scaler CB id
 * \param dst_cb_id Destination CB id
 * \param reduce_kind Reduction kind string ("sum" / "max")
 * \param reduce_dim Reduction dim string ("row" / "col")
 */
TVM_DLL const Op& blackhole_reduce_init();

/*!
 * \brief Execute one exact TT-Metal reduction tile op into DST.
 * \param src_cb_id Source CB id
 * \param scaler_cb_id Scaler CB id
 * \param src_tile_index Source tile index in the CB front window
 * \param scaler_tile_index Scaler tile index in the scaler CB front window
 * \param dst_tile_index Destination tile index within DST registers
 * \param reduce_kind Reduction kind string ("sum" / "max")
 * \param reduce_dim Reduction dim string ("row" / "col")
 */
TVM_DLL const Op& blackhole_reduce_tile();

/*!
 * \brief Uninitialize exact TT-Metal reduction state.
 * \param reduce_kind Reduction kind string ("sum" / "max")
 * \param reduce_dim Reduction dim string ("row" / "col")
 */
TVM_DLL const Op& blackhole_reduce_uninit();

/*!
 * \brief Initialize exact TT-Metal binary max tile sequence.
 */
TVM_DLL const Op& blackhole_binary_max_tile_init();

/*!
 * \brief Execute exact TT-Metal binary max on DST tiles.
 * \param lhs_dst_tile_index Left-hand-side DST tile index
 * \param rhs_dst_tile_index Right-hand-side DST tile index
 * \param dst_tile_index Destination DST tile index
 */
TVM_DLL const Op& blackhole_binary_max_tile();

/*!
 * \brief Initialize exact TT-Metal binary divide tile sequence.
 */
TVM_DLL const Op& blackhole_div_binary_tile_init();

/*!
 * \brief Execute exact TT-Metal binary divide on DST tiles.
 * \param lhs_dst_tile_index Left-hand-side DST tile index
 * \param rhs_dst_tile_index Right-hand-side DST tile index
 * \param dst_tile_index Destination DST tile index
 */
TVM_DLL const Op& blackhole_div_binary_tile();

/*!
 * \brief Initialize exact TT-Metal exp tile sequence.
 */
TVM_DLL const Op& blackhole_exp_tile_init();

/*!
 * \brief Execute exact TT-Metal exp on one DST tile.
 * \param dst_tile_index Destination DST tile index
 */
TVM_DLL const Op& blackhole_exp_tile();

/*!
 * \brief Initialize exact TT-Metal exp2 tile sequence.
 */
TVM_DLL const Op& blackhole_exp2_tile_init();

/*!
 * \brief Execute exact TT-Metal exp2 on one DST tile.
 * \param dst_tile_index Destination DST tile index
 */
TVM_DLL const Op& blackhole_exp2_tile();

/*!
 * \brief Initialize exact TT-Metal reciprocal tile sequence.
 */
TVM_DLL const Op& blackhole_recip_tile_init();

/*!
 * \brief Execute exact TT-Metal reciprocal on one DST tile.
 * \param dst_tile_index Destination DST tile index
 */
TVM_DLL const Op& blackhole_recip_tile();

/*!
 * \brief Pack/untilize a contiguous local fragment slice into the reserved CB write window.
 * \param src_buffer Source local fragment buffer handle
 * \param dst_cb_id Destination CB id
 * \param dst_offset_elements Element offset from the beginning of the reserved CB write window
 * \param num_elements Number of contiguous elements to copy
 * \param src_offset_elements Optional element offset from the beginning of the local fragment slice
 */
TVM_DLL const Op& blackhole_pack_untilize_slice();

/*!
 * \brief Pack/untilize one 32x32 local fragment tile into tiled-nfaces layout.
 * \param src_buffer Source local fragment buffer handle
 * \param dst_cb_id Destination CB id
 * \param dst_tile_index Destination tile index in the reserved CB write window
 * \param src_offset_elements Element offset from the beginning of the local fragment tile
 */
TVM_DLL const Op& blackhole_pack_untilize_tile();

/*!
 * \brief Tilize a row-major local fragment slice into tiled-nfaces layout.
 * \param src_buffer Source local fragment buffer handle
 * \param dst_cb_id Destination CB id
 * \param dst_offset_elements Logical element offset within the destination tensor view
 * \param num_elements Number of contiguous source elements to materialize
 * \param row_width Logical row width of the destination tiled tensor view
 * \param src_offset_elements Optional source element offset from the beginning of the local fragment slice
 */
TVM_DLL const Op& blackhole_tilize_local_fragment_slice();

/*!
 * \brief Tilize a casted row-major local fragment slice into tiled-nfaces layout.
 * \param dst_buffer Destination CB-backed fragment buffer handle (dtype/source-of-truth only)
 * \param src_buffer Source local fragment buffer handle
 * \param dst_cb_id Destination CB id
 * \param dst_offset_elements Logical destination element offset within the tiled tensor view
 * \param src_offset_elements Source element offset within the local fragment
 * \param num_elements Number of contiguous source elements to cast/materialize
 * \param row_width Logical row width of the destination tiled tensor view
 */
TVM_DLL const Op& blackhole_tilize_cast_fragment_slice();

/*!
 * \brief Fill a tiled CB page directly from the compute PACK thread.
 * \param dst_buffer Destination CB-backed fragment buffer handle (dtype/source-of-truth only)
 * \param dst_cb_id Destination CB id
 * \param dst_offset_elements Logical destination element offset within the tiled tensor view
 * \param num_elements Number of logical elements to fill
 * \param row_width Logical row width of the destination tiled tensor view
 * \param value Scalar fill value
 */
TVM_DLL const Op& blackhole_pack_fill_fragment_to_tiled_cb();

/*!
 * \brief Untilize a tile from the current CB front window into a local fragment slice.
 * \param dst_buffer Destination local fragment buffer handle
 * \param src_cb_id Source CB id
 * \param src_tile_index Source tile index in the current CB front window
 * \param dst_offset_elements Element offset from the beginning of the destination local fragment
 * \param num_elements Number of contiguous elements to copy from the source tile
 */
TVM_DLL const Op& blackhole_untilize_cb_front_tile();

/*!
 * \brief Untilize one 32x32 tile from the current CB front window into row-major local fragment layout.
 * \param dst_buffer Destination local fragment buffer handle
 * \param src_cb_id Source CB id
 * \param src_tile_index Source tile index in the current CB front window
 * \param dst_offset_elements Element offset from the beginning of the destination local fragment tile
 */
TVM_DLL const Op& blackhole_untilize_cb_front_tile_fragment();

/*!
 * \brief Fill a local fragment buffer with a scalar literal value.
 * \param dst_buffer Destination local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 * \param value Scalar literal fill value
 */
TVM_DLL const Op& blackhole_fill_fragment();

/*!
 * \brief Add a contiguous local fragment slice into another local fragment buffer in-place.
 * \param dst_buffer Destination/source local fragment buffer handle
 * \param src_buffer Source local fragment buffer handle
 * \param num_elements Number of contiguous destination elements
 */
TVM_DLL const Op& blackhole_add_fragment();

/*!
 * \brief Add the front page(s) of a CB-backed scratch fragment into a local fragment buffer.
 * \param dst_buffer Destination/source local fragment buffer handle
 * \param src_cb_id Source CB id whose current front page(s) contain the packed fragment
 * \param num_elements Number of contiguous destination elements
 */
TVM_DLL const Op& blackhole_add_fragment_from_cb_front();

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
