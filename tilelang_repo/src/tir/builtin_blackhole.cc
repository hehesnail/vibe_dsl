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
 * \file builtin_blackhole.cc
 * \brief Blackhole (TT-Metal) builtin function implementations.
 */

#include "builtin_blackhole.h"

namespace tvm {
namespace tir {
namespace builtin {

// Helper macro to register a builtin
#define TIR_DEFINE_BUILTIN(NAME) \
  const Op& blackhole_##NAME() { \
    static const Op& op = Op::Get("tl.blackhole." #NAME); \
    return op; \
  }

// Circular Buffer Operations
TIR_DEFINE_BUILTIN(cb_reserve_back)
TIR_DEFINE_BUILTIN(cb_push_back)
TIR_DEFINE_BUILTIN(cb_wait_front)
TIR_DEFINE_BUILTIN(cb_pop_front)

// NOC Operations
TIR_DEFINE_BUILTIN(noc_async_read)
TIR_DEFINE_BUILTIN(noc_async_write)
TIR_DEFINE_BUILTIN(noc_async_read_barrier)
TIR_DEFINE_BUILTIN(noc_async_write_barrier)
TIR_DEFINE_BUILTIN(read_tile_to_cb)
TIR_DEFINE_BUILTIN(read_page_to_cb)
TIR_DEFINE_BUILTIN(read_bcast_cols_to_cb)
TIR_DEFINE_BUILTIN(write_tile_from_cb)
TIR_DEFINE_BUILTIN(write_page_from_cb)
TIR_DEFINE_BUILTIN(get_semaphore)
TIR_DEFINE_BUILTIN(runtime_arg_u32)
TIR_DEFINE_BUILTIN(semaphore_wait)
TIR_DEFINE_BUILTIN(semaphore_set)
TIR_DEFINE_BUILTIN(semaphore_inc_remote)
TIR_DEFINE_BUILTIN(semaphore_set_remote)

// Compute Operations
TIR_DEFINE_BUILTIN(mm_init)
TIR_DEFINE_BUILTIN(reconfig_data_format)
TIR_DEFINE_BUILTIN(mm_init_short)
TIR_DEFINE_BUILTIN(mm_init_short_with_dt)
TIR_DEFINE_BUILTIN(matmul_tiles)
TIR_DEFINE_BUILTIN(tile_regs_acquire)
TIR_DEFINE_BUILTIN(tile_regs_commit)
TIR_DEFINE_BUILTIN(tile_regs_wait)
TIR_DEFINE_BUILTIN(tile_regs_release)
TIR_DEFINE_BUILTIN(pack_tile)
TIR_DEFINE_BUILTIN(pack_reconfig_data_format)
TIR_DEFINE_BUILTIN(copy_tile_to_dst_init_short)
TIR_DEFINE_BUILTIN(copy_tile_to_dst_init_short_with_dt)
TIR_DEFINE_BUILTIN(copy_tile)
TIR_DEFINE_BUILTIN(binary_op_init_common)
TIR_DEFINE_BUILTIN(unary_op_init_common)
TIR_DEFINE_BUILTIN(add_tiles_init)
TIR_DEFINE_BUILTIN(add_tiles)
TIR_DEFINE_BUILTIN(sub_tiles_init)
TIR_DEFINE_BUILTIN(sub_tiles)
TIR_DEFINE_BUILTIN(add_bcast_rows_init_short)
TIR_DEFINE_BUILTIN(add_bcast_cols_init_short)
TIR_DEFINE_BUILTIN(add_tiles_bcast_rows)
TIR_DEFINE_BUILTIN(add_tiles_bcast_cols)
TIR_DEFINE_BUILTIN(mul_tiles_init)
TIR_DEFINE_BUILTIN(mul_tiles)
TIR_DEFINE_BUILTIN(mul_bcast_rows_init_short)
TIR_DEFINE_BUILTIN(mul_bcast_cols_init_short)
TIR_DEFINE_BUILTIN(mul_tiles_bcast_rows)
TIR_DEFINE_BUILTIN(mul_tiles_bcast_cols)
TIR_DEFINE_BUILTIN(reduce_init)
TIR_DEFINE_BUILTIN(reduce_tile)
TIR_DEFINE_BUILTIN(reduce_uninit)
TIR_DEFINE_BUILTIN(binary_max_tile_init)
TIR_DEFINE_BUILTIN(binary_max_tile)
TIR_DEFINE_BUILTIN(div_binary_tile_init)
TIR_DEFINE_BUILTIN(div_binary_tile)
TIR_DEFINE_BUILTIN(exp_tile_init)
TIR_DEFINE_BUILTIN(exp_tile)
TIR_DEFINE_BUILTIN(exp2_tile_init)
TIR_DEFINE_BUILTIN(exp2_tile)
TIR_DEFINE_BUILTIN(recip_tile_init)
TIR_DEFINE_BUILTIN(recip_tile)
TIR_DEFINE_BUILTIN(pack_untilize_slice)
TIR_DEFINE_BUILTIN(pack_untilize_tile)
TIR_DEFINE_BUILTIN(tilize_local_fragment_slice)
TIR_DEFINE_BUILTIN(tilize_cast_fragment_slice)
TIR_DEFINE_BUILTIN(pack_fill_fragment_to_tiled_cb)
TIR_DEFINE_BUILTIN(untilize_cb_front_tile)
TIR_DEFINE_BUILTIN(untilize_cb_front_tile_fragment)
TIR_DEFINE_BUILTIN(fill_fragment)
TIR_DEFINE_BUILTIN(add_fragment)
TIR_DEFINE_BUILTIN(add_fragment_from_cb_front)
TIR_DEFINE_BUILTIN(cast_fragment_slice)

// Register all builtins in TVM's op registry
TVM_REGISTER_OP("tl.blackhole.cb_reserve_back")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Circular buffer ID")
    .add_argument("num_tiles", "int", "Number of tiles to reserve");

TVM_REGISTER_OP("tl.blackhole.cb_push_back")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Circular buffer ID")
    .add_argument("num_tiles", "int", "Number of tiles to push");

TVM_REGISTER_OP("tl.blackhole.cb_wait_front")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Circular buffer ID")
    .add_argument("num_tiles", "int", "Number of tiles to wait for");

TVM_REGISTER_OP("tl.blackhole.cb_pop_front")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Circular buffer ID")
    .add_argument("num_tiles", "int", "Number of tiles to pop");

TVM_REGISTER_OP("tl.blackhole.noc_async_read")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_addr", "uint64", "Source address in DRAM")
    .add_argument("dst_addr", "uint32", "Destination address in L1")
    .add_argument("size", "int", "Size in bytes");

TVM_REGISTER_OP("tl.blackhole.noc_async_write")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_addr", "uint32", "Source address in L1")
    .add_argument("dst_addr", "uint64", "Destination address in DRAM")
    .add_argument("size", "int", "Size in bytes");

TVM_REGISTER_OP("tl.blackhole.noc_async_read_barrier")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.noc_async_write_barrier")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.read_tile_to_cb")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("buffer", "handle", "Source backing buffer handle")
    .add_argument("tile_index", "int", "Logical tile index in the source buffer")
    .add_argument("cb_id", "int", "Destination CB ID")
    .add_argument("tile_bytes", "int", "Tile size in bytes")
    .add_argument("accessor_slot", "int", "Accessor slot for later TT-Metal mapping");

TVM_REGISTER_OP("tl.blackhole.read_page_to_cb")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("buffer", "handle", "Source backing buffer handle")
    .add_argument("page_id", "int", "Logical page id in the source buffer")
    .add_argument("cb_id", "int", "Destination CB ID")
    .add_argument("page_bytes", "int", "Page size in bytes")
    .add_argument("accessor_slot", "int", "Accessor slot for later TT-Metal mapping")
    .add_argument("cb_offset_bytes", "int", "Byte offset within the current CB page");

TVM_REGISTER_OP("tl.blackhole.read_bcast_cols_to_cb")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("buffer", "handle", "Source backing buffer handle")
    .add_argument("page_id", "int", "Logical page id in the source buffer")
    .add_argument("cb_id", "int", "Destination CB ID")
    .add_argument("page_bytes", "int", "Source page size in bytes")
    .add_argument("accessor_slot", "int", "Accessor slot for later TT-Metal mapping")
    .add_argument("vector_len", "int", "Number of vector elements in the broadcast column");

TVM_REGISTER_OP("tl.blackhole.write_tile_from_cb")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Source CB ID")
    .add_argument("buffer", "handle", "Destination backing buffer handle")
    .add_argument("tile_index", "int", "Logical tile index in the destination buffer")
    .add_argument("tile_bytes", "int", "Tile size in bytes")
    .add_argument("accessor_slot", "int", "Accessor slot for later TT-Metal mapping");

TVM_REGISTER_OP("tl.blackhole.write_page_from_cb")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("cb_id", "int", "Source CB ID")
    .add_argument("buffer", "handle", "Destination backing buffer handle")
    .add_argument("page_id", "int", "Logical page id in the destination buffer")
    .add_argument("page_bytes", "int", "Page size in bytes")
    .add_argument("accessor_slot", "int", "Accessor slot for later TT-Metal mapping")
    .add_argument("cb_offset_bytes", "int", "Byte offset within the current CB page");

TVM_REGISTER_OP("tl.blackhole.get_semaphore")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .add_argument("semaphore_id", "uint32", "Program-local semaphore id");

TVM_REGISTER_OP("tl.blackhole.runtime_arg_u32")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .add_argument("name", "string", "Named uint32 runtime arg");

TVM_REGISTER_OP("tl.blackhole.semaphore_wait")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("semaphore_addr", "uint32", "Local L1 semaphore address")
    .add_argument("value", "uint32", "Target semaphore value");

TVM_REGISTER_OP("tl.blackhole.semaphore_set")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("semaphore_addr", "uint32", "Local L1 semaphore address")
    .add_argument("value", "uint32", "Value to store");

TVM_REGISTER_OP("tl.blackhole.semaphore_inc_remote")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("remote_l1_addr", "uint32", "Destination core's local semaphore address")
    .add_argument("remote_core_x", "uint32", "Destination worker core x coordinate")
    .add_argument("remote_core_y", "uint32", "Destination worker core y coordinate")
    .add_argument("value", "uint32", "Increment amount");

TVM_REGISTER_OP("tl.blackhole.semaphore_set_remote")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_local_l1_addr", "uint32",
                  "Local L1 address containing the value to forward")
    .add_argument("remote_core_x", "uint32", "Destination worker core x coordinate")
    .add_argument("remote_core_y", "uint32", "Destination worker core y coordinate")
    .add_argument("remote_l1_addr", "uint32", "Destination core's local semaphore address");

TVM_REGISTER_OP("tl.blackhole.mm_init")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("in0_cb_id", "int", "Input CB 0 ID (A matrix)")
    .add_argument("in1_cb_id", "int", "Input CB 1 ID (B matrix)")
    .add_argument("out_cb_id", "int", "Output CB ID (C matrix)");

TVM_REGISTER_OP("tl.blackhole.reconfig_data_format")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("in0_cb_id", "int", "New SrcA CB ID")
    .add_argument("in1_cb_id", "int", "New SrcB CB ID");

TVM_REGISTER_OP("tl.blackhole.mm_init_short")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("in0_cb_id", "int", "Input CB 0 ID (A matrix)")
    .add_argument("in1_cb_id", "int", "Input CB 1 ID (B matrix)");

TVM_REGISTER_OP("tl.blackhole.mm_init_short_with_dt")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("in0_cb_id", "int", "Input CB 0 ID (A matrix)")
    .add_argument("in1_cb_id", "int", "Input CB 1 ID (B matrix)")
    .add_argument("old_srca_cb_id", "int", "Previously configured SrcA CB ID");

TVM_REGISTER_OP("tl.blackhole.matmul_tiles")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("in0_cb_id", "int", "Input CB 0 ID")
    .add_argument("in1_cb_id", "int", "Input CB 1 ID")
    .add_argument("in0_tile_index", "int", "Tile index in CB 0")
    .add_argument("in1_tile_index", "int", "Tile index in CB 1")
    .add_argument("dst_tile_index", "int", "Destination tile index");

TVM_REGISTER_OP("tl.blackhole.tile_regs_acquire")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.tile_regs_commit")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.tile_regs_wait")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.tile_regs_release")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.pack_tile")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_tile_index", "int", "Source tile index in DST")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_tile_index", "int",
                  "Optional destination tile index within the reserved CB window");

TVM_REGISTER_OP("tl.blackhole.pack_reconfig_data_format")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_cb_id", "int", "Destination CB ID");

TVM_REGISTER_OP("tl.blackhole.copy_tile_to_dst_init_short")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_cb_id", "int", "Source CB ID");

TVM_REGISTER_OP("tl.blackhole.copy_tile_to_dst_init_short_with_dt")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("old_srca_cb_id", "int", "Previously configured SrcA CB ID")
    .add_argument("src_cb_id", "int", "Source CB ID");

TVM_REGISTER_OP("tl.blackhole.copy_tile")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_cb_id", "int", "Source CB ID")
    .add_argument("src_tile_index", "int", "Source tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.binary_op_init_common")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("out_cb_id", "int", "Output CB ID");

TVM_REGISTER_OP("tl.blackhole.unary_op_init_common")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("input_cb_id", "int", "Input CB ID")
    .add_argument("out_cb_id", "int", "Output CB ID");

TVM_REGISTER_OP("tl.blackhole.add_tiles_init")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("acc_to_dest", "int", "Whether to accumulate into the destination register");

TVM_REGISTER_OP("tl.blackhole.add_tiles")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.sub_tiles_init")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.sub_tiles")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.add_bcast_rows_init_short")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.add_bcast_cols_init_short")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.add_tiles_bcast_rows")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.add_tiles_bcast_cols")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.mul_tiles_init")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.mul_tiles")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.mul_bcast_rows_init_short")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.mul_bcast_cols_init_short")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID");

TVM_REGISTER_OP("tl.blackhole.mul_tiles_bcast_rows")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.mul_tiles_bcast_cols")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_cb_id", "int", "Left-hand-side CB ID")
    .add_argument("rhs_cb_id", "int", "Right-hand-side CB ID")
    .add_argument("lhs_tile_index", "int", "Left-hand-side tile index in the CB front window")
    .add_argument("rhs_tile_index", "int", "Right-hand-side tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST");

TVM_REGISTER_OP("tl.blackhole.reduce_init")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_cb_id", "int", "Source CB ID")
    .add_argument("scaler_cb_id", "int", "Scaler CB ID")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("reduce_kind", "string", "Reduction kind string")
    .add_argument("reduce_dim", "string", "Reduction dim string");

TVM_REGISTER_OP("tl.blackhole.reduce_tile")
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_cb_id", "int", "Source CB ID")
    .add_argument("scaler_cb_id", "int", "Scaler CB ID")
    .add_argument("src_tile_index", "int", "Source tile index in the CB front window")
    .add_argument("scaler_tile_index", "int", "Scaler tile index in the CB front window")
    .add_argument("dst_tile_index", "int", "Destination tile index in DST")
    .add_argument("reduce_kind", "string", "Reduction kind string")
    .add_argument("reduce_dim", "string", "Reduction dim string");

TVM_REGISTER_OP("tl.blackhole.reduce_uninit")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("reduce_kind", "string", "Reduction kind string")
    .add_argument("reduce_dim", "string", "Reduction dim string");

TVM_REGISTER_OP("tl.blackhole.binary_max_tile_init")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.binary_max_tile")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_dst_tile_index", "int", "Left-hand-side DST tile index")
    .add_argument("rhs_dst_tile_index", "int", "Right-hand-side DST tile index")
    .add_argument("dst_tile_index", "int", "Destination DST tile index")
    .add_argument("vector_mode", "str", "Optional TT-Metal VectorMode suffix");

TVM_REGISTER_OP("tl.blackhole.div_binary_tile_init")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.div_binary_tile")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("lhs_dst_tile_index", "int", "Left-hand-side DST tile index")
    .add_argument("rhs_dst_tile_index", "int", "Right-hand-side DST tile index")
    .add_argument("dst_tile_index", "int", "Destination DST tile index");

TVM_REGISTER_OP("tl.blackhole.exp_tile_init")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.exp_tile")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_tile_index", "int", "Destination DST tile index");

TVM_REGISTER_OP("tl.blackhole.exp2_tile_init")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.blackhole.exp2_tile")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_tile_index", "int", "Destination DST tile index");

TVM_REGISTER_OP("tl.blackhole.recip_tile_init")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("legacy_compat", "int", "Optional TT-Metal reciprocal legacy compatibility flag");

TVM_REGISTER_OP("tl.blackhole.recip_tile")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_tile_index", "int", "Destination DST tile index")
    .add_argument("vector_mode", "string", "Optional TT-Metal VectorMode name")
    .add_argument("legacy_compat", "int", "Optional TT-Metal reciprocal legacy compatibility flag");

TVM_REGISTER_OP("tl.blackhole.pack_untilize_slice")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_offset_elements", "int", "Destination element offset within the reserved CB write window")
    .add_argument("num_elements", "int", "Number of contiguous elements to copy")
    .add_argument("src_offset_elements", "int",
                  "Optional source element offset within the local fragment");

TVM_REGISTER_OP("tl.blackhole.pack_untilize_tile")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_tile_index", "int", "Destination tile index within the reserved CB write window")
    .add_argument("src_offset_elements", "int", "Source element offset within the local fragment");

TVM_REGISTER_OP("tl.blackhole.tilize_local_fragment_slice")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_offset_elements", "int", "Logical destination element offset")
    .add_argument("num_elements", "int", "Number of contiguous elements to materialize")
    .add_argument("row_width", "int", "Logical row width of the destination tiled tensor")
    .add_argument("src_offset_elements", "int",
                  "Optional source element offset within the local fragment");

TVM_REGISTER_OP("tl.blackhole.tilize_cast_fragment_slice")
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination CB-backed fragment buffer handle")
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_offset_elements", "int", "Logical destination element offset")
    .add_argument("src_offset_elements", "int", "Source element offset within the local fragment")
    .add_argument("num_elements", "int", "Number of contiguous elements to materialize")
    .add_argument("row_width", "int", "Logical row width of the destination tiled tensor");

TVM_REGISTER_OP("tl.blackhole.pack_fill_fragment_to_tiled_cb")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination CB-backed fragment buffer handle")
    .add_argument("dst_cb_id", "int", "Destination CB ID")
    .add_argument("dst_offset_elements", "int", "Logical destination element offset")
    .add_argument("num_elements", "int", "Number of logical elements to fill")
    .add_argument("row_width", "int", "Logical row width of the destination tiled tensor")
    .add_argument("value", "float", "Scalar fill value");

TVM_REGISTER_OP("tl.blackhole.untilize_cb_front_tile")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination local fragment buffer handle")
    .add_argument("src_cb_id", "int", "Source CB ID")
    .add_argument("src_tile_index", "int", "Source tile index in the current CB front window")
    .add_argument("dst_offset_elements", "int", "Destination element offset in the local fragment")
    .add_argument("num_elements", "int", "Number of contiguous elements to copy");

TVM_REGISTER_OP("tl.blackhole.untilize_cb_front_tile_fragment")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination local fragment buffer handle")
    .add_argument("src_cb_id", "int", "Source CB ID")
    .add_argument("src_tile_index", "int", "Source tile index in the current CB front window")
    .add_argument("dst_offset_elements", "int", "Destination element offset in the local fragment");

TVM_REGISTER_OP("tl.blackhole.fill_fragment")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination local fragment buffer handle")
    .add_argument("num_elements", "int", "Number of contiguous destination elements")
    .add_argument("value", "float", "Scalar literal fill value");

TVM_REGISTER_OP("tl.blackhole.add_fragment")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination/source local fragment buffer handle")
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("num_elements", "int", "Number of contiguous destination elements");

TVM_REGISTER_OP("tl.blackhole.add_fragment_from_cb_front")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination/source local fragment buffer handle")
    .add_argument("src_cb_id", "int", "Source CB ID whose current front page(s) hold the fragment")
    .add_argument("num_elements", "int", "Number of contiguous destination elements");

TVM_REGISTER_OP("tl.blackhole.cast_fragment_slice")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("dst_buffer", "handle", "Destination local fragment buffer handle")
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_offset", "int", "Destination element offset")
    .add_argument("src_offset", "int", "Source element offset")
    .add_argument("num_elements", "int", "Number of contiguous elements");

}  // namespace builtin
}  // namespace tir
}  // namespace tvm
