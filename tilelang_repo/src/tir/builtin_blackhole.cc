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
TIR_DEFINE_BUILTIN(matmul_tiles)
TIR_DEFINE_BUILTIN(tile_regs_acquire)
TIR_DEFINE_BUILTIN(tile_regs_commit)
TIR_DEFINE_BUILTIN(tile_regs_wait)
TIR_DEFINE_BUILTIN(tile_regs_release)
TIR_DEFINE_BUILTIN(pack_tile)
TIR_DEFINE_BUILTIN(reduce_row)

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
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_tile_index", "int", "Source tile index in DST")
    .add_argument("dst_cb_id", "int", "Destination CB ID");

TVM_REGISTER_OP("tl.blackhole.reduce_row")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .add_argument("src_buffer", "handle", "Source local fragment buffer handle")
    .add_argument("dst_buffer", "handle", "Destination scalar local fragment buffer handle")
    .add_argument("num_elements", "int", "Number of contiguous source elements")
    .add_argument("reduce_kind", "string", "Reduction kind string")
    .add_argument("clear", "bool", "Whether to clear destination before reduction");

}  // namespace builtin
}  // namespace tir
}  // namespace tvm
