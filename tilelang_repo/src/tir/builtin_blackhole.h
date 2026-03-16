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

}  // namespace builtin
}  // namespace tir
}  // namespace tvm

#endif  // TL_TIR_BUILTIN_BLACKHOLE_H_
