/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file blackhole_runtime_arg_schema.h
 * \brief Shared per-work runtime-arg schema keys for the Blackhole backend
 */

#ifndef TVM_TL_BLACKHOLE_RUNTIME_ARG_SCHEMA_H_
#define TVM_TL_BLACKHOLE_RUNTIME_ARG_SCHEMA_H_

namespace tvm {
namespace tl {
namespace blackhole_runtime_arg_schema {

inline constexpr const char* kPerWorkArgSpecs = "per_work_arg_specs";
inline constexpr const char* kArgKind = "arg_kind";
inline constexpr const char* kArgIdentity = "arg_identity";
inline constexpr const char* kBuffer = "buffer";
inline constexpr const char* kDescriptorKind = "descriptor_kind";
inline constexpr const char* kValueKind = "value_kind";
inline constexpr const char* kValueSource = "value_source";
inline constexpr const char* kConstantValue = "constant_value";

inline constexpr const char* kValueCurrentWorkLinearId = "current_work_linear_id";
inline constexpr const char* kValueLogicalBlockX = "logical_block_x";
inline constexpr const char* kValueLogicalBlockY = "logical_block_y";
inline constexpr const char* kValueGemmNumKTiles = "gemm_num_k_tiles";
inline constexpr const char* kValueGemmLogicalNTiles = "gemm_logical_n_tiles";
inline constexpr const char* kValueConstant = "constant";

inline constexpr const char* kDescriptorTileStart = "tile_start";
inline constexpr const char* kDescriptorTileCount = "tile_count";
inline constexpr const char* kDescriptorTileStride = "tile_stride";
inline constexpr const char* kDescriptorKTileStart = "k_tile_start";
inline constexpr const char* kDescriptorKTileCount = "k_tile_count";

inline constexpr const char* kValueSourceWorkLinearId = "work_linear_id";
inline constexpr const char* kValueSourceLogicalBlockX = "logical_block_x";
inline constexpr const char* kValueSourceLogicalBlockY = "logical_block_y";
inline constexpr const char* kValueSourceComputeNumKTiles = "compute_op_num_k_tiles";
inline constexpr const char* kValueSourceComputeLogicalNTiles = "compute_op_logical_n_tiles";
inline constexpr const char* kValueSourceConstant = "constant";

}  // namespace blackhole_runtime_arg_schema
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_BLACKHOLE_RUNTIME_ARG_SCHEMA_H_
