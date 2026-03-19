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
 * \file blackhole_cb_common.h
 * \brief Shared Circular Buffer planning protocol types for Blackhole backend
 */

#ifndef TVM_TL_BLACKHOLE_CB_COMMON_H_
#define TVM_TL_BLACKHOLE_CB_COMMON_H_

#include <string>

namespace tvm {
namespace tl {

/*!
 * \brief CB type classification
 */
enum class CBType {
  kInput,        // CB 0-15: Input buffers (Reader -> Compute)
  kOutput,       // CB 16-31: Output buffers (Compute -> Writer)
  kIntermediate  // CB 32-63: Intermediate buffers
};

/*!
 * \brief CB requirement description shared between extraction and planning.
 */
struct CBRequirement {
  std::string name;        // Buffer name
  CBType type;             // CB classification
  int page_size;           // Size of each page in bytes
  int num_pages;           // Number of pages (for double buffering)
  std::string data_format; // Data format string (e.g., "Float16", "Float32")
  int lifetime_begin;      // First requirement slot where this CB is live
  int lifetime_end;        // Last requirement slot where this CB is live

  CBRequirement()
      : type(CBType::kIntermediate),
        page_size(2048),
        num_pages(2),
        data_format("Float16"),
        lifetime_begin(0),
        lifetime_end(0) {}
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_BLACKHOLE_CB_COMMON_H_
