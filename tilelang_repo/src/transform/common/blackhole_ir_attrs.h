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
 * \file blackhole_ir_attrs.h
 * \brief Shared Blackhole TIR annotation keys.
 */

#ifndef TVM_TL_BLACKHOLE_IR_ATTRS_H_
#define TVM_TL_BLACKHOLE_IR_ATTRS_H_

namespace tvm {
namespace tl {
namespace blackhole_ir_attrs {

inline constexpr const char* kCBRequirementIndex =
    "tl.blackhole.cb_requirement_index";

}  // namespace blackhole_ir_attrs
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_BLACKHOLE_IR_ATTRS_H_
