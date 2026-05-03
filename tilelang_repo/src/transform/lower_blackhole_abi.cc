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
 * \file lower_blackhole_abi.cc
 * \brief Segment, accessor, kernel, and ABI planning for Blackhole lowering.
 */

#include "lower_blackhole_ops.h"

#include "common/blackhole_runtime_arg_schema.h"
#include "common/blackhole_tile_compute_covering.h"
#include "common/blackhole_tile_compute_legalizer.h"
#include "common/blackhole_utils.h"

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using tir::Buffer;
using tir::CallNode;
using tir::PrimFunc;
using tir::Stmt;
using tvm::Bool;
using tvm::DataType;
using tvm::Integer;
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;

namespace {

constexpr int kBlackholeTileRows = 32;
constexpr int kBlackholeTileCols = 32;

static std::string DataTypeToDataFormatForBlackhole(DataType dtype) {
  if (dtype.is_bfloat16())
    return "Float16_b";
  if (dtype.is_float16())
    return "Float16";
  if (dtype.is_float() && dtype.bits() == 32)
    return "Float32";
  if (dtype.is_float() && dtype.bits() == 8)
    return "Bfp8";
  if (dtype.is_uint() && dtype.bits() == 32)
    return "UInt32";
  if (dtype.is_uint() && dtype.bits() == 16)
    return "UInt16";
  if (dtype.is_int() && dtype.bits() == 32)
    return "Int32";
  if (dtype.is_int() && dtype.bits() == 16)
    return "Int16";
  return "Float16_b";
}

static int CeilDivToInt(int64_t value, int64_t divisor) {
  ICHECK_GT(divisor, 0);
  if (value <= 0) {
    return 1;
  }
  return static_cast<int>((value + divisor - 1) / divisor);
}

static std::string GemmWarpPolicyTypeToStringForBlackhole(int policy_type) {
  switch (policy_type) {
  case 0:
    return "Square";
  case 1:
    return "FullRow";
  case 2:
    return "FullCol";
  case 3:
    return "Free";
  default:
    return "Unknown";
  }
}

static Array<TTKernelDefine> EncodeTTKernelDefines(
    const std::vector<std::pair<std::string, std::string>> &entries) {
  Array<TTKernelDefine> encoded_entries;
  for (const auto &[name, value] : entries) {
    encoded_entries.push_back(TTKernelDefine(String(name), String(value)));
  }
  return encoded_entries;
}

static Array<TTKernelNamedCompileArg> EncodeTTKernelNamedCompileArgs(
    const std::vector<std::pair<std::string, uint32_t>> &entries) {
  Array<TTKernelNamedCompileArg> encoded_entries;
  for (const auto &[name, value] : entries) {
    encoded_entries.push_back(TTKernelNamedCompileArg(String(name), value));
  }
  return encoded_entries;
}

static Array<Integer> BuildIntegerArray(std::initializer_list<int64_t> values) {
  Array<Integer> result;
  for (int64_t value : values) {
    result.push_back(Integer(value));
  }
  return result;
}

static Array<String>
BuildStringArray(std::initializer_list<const char *> values) {
  Array<String> result;
  for (const char *value : values) {
    result.push_back(String(value));
  }
  return result;
}

static String ResolveComputeOperandHostBuffer(
    const std::unordered_map<std::string, std::string> &host_buffer_by_operand,
    const std::string &buffer) {
  auto it = host_buffer_by_operand.find(buffer);
  if (it != host_buffer_by_operand.end() && !it->second.empty()) {
    return String(it->second);
  }
  return String(buffer);
}

static TTComputeOperandBindingPlan BuildGemmComputeOperandBindingPlan(
    const GemmComputeOpFact &fact,
    const std::unordered_map<std::string, std::string> &host_buffer_by_operand,
    const char *role) {
  std::string buffer;
  DataType dtype;
  bool transpose = false;
  const std::string role_string(role);
  if (role_string == "a") {
    buffer = fact.a_buffer;
    dtype = fact.a_dtype;
    transpose = fact.transpose_a;
  } else if (role_string == "b") {
    buffer = fact.b_buffer;
    dtype = fact.b_dtype;
    transpose = fact.transpose_b;
  } else {
    buffer = fact.c_buffer;
    dtype = fact.c_dtype;
  }
  const String host_buffer =
      ResolveComputeOperandHostBuffer(host_buffer_by_operand, buffer);
  const String data_format = String(DataTypeToDataFormatForBlackhole(dtype));

  return TTComputeOperandBindingPlan(
      String(role), String(buffer), host_buffer, data_format, data_format,
      String(transpose ? "transpose" : "identity"));
}

static TTComputeOpPlan BuildTTComputeOpPlanFromFact(
    const GemmComputeOpFact &fact,
    const std::unordered_map<std::string, std::string> &host_buffer_by_operand,
    const String &kernel_name, int64_t kernel_plan_index, int64_t ordinal) {
  const BlackholeTileComputeCoveringDecision covering =
      SelectBlackholeTileComputeCovering("matmul_tiles");
  ICHECK(covering.selected)
      << "TileCompute covering rejected operation matmul_tiles: "
      << covering.reject_reason;

  Array<TTComputeOperandBindingPlan> operand_bindings;
  const TTComputeOperandBindingPlan a_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "a");
  const TTComputeOperandBindingPlan b_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "b");
  const TTComputeOperandBindingPlan c_binding =
      BuildGemmComputeOperandBindingPlan(fact, host_buffer_by_operand, "c");
  operand_bindings.push_back(a_binding);
  operand_bindings.push_back(b_binding);
  operand_bindings.push_back(c_binding);

  const int64_t mt = fact.m > 0 ? fact.m / 32 : 0;
  const int64_t nt = fact.n > 0 ? fact.n / 32 : 0;
  const int64_t kt = fact.k > 0 ? fact.k / 32 : 0;

  Array<String> mbarrier_index_exprs;
  for (const auto &expr : fact.mbarrier_index_exprs) {
    mbarrier_index_exprs.push_back(String(expr));
  }

  const std::string name = "compute_op_" +
                           static_cast<std::string>(kernel_name) + "_" +
                           std::to_string(ordinal);
  TTComputeOpPlan plan = TTComputeOpPlan(
      String(name), kernel_name, kernel_plan_index,
      String(covering.result_kind), String(covering.operation_name), Bool(true),
      operand_bindings, BuildStringArray({"M", "N", "K"}),
      BuildIntegerArray({fact.m, fact.n, fact.k}),
      BuildIntegerArray({mt, nt, kt}), BuildIntegerArray({mt, nt, kt}),
      BuildIntegerArray({mt, nt}),
      String(DataTypeToDataFormatForBlackhole(fact.c_dtype)),
      String(fact.mbarrier_buffer), String(fact.mbarrier_scope),
      mbarrier_index_exprs,
      /*tile_compute_dag_node_id=*/-1, String(), String(),
      /*tile_compute_fanout_use_count=*/0, String());
  RequireLegalBlackholeTileComputePlan(plan);
  return plan;
}

static std::string
MakeBlackholeRuntimeArgIdentity(const std::string &kind,
                                const std::string &name,
                                const std::string &buffer_name = "") {
  if (!buffer_name.empty()) {
    return kind + ":" + buffer_name;
  }
  return !kind.empty() ? kind : name;
}

static std::string MakeSegmentBufferKey(const std::string &segment_kind,
                                        const tir::Buffer &buffer) {
  std::ostringstream os;
  os << segment_kind << ":"
     << reinterpret_cast<uintptr_t>(BufferDataIdentity(buffer));
  return os.str();
}

static bool IsBufferAddrRuntimeArgKind(const std::string &kind) {
  return kind == "input_buffer_addr32" || kind == "input_buffer_addr" ||
         kind == "output_buffer_addr32" || kind == "output_buffer_addr";
}

static std::string GetRuntimeArgKind(const Any &arg_item) {
  auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
  if (arg.empty() || !arg.Get("kind")) {
    return "";
  }
  return static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
}

static std::string GetRuntimeArgBufferName(const Any &arg_item) {
  auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
  if (arg.empty() || !arg.Get("buffer")) {
    return "";
  }
  return static_cast<std::string>(Downcast<String>(arg.Get("buffer").value()));
}

static int FindRuntimeArgIndex(const Array<Any> &runtime_args,
                               const std::string &kind,
                               const std::string &buffer_name = "") {
  for (int i = 0, n = static_cast<int>(runtime_args.size()); i < n; ++i) {
    if (GetRuntimeArgKind(runtime_args[i]) != kind) {
      continue;
    }
    if (!buffer_name.empty()) {
      const std::string existing_buffer =
          GetRuntimeArgBufferName(runtime_args[i]);
      if (!existing_buffer.empty() && existing_buffer != buffer_name) {
        continue;
      }
    }
    return i;
  }
  return -1;
}

static Map<String, Any> MakeRuntimeArg(const std::string &name,
                                       const std::string &kind,
                                       const std::string &dtype,
                                       const std::string &buffer_name = "") {
  Map<String, Any> arg;
  arg.Set("name", String(name));
  arg.Set("kind", String(kind));
  arg.Set("dtype", String(dtype));
  if (!buffer_name.empty()) {
    arg.Set("buffer", String(buffer_name));
  }
  arg.Set("identity",
          String(MakeBlackholeRuntimeArgIdentity(kind, name, buffer_name)));
  return arg;
}

static std::string
ResolveAccessorBufferNameByCompileTimeArgOffset(const Array<Any> &accessors,
                                                int offset) {
  for (const auto &accessor_item : accessors) {
    auto accessor =
        accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (accessor.empty() || !accessor.Get("buffer")) {
      continue;
    }
    if (!accessor.Get("compile_time_arg_offset")) {
      continue;
    }
    const int accessor_offset =
        Downcast<Integer>(accessor.Get("compile_time_arg_offset").value())
            .IntValue();
    if (accessor_offset == offset) {
      return static_cast<std::string>(
          Downcast<String>(accessor.Get("buffer").value()));
    }
  }
  return "";
}

static Array<Any>
EnsureSegmentBufferRuntimeArgs(const std::string &segment_kind,
                               const Array<Any> &accessors,
                               const Optional<Any> &runtime_args_opt,
                               const std::string &input_buffer_name = "",
                               const std::string &output_buffer_name = "",
                               const std::vector<std::string> &input_buffer_names = {},
                               const std::vector<std::string> &output_buffer_names = {}) {
  Array<Any> existing_runtime_args =
      runtime_args_opt ? Downcast<Array<Any>>(runtime_args_opt.value())
                       : Array<Any>();
  if (segment_kind == "fused_dataflow") {
    std::vector<std::string> resolved_input_buffer_names = input_buffer_names;
    std::vector<std::string> resolved_output_buffer_names = output_buffer_names;
    if (resolved_input_buffer_names.empty()) {
      const std::string resolved_input_buffer_name =
          !input_buffer_name.empty()
              ? input_buffer_name
              : ResolveAccessorBufferNameByCompileTimeArgOffset(accessors, 0);
      if (!resolved_input_buffer_name.empty()) {
        resolved_input_buffer_names.push_back(resolved_input_buffer_name);
      }
    }
    if (resolved_output_buffer_names.empty()) {
      const std::string resolved_output_buffer_name =
          !output_buffer_name.empty()
              ? output_buffer_name
              : ResolveAccessorBufferNameByCompileTimeArgOffset(
                    accessors, resolved_input_buffer_names.empty() ? 0 : 2);
      if (!resolved_output_buffer_name.empty()) {
        resolved_output_buffer_names.push_back(resolved_output_buffer_name);
      }
    }
    std::vector<bool> consumed(existing_runtime_args.size(), false);
    Array<Any> runtime_args;
    auto push_existing_or_synthesized = [&](const std::string &kind,
                                            const std::string &name,
                                            const std::string &buffer_name =
                                                "") {
      const int existing_index =
          FindRuntimeArgIndex(existing_runtime_args, kind, buffer_name);
      if (existing_index >= 0) {
        runtime_args.push_back(existing_runtime_args[existing_index]);
        consumed[existing_index] = true;
        return;
      }
      runtime_args.push_back(MakeRuntimeArg(name, kind, "uint32", buffer_name));
    };

    for (const std::string &buffer_name : resolved_input_buffer_names) {
      push_existing_or_synthesized("input_buffer_addr32",
                                   buffer_name + "_addr",
                                   buffer_name);
    }
    for (const std::string &buffer_name : resolved_output_buffer_names) {
      push_existing_or_synthesized("output_buffer_addr32",
                                   buffer_name + "_addr",
                                   buffer_name);
    }
    push_existing_or_synthesized("work_linear_id", "work_linear_id");
    if (!resolved_input_buffer_names.empty()) {
      push_existing_or_synthesized("a_tile_start_id", "a_tile_start_id");
      push_existing_or_synthesized("a_tile_num_tiles", "a_tile_num_tiles");
      push_existing_or_synthesized("a_tile_stride", "a_tile_stride");
    }
    if (!resolved_output_buffer_names.empty()) {
      push_existing_or_synthesized("output_tile_start_id",
                                   "output_tile_start_id");
      push_existing_or_synthesized("output_tile_num_tiles",
                                   "output_tile_num_tiles");
      push_existing_or_synthesized("output_tile_stride", "output_tile_stride");
    }
    for (int i = 0, n = static_cast<int>(existing_runtime_args.size()); i < n;
         ++i) {
      if (!consumed[i]) {
        runtime_args.push_back(existing_runtime_args[i]);
      }
    }
    return runtime_args;
  }

  const bool is_reader = segment_kind == "reader";
  const bool is_writer = segment_kind == "writer";
  if (!is_reader && !is_writer) {
    return existing_runtime_args;
  }

  Array<Any> buffer_args;
  Array<Any> other_args;
  std::vector<std::string> bound_buffers;

  for (const auto &arg_item : existing_runtime_args) {
    auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (arg.empty()) {
      other_args.push_back(arg_item);
      continue;
    }
    const std::string arg_kind =
        arg.Get("kind") ? static_cast<std::string>(
                              Downcast<String>(arg.Get("kind").value()))
                        : std::string();
    if (IsBufferAddrRuntimeArgKind(arg_kind) && arg.Get("buffer")) {
      bound_buffers.push_back(static_cast<std::string>(
          Downcast<String>(arg.Get("buffer").value())));
      buffer_args.push_back(arg_item);
    } else {
      other_args.push_back(arg_item);
    }
  }

  auto has_bound_buffer = [&](const std::string &buffer_name) {
    return std::find(bound_buffers.begin(), bound_buffers.end(), buffer_name) !=
           bound_buffers.end();
  };

  for (const auto &accessor_item : accessors) {
    auto accessor =
        accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (accessor.empty() || !accessor.Get("buffer")) {
      continue;
    }
    const std::string buffer_name = static_cast<std::string>(
        Downcast<String>(accessor.Get("buffer").value()));
    if (buffer_name.empty() || has_bound_buffer(buffer_name)) {
      continue;
    }

    Map<String, Any> arg;
    const std::string arg_kind =
        is_reader ? "input_buffer_addr32" : "output_buffer_addr32";
    const std::string arg_name = buffer_name + "_addr";
    arg.Set("name", String(arg_name));
    arg.Set("kind", String(arg_kind));
    arg.Set("dtype", String("uint32"));
    arg.Set("buffer", String(buffer_name));
    arg.Set("identity", String(MakeBlackholeRuntimeArgIdentity(
                            arg_kind, arg_name, buffer_name)));
    buffer_args.push_back(arg);
    bound_buffers.push_back(buffer_name);
  }

  Array<Any> runtime_args;
  for (const auto &item : buffer_args) {
    runtime_args.push_back(item);
  }
  auto append_runtime_arg_if_missing = [&](const std::string &name,
                                           const std::string &kind,
                                           const std::string &buffer_name =
                                               "") {
    if (FindRuntimeArgIndex(runtime_args, kind, buffer_name) >= 0) {
      return;
    }
    runtime_args.push_back(MakeRuntimeArg(name, kind, "uint32", buffer_name));
  };
  if (is_reader) {
    std::vector<std::string> input_buffers;
    for (const auto &item : buffer_args) {
      auto arg = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (arg.empty() || !arg.Get("kind")) {
        continue;
      }
      const std::string arg_kind =
          static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
      if (arg_kind != "input_buffer_addr32" &&
          arg_kind != "input_buffer_addr") {
        continue;
      }
      input_buffers.push_back(arg.Get("buffer")
                                  ? static_cast<std::string>(Downcast<String>(
                                        arg.Get("buffer").value()))
                                  : std::string());
    }
    if (!input_buffers.empty()) {
      append_runtime_arg_if_missing("a_tile_start_id", "a_tile_start_id",
                                    input_buffers[0]);
      append_runtime_arg_if_missing("a_tile_num_tiles", "a_tile_num_tiles",
                                    input_buffers[0]);
      append_runtime_arg_if_missing("a_tile_stride", "a_tile_stride",
                                    input_buffers[0]);
    }
    if (input_buffers.size() > 1) {
      append_runtime_arg_if_missing("b_tile_start_id", "b_tile_start_id",
                                    input_buffers[1]);
      append_runtime_arg_if_missing("b_tile_num_tiles", "b_tile_num_tiles",
                                    input_buffers[1]);
      append_runtime_arg_if_missing("b_tile_stride", "b_tile_stride",
                                    input_buffers[1]);
    }
    append_runtime_arg_if_missing("k_tile_start_id", "k_tile_start_id");
    append_runtime_arg_if_missing("num_k_tiles", "num_k_tiles");
  } else if (is_writer) {
    std::string resolved_output_buffer_name =
        !output_buffer_name.empty() ? output_buffer_name : "";
    if (resolved_output_buffer_name.empty()) {
      for (const auto &item : buffer_args) {
        auto arg = item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (arg.empty() || !arg.Get("kind")) {
          continue;
        }
        const std::string arg_kind =
            static_cast<std::string>(Downcast<String>(arg.Get("kind").value()));
        if (arg_kind != "output_buffer_addr32" &&
            arg_kind != "output_buffer_addr") {
          continue;
        }
        if (arg.Get("buffer")) {
          resolved_output_buffer_name = static_cast<std::string>(
              Downcast<String>(arg.Get("buffer").value()));
        }
        break;
      }
    }
    append_runtime_arg_if_missing("output_tile_start_id",
                                  "output_tile_start_id",
                                  resolved_output_buffer_name);
    append_runtime_arg_if_missing("output_tile_num_tiles",
                                  "output_tile_num_tiles",
                                  resolved_output_buffer_name);
    append_runtime_arg_if_missing("output_tile_stride", "output_tile_stride",
                                  resolved_output_buffer_name);
  }
  for (const auto &item : other_args) {
    runtime_args.push_back(item);
  }
  return runtime_args;
}

static Map<String, Any> MakeCompileTimeArgSpec(
    const std::string &name, const std::string &kind, const std::string &dtype,
    int offset, int count, const std::string &segment_role,
    const std::string &buffer = "", const std::vector<uint32_t> &values = {},
    int args_config_bits = 0, int transport_page_size_bytes = 0,
    const std::string &layout = "", const std::string &memory_space = "",
    std::vector<int64_t> host_axis_order = {}, bool transpose_2d = false) {
  Map<String, Any> spec;
  spec.Set("name", String(name));
  spec.Set("kind", String(kind));
  spec.Set("dtype", String(dtype));
  spec.Set("offset", Integer(offset));
  spec.Set("count", Integer(count));
  if (!buffer.empty()) {
    spec.Set("buffer", String(buffer));
  }
  if (!segment_role.empty()) {
    spec.Set("segment_role", String(segment_role));
  }
  if (!values.empty()) {
    Array<Any> encoded_values;
    for (uint32_t value : values) {
      encoded_values.push_back(Integer(static_cast<int>(value)));
    }
    spec.Set("values", encoded_values);
  }
  if (args_config_bits != 0) {
    spec.Set("args_config_bits", Integer(args_config_bits));
  }
  if (transport_page_size_bytes > 0) {
    spec.Set("transport_page_size", Integer(transport_page_size_bytes));
  }
  if (!layout.empty()) {
    spec.Set("layout", String(layout));
  }
  if (!memory_space.empty()) {
    spec.Set("memory_space", String(memory_space));
  }
  if (!host_axis_order.empty()) {
    Array<Any> axis_order;
    for (int64_t axis : host_axis_order) {
      axis_order.push_back(Integer(axis));
    }
    spec.Set("host_axis_order", axis_order);
  }
  if (transpose_2d) {
    spec.Set("transpose_2d", Bool(true));
  }
  return spec;
}

static std::string AccessorCompileTimeArgKind(const std::string &layout,
                                              const std::string &memory_space) {
  if (layout == "interleaved" &&
      (memory_space == "dram" || memory_space == "DRAM")) {
    return "interleaved_accessor_cta";
  }
  if (layout == "sharded" || memory_space == "l1" || memory_space == "L1") {
    return "sharded_accessor_cta";
  }
  return "page_indexed_accessor_cta";
}

static TTPerWorkArgSpec MakePerWorkArgSpec(const std::string &arg_kind,
                                           const std::string &arg_identity,
                                           const std::string &descriptor_kind,
                                           const std::string &value_source,
                                           const std::string &buffer = "",
                                           uint32_t constant_value = 0) {
  return TTPerWorkArgSpec(String(arg_kind), String(arg_identity),
                          String(buffer), String(descriptor_kind),
                          String(value_source),
                          static_cast<int64_t>(constant_value));
}

static TTKernelLaunchSpec MakeLaunchSpec(const std::string &core_type,
                                         const std::string &processor,
                                         const std::string &noc) {
  return TTKernelLaunchSpec(String(core_type), String(processor), String(noc));
}

static TTKernelComputeConfig MakeEmptyComputeConfig() {
  return TTKernelComputeConfig(String(""), false, false, false, Array<String>{},
                               false, Array<TTKernelDefine>{},
                               Array<TTKernelNamedCompileArg>{}, false, 1, 0, 0,
                               String(""));
}

static Map<String, Any> AsStringAnyMap(const Any &item) {
  return item.as<Map<String, Any>>().value_or(Map<String, Any>());
}

static String GetMapString(const Map<String, Any> &item, const char *key) {
  if (auto value = item.Get(key)) {
    return Downcast<String>(value.value());
  }
  return String("");
}

static int64_t GetMapInteger(const Map<String, Any> &item, const char *key,
                             int64_t default_value = 0) {
  if (auto value = item.Get(key)) {
    return Downcast<Integer>(value.value()).IntValue();
  }
  return default_value;
}

static bool GetMapBool(const Map<String, Any> &item, const char *key,
                       bool default_value = false) {
  if (auto value = item.Get(key)) {
    return Downcast<Bool>(value.value());
  }
  return default_value;
}

static Array<Integer> GetMapIntegerArray(const Map<String, Any> &item,
                                         const char *key) {
  Array<Integer> values;
  if (auto value = item.Get(key)) {
    for (const Any &element : Downcast<Array<Any>>(value.value())) {
      values.push_back(Downcast<Integer>(element));
    }
  }
  return values;
}

static TTRuntimeArgSpec DecodeRuntimeArgSpec(const Any &item) {
  const Map<String, Any> arg = AsStringAnyMap(item);
  return TTRuntimeArgSpec(
      GetMapString(arg, "name"), GetMapString(arg, "kind"),
      GetMapString(arg, "dtype"), GetMapString(arg, "buffer"),
      GetMapString(arg, "identity"), GetMapInteger(arg, "core_x", -1),
      GetMapInteger(arg, "core_y", -1));
}

static Array<TTRuntimeArgSpec>
DecodeRuntimeArgSpecs(const Optional<Any> &items_opt) {
  Array<TTRuntimeArgSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any &item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeRuntimeArgSpec(item));
  }
  return specs;
}

static TTCompileTimeArgSpec DecodeCompileTimeArgSpec(const Any &item) {
  const Map<String, Any> spec = AsStringAnyMap(item);
  return TTCompileTimeArgSpec(
      GetMapString(spec, "name"), GetMapString(spec, "kind"),
      GetMapString(spec, "dtype"), GetMapInteger(spec, "offset"),
      GetMapInteger(spec, "count"), GetMapString(spec, "buffer"),
      GetMapString(spec, "segment_role"), GetMapIntegerArray(spec, "values"),
      GetMapInteger(spec, "args_config_bits"),
      GetMapInteger(spec, "transport_page_size"), GetMapString(spec, "layout"),
      GetMapString(spec, "memory_space"),
      GetMapIntegerArray(spec, "host_axis_order"),
      GetMapBool(spec, "transpose_2d"));
}

static Array<TTCompileTimeArgSpec>
DecodeCompileTimeArgSpecs(const Optional<Any> &items_opt) {
  Array<TTCompileTimeArgSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any &item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeCompileTimeArgSpec(item));
  }
  return specs;
}

static TTAccessorSpec DecodeAccessorSpec(const Any &item) {
  const Map<String, Any> accessor = AsStringAnyMap(item);
  return TTAccessorSpec(GetMapString(accessor, "buffer"),
                        GetMapInteger(accessor, "compile_time_arg_offset"),
                        GetMapInteger(accessor, "compile_time_arg_count"),
                        GetMapInteger(accessor, "common_runtime_arg_offset"),
                        GetMapInteger(accessor, "common_runtime_arg_count"),
                        GetMapInteger(accessor, "args_config_bits"),
                        GetMapInteger(accessor, "transport_page_size"),
                        GetMapString(accessor, "layout"),
                        GetMapString(accessor, "memory_space"),
                        GetMapIntegerArray(accessor, "host_axis_order"),
                        GetMapBool(accessor, "transpose_2d"));
}

static Array<TTAccessorSpec>
DecodeAccessorSpecs(const Optional<Any> &items_opt) {
  Array<TTAccessorSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any &item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeAccessorSpec(item));
  }
  return specs;
}

static TTSemaphoreBindingSpec DecodeSemaphoreBindingSpec(const Any &item) {
  const Map<String, Any> binding = AsStringAnyMap(item);
  return TTSemaphoreBindingSpec(GetMapString(binding, "name"),
                                GetMapInteger(binding, "semaphore_id"),
                                GetMapString(binding, "arg_kind"));
}

static Array<TTSemaphoreBindingSpec>
DecodeSemaphoreBindingSpecs(const Optional<Any> &items_opt) {
  Array<TTSemaphoreBindingSpec> specs;
  if (!items_opt) {
    return specs;
  }
  for (const Any &item : Downcast<Array<Any>>(items_opt.value())) {
    specs.push_back(DecodeSemaphoreBindingSpec(item));
  }
  return specs;
}

static void BuildTTKernelAndABISeeds(const Array<Any> &segment_plan,
                                     Array<TTKernel> *kernels_out,
                                     Array<TTABIPlan> *abi_plans_out) {
  Array<TTKernel> kernels;
  Array<TTABIPlan> abi_plans;
  int index = 0;
  for (const Any &item : segment_plan) {
    Map<String, Any> segment =
        item.as<Map<String, Any>>().value_or(Map<String, Any>());
    if (segment.empty()) {
      continue;
    }
    String kernel_name = segment.Get("name")
                             ? Downcast<String>(segment.Get("name").value())
                             : String("kernel_" + std::to_string(index));
    String kernel_kind = segment.Get("kind")
                             ? Downcast<String>(segment.Get("kind").value())
                             : String("fused_dataflow");
    String core_type = segment.Get("core_type")
                           ? Downcast<String>(segment.Get("core_type").value())
                           : String("brisc");
    Array<TTRuntimeArgSpec> runtime_args =
        DecodeRuntimeArgSpecs(segment.Get("runtime_args"));
    Array<TTRuntimeArgSpec> common_runtime_args =
        DecodeRuntimeArgSpecs(segment.Get("common_runtime_args"));
    Array<TTCompileTimeArgSpec> compile_time_arg_specs =
        DecodeCompileTimeArgSpecs(segment.Get("compile_time_arg_specs"));
    Array<TTAccessorSpec> accessors =
        DecodeAccessorSpecs(segment.Get("accessors"));
    Array<TTSemaphoreBindingSpec> semaphore_bindings =
        DecodeSemaphoreBindingSpecs(segment.Get("semaphore_bindings"));
    TTKernelLaunchSpec launch_spec =
        segment.Get("launch_spec")
            ? Downcast<TTKernelLaunchSpec>(segment.Get("launch_spec").value())
            : TTKernelLaunchSpec(String(""), String(""), String(""));
    TTKernelComputeConfig compute_config =
        segment.Get("compute_config")
            ? Downcast<TTKernelComputeConfig>(
                  segment.Get("compute_config").value())
            : MakeEmptyComputeConfig();
    Array<TTPerWorkArgSpec> per_work_arg_specs =
        segment.Get(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs))
            ? Downcast<Array<TTPerWorkArgSpec>>(
                  segment
                      .Get(String(
                          blackhole_runtime_arg_schema::kPerWorkArgSpecs))
                      .value())
            : Array<TTPerWorkArgSpec>();
    abi_plans.push_back(TTABIPlan(String("abi_" + std::to_string(index)),
                                  kernel_name, runtime_args,
                                  common_runtime_args, compile_time_arg_specs,
                                  accessors, semaphore_bindings));
    kernels.push_back(TTKernel(kernel_name, kernel_kind, core_type, index,
                               launch_spec, compute_config,
                               per_work_arg_specs));
    ++index;
  }
  *kernels_out = kernels;
  *abi_plans_out = abi_plans;
}

static std::vector<std::string> CollectSegmentKindsFromBody(const Stmt &body) {
  class SegmentKindCollector : public tir::StmtVisitor {
  public:
    void VisitStmt_(const tir::AttrStmtNode *op) final {
      if (op->attr_key == "blackhole.segment_kind") {
        if (const auto *kind = op->value.as<tir::StringImmNode>()) {
          const std::string segment_kind = kind->value;
          if (seen_.insert(segment_kind).second) {
            segment_kinds_.push_back(segment_kind);
          }
        }
      }
      tir::StmtVisitor::VisitStmt_(op);
    }

    const std::vector<std::string> &segment_kinds() const {
      return segment_kinds_;
    }

  private:
    std::unordered_set<std::string> seen_;
    std::vector<std::string> segment_kinds_;
  };

  SegmentKindCollector collector;
  collector(body);
  return collector.segment_kinds();
}

static std::string CoreTypeForSegmentKind(const std::string &segment_kind) {
  if (segment_kind == "reader") {
    return "brisc";
  }
  if (segment_kind == "compute") {
    return "trisc";
  }
  if (segment_kind == "writer") {
    return "ncrisc";
  }
  return "brisc";
}

} // namespace

void PlanTTKernelABI::LoadSeededComputeOpPlans(const PrimFunc &func) {
  auto staged_program = func->GetAttr<TTProgram>(attr::kTLTTProgram);
  if (!staged_program) {
    return;
  }
  for (const TTComputeOpPlan &plan : staged_program.value()->compute_op_plans) {
    tt_compute_op_plans_.push_back(plan);
    const std::string operation_name = plan->operation_name;
    const bool is_broadcast_cols_op =
        operation_name.find("_bcast_cols") != std::string::npos;
    for (const TTComputeOperandBindingPlan &binding : plan->operand_bindings) {
      const std::string role = binding->role;
      const std::string buffer = binding->buffer;
      if (buffer.empty() || role == "output" || role == "c") {
        continue;
      }
      tile_compute_input_buffers_.insert(buffer);
      if (is_broadcast_cols_op && role == "rhs" &&
          static_cast<std::string>(binding->transform_kind) == "broadcast") {
        broadcast_cols_rhs_buffers_.insert(buffer);
      }
    }
  }
}

void PlanTTKernelABI::StoreSegmentPlan(PrimFunc &func) {
  const std::vector<std::string> segment_kinds =
      CollectSegmentKindsFromBody(func->body);
  if (segment_kinds.empty() && !needs_copy_runtime_args_ &&
      !requires_compute_segment_) {
    segment_plan_ = Array<Any>();
    return;
  }
  ICHECK(!requires_compute_segment_ || !segment_kinds.empty())
      << "PlanTTKernelABI requires explicit segment_kind truth for "
         "compute-bearing fragment "
         "workloads; do not recover them as fused_dataflow";

  Array<Any> kernels;
  if (segment_kinds.empty()) {
    Map<String, Any> kernel;
    kernel.Set("name", String("main"));
    kernel.Set("kind", String("fused_dataflow"));
    kernel.Set("core_type", String("brisc"));
    kernels.push_back(kernel);
  } else {
    for (const std::string &kind : segment_kinds) {
      Map<String, Any> kernel;
      kernel.Set("name", String(kind));
      kernel.Set("kind", String(kind));
      kernel.Set("core_type", String(CoreTypeForSegmentKind(kind)));
      kernels.push_back(kernel);
    }
  }
  segment_plan_ = kernels;
}

void PlanTTKernelABI::StoreAccessorDescriptors(PrimFunc &func) {
  auto make_compute_config_from_gemm_state = [&]() -> TTKernelComputeConfig {
    return TTKernelComputeConfig(
        String("HiFi4"), true, gemm_dst_full_sync_en_, false, Array<String>{},
        gemm_bfp8_pack_precise_, EncodeTTKernelDefines(gemm_defines_),
        EncodeTTKernelNamedCompileArgs(gemm_named_compile_args_),
        gemm_clear_accum_, gemm_k_pack_, gemm_wg_wait_, gemm_policy_type_,
        String(GemmWarpPolicyTypeToStringForBlackhole(gemm_policy_type_)));
  };

  auto make_launch_spec =
      [](const std::string &core_type) -> TTKernelLaunchSpec {
    if (core_type == "brisc") {
      return MakeLaunchSpec(core_type, "riscv_0", "riscv_0_default");
    }
    if (core_type == "ncrisc") {
      return MakeLaunchSpec(core_type, "riscv_1", "riscv_1_default");
    }
    if (core_type == "trisc") {
      return MakeLaunchSpec(core_type, "", "");
    }
    return TTKernelLaunchSpec(String(""), String(""), String(""));
  };

  auto make_accessor_cta_specs = [&](const std::string &kind,
                                     const Array<Any> &accessors) {
    Array<Any> compile_time_arg_specs;
    for (const auto &accessor_item : accessors) {
      auto accessor =
          accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (accessor.empty()) {
        continue;
      }
      const std::string buffer =
          accessor.Get("buffer") ? static_cast<std::string>(Downcast<String>(
                                       accessor.Get("buffer").value()))
                                 : std::string();
      ICHECK(accessor.Get("compile_time_arg_offset"))
          << "PlanTTKernelABI requires accessor compile_time_arg_offset for "
          << buffer;
      ICHECK(accessor.Get("compile_time_arg_count"))
          << "PlanTTKernelABI requires accessor compile_time_arg_count for "
          << buffer;
      ICHECK(accessor.Get("layout"))
          << "PlanTTKernelABI requires accessor layout for " << buffer;
      ICHECK(accessor.Get("memory_space"))
          << "PlanTTKernelABI requires accessor memory_space for " << buffer;
      ICHECK(accessor.Get("args_config_bits"))
          << "PlanTTKernelABI requires accessor args_config_bits for "
          << buffer;
      const int compile_time_arg_offset =
          Downcast<Integer>(accessor.Get("compile_time_arg_offset").value())
              .IntValue();
      const int compile_time_arg_count =
          Downcast<Integer>(accessor.Get("compile_time_arg_count").value())
              .IntValue();
      const std::string layout = static_cast<std::string>(
          Downcast<String>(accessor.Get("layout").value()));
      const std::string memory_space = static_cast<std::string>(
          Downcast<String>(accessor.Get("memory_space").value()));
      const int args_config_bits =
          Downcast<Integer>(accessor.Get("args_config_bits").value())
              .IntValue();
      const int transport_page_size =
          accessor.Get("transport_page_size")
              ? Downcast<Integer>(accessor.Get("transport_page_size").value())
                    .IntValue()
              : 0;
      std::vector<int64_t> host_axis_order;
      if (auto axis_order_value = accessor.Get("host_axis_order")) {
        for (const Any &axis : Downcast<Array<Any>>(axis_order_value.value())) {
          host_axis_order.push_back(Downcast<Integer>(axis).IntValue());
        }
      }
      const bool transpose_2d =
          accessor.Get("transpose_2d")
              ? Downcast<Bool>(accessor.Get("transpose_2d").value())
              : false;

      compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
          buffer, AccessorCompileTimeArgKind(layout, memory_space), "uint32",
          compile_time_arg_offset, compile_time_arg_count, kind, buffer, {},
          args_config_bits, transport_page_size, layout, memory_space,
          host_axis_order, transpose_2d));
    }
    return compile_time_arg_specs;
  };

  std::unordered_map<std::string, int> accessor_transport_page_sizes;
  PostOrderVisit(func->body, [&](const ObjectRef &node_ref) {
    const auto *call = node_ref.as<CallNode>();
    if (call == nullptr || !call->op->IsInstance<OpNode>() ||
        call->args.size() < 4) {
      return;
    }
    const std::string op_name = Downcast<Op>(call->op)->name;
    const auto *page_bytes = call->args[3].as<IntImmNode>();
    if (page_bytes == nullptr) {
      return;
    }
    if (op_name == "tl.blackhole.read_page_to_cb" ||
        op_name == "tl.blackhole.read_bcast_cols_to_cb") {
      if (!copy_input_buffer_name_.empty()) {
        accessor_transport_page_sizes[copy_input_buffer_name_] =
            page_bytes->value;
      }
    } else if (op_name == "tl.blackhole.write_page_from_cb") {
      if (!copy_output_buffer_name_.empty()) {
        accessor_transport_page_sizes[copy_output_buffer_name_] =
            page_bytes->value;
      }
    }
  });

  auto make_gemm_compute_cta_specs = [&]() {
    Array<Any> compile_time_arg_specs;
    if (gemm_a_buffer_name_.empty() || gemm_b_buffer_name_.empty() ||
        gemm_c_buffer_name_.empty()) {
      return compile_time_arg_specs;
    }
    ICHECK_EQ(gemm_m_ % kBlackholeTileRows, 0)
        << "Blackhole GEMM compile-time ABI requires M to be 32-aligned";
    ICHECK_EQ(gemm_k_ % kBlackholeTileCols, 0)
        << "Blackhole GEMM compile-time ABI requires K to be 32-aligned";
    ICHECK_EQ(gemm_n_ % kBlackholeTileCols, 0)
        << "Blackhole GEMM compile-time ABI requires N to be 32-aligned";
    const int mt = gemm_m_ / kBlackholeTileRows;
    const int kt = gemm_k_ / kBlackholeTileCols;
    const int nt = gemm_n_ / kBlackholeTileCols;
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_shape", "gemm_shape", "uint32", 0, 3, "compute", "",
        {static_cast<uint32_t>(mt), static_cast<uint32_t>(kt),
         static_cast<uint32_t>(nt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_transpose_flags", "gemm_transpose_flags", "uint32", 3, 2,
        "compute", "",
        {static_cast<uint32_t>(gemm_transpose_a_ ? 1 : 0),
         static_cast<uint32_t>(gemm_transpose_b_ ? 1 : 0)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_block_shape", "gemm_block_shape", "uint32", 5, 3, "compute", "",
        {static_cast<uint32_t>(mt), static_cast<uint32_t>(nt),
         static_cast<uint32_t>(kt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_subblock_shape", "gemm_subblock_shape", "uint32", 8, 2, "compute",
        "", {static_cast<uint32_t>(mt), static_cast<uint32_t>(nt)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_clear_accum", "gemm_clear_accum", "uint32", 10, 1, "compute", "",
        {static_cast<uint32_t>(gemm_clear_accum_ ? 1 : 0)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_k_pack", "gemm_k_pack", "uint32", 11, 1, "compute", "",
        {static_cast<uint32_t>(gemm_k_pack_)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_wg_wait", "gemm_wg_wait", "uint32", 12, 1, "compute", "",
        {static_cast<uint32_t>(gemm_wg_wait_)}));
    compile_time_arg_specs.push_back(MakeCompileTimeArgSpec(
        "gemm_policy", "gemm_policy", "uint32", 13, 1, "compute", "",
        {static_cast<uint32_t>(gemm_policy_type_)}));
    return compile_time_arg_specs;
  };

  auto make_segment_per_work_arg_specs = [&](const std::string &kind,
                                             const Array<Any> &runtime_args,
                                             const Optional<Any>
                                                 &existing_specs_opt) {
    Array<TTPerWorkArgSpec> per_work_arg_specs =
        existing_specs_opt
            ? Downcast<Array<TTPerWorkArgSpec>>(existing_specs_opt.value())
            : Array<TTPerWorkArgSpec>();
    const std::string copy_input_buffer_name =
        copy_input_buffer_.defined() ? BufferIdentityName(copy_input_buffer_)
                                     : copy_input_buffer_name_;
    const std::string copy_output_buffer_name =
        copy_output_buffer_.defined() ? BufferIdentityName(copy_output_buffer_)
                                      : copy_output_buffer_name_;

    auto runtime_arg_identity_for_kind =
        [&](const char *arg_kind) -> std::string {
      for (const Any &arg_item : runtime_args) {
        auto arg = arg_item.as<Map<String, Any>>().value_or(Map<String, Any>());
        if (!arg.Get("kind") || static_cast<std::string>(Downcast<String>(
                                    arg.Get("kind").value())) != arg_kind) {
          continue;
        }
        if (auto identity = arg.Get("identity")) {
          return Downcast<String>(identity.value());
        }
        return MakeBlackholeRuntimeArgIdentity(arg_kind, arg_kind);
      }
      return "";
    };
    auto runtime_args_contain_kind = [&](const char *arg_kind) {
      return !runtime_arg_identity_for_kind(arg_kind).empty();
    };
    auto upsert_spec = [&](const TTPerWorkArgSpec &spec) {
      const std::string arg_identity =
          static_cast<std::string>(spec->arg_identity);
      for (int i = 0; i < per_work_arg_specs.size(); ++i) {
        const TTPerWorkArgSpec &existing = per_work_arg_specs[i];
        const std::string existing_arg_identity =
            static_cast<std::string>(existing->arg_identity);
        if (existing_arg_identity == arg_identity) {
          per_work_arg_specs.Set(i, spec);
          return;
        }
      }
      per_work_arg_specs.push_back(spec);
    };

    if (kind == "fused_dataflow") {
      if (runtime_args_contain_kind("a_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_start_id", runtime_arg_identity_for_kind("a_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            copy_input_buffer_name));
      }
      if (runtime_args_contain_kind("a_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_num_tiles",
            runtime_arg_identity_for_kind("a_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_input_buffer_name, 1));
      }
      if (runtime_args_contain_kind("a_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_stride", runtime_arg_identity_for_kind("a_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_input_buffer_name, 1));
      }
      if (runtime_args_contain_kind("output_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_start_id",
            runtime_arg_identity_for_kind("output_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            copy_output_buffer_name));
      }
      if (runtime_args_contain_kind("output_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_num_tiles",
            runtime_arg_identity_for_kind("output_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_output_buffer_name, 1));
      }
      if (runtime_args_contain_kind("output_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_stride",
            runtime_arg_identity_for_kind("output_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            copy_output_buffer_name, 1));
      }
    }
    if (kind == "reader") {
      const bool has_gemm_reader_contract =
          !gemm_a_buffer_name_.empty() && !gemm_b_buffer_name_.empty();
      if (runtime_args_contain_kind("a_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_start_id", runtime_arg_identity_for_kind("a_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            has_gemm_reader_contract
                ? blackhole_runtime_arg_schema::kValueSourceLogicalBlockY
                : blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            gemm_a_buffer_name_, has_gemm_reader_contract ? 0 : 0));
      }
      if (runtime_args_contain_kind("a_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_num_tiles",
            runtime_arg_identity_for_kind("a_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            has_gemm_reader_contract
                ? blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles
                : blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_a_buffer_name_, has_gemm_reader_contract ? 0 : 1));
      }
      if (runtime_args_contain_kind("a_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "a_tile_stride", runtime_arg_identity_for_kind("a_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_a_buffer_name_, 1));
      }
      if (runtime_args_contain_kind("b_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_start_id", runtime_arg_identity_for_kind("b_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            has_gemm_reader_contract
                ? blackhole_runtime_arg_schema::kValueSourceLogicalBlockX
                : blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            gemm_b_buffer_name_, has_gemm_reader_contract ? 0 : 0));
      }
      if (runtime_args_contain_kind("b_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_num_tiles",
            runtime_arg_identity_for_kind("b_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            has_gemm_reader_contract
                ? blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles
                : blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_b_buffer_name_, has_gemm_reader_contract ? 0 : 1));
      }
      if (runtime_args_contain_kind("b_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "b_tile_stride", runtime_arg_identity_for_kind("b_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            has_gemm_reader_contract
                ? blackhole_runtime_arg_schema::kValueSourceComputeLogicalNTiles
                : blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_b_buffer_name_, has_gemm_reader_contract ? 0 : 1));
      }
    }
    if (kind == "reader" || kind == "compute") {
      if (runtime_args_contain_kind("k_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "k_tile_start_id", runtime_arg_identity_for_kind("k_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorKTileStart,
            blackhole_runtime_arg_schema::kValueSourceConstant, "", 0));
      }
      if (runtime_args_contain_kind("num_k_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "num_k_tiles", runtime_arg_identity_for_kind("num_k_tiles"),
            blackhole_runtime_arg_schema::kDescriptorKTileCount,
            !gemm_a_buffer_name_.empty()
                ? blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles
                : blackhole_runtime_arg_schema::kValueSourceConstant,
            "", !gemm_a_buffer_name_.empty() ? 0 : 1));
      }
    }
    if (kind == "writer") {
      if (runtime_args_contain_kind("output_tile_start_id")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_start_id",
            runtime_arg_identity_for_kind("output_tile_start_id"),
            blackhole_runtime_arg_schema::kDescriptorTileStart,
            blackhole_runtime_arg_schema::kValueSourceWorkLinearId,
            gemm_c_buffer_name_));
      }
      if (runtime_args_contain_kind("output_tile_num_tiles")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_num_tiles",
            runtime_arg_identity_for_kind("output_tile_num_tiles"),
            blackhole_runtime_arg_schema::kDescriptorTileCount,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_c_buffer_name_, 1));
      }
      if (runtime_args_contain_kind("output_tile_stride")) {
        upsert_spec(MakePerWorkArgSpec(
            "output_tile_stride",
            runtime_arg_identity_for_kind("output_tile_stride"),
            blackhole_runtime_arg_schema::kDescriptorTileStride,
            blackhole_runtime_arg_schema::kValueSourceConstant,
            gemm_c_buffer_name_, 1));
      }
    }
    return per_work_arg_specs;
  };

  if (!segment_plan_.empty()) {
    Array<Any> rewritten_segments;
    for (const auto &item : segment_plan_) {
      auto segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (segment.empty()) {
        rewritten_segments.push_back(item);
        continue;
      }
      const std::string kind = segment.Get("kind")
                                   ? static_cast<std::string>(Downcast<String>(
                                         segment.Get("kind").value()))
                                   : std::string();
      Array<Any> accessors;
      Array<Any> compile_time_arg_specs;
      if (auto accessor_items = segment.Get("accessors")) {
        for (const auto &accessor_item :
             Downcast<Array<Any>>(accessor_items.value())) {
          auto accessor =
              accessor_item.as<Map<String, Any>>().value_or(Map<String, Any>());
          if (accessor.empty()) {
            accessors.push_back(accessor_item);
            continue;
          }

          const std::string buffer_name =
              accessor.Get("buffer")
                  ? static_cast<std::string>(
                        Downcast<String>(accessor.Get("buffer").value()))
                  : std::string();
          ICHECK(accessor.Get("compile_time_arg_offset"))
              << "PlanTTKernelABI requires accessor compile_time_arg_offset "
                 "for "
              << buffer_name;
          ICHECK(accessor.Get("compile_time_arg_count"))
              << "PlanTTKernelABI requires accessor compile_time_arg_count for "
              << buffer_name;
          ICHECK(accessor.Get("common_runtime_arg_offset"))
              << "PlanTTKernelABI requires accessor common_runtime_arg_offset "
                 "for "
              << buffer_name;
          ICHECK(accessor.Get("common_runtime_arg_count"))
              << "PlanTTKernelABI requires accessor common_runtime_arg_count "
                 "for "
              << buffer_name;
          ICHECK(accessor.Get("layout"))
              << "PlanTTKernelABI requires accessor layout for " << buffer_name;
          ICHECK(accessor.Get("memory_space"))
              << "PlanTTKernelABI requires accessor memory_space for "
              << buffer_name;
          ICHECK(accessor.Get("args_config_bits"))
              << "PlanTTKernelABI requires accessor args_config_bits for "
              << buffer_name;
          const int compile_time_arg_offset =
              Downcast<Integer>(accessor.Get("compile_time_arg_offset").value())
                  .IntValue();
          const int compile_time_arg_count =
              Downcast<Integer>(accessor.Get("compile_time_arg_count").value())
                  .IntValue();
          const int common_runtime_arg_offset =
              Downcast<Integer>(
                  accessor.Get("common_runtime_arg_offset").value())
                  .IntValue();
          const int common_runtime_arg_count =
              Downcast<Integer>(
                  accessor.Get("common_runtime_arg_count").value())
                  .IntValue();
          const std::string layout = static_cast<std::string>(
              Downcast<String>(accessor.Get("layout").value()));
          const std::string memory_space = static_cast<std::string>(
              Downcast<String>(accessor.Get("memory_space").value()));
          const int args_config_bits =
              Downcast<Integer>(accessor.Get("args_config_bits").value())
                  .IntValue();
          const int transport_page_size =
              accessor.Get("transport_page_size")
                  ? Downcast<Integer>(
                        accessor.Get("transport_page_size").value())
                        .IntValue()
                  : 0;
          const AccessorDescriptor *matched_descriptor = nullptr;
          for (const AccessorDescriptor &desc : accessor_descriptors_) {
            const bool matches = desc.segment_kind == kind &&
                                 desc.buffer_name == buffer_name &&
                                 desc.compile_time_arg_offset ==
                                     compile_time_arg_offset &&
                                 desc.compile_time_arg_count ==
                                     compile_time_arg_count &&
                                 desc.common_runtime_arg_offset ==
                                     common_runtime_arg_offset &&
                                 desc.common_runtime_arg_count ==
                                     common_runtime_arg_count &&
                                 desc.args_config_bits == args_config_bits;
            if (!matches) {
              continue;
            }
            if (matched_descriptor != nullptr) {
              ICHECK(matched_descriptor->layout == desc.layout &&
                     matched_descriptor->memory_space == desc.memory_space &&
                     matched_descriptor->transport_page_size_bytes ==
                         desc.transport_page_size_bytes)
                  << "PlanTTKernelABI found conflicting accessor descriptors "
                     "for buffer "
                  << buffer_name
                  << "; a single accessor slot cannot mix page-indexed and "
                     "interleaved transport metadata";
              continue;
            }
            matched_descriptor = &desc;
          }
          std::string resolved_layout = layout;
          std::string resolved_memory_space = memory_space;
          int resolved_transport_page_size = transport_page_size;
          if (matched_descriptor != nullptr) {
            resolved_layout = matched_descriptor->layout;
            resolved_memory_space = matched_descriptor->memory_space;
            if (matched_descriptor->transport_page_size_bytes > 0) {
              resolved_transport_page_size =
                  matched_descriptor->transport_page_size_bytes;
            }
          }
          if (resolved_transport_page_size == 0 && accessor.Get("buffer")) {
            auto transport_it = accessor_transport_page_sizes.find(buffer_name);
            if (transport_it != accessor_transport_page_sizes.end()) {
              resolved_transport_page_size = transport_it->second;
            }
          }

          accessor.Set("compile_time_arg_offset",
                       Integer(compile_time_arg_offset));
          accessor.Set("compile_time_arg_count",
                       Integer(compile_time_arg_count));
          accessor.Set("common_runtime_arg_offset",
                       Integer(common_runtime_arg_offset));
          accessor.Set("common_runtime_arg_count",
                       Integer(common_runtime_arg_count));
          accessor.Set("args_config_bits", Integer(args_config_bits));
          if (resolved_transport_page_size > 0) {
            accessor.Set("transport_page_size",
                         Integer(resolved_transport_page_size));
          }
          accessor.Set("layout", String(resolved_layout));
          accessor.Set("memory_space", String(resolved_memory_space));
          accessors.push_back(accessor);
        }
      }
      if (accessors.empty()) {
        accessors = EncodeAccessorDescriptors(kind);
      }
      Array<Any> runtime_args = EnsureSegmentBufferRuntimeArgs(
          kind, accessors, segment.Get("runtime_args"),
          copy_input_buffer_.defined() ? BufferIdentityName(copy_input_buffer_)
                                       : copy_input_buffer_name_,
          copy_output_buffer_.defined()
              ? BufferIdentityName(copy_output_buffer_)
              : copy_output_buffer_name_,
          copy_input_buffer_names_, copy_output_buffer_names_);
      Array<TTPerWorkArgSpec> per_work_arg_specs =
          make_segment_per_work_arg_specs(
              kind, runtime_args,
              segment.Get(
                  String(blackhole_runtime_arg_schema::kPerWorkArgSpecs)));
      compile_time_arg_specs = make_accessor_cta_specs(kind, accessors);
      if (kind == "compute") {
        if (compute_op_signatures_.size() == 1) {
          auto gemm_compile_time_arg_specs = make_gemm_compute_cta_specs();
          for (const auto &spec : gemm_compile_time_arg_specs) {
            compile_time_arg_specs.push_back(spec);
          }
        }
        segment.Set("compute_config", make_compute_config_from_gemm_state());
      }
      segment.Set("accessors", accessors);
      if (!runtime_args.empty()) {
        segment.Set("runtime_args", runtime_args);
      }
      if (!per_work_arg_specs.empty()) {
        segment.Set(String(blackhole_runtime_arg_schema::kPerWorkArgSpecs),
                    per_work_arg_specs);
      }
      segment.Set("compile_time_arg_specs", compile_time_arg_specs);
      TTKernelLaunchSpec launch_spec = make_launch_spec(
          segment.Get("core_type") ? static_cast<std::string>(Downcast<String>(
                                         segment.Get("core_type").value()))
                                   : std::string());
      if (!launch_spec->core_type.empty()) {
        segment.Set("launch_spec", launch_spec);
      }
      segment.Set("common_runtime_args", EncodeCommonRuntimeArgs(kind));
      rewritten_segments.push_back(segment);
    }
    segment_plan_ = rewritten_segments;
    FinalizeMaterializationPlanHostBuffers();
    String compute_kernel_name;
    for (const Any &item : segment_plan_) {
      auto segment = item.as<Map<String, Any>>().value_or(Map<String, Any>());
      if (segment.empty()) {
        continue;
      }
      const std::string kind = segment.Get("kind")
                                   ? static_cast<std::string>(Downcast<String>(
                                         segment.Get("kind").value()))
                                   : std::string();
      const std::string core_type =
          segment.Get("core_type") ? static_cast<std::string>(Downcast<String>(
                                         segment.Get("core_type").value()))
                                   : std::string();
      if (kind == "compute" || core_type == "trisc") {
        compute_kernel_name =
            segment.Get("name") ? Downcast<String>(segment.Get("name").value())
                                : String("compute");
        break;
      }
    }
    Array<TTComputeOpPlan> exact_compute_op_plans = tt_compute_op_plans_;
    tt_compute_op_plans_.clear();
    for (const TTComputeOpPlan &plan : exact_compute_op_plans) {
      tt_compute_op_plans_.push_back(plan);
    }
    if (!gemm_compute_op_facts_.empty()) {
      ICHECK(!compute_kernel_name.empty())
          << "PlanTTKernelABI produced GEMM compute op facts without a compute "
             "kernel segment";
      int64_t ordinal = 0;
      for (const auto &fact : gemm_compute_op_facts_) {
        tt_compute_op_plans_.push_back(BuildTTComputeOpPlanFromFact(
            fact, host_buffer_by_compute_operand_buffer_, compute_kernel_name,
            /*kernel_plan_index=*/-1, ordinal++));
      }
    }
    BuildTTKernelAndABISeeds(segment_plan_, &tt_kernels_, &tt_abi_plans_);
    FinalizeConsumerBindingABIIndices();
  }
}

Array<Any> PlanTTKernelABI::EncodeAccessorDescriptors(
    const std::string &segment_kind) const {
  Array<Any> accessors;
  for (const auto &desc : accessor_descriptors_) {
    if (desc.segment_kind != segment_kind) {
      continue;
    }
    Map<String, Any> accessor;
    accessor.Set("buffer", String(desc.buffer_name));
    accessor.Set("compile_time_arg_offset",
                 Integer(desc.compile_time_arg_offset));
    accessor.Set("compile_time_arg_count",
                 Integer(desc.compile_time_arg_count));
    accessor.Set("common_runtime_arg_offset",
                 Integer(desc.common_runtime_arg_offset));
    accessor.Set("common_runtime_arg_count",
                 Integer(desc.common_runtime_arg_count));
    accessor.Set("args_config_bits", Integer(desc.args_config_bits));
    if (desc.transport_page_size_bytes > 0) {
      accessor.Set("transport_page_size",
                   Integer(desc.transport_page_size_bytes));
    }
    accessor.Set("layout", String(desc.layout));
    accessor.Set("memory_space", String(desc.memory_space));
    if (!desc.host_axis_order.empty()) {
      Array<Any> axis_order;
      for (int64_t axis : desc.host_axis_order) {
        axis_order.push_back(Integer(axis));
      }
      accessor.Set("host_axis_order", axis_order);
    }
    if (desc.transpose_2d) {
      accessor.Set("transpose_2d", Bool(true));
    }
    accessors.push_back(accessor);
  }
  return accessors;
}

Array<Any> PlanTTKernelABI::EncodeCommonRuntimeArgs(
    const std::string &segment_kind) const {
  (void)segment_kind;
  return Array<Any>{};
}

std::string PlanTTKernelABI::ResolveHostBufferForComputeOperand(
    const Buffer &buffer) const {
  const std::string buffer_name = BufferIdentityName(buffer);
  auto it = host_buffer_by_compute_operand_buffer_.find(buffer_name);
  if (it != host_buffer_by_compute_operand_buffer_.end() &&
      !it->second.empty()) {
    return it->second;
  }
  return "";
}

std::string PlanTTKernelABI::ComputeKernelNameForCurrentPlan() const {
  if (!current_segment_kind_.empty()) {
    return current_segment_kind_;
  }
  return requires_compute_segment_ ? std::string("compute")
                                   : std::string("main");
}

void PlanTTKernelABI::RecordExactComputeOpPlan(
    const std::string &kind, const std::string &operation_name,
    const std::vector<ComputeOperandPlanSeed> &operands) {
  if (operation_name.empty() || operands.empty()) {
    return;
  }

  Array<TTComputeOperandBindingPlan> operand_bindings;
  const Buffer *output_buffer = nullptr;
  for (const ComputeOperandPlanSeed &operand : operands) {
    if (!operand.buffer.defined()) {
      continue;
    }
    const std::string buffer_name = BufferIdentityName(operand.buffer);
    if (buffer_name.empty()) {
      continue;
    }
    if (operand.role == "output" || operand.role == "c") {
      output_buffer = &operand.buffer;
    }
    const std::string data_format =
        DataTypeToDataFormatForBlackhole(operand.buffer->dtype);
    operand_bindings.push_back(TTComputeOperandBindingPlan(
        String(operand.role), String(buffer_name),
        String(ResolveHostBufferForComputeOperand(operand.buffer)),
        String(data_format), String(data_format),
        String(operand.transform_kind.empty() ? "identity"
                                              : operand.transform_kind)));
  }
  if (operand_bindings.empty()) {
    return;
  }
  std::vector<std::string> operand_roles;
  operand_roles.reserve(operand_bindings.size());
  for (const TTComputeOperandBindingPlan &binding : operand_bindings) {
    operand_roles.push_back(static_cast<std::string>(binding->role));
  }
  const BlackholeTileComputeCoveringDecision covering =
      SelectBlackholeTileComputeCovering(operation_name);
  ICHECK(covering.selected) << "TileCompute covering rejected operation "
                            << operation_name << ": " << covering.reject_reason;
  ICHECK_EQ(kind, covering.result_kind)
      << "TileCompute covering selected result kind " << covering.result_kind
      << " for " << operation_name << ", but caller recorded " << kind;
  RequireLegalBlackholeTileComputeSelection(
      covering.result_kind, covering.operation_name, operand_roles);

  if (output_buffer == nullptr) {
    output_buffer = &operands.back().buffer;
  }

  Array<String> problem_shape_axes;
  Array<Integer> problem_shape;
  Array<Integer> tile_shape;
  std::string accumulator_dtype;
  if (output_buffer != nullptr && output_buffer->defined()) {
    const std::vector<int64_t> logical_shape =
        GetLogicalBufferShape(*output_buffer);
    for (size_t i = 0; i < logical_shape.size(); ++i) {
      if (logical_shape.size() == 1U) {
        problem_shape_axes.push_back(String("elements"));
      } else if (logical_shape.size() == 2U) {
        problem_shape_axes.push_back(String(i == 0 ? "rows" : "cols"));
      } else {
        problem_shape_axes.push_back(String("dim" + std::to_string(i)));
      }
      problem_shape.push_back(Integer(logical_shape[i]));
    }
    if (logical_shape.size() >= 2U) {
      const int64_t rows = logical_shape[logical_shape.size() - 2];
      const int64_t cols = logical_shape[logical_shape.size() - 1];
      tile_shape.push_back(Integer(
          std::max<int64_t>(1, CeilDivToInt(rows, kBlackholeTileRows))));
      tile_shape.push_back(Integer(
          std::max<int64_t>(1, CeilDivToInt(cols, kBlackholeTileCols))));
    } else if (logical_shape.size() == 1U) {
      tile_shape.push_back(Integer(std::max<int64_t>(
          1, CeilDivToInt(logical_shape[0],
                          kBlackholeTileRows * kBlackholeTileCols))));
    }
    accumulator_dtype =
        DataTypeToDataFormatForBlackhole((*output_buffer)->dtype);
  }

  const std::string kernel_name = ComputeKernelNameForCurrentPlan();
  const std::string plan_name = "compute_op_" + kernel_name + "_" +
                                operation_name + "_" +
                                std::to_string(tt_compute_op_plans_.size());
  tt_compute_op_plans_.push_back(TTComputeOpPlan(
      String(plan_name), String(kernel_name), /*kernel_plan_index=*/-1,
      String(covering.result_kind), String(covering.operation_name), Bool(true),
      operand_bindings, problem_shape_axes, problem_shape, tile_shape,
      Array<Integer>{}, Array<Integer>{}, String(accumulator_dtype), String(""),
      String(""), Array<String>{}, CurrentTileComputeDAGNodeId(),
      CurrentTileComputeDAGSourceEmitter(),
      CurrentTileComputeDAGMaterializationPolicy(),
      CurrentTileComputeDAGFanoutUseCount(),
      CurrentTileComputeDAGFanoutPolicy()));
}

void PlanTTKernelABI::RegisterAccessor(
    const std::string &segment_kind, const Buffer &buffer,
    int compile_time_arg_offset, int compile_time_arg_count,
    int common_runtime_arg_offset, int common_runtime_arg_count,
    int args_config_bits, int transport_page_size_bytes,
    std::vector<int64_t> host_axis_order, bool transpose_2d,
    const std::string &layout, const std::string &memory_space) {
  auto it = std::find_if(
      accessor_descriptors_.begin(), accessor_descriptors_.end(),
      [&](const AccessorDescriptor &desc) {
        return desc.segment_kind == segment_kind &&
               SameBufferIdentity(desc.buffer, buffer) &&
               desc.compile_time_arg_offset == compile_time_arg_offset &&
               desc.compile_time_arg_count == compile_time_arg_count &&
               desc.common_runtime_arg_offset == common_runtime_arg_offset &&
               desc.common_runtime_arg_count == common_runtime_arg_count &&
               desc.args_config_bits == args_config_bits &&
               desc.transport_page_size_bytes == transport_page_size_bytes &&
               desc.host_axis_order == host_axis_order &&
               desc.transpose_2d == transpose_2d &&
               desc.layout == layout &&
               desc.memory_space == memory_space;
      });
  if (it != accessor_descriptors_.end()) {
    return;
  }
  accessor_descriptors_.push_back(AccessorDescriptor{
      segment_kind, buffer, BufferIdentityName(buffer), compile_time_arg_offset,
      compile_time_arg_count, common_runtime_arg_offset,
      common_runtime_arg_count, args_config_bits, transport_page_size_bytes,
      layout, memory_space, std::move(host_axis_order), transpose_2d});
}

std::string
PlanTTKernelABI::ResolveAccessorSegmentKind(CopyDirection direction) const {
  if (!current_segment_kind_.empty()) {
    return current_segment_kind_;
  }
  if (requires_compute_segment_) {
    if (direction == CopyDirection::kDramToCB) {
      return "reader";
    }
    if (direction == CopyDirection::kCBToDram) {
      return "writer";
    }
    return "fused_dataflow";
  }
  if (direction == CopyDirection::kDramToCB) {
    return !gemm_a_buffer_name_.empty() ? "reader" : "fused_dataflow";
  }
  if (direction == CopyDirection::kCBToDram ||
      direction == CopyDirection::kLocalToCB) {
    return !gemm_a_buffer_name_.empty() ? "writer" : "fused_dataflow";
  }
  return "fused_dataflow";
}

int PlanTTKernelABI::GetOrAllocateSegmentAccessorSlot(
    std::unordered_map<std::string, int> *slot_map,
    const std::string &segment_kind, const Buffer &buffer) {
  const std::string key = MakeSegmentBufferKey(segment_kind, buffer);
  auto it = slot_map->find(key);
  if (it != slot_map->end()) {
    return it->second;
  }
  int next_slot = 0;
  for (const auto &[existing_key, slot] : *slot_map) {
    if (existing_key.rfind(segment_kind + ":", 0) == 0) {
      next_slot = std::max(next_slot, slot + 2);
    }
  }
  slot_map->emplace(key, next_slot);
  return next_slot;
}

int PlanTTKernelABI::GetReadAccessorSlot(const std::string &segment_kind,
                                         const Buffer &buffer,
                                         CopyDirection direction) {
  if (segment_kind == "fused_dataflow") {
    return GetOrAllocateSegmentAccessorSlot(&fused_dataflow_accessor_slots_,
                                            segment_kind, buffer);
  }
  if (direction == CopyDirection::kDramToCB) {
    return GetOrAllocateSegmentAccessorSlot(&read_accessor_slots_, segment_kind,
                                            buffer);
  }
  return 0;
}

int PlanTTKernelABI::GetWriteAccessorSlot(const std::string &segment_kind,
                                          const Buffer &buffer,
                                          CopyDirection direction) {
  if (segment_kind == "fused_dataflow") {
    return GetOrAllocateSegmentAccessorSlot(&fused_dataflow_accessor_slots_,
                                            segment_kind, buffer);
  }
  if (direction == CopyDirection::kCBToDram) {
    return GetOrAllocateSegmentAccessorSlot(&write_accessor_slots_,
                                            segment_kind, buffer);
  }
  return 0;
}

} // namespace tl
} // namespace tvm
