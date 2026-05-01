/*!
 * \file target/blackhole_module.cc
 * \brief Unified Blackhole module implementation
 *
 * This file provides the direct TT-Metal execution path for Blackhole kernels.
 */

#include "blackhole_module.h"

#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"

#ifdef TILELANG_BLACKHOLE_DIRECT
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#endif

namespace tvm {
namespace runtime {

static std::string NormalizeBlackholeKernelSource(std::string source) {
  const std::string old_compute_include = "#include \"compute_kernel_api.h\"";
  const std::string new_compute_include = "#include \"api/compute/compute_kernel_api.h\"";
  size_t pos = source.find(old_compute_include);
  if (pos != std::string::npos) {
    source.replace(pos, old_compute_include.size(), new_compute_include);
  }
  return source;
}

static std::string EncodeExecutableSpecMetadata(const ExecutableSpec& spec) {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  spec.Save(&writer);
  return os.str();
}

static constexpr const char* kBlackholeModuleSerializationMagic =
    "tilelang.blackhole.module.v1";

static uint64_t ReadUInt64(dmlc::Stream* stream, const char* field) {
  uint64_t value = 0;
  ICHECK(stream->Read(&value)) << "BlackholeModule LoadFromBytes missing " << field;
  return value;
}

static int64_t ReadInt64(dmlc::Stream* stream, const char* field) {
  int64_t value = 0;
  ICHECK(stream->Read(&value)) << "BlackholeModule LoadFromBytes missing " << field;
  return value;
}

static uint32_t ReadUInt32(dmlc::Stream* stream, const char* field) {
  const uint64_t value = ReadUInt64(stream, field);
  ICHECK_LE(value, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      << "BlackholeModule LoadFromBytes field " << field << " exceeds uint32 range";
  return static_cast<uint32_t>(value);
}

static int32_t ReadInt32(dmlc::Stream* stream, const char* field) {
  const int64_t value = ReadInt64(stream, field);
  ICHECK_GE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::min()))
      << "BlackholeModule LoadFromBytes field " << field << " is below int32 range";
  ICHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "BlackholeModule LoadFromBytes field " << field << " exceeds int32 range";
  return static_cast<int32_t>(value);
}

static bool ReadBool(dmlc::Stream* stream, const char* field) {
  bool value = false;
  ICHECK(stream->Read(&value)) << "BlackholeModule LoadFromBytes missing " << field;
  return value;
}

static std::string ReadString(dmlc::Stream* stream, const char* field) {
  std::string value;
  ICHECK(stream->Read(&value)) << "BlackholeModule LoadFromBytes missing " << field;
  return value;
}

static void WriteUInt64(dmlc::Stream* stream, uint64_t value) {
  stream->Write(value);
}

static void WriteInt64(dmlc::Stream* stream, int64_t value) {
  stream->Write(value);
}

static void WriteUInt32(dmlc::Stream* stream, uint32_t value) {
  WriteUInt64(stream, static_cast<uint64_t>(value));
}

static void WriteInt32(dmlc::Stream* stream, int32_t value) {
  WriteInt64(stream, static_cast<int64_t>(value));
}

static void WriteBool(dmlc::Stream* stream, bool value) {
  stream->Write(value);
}

static void WriteString(dmlc::Stream* stream, const std::string& value) {
  stream->Write(value);
}

template <typename T, typename WriteOne>
static void WriteVectorField(dmlc::Stream* stream, const std::vector<T>& values,
                             WriteOne write_one) {
  WriteUInt64(stream, static_cast<uint64_t>(values.size()));
  for (const auto& value : values) {
    write_one(stream, value);
  }
}

template <typename T, typename ReadOne>
static std::vector<T> ReadVectorField(dmlc::Stream* stream, const char* field,
                                      ReadOne read_one) {
  const uint64_t size = ReadUInt64(stream, field);
  ICHECK_LE(size, 10000000ULL)
      << "BlackholeModule LoadFromBytes field " << field << " has unreasonable size";
  std::vector<T> values;
  values.reserve(static_cast<size_t>(size));
  for (uint64_t i = 0; i < size; ++i) {
    values.push_back(read_one(stream));
  }
  return values;
}

static void WriteUInt32Vector(dmlc::Stream* stream, const std::vector<uint32_t>& values) {
  WriteVectorField<uint32_t>(stream, values,
                             [](dmlc::Stream* stream, uint32_t value) {
                               WriteUInt32(stream, value);
                             });
}

static std::vector<uint32_t> ReadUInt32Vector(dmlc::Stream* stream, const char* field) {
  return ReadVectorField<uint32_t>(stream, field, [field](dmlc::Stream* stream) {
    return ReadUInt32(stream, field);
  });
}

static void WriteInt64Vector(dmlc::Stream* stream, const std::vector<int64_t>& values) {
  WriteVectorField<int64_t>(stream, values,
                            [](dmlc::Stream* stream, int64_t value) {
                              WriteInt64(stream, value);
                            });
}

static std::vector<int64_t> ReadInt64Vector(dmlc::Stream* stream, const char* field) {
  return ReadVectorField<int64_t>(stream, field, [field](dmlc::Stream* stream) {
    return ReadInt64(stream, field);
  });
}

static void WriteStringVector(dmlc::Stream* stream, const std::vector<std::string>& values) {
  WriteVectorField<std::string>(stream, values,
                                [](dmlc::Stream* stream, const std::string& value) {
                                  WriteString(stream, value);
                                });
}

static std::vector<std::string> ReadStringVector(dmlc::Stream* stream, const char* field) {
  return ReadVectorField<std::string>(stream, field, [field](dmlc::Stream* stream) {
    return ReadString(stream, field);
  });
}

static void WriteBoolVector(dmlc::Stream* stream, const std::vector<bool>& values) {
  WriteUInt64(stream, static_cast<uint64_t>(values.size()));
  for (bool value : values) {
    WriteBool(stream, value);
  }
}

static std::vector<bool> ReadBoolVector(dmlc::Stream* stream, const char* field) {
  const uint64_t size = ReadUInt64(stream, field);
  ICHECK_LE(size, 10000000ULL)
      << "BlackholeModule LoadFromBytes field " << field << " has unreasonable size";
  std::vector<bool> values;
  values.reserve(static_cast<size_t>(size));
  for (uint64_t i = 0; i < size; ++i) {
    values.push_back(ReadBool(stream, field));
  }
  return values;
}

static void WriteDLDataType(dmlc::Stream* stream, DLDataType dtype) {
  WriteUInt32(stream, static_cast<uint32_t>(dtype.code));
  WriteUInt32(stream, static_cast<uint32_t>(dtype.bits));
  WriteUInt32(stream, static_cast<uint32_t>(dtype.lanes));
}

static DLDataType ReadDLDataType(dmlc::Stream* stream) {
  const uint32_t code = ReadUInt32(stream, "dtype.code");
  const uint32_t bits = ReadUInt32(stream, "dtype.bits");
  const uint32_t lanes = ReadUInt32(stream, "dtype.lanes");
  ICHECK_LE(code, static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()));
  ICHECK_LE(bits, static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()));
  ICHECK_LE(lanes, static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));
  return DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                    static_cast<uint16_t>(lanes)};
}

static void WriteDLDataTypeVector(dmlc::Stream* stream, const std::vector<DLDataType>& values) {
  WriteVectorField<DLDataType>(stream, values,
                               [](dmlc::Stream* stream, DLDataType dtype) {
                                 WriteDLDataType(stream, dtype);
                               });
}

static std::vector<DLDataType> ReadDLDataTypeVector(dmlc::Stream* stream, const char* field) {
  return ReadVectorField<DLDataType>(stream, field,
                                     [](dmlc::Stream* stream) {
                                       return ReadDLDataType(stream);
                                     });
}

static void WritePhysicalCore(dmlc::Stream* stream, const PhysicalCore& spec) {
  WriteUInt32(stream, spec.core_x);
  WriteUInt32(stream, spec.core_y);
}

static PhysicalCore ReadPhysicalCore(dmlc::Stream* stream) {
  PhysicalCore spec;
  spec.core_x = ReadUInt32(stream, "physical_core.core_x");
  spec.core_y = ReadUInt32(stream, "physical_core.core_y");
  return spec;
}

static void WriteWorkPacket(dmlc::Stream* stream, const WorkPacket& spec) {
  WriteUInt32(stream, spec.core_x);
  WriteUInt32(stream, spec.core_y);
  WriteUInt32(stream, spec.work_offset);
  WriteUInt32(stream, spec.work_count);
}

static WorkPacket ReadWorkPacket(dmlc::Stream* stream) {
  WorkPacket spec;
  spec.core_x = ReadUInt32(stream, "work_packet.core_x");
  spec.core_y = ReadUInt32(stream, "work_packet.core_y");
  spec.work_offset = ReadUInt32(stream, "work_packet.work_offset");
  spec.work_count = ReadUInt32(stream, "work_packet.work_count");
  return spec;
}

static void WriteCorePlan(dmlc::Stream* stream, const CorePlan& spec) {
  WriteUInt32(stream, spec.logical_grid_x);
  WriteUInt32(stream, spec.logical_grid_y);
  WriteString(stream, spec.linearization);
  WriteVectorField<PhysicalCore>(stream, spec.physical_cores, WritePhysicalCore);
  WriteVectorField<WorkPacket>(stream, spec.work_packets, WriteWorkPacket);
}

static CorePlan ReadCorePlan(dmlc::Stream* stream) {
  CorePlan spec;
  spec.logical_grid_x = ReadUInt32(stream, "core_plan.logical_grid_x");
  spec.logical_grid_y = ReadUInt32(stream, "core_plan.logical_grid_y");
  spec.linearization = ReadString(stream, "core_plan.linearization");
  spec.physical_cores = ReadVectorField<PhysicalCore>(
      stream, "core_plan.physical_cores", ReadPhysicalCore);
  spec.work_packets = ReadVectorField<WorkPacket>(
      stream, "core_plan.work_packets", ReadWorkPacket);
  return spec;
}

static void WriteCoreRangeSpec(dmlc::Stream* stream, const CoreRangeSpec& spec) {
  WritePhysicalCore(stream, spec.start);
  WritePhysicalCore(stream, spec.end);
}

static CoreRangeSpec ReadCoreRangeSpec(dmlc::Stream* stream) {
  CoreRangeSpec spec;
  spec.start = ReadPhysicalCore(stream);
  spec.end = ReadPhysicalCore(stream);
  return spec;
}

static void WriteSemaphoreSpec(dmlc::Stream* stream, const SemaphoreSpec& spec) {
  WriteUInt32(stream, spec.id);
  WriteUInt32(stream, spec.initial_value);
  WriteString(stream, spec.core_type);
  WriteVectorField<CoreRangeSpec>(stream, spec.core_ranges, WriteCoreRangeSpec);
}

static SemaphoreSpec ReadSemaphoreSpec(dmlc::Stream* stream) {
  SemaphoreSpec spec;
  spec.id = ReadUInt32(stream, "semaphore.id");
  spec.initial_value = ReadUInt32(stream, "semaphore.initial_value");
  spec.core_type = ReadString(stream, "semaphore.core_type");
  spec.core_ranges = ReadVectorField<CoreRangeSpec>(
      stream, "semaphore.core_ranges", ReadCoreRangeSpec);
  return spec;
}

static void WriteCBConfig(dmlc::Stream* stream, const CBConfig& spec) {
  WriteUInt32(stream, spec.cb_id);
  WriteString(stream, spec.name);
  WriteString(stream, spec.role);
  WriteUInt32(stream, spec.num_pages);
  WriteUInt32(stream, spec.page_size_bytes);
  WriteUInt32(stream, spec.initial_reserve_pages);
  WriteString(stream, spec.flow_class);
  WriteUInt32(stream, spec.publish_pages_per_event);
  WriteUInt32(stream, spec.consume_pages_per_event);
  WriteString(stream, spec.data_format);
}

static CBConfig ReadCBConfig(dmlc::Stream* stream) {
  CBConfig spec;
  spec.cb_id = ReadUInt32(stream, "cb_config.cb_id");
  spec.name = ReadString(stream, "cb_config.name");
  spec.role = ReadString(stream, "cb_config.role");
  spec.num_pages = ReadUInt32(stream, "cb_config.num_pages");
  spec.page_size_bytes = ReadUInt32(stream, "cb_config.page_size_bytes");
  spec.initial_reserve_pages = ReadUInt32(stream, "cb_config.initial_reserve_pages");
  spec.flow_class = ReadString(stream, "cb_config.flow_class");
  spec.publish_pages_per_event = ReadUInt32(stream, "cb_config.publish_pages_per_event");
  spec.consume_pages_per_event = ReadUInt32(stream, "cb_config.consume_pages_per_event");
  spec.data_format = ReadString(stream, "cb_config.data_format");
  return spec;
}

static void WriteKernelArgSpec(dmlc::Stream* stream, const KernelArgSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.kind);
  WriteString(stream, spec.dtype);
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.identity);
  WriteUInt32(stream, spec.core_x);
  WriteUInt32(stream, spec.core_y);
  WriteBool(stream, spec.has_core_coord);
}

static KernelArgSpec ReadKernelArgSpec(dmlc::Stream* stream) {
  KernelArgSpec spec;
  spec.name = ReadString(stream, "kernel_arg.name");
  spec.kind = ReadString(stream, "kernel_arg.kind");
  spec.dtype = ReadString(stream, "kernel_arg.dtype");
  spec.buffer = ReadString(stream, "kernel_arg.buffer");
  spec.identity = ReadString(stream, "kernel_arg.identity");
  spec.core_x = ReadUInt32(stream, "kernel_arg.core_x");
  spec.core_y = ReadUInt32(stream, "kernel_arg.core_y");
  spec.has_core_coord = ReadBool(stream, "kernel_arg.has_core_coord");
  return spec;
}

static void WriteCompileTimeArgSpec(dmlc::Stream* stream, const CompileTimeArgSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.kind);
  WriteString(stream, spec.dtype);
  WriteUInt32(stream, spec.offset);
  WriteUInt32(stream, spec.count);
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.segment_role);
  WriteUInt32Vector(stream, spec.values);
  WriteUInt32(stream, spec.args_config_bits);
  WriteUInt32(stream, spec.transport_page_size_bytes);
  WriteString(stream, spec.layout);
  WriteString(stream, spec.memory_space);
  WriteInt64Vector(stream, spec.host_axis_order);
  WriteBool(stream, spec.transpose_2d);
}

static CompileTimeArgSpec ReadCompileTimeArgSpec(dmlc::Stream* stream) {
  CompileTimeArgSpec spec;
  spec.name = ReadString(stream, "compile_time_arg.name");
  spec.kind = ReadString(stream, "compile_time_arg.kind");
  spec.dtype = ReadString(stream, "compile_time_arg.dtype");
  spec.offset = ReadUInt32(stream, "compile_time_arg.offset");
  spec.count = ReadUInt32(stream, "compile_time_arg.count");
  spec.buffer = ReadString(stream, "compile_time_arg.buffer");
  spec.segment_role = ReadString(stream, "compile_time_arg.segment_role");
  spec.values = ReadUInt32Vector(stream, "compile_time_arg.values");
  spec.args_config_bits = ReadUInt32(stream, "compile_time_arg.args_config_bits");
  spec.transport_page_size_bytes = ReadUInt32(
      stream, "compile_time_arg.transport_page_size_bytes");
  spec.layout = ReadString(stream, "compile_time_arg.layout");
  spec.memory_space = ReadString(stream, "compile_time_arg.memory_space");
  spec.host_axis_order = ReadInt64Vector(stream, "compile_time_arg.host_axis_order");
  spec.transpose_2d = ReadBool(stream, "compile_time_arg.transpose_2d");
  return spec;
}

static void WritePerWorkArgSpec(dmlc::Stream* stream, const PerWorkArgSpec& spec) {
  WriteString(stream, spec.arg_kind);
  WriteString(stream, spec.arg_identity);
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.descriptor_kind);
  WriteString(stream, spec.value_source);
  WriteUInt32(stream, spec.constant_value);
}

static PerWorkArgSpec ReadPerWorkArgSpec(dmlc::Stream* stream) {
  PerWorkArgSpec spec;
  spec.arg_kind = ReadString(stream, "per_work_arg.arg_kind");
  spec.arg_identity = ReadString(stream, "per_work_arg.arg_identity");
  spec.buffer = ReadString(stream, "per_work_arg.buffer");
  spec.descriptor_kind = ReadString(stream, "per_work_arg.descriptor_kind");
  spec.value_source = ReadString(stream, "per_work_arg.value_source");
  spec.constant_value = ReadUInt32(stream, "per_work_arg.constant_value");
  return spec;
}

static void WriteKernelLaunchSpec(dmlc::Stream* stream, const KernelLaunchSpec& spec) {
  WriteString(stream, spec.core_type);
  WriteString(stream, spec.processor);
  WriteString(stream, spec.noc);
}

static KernelLaunchSpec ReadKernelLaunchSpec(dmlc::Stream* stream) {
  KernelLaunchSpec spec;
  spec.core_type = ReadString(stream, "kernel_launch.core_type");
  spec.processor = ReadString(stream, "kernel_launch.processor");
  spec.noc = ReadString(stream, "kernel_launch.noc");
  return spec;
}

static void WriteKernelDefineSpec(dmlc::Stream* stream, const KernelDefineSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.value);
}

static KernelDefineSpec ReadKernelDefineSpec(dmlc::Stream* stream) {
  KernelDefineSpec spec;
  spec.name = ReadString(stream, "kernel_define.name");
  spec.value = ReadString(stream, "kernel_define.value");
  return spec;
}

static void WriteNamedCompileArgSpec(dmlc::Stream* stream,
                                     const NamedCompileArgSpec& spec) {
  WriteString(stream, spec.name);
  WriteUInt32(stream, spec.value);
}

static NamedCompileArgSpec ReadNamedCompileArgSpec(dmlc::Stream* stream) {
  NamedCompileArgSpec spec;
  spec.name = ReadString(stream, "named_compile_arg.name");
  spec.value = ReadUInt32(stream, "named_compile_arg.value");
  return spec;
}

static void WriteKernelComputeConfigSpec(dmlc::Stream* stream,
                                         const KernelComputeConfigSpec& spec) {
  WriteString(stream, spec.math_fidelity);
  WriteBool(stream, spec.fp32_dest_acc_en);
  WriteBool(stream, spec.dst_full_sync_en);
  WriteBool(stream, spec.math_approx_mode);
  WriteStringVector(stream, spec.unpack_to_dest_mode);
  WriteBool(stream, spec.bfp8_pack_precise);
  WriteBool(stream, spec.clear_accum);
  WriteUInt32(stream, spec.k_pack);
  WriteInt32(stream, spec.wg_wait);
  WriteInt32(stream, spec.policy_type);
  WriteString(stream, spec.policy_name);
  WriteVectorField<KernelDefineSpec>(stream, spec.defines, WriteKernelDefineSpec);
  WriteVectorField<NamedCompileArgSpec>(
      stream, spec.named_compile_args, WriteNamedCompileArgSpec);
}

static KernelComputeConfigSpec ReadKernelComputeConfigSpec(dmlc::Stream* stream) {
  KernelComputeConfigSpec spec;
  spec.math_fidelity = ReadString(stream, "compute_config.math_fidelity");
  spec.fp32_dest_acc_en = ReadBool(stream, "compute_config.fp32_dest_acc_en");
  spec.dst_full_sync_en = ReadBool(stream, "compute_config.dst_full_sync_en");
  spec.math_approx_mode = ReadBool(stream, "compute_config.math_approx_mode");
  spec.unpack_to_dest_mode = ReadStringVector(stream, "compute_config.unpack_to_dest_mode");
  spec.bfp8_pack_precise = ReadBool(stream, "compute_config.bfp8_pack_precise");
  spec.clear_accum = ReadBool(stream, "compute_config.clear_accum");
  spec.k_pack = ReadUInt32(stream, "compute_config.k_pack");
  spec.wg_wait = ReadInt32(stream, "compute_config.wg_wait");
  spec.policy_type = ReadInt32(stream, "compute_config.policy_type");
  spec.policy_name = ReadString(stream, "compute_config.policy_name");
  spec.defines = ReadVectorField<KernelDefineSpec>(
      stream, "compute_config.defines", ReadKernelDefineSpec);
  spec.named_compile_args = ReadVectorField<NamedCompileArgSpec>(
      stream, "compute_config.named_compile_args", ReadNamedCompileArgSpec);
  return spec;
}

static void WriteComputeOperandBindingSpec(dmlc::Stream* stream,
                                           const ComputeOperandBindingSpec& spec) {
  WriteString(stream, spec.role);
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.host_buffer);
}

static ComputeOperandBindingSpec ReadComputeOperandBindingSpec(dmlc::Stream* stream) {
  ComputeOperandBindingSpec spec;
  spec.role = ReadString(stream, "compute_operand.role");
  spec.buffer = ReadString(stream, "compute_operand.buffer");
  spec.host_buffer = ReadString(stream, "compute_operand.host_buffer");
  return spec;
}

static void WriteKernelComputeOpSpec(dmlc::Stream* stream, const KernelComputeOpSpec& spec) {
  WriteBool(stream, spec.enabled);
  WriteString(stream, spec.kind);
  WriteString(stream, spec.operation_name);
  WriteString(stream, spec.a_buffer);
  WriteString(stream, spec.b_buffer);
  WriteString(stream, spec.c_buffer);
  WriteVectorField<ComputeOperandBindingSpec>(
      stream, spec.operand_bindings, WriteComputeOperandBindingSpec);
  WriteUInt32(stream, spec.M);
  WriteUInt32(stream, spec.N);
  WriteUInt32(stream, spec.K);
  WriteUInt32(stream, spec.Mt);
  WriteUInt32(stream, spec.Nt);
  WriteUInt32(stream, spec.Kt);
  WriteUInt32(stream, spec.block_m_tiles);
  WriteUInt32(stream, spec.block_n_tiles);
  WriteUInt32(stream, spec.block_k_tiles);
  WriteUInt32(stream, spec.subblock_m_tiles);
  WriteUInt32(stream, spec.subblock_n_tiles);
  WriteBool(stream, spec.transpose_A);
  WriteBool(stream, spec.transpose_B);
  WriteString(stream, spec.a_tensor_dtype);
  WriteString(stream, spec.b_tensor_dtype);
  WriteString(stream, spec.c_tensor_dtype);
  WriteString(stream, spec.a_cb_dtype);
  WriteString(stream, spec.b_cb_dtype);
  WriteString(stream, spec.c_cb_dtype);
  WriteString(stream, spec.accumulator_dtype);
  WriteBool(stream, spec.has_mbarrier);
  WriteString(stream, spec.mbarrier_buffer);
  WriteString(stream, spec.mbarrier_scope);
  WriteStringVector(stream, spec.mbarrier_index_exprs);
}

static KernelComputeOpSpec ReadKernelComputeOpSpec(dmlc::Stream* stream) {
  KernelComputeOpSpec spec;
  spec.enabled = ReadBool(stream, "compute_op.enabled");
  spec.kind = ReadString(stream, "compute_op.kind");
  spec.operation_name = ReadString(stream, "compute_op.operation_name");
  spec.a_buffer = ReadString(stream, "compute_op.a_buffer");
  spec.b_buffer = ReadString(stream, "compute_op.b_buffer");
  spec.c_buffer = ReadString(stream, "compute_op.c_buffer");
  spec.operand_bindings = ReadVectorField<ComputeOperandBindingSpec>(
      stream, "compute_op.operand_bindings", ReadComputeOperandBindingSpec);
  spec.M = ReadUInt32(stream, "compute_op.M");
  spec.N = ReadUInt32(stream, "compute_op.N");
  spec.K = ReadUInt32(stream, "compute_op.K");
  spec.Mt = ReadUInt32(stream, "compute_op.Mt");
  spec.Nt = ReadUInt32(stream, "compute_op.Nt");
  spec.Kt = ReadUInt32(stream, "compute_op.Kt");
  spec.block_m_tiles = ReadUInt32(stream, "compute_op.block_m_tiles");
  spec.block_n_tiles = ReadUInt32(stream, "compute_op.block_n_tiles");
  spec.block_k_tiles = ReadUInt32(stream, "compute_op.block_k_tiles");
  spec.subblock_m_tiles = ReadUInt32(stream, "compute_op.subblock_m_tiles");
  spec.subblock_n_tiles = ReadUInt32(stream, "compute_op.subblock_n_tiles");
  spec.transpose_A = ReadBool(stream, "compute_op.transpose_A");
  spec.transpose_B = ReadBool(stream, "compute_op.transpose_B");
  spec.a_tensor_dtype = ReadString(stream, "compute_op.a_tensor_dtype");
  spec.b_tensor_dtype = ReadString(stream, "compute_op.b_tensor_dtype");
  spec.c_tensor_dtype = ReadString(stream, "compute_op.c_tensor_dtype");
  spec.a_cb_dtype = ReadString(stream, "compute_op.a_cb_dtype");
  spec.b_cb_dtype = ReadString(stream, "compute_op.b_cb_dtype");
  spec.c_cb_dtype = ReadString(stream, "compute_op.c_cb_dtype");
  spec.accumulator_dtype = ReadString(stream, "compute_op.accumulator_dtype");
  spec.has_mbarrier = ReadBool(stream, "compute_op.has_mbarrier");
  spec.mbarrier_buffer = ReadString(stream, "compute_op.mbarrier_buffer");
  spec.mbarrier_scope = ReadString(stream, "compute_op.mbarrier_scope");
  spec.mbarrier_index_exprs = ReadStringVector(stream, "compute_op.mbarrier_index_exprs");
  return spec;
}

static void WriteAccessorSpec(dmlc::Stream* stream, const AccessorSpec& spec) {
  WriteString(stream, spec.buffer);
  WriteUInt32(stream, spec.compile_time_arg_offset);
  WriteUInt32(stream, spec.compile_time_arg_count);
  WriteUInt32(stream, spec.common_runtime_arg_offset);
  WriteUInt32(stream, spec.common_runtime_arg_count);
  WriteUInt32(stream, spec.args_config_bits);
  WriteUInt32(stream, spec.transport_page_size_bytes);
  WriteString(stream, spec.layout);
  WriteString(stream, spec.memory_space);
  WriteInt64Vector(stream, spec.host_axis_order);
  WriteBool(stream, spec.transpose_2d);
}

static AccessorSpec ReadAccessorSpec(dmlc::Stream* stream) {
  AccessorSpec spec;
  spec.buffer = ReadString(stream, "accessor.buffer");
  spec.compile_time_arg_offset = ReadUInt32(stream, "accessor.compile_time_arg_offset");
  spec.compile_time_arg_count = ReadUInt32(stream, "accessor.compile_time_arg_count");
  spec.common_runtime_arg_offset = ReadUInt32(stream, "accessor.common_runtime_arg_offset");
  spec.common_runtime_arg_count = ReadUInt32(stream, "accessor.common_runtime_arg_count");
  spec.args_config_bits = ReadUInt32(stream, "accessor.args_config_bits");
  spec.transport_page_size_bytes = ReadUInt32(stream, "accessor.transport_page_size_bytes");
  spec.layout = ReadString(stream, "accessor.layout");
  spec.memory_space = ReadString(stream, "accessor.memory_space");
  spec.host_axis_order = ReadInt64Vector(stream, "accessor.host_axis_order");
  spec.transpose_2d = ReadBool(stream, "accessor.transpose_2d");
  return spec;
}

static void WriteSemaphoreBindingSpec(dmlc::Stream* stream,
                                      const SemaphoreBindingSpec& spec) {
  WriteString(stream, spec.name);
  WriteUInt32(stream, spec.semaphore_id);
  WriteString(stream, spec.arg_kind);
}

static SemaphoreBindingSpec ReadSemaphoreBindingSpec(dmlc::Stream* stream) {
  SemaphoreBindingSpec spec;
  spec.name = ReadString(stream, "semaphore_binding.name");
  spec.semaphore_id = ReadUInt32(stream, "semaphore_binding.semaphore_id");
  spec.arg_kind = ReadString(stream, "semaphore_binding.arg_kind");
  return spec;
}

static void WriteRemoteCoreDescriptorSpec(dmlc::Stream* stream,
                                          const RemoteCoreDescriptorSpec& spec) {
  WriteString(stream, spec.identity);
  WriteUInt32(stream, spec.core_x);
  WriteUInt32(stream, spec.core_y);
}

static RemoteCoreDescriptorSpec ReadRemoteCoreDescriptorSpec(dmlc::Stream* stream) {
  RemoteCoreDescriptorSpec spec;
  spec.identity = ReadString(stream, "remote_core_descriptor.identity");
  spec.core_x = ReadUInt32(stream, "remote_core_descriptor.core_x");
  spec.core_y = ReadUInt32(stream, "remote_core_descriptor.core_y");
  return spec;
}

static void WriteBufferMaterializationSpec(dmlc::Stream* stream,
                                           const BufferMaterializationSpec& spec) {
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.materialization_kind);
  WriteString(stream, spec.layout);
  WriteString(stream, spec.memory_space);
  WriteUInt32(stream, spec.transport_page_size_bytes);
  WriteInt64Vector(stream, spec.host_axis_order);
  WriteBool(stream, spec.transpose_2d);
  WriteString(stream, spec.live_form_kind);
  WriteString(stream, spec.execution_topology_kind);
  WriteUInt32(stream, spec.physical_local_extent);
  WriteUInt32(stream, spec.logical_element_count);
  WriteString(stream, spec.producer_kernel);
  WriteString(stream, spec.materialization_protocol);
  WriteString(stream, spec.publication_protocol);
}

static BufferMaterializationSpec ReadBufferMaterializationSpec(dmlc::Stream* stream) {
  BufferMaterializationSpec spec;
  spec.buffer = ReadString(stream, "buffer_materialization.buffer");
  spec.materialization_kind = ReadString(stream, "buffer_materialization.materialization_kind");
  spec.layout = ReadString(stream, "buffer_materialization.layout");
  spec.memory_space = ReadString(stream, "buffer_materialization.memory_space");
  spec.transport_page_size_bytes = ReadUInt32(
      stream, "buffer_materialization.transport_page_size_bytes");
  spec.host_axis_order = ReadInt64Vector(stream, "buffer_materialization.host_axis_order");
  spec.transpose_2d = ReadBool(stream, "buffer_materialization.transpose_2d");
  spec.live_form_kind = ReadString(stream, "buffer_materialization.live_form_kind");
  spec.execution_topology_kind = ReadString(
      stream, "buffer_materialization.execution_topology_kind");
  spec.physical_local_extent = ReadUInt32(
      stream, "buffer_materialization.physical_local_extent");
  spec.logical_element_count = ReadUInt32(
      stream, "buffer_materialization.logical_element_count");
  spec.producer_kernel = ReadString(stream, "buffer_materialization.producer_kernel");
  spec.materialization_protocol = ReadString(
      stream, "buffer_materialization.materialization_protocol");
  spec.publication_protocol = ReadString(
      stream, "buffer_materialization.publication_protocol");
  return spec;
}

static void WriteBufferDistributionSpec(dmlc::Stream* stream,
                                        const BufferDistributionSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.buffer);
  WriteString(stream, spec.mesh_plan);
  WriteInt64(stream, spec.mesh_plan_index);
  WriteString(stream, spec.distribution_kind);
  WriteString(stream, spec.layout);
  WriteString(stream, spec.memory_space);
  WriteUInt32(stream, spec.page_size_bytes);
  WriteInt64Vector(stream, spec.shard_shape);
  WriteInt64Vector(stream, spec.shard_grid_shape);
  WriteString(stream, spec.sharding_strategy);
  WriteString(stream, spec.shard_orientation);
  WriteString(stream, spec.source_buffer);
  WriteString(stream, spec.source_region_kind);
  WriteInt64Vector(stream, spec.source_region_shape);
  WriteString(stream, spec.logical_index_mapping);
  WriteString(stream, spec.core_local_address_mapping);
  WriteString(stream, spec.host_visibility);
  WriteString(stream, spec.attached_core_group);
  WriteInt64(stream, spec.attached_core_group_index);
}

static BufferDistributionSpec ReadBufferDistributionSpec(dmlc::Stream* stream) {
  BufferDistributionSpec spec;
  spec.name = ReadString(stream, "buffer_distribution.name");
  spec.buffer = ReadString(stream, "buffer_distribution.buffer");
  spec.mesh_plan = ReadString(stream, "buffer_distribution.mesh_plan");
  spec.mesh_plan_index = ReadInt64(stream, "buffer_distribution.mesh_plan_index");
  spec.distribution_kind = ReadString(stream, "buffer_distribution.distribution_kind");
  spec.layout = ReadString(stream, "buffer_distribution.layout");
  spec.memory_space = ReadString(stream, "buffer_distribution.memory_space");
  spec.page_size_bytes = ReadUInt32(stream, "buffer_distribution.page_size_bytes");
  spec.shard_shape = ReadInt64Vector(stream, "buffer_distribution.shard_shape");
  spec.shard_grid_shape = ReadInt64Vector(stream, "buffer_distribution.shard_grid_shape");
  spec.sharding_strategy = ReadString(stream, "buffer_distribution.sharding_strategy");
  spec.shard_orientation = ReadString(stream, "buffer_distribution.shard_orientation");
  spec.source_buffer = ReadString(stream, "buffer_distribution.source_buffer");
  spec.source_region_kind = ReadString(stream, "buffer_distribution.source_region_kind");
  spec.source_region_shape =
      ReadInt64Vector(stream, "buffer_distribution.source_region_shape");
  spec.logical_index_mapping =
      ReadString(stream, "buffer_distribution.logical_index_mapping");
  spec.core_local_address_mapping =
      ReadString(stream, "buffer_distribution.core_local_address_mapping");
  spec.host_visibility = ReadString(stream, "buffer_distribution.host_visibility");
  spec.attached_core_group = ReadString(stream, "buffer_distribution.attached_core_group");
  spec.attached_core_group_index =
      ReadInt64(stream, "buffer_distribution.attached_core_group_index");
  return spec;
}

static void WriteTensorMemoryConfigSpec(dmlc::Stream* stream,
                                        const TensorMemoryConfigSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.subject);
  WriteString(stream, spec.value_identity);
  WriteInt64Vector(stream, spec.logical_shape);
  WriteString(stream, spec.dtype);
  WriteString(stream, spec.memory_layout);
  WriteString(stream, spec.buffer_type);
  WriteString(stream, spec.grid_ref);
  WriteInt64Vector(stream, spec.shard_grid_shape);
  WriteInt64Vector(stream, spec.shard_shape);
  WriteString(stream, spec.shard_orientation);
  WriteString(stream, spec.shard_distribution_strategy);
  WriteInt64Vector(stream, spec.page_shape);
  WriteString(stream, spec.origin);
  WriteString(stream, spec.source_buffer);
  WriteString(stream, spec.buffer_distribution_plan);
  WriteInt64(stream, spec.buffer_distribution_plan_index);
  WriteBool(stream, spec.has_runtime_accessor);
  WriteBool(stream, spec.requires_materialization);
}

static TensorMemoryConfigSpec ReadTensorMemoryConfigSpec(dmlc::Stream* stream) {
  TensorMemoryConfigSpec spec;
  spec.name = ReadString(stream, "tensor_memory_config.name");
  spec.subject = ReadString(stream, "tensor_memory_config.subject");
  spec.value_identity = ReadString(stream, "tensor_memory_config.value_identity");
  spec.logical_shape =
      ReadInt64Vector(stream, "tensor_memory_config.logical_shape");
  spec.dtype = ReadString(stream, "tensor_memory_config.dtype");
  spec.memory_layout = ReadString(stream, "tensor_memory_config.memory_layout");
  spec.buffer_type = ReadString(stream, "tensor_memory_config.buffer_type");
  spec.grid_ref = ReadString(stream, "tensor_memory_config.grid_ref");
  spec.shard_grid_shape =
      ReadInt64Vector(stream, "tensor_memory_config.shard_grid_shape");
  spec.shard_shape = ReadInt64Vector(stream, "tensor_memory_config.shard_shape");
  spec.shard_orientation =
      ReadString(stream, "tensor_memory_config.shard_orientation");
  spec.shard_distribution_strategy =
      ReadString(stream, "tensor_memory_config.shard_distribution_strategy");
  spec.page_shape = ReadInt64Vector(stream, "tensor_memory_config.page_shape");
  spec.origin = ReadString(stream, "tensor_memory_config.origin");
  spec.source_buffer = ReadString(stream, "tensor_memory_config.source_buffer");
  spec.buffer_distribution_plan =
      ReadString(stream, "tensor_memory_config.buffer_distribution_plan");
  spec.buffer_distribution_plan_index =
      ReadInt64(stream, "tensor_memory_config.buffer_distribution_plan_index");
  spec.has_runtime_accessor =
      ReadBool(stream, "tensor_memory_config.has_runtime_accessor");
  spec.requires_materialization =
      ReadBool(stream, "tensor_memory_config.requires_materialization");
  return spec;
}

static void WriteReshardPlanSpec(dmlc::Stream* stream,
                                 const ReshardPlanSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.source_value);
  WriteString(stream, spec.target_value);
  WriteString(stream, spec.source_memory_config_plan);
  WriteInt64(stream, spec.source_memory_config_plan_index);
  WriteString(stream, spec.target_memory_config_plan);
  WriteInt64(stream, spec.target_memory_config_plan_index);
  WriteString(stream, spec.conversion_kind);
  WriteString(stream, spec.source_region_kind);
  WriteInt64Vector(stream, spec.source_region_shape);
  WriteString(stream, spec.materialization_plan);
  WriteInt64(stream, spec.materialization_plan_index);
  WriteString(stream, spec.materialization_protocol);
  WriteInt64Vector(stream, spec.required_cb_plan_indices);
  WriteInt64Vector(stream, spec.required_sync_plan_indices);
  WriteString(stream, spec.scheduling_kind);
  WriteString(stream, spec.inserted_by);
  WriteString(stream, spec.admission_status);
  WriteString(stream, spec.unsupported_reason);
}

static ReshardPlanSpec ReadReshardPlanSpec(dmlc::Stream* stream) {
  ReshardPlanSpec spec;
  spec.name = ReadString(stream, "reshard_plan.name");
  spec.source_value = ReadString(stream, "reshard_plan.source_value");
  spec.target_value = ReadString(stream, "reshard_plan.target_value");
  spec.source_memory_config_plan =
      ReadString(stream, "reshard_plan.source_memory_config_plan");
  spec.source_memory_config_plan_index =
      ReadInt64(stream, "reshard_plan.source_memory_config_plan_index");
  spec.target_memory_config_plan =
      ReadString(stream, "reshard_plan.target_memory_config_plan");
  spec.target_memory_config_plan_index =
      ReadInt64(stream, "reshard_plan.target_memory_config_plan_index");
  spec.conversion_kind = ReadString(stream, "reshard_plan.conversion_kind");
  spec.source_region_kind = ReadString(stream, "reshard_plan.source_region_kind");
  spec.source_region_shape =
      ReadInt64Vector(stream, "reshard_plan.source_region_shape");
  spec.materialization_plan =
      ReadString(stream, "reshard_plan.materialization_plan");
  spec.materialization_plan_index =
      ReadInt64(stream, "reshard_plan.materialization_plan_index");
  spec.materialization_protocol =
      ReadString(stream, "reshard_plan.materialization_protocol");
  spec.required_cb_plan_indices =
      ReadInt64Vector(stream, "reshard_plan.required_cb_plan_indices");
  spec.required_sync_plan_indices =
      ReadInt64Vector(stream, "reshard_plan.required_sync_plan_indices");
  spec.scheduling_kind = ReadString(stream, "reshard_plan.scheduling_kind");
  spec.inserted_by = ReadString(stream, "reshard_plan.inserted_by");
  spec.admission_status = ReadString(stream, "reshard_plan.admission_status");
  spec.unsupported_reason = ReadString(stream, "reshard_plan.unsupported_reason");
  return spec;
}

static void WriteLiveFormPlanSpec(dmlc::Stream* stream, const LiveFormPlanSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.logical_value);
  WriteString(stream, spec.spatial_live_value);
  WriteInt64(stream, spec.spatial_live_value_index);
  WriteString(stream, spec.producer_kernel);
  WriteString(stream, spec.physical_form);
  WriteString(stream, spec.execution_topology);
  WriteUInt32(stream, spec.physical_local_extent);
  WriteUInt32(stream, spec.logical_element_count);
  WriteString(stream, spec.ownership_kind);
}

static LiveFormPlanSpec ReadLiveFormPlanSpec(dmlc::Stream* stream) {
  LiveFormPlanSpec spec;
  spec.name = ReadString(stream, "live_form.name");
  spec.logical_value = ReadString(stream, "live_form.logical_value");
  spec.spatial_live_value = ReadString(stream, "live_form.spatial_live_value");
  spec.spatial_live_value_index = ReadInt64(stream, "live_form.spatial_live_value_index");
  spec.producer_kernel = ReadString(stream, "live_form.producer_kernel");
  spec.physical_form = ReadString(stream, "live_form.physical_form");
  spec.execution_topology = ReadString(stream, "live_form.execution_topology");
  spec.physical_local_extent = ReadUInt32(stream, "live_form.physical_local_extent");
  spec.logical_element_count = ReadUInt32(stream, "live_form.logical_element_count");
  spec.ownership_kind = ReadString(stream, "live_form.ownership_kind");
  return spec;
}

static void WriteMaterializationPlanSpec(dmlc::Stream* stream,
                                         const MaterializationPlanSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.source_live_form);
  WriteString(stream, spec.materialization_boundary);
  WriteInt64(stream, spec.materialization_boundary_index);
  WriteString(stream, spec.target_buffer);
  WriteString(stream, spec.host_buffer);
  WriteString(stream, spec.target_kernel);
  WriteString(stream, spec.bridge_kind);
  WriteString(stream, spec.materialization_kind);
  WriteString(stream, spec.materialization_protocol);
  WriteString(stream, spec.publication_protocol);
  WriteInt64Vector(stream, spec.required_cb_plan_indices);
  WriteInt64Vector(stream, spec.required_sync_plan_indices);
  WriteString(stream, spec.produced_live_form);
}

static MaterializationPlanSpec ReadMaterializationPlanSpec(dmlc::Stream* stream) {
  MaterializationPlanSpec spec;
  spec.name = ReadString(stream, "materialization_plan.name");
  spec.source_live_form = ReadString(stream, "materialization_plan.source_live_form");
  spec.materialization_boundary = ReadString(
      stream, "materialization_plan.materialization_boundary");
  spec.materialization_boundary_index = ReadInt64(
      stream, "materialization_plan.materialization_boundary_index");
  spec.target_buffer = ReadString(stream, "materialization_plan.target_buffer");
  spec.host_buffer = ReadString(stream, "materialization_plan.host_buffer");
  spec.target_kernel = ReadString(stream, "materialization_plan.target_kernel");
  spec.bridge_kind = ReadString(stream, "materialization_plan.bridge_kind");
  spec.materialization_kind = ReadString(stream, "materialization_plan.materialization_kind");
  spec.materialization_protocol = ReadString(
      stream, "materialization_plan.materialization_protocol");
  spec.publication_protocol = ReadString(stream, "materialization_plan.publication_protocol");
  spec.required_cb_plan_indices = ReadInt64Vector(
      stream, "materialization_plan.required_cb_plan_indices");
  spec.required_sync_plan_indices = ReadInt64Vector(
      stream, "materialization_plan.required_sync_plan_indices");
  spec.produced_live_form = ReadString(stream, "materialization_plan.produced_live_form");
  return spec;
}

static void WriteConsumerBindingPlanSpec(dmlc::Stream* stream,
                                         const ConsumerBindingPlanSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.consumer_kernel);
  WriteString(stream, spec.consumer_op_kind);
  WriteString(stream, spec.source_live_form);
  WriteString(stream, spec.live_value_edge);
  WriteInt64(stream, spec.live_value_edge_index);
  WriteBool(stream, spec.accepts_distributed_slice);
  WriteBool(stream, spec.requires_full_logical_tile);
  WriteInt64(stream, spec.abi_plan_index);
  WriteString(stream, spec.target_buffer);
  WriteString(stream, spec.materialization_plan);
}

static ConsumerBindingPlanSpec ReadConsumerBindingPlanSpec(dmlc::Stream* stream) {
  ConsumerBindingPlanSpec spec;
  spec.name = ReadString(stream, "consumer_binding.name");
  spec.consumer_kernel = ReadString(stream, "consumer_binding.consumer_kernel");
  spec.consumer_op_kind = ReadString(stream, "consumer_binding.consumer_op_kind");
  spec.source_live_form = ReadString(stream, "consumer_binding.source_live_form");
  spec.live_value_edge = ReadString(stream, "consumer_binding.live_value_edge");
  spec.live_value_edge_index = ReadInt64(stream, "consumer_binding.live_value_edge_index");
  spec.accepts_distributed_slice = ReadBool(
      stream, "consumer_binding.accepts_distributed_slice");
  spec.requires_full_logical_tile = ReadBool(
      stream, "consumer_binding.requires_full_logical_tile");
  spec.abi_plan_index = ReadInt64(stream, "consumer_binding.abi_plan_index");
  spec.target_buffer = ReadString(stream, "consumer_binding.target_buffer");
  spec.materialization_plan = ReadString(stream, "consumer_binding.materialization_plan");
  return spec;
}

static void WriteKernelSpec(dmlc::Stream* stream, const KernelSpec& spec) {
  WriteString(stream, spec.name);
  WriteString(stream, spec.kind);
  WriteString(stream, spec.core_type);
  WriteString(stream, spec.source_code);
  WriteUInt32Vector(stream, spec.compile_time_args);
  WriteVectorField<KernelArgSpec>(stream, spec.runtime_args, WriteKernelArgSpec);
  WriteVectorField<KernelArgSpec>(stream, spec.common_runtime_args, WriteKernelArgSpec);
  WriteVectorField<CompileTimeArgSpec>(
      stream, spec.compile_time_arg_specs, WriteCompileTimeArgSpec);
  WriteVectorField<PerWorkArgSpec>(stream, spec.per_work_arg_specs, WritePerWorkArgSpec);
  WriteBool(stream, spec.has_launch_spec);
  WriteKernelLaunchSpec(stream, spec.launch_spec);
  WriteBool(stream, spec.has_compute_config);
  WriteKernelComputeConfigSpec(stream, spec.compute_config);
  WriteVectorField<KernelComputeOpSpec>(stream, spec.compute_ops, WriteKernelComputeOpSpec);
  WriteVectorField<AccessorSpec>(stream, spec.accessors, WriteAccessorSpec);
  WriteVectorField<SemaphoreBindingSpec>(
      stream, spec.semaphore_bindings, WriteSemaphoreBindingSpec);
  WriteVectorField<RemoteCoreDescriptorSpec>(
      stream, spec.remote_core_descriptors, WriteRemoteCoreDescriptorSpec);
}

static KernelSpec ReadKernelSpec(dmlc::Stream* stream) {
  KernelSpec spec;
  spec.name = ReadString(stream, "kernel.name");
  spec.kind = ReadString(stream, "kernel.kind");
  spec.core_type = ReadString(stream, "kernel.core_type");
  spec.source_code = ReadString(stream, "kernel.source_code");
  spec.compile_time_args = ReadUInt32Vector(stream, "kernel.compile_time_args");
  spec.runtime_args = ReadVectorField<KernelArgSpec>(
      stream, "kernel.runtime_args", ReadKernelArgSpec);
  spec.common_runtime_args = ReadVectorField<KernelArgSpec>(
      stream, "kernel.common_runtime_args", ReadKernelArgSpec);
  spec.compile_time_arg_specs = ReadVectorField<CompileTimeArgSpec>(
      stream, "kernel.compile_time_arg_specs", ReadCompileTimeArgSpec);
  spec.per_work_arg_specs = ReadVectorField<PerWorkArgSpec>(
      stream, "kernel.per_work_arg_specs", ReadPerWorkArgSpec);
  spec.has_launch_spec = ReadBool(stream, "kernel.has_launch_spec");
  spec.launch_spec = ReadKernelLaunchSpec(stream);
  spec.has_compute_config = ReadBool(stream, "kernel.has_compute_config");
  spec.compute_config = ReadKernelComputeConfigSpec(stream);
  spec.compute_ops = ReadVectorField<KernelComputeOpSpec>(
      stream, "kernel.compute_ops", ReadKernelComputeOpSpec);
  spec.accessors = ReadVectorField<AccessorSpec>(
      stream, "kernel.accessors", ReadAccessorSpec);
  spec.semaphore_bindings = ReadVectorField<SemaphoreBindingSpec>(
      stream, "kernel.semaphore_bindings", ReadSemaphoreBindingSpec);
  spec.remote_core_descriptors = ReadVectorField<RemoteCoreDescriptorSpec>(
      stream, "kernel.remote_core_descriptors", ReadRemoteCoreDescriptorSpec);
  return spec;
}

static void WriteExecutableSpec(dmlc::Stream* stream, const ExecutableSpec& spec) {
  WriteString(stream, spec.entry_name);
  WriteVectorField<CBConfig>(stream, spec.cb_configs, WriteCBConfig);
  WriteCorePlan(stream, spec.core_plan);
  WriteVectorField<SemaphoreSpec>(stream, spec.semaphores, WriteSemaphoreSpec);
  WriteVectorField<BufferDistributionSpec>(
      stream, spec.buffer_distribution_plans, WriteBufferDistributionSpec);
  WriteVectorField<TensorMemoryConfigSpec>(
      stream, spec.tensor_memory_config_plans, WriteTensorMemoryConfigSpec);
  WriteVectorField<ReshardPlanSpec>(
      stream, spec.reshard_plans, WriteReshardPlanSpec);
  WriteVectorField<BufferMaterializationSpec>(
      stream, spec.buffer_materializations, WriteBufferMaterializationSpec);
  WriteVectorField<KernelArgSpec>(stream, spec.runtime_args, WriteKernelArgSpec);
  WriteVectorField<KernelArgSpec>(stream, spec.common_runtime_args, WriteKernelArgSpec);
  WriteVectorField<PerWorkArgSpec>(stream, spec.per_work_arg_specs, WritePerWorkArgSpec);
  WriteVectorField<KernelSpec>(stream, spec.kernels, WriteKernelSpec);
  WriteVectorField<LiveFormPlanSpec>(stream, spec.live_form_plans, WriteLiveFormPlanSpec);
  WriteVectorField<MaterializationPlanSpec>(
      stream, spec.materialization_plans, WriteMaterializationPlanSpec);
  WriteVectorField<ConsumerBindingPlanSpec>(
      stream, spec.consumer_binding_plans, WriteConsumerBindingPlanSpec);
  WriteStringVector(stream, spec.direct_runtime_unsupported_reasons);
  WriteStringVector(stream, spec.tvm_arg_names);
  WriteDLDataTypeVector(stream, spec.tvm_arg_types);
  WriteBoolVector(stream, spec.tvm_is_buffer_arg);
}

static ExecutableSpec ReadExecutableSpec(dmlc::Stream* stream) {
  ExecutableSpec spec;
  spec.entry_name = ReadString(stream, "executable.entry_name");
  spec.cb_configs = ReadVectorField<CBConfig>(
      stream, "executable.cb_configs", ReadCBConfig);
  spec.core_plan = ReadCorePlan(stream);
  spec.semaphores = ReadVectorField<SemaphoreSpec>(
      stream, "executable.semaphores", ReadSemaphoreSpec);
  spec.buffer_distribution_plans = ReadVectorField<BufferDistributionSpec>(
      stream, "executable.buffer_distribution_plans", ReadBufferDistributionSpec);
  spec.tensor_memory_config_plans = ReadVectorField<TensorMemoryConfigSpec>(
      stream, "executable.tensor_memory_config_plans", ReadTensorMemoryConfigSpec);
  spec.reshard_plans = ReadVectorField<ReshardPlanSpec>(
      stream, "executable.reshard_plans", ReadReshardPlanSpec);
  spec.buffer_materializations = ReadVectorField<BufferMaterializationSpec>(
      stream, "executable.buffer_materializations", ReadBufferMaterializationSpec);
  spec.runtime_args = ReadVectorField<KernelArgSpec>(
      stream, "executable.runtime_args", ReadKernelArgSpec);
  spec.common_runtime_args = ReadVectorField<KernelArgSpec>(
      stream, "executable.common_runtime_args", ReadKernelArgSpec);
  spec.per_work_arg_specs = ReadVectorField<PerWorkArgSpec>(
      stream, "executable.per_work_arg_specs", ReadPerWorkArgSpec);
  spec.kernels = ReadVectorField<KernelSpec>(
      stream, "executable.kernels", ReadKernelSpec);
  spec.live_form_plans = ReadVectorField<LiveFormPlanSpec>(
      stream, "executable.live_form_plans", ReadLiveFormPlanSpec);
  spec.materialization_plans = ReadVectorField<MaterializationPlanSpec>(
      stream, "executable.materialization_plans", ReadMaterializationPlanSpec);
  spec.consumer_binding_plans = ReadVectorField<ConsumerBindingPlanSpec>(
      stream, "executable.consumer_binding_plans", ReadConsumerBindingPlanSpec);
  spec.direct_runtime_unsupported_reasons = ReadStringVector(
      stream, "executable.direct_runtime_unsupported_reasons");
  spec.tvm_arg_names = ReadStringVector(stream, "executable.tvm_arg_names");
  spec.tvm_arg_types = ReadDLDataTypeVector(stream, "executable.tvm_arg_types");
  spec.tvm_is_buffer_arg = ReadBoolVector(stream, "executable.tvm_is_buffer_arg");
  return spec;
}

static void WriteExecutableSpecMap(
    dmlc::Stream* stream,
    const std::unordered_map<std::string, ExecutableSpec>& fmap) {
  WriteUInt64(stream, static_cast<uint64_t>(fmap.size()));
  for (const auto& entry : fmap) {
    WriteString(stream, entry.first);
    WriteExecutableSpec(stream, entry.second);
  }
}

static std::unordered_map<std::string, ExecutableSpec> ReadExecutableSpecMap(
    dmlc::Stream* stream) {
  const uint64_t size = ReadUInt64(stream, "module.fmap");
  ICHECK_LE(size, 1000000ULL)
      << "BlackholeModule LoadFromBytes module.fmap has unreasonable size";
  std::unordered_map<std::string, ExecutableSpec> fmap;
  fmap.reserve(static_cast<size_t>(size));
  for (uint64_t i = 0; i < size; ++i) {
    std::string name = ReadString(stream, "module.fmap.name");
    ExecutableSpec spec = ReadExecutableSpec(stream);
    auto inserted = fmap.emplace(std::move(name), std::move(spec));
    ICHECK(inserted.second)
        << "BlackholeModule LoadFromBytes duplicate function " << inserted.first->first;
  }
  return fmap;
}

static bool BlackholeDebugIOEnabled() {
  const char* value = std::getenv("TILELANG_BLACKHOLE_DEBUG_IO");
  return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

static float BFloat16BitsToFloat(uint16_t bits) {
  const uint32_t value = static_cast<uint32_t>(bits) << 16;
  float result = 0.0f;
  static_assert(sizeof(result) == sizeof(value), "float and uint32_t must have equal size");
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

template <typename T>
static std::string PreviewTensorValues(const std::vector<T>& values, size_t limit = 8) {
  std::ostringstream os;
  os << "[";
  const size_t count = std::min(limit, values.size());
  for (size_t i = 0; i < count; ++i) {
    if (i != 0) {
      os << ", ";
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
      os << BFloat16BitsToFloat(values[i]);
    } else {
      os << values[i];
    }
  }
  if (values.size() > count) {
    os << ", ...";
  }
  os << "]";
  return os.str();
}

// Forward declaration
class BlackholeWrappedFunc;

// ============================================================================
// BlackholeWrappedFunc declaration
// ============================================================================

class BlackholeWrappedFunc {
 public:
  void Init(BlackholeModuleNode* m, ObjectPtr<Object> sptr,
            const std::string& func_name,
            const ExecutableSpec& info) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    info_ = info;
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const;

 private:
  BlackholeModuleNode* m_;
  ObjectPtr<Object> sptr_;
  std::string func_name_;
  ExecutableSpec info_;
};

// Argument extraction helpers
template <typename To, typename From>
static To BitCastScalar(const From& value) {
  static_assert(sizeof(To) == sizeof(From), "bit cast requires equal-sized types");
  To out;
  std::memcpy(&out, &value, sizeof(To));
  return out;
}

static uint32_t ExtractSignedScalarAsU32(int64_t value, DLDataType dtype) {
  ICHECK_GT(dtype.bits, 0U) << "Blackhole unsupported scalar argument: zero-width int";
  ICHECK_LE(dtype.bits, 32U)
      << "Blackhole unsupported scalar argument: int" << dtype.bits
      << " cannot be passed as a uint32 runtime argument";
  const int64_t min_value =
      dtype.bits == 64U ? std::numeric_limits<int64_t>::min()
                        : -(int64_t{1} << (static_cast<int64_t>(dtype.bits) - 1));
  const int64_t max_value =
      dtype.bits == 64U ? std::numeric_limits<int64_t>::max()
                        : ((int64_t{1} << (static_cast<int64_t>(dtype.bits) - 1)) - 1);
  ICHECK_GE(value, min_value)
      << "Blackhole scalar int argument " << value << " is below int" << dtype.bits
      << " range";
  ICHECK_LE(value, max_value)
      << "Blackhole scalar int argument " << value << " exceeds int" << dtype.bits
      << " range";
  return static_cast<uint32_t>(static_cast<int32_t>(value));
}

static uint32_t ExtractUnsignedScalarAsU32(uint64_t value, DLDataType dtype) {
  ICHECK_GT(dtype.bits, 0U) << "Blackhole unsupported scalar argument: zero-width uint";
  ICHECK_LE(dtype.bits, 32U)
      << "Blackhole unsupported scalar argument: uint" << dtype.bits
      << " cannot be passed as a uint32 runtime argument";
  const uint64_t max_value =
      dtype.bits == 32U ? std::numeric_limits<uint32_t>::max()
                        : ((uint64_t{1} << static_cast<uint64_t>(dtype.bits)) - 1U);
  ICHECK_LE(value, max_value)
      << "Blackhole scalar uint argument " << value << " exceeds uint" << dtype.bits
      << " range";
  return static_cast<uint32_t>(value);
}

uint32_t ExtractScalar(const ffi::AnyView& arg, DLDataType dtype) {
  if (dtype.code == kDLInt) {
    auto opt_i32 = arg.try_cast<int32_t>();
    if (opt_i32.has_value()) {
      return ExtractSignedScalarAsU32(opt_i32.value(), dtype);
    }
    auto opt_i64 = arg.try_cast<int64_t>();
    if (opt_i64.has_value()) {
      return ExtractSignedScalarAsU32(opt_i64.value(), dtype);
    }
    LOG(FATAL) << "Blackhole unsupported scalar argument: expected int" << dtype.bits;
  }
  if (dtype.code == kDLUInt) {
    auto opt_u32 = arg.try_cast<uint32_t>();
    if (opt_u32.has_value()) {
      return ExtractUnsignedScalarAsU32(opt_u32.value(), dtype);
    }
    auto opt_u64 = arg.try_cast<uint64_t>();
    if (opt_u64.has_value()) {
      return ExtractUnsignedScalarAsU32(opt_u64.value(), dtype);
    }
    LOG(FATAL) << "Blackhole unsupported scalar argument: expected uint" << dtype.bits;
  }
  if (dtype.code == kDLFloat) {
    ICHECK_EQ(dtype.bits, 32U)
        << "Blackhole unsupported scalar argument: float" << dtype.bits
        << " cannot be passed as a uint32 runtime argument";
    auto opt_f = arg.try_cast<float>();
    if (opt_f.has_value()) {
      return BitCastScalar<uint32_t>(opt_f.value());
    }
    auto opt_d = arg.try_cast<double>();
    if (opt_d.has_value()) {
      const float f = static_cast<float>(opt_d.value());
      return BitCastScalar<uint32_t>(f);
    }
    LOG(FATAL) << "Blackhole unsupported scalar argument: expected float32";
  }
  LOG(FATAL) << "Blackhole unsupported scalar argument type code " << dtype.code;
  return 0;
}

DLTensor* ExtractTensorArg(const ffi::AnyView& arg, void* void_arg) {
  auto opt_tensor = arg.try_cast<DLTensor*>();
  if (opt_tensor.has_value()) {
    return opt_tensor.value();
  }
  if (void_arg != nullptr) {
    DLTensor* tensor = *reinterpret_cast<DLTensor**>(void_arg);
    if (tensor != nullptr) {
      return tensor;
    }
  }
  LOG(FATAL) << "Cannot extract DLTensor* from packed argument";
  return nullptr;
}

// ============================================================================
// Direct TT-Metal path helpers (only when linked against TT-Metal)
// ============================================================================

#ifdef TILELANG_BLACKHOLE_DIRECT

using namespace tt::tt_metal;

template <typename T>
static std::vector<T> TransposeRowMajor2D(const T* src, uint32_t rows, uint32_t cols) {
  std::vector<T> out(static_cast<size_t>(rows) * cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
    }
  }
  return out;
}

static bool IsTwoDimTensor(const DLTensor* tensor) {
  return tensor != nullptr && tensor->ndim == 2;
}

static std::pair<uint32_t, uint32_t> GetTensorShape2D(const DLTensor* tensor) {
  ICHECK(IsTwoDimTensor(tensor));
  return {static_cast<uint32_t>(tensor->shape[0]), static_cast<uint32_t>(tensor->shape[1])};
}

static std::string DLTensorDataFormat(const DLTensor& tensor) {
  const DLDataType& dtype = tensor.dtype;
  if (dtype.code == kDLBfloat && dtype.bits == 16) return "Float16_b";
  if (dtype.code == kDLFloat && dtype.bits == 16) return "Float16";
  if (dtype.code == kDLFloat && dtype.bits == 32) return "Float32";
  if (dtype.code == kDLUInt && dtype.bits == 16) return "UInt16";
  if (dtype.code == kDLUInt && dtype.bits == 32) return "UInt32";
  if (dtype.code == kDLInt && dtype.bits == 16) return "Int16";
  if (dtype.code == kDLInt && dtype.bits == 32) return "Int32";
  return "unknown";
}

static void ValidateGemmTensorDType(const RuntimeTensorBinding& binding,
                                    const std::string& expected_dtype) {
  ICHECK(binding.tensor != nullptr);
  const std::string actual_dtype = DLTensorDataFormat(*binding.tensor);
  ICHECK_EQ(actual_dtype, expected_dtype)
      << "Unexpected tensor dtype for GEMM binding " << binding.name << ": got " << actual_dtype
      << ", expected " << expected_dtype;
}

static std::vector<const KernelComputeOpSpec*> CollectGemmComputeOps(const ExecutableSpec& spec) {
  std::vector<const KernelComputeOpSpec*> gemm_ops;
  for (const auto& kernel : spec.kernels) {
    for (const auto& compute_op : kernel.compute_ops) {
      if (!compute_op.enabled || compute_op.kind != "gemm") {
        continue;
      }
      gemm_ops.push_back(&compute_op);
    }
  }
  return gemm_ops;
}

static KernelComputeOpSpec GetPrimaryGemmCompute(const ExecutableSpec& spec) {
  const auto gemm_ops = CollectGemmComputeOps(spec);
  if (gemm_ops.empty()) {
    return KernelComputeOpSpec();
  }
  return *gemm_ops.front();
}

static void ValidateGemmComputeOpsShareDirectRuntimeWorkDecomposition(
    const ExecutableSpec& spec) {
  const auto gemm_ops = CollectGemmComputeOps(spec);
  if (gemm_ops.size() <= 1) {
    return;
  }

  const KernelComputeOpSpec& reference = *gemm_ops.front();
  for (size_t i = 1; i < gemm_ops.size(); ++i) {
    const KernelComputeOpSpec& gemm = *gemm_ops[i];
    ICHECK_EQ(gemm.M, reference.M)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on M";
    ICHECK_EQ(gemm.N, reference.N)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on N";
    ICHECK_EQ(gemm.K, reference.K)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on K";
    ICHECK_EQ(gemm.Mt, reference.Mt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Mt";
    ICHECK_EQ(gemm.Nt, reference.Nt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Nt";
    ICHECK_EQ(gemm.Kt, reference.Kt)
        << "Blackhole direct runtime requires compute_ops GEMM specs to agree on Kt";
  }
}

static void ValidateGemmComputeDirectRuntimeConstraints(const ExecutableSpec& spec) {
  ValidateGemmComputeOpsShareDirectRuntimeWorkDecomposition(spec);
  for (const KernelComputeOpSpec* gemm_op : CollectGemmComputeOps(spec)) {
    ICHECK(!gemm_op->has_mbarrier)
        << "Blackhole direct runtime does not yet support GEMM compute_ops.mbarrier bindings";
  }
}

static void ValidateGemmInputShape(const ExecutableSpec& spec,
                                   const RuntimeTensorBinding& binding,
                                   uint32_t rows,
                                   uint32_t cols) {
  const auto gemm = GetPrimaryGemmCompute(spec);
  const uint32_t logical_grid_x = std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
  const uint32_t logical_grid_y = std::max<uint32_t>(1, spec.core_plan.logical_grid_y);

  if (binding.name == gemm.a_buffer) {
    const uint32_t expected_rows = gemm.transpose_A ? gemm.K * logical_grid_y
                                                    : gemm.M * logical_grid_y;
    const uint32_t expected_cols = gemm.transpose_A ? gemm.M : gemm.K;
    ICHECK(rows == expected_rows && cols == expected_cols)
        << "Unexpected A tensor shape for GEMM direct path: got (" << rows << ", " << cols
        << "), expected (" << expected_rows << ", " << expected_cols
        << ") for logical grid " << logical_grid_y << "x" << logical_grid_x;
    return;
  }

  if (binding.name == gemm.b_buffer) {
    const uint32_t expected_rows = gemm.transpose_B ? gemm.N * logical_grid_x
                                                    : gemm.K * logical_grid_x;
    const uint32_t expected_cols = gemm.transpose_B ? gemm.K : gemm.N;
    ICHECK(rows == expected_rows && cols == expected_cols)
        << "Unexpected B tensor shape for GEMM direct path: got (" << rows << ", " << cols
        << "), expected (" << expected_rows << ", " << expected_cols
        << ") for transpose_B=" << gemm.transpose_B
        << " and logical grid " << logical_grid_x << "x" << logical_grid_y;
    return;
  }
}

static void ValidateGemmOutputShape(const ExecutableSpec& spec,
                                    uint32_t rows,
                                    uint32_t cols) {
  const auto gemm = GetPrimaryGemmCompute(spec);
  const uint32_t logical_grid_x = std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
  const uint32_t logical_grid_y = std::max<uint32_t>(1, spec.core_plan.logical_grid_y);
  const uint32_t expected_rows = gemm.M * logical_grid_y;
  const uint32_t expected_cols = gemm.N * logical_grid_x;
  ICHECK(rows == expected_rows && cols == expected_cols)
      << "Unexpected C tensor shape for GEMM direct path: got (" << rows << ", " << cols
      << "), expected (" << expected_rows << ", " << expected_cols
      << ") for logical grid " << logical_grid_x << "x" << logical_grid_y;
}

static const BufferMaterializationSpec& ResolveBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name);
static const BufferMaterializationSpec* FindBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name);

template <typename T>
static const T* GetTensorData(const DLTensor* tensor) {
  const uint8_t* base = static_cast<const uint8_t*>(tensor->data);
  return reinterpret_cast<const T*>(base + tensor->byte_offset);
}

template <typename T>
static T* GetTensorData(DLTensor* tensor) {
  uint8_t* base = static_cast<uint8_t*>(tensor->data);
  return reinterpret_cast<T*>(base + tensor->byte_offset);
}

static bool HasCompactRowMajorLayout(const DLTensor* tensor) {
  if (tensor == nullptr || tensor->ndim <= 0) {
    return false;
  }
  if (tensor->strides == nullptr) {
    return true;
  }
  int64_t expected_stride = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    if (tensor->strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= tensor->shape[i];
  }
  return true;
}

static void RequireCompactRowMajorLayout(const DLTensor* tensor,
                                         const char* context,
                                         const std::string& buffer_name) {
  ICHECK(HasCompactRowMajorLayout(tensor))
      << "Blackhole direct runtime " << context
      << " requires compact row-major DLTensor layout for buffer " << buffer_name;
}

static std::vector<int64_t> GetTensorShape(const DLTensor* tensor) {
  std::vector<int64_t> shape;
  shape.reserve(tensor->ndim);
  for (int i = 0; i < tensor->ndim; ++i) {
    shape.push_back(tensor->shape[i]);
  }
  return shape;
}

static int64_t ShapeProduct(const std::vector<int64_t>& shape, size_t begin, size_t end) {
  int64_t product = 1;
  for (size_t i = begin; i < end; ++i) {
    product *= shape[i];
  }
  return product;
}

static std::vector<int64_t> InvertAxisOrder(const std::vector<int64_t>& axis_order) {
  std::vector<int64_t> inverse(axis_order.size(), -1);
  for (size_t i = 0; i < axis_order.size(); ++i) {
    inverse[static_cast<size_t>(axis_order[i])] = static_cast<int64_t>(i);
  }
  return inverse;
}

static std::vector<int64_t> PermuteShape(const std::vector<int64_t>& shape,
                                         const std::vector<int64_t>& axis_order) {
  std::vector<int64_t> permuted_shape;
  permuted_shape.reserve(axis_order.size());
  for (int64_t axis : axis_order) {
    permuted_shape.push_back(shape[static_cast<size_t>(axis)]);
  }
  return permuted_shape;
}

static std::vector<int64_t> ComputeRowMajorStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = running;
    running *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

static bool IsValidAxisOrder(const std::vector<int64_t>& axis_order, int ndim) {
  if (static_cast<int>(axis_order.size()) != ndim) {
    return false;
  }
  std::vector<bool> seen(static_cast<size_t>(ndim), false);
  for (int64_t axis : axis_order) {
    if (axis < 0 || axis >= ndim || seen[static_cast<size_t>(axis)]) {
      return false;
    }
    seen[static_cast<size_t>(axis)] = true;
  }
  return true;
}

template <typename T>
static std::vector<T> PermuteContiguousTensorAxes(const T* src,
                                                  const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& axis_order) {
  const std::vector<int64_t> permuted_shape = PermuteShape(shape, axis_order);
  const std::vector<int64_t> input_strides = ComputeRowMajorStrides(shape);
  const std::vector<int64_t> output_strides = ComputeRowMajorStrides(permuted_shape);
  const size_t numel = static_cast<size_t>(ShapeProduct(shape, 0, shape.size()));
  std::vector<T> output(numel);
  for (size_t out_linear = 0; out_linear < numel; ++out_linear) {
    size_t remainder = out_linear;
    int64_t input_linear = 0;
    for (size_t i = 0; i < axis_order.size(); ++i) {
      const int64_t stride = output_strides[i];
      const int64_t coord =
          stride == 0 ? 0 : static_cast<int64_t>(remainder / static_cast<size_t>(stride));
      remainder %= static_cast<size_t>(stride);
      input_linear += coord * input_strides[static_cast<size_t>(axis_order[i])];
    }
    output[out_linear] = src[static_cast<size_t>(input_linear)];
  }
  return output;
}

struct InterleavedTilePlan {
  bool enabled = false;
  uint32_t tile_rows = 0;
  uint32_t tile_cols = 0;
  std::vector<int64_t> axis_order;
  bool transpose_2d = false;
};

static InterleavedTilePlan BuildInterleavedTilePlan(const ExecutableSpec& spec,
                                                    const BufferMaterializationSpec& materialization,
                                                    const DLTensor* tensor) {
  (void)spec;
  InterleavedTilePlan plan;
  if (tensor == nullptr || tensor->ndim < 2 || !HasCompactRowMajorLayout(tensor)) {
    return plan;
  }
  if (tensor->dtype.lanes != 1 || tensor->dtype.bits == 0) {
    return plan;
  }

  const uint32_t element_size_bytes = static_cast<uint32_t>((tensor->dtype.bits + 7) / 8);
  if (element_size_bytes == 0 || materialization.transport_page_size_bytes == 0 ||
      materialization.transport_page_size_bytes % element_size_bytes != 0) {
    return plan;
  }

  constexpr uint32_t kBlackholeTileCols = 32;
  const uint32_t tile_elements = materialization.transport_page_size_bytes / element_size_bytes;
  const uint32_t tile_rows = tile_elements / kBlackholeTileCols;
  if (tile_elements == 0 || tile_elements % kBlackholeTileCols != 0) {
    return plan;
  }

  if (!materialization.host_axis_order.empty()) {
    ICHECK(IsValidAxisOrder(materialization.host_axis_order, tensor->ndim))
        << "Invalid Blackhole host_axis_order materialization contract for buffer "
        << materialization.buffer;
    plan.axis_order = materialization.host_axis_order;
  } else {
    ICHECK(!materialization.host_axis_order.empty())
        << "Blackhole runtime requires explicit host_axis_order materialization contract for buffer "
        << materialization.buffer;
  }
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  int64_t total_rows = ShapeProduct(device_shape, 0, device_shape.size() - 1);
  int64_t cols = device_shape.back();
  if (materialization.transpose_2d) {
    std::swap(total_rows, cols);
  }
  if (tile_rows == 0 || cols <= 0 || cols % kBlackholeTileCols != 0 || total_rows <= 0 ||
      total_rows % tile_rows != 0) {
    return plan;
  }

  plan.enabled = true;
  plan.tile_rows = tile_rows;
  plan.tile_cols = kBlackholeTileCols;
  plan.transpose_2d = materialization.transpose_2d;
  return plan;
}

template <typename T>
static std::vector<uint8_t> BuildInterleavedTiledTransferData(const DLTensor* tensor,
                                                              const InterleavedTilePlan& plan) {
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<T> permuted = PermuteContiguousTensorAxes(
      GetTensorData<T>(tensor), host_shape, plan.axis_order);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  uint32_t rows = static_cast<uint32_t>(ShapeProduct(device_shape, 0, device_shape.size() - 1));
  uint32_t cols = static_cast<uint32_t>(device_shape.back());
  std::vector<T> row_major = permuted;
  if (plan.transpose_2d) {
    row_major = TransposeRowMajor2D(permuted.data(), rows, cols);
    std::swap(rows, cols);
  }
  std::vector<T> tiled = tilize_nfaces(row_major, rows, cols);
  std::vector<uint8_t> bytes(tiled.size() * sizeof(T));
  std::memcpy(bytes.data(), tiled.data(), bytes.size());
  return bytes;
}

template <typename T>
static void CopyInterleavedTiledOutputToTensor(DLTensor* tensor,
                                               const InterleavedTilePlan& plan,
                                               const std::vector<uint8_t>& output_data) {
  const std::vector<int64_t> host_shape = GetTensorShape(tensor);
  const std::vector<int64_t> device_shape = PermuteShape(host_shape, plan.axis_order);
  uint32_t rows = static_cast<uint32_t>(ShapeProduct(device_shape, 0, device_shape.size() - 1));
  uint32_t cols = static_cast<uint32_t>(device_shape.back());
  if (plan.transpose_2d) {
    std::swap(rows, cols);
  }
  const size_t numel = static_cast<size_t>(ShapeProduct(device_shape, 0, device_shape.size()));
  ICHECK_EQ(output_data.size(), numel * sizeof(T))
      << "Unexpected interleaved tiled output buffer size: got " << output_data.size()
      << " bytes, expected " << (numel * sizeof(T)) << " bytes";
  const auto* tiled = reinterpret_cast<const T*>(output_data.data());
  std::vector<T> tiled_vec(tiled, tiled + numel);
  if (BlackholeDebugIOEnabled()) {
    LOG(INFO) << "Blackhole debug output: tiled preview=" << PreviewTensorValues(tiled_vec);
  }
  std::vector<T> device_row_major = untilize_nfaces(tiled_vec, rows, cols);
  if (plan.transpose_2d) {
    device_row_major = TransposeRowMajor2D(device_row_major.data(), rows, cols);
  }
  const std::vector<int64_t> inverse_axis_order = InvertAxisOrder(plan.axis_order);
  std::vector<T> host_row_major =
      PermuteContiguousTensorAxes(device_row_major.data(), device_shape, inverse_axis_order);
  if (BlackholeDebugIOEnabled()) {
    LOG(INFO) << "Blackhole debug output: host preview=" << PreviewTensorValues(host_row_major);
  }
  std::memcpy(GetTensorData<T>(tensor), host_row_major.data(), host_row_major.size() * sizeof(T));
}

static std::vector<uint8_t> BuildInputTransferData(const ExecutableSpec& spec,
                                                   const RuntimeTensorBinding& binding) {
  const DLTensor* tensor = binding.tensor;
  ICHECK(tensor != nullptr);
  RequireCompactRowMajorLayout(tensor, "input transfer", binding.name);
  const size_t tensor_size = GetDataSize(*tensor);
  const auto gemm = GetPrimaryGemmCompute(spec);

  if (!gemm.enabled || gemm.kind != "gemm" || !IsTwoDimTensor(tensor)) {
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    const InterleavedTilePlan tile_plan =
        BuildInterleavedTilePlan(spec, materialization, tensor);
    if (tile_plan.enabled) {
      if (tensor->dtype.bits == 16) {
        auto tiled = BuildInterleavedTiledTransferData<uint16_t>(tensor, tile_plan);
        if (BlackholeDebugIOEnabled()) {
          const auto* values = reinterpret_cast<const uint16_t*>(tiled.data());
          std::vector<uint16_t> preview(values, values + tiled.size() / sizeof(uint16_t));
          LOG(INFO) << "Blackhole debug input " << binding.name
                    << ": tiled preview=" << PreviewTensorValues(preview);
        }
        return tiled;
      }
      if (tensor->dtype.bits == 32) {
        auto tiled = BuildInterleavedTiledTransferData<uint32_t>(tensor, tile_plan);
        if (BlackholeDebugIOEnabled()) {
          const auto* values = reinterpret_cast<const uint32_t*>(tiled.data());
          std::vector<uint32_t> preview(values, values + tiled.size() / sizeof(uint32_t));
          LOG(INFO) << "Blackhole debug input " << binding.name
                    << ": tiled preview=" << PreviewTensorValues(preview);
        }
        return tiled;
      }
    }
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), GetTensorData<uint8_t>(tensor), tensor_size);
    return raw;
  }

  if (binding.name != gemm.a_buffer && binding.name != gemm.b_buffer) {
    std::vector<uint8_t> raw(tensor_size);
    std::memcpy(raw.data(), GetTensorData<uint8_t>(tensor), tensor_size);
    return raw;
  }

  const std::string expected_tensor_dtype =
      binding.name == gemm.a_buffer ? gemm.a_tensor_dtype : gemm.b_tensor_dtype;
  const std::string expected_cb_dtype =
      binding.name == gemm.a_buffer ? gemm.a_cb_dtype : gemm.b_cb_dtype;
  ValidateGemmTensorDType(binding, expected_tensor_dtype);
  ICHECK_EQ(expected_tensor_dtype, expected_cb_dtype)
      << "Blackhole direct GEMM currently requires identical tensor and CB dtype for "
      << binding.name;
  ICHECK_EQ(expected_tensor_dtype, "Float16_b")
      << "Blackhole direct GEMM currently supports only bfloat16 inputs, but " << binding.name
      << " requested " << expected_tensor_dtype;
  const auto* raw = GetTensorData<uint16_t>(tensor);
  const auto [rows, cols] = GetTensorShape2D(tensor);
  ValidateGemmInputShape(spec, binding, rows, cols);

  std::vector<uint16_t> tiled;
  if ((binding.name == gemm.a_buffer && gemm.transpose_A) ||
      (binding.name == gemm.b_buffer && gemm.transpose_B)) {
    tiled = tilize_nfaces(TransposeRowMajor2D(raw, rows, cols), cols, rows);
  } else {
    std::vector<uint16_t> row_major(raw, raw + static_cast<size_t>(rows) * cols);
    tiled = tilize_nfaces(row_major, rows, cols);
  }

  std::vector<uint8_t> bytes(tiled.size() * sizeof(uint16_t));
  std::memcpy(bytes.data(), tiled.data(), bytes.size());
  return bytes;
}

static void CopyOutputFromDeviceBuffer(const ExecutableSpec& spec,
                                       const RuntimeTensorBinding& binding,
                                       const std::vector<uint8_t>& output_data) {
  ICHECK(binding.tensor != nullptr);
  RequireCompactRowMajorLayout(binding.tensor, "output transfer", binding.name);
  const size_t tensor_size = GetDataSize(*binding.tensor);
  ICHECK(output_data.size() >= tensor_size)
      << "Output data size mismatch for " << binding.name;
  const auto gemm = GetPrimaryGemmCompute(spec);
  const BufferMaterializationSpec* materialized_output =
      FindBufferMaterializationSpec(spec, binding.name);

  if (!gemm.enabled || gemm.kind != "gemm" || binding.name != gemm.c_buffer ||
      (materialized_output != nullptr && !materialized_output->live_form_kind.empty()) ||
      !IsTwoDimTensor(binding.tensor)) {
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    const InterleavedTilePlan tile_plan =
        BuildInterleavedTilePlan(spec, materialization, binding.tensor);
    if (tile_plan.enabled) {
      if (binding.tensor->dtype.bits == 16) {
        CopyInterleavedTiledOutputToTensor<uint16_t>(binding.tensor, tile_plan, output_data);
        return;
      }
      if (binding.tensor->dtype.bits == 32) {
        CopyInterleavedTiledOutputToTensor<uint32_t>(binding.tensor, tile_plan, output_data);
        return;
      }
    }
    std::memcpy(GetTensorData<uint8_t>(binding.tensor), output_data.data(), tensor_size);
    return;
  }

  ValidateGemmTensorDType(binding, gemm.c_tensor_dtype);
  ICHECK_EQ(gemm.c_cb_dtype, gemm.accumulator_dtype)
      << "Blackhole direct GEMM currently requires identical output CB and accumulator dtypes";
  ICHECK_EQ(gemm.c_tensor_dtype, "Float32")
      << "Blackhole direct GEMM currently supports only float32 outputs, but "
      << gemm.c_buffer << " requested " << gemm.c_tensor_dtype;
  ICHECK_EQ(gemm.accumulator_dtype, "Float32")
      << "Blackhole direct GEMM currently supports only float32 accumulators, but requested "
      << gemm.accumulator_dtype;
  const auto [rows, cols] = GetTensorShape2D(binding.tensor);
  ValidateGemmOutputShape(spec, rows, cols);
  const size_t numel = static_cast<size_t>(rows) * cols;
  ICHECK(output_data.size() == numel * sizeof(float))
      << "Unexpected GEMM output buffer size for " << binding.name << ": got "
      << output_data.size() << " bytes, expected " << (numel * sizeof(float));
  const auto* tiled = reinterpret_cast<const float*>(output_data.data());
  std::vector<float> tiled_vec(tiled, tiled + numel);
  std::vector<float> row_major = untilize_nfaces(tiled_vec, rows, cols);
  std::memcpy(GetTensorData<float>(binding.tensor), row_major.data(),
              row_major.size() * sizeof(float));
}

static tt::DataFormat ParseDataFormat(const std::string& value) {
  if (value == "Float16") return tt::DataFormat::Float16;
  if (value == "Float16_b") return tt::DataFormat::Float16_b;
  if (value == "Float32") return tt::DataFormat::Float32;
  if (value == "UInt16") return tt::DataFormat::UInt16;
  if (value == "UInt32") return tt::DataFormat::UInt32;
  LOG(FATAL) << "Unsupported data format: " << value;
  return tt::DataFormat::Float16_b;
}

static const BufferMaterializationSpec& ResolveBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name) {
  const BufferMaterializationSpec* materialization =
      FindBufferMaterializationSpec(spec, buffer_name);
  ICHECK(materialization != nullptr)
      << "Missing Blackhole buffer materialization spec for buffer " << buffer_name;
  ICHECK_EQ(materialization->materialization_kind, "replicated")
      << "Unsupported Blackhole buffer materialization kind for " << buffer_name << ": "
      << materialization->materialization_kind;
  ICHECK_EQ(materialization->memory_space, "dram")
      << "Unsupported Blackhole buffer memory_space for " << buffer_name << ": "
      << materialization->memory_space;
  ICHECK_GT(materialization->transport_page_size_bytes, 0U)
      << "Blackhole buffer materialization requires transport_page_size for " << buffer_name;
  return *materialization;
}

static const BufferMaterializationSpec* FindBufferMaterializationSpec(
    const ExecutableSpec& spec,
    const std::string& buffer_name) {
  auto it = std::find_if(spec.buffer_materializations.begin(), spec.buffer_materializations.end(),
                         [&](const BufferMaterializationSpec& materialization) {
                           return materialization.buffer == buffer_name;
                         });
  if (it == spec.buffer_materializations.end()) {
    return nullptr;
  }
  return &*it;
}

static void ValidateSemaphoreCoreType(const std::string& core_type) {
  ICHECK(core_type.empty() || core_type == "worker")
      << "Unsupported Blackhole semaphore core_type: " << core_type;
}

static CoreRangeSet BuildSemaphoreCoreRangeSet(const SemaphoreSpec& semaphore) {
  std::vector<CoreRange> core_ranges;
  core_ranges.reserve(semaphore.core_ranges.size());
  for (const auto& range : semaphore.core_ranges) {
    const CoreCoord start{range.start.core_x, range.start.core_y};
    const CoreCoord end{range.end.core_x, range.end.core_y};
    core_ranges.emplace_back(start, end);
  }
  return CoreRangeSet(std::move(core_ranges));
}

static std::unordered_map<uint32_t, uint32_t> CreateSemaphoresFromSpec(Program& program,
                                                                       const ExecutableSpec& spec) {
  std::unordered_map<uint32_t, uint32_t> semaphore_ids;
  for (const auto& semaphore : spec.semaphores) {
    ICHECK(!semaphore.core_ranges.empty())
        << "Blackhole semaphore_plan entry id=" << semaphore.id
        << " must define at least one core range";
    ValidateSemaphoreCoreType(semaphore.core_type);
    const CoreRangeSet core_ranges = BuildSemaphoreCoreRangeSet(semaphore);
    uint32_t created_id = CreateSemaphore(program, core_ranges, semaphore.initial_value);
    ICHECK_EQ(created_id, semaphore.id)
        << "Blackhole semaphore_plan id mismatch: requested " << semaphore.id
        << ", TT-Metal allocated " << created_id;
    semaphore_ids.emplace(semaphore.id, created_id);
  }
  return semaphore_ids;
}

static uint32_t GetRuntimeNumKTiles(const ExecutableSpec& spec) {
  const auto gemm = GetPrimaryGemmCompute(spec);
  if (gemm.enabled && gemm.kind == "gemm") {
    ICHECK_GT(gemm.Kt, 0U)
        << "Blackhole GEMM direct path requires compute_ops GEMM Kt to be populated";
    return std::max<uint32_t>(1, gemm.Kt);
  }
  return 0;
}

static uint32_t GetRuntimeLogicalGridX(const ExecutableSpec& spec) {
  return std::max<uint32_t>(1, spec.core_plan.logical_grid_x);
}

static uint32_t GetRuntimeLogicalNTiles(const ExecutableSpec& spec) {
  const auto gemm = GetPrimaryGemmCompute(spec);
  ICHECK(gemm.enabled && gemm.kind == "gemm")
      << "logical_n_tiles is only defined for GEMM kernels in Blackhole direct runtime";
  ICHECK_GT(gemm.Nt, 0U)
      << "Blackhole GEMM direct path requires compute_ops GEMM Nt to be populated";
  const uint32_t local_n_tiles = std::max<uint32_t>(1, gemm.Nt);
  const uint32_t logical_grid_x = GetRuntimeLogicalGridX(spec);
  return local_n_tiles * logical_grid_x;
}

static void CreateCircularBuffersFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const ExecutableSpec& spec) {
  for (const auto& cb : spec.cb_configs) {
    uint32_t total_size = cb.num_pages * cb.page_size_bytes;
    CircularBufferConfig cb_config(
        total_size,
        {{static_cast<uint8_t>(cb.cb_id), ParseDataFormat(cb.data_format)}});
    cb_config.set_page_size(static_cast<uint8_t>(cb.cb_id), cb.page_size_bytes);
    CreateCircularBuffer(program, core_spec, cb_config);
  }
}

struct RuntimeBufferBinding {
  std::shared_ptr<distributed::MeshBuffer> mesh_buffer;
  size_t size_bytes{0};
  bool is_output{false};
};

static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::string& kernel_path);
static std::vector<uint32_t> BuildCommonRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids);
static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<uint32_t>& scalar_args);

struct DirectWorkItem {
  uint32_t work_id;
  CoreCoord core;
};

struct DirectLaunchWave {
  std::vector<DirectWorkItem> work_items;
  std::vector<CoreCoord> launch_cores;
};

struct DirectRuntimeBufferState {
  std::unordered_map<std::string, RuntimeBufferBinding> runtime_buffers;
  std::vector<std::string> input_names;
  std::vector<std::string> ordered_output_names;
};

struct SynchronizationRuntimeContext {
  const IDevice* device{nullptr};
  const std::unordered_map<uint32_t, uint32_t>* semaphore_ids{nullptr};
};

static uint64_t EncodeDirectLaunchCoreKey(const CoreCoord& core) {
  return (static_cast<uint64_t>(core.x) << 32) | static_cast<uint64_t>(core.y);
}

static bool HasDirectLaunchSynchronizationContract(const ExecutableSpec& spec) {
  if (!spec.semaphores.empty()) {
    return true;
  }
  return std::any_of(spec.kernels.begin(), spec.kernels.end(), [](const KernelSpec& kernel) {
    return !kernel.semaphore_bindings.empty() || !kernel.remote_core_descriptors.empty();
  });
}

static std::vector<DirectLaunchWave> BuildDirectLaunchWaves(const ExecutableSpec& spec,
                                                            const std::string& func_name) {
  const auto& work_packets = spec.core_plan.work_packets;
  uint64_t total_work_items = 0;
  uint32_t max_work_count = 0;
  for (const auto& packet : work_packets) {
    total_work_items += packet.work_count;
    max_work_count = std::max(max_work_count, packet.work_count);
  }
  ICHECK_GT(total_work_items, 0U)
      << "Blackhole planner/runtime contract requires at least one logical work item for "
      << func_name;
  ICHECK_GT(max_work_count, 0U)
      << "Blackhole planner/runtime contract requires positive work_count in core_plan.work_packets "
         "for "
      << func_name;

  const bool oversubscribed = total_work_items > work_packets.size();
  if (oversubscribed) {
    ICHECK(!HasDirectLaunchSynchronizationContract(spec))
        << "Blackhole direct runtime oversubscribed launch currently wave-schedules logical work "
           "items across repeated program launches. This is only supported when the executable "
           "has no explicit semaphore or remote-core synchronization contract. Function "
        << func_name << " maps " << total_work_items << " logical work items onto "
        << work_packets.size() << " physical-core packets.";
  }

  std::vector<DirectLaunchWave> waves;
  waves.reserve(max_work_count);
  for (uint32_t wave_index = 0; wave_index < max_work_count; ++wave_index) {
    DirectLaunchWave wave;
    wave.work_items.reserve(work_packets.size());
    wave.launch_cores.reserve(work_packets.size());
    std::unordered_set<uint64_t> seen_cores;
    seen_cores.reserve(work_packets.size());
    for (const auto& packet : work_packets) {
      if (wave_index >= packet.work_count) {
        continue;
      }
      const CoreCoord packet_core{packet.core_x, packet.core_y};
      ICHECK(seen_cores.insert(EncodeDirectLaunchCoreKey(packet_core)).second)
          << "Blackhole planner/runtime contract requires each launch wave to map at most one "
             "logical work item to a physical core. Function "
          << func_name << " duplicates core (" << packet_core.x << "," << packet_core.y
          << ") within wave " << wave_index;
      wave.work_items.push_back({packet.work_offset + wave_index, packet_core});
      wave.launch_cores.push_back(packet_core);
    }
    if (!wave.work_items.empty()) {
      waves.push_back(std::move(wave));
    }
  }

  ICHECK(!waves.empty()) << "No launch waves resolved for direct execution";
  return waves;
}

static DirectRuntimeBufferState MaterializeRuntimeBuffers(
    distributed::MeshCommandQueue& cq,
    distributed::MeshDevice* mesh_device,
    const ExecutableSpec& spec,
    const std::vector<RuntimeTensorBinding>& buffer_args,
    const std::vector<std::string>& output_names) {
  DirectRuntimeBufferState state;
  for (const auto& binding : buffer_args) {
    ICHECK(binding.tensor != nullptr) << "Null tensor passed to Blackhole direct path";
    const size_t tensor_size = GetDataSize(*binding.tensor);
    const auto& materialization = ResolveBufferMaterializationSpec(spec, binding.name);
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = materialization.transport_page_size_bytes,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = tensor_size};
    auto mesh_buffer =
        distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device);
    state.runtime_buffers.emplace(binding.name, RuntimeBufferBinding{
                                                .mesh_buffer = mesh_buffer,
                                                .size_bytes = tensor_size,
                                                .is_output = binding.is_output,
                                            });
    if (binding.is_output) {
      state.ordered_output_names.push_back(binding.name);
    } else {
      state.input_names.push_back(binding.name);
    }
    std::vector<uint8_t> initial_data = BuildInputTransferData(spec, binding);
    EnqueueWriteMeshBuffer(cq, mesh_buffer, initial_data, /*blocking=*/true);
  }
  if (!output_names.empty()) {
    state.ordered_output_names = output_names;
  }
  return state;
}

static std::vector<std::string> WriteKernelSourceFiles(const ExecutableSpec& spec,
                                                       const std::string& func_name,
                                                       const std::string& tmp_dir) {
  std::vector<std::string> kernel_paths;
  kernel_paths.reserve(spec.kernels.size());
  for (size_t i = 0; i < spec.kernels.size(); ++i) {
    const auto& kernel = spec.kernels[i];
    std::string kernel_path = tmp_dir + "/" + func_name + "_" + std::to_string(i) + "_" +
                              kernel.kind + ".cpp";
    std::ofstream ofs(kernel_path);
    if (!ofs) {
      LOG(FATAL) << "Failed to write kernel file: " << kernel_path;
    }
    ofs << NormalizeBlackholeKernelSource(kernel.source_code);
    kernel_paths.push_back(kernel_path);
  }
  return kernel_paths;
}

static std::vector<KernelHandle> CreateProgramKernelsFromSpec(
    Program& program,
    const CoreRangeSet& launch_core_ranges,
    const ExecutableSpec& spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& runtime_buffers,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<std::string>& kernel_paths) {
  std::vector<KernelHandle> kernels;
  kernels.reserve(spec.kernels.size());
  for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
    const auto& kernel_spec = spec.kernels[ki];
    LOG(INFO) << "Direct path: create kernel[" << ki << "] kind=" << kernel_spec.kind
              << " core_type=" << kernel_spec.core_type;
    kernels.push_back(CreateKernelFromSpec(program, launch_core_ranges, kernel_spec,
                                           runtime_buffers, kernel_paths[ki]));
    const auto common_runtime_args =
        BuildCommonRuntimeArgsFromSpec(kernel_spec, runtime_buffers, semaphore_ids);
    if (!common_runtime_args.empty()) {
      LOG(INFO) << "Direct path: set common runtime args kernel[" << ki
                << "] count=" << common_runtime_args.size();
      SetCommonRuntimeArgs(program, kernels.back(), common_runtime_args);
    }
  }
  return kernels;
}

static void ApplyWorkItemRuntimeArgs(
    Program& program,
    const ExecutableSpec& spec,
    const std::vector<KernelHandle>& kernels,
    const std::vector<DirectWorkItem>& work_items,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& runtime_buffers,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<uint32_t>& scalar_args) {
  for (const auto& item : work_items) {
    for (size_t ki = 0; ki < spec.kernels.size(); ++ki) {
      const auto& kernel_spec = spec.kernels[ki];
      auto runtime_args = BuildRuntimeArgsFromSpec(
          kernel_spec, spec, item.work_id, device, runtime_buffers, semaphore_ids, scalar_args);
      SetRuntimeArgs(program, kernels[ki], item.core, runtime_args);
    }
  }
}

static std::vector<uint32_t> BuildKernelCompileTimeArgs(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings);

static DataMovementProcessor ParseDataMovementProcessor(const std::string& processor) {
  if (processor == "riscv_0") {
    return DataMovementProcessor::RISCV_0;
  }
  if (processor == "riscv_1") {
    return DataMovementProcessor::RISCV_1;
  }
  LOG(FATAL) << "Unsupported Blackhole launch_spec processor: " << processor;
}

static MathFidelity ParseMathFidelity(const std::string& math_fidelity) {
  if (math_fidelity == "LoFi") {
    return MathFidelity::LoFi;
  }
  if (math_fidelity == "HiFi2") {
    return MathFidelity::HiFi2;
  }
  if (math_fidelity == "HiFi3") {
    return MathFidelity::HiFi3;
  }
  if (math_fidelity.empty() || math_fidelity == "HiFi4") {
    return MathFidelity::HiFi4;
  }
  LOG(FATAL) << "Unsupported Blackhole compute math_fidelity: " << math_fidelity;
}

static UnpackToDestMode ParseUnpackToDestMode(const std::string& mode) {
  if (mode == "Default") {
    return UnpackToDestMode::Default;
  }
  if (mode == "UnpackToDestFp32") {
    return UnpackToDestMode::UnpackToDestFp32;
  }
  LOG(FATAL) << "Unsupported Blackhole unpack_to_dest_mode: " << mode;
}

static NOC ParseNoc(const std::string& noc) {
  if (noc == "riscv_0_default") {
    return NOC::RISCV_0_default;
  }
  if (noc == "riscv_1_default") {
    return NOC::RISCV_1_default;
  }
  LOG(FATAL) << "Unsupported Blackhole launch_spec noc: " << noc;
}

static void AppendCompileTimeArgValues(const CompileTimeArgSpec& spec,
                                       std::vector<uint32_t>* compile_time_args) {
  const size_t before = compile_time_args->size();
  for (uint32_t value : spec.values) {
    compile_time_args->push_back(value);
  }
  const uint32_t emitted_count =
      static_cast<uint32_t>(compile_time_args->size() - before);
  ICHECK_GT(emitted_count, 0U)
      << "Blackhole compile-time ABI kind " << spec.kind
      << " did not materialize any values";
  if (spec.count != 0U) {
    ICHECK_EQ(spec.count, emitted_count)
        << "Blackhole compile-time ABI kind " << spec.kind
        << " has count mismatch: expected " << spec.count
        << ", materialized " << emitted_count;
  }
}

static void ValidateDirectRuntimeAccessorSpec(const std::string& buffer_name,
                                              const std::string& layout,
                                              const std::string& memory_space,
                                              uint32_t common_runtime_arg_count,
                                              uint32_t args_config_bits) {
  ICHECK_EQ(layout, "interleaved")
      << "Blackhole direct runtime currently supports only interleaved accessors";
  ICHECK_EQ(memory_space, "dram")
      << "Blackhole direct runtime currently supports only DRAM accessors";
  ICHECK_EQ(common_runtime_arg_count, 0U)
      << "Blackhole direct runtime currently supports only interleaved accessors without common runtime args";
  ICHECK_EQ(args_config_bits, 2U)
      << "Blackhole direct runtime expects interleaved DRAM accessor args_config_bits == 2";
}

static void AppendInterleavedAccessorCompileTimeArgs(
    const std::string& buffer_name,
    uint32_t expected_count,
    uint32_t args_config_bits,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* compile_time_args) {
  ICHECK(!buffer_name.empty())
      << "Blackhole interleaved accessor compile-time ABI entry is missing a buffer name";
  ValidateDirectRuntimeAccessorSpec(buffer_name, "interleaved", "dram",
                                    /*common_runtime_arg_count=*/0, args_config_bits);
  auto it = buffer_bindings.find(buffer_name);
  ICHECK(it != buffer_bindings.end())
      << "Missing runtime buffer binding for accessor buffer " << buffer_name;
  const auto underlying_args_config_bits = static_cast<tensor_accessor::ArgsConfig::Underlying>(
      args_config_bits);
  const tensor_accessor::ArgsConfig args_config(underlying_args_config_bits);
  ICHECK((args_config & tensor_accessor::ArgConfig::Runtime).raw() == 0)
      << "Blackhole direct runtime does not yet support accessor common runtime args";

  const size_t before = compile_time_args->size();
  TensorAccessorArgs(*(it->second.mesh_buffer), args_config).append_to(*compile_time_args);
  const uint32_t emitted_count =
      static_cast<uint32_t>(compile_time_args->size() - before);
  ICHECK_EQ(emitted_count, 2U)
      << "Blackhole interleaved accessor compile-time ABI for buffer " << buffer_name
      << " must materialize exactly two uint32 values";
  if (expected_count != 0U) {
    ICHECK_EQ(expected_count, emitted_count)
        << "Blackhole interleaved accessor compile-time ABI count mismatch for buffer "
        << buffer_name;
  }
}

static void AppendAccessorCompileTimeArgs(
    const CompileTimeArgSpec& spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* compile_time_args) {
  ICHECK(!spec.buffer.empty())
      << "Blackhole interleaved accessor compile-time ABI requires explicit buffer binding for "
      << spec.name;
  AppendInterleavedAccessorCompileTimeArgs(spec.buffer, spec.count, spec.args_config_bits,
                                           buffer_bindings, compile_time_args);
}

static bool IsSupportedCommonRuntimeArgKind(const std::string& kind) {
  return kind == "input_buffer_addr" || kind == "input_buffer_addr32" ||
         kind == "output_buffer_addr" || kind == "output_buffer_addr32" ||
         kind == "semaphore_id_u32";
}

static bool IsLogicalCoreNocRuntimeArgKind(const std::string& kind) {
  return kind == "logical_core_noc_x" || kind == "logical_core_noc_y";
}

static const RemoteCoreDescriptorSpec* ResolveRemoteCoreDescriptorSpec(
    const KernelSpec& kernel, const KernelArgSpec& arg_spec) {
  ICHECK(!arg_spec.identity.empty())
      << "Blackhole synchronization schema requires identity for runtime arg " << arg_spec.name
      << " kind=" << arg_spec.kind;
  const RemoteCoreDescriptorSpec* matched = nullptr;
  for (const auto& descriptor : kernel.remote_core_descriptors) {
    if (descriptor.identity != arg_spec.identity) {
      continue;
    }
    matched = &descriptor;
    break;
  }
  ICHECK(matched != nullptr)
      << "Blackhole synchronization schema requires a matching remote core descriptor for runtime arg "
      << arg_spec.name << " identity=" << arg_spec.identity;
  return matched;
}

static const SemaphoreBindingSpec* ResolveSemaphoreBindingSpec(const KernelSpec& kernel,
                                                              const KernelArgSpec& arg_spec) {
  const SemaphoreBindingSpec* matched = nullptr;
  for (const auto& binding : kernel.semaphore_bindings) {
    if (binding.name != arg_spec.name || binding.arg_kind != arg_spec.kind) {
      continue;
    }
    ICHECK(matched == nullptr)
        << "Blackhole synchronization schema requires a unique semaphore binding for runtime arg "
        << arg_spec.name << " kind=" << arg_spec.kind;
    matched = &binding;
  }
  ICHECK(matched != nullptr)
      << "Blackhole synchronization schema requires a matching semaphore binding for runtime arg "
      << arg_spec.name << " kind=" << arg_spec.kind;
  return matched;
}

static void ValidateLogicalCoreNocRuntimeArgs(const KernelSpec& kernel) {
  struct LogicalCorePairState {
    const KernelArgSpec* x{nullptr};
    const KernelArgSpec* y{nullptr};
  };

  std::unordered_map<std::string, LogicalCorePairState> pair_by_identity;
  for (const auto& arg_spec : kernel.runtime_args) {
    if (!IsLogicalCoreNocRuntimeArgKind(arg_spec.kind)) {
      continue;
    }
    ICHECK(!arg_spec.identity.empty())
        << "Blackhole synchronization schema requires identity for runtime arg " << arg_spec.name
        << " kind=" << arg_spec.kind;
    ICHECK(arg_spec.has_core_coord)
        << "Blackhole synchronization schema requires core_x/core_y for runtime arg "
        << arg_spec.name << " kind=" << arg_spec.kind;
    auto& pair = pair_by_identity[arg_spec.identity];
    if (arg_spec.kind == "logical_core_noc_x") {
      ICHECK(pair.x == nullptr)
          << "Blackhole synchronization core descriptor " << arg_spec.identity
          << " cannot define logical_core_noc_x more than once";
      pair.x = &arg_spec;
    } else {
      ICHECK(pair.y == nullptr)
          << "Blackhole synchronization core descriptor " << arg_spec.identity
          << " cannot define logical_core_noc_y more than once";
      pair.y = &arg_spec;
    }
  }

  for (const auto& entry : pair_by_identity) {
    const auto& pair = entry.second;
    ICHECK(pair.x != nullptr && pair.y != nullptr)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must define both logical_core_noc_x and logical_core_noc_y";
    ICHECK_EQ(pair.x->core_x, pair.y->core_x)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must use one logical core for logical_core_noc_x/y";
    ICHECK_EQ(pair.x->core_y, pair.y->core_y)
        << "Blackhole synchronization core descriptor " << entry.first
        << " must use one logical core for logical_core_noc_x/y";

    auto descriptor_it =
        std::find_if(kernel.remote_core_descriptors.begin(), kernel.remote_core_descriptors.end(),
                     [&](const RemoteCoreDescriptorSpec& descriptor) {
                       return descriptor.identity == entry.first;
                     });
    ICHECK(descriptor_it != kernel.remote_core_descriptors.end())
        << "Blackhole synchronization core descriptor " << entry.first
        << " must be materialized into KernelSpec.remote_core_descriptors";
    ICHECK_EQ(descriptor_it->core_x, pair.x->core_x)
        << "Blackhole synchronization core descriptor " << entry.first
        << " core_x mismatch between runtime args and KernelSpec.remote_core_descriptors";
    ICHECK_EQ(descriptor_it->core_y, pair.x->core_y)
        << "Blackhole synchronization core descriptor " << entry.first
        << " core_y mismatch between runtime args and KernelSpec.remote_core_descriptors";
  }
}

static void ValidateKernelSynchronizationSchema(
    const KernelSpec& kernel, const std::unordered_set<uint32_t>& planned_semaphore_ids) {
  ValidateLogicalCoreNocRuntimeArgs(kernel);

  auto validate_semaphore_runtime_arg = [&](const KernelArgSpec& arg_spec) {
    if (arg_spec.kind != "semaphore_id_u32") {
      return;
    }
    const auto* binding = ResolveSemaphoreBindingSpec(kernel, arg_spec);
    ICHECK(planned_semaphore_ids.count(binding->semaphore_id))
        << "Blackhole synchronization schema requires semaphore binding " << binding->name
        << " to reference a planned semaphore id; missing id " << binding->semaphore_id;
  };

  for (const auto& arg_spec : kernel.common_runtime_args) {
    validate_semaphore_runtime_arg(arg_spec);
  }
  for (const auto& arg_spec : kernel.runtime_args) {
    validate_semaphore_runtime_arg(arg_spec);
  }
}

static void ValidateKernelDirectRuntimeSchema(const KernelSpec& kernel) {
  for (const auto& arg_spec : kernel.common_runtime_args) {
    ICHECK(IsSupportedCommonRuntimeArgKind(arg_spec.kind))
        << "Blackhole direct runtime only supports shared common runtime args for "
           "buffer addresses and semaphores; unsupported common runtime arg kind: "
        << arg_spec.kind;
  }

  for (const auto& accessor : kernel.accessors) {
    ValidateDirectRuntimeAccessorSpec(accessor.buffer, accessor.layout, accessor.memory_space,
                                      accessor.common_runtime_arg_count,
                                      accessor.args_config_bits);
  }

  for (const auto& spec : kernel.compile_time_arg_specs) {
    if (spec.kind != "interleaved_accessor_cta") {
      continue;
    }
    ICHECK(!spec.buffer.empty())
        << "Blackhole direct runtime requires explicit buffer binding for compile-time ABI "
        << spec.name;
    ValidateDirectRuntimeAccessorSpec(spec.buffer, spec.layout, spec.memory_space,
                                      /*common_runtime_arg_count=*/0, spec.args_config_bits);
  }
}

static void ValidateKernelDirectRuntimeConstraints(const KernelSpec& kernel) {
  if (kernel.has_launch_spec && !kernel.launch_spec.core_type.empty()) {
    ICHECK_EQ(kernel.launch_spec.core_type, kernel.core_type)
        << "Blackhole launch_spec.core_type mismatch for kernel " << kernel.name
        << ": launch_spec.core_type=" << kernel.launch_spec.core_type
        << ", kernel.core_type=" << kernel.core_type;
  }
  ValidateKernelDirectRuntimeSchema(kernel);
}

static void ValidateExecutableSpecSynchronizationSchema(const std::string& func_name,
                                                        const ExecutableSpec& spec) {
  std::unordered_set<uint32_t> planned_semaphore_ids;
  for (const auto& semaphore : spec.semaphores) {
    planned_semaphore_ids.insert(semaphore.id);
  }
  for (const auto& kernel : spec.kernels) {
    ValidateKernelSynchronizationSchema(kernel, planned_semaphore_ids);
  }
}

static std::vector<uint32_t> BuildKernelCompileTimeArgsFromSchema(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  std::vector<uint32_t> compile_time_args;
  std::vector<CompileTimeArgSpec> compile_time_arg_specs = kernel.compile_time_arg_specs;
  std::sort(compile_time_arg_specs.begin(), compile_time_arg_specs.end(),
            [](const CompileTimeArgSpec& a, const CompileTimeArgSpec& b) {
              if (a.offset != b.offset) {
                return a.offset < b.offset;
              }
              return a.name < b.name;
            });

  uint32_t expected_offset = 0;
  for (const auto& spec : compile_time_arg_specs) {
    ICHECK_EQ(spec.offset, expected_offset)
        << "Blackhole compile-time ABI offset mismatch for " << spec.name
        << ": got " << spec.offset << ", expected " << expected_offset;

    if (spec.kind == "interleaved_accessor_cta") {
      AppendAccessorCompileTimeArgs(spec, buffer_bindings, &compile_time_args);
    } else if (spec.kind == "gemm_shape" || spec.kind == "gemm_transpose_flags" ||
               spec.kind == "gemm_block_shape" || spec.kind == "gemm_subblock_shape" ||
               spec.kind == "gemm_clear_accum" || spec.kind == "gemm_k_pack" ||
               spec.kind == "gemm_wg_wait" || spec.kind == "gemm_policy" ||
               spec.kind == "literal_u32") {
      AppendCompileTimeArgValues(spec, &compile_time_args);
    } else {
      LOG(FATAL) << "Unsupported Blackhole compile-time ABI kind: " << spec.kind;
    }

    expected_offset = static_cast<uint32_t>(compile_time_args.size());
  }

  return compile_time_args;
}

static KernelHandle CreateKernelFromSpec(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::string& kernel_path) {
  ValidateKernelDirectRuntimeConstraints(kernel);
  const std::vector<uint32_t> compile_time_args =
      BuildKernelCompileTimeArgs(kernel, buffer_bindings);
  const std::string core_type = kernel.core_type;
  if (core_type == "trisc" || kernel.kind == "compute") {
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool dst_full_sync_en = false;
    bool math_approx_mode = false;
    bool bfp8_pack_precise = false;
    if (kernel.has_compute_config) {
      math_fidelity = ParseMathFidelity(kernel.compute_config.math_fidelity);
      fp32_dest_acc_en = kernel.compute_config.fp32_dest_acc_en;
      dst_full_sync_en = kernel.compute_config.dst_full_sync_en;
      math_approx_mode = kernel.compute_config.math_approx_mode;
      bfp8_pack_precise = kernel.compute_config.bfp8_pack_precise;
      for (const auto& define : kernel.compute_config.defines) {
        defines.emplace(define.name, define.value);
      }
      for (const auto& arg : kernel.compute_config.named_compile_args) {
        named_compile_args.emplace(arg.name, arg.value);
      }
      for (const auto& mode : kernel.compute_config.unpack_to_dest_mode) {
        unpack_to_dest_mode.push_back(ParseUnpackToDestMode(mode));
      }
    }
    return CreateKernel(
        program,
        kernel_path,
        core_spec,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compile_time_args,
            .defines = defines,
            .named_compile_args = named_compile_args});
  }

  DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
  NOC noc = NOC::RISCV_0_default;
  if (kernel.has_launch_spec) {
    if (!kernel.launch_spec.processor.empty()) {
      processor = ParseDataMovementProcessor(kernel.launch_spec.processor);
    } else if (core_type == "ncrisc") {
      processor = DataMovementProcessor::RISCV_1;
    }
    if (!kernel.launch_spec.noc.empty()) {
      noc = ParseNoc(kernel.launch_spec.noc);
    } else if (core_type == "ncrisc") {
      noc = NOC::RISCV_1_default;
    }
  } else if (core_type == "ncrisc") {
    processor = DataMovementProcessor::RISCV_1;
    noc = NOC::RISCV_1_default;
  }

  return CreateKernel(
      program,
      kernel_path,
      core_spec,
      DataMovementConfig{
          .processor = processor,
          .noc = noc,
          .compile_args = compile_time_args});
}

static std::vector<uint32_t> BuildKernelCompileTimeArgs(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  ValidateKernelDirectRuntimeSchema(kernel);

  if (!kernel.compile_time_arg_specs.empty()) {
    return BuildKernelCompileTimeArgsFromSchema(kernel, buffer_bindings);
  }

  std::vector<uint32_t> compile_time_args = kernel.compile_time_args;
  if (kernel.accessors.empty()) {
    return compile_time_args;
  }

  std::vector<AccessorSpec> accessors = kernel.accessors;
  std::sort(accessors.begin(), accessors.end(),
            [](const AccessorSpec& a, const AccessorSpec& b) {
              return a.compile_time_arg_offset < b.compile_time_arg_offset;
            });

  uint32_t expected_slot = static_cast<uint32_t>(compile_time_args.size());
  for (const auto& accessor : accessors) {
    ValidateDirectRuntimeAccessorSpec(accessor.buffer, accessor.layout, accessor.memory_space,
                                      accessor.common_runtime_arg_count,
                                      accessor.args_config_bits);
    ICHECK_EQ(accessor.compile_time_arg_count, 2U)
        << "Blackhole direct runtime currently supports only interleaved accessors with two compile-time args";
    ICHECK_EQ(accessor.compile_time_arg_offset, expected_slot)
        << "Accessor compile-time offset mismatch for buffer " << accessor.buffer
        << ": got " << accessor.compile_time_arg_offset << ", expected " << expected_slot;
    AppendInterleavedAccessorCompileTimeArgs(accessor.buffer, accessor.compile_time_arg_count,
                                             accessor.args_config_bits, buffer_bindings,
                                             &compile_time_args);
    expected_slot += accessor.compile_time_arg_count;
  }
  return compile_time_args;
}

static const RuntimeBufferBinding& ResolveRuntimeBufferBinding(
    const KernelArgSpec& arg_spec,
    bool expect_output,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings) {
  ICHECK(!arg_spec.buffer.empty())
      << "Direct runtime requires explicit buffer binding for arg kind " << arg_spec.kind;
  auto it = buffer_bindings.find(arg_spec.buffer);
  ICHECK(it != buffer_bindings.end())
      << "Missing runtime buffer binding for " << arg_spec.buffer;
  ICHECK(it->second.is_output == expect_output)
      << "Runtime buffer role mismatch for " << arg_spec.buffer
      << ": expected output=" << expect_output;
  return it->second;
}

static uint32_t ResolveRuntimeSemaphoreId(
    const KernelSpec& kernel,
    const KernelArgSpec& arg_spec,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  const auto* binding = ResolveSemaphoreBindingSpec(kernel, arg_spec);
  auto semaphore_it = semaphore_ids.find(binding->semaphore_id);
  ICHECK(semaphore_it != semaphore_ids.end())
      << "Blackhole kernel semaphore binding " << binding->name
      << " references missing planned semaphore id " << binding->semaphore_id;
  return semaphore_it->second;
}

static CoreCoord ResolveLogicalCoreNocCoord(const KernelArgSpec& arg_spec,
                                            const KernelSpec& kernel,
                                            const IDevice& device) {
  const auto* descriptor = ResolveRemoteCoreDescriptorSpec(kernel, arg_spec);
  return device.worker_core_from_logical_core(
      CoreCoord{descriptor->core_x, descriptor->core_y});
}

static SynchronizationRuntimeContext BuildSynchronizationRuntimeContext(
    const IDevice& device, const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  return SynchronizationRuntimeContext{
      .device = &device,
      .semaphore_ids = &semaphore_ids,
  };
}

static SynchronizationRuntimeContext BuildCommonSynchronizationRuntimeContext(
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  return SynchronizationRuntimeContext{
      .device = nullptr,
      .semaphore_ids = &semaphore_ids,
  };
}

static bool TryAppendSynchronizationRuntimeArg(const KernelSpec& kernel,
                                               const KernelArgSpec& arg_spec,
                                               const SynchronizationRuntimeContext& sync_context,
                                               std::vector<uint32_t>* args) {
  if (arg_spec.kind == "semaphore_id_u32") {
    ICHECK(sync_context.semaphore_ids != nullptr)
        << "Blackhole synchronization runtime context is missing semaphore ids";
    args->push_back(ResolveRuntimeSemaphoreId(kernel, arg_spec, *sync_context.semaphore_ids));
    return true;
  }
  if (arg_spec.kind == "logical_core_noc_x") {
    ICHECK(sync_context.device != nullptr)
        << "Blackhole synchronization runtime context is missing device";
    const CoreCoord noc_core = ResolveLogicalCoreNocCoord(arg_spec, kernel, *sync_context.device);
    args->push_back(static_cast<uint32_t>(noc_core.x));
    return true;
  }
  if (arg_spec.kind == "logical_core_noc_y") {
    ICHECK(sync_context.device != nullptr)
        << "Blackhole synchronization runtime context is missing device";
    const CoreCoord noc_core = ResolveLogicalCoreNocCoord(arg_spec, kernel, *sync_context.device);
    args->push_back(static_cast<uint32_t>(noc_core.y));
    return true;
  }
  return false;
}

static void AppendRuntimeBufferAddressArg(
    const KernelArgSpec& arg_spec,
    bool expect_output,
    bool use_32bit_addr,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* args) {
  const auto& binding =
      ResolveRuntimeBufferBinding(arg_spec, expect_output, buffer_bindings);
  const uint64_t addr = binding.mesh_buffer->address();
  args->push_back(static_cast<uint32_t>(addr & 0xFFFFFFFF));
  if (!use_32bit_addr) {
    args->push_back(static_cast<uint32_t>(addr >> 32));
  }
}

static bool TryAppendSharedRuntimeArg(
    const KernelArgSpec& arg_spec,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    std::vector<uint32_t>* args) {
  if (arg_spec.kind == "input_buffer_addr") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/false, /*use_32bit_addr=*/false,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "input_buffer_addr32") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/false, /*use_32bit_addr=*/true,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "output_buffer_addr") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/true, /*use_32bit_addr=*/false,
                                  buffer_bindings, args);
    return true;
  }
  if (arg_spec.kind == "output_buffer_addr32") {
    AppendRuntimeBufferAddressArg(arg_spec, /*expect_output=*/true, /*use_32bit_addr=*/true,
                                  buffer_bindings, args);
    return true;
  }
  return false;
}

struct DirectRuntimeWorkContext {
  uint32_t num_k_tiles = 0;
  uint32_t logical_grid_x = 0;
  uint32_t logical_n_tiles = 0;
  uint32_t work_linear_id = 0;
  uint32_t bx = 0;
  uint32_t by = 0;
  bool has_gemm_compute_op = false;
};

static DirectRuntimeWorkContext BuildDirectRuntimeWorkContext(const KernelSpec& kernel,
                                                             const ExecutableSpec& spec,
                                                             uint32_t current_work_linear_id) {
  (void)kernel;
  DirectRuntimeWorkContext context;
  context.logical_grid_x = GetRuntimeLogicalGridX(spec);
  context.work_linear_id = current_work_linear_id;
  context.bx = context.logical_grid_x == 0 ? 0 : (context.work_linear_id % context.logical_grid_x);
  context.by = context.logical_grid_x == 0 ? 0 : (context.work_linear_id / context.logical_grid_x);
  const auto gemm_op = GetPrimaryGemmCompute(spec);
  context.has_gemm_compute_op = gemm_op.enabled && gemm_op.kind == "gemm";
  if (context.has_gemm_compute_op) {
    context.num_k_tiles = GetRuntimeNumKTiles(spec);
    context.logical_n_tiles = GetRuntimeLogicalNTiles(spec);
  }
  return context;
}

static const PerWorkArgSpec* FindPerWorkArgSpec(const std::vector<PerWorkArgSpec>& per_work_arg_specs,
                                                const KernelArgSpec& arg_spec) {
  ICHECK(!arg_spec.identity.empty())
      << "Blackhole direct runtime per-work binding requires runtime arg identity for "
      << arg_spec.name << " kind=" << arg_spec.kind;
  auto it = std::find_if(per_work_arg_specs.begin(), per_work_arg_specs.end(),
                         [&](const PerWorkArgSpec& spec) {
                           return spec.arg_identity == arg_spec.identity;
                         });
  return it == per_work_arg_specs.end() ? nullptr : &(*it);
}

static uint32_t EvaluatePerWorkArgSpec(const PerWorkArgSpec& spec,
                                       const DirectRuntimeWorkContext& context) {
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceWorkLinearId) {
    return context.work_linear_id;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockX) {
    return context.bx;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceLogicalBlockY) {
    return context.by;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceComputeNumKTiles) {
    ICHECK(context.has_gemm_compute_op)
        << "Blackhole direct runtime per_work_arg_spec requires GEMM compute_op for "
        << spec.arg_identity;
    return context.num_k_tiles;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceComputeLogicalNTiles) {
    ICHECK(context.has_gemm_compute_op)
        << "Blackhole direct runtime per_work_arg_spec requires GEMM compute_op for "
        << spec.arg_identity;
    return context.logical_n_tiles;
  }
  if (spec.value_source == tl::blackhole_runtime_arg_schema::kValueSourceConstant) {
    return spec.constant_value;
  }
  LOG(FATAL) << "Unsupported Blackhole per_work_arg_spec value_source " << spec.value_source
             << " for arg " << spec.arg_identity;
  return 0;
}

static bool TryAppendPerWorkRuntimeArg(const KernelSpec& kernel,
                                       const KernelArgSpec& arg_spec,
                                       const std::vector<PerWorkArgSpec>& per_work_arg_specs,
                                       const DirectRuntimeWorkContext& context,
                                       size_t* scalar_index,
                                       const std::vector<uint32_t>& scalar_args,
                                       std::vector<uint32_t>* args) {
  (void)kernel;

  if (arg_spec.kind == "work_linear_id" || arg_spec.kind == "current_work_linear_id") {
    args->push_back(context.work_linear_id);
    return true;
  }
  if (const PerWorkArgSpec* spec = FindPerWorkArgSpec(per_work_arg_specs, arg_spec)) {
    args->push_back(EvaluatePerWorkArgSpec(*spec, context));
    return true;
  }
  if (arg_spec.kind == "scalar_u32") {
    ICHECK(*scalar_index < scalar_args.size())
        << "Spec requested more scalar args than provided";
    args->push_back(scalar_args[(*scalar_index)++]);
    return true;
  }
  return false;
}

static std::vector<uint32_t> BuildCommonRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids) {
  std::vector<uint32_t> args;
  const auto sync_context = BuildCommonSynchronizationRuntimeContext(semaphore_ids);
  for (const auto& arg_spec : kernel.common_runtime_args) {
    if (TryAppendSynchronizationRuntimeArg(kernel, arg_spec, sync_context, &args)) {
      continue;
    }
    if (!TryAppendSharedRuntimeArg(arg_spec, buffer_bindings, &args)) {
      LOG(FATAL) << "Unsupported common runtime arg kind: " << arg_spec.kind;
    }
  }
  return args;
}

static std::vector<uint32_t> BuildRuntimeArgsFromSpec(
    const KernelSpec& kernel,
    const ExecutableSpec& spec,
    uint32_t current_work_linear_id,
    const IDevice& device,
    const std::unordered_map<std::string, RuntimeBufferBinding>& buffer_bindings,
    const std::unordered_map<uint32_t, uint32_t>& semaphore_ids,
    const std::vector<uint32_t>& scalar_args) {
  std::vector<uint32_t> args;
  size_t scalar_index = 0;
  const DirectRuntimeWorkContext context =
      BuildDirectRuntimeWorkContext(kernel, spec, current_work_linear_id);
  const auto sync_context = BuildSynchronizationRuntimeContext(device, semaphore_ids);

  for (const auto& arg_spec : kernel.runtime_args) {
    if (TryAppendSynchronizationRuntimeArg(kernel, arg_spec, sync_context, &args)) {
      continue;
    }
    if (TryAppendSharedRuntimeArg(arg_spec, buffer_bindings, &args)) {
      continue;
    }
    if (!TryAppendPerWorkRuntimeArg(kernel, arg_spec, kernel.per_work_arg_specs, context,
                                    &scalar_index, scalar_args,
                                    &args)) {
      LOG(FATAL) << "Unsupported runtime arg kind: " << arg_spec.kind;
    }
  }

  return args;
}

#endif  // TILELANG_BLACKHOLE_DIRECT

// ============================================================================
// BlackholeModuleNode implementation
// ============================================================================

static void ValidateExecutableSpecCorePlan(const std::string& func_name,
                                           const ExecutableSpec& spec) {
  const auto& core_plan = spec.core_plan;
  ICHECK(!core_plan.work_packets.empty())
      << "Blackhole planner/runtime contract requires non-empty core_plan.work_packets for "
      << func_name;
  uint64_t total_work_items = 0;
  for (const auto& packet : core_plan.work_packets) {
    ICHECK_GT(packet.work_count, 0U)
        << "Blackhole planner/runtime contract requires positive work_count in core_plan."
           "work_packets for "
        << func_name;
    total_work_items += packet.work_count;
  }
  ICHECK_GT(total_work_items, 0U)
      << "Blackhole planner/runtime contract requires at least one logical work item for "
      << func_name;
}

static bool HasPositiveShape(const std::vector<int64_t>& shape) {
  return !shape.empty() &&
         std::all_of(shape.begin(), shape.end(), [](int64_t value) { return value > 0; });
}

static bool HasShardedSourceBinding(const BufferDistributionSpec& plan) {
  return !plan.source_buffer.empty() ||
         (!plan.source_region_kind.empty() && plan.source_region_kind != "none") ||
         !plan.source_region_shape.empty();
}

static std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static void ValidateExecutableSpecBufferDistributionPlans(const std::string& func_name,
                                                          const ExecutableSpec& spec) {
  std::unordered_set<std::string> buffers;
  for (const auto& plan : spec.buffer_distribution_plans) {
    ICHECK(!plan.name.empty())
        << "Blackhole executable buffer distribution requires name for " << func_name;
    ICHECK(!plan.buffer.empty())
        << "Blackhole executable buffer distribution requires buffer for " << func_name;
    ICHECK(buffers.insert(plan.buffer).second)
        << "Blackhole executable buffer distribution has duplicate buffer "
        << plan.buffer << " for " << func_name;
    ICHECK(!plan.mesh_plan.empty())
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires mesh_plan";
    ICHECK_GE(plan.mesh_plan_index, 0)
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires mesh_plan_index";
    ICHECK(!plan.distribution_kind.empty())
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires distribution_kind";
    ICHECK(plan.distribution_kind == "interleaved" || plan.distribution_kind == "sharded" ||
           plan.distribution_kind == "replicated")
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " has unsupported distribution_kind " << plan.distribution_kind;
    ICHECK(!plan.layout.empty())
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires layout";
    ICHECK(!plan.memory_space.empty())
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires memory_space";
    ICHECK_GT(plan.page_size_bytes, 0U)
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires page_size_bytes";
    ICHECK(!plan.host_visibility.empty())
        << "Blackhole executable buffer distribution for " << plan.buffer
        << " requires host_visibility";

    const std::string memory_space = LowerAscii(plan.memory_space);
    if (plan.distribution_kind == "interleaved") {
      ICHECK_EQ(plan.layout, "interleaved")
          << "Blackhole executable interleaved buffer distribution for "
          << plan.buffer << " requires interleaved layout";
      ICHECK_EQ(memory_space, "dram")
          << "Blackhole executable interleaved buffer distribution for "
          << plan.buffer << " requires DRAM memory_space";
      ICHECK(plan.shard_shape.empty() && plan.shard_grid_shape.empty())
          << "Blackhole executable interleaved buffer distribution for "
          << plan.buffer << " cannot carry shard shape";
      ICHECK_EQ(plan.logical_index_mapping, "interleaved_page_index")
          << "Blackhole executable interleaved buffer distribution for "
          << plan.buffer << " requires interleaved_page_index";
    } else if (plan.distribution_kind == "sharded") {
      ICHECK_EQ(memory_space, "l1")
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires L1 memory_space";
      ICHECK(plan.sharding_strategy == "height" ||
             plan.sharding_strategy == "width" ||
             plan.sharding_strategy == "block")
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer
          << " requires sharding_strategy height, width, or block";
      ICHECK(plan.shard_orientation == "row_major" ||
             plan.shard_orientation == "col_major")
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer
          << " requires shard_orientation row_major or col_major";
      ICHECK(HasPositiveShape(plan.shard_shape))
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires shard_shape";
      ICHECK(HasPositiveShape(plan.shard_grid_shape))
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires shard_grid_shape";
      const bool has_source_binding = HasShardedSourceBinding(plan);
      if (has_source_binding) {
        ICHECK(!plan.source_buffer.empty())
            << "Blackhole executable sharded buffer distribution for "
            << plan.buffer << " source binding requires source_buffer";
        ICHECK_EQ(plan.source_region_kind, "per_work_tile")
            << "Blackhole executable sharded buffer distribution for "
            << plan.buffer << " source binding requires per_work_tile source_region_kind";
        ICHECK(HasPositiveShape(plan.source_region_shape))
            << "Blackhole executable sharded buffer distribution for "
            << plan.buffer << " source binding requires source_region_shape";
      } else {
        ICHECK(plan.source_region_kind.empty() || plan.source_region_kind == "none")
            << "Blackhole executable pure-local sharded buffer distribution for "
            << plan.buffer << " cannot carry source_region_kind without source_buffer";
        ICHECK(plan.source_region_shape.empty())
            << "Blackhole executable pure-local sharded buffer distribution for "
            << plan.buffer << " cannot carry source_region_shape without source_buffer";
      }
      ICHECK_EQ(plan.logical_index_mapping, "work_packet_row_major")
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires work_packet_row_major";
      ICHECK_EQ(plan.core_local_address_mapping, "l1_shard_linear")
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires l1_shard_linear";
      ICHECK(!plan.attached_core_group.empty())
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires attached_core_group";
      ICHECK_GE(plan.attached_core_group_index, 0)
          << "Blackhole executable sharded buffer distribution for "
          << plan.buffer << " requires attached_core_group_index";
    }
  }
}

static bool IsShardedTensorMemoryLayout(const std::string& layout) {
  return layout == "HEIGHT_SHARDED" || layout == "WIDTH_SHARDED" ||
         layout == "BLOCK_SHARDED" || layout == "ND_SHARDED";
}

static void ValidateExecutableSpecPlacementRecords(const std::string& func_name,
                                                   const ExecutableSpec& spec) {
  std::unordered_map<std::string, const TensorMemoryConfigSpec*> memory_by_name;
  std::unordered_map<std::string, const TensorMemoryConfigSpec*> memory_by_subject;
  for (const auto& plan : spec.tensor_memory_config_plans) {
    ICHECK(!plan.name.empty())
        << "Blackhole executable tensor memory config requires name for " << func_name;
    ICHECK(!plan.subject.empty())
        << "Blackhole executable tensor memory config requires subject for " << func_name;
    ICHECK(!plan.memory_layout.empty())
        << "Blackhole executable tensor memory config " << plan.name
        << " requires memory_layout";
    ICHECK(!plan.buffer_type.empty())
        << "Blackhole executable tensor memory config " << plan.name
        << " requires buffer_type";
    ICHECK(!plan.origin.empty())
        << "Blackhole executable tensor memory config " << plan.name << " requires origin";
    ICHECK(memory_by_name.emplace(plan.name, &plan).second)
        << "Blackhole executable has duplicate tensor memory config name " << plan.name
        << " for " << func_name;
    ICHECK(memory_by_subject.emplace(plan.subject, &plan).second)
        << "Blackhole executable has duplicate tensor memory config subject "
        << plan.subject << " for " << func_name;
    if (IsShardedTensorMemoryLayout(plan.memory_layout)) {
      ICHECK(!plan.grid_ref.empty())
          << "Blackhole executable sharded tensor memory config " << plan.name
          << " requires grid_ref";
      ICHECK(HasPositiveShape(plan.shard_grid_shape))
          << "Blackhole executable sharded tensor memory config " << plan.name
          << " requires shard_grid_shape";
      ICHECK(HasPositiveShape(plan.shard_shape))
          << "Blackhole executable sharded tensor memory config " << plan.name
          << " requires shard_shape";
      ICHECK(plan.shard_orientation == "row_major" ||
             plan.shard_orientation == "col_major")
          << "Blackhole executable sharded tensor memory config " << plan.name
          << " requires row_major or col_major shard_orientation";
      ICHECK(plan.shard_distribution_strategy == "height" ||
             plan.shard_distribution_strategy == "width" ||
             plan.shard_distribution_strategy == "block" ||
             plan.shard_distribution_strategy == "nd")
          << "Blackhole executable sharded tensor memory config " << plan.name
          << " requires sharding strategy";
    }
  }

  for (const auto& plan : spec.reshard_plans) {
    ICHECK(!plan.name.empty())
        << "Blackhole executable reshard plan requires name for " << func_name;
    ICHECK(!plan.source_value.empty())
        << "Blackhole executable reshard plan " << plan.name << " requires source_value";
    ICHECK(!plan.target_value.empty())
        << "Blackhole executable reshard plan " << plan.name << " requires target_value";
    ICHECK_NE(plan.source_value, plan.target_value)
        << "Blackhole executable reshard plan " << plan.name
        << " source_value and target_value must differ";
    ICHECK_GE(plan.source_memory_config_plan_index, 0)
        << "Blackhole executable reshard plan " << plan.name
        << " requires source_memory_config_plan_index";
    ICHECK_GE(plan.target_memory_config_plan_index, 0)
        << "Blackhole executable reshard plan " << plan.name
        << " requires target_memory_config_plan_index";
    ICHECK_LT(plan.source_memory_config_plan_index,
              static_cast<int64_t>(spec.tensor_memory_config_plans.size()))
        << "Blackhole executable reshard plan " << plan.name
        << " source_memory_config_plan_index out of bounds";
    ICHECK_LT(plan.target_memory_config_plan_index,
              static_cast<int64_t>(spec.tensor_memory_config_plans.size()))
        << "Blackhole executable reshard plan " << plan.name
        << " target_memory_config_plan_index out of bounds";
    const auto* source_config =
        spec.tensor_memory_config_plans.empty()
            ? nullptr
            : &spec.tensor_memory_config_plans[static_cast<size_t>(
                  plan.source_memory_config_plan_index)];
    const auto* target_config =
        spec.tensor_memory_config_plans.empty()
            ? nullptr
            : &spec.tensor_memory_config_plans[static_cast<size_t>(
                  plan.target_memory_config_plan_index)];
    ICHECK(source_config != nullptr && target_config != nullptr)
        << "Blackhole executable reshard plan " << plan.name
        << " requires tensor memory config records";
    ICHECK_EQ(plan.source_memory_config_plan, source_config->name)
        << "Blackhole executable reshard plan " << plan.name
        << " source_memory_config_plan must match indexed config";
    ICHECK_EQ(plan.target_memory_config_plan, target_config->name)
        << "Blackhole executable reshard plan " << plan.name
        << " target_memory_config_plan must match indexed config";
    ICHECK_EQ(plan.source_value, source_config->subject)
        << "Blackhole executable reshard plan " << plan.name
        << " source_value must match source config subject";
    ICHECK_EQ(plan.target_value, target_config->subject)
        << "Blackhole executable reshard plan " << plan.name
        << " target_value must match target config subject";
    ICHECK(plan.conversion_kind == "interleaved_to_sharded" ||
           plan.conversion_kind == "sharded_to_interleaved" ||
           plan.conversion_kind == "reshard" || plan.conversion_kind == "unsupported")
        << "Blackhole executable reshard plan " << plan.name
        << " has unsupported conversion_kind " << plan.conversion_kind;
    ICHECK(plan.admission_status == "admitted" ||
           plan.admission_status == "unsupported")
        << "Blackhole executable reshard plan " << plan.name
        << " has unsupported admission_status " << plan.admission_status;
    if (plan.admission_status == "admitted") {
      ICHECK(plan.unsupported_reason.empty())
          << "Blackhole executable reshard plan " << plan.name
          << " admitted conversion cannot carry unsupported_reason";
    } else {
      ICHECK(!plan.unsupported_reason.empty())
          << "Blackhole executable reshard plan " << plan.name
          << " unsupported conversion requires unsupported_reason";
    }
  }
}

BlackholeModuleNode::BlackholeModuleNode(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir)
    : fmap_(std::move(fmap)),
      kernel_dir_(std::move(kernel_dir)) {
  for (const auto& entry : fmap_) {
    ValidateExecutableSpecCorePlan(entry.first, entry.second);
    ValidateExecutableSpecBufferDistributionPlans(entry.first, entry.second);
    ValidateExecutableSpecPlacementRecords(entry.first, entry.second);
    ValidateExecutableSpecSynchronizationSchema(entry.first, entry.second);
  }
}


ffi::Optional<ffi::Function> BlackholeModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);

  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return ffi::Function();
  }

  const ExecutableSpec& info = it->second;
  BlackholeWrappedFunc f;
  f.Init(this, sptr_to_self, name, info);

  std::vector<FunctionInfo::ArgExtraTags> arg_extra_tags;
  return PackFuncVoidAddr(f, info.tvm_arg_types, arg_extra_tags);
}

ffi::Optional<ffi::String> BlackholeModuleNode::GetFunctionMetadata(const ffi::String& name) {
  auto it = fmap_.find(name);
  if (it == fmap_.end()) {
    return std::nullopt;
  }
  return ffi::String(EncodeExecutableSpecMetadata(it->second));
}

void BlackholeModuleNode::WriteToFile(const ffi::String& file_name,
                                      const ffi::String& format) const {
  LOG(FATAL) << "BlackholeModule WriteToFile not implemented";
}

ffi::Bytes BlackholeModuleNode::SaveToBytes() const {
  std::string buffer;
  dmlc::MemoryStringStream ms(&buffer);
  dmlc::Stream* stream = &ms;
  WriteString(stream, kBlackholeModuleSerializationMagic);
  WriteString(stream, kernel_dir_);
  WriteExecutableSpecMap(stream, fmap_);
  return ffi::Bytes(buffer);
}

ffi::String BlackholeModuleNode::InspectSource(const ffi::String& format) const {
  LOG(INFO) << "BlackholeModuleNode::InspectSource called";
  auto it = fmap_.find("default");
  if (it != fmap_.end()) {
    const auto& spec = it->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Found 'default' function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  if (!fmap_.empty()) {
    const auto& spec = fmap_.begin()->second;
    const std::string source = spec.kernels.empty() ? std::string() : spec.kernels.front().source_code;
    LOG(INFO) << "Using first function, kernel_code size: " << source.size();
    return ffi::String(source);
  }
  LOG(WARNING) << "No functions found in fmap_";
  return ffi::String("");
}

// ============================================================================
// Unique temp-directory helper (used by both execution paths)
// ============================================================================

static std::string MakeUniqueTempDir(const std::string& prefix) {
  static std::atomic<uint64_t> counter{0};
  const auto id = counter.fetch_add(1, std::memory_order_relaxed);
  std::filesystem::path dir = std::filesystem::temp_directory_path() /
                              (prefix + std::to_string(getpid()) + "_" + std::to_string(id));
  std::filesystem::create_directories(dir);
  return dir.string();
}

// ============================================================================
// Direct TT-Metal execution path
// ============================================================================

void BlackholeModuleNode::ExecuteDirect(
    const std::string& func_name,
    const std::vector<RuntimeTensorBinding>& buffer_args,
    const std::vector<uint32_t>& scalar_args,
    const std::vector<std::string>& output_names) {
#ifdef TILELANG_BLACKHOLE_DIRECT
  using namespace tt::tt_metal;

  auto fit = fmap_.find(func_name);
  if (fit == fmap_.end()) {
    LOG(FATAL) << "Function not found: " << func_name;
  }
  const ExecutableSpec& spec = fit->second;
  if (spec.kernels.empty()) {
    LOG(FATAL) << "ExecutableSpec has no kernels for function: " << func_name;
  }
  if (!spec.direct_runtime_unsupported_reasons.empty()) {
    std::ostringstream os;
    for (size_t i = 0; i < spec.direct_runtime_unsupported_reasons.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << spec.direct_runtime_unsupported_reasons[i];
    }
    LOG(FATAL) << "Blackhole direct runtime is not supported for " << func_name << ": "
               << os.str();
  }
  ValidateGemmComputeDirectRuntimeConstraints(spec);
  for (const auto& kernel_spec : spec.kernels) {
    ValidateKernelDirectRuntimeConstraints(kernel_spec);
  }

  const std::vector<DirectLaunchWave> launch_waves = BuildDirectLaunchWaves(spec, func_name);
  uint64_t total_work_items = 0;
  for (const auto& wave : launch_waves) {
    total_work_items += wave.work_items.size();
  }

  // Keep direct execution hermetic per call. Reusing a persistent MeshDevice across
  // multiple Python direct-call tests can leave simulator/device state behind and
  // cause cross-test contamination across cases.
  LOG(INFO) << "Initializing Blackhole TT-Metal device...";
  std::shared_ptr<distributed::MeshDevice> mesh_device;
  try {
    mesh_device = distributed::MeshDevice::create_unit_mesh(0);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Blackhole device: " << e.what();
  }
  ICHECK(mesh_device != nullptr);
  LOG(INFO) << "Blackhole device initialized successfully";

  distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

  DirectRuntimeBufferState runtime_buffer_state =
      MaterializeRuntimeBuffers(cq, mesh_device.get(), spec, buffer_args, output_names);

  // Write kernel source files to temp directory
  std::string tmp_dir = MakeUniqueTempDir("tilelang_bh_direct_");
  std::vector<std::string> kernel_paths = WriteKernelSourceFiles(spec, func_name, tmp_dir);

  LOG(INFO) << "Direct path: executing " << total_work_items << " logical work items across "
            << launch_waves.size() << " launch wave(s) for " << func_name;

  for (size_t wave_index = 0; wave_index < launch_waves.size(); ++wave_index) {
    const auto& launch_wave = launch_waves[wave_index];
    const CoreRangeSet launch_core_ranges(launch_wave.launch_cores);
    LOG(INFO) << "Direct path: launch wave " << (wave_index + 1) << "/" << launch_waves.size()
              << " with " << launch_wave.work_items.size() << " logical work items across "
              << launch_wave.launch_cores.size() << " launch cores for " << func_name;

    Program program = CreateProgram();
    const std::unordered_map<uint32_t, uint32_t> semaphore_ids =
        CreateSemaphoresFromSpec(program, spec);

    CreateCircularBuffersFromSpec(program, launch_core_ranges, spec);

    std::vector<KernelHandle> kernels = CreateProgramKernelsFromSpec(
        program, launch_core_ranges, spec, runtime_buffer_state.runtime_buffers, semaphore_ids,
        kernel_paths);

    ApplyWorkItemRuntimeArgs(program, spec, kernels, launch_wave.work_items, *mesh_device,
                             runtime_buffer_state.runtime_buffers, semaphore_ids, scalar_args);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    LOG(INFO) << "Direct path: enqueue multi-core workload for " << func_name << " wave "
              << (wave_index + 1) << "/" << launch_waves.size();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
  }

  // Read back results
  for (const auto& binding : buffer_args) {
    if (!binding.is_output) {
      continue;
    }
    auto it = runtime_buffer_state.runtime_buffers.find(binding.name);
    ICHECK(it != runtime_buffer_state.runtime_buffers.end())
        << "Missing runtime output binding for " << binding.name;
    std::vector<uint8_t> output_data;
    distributed::EnqueueReadMeshBuffer(cq, output_data, it->second.mesh_buffer,
                                       /*blocking=*/true);
    CopyOutputFromDeviceBuffer(spec, binding, output_data);
  }

  // Cleanup kernel temp files
  std::filesystem::remove_all(tmp_dir);

  LOG(INFO) << "Direct path execution completed for " << func_name;

#else
  LOG(FATAL) << "Direct TT-Metal path not available. "
             << "Rebuild with TILELANG_BLACKHOLE_DIRECT=ON.";
#endif  // TILELANG_BLACKHOLE_DIRECT
}

// ============================================================================
// Execution dispatch
// ============================================================================

void BlackholeWrappedFunc::operator()(ffi::PackedArgs args, ffi::Any* rv,
                                       void** void_args) const {
  // Direct runtime requires explicit schema-derived name->role bindings.
  std::unordered_map<std::string, bool> buffer_is_output_by_name;
  auto append_buffer_contract = [&](const std::vector<KernelArgSpec>& runtime_args) {
    for (const auto& arg : runtime_args) {
      const bool is_input = arg.kind == "input_buffer_addr32" || arg.kind == "input_buffer_addr";
      const bool is_output = arg.kind == "output_buffer_addr32" || arg.kind == "output_buffer_addr";
      if (!is_input && !is_output) {
        continue;
      }
      ICHECK(!arg.buffer.empty())
          << "Blackhole direct runtime requires explicit buffer role schema for arg "
          << arg.name << " kind=" << arg.kind;
      auto [it, inserted] = buffer_is_output_by_name.emplace(arg.buffer, is_output);
      ICHECK(inserted || it->second == is_output)
          << "Blackhole direct runtime buffer role mismatch for " << arg.buffer;
      if (is_input) {
        ICHECK(!is_output);
      }
    }
  };
  append_buffer_contract(info_.runtime_args);
  append_buffer_contract(info_.common_runtime_args);
  ICHECK(!buffer_is_output_by_name.empty())
      << "Blackhole direct runtime requires explicit buffer role schema";

  // Collect arguments
  std::vector<RuntimeTensorBinding> buffer_args;
  std::vector<uint32_t> scalars;
  std::vector<std::string> output_names;

  for (size_t i = 0; i < info_.tvm_arg_types.size(); ++i) {
    if (info_.tvm_is_buffer_arg[i]) {
      DLTensor* tensor = ExtractTensorArg(args[i], void_args != nullptr ? void_args[i] : nullptr);
      ICHECK_LT(i, info_.tvm_arg_names.size())
          << "Blackhole direct runtime requires formal buffer identity for arg index " << i;
      const std::string buffer_name = info_.tvm_arg_names[i];
      ICHECK(!buffer_name.empty())
          << "Blackhole direct runtime requires formal buffer identity for arg index " << i;
      auto role_it = buffer_is_output_by_name.find(buffer_name);
      ICHECK(role_it != buffer_is_output_by_name.end())
          << "Blackhole direct runtime requires explicit buffer role binding for "
          << buffer_name;
      bool is_out = role_it->second;
      buffer_args.push_back(RuntimeTensorBinding{buffer_name, tensor, is_out});
      if (is_out) {
        output_names.push_back(buffer_name);
      }
    } else {
      ffi::AnyView arg = args[i];
      uint32_t val = ExtractScalar(arg, info_.tvm_arg_types[i]);
      scalars.push_back(val);
    }
  }

  m_->ExecuteDirect(func_name_, buffer_args, scalars, output_names);
}

// ============================================================================
// Module creation and registration
// ============================================================================

ffi::Module BlackholeModuleCreate(
    std::unordered_map<std::string, ExecutableSpec> fmap,
    std::string kernel_dir) {
  auto n = ffi::make_object<BlackholeModuleNode>(std::move(fmap), std::move(kernel_dir));
  return ffi::Module(std::move(n));
}

ffi::Module BlackholeModuleLoadFromBytes(const ffi::Bytes& bytes) {
  dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
  dmlc::Stream* stream = &ms;
  const std::string magic = ReadString(stream, "module.magic");
  ICHECK_EQ(magic, kBlackholeModuleSerializationMagic)
      << "BlackholeModule LoadFromBytes magic mismatch";
  std::string kernel_dir = ReadString(stream, "module.kernel_dir");
  auto fmap = ReadExecutableSpecMap(stream);
  return BlackholeModuleCreate(std::move(fmap), std::move(kernel_dir));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.module.loadbinary_blackhole", BlackholeModuleLoadFromBytes)
      .def("ffi.Module.load_from_bytes.blackhole", BlackholeModuleLoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm
