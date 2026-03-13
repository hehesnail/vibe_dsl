# TileLang Quantization 模块详解

## 模块概述

Quantization 是 TileLang 的量化支持模块，位于 `/root/dev/vibe_dsl/tilelang/tilelang/quantize/` 目录。该模块提供了低精度数据类型的转换和量化/反量化功能，支持 INT4/INT8/FP4/FP8 等格式。

目录结构：
```
tilelang/quantize/
├── __init__.py          # 模块导出
├── quantization.py      # TIR 量化原语
├── lop3.py             # LOP3 指令解码
├── mxfp.py             # MXFP 格式支持
└── utils.py            # 量化工具函数
```

核心功能：
- INT4/INT8 量化和反量化
- FP4/FP8 (E4M3/E5M2) 格式转换
- LOP3 指令加速解码
- MXFP 微缩放浮点格式

## 核心函数详解

### 1. TIR 量化原语

#### _tir_u8_to_f4_to_bf16

```python
def _tir_u8_to_f4_to_bf16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, scale: tir.PrimExpr, dtype: str):
    """Convert a packed 4-bit field stored in a uint8 into a bfloat16 value."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:30-73`

将 uint8 中打包的 4-bit 字段转换为 bfloat16：
- 提取 4-bit 字段（符号、指数、尾数）
- 将 2-bit 指数转换为 bf16 指数空间（加偏置 126）
- 应用缩放因子
- 组装 16-bit bfloat16 位模式

#### _tir_f32x2_to_bf16x2_to_u32

```python
def _tir_f32x2_to_bf16x2_to_u32(v0: tir.PrimExpr, v1: tir.PrimExpr, round_to_even: bool = True):
    """Convert two float32 values to bfloat16 and pack them into a single uint32."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:75-100`

将两个 float32 转换为 bfloat16 并打包到 uint32：
- 可选的 round-to-nearest-even 舍入
- v0 在低 16 位，v1 在高 16 位

#### _tir_f32_to_uint_to_f4

```python
def _tir_f32_to_uint_to_f4(val: tir.PrimExpr):
    """Convert float32 to 4-bit floating point format."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:141-155`

FP32 到 FP4 的量化：
- 提取符号位、指数位、尾数位
- 根据指数值确定 FP4 指数
- 支持非规格化数处理

#### _tir_packed_to_fp4_to_f16

```python
def _tir_packed_to_fp4_to_f16(storage_type="uint", storage_nbit=8):
    """Factory function for FP4 to FP16 conversion."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:202-217`

创建 FP4 到 FP16 的转换函数：
- 支持多种存储类型（uint8, int32）
- 处理打包数据布局

#### _tir_u8_to_f8_e4m3_to_f16

```python
def _tir_u8_to_f8_e4m3_to_f16(nbit: int, val: tir.PrimExpr, dtype: str):
    """Convert FP8 E4M3 format to FP16."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:230-237`

FP8 E4M3 到 FP16 的转换：
- E4M3: 4 位指数，3 位尾数
- 使用位操作进行高效转换

#### _tir_u8_to_f8_e5m2_to_f16

```python
def _tir_u8_to_f8_e5m2_to_f16(nbit: int, val: tir.PrimExpr, dtype: str):
    """Convert FP8 E5M2 format to FP16."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:240-244`

FP8 E5M2 到 FP16 的转换，使用 TVM 内置类型转换。

### 2. 打包整数转换

#### _tir_packed_int_to_int_convert

```python
def _tir_packed_int_to_int_convert(storage_type="uint", storage_nbit=8):
    """Factory for packed signed integer to int conversion with sign extension."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:281-291`

带符号扩展的打包整数转换：
```python
# 示例：将打包的 4-bit 有符号整数转换为 int32
mask = tir.const((1 << nbit) - 1, T.int32)
unextended = (val >> (pos * nbit)) & mask
# 符号扩展
result = (unextended << (32 - nbit)) >> (32 - nbit)
```

#### _tir_packed_to_unsigned_convert

```python
def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    """Factory for packed unsigned integer conversion."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py:258-266`

无符号打包整数转换，支持零点偏移：
```python
def _tir_packed_to_unsigned_convert_with_zeros(storage_type="uint", storage_nbit=8):
    def f_convert(nbit: int, val, pos, zero, dtype):
        mask = tir.const((1 << nbit) - 1, storage_dtype)
        return (((val >> (pos * nbit)) & mask) - zero).astype(dtype)
```

### 3. LOP3 解码

#### get_lop3_intrin_group

```python
def get_lop3_intrin_group(
    out_dtype: Literal[T.float16, T.bfloat16] = T.float16,
    storage_dtype: Literal[T.int8, T.uint8] = T.uint8,
    source_format: Literal["uint", "int"] = "uint",
    source_bit: int = 4,
    with_scaling: bool = False,
    dequantize_scale_dtype: Literal[T.float16, T.bfloat16, T.float32] = T.float16,
    quantize_type: Literal["per_channel", "per_block", "per_block_with_offset"] = "per_channel",
):
    """Return metadata for a LOP3-based dequantization intrinsic."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/lop3.py`

LOP3（3 输入逻辑操作）指令加速解码：
- 使用 CUDA LOP3 指令（.b32）进行位操作
- 支持 INT4 到 FP16/BF16 的快速解码
- 支持有符号/无符号格式
- 支持缩放因子应用

**LOP3 解码原理**:
```c
// LOP3 指令执行三输入位操作
lop3.b32 dst, src1, src2, src3, immLut;
// immLut 是查找表，定义逻辑操作
```

### 4. MXFP 微缩放浮点

#### get_mxfp_intrin_group

```python
def get_mxfp_intrin_group(
    out_dtype: Literal[T.float16, T.bfloat16] = T.bfloat16,
    source_format: Literal[T.int, T.uint] = T.uint,
    source_bit: int = 4,
    storage_dtype: Literal[T.int32, T.int8, T.uint8] = T.uint8,
    use_twiddling: bool = False,
) -> dict[str, str]:
    """Return metadata for an MXFP decoding intrinsic."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/mxfp.py:52-105`

MXFP (Microscaling Floating Point) 格式支持：
- 支持 FP4 到 BF16/F16 的转换
- 使用 twiddling 技术进行高效位操作
- 内联 PTX 汇编实现

**Twiddling 解码**:
```c
// 使用位操作技巧而非查找表
prmt.b32 tmp, val, 0, 0x0123;  // 处理字节序
mov.b32 magic, 0x7e807e80;     // 魔术常量
and.b32 masked, tmp, 0b10000001110000001000000111000000;
mul.bf16x2 result, masked, magic;
```

### 5. 量化工具函数

#### gen_quant4

```python
def gen_quant4(k, n, groupsize=-1):
    """Generate 4-bit quantized weights for testing."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py:1-47`

生成 4-bit 量化权重（用于测试）：
- 支持按组量化
- 计算缩放因子
- 返回原始权重、反量化权重、缩放因子和量化值

#### general_compress

```python
def general_compress(lowprecision_weight, source_bits=4, storage_dtype=None):
    """Compress low-precision weights into dense storage."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py:50-67`

将低精度权重压缩到密集存储：
- 支持 1/2/4/8 bit 源精度
- 打包到 int8 存储
- 计算元素打包比例

#### interleave_weight

```python
def interleave_weight(qweight, nbits=4, target_dtype="float16"):
    """Interleave the weight to the target data type layout."""
```

**位置**: `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py:71-125`

权重交错重排（用于 Tensor Core 布局）：
- 支持 1/2/4 bit 量化
- 针对 float16/int8 目标类型优化
- 处理特定的位排列模式

## 实现逻辑分析

### 量化流程

```
┌──────────────────┐
│  原始 FP32/FP16  │
│     权重         │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  计算缩放因子     │
│  s = max(|w|) * 2/maxq │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  量化：w_q = round(w/s) │
│  偏移：w_q += maxq/2    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  打包到 uint8    │
│  (2x4bit 或 8x1bit) │
└──────────────────┘
```

### 反量化流程

```
┌──────────────────┐
│  打包的 uint8    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  解包位字段      │
│  (移位 + 掩码)   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  转换为 FP16/FP32 │
│  (位操作/LOP3)   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  应用缩放因子    │
│  w = (w_q - zero) * s │
└──────────────────┘
```

### LOP3 加速原理

LOP3 指令可以在单个时钟周期内执行任意 3 输入布尔函数：

```python
# 传统方式：多个指令
and_result = a & b
or_result = and_result | c

# LOP3 方式：单个指令
# immLut 编码了真值表
# 对于 AND-OR: immLut = 0b11101000 = 0xE8
lop3.b32 result, a, b, c, 0xE8
```

## 使用示例

### INT4 量化/反量化

```python
from tilelang import language as T
from tilelang.quantize import _tir_u8_to_f4_to_bf16, _tir_packed_to_fp4_to_f16

# 创建反量化函数
fp4_to_f16 = _tir_packed_to_fp4_to_f16(storage_type="uint", storage_nbit=8)

# 在 TileLang kernel 中使用
@T.prim_func
def dequantize_kernel(A: T.Buffer((1024,), "uint8"), B: T.Buffer((2048,), "float16")):
    for i in T.Parallel(2048):
        byte_idx = i // 2
        nibble = i % 2
        packed_val = A[byte_idx]
        B[i] = fp4_to_f16(4, packed_val, nibble, "float16")
```

### LOP3 解码

```python
from tilelang.quantize import get_lop3_intrin_group

# 获取 LOP3 解码元数据
intrin = get_lop3_intrin_group(
    out_dtype=T.float16,
    storage_dtype=T.uint8,
    source_format="uint",
    source_bit=4,
    with_scaling=True,
)

# intrin 包含：
# - func_name: 函数名
# - c_source: CUDA C 源代码
# - storage_dtype: 存储类型
# - n_elements: 每次处理的元素数
```

### 权重压缩

```python
import torch
from tilelang.quantize import general_compress, interleave_weight

# 原始量化权重 (4-bit)
quantized = torch.randint(0, 16, (1024, 512), dtype=torch.int8)

# 压缩到密集存储
compressed = general_compress(quantized, source_bits=4, storage_dtype=torch.int8)
# compressed shape: (1024, 256)

# 交错重排用于 Tensor Core
interleaved = interleave_weight(compressed, nbits=4, target_dtype="float16")
```

### MXFP 解码

```python
from tilelang.quantize import get_mxfp_intrin_group

# 获取 MXFP 解码函数
mxfp_intrin = get_mxfp_intrin_group(
    out_dtype=T.bfloat16,
    source_format=T.uint,
    source_bit=4,
    storage_dtype=T.uint8,
    use_twiddling=True,
)

# 在内核中调用生成的函数
# decode_fp4_to_bf16_twiddling(packed_data, output_buffer)
```

## 关键代码引用

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| _tir_u8_to_f4_to_bf16 | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py` | 30-73 |
| _tir_f32_to_uint_to_f4 | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py` | 141-155 |
| _tir_packed_to_fp4_to_f16 | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py` | 202-217 |
| _tir_u8_to_f8_e4m3_to_f16 | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py` | 230-237 |
| _tir_packed_int_to_int_convert | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/quantization.py` | 281-291 |
| get_lop3_intrin_group | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/lop3.py` | 全文 |
| get_mxfp_intrin_group | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/mxfp.py` | 52-105 |
| gen_quant4 | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py` | 1-47 |
| general_compress | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py` | 50-67 |
| interleave_weight | `/root/dev/vibe_dsl/tilelang/tilelang/quantize/utils.py` | 71-125 |
