# Compute Kernel API 参考手册

> **文档版本**: 1.0
> **最后更新**: 2026-03-12
> **适用范围**: TT-Metalium 计算内核开发

---

## 目录

1. [概述](#1-概述)
2. [Tile 寄存器管理](#2-tile-寄存器管理)
3. [矩阵运算 API](#3-矩阵运算-api)
4. [逐元素二元操作](#4-逐元素二元操作)
5. [逐元素一元操作 (SFPU)](#5-逐元素一元操作-sfpu)
6. [归约操作](#6-归约操作)
7. [数据格式转换](#7-数据格式转换)
8. [打包操作](#8-打包操作)
9. [SFPI 条件执行](#9-sfpi-条件执行)
10. [使用示例](#10-使用示例)

---

## 1. 概述

Compute Kernel 在 Tensix 核心的 TRISC（三个 RISC-V 核心）上运行，负责执行实际的数学运算。TRISC 核心分为：

| 核心 | 功能 | 主要职责 |
|------|------|----------|
| TRISC0 (Unpack) | 数据解包 | 从 CB 读取数据并解包到寄存器 |
| TRISC1 (Math) | 数学运算 | 执行矩阵乘法、逐元素操作等 |
| TRISC2 (Pack) | 数据打包 | 将结果从寄存器打包回 CB |

### 1.1 头文件包含

```cpp
// 主计算 API
#include "compute_kernel_api/compute_kernel_api.h"

// 矩阵乘法
#include "compute_kernel_api/matmul.h"

// 逐元素二元操作
#include "compute_kernel_api/eltwise_binary.h"

// 逐元素一元操作
#include "compute_kernel_api/eltwise_unary/sigmoid.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/log.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"

// 归约操作
#include "compute_kernel_api/reduce.h"

// 打包操作
#include "compute_kernel_api/pack.h"

// 数据格式转换
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
```

### 1.2 常用宏定义

```cpp
#define ALWI inline __attribute__((always_inline))
#define MAIN extern "C" void _ZN7tt_metal7kernels4mainEv()
```

---

## 2. Tile 寄存器管理

Tile 寄存器是计算核心上的临时存储，用于存放中间计算结果。

### 2.1 tile_regs_acquire

**函数签名**:
```cpp
ALWI void tile_regs_acquire();
```

**描述**: 获取 Tile 寄存器的访问权，开始一个计算序列。

**参数**: 无

**返回值**: 无

**使用场景**: 在执行任何计算操作之前调用，标记计算阶段的开始。

**示例**:
```cpp
void MAIN {
    tile_regs_acquire();  // 开始计算阶段
    // ... 执行计算操作 ...
    tile_regs_commit();
}
```

---

### 2.2 tile_regs_commit

**函数签名**:
```cpp
ALWI void tile_regs_commit();
```

**描述**: 提交 Tile 寄存器中的计算结果，表示计算阶段完成。

**参数**: 无

**返回值**: 无

**使用场景**: 在所有计算操作完成后调用，准备将结果打包到输出 CB。

**示例**:
```cpp
void MAIN {
    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);  // 执行加法
    tile_regs_commit();  // 计算完成
}
```

---

### 2.3 tile_regs_wait

**函数签名**:
```cpp
ALWI void tile_regs_wait();
```

**描述**: 等待 Tile 寄存器中的数据准备好，确保计算已完成。

**参数**: 无

**返回值**: 无

**使用场景**: 在打包操作之前调用，确保所有计算指令已执行完毕。

**示例**:
```cpp
void MAIN {
    tile_regs_acquire();
    matmul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();  // 等待计算完成
    pack_tile(0, cb_out);
}
```

---

### 2.4 tile_regs_release

**函数签名**:
```cpp
ALWI void tile_regs_release();
```

**描述**: 释放 Tile 寄存器，使其可用于下一个计算周期。

**参数**: 无

**返回值**: 无

**使用场景**: 在打包完成后调用，释放寄存器资源。

**示例**:
```cpp
void MAIN {
    tile_regs_acquire();
    mul_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();  // 释放寄存器
}
```

---

### 2.5 完整 Tile 寄存器生命周期

```cpp
void MAIN {
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // 1. 获取寄存器
        tile_regs_acquire();

        // 2. 执行计算
        add_tiles(cb_in0, cb_in1, 0, 0, 0);

        // 3. 提交计算
        tile_regs_commit();

        // 4. 等待就绪
        tile_regs_wait();

        // 5. 打包结果
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        // 6. 释放寄存器
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
```

---

## 3. 矩阵运算 API

### 3.1 mm_init

**函数签名**:
```cpp
ALWI void mm_init(
    uint32_t in0_cb_id,      // 输入 A 的 CB ID
    uint32_t in1_cb_id,      // 输入 B 的 CB ID
    uint32_t out_cb_id,      // 输出 CB ID
    const uint32_t transpose = 0,  // 是否转置 (0=否, 1=是)
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 初始化矩阵乘法引擎，配置输入输出 CB。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| out_cb_id | uint32_t | 结果输出的 CB ID |
| transpose | uint32_t | 是否对 in1 进行转置 (0=否, 1=是) |
| call_line | uint32_t | 调用行号（用于调试，自动填充） |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    mm_init(cb_in0, cb_in1, cb_out, 0);  // 标准矩阵乘法

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
```

---

### 3.2 mm_init_short

**函数签名**:
```cpp
ALWI void mm_init_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 短初始化版本，不配置输出 CB，用于连续执行多个 matmul 的场景。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| transpose | uint32_t | 是否对 in1 进行转置 |
| call_line | uint32_t | 调用行号（用于调试） |

**返回值**: 无

**使用场景**: 当输出 CB 已在之前的初始化中配置，或需要动态切换输出 CB 时使用。

**示例**:
```cpp
void MAIN {
    // 完整初始化一次
    mm_init(cb_in0, cb_in1, cb_out, 0);

    // 后续使用短初始化（如果输入 CB 改变但输出不变）
    mm_init_short(cb_in0, cb_in1, 0);
}
```

---

### 3.3 mm_init_short_with_dt

**函数签名**:
```cpp
ALWI void mm_init_short_with_dt(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t c_in_old_srca,   // 旧的数据类型配置
    const uint32_t transpose = 0
);
```

**描述**: 带数据类型配置的短初始化，用于在运行时切换数据格式。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| c_in_old_srca | uint32_t | 源 A 的旧数据类型配置 |
| transpose | uint32_t | 是否对 in1 进行转置 |

**返回值**: 无

**使用场景**: 需要在同一内核中处理不同数据类型的矩阵乘法时使用。

---

### 3.4 mm_block_init

**函数签名**:
```cpp
ALWI void mm_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,     // C Tile 维度 (输出列方向)
    uint32_t rt_dim = 1,     // R Tile 维度 (输出行方向)
    uint32_t kt_dim = 1,     // K Tile 维度 (累加维度)
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 初始化块矩阵乘法引擎，支持更大的 Tile 块操作。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| out_cb_id | uint32_t | 输出 CB ID |
| transpose | uint32_t | 是否对 in1 进行转置 |
| ct_dim | uint32_t | C Tile 维度（输出列方向 Tile 数） |
| rt_dim | uint32_t | R Tile 维度（输出行方向 Tile 数） |
| kt_dim | uint32_t | K Tile 维度（累加维度 Tile 数） |
| call_line | uint32_t | 调用行号（用于调试） |

**返回值**: 无

**使用场景**: 用于大规模矩阵乘法，通过块操作提高数据复用率。

**示例**:
```cpp
void MAIN {
    // 初始化 2x2 块矩阵乘法
    mm_block_init(cb_in0, cb_in1, cb_out, 0, 2, 2, 2);

    tile_regs_acquire();
    matmul_block(cb_in0, cb_in1, 0, 0, 0, 0, 2, 2, 2);
    tile_regs_commit();
}
```

---

### 3.5 matmul_tiles

**函数签名**:
```cpp
ALWI void matmul_tiles(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,  // 输入 A 的 Tile 索引
    uint32_t in1_tile_index,  // 输入 B 的 Tile 索引
    uint32_t idst             // 目标寄存器 ID
);
```

**描述**: 执行两个 Tile 的矩阵乘法，结果累加到目标寄存器。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| in0_tile_index | uint32_t | 输入 A 中的 Tile 索引（0 表示当前 CB 前端） |
| in1_tile_index | uint32_t | 输入 B 中的 Tile 索引 |
| idst | uint32_t | 目标寄存器 ID（0-15） |

**返回值**: 无

**注意**: 结果会累加到目标寄存器的现有值。如需清零，请先调用 zero_acc 或首次调用时确保寄存器为空。

**示例**:
```cpp
void MAIN {
    mm_init(cb_in0, cb_in1, cb_out);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();
    // 计算: dst[0] += A[0] @ B[0]
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
```

---

### 3.6 matmul_block

**函数签名**:
```cpp
ALWI void matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 执行块矩阵乘法，处理多个 Tile 组成的矩阵块。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| in0_tile_index | uint32_t | 输入 A 的起始 Tile 索引 |
| in1_tile_index | uint32_t | 输入 B 的起始 Tile 索引 |
| idst | uint32_t | 目标寄存器 ID |
| transpose | uint32_t | 是否对 in1 进行转置 |
| ct_dim | uint32_t | C Tile 维度 |
| rt_dim | uint32_t | R Tile 维度 |
| kt_dim | uint32_t | K Tile 维度 |
| call_line | uint32_t | 调用行号（用于调试） |

**返回值**: 无

**使用场景**: 大规模矩阵乘法的优化实现，通过增加每次处理的数据量来减少开销。

---

### 3.7 matmul_block_math_dynamic_throttle

**函数签名**:
```cpp
ALWI void matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim
);
```

**描述**: 带动态节流的块矩阵乘法（仅 Blackhole 架构支持），根据系统负载动态调整计算速度。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in0_cb_id | uint32_t | 左矩阵 (A) 所在的 CB ID |
| in1_cb_id | uint32_t | 右矩阵 (B) 所在的 CB ID |
| idst | uint32_t | 目标寄存器 ID |
| transpose | uint32_t | 是否对 in1 进行转置 |
| ct_dim | uint32_t | C Tile 维度 |
| rt_dim | uint32_t | R Tile 维度 |

**返回值**: 无

**注意**: 此函数仅在 Blackhole 架构上可用，用于优化功耗和散热。

---

## 4. 逐元素二元操作

### 4.1 binary_op_init_common

**函数签名**:
```cpp
ALWI void binary_op_init_common(
    uint32_t icb0,           // 输入 CB 0 ID
    uint32_t icb1,           // 输入 CB 1 ID
    uint32_t ocb,            // 输出 CB ID
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 通用二元操作初始化，配置输入输出 CB 用于逐元素操作。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb0 | uint32_t | 第一个输入 CB ID |
| icb1 | uint32_t | 第二个输入 CB ID |
| ocb | uint32_t | 输出 CB ID |
| call_line | uint32_t | 调用行号（用于调试） |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    binary_op_init_common(cb_in0, cb_in1, cb_out);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
```

---

### 4.2 add_tiles

**函数签名**:
```cpp
ALWI void add_tiles(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t itile0,         // 输入 0 的 Tile 索引
    uint32_t itile1,         // 输入 1 的 Tile 索引
    uint32_t idst            // 目标寄存器 ID
);
```

**描述**: 逐元素加法: `dst[idst] = src0[itile0] + src1[itile1]`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb0 | uint32_t | 第一个输入 CB ID |
| icb1 | uint32_t | 第二个输入 CB ID |
| itile0 | uint32_t | 输入 0 的 Tile 索引 |
| itile1 | uint32_t | 输入 1 的 Tile 索引 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

### 4.3 sub_tiles

**函数签名**:
```cpp
ALWI void sub_tiles(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t itile0,
    uint32_t itile1,
    uint32_t idst
);
```

**描述**: 逐元素减法: `dst[idst] = src0[itile0] - src1[itile1]`

**参数说明**: 同 add_tiles

**返回值**: 无

---

### 4.4 mul_tiles

**函数签名**:
```cpp
ALWI void mul_tiles(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t itile0,
    uint32_t itile1,
    uint32_t idst
);
```

**描述**: 逐元素乘法: `dst[idst] = src0[itile0] * src1[itile1]`

**参数说明**: 同 add_tiles

**返回值**: 无

---

### 4.5 binary_tiles_init

**函数签名**:
```cpp
template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init(
    uint32_t icb0,
    uint32_t icb1,
    bool acc_to_dest = false,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 模板化的二元操作初始化，支持指定操作类型。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| full_init | bool | 是否执行完整初始化 |
| eltwise_binary_type | EltwiseBinaryType | 操作类型（ELWADD, ELWSUB, ELWMUL 等） |
| icb0 | uint32_t | 第一个输入 CB ID |
| icb1 | uint32_t | 第二个输入 CB ID |
| acc_to_dest | bool | 是否累加到目标 |
| call_line | uint32_t | 调用行号 |

**EltwiseBinaryType 枚举**:
```cpp
enum EltwiseBinaryType {
    ELWADD,      // 加法
    ELWSUB,      // 减法
    ELWMUL,      // 乘法
    ELWMAX,      // 最大值
    ELWMIN,      // 最小值
    // ... 其他类型
};
```

**示例**:
```cpp
void MAIN {
    // 初始化乘法操作
    binary_tiles_init<true, ELWMUL>(cb_in0, cb_in1);

    tile_regs_acquire();
    mul_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
}
```

---

### 4.6 mul_tiles_init

**函数签名**:
```cpp
ALWI void mul_tiles_init(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 乘法操作的专用初始化。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb0 | uint32_t | 第一个输入 CB ID |
| icb1 | uint32_t | 第二个输入 CB ID |
| call_line | uint32_t | 调用行号 |

**返回值**: 无

---

### 4.7 add_tiles_init

**函数签名**:
```cpp
ALWI void add_tiles_init(
    uint32_t icb0,
    uint32_t icb1,
    bool acc_to_dest = false,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 加法操作的专用初始化。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb0 | uint32_t | 第一个输入 CB ID |
| icb1 | uint32_t | 第二个输入 CB ID |
| acc_to_dest | bool | 是否累加到目标 |
| call_line | uint32_t | 调用行号 |

**返回值**: 无

---

### 4.8 sub_tiles_init

**函数签名**:
```cpp
ALWI void sub_tiles_init(
    uint32_t icb0,
    uint32_t icb1,
    bool acc_to_dest = false,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 减法操作的专用初始化。

**参数说明**: 同 add_tiles_init

**返回值**: 无

---

### 4.9 binary_dest_reuse_tiles_init

**函数签名**:
```cpp
template <EltwiseBinaryType eltwise_binary_type = ELWADD,
          EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles_init(
    uint32_t icb0,
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 初始化目标复用模式的二元操作，允许将前一次计算结果作为输入。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| eltwise_binary_type | EltwiseBinaryType | 操作类型 |
| binary_reuse_dest | EltwiseBinaryReuseDestType | 目标复用模式 |
| icb0 | uint32_t | 输入 CB ID |
| call_line | uint32_t | 调用行号 |

**EltwiseBinaryReuseDestType 枚举**:
```cpp
enum class EltwiseBinaryReuseDestType {
    NONE,           // 不复用
    DEST_TO_SRCA,   // 目标作为源 A
    DEST_TO_SRCB    // 目标作为源 B
};
```

**返回值**: 无

**使用场景**: 链式计算，如 `((a + b) * c) - d`，避免中间结果的打包/解包开销。

---

### 4.10 binary_dest_reuse_tiles

**函数签名**:
```cpp
template <EltwiseBinaryType eltwise_binary_type = ELWADD,
          EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
ALWI void binary_dest_reuse_tiles(
    uint32_t in_cb_id,
    uint32_t in_tile_index,
    uint32_t dst_tile_index
);
```

**描述**: 执行目标复用模式的二元操作。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| in_cb_id | uint32_t | 输入 CB ID |
| in_tile_index | uint32_t | 输入 Tile 索引 |
| dst_tile_index | uint32_t | 目标寄存器索引 |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    // 链式计算: ((a + b) * c)
    binary_op_init_common(cb_a, cb_b, cb_out);

    tile_regs_acquire();
    // 第一步: dst = a + b
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();

    // 第二步: dst = dst * c (复用目标)
    binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_c);
    binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_c, 0, 0);

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
}
```

---

## 5. 逐元素一元操作 (SFPU)

SFPU（Special Function Processing Unit）专门执行逐元素数学函数。

### 5.1 激活函数

#### 5.1.1 sigmoid_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void sigmoid_tile_init();

template <int vec_mode = VectorMode::RC, bool fast_and_approx = false>
ALWI void sigmoid_tile(uint32_t idst);
```

**描述**: Sigmoid 激活函数: `sigmoid(x) = 1 / (1 + exp(-x))`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式（精度换速度） |
| vec_mode | int | 向量模式（VectorMode::RC 表示行列模式） |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    sigmoid_tile_init<false>();  // 精确模式初始化

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);      // 加载输入
    sigmoid_tile< VectorMode::RC, false>(0);  // 应用 sigmoid
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
}
```

---

#### 5.1.2 tanh_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void tanh_tile_init();

template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst);
```

**描述**: 双曲正切激活函数: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.1.3 relu_tile

**函数签名**:
```cpp
ALWI void relu_tile_init();
ALWI void relu_tile(uint32_t idst);
```

**描述**: ReLU 激活函数: `relu(x) = max(0, x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.1.4 gelu_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void gelu_tile_init();

template <bool fast_and_approx = false>
ALWI void gelu_tile(uint32_t idst);
```

**描述**: GELU 激活函数: `gelu(x) = x * Φ(x)`，其中 Φ(x) 是标准正态分布的累积分布函数

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.1.5 silu_tile (Swish)

**函数签名**:
```cpp
ALWI void silu_tile_init();
ALWI void silu_tile(uint32_t idst);
```

**描述**: SiLU（Swish）激活函数: `silu(x) = x * sigmoid(x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

**使用场景**: 现代 Transformer 模型（如 SwiGLU）中常用的激活函数。

---

### 5.2 指数和对数函数

#### 5.2.1 exp_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void exp_tile_init();

template <bool fast_and_approx = false>
ALWI void exp_tile(uint32_t idst);
```

**描述**: 自然指数函数: `exp(x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.2.2 exp2_tile

**函数签名**:
```cpp
ALWI void exp2_tile_init();
ALWI void exp2_tile(uint32_t idst);
```

**描述**: 以 2 为底的指数函数: `2^x`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.2.3 expm1_tile

**函数签名**:
```cpp
template <bool approx = false>
ALWI void expm1_tile_init();

template <bool approx = false>
ALWI void expm1_tile(uint32_t idst);
```

**描述**: 指数减 1: `expm1(x) = exp(x) - 1`，对小值 x 更精确

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| approx | bool | 使用近似模式 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.2.4 log_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void log_tile_init();

template <bool fast_and_approx = false>
ALWI void log_tile(uint32_t idst);
```

**描述**: 自然对数: `ln(x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.2.5 log_with_base_tile

**函数签名**:
```cpp
template <bool fast_and_approx = false>
ALWI void log_with_base_tile_init();

template <bool fast_and_approx = false>
ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale);
```

**描述**: 任意底数的对数: `log_base(x) = ln(x) / ln(base)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| fast_and_approx | bool | 使用快速近似模式 |
| idst | uint32_t | 目标寄存器 ID |
| base_scale | uint32_t | 底数的缩放因子（预计算的 1/ln(base)） |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    log_with_base_tile_init<false>();

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    // 计算以 10 为底的对数，base_scale = 1/ln(10) ≈ 0.4343
    log_with_base_tile<false>(0, 0x3EDE5BD9);  // FP16 格式的 0.4343
    tile_regs_commit();
}
```

---

### 5.3 幂和根函数

#### 5.3.1 sqrt_tile

**函数签名**:
```cpp
ALWI void sqrt_tile_init();
ALWI void sqrt_tile(uint32_t idst);
```

**描述**: 平方根: `sqrt(x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.3.2 power_tile

**函数签名**:
```cpp
ALWI void power_tile_init();
ALWI void power_tile(uint32_t idst, uint32_t param0);
```

**描述**: 幂运算: `power(x, n) = x^n`，n 为整数指数

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |
| param0 | uint32_t | 指数 n（整数） |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    power_tile_init();

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    power_tile(0, 3);  // 计算 x^3
    tile_regs_commit();
}
```

---

#### 5.3.3 power_iterative_tile

**函数签名**:
```cpp
ALWI void power_iterative_tile_init();
ALWI void power_iterative_tile(uint32_t idst, uint32_t param0);
```

**描述**: 迭代幂运算，用于较大的整数指数。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |
| param0 | uint32_t | 指数 n（整数） |

**返回值**: 无

---

#### 5.3.4 square_tile

**函数签名**:
```cpp
ALWI void square_tile_init();
ALWI void square_tile(uint32_t idst);
```

**描述**: 平方: `square(x) = x^2`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

### 5.4 符号和绝对值函数

#### 5.4.1 abs_tile

**函数签名**:
```cpp
ALWI void abs_tile_init();
ALWI void abs_tile(uint32_t idst);
ALWI void abs_tile_int32(uint32_t idst);
```

**描述**: 绝对值: `abs(x)`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

**注意**: `abs_tile_int32` 专用于 Int32 数据类型。

---

#### 5.4.2 sign_tile

**函数签名**:
```cpp
ALWI void sign_tile_init();
ALWI void sign_tile(uint32_t idst);
```

**描述**: 符号函数: `sign(x) = -1 if x < 0, 0 if x = 0, 1 if x > 0`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

#### 5.4.3 signbit_tile

**函数签名**:
```cpp
ALWI void signbit_tile_init();
ALWI void signbit_tile(uint32_t idst);
ALWI void signbit_tile_int32(uint32_t idst);
```

**描述**: 符号位检测: `signbit(x) = 1 if x < 0 else 0`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

### 5.5 其他特殊函数

#### 5.5.1 heaviside_tile

**函数签名**:
```cpp
ALWI void heaviside_tile_init();
ALWI void heaviside_tile(uint32_t idst, uint32_t param0);
```

**描述**: Heaviside 阶跃函数: `H(x) = 0 if x < 0, param0 if x = 0, 1 if x > 0`

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |
| param0 | uint32_t | x=0 时的值（通常为 0 或 1） |

**返回值**: 无

---

#### 5.5.2 tiled_prod_tile

**函数签名**:
```cpp
ALWI void tiled_prod_tile_init();
ALWI void tiled_prod_tile(uint32_t idst);
```

**描述**: Tile 内所有元素的乘积。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

### 5.6 SFPU 操作汇总表

| 函数 | 数学表达式 | 初始化函数 | 近似模式支持 |
|------|-----------|-----------|-------------|
| sigmoid_tile | 1/(1+exp(-x)) | sigmoid_tile_init() | 是 |
| tanh_tile | (e^x - e^-x)/(e^x + e^-x) | tanh_tile_init() | 是 |
| relu_tile | max(0, x) | relu_tile_init() | 否 |
| gelu_tile | x * Φ(x) | gelu_tile_init() | 是 |
| silu_tile | x * sigmoid(x) | silu_tile_init() | 否 |
| exp_tile | e^x | exp_tile_init() | 是 |
| exp2_tile | 2^x | exp2_tile_init() | 否 |
| expm1_tile | e^x - 1 | expm1_tile_init() | 是 |
| log_tile | ln(x) | log_tile_init() | 是 |
| log_with_base_tile | log_base(x) | log_with_base_tile_init() | 是 |
| sqrt_tile | √x | sqrt_tile_init() | 否 |
| power_tile | x^n | power_tile_init() | 否 |
| square_tile | x^2 | square_tile_init() | 否 |
| abs_tile | \|x\| | abs_tile_init() | 否 |
| sign_tile | sign(x) | sign_tile_init() | 否 |
| signbit_tile | x < 0 ? 1 : 0 | signbit_tile_init() | 否 |
| heaviside_tile | H(x) | heaviside_tile_init() | 否 |

---

## 6. 归约操作

### 6.1 reduce_init

**函数签名**:
```cpp
template <PoolType reduce_type = REDUCE_OP,
          ReduceDim reduce_dim = REDUCE_DIM,
          bool enforce_fp32_accumulation = false>
ALWI void reduce_init(
    uint32_t icb,            // 输入 CB ID
    uint32_t icb_scaler,     // 缩放因子 CB ID
    uint32_t ocb,            // 输出 CB ID
    uint32_t call_line = __builtin_LINE()
);
```

**描述**: 初始化归约操作引擎。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| reduce_type | PoolType | 归约类型（SUM, AVG, MAX） |
| reduce_dim | ReduceDim | 归约维度（REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR） |
| enforce_fp32_accumulation | bool | 强制使用 FP32 累加 |
| icb | uint32_t | 输入 CB ID |
| icb_scaler | uint32_t | 缩放因子 CB ID（用于 AVG） |
| ocb | uint32_t | 输出 CB ID |
| call_line | uint32_t | 调用行号 |

**PoolType 枚举**:
```cpp
enum PoolType {
    SUM,    // 求和
    AVG,    // 平均值
    MAX     // 最大值
};
```

**ReduceDim 枚举**:
```cpp
enum ReduceDim {
    REDUCE_ROW,     // 按行归约
    REDUCE_COL,     // 按列归约
    REDUCE_SCALAR   // 全局归约到标量
};
```

**返回值**: 无

---

### 6.2 reduce_uninit

**函数签名**:
```cpp
template <bool enforce_fp32_accumulation = false>
ALWI void reduce_uninit(uint32_t icb = 0);
```

**描述**: 反初始化归约操作引擎，释放资源。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| enforce_fp32_accumulation | bool | 是否使用 FP32 累加模式 |
| icb | uint32_t | 输入 CB ID |

**返回值**: 无

---

### 6.3 reduce_tile

**函数签名**:
```cpp
template <PoolType reduce_type = REDUCE_OP,
          ReduceDim reduce_dim = REDUCE_DIM,
          bool enforce_fp32_accumulation = false>
ALWI void reduce_tile(
    uint32_t icb,
    uint32_t icb_scaler,
    uint32_t itile,          // 输入 Tile 索引
    uint32_t itile_scaler,   // 缩放因子 Tile 索引
    uint32_t idst            // 目标寄存器 ID
);
```

**描述**: 执行 Tile 级别的归约操作。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb | uint32_t | 输入 CB ID |
| icb_scaler | uint32_t | 缩放因子 CB ID |
| itile | uint32_t | 输入 Tile 索引 |
| itile_scaler | uint32_t | 缩放因子 Tile 索引 |
| idst | uint32_t | 目标寄存器 ID |

**返回值**: 无

---

### 6.4 reduce_tile_math

**函数签名**:
```cpp
template <PoolType reduce_type = REDUCE_OP,
          ReduceDim reduce_dim = REDUCE_DIM,
          bool enforce_fp32_accumulation = false>
ALWI void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4);
```

**描述**: 仅执行归约的数学运算部分（不含数据移动）。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 目标寄存器 ID |
| num_faces | uint32_t | Tile 的面数（默认 4） |

**返回值**: 无

**使用场景**: 当数据已在寄存器中，只需执行归约计算时使用。

---

### 6.5 归约操作示例

```cpp
#include "compute_kernel_api/reduce.h"

void MAIN {
    // 初始化行方向求和归约
    reduce_init<SUM, REDUCE_ROW, false>(cb_in, cb_scaler, cb_out);

    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    // 将 32x32 Tile 归约为 1x32（每行求和）
    reduce_tile<SUM, REDUCE_ROW, false>(cb_in, cb_scaler, 0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_in, 1);

    reduce_uninit<false>(cb_in);
}
```

---

## 7. 数据格式转换

### 7.1 tilize

**函数签名**:
```cpp
ALWI void tilize_init_short(uint32_t icb);
ALWI void tilize_block(uint32_t icb, uint32_t num_tiles, uint32_t ocb);
```

**描述**: 将线性数据转换为 Tile 格式。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb | uint32_t | 输入 CB ID（线性数据） |
| num_tiles | uint32_t | 要转换的 Tile 数量 |
| ocb | uint32_t | 输出 CB ID（Tile 格式） |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    tilize_init_short(cb_linear);

    tile_regs_acquire();
    tilize_block(cb_linear, num_tiles, cb_tiled);
    tile_regs_commit();
}
```

---

### 7.2 untilize

**函数签名**:
```cpp
ALWI void untilize_init_short(uint32_t icb);
ALWI void untilize_block(uint32_t icb, uint32_t num_tiles, uint32_t ocb);
```

**描述**: 将 Tile 格式数据转换为线性格式。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| icb | uint32_t | 输入 CB ID（Tile 格式） |
| num_tiles | uint32_t | 要转换的 Tile 数量 |
| ocb | uint32_t | 输出 CB ID（线性数据） |

**返回值**: 无

---

### 7.3 pack_reconfig_data_format

**函数签名**:
```cpp
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t new_cb_id);

template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const uint32_t old_cb_id, const uint32_t new_cb_id);
```

**描述**: 重新配置打包器的数据格式，用于在运行时切换输出数据类型。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| is_tile_dim_reconfig_en | bool | 是否启用 Tile 维度重配置 |
| new_cb_id | uint32_t | 新的输出 CB ID |
| old_cb_id | uint32_t | 旧的输出 CB ID |

**返回值**: 无

**使用场景**: 当需要将同一计算结果以不同格式输出到不同 CB 时使用。

---

## 8. 打包操作

### 8.1 pack_tile

**函数签名**:
```cpp
template <bool out_of_order_output = false>
ALWI void pack_tile(
    uint32_t ifrom_dst,      // 源寄存器 ID
    uint32_t icb,            // 目标 CB ID
    std::uint32_t output_tile_index = 0  // 输出 Tile 索引
);
```

**描述**: 将 Tile 寄存器中的数据打包到输出 CB。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| out_of_order_output | bool | 是否支持乱序输出 |
| ifrom_dst | uint32_t | 源寄存器 ID |
| icb | uint32_t | 目标 CB ID |
| output_tile_index | uint32_t | 输出 Tile 在 CB 中的索引 |

**返回值**: 无

**示例**:
```cpp
void MAIN {
    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);  // 将寄存器 0 打包到 cb_out
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
```

---

### 8.2 pack_tile_block

**函数签名**:
```cpp
ALWI void pack_tile_block(
    uint32_t ifrom_dst,      // 起始源寄存器 ID
    uint32_t icb,
    uint32_t ntiles          // 要打包的 Tile 数量
);
```

**描述**: 批量打包多个 Tile 寄存器到输出 CB。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| ifrom_dst | uint32_t | 起始源寄存器 ID |
| icb | uint32_t | 目标 CB ID |
| ntiles | uint32_t | 要打包的 Tile 数量 |

**返回值**: 无

---

### 8.3 pack_reconfig_l1_acc

**函数签名**:
```cpp
ALWI void pack_reconfig_l1_acc(const uint32_t l1_acc_en);
```

**描述**: 重新配置 L1 累加器模式。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| l1_acc_en | uint32_t | 是否启用 L1 累加（1=启用，0=禁用） |

**返回值**: 无

**使用场景**: 需要在 L1 内存中累加部分结果时使用，常用于大规模矩阵乘法的分块累加。

---

### 8.4 pack_rows_init

**函数签名**:
```cpp
ALWI void pack_rows_init(uint32_t num_rows);
```

**描述**: 初始化行打包模式。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| num_rows | uint32_t | 要打包的行数 |

**返回值**: 无

---

### 8.5 pack_rows

**函数签名**:
```cpp
ALWI void pack_rows(
    uint32_t idst,
    uint32_t ocb,
    uint32_t output_index = 0
);
```

**描述**: 从寄存器打包指定行到输出 CB。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| idst | uint32_t | 源寄存器 ID |
| ocb | uint32_t | 输出 CB ID |
| output_index | uint32_t | 输出索引 |

**返回值**: 无

---

### 8.6 pack_rows_uninit

**函数签名**:
```cpp
ALWI void pack_rows_uninit();
```

**描述**: 反初始化行打包模式。

**参数**: 无

**返回值**: 无

---

## 9. SFPI 条件执行

SFPI（Special Function Processor Interface）提供向量级条件执行能力。

### 9.1 v_if

**语法**:
```cpp
v_if(condition);
    // 条件为真时执行的代码
v_endif;
```

**描述**: 向量条件执行的开始。条件应用于向量中的每个元素。

**参数说明**:
| 参数 | 类型 | 描述 |
|------|------|------|
| condition | 表达式 | 向量条件表达式 |

**返回值**: 无

---

### 9.2 v_else

**语法**:
```cpp
v_if(condition);
    // 条件为真时执行
v_else;
    // 条件为假时执行
v_endif;
```

**描述**: 向量条件执行的 else 分支。

**参数**: 无

**返回值**: 无

---

### 9.3 v_endif

**语法**:
```cpp
v_if(condition);
    // 条件代码
v_endif;
```

**描述**: 标记向量条件执行块的结束。

**参数**: 无

**返回值**: 无

---

### 9.4 SFPI 使用示例

#### 示例 1: 条件 ReLU
```cpp
#include "compute_kernel_api/sfpi.h"

void MAIN {
    unary_op_init_common(cb_in, cb_out);

    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);

    // 向量条件: 如果元素 < 0，设为 0
    v_if(sfpi::dst_reg[0] < 0.0f);
        sfpi::dst_reg[0] = 0.0f;
    v_endif;

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    cb_pop_front(cb_in, 1);
}
```

#### 示例 2: If-Else 条件
```cpp
void MAIN {
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);

    // 如果 x > 0，y = sqrt(x)，否则 y = 0
    v_if(sfpi::dst_reg[0] > 0.0f);
        sfpi::dst_reg[0] = sfpi::sqrt(sfpi::dst_reg[0]);
    v_else;
        sfpi::dst_reg[0] = 0.0f;
    v_endif;

    tile_regs_commit();
}
```

#### 示例 3: 嵌套条件（使用逻辑与）
```cpp
void MAIN {
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);

    // 如果 0 < x < 1，y = x，否则 y = 0
    v_if((sfpi::dst_reg[0] > 0.0f) && (sfpi::dst_reg[0] < 1.0f));
        // 保持原值
    v_else;
        sfpi::dst_reg[0] = 0.0f;
    v_endif;

    tile_regs_commit();
}
```

---

### 9.5 SFPI 注意事项

1. **性能影响**: 条件执行会降低 SFPU 吞吐量，尽量避免在性能关键路径上使用复杂条件。

2. **嵌套限制**: SFPI 支持有限的嵌套深度（通常 2-3 层），过度嵌套会导致编译错误。

3. **数据类型**: SFPI 操作默认使用 FP16 格式，混合精度需要显式转换。

4. **调试困难**: SFPI 代码难以调试，建议在 Host 端验证算法逻辑。

---

## 10. 使用示例

### 10.1 完整计算内核模板

```cpp
// compute_kernel.cpp
#include "compute_kernel_api/compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sigmoid.h"

namespace NAMESPACE {
void MAIN {
    // 获取编译时参数
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    // 初始化操作
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // 等待输入数据
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        // 计算阶段
        tile_regs_acquire();
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();

        // 打包阶段
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        // 释放输入
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
    }
}
}  // namespace NAMESPACE
```

---

### 10.2 带激活函数的矩阵乘法

```cpp
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"

void MAIN {
    mm_init(cb_in0, cb_in1, cb_intermediate);
    gelu_tile_init<false>();

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();

            // 矩阵乘法累加
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // 应用 GELU 激活
            gelu_tile<false>(0);

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
    }
}
```

---

### 10.3 多操作链式计算

```cpp
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sigmoid.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"

void MAIN {
    binary_op_init_common(cb_a, cb_b, cb_out);
    sigmoid_tile_init<false>();
    sqrt_tile_init();

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        tile_regs_acquire();

        // 步骤 1: c = a + b
        add_tiles(cb_a, cb_b, 0, 0, 0);

        // 步骤 2: d = sigmoid(c)
        sigmoid_tile<false>(0);

        // 步骤 3: e = sqrt(d)
        sqrt_tile(0);

        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}
```

---

### 10.4 归约操作示例

```cpp
#include "compute_kernel_api/reduce.h"

void MAIN {
    // 全局求和归约
    reduce_init<SUM, REDUCE_SCALAR, false>(cb_in, cb_scaler, cb_out);

    tile_regs_acquire();

    // 累加所有输入 Tile
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);
        reduce_tile<SUM, REDUCE_SCALAR, false>(cb_in, cb_scaler, 0, 0, 0);
        cb_pop_front(cb_in, 1);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();

    reduce_uninit<false>(cb_in);
}
```

---

### 10.5 块矩阵乘法

```cpp
#include "compute_kernel_api/matmul.h"

void MAIN {
    // 初始化 4x4 块矩阵乘法
    mm_block_init(cb_in0, cb_in1, cb_out, 0, 4, 4, 4);

    for (uint32_t mt = 0; mt < Mt; mt += 4) {
        for (uint32_t nt = 0; nt < Nt; nt += 4) {
            tile_regs_acquire();

            // 执行块乘法
            matmul_block(cb_in0, cb_in1, mt, nt, 0, 0, 4, 4, 4);

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 16);  // 4x4 = 16 tiles
            pack_tile_block(0, cb_out, 16);
            cb_push_back(cb_out, 16);
            tile_regs_release();
        }
    }
}
```

---

## 附录 A: 常见错误与解决

### A.1 Tile 寄存器未正确释放

**错误现象**: 内核挂起或产生错误结果

**原因**: `tile_regs_release()` 未调用或调用顺序错误

**正确做法**:
```cpp
tile_regs_acquire();
// ... 计算 ...
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_out);
tile_regs_release();  // 必须在 pack 之后
```

### A.2 CB 操作顺序错误

**错误现象**: 数据损坏或内核挂起

**原因**: 未等待 CB 数据就执行操作

**正确做法**:
```cpp
cb_wait_front(cb_in, num_tiles);  // 先等待
// ... 使用数据 ...
cb_pop_front(cb_in, num_tiles);   // 最后释放
```

### A.3 未初始化就执行操作

**错误现象**: 未定义行为或错误结果

**原因**: 未调用对应的 init 函数

**正确做法**:
```cpp
sigmoid_tile_init<false>();  // 先初始化
sigmoid_tile<false>(0);       // 再执行
```

---

## 附录 B: 性能优化建议

1. **批处理**: 尽可能一次处理多个 Tile，减少循环开销
2. **避免重复初始化**: 在循环外调用 init 函数
3. **使用近似模式**: 如果精度允许，使用 `fast_and_approx=true` 模式
4. **复用目标寄存器**: 使用 `binary_dest_reuse_tiles` 减少数据移动
5. **合理选择块大小**: 块矩阵乘法时，选择合适的 ct_dim/rt_dim/kt_dim

---

*文档结束*
