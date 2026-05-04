# TT-Sim Simulator Fatal Constraint Scan

> 当前扫描对象: `/root/dev/vibe_dsl/tt_metal_repo/sim/libttsim.so`
> 实际二进制: `/root/dev/vibe_dsl/tt_metal_repo/sim/libttsim_bh.so`
> 日期: `2026-04-09`

## 1. 目的

这份文档记录当前 Blackhole TT-Sim 二进制里，
通过公共 fatal helper 显式编码的 simulator 约束。

它的用途不是替代源码文档，
而是在 direct runtime / TT-Sim triage 时回答两个问题：

- 当前报错更像 simulator 自身的 hard gate，还是 target contract 回归
- 除了当前已经遇到的 `fp16` unpack 路径，还有哪些显式 fatal taxonomy

## 2. 扫描方法

当前 `libttsim_bh.so` 是 stripped binary，
不能直接依赖符号表。

本轮扫描的方法是：

1. 对二进制做 `objdump -d -M intel`
2. 定位公共 fatal helper `0x3cff`
3. 枚举所有：
   - `mov edi,0x5` -> `UntestedFunctionality`
   - `mov edi,0x6` -> `UnimplementedFunctionality`
   - `mov edi,0x7` -> `UnsupportedFunctionality`
   且随后 `call 0x3cff` 的 call site
4. 从 call site 前的 `lea rsi, ...` / `lea rdx, ...`
   恢复函数名字符串和细节字符串

当前已确认：

- `0x3cff` 不是普通日志函数
- 命中 fatal taxonomy 后会打印并直接 `_Exit(1)`
- 因此这些 taxonomy 都是 hard gate，不是 warning

## 3. Fatal Taxonomy

公共 helper 当前包含下面这些类别：

| `edi` | 类别 |
|---:|---|
| `0` | `UndefinedBehavior` |
| `1` | `UnpredictableValueUsed` |
| `2` | `NonContractualBehavior` |
| `3` | `AssertionFailure` |
| `4` | `MissingSpecification` |
| `5` | `UntestedFunctionality` |
| `6` | `UnimplementedFunctionality` |
| `7` | `UnsupportedFunctionality` |
| `8` | `SystemError` |
| `9` | `ConfigurationError` |

本轮重点扫描的是 `5 / 6 / 7`。

## 4. 全量扫描摘要

| Category | Call Sites | Unique Function+Detail Pairs | Unique Functions |
|---|---:|---:|---:|
| `UntestedFunctionality` | 18 | 17 | 13 |
| `UnimplementedFunctionality` | 655 | 493 | 212 |
| `UnsupportedFunctionality` | 75 | 67 | 47 |

结论：

- `fp16` 不是孤立边界
- 当前 TT-Sim 里还存在大量 `Unimplemented` / `Unsupported` hard gate
- 看到这三类错误时，不能默认先怀疑 TileLang target contract

## 5. 当前 `UntestedFunctionality` 明细

下面这张表是当前最值得直接复用的“硬门”列表。

| Function | Detail | Count | Representative Call Site |
|---|---|---:|---|
| `noc_cmd_ctrl` | `atomic noc_at_data=0x%x\n` | 2 | `0x90c4` |
| `riscv_debug_regs_wr32` | `trisc_reset_pc_override=0x%x\n` | 1 | `0x937f` |
| `tensix_execute_addrcrzw` | `inst=0x%x\n` | 1 | `0x444c` |
| `tensix_execute_elw_op` | `elwsub: ALU_ACC_CTRL_INT8_math_enabled\n` | 1 | `0x6367` |
| `tensix_execute_incadcxy` | `cnt_set_mask=%d\n` | 1 | `0x434a` |
| `tensix_execute_incadczw` | `cnt_set_mask=%d\n` | 1 | `0x43dd` |
| `tensix_execute_pacr` | `edge_mask=0x%x\n` | 1 | `0x680e` |
| `tensix_execute_pacr` | `fp16 data format\n` | 1 | `0x69d3` |
| `tensix_execute_setadc` | `` | 1 | `0x42b5` |
| `tensix_execute_setadcxx` | `cnt_set_mask=%d\n` | 1 | `0x7fa3` |
| `tensix_execute_setadcxy` | `cnt_set_mask=%d\n` | 1 | `0x4104` |
| `tensix_execute_setadczw` | `cnt_set_mask=%d\n` | 1 | `0x419a` |
| `tensix_execute_unpacr` | `ch1_w=%d\n` | 1 | `0x6f20` |
| `tensix_execute_unpacr` | `fp16\n` | 1 | `0x71b5` |
| `tensix_execute_unpacr` | `int8 is_unsigned=%d\n` | 1 | `0x70f2` |
| `tensix_execute_unpacr` | `zero_write2=%d\n` | 1 | `0x711a` |
| `tensix_execute_zerosrc` | `src_mask=%d\n` | 1 | `0x5619` |

高信号解释：

- `tensix_execute_unpacr : fp16`
  说明当前 TT-Sim 明确把这条 `fp16` unpack 路径作为 `UntestedFunctionality`
- `tensix_execute_pacr : fp16 data format`
  说明 pack 侧也有对应的 `fp16` data-format hard gate
- `noc_cmd_ctrl : atomic noc_at_data`
  说明某些 atomic NOC path 也在 current TT-Sim 约束面之外

## 6. 高频函数分布

这些表不是完整 pair dump，
而是当前 binary 里最常出现 hard gate 的函数分布。

### 6.1 `UntestedFunctionality`

| Function | Call Sites |
|---|---:|
| `tensix_execute_unpacr` | 4 |
| `noc_cmd_ctrl` | 2 |
| `tensix_execute_pacr` | 2 |
| `riscv_debug_regs_wr32` | 1 |
| `tensix_execute_addrcrzw` | 1 |
| `tensix_execute_elw_op` | 1 |
| `tensix_execute_incadcxy` | 1 |
| `tensix_execute_incadczw` | 1 |
| `tensix_execute_setadc` | 1 |
| `tensix_execute_setadcxx` | 1 |
| `tensix_execute_setadcxy` | 1 |
| `tensix_execute_setadczw` | 1 |
| `tensix_execute_zerosrc` | 1 |

### 6.2 `UnimplementedFunctionality`

| Function | Call Sites |
|---|---:|
| `noc_cmd_ctrl` | 90 |
| `(unrecovered)` | 59 |
| `tensix_execute_elw_op` | 28 |
| `tensix_execute_pacr` | 23 |
| `tensix_execute_unpacr` | 23 |
| `tensix_execute_matmul_op` | 20 |
| `eth_txq_regs_wr32` | 13 |
| `tensix_execute_gmpool` | 11 |
| `noc_overlay_wr32` | 10 |
| `tensix_execute_movd2a` | 9 |
| `decode_and_execute_alu` | 8 |
| `noc_regs_wr32` | 8 |
| `tensix_execute_mova2d` | 8 |
| `tensix_execute_movb2d` | 8 |
| `tensix_execute_movd2b` | 8 |
| `tile_wr_bytes` | 8 |
| `tensix_execute_cfgshiftmask` | 7 |
| `mop_expander` | 6 |
| `tensix_execute_setrwc` | 6 |
| `tensix_execute_sfpstore` | 6 |
| `tensix_execute_unpacr_nop` | 6 |
| `tensix_execute_zeroacc` | 6 |

### 6.3 `UnsupportedFunctionality`

| Function | Call Sites |
|---|---:|
| `tensix_execute_unpacr` | 10 |
| `tensix_execute_elw_op` | 9 |
| `tensix_execute_matmul_op` | 4 |
| `tensix_execute_gmpool` | 3 |
| `decode_and_execute_ecall_ebreak` | 2 |
| `tensix_execute_cleardvalid` | 2 |
| `tensix_execute_movb2d` | 2 |
| `tensix_execute_pacr` | 2 |
| `tensix_execute_sfpconfig` | 2 |
| `tensix_execute_unpacr_nop` | 2 |
| `(unrecovered)` | 1 |
| `decode_and_execute_alu_imm` | 1 |
| `decode_and_execute_fence_i` | 1 |
| `libttsim_pci_mem_wr_bytes` | 1 |
| `math_update_counters` | 1 |
| `mop_expander` | 1 |
| `remap_virtual_coordinate` | 1 |
| `tensix_decode_and_execute_sfpswap` | 1 |
| `tensix_execute_apool3s1` | 1 |
| `tensix_execute_apool3s2` | 1 |
| `tensix_execute_conv3s1` | 1 |
| `tensix_execute_conv3s2` | 1 |
| `tensix_execute_dotpv` | 1 |
| `tensix_execute_gapool` | 1 |
| `tensix_execute_mfconv3s1` | 1 |
| `tensix_execute_mova2d` | 1 |
| `tensix_execute_mpool3s1` | 1 |
| `tensix_execute_mpool3s2` | 1 |
| `tensix_execute_reg2flop` | 1 |
| `tensix_execute_semget` | 1 |
| `tensix_execute_seminit` | 1 |
| `tensix_execute_sempost` | 1 |
| `tensix_execute_semwait` | 1 |
| `tensix_execute_setdmareg` | 1 |
| `tensix_execute_sfpcast` | 1 |
| `tensix_execute_sfpiadd` | 1 |
| `tensix_execute_sfplz` | 1 |
| `tensix_execute_sfpmul24` | 1 |
| `tensix_execute_sfpsetexp` | 1 |
| `tensix_execute_sfpshft` | 1 |
| `tensix_execute_shiftxa` | 1 |
| `tensix_execute_stallwait` | 1 |
| `tensix_execute_storereg` | 1 |
| `tensix_execute_zerosrc` | 1 |
| `tile_rd_bytes` | 1 |
| `tile_wr_bytes` | 1 |
| `tlb_translate` | 1 |

## 7. 当前结论

1. `Untested / Unimplemented / Unsupported` 都是 fatal taxonomy，
   不是无害 warning。
2. `fp16` 路径确实是当前 TT-Sim 里显式编码的 hard gate，
   但它不是唯一约束面。
3. 当前 TT-Sim 还对多类：
   - `pack / unpack`
   - `elw_op`
   - `matmul_op`
   - `gmpool`
   - `noc_cmd_ctrl`
   - 若干 `sfp* / semaphore / tile io`
   存在 simulator-side hard gate。
   本轮额外确认：
   32B bf16 staged stick page transport 会在 `noc_cmd_ctrl`
   命中 read alignment mismatch fatal，
   当前 direct-runtime page transport 仍按 64B page alignment admission。
   2026-05-02 另确认：standalone bf16 row `reduce_tile` 在 writer
   CB binding、rank-1 scalar page writeback、final writer barrier/pop 都修正后，
   放开 direct-runtime gate 真跑仍在 workload enqueue 后命中
   `UnimplementedFunctionality: tensix_execute_pacr: count=1`。
   2026-05-05 另确认：bf16 flash-attn seq128/256/512 的 loop-carried
   input exact-CB backedge source/spec admission 后，当前 TT-Sim 也会命中
   同一 `tensix_execute_pacr: count=1` fatal；seq64 accumulator-only
   loop-carried exact-CB direct runtime 仍是正例，因此 gate 必须按 typed
   input-CB backedge release 收窄。
4. 当 direct runtime 首次命中这三类 taxonomy 时，
   应先把它和 target contract 回归分开判断。

## 8. 使用方式

出现类似报错时，按下面顺序判断：

1. 先看错误 taxonomy 是否属于 `Untested / Unimplemented / Unsupported`
2. 若是，先查本文件中是否已有同类函数/细节
3. 如果命中，优先把它视为 simulator capability boundary
4. 只有在已知 gate 之外，才继续怀疑 TileLang target contract

## 9. 再生成方法

如果后续需要对新的 simulator binary 重扫，
复用下面的流程即可：

```bash
objdump -d -M intel /root/dev/vibe_dsl/tt_metal_repo/sim/libttsim_bh.so > /tmp/libttsim_bh.objdump
```

然后枚举所有：

- `mov edi,0x5` 且随后 `call 0x3cff`
- `mov edi,0x6` 且随后 `call 0x3cff`
- `mov edi,0x7` 且随后 `call 0x3cff`

并从前几行的 `lea rsi` / `lea rdx` 恢复函数名和细节字符串。

`(unrecovered)` 表示当前 stripped binary 周围没有稳定函数名字符串可恢复。
