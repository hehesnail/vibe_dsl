# jit_build/ 模块源码解析

## 1. 模块概述

### 1.1 模块职责

`jit_build/` 模块是 TT-Metal 框架的**JIT（Just-In-Time）编译支持核心**，负责在运行时动态编译内核代码和设备固件。主要功能包括：

- **内核代码编译**：将用户编写的 C++ 内核代码编译为 RISC-V 可执行文件
- **固件编译**：编译设备固件（firmware）二进制文件
- **编译缓存管理**：实现增量编译，避免重复编译未变更的代码
- **代码生成**：自动生成内核描述符、数据格式配置等中间文件
- **多处理器支持**：支持 Tensix、Ethernet 等多种核心类型的编译

### 1.2 在系统中的位置

```
tt_metal/
├── jit_build/          # JIT编译核心模块（本文档）
│   ├── build.cpp/hpp   # 核心编译逻辑
│   ├── build_env_manager.cpp/hpp  # 构建环境管理
│   └── ...
├── llrt/hal.hpp        # HAL硬件抽象层接口
├── impl/kernels/       # 内核实现
└── hw/                 # 硬件相关代码和固件源码
```

### 1.3 与其他模块的交互

| 模块 | 交互方式 | 说明 |
|------|----------|------|
| `llrt/hal.hpp` | HAL 查询接口 | 获取架构特定的编译参数（源文件、链接脚本、编译选项等） |
| `impl/kernels/` | 内核设置 | 接收内核特定的编译设置（defines、include路径等） |
| `impl/context/` | 设备配置 | 创建设备特定的 JIT 编译配置 |
| `llrt/rtoptions.hpp` | 运行时选项 | 获取编译器优化级别、调试选项等 |
| `common/stable_hash.hpp` | 哈希计算 | 计算编译缓存的哈希键 |

---

## 2. 目录结构

```
tt_metal/jit_build/
├── build.cpp                    # 核心编译实现（800行）
├── build.hpp                    # JitBuildEnv/JitBuildState 类定义
├── build_env_manager.cpp        # 构建环境管理器实现
├── build_env_manager.hpp        # BuildEnvManager 单例类
├── jit_device_config.cpp        # 设备配置工厂
├── jit_device_config.hpp        # JitDeviceConfig 结构体
├── jit_build_settings.hpp       # JitBuildSettings 抽象基类
├── jit_build_options.cpp        # 构建选项实现
├── jit_build_options.hpp        # JitBuildOptions 类
├── jit_build_cache.cpp          # 编译缓存实现
├── jit_build_cache.hpp          # JitBuildCache 单例类
├── jit_build_utils.cpp          # 工具函数（命令执行、文件操作）
├── jit_build_utils.hpp          # FileRenamer RAII 类
├── genfiles.cpp                 # 代码生成实现（540行）
├── genfiles.hpp                 # 生成文件接口
├── data_format.cpp              # 数据格式处理
├── data_format.hpp              # 数据格式转换函数
├── hlk_desc.hpp                 # HLK（High-Level Kernel）描述符
├── depend.cpp                   # 依赖追踪实现
├── depend.hpp                   # 依赖解析接口
├── kernel_args.cpp              # 内核参数日志
├── kernel_args.hpp              # 参数日志接口
├── precompiled.cpp              # 预编译固件支持
├── precompiled.hpp              # 预编译固件接口
├── fake_kernels_target/         # 伪内核目标（测试用）
│   └── fake_jit_prelude.h       # 硬编码的描述符值
├── CMakeLists.txt               # CMake 构建配置
└── sources.cmake                # 源文件列表
```

---

## 3. 核心组件解析

### 3.1 JitBuildEnv - 构建环境

`JitBuildEnv` 类封装了设备特定的**全局编译环境**，包括：

```cpp
class JitBuildEnv {
    // 路径配置
    std::string root_;              // tt-metal 源码根目录
    std::string out_root_;          // 编译输出根目录（默认 ~/.cache/tt-metal-cache/）
    std::string out_firmware_root_; // 固件输出目录
    std::string out_kernel_root_;   // 内核输出目录

    // 编译工具
    std::string gpp_;               // 编译器路径（riscv-tt-elf-g++）
    std::string gpp_include_dir_;   // SFPI 包含目录

    // 编译选项
    std::string cflags_;            // 编译器标志
    std::string defines_;           // 预处理器定义
    std::string includes_;          // 包含路径
    std::string lflags_;            // 链接器标志

    uint64_t build_key_;            // 构建缓存键（哈希值）
};
```

**关键方法**：
- `init()`：根据设备配置初始化编译环境，计算 build_key
- `get_build_key()`：返回用于缓存的唯一标识符

### 3.2 JitBuildState - 构建状态

`JitBuildState` 类表示**单个编译目标**的完整状态（固件或内核）：

```cpp
class alignas(CACHE_LINE_ALIGNMENT) JitBuildState {
    const JitBuildEnv& env_;        // 关联的构建环境

    // 目标配置
    bool is_fw_;                    // 是否为固件编译
    std::string target_name_;       // 目标名称（如 "brisc"、"ncrisc"）
    std::string out_path_;          // 输出路径

    // 编译参数
    std::string cflags_, defines_, includes_, lflags_;
    std::string linker_script_;     // 链接器脚本路径

    // 源文件和对象文件
    vector_cache_aligned<std::string> srcs_;   // 源文件列表
    vector_cache_aligned<std::string> objs_;   // 对象文件列表

    uint64_t build_state_hash_;     // 编译状态哈希（用于增量编译）
};
```

**核心方法**：
- `build()`：执行完整编译流程（编译 + 链接 + 可选的 weaken 步骤）
- `compile()`：并行编译所有源文件
- `link()`：链接对象文件生成 ELF
- `weaken()`：弱化符号，使固件可作为库被内核链接

### 3.3 BuildEnvManager - 构建环境管理器

**单例模式**管理所有设备的构建环境：

```cpp
class BuildEnvManager {
    std::unordered_map<ChipId, DeviceBuildEnv> device_id_to_build_env_;

    // 处理器到构建状态的映射
    ProgCoreMapping kernel_build_state_indices_;
    ProgCoreMapping firmware_build_state_indices_;
};

struct DeviceBuildEnv {
    JitBuildEnv build_env;
    std::vector<JitBuildState> firmware_build_states;
    std::vector<JitBuildState> kernel_build_states;
    bool firmware_precompiled = false;
};
```

**关键功能**：
- `add_build_env()`：为新设备创建构建环境
- `get_firmware_build_state()` / `get_kernel_build_state()`：获取特定处理器的构建状态
- `build_firmware()`：触发固件编译（支持预编译固件复用）

### 3.4 JitBuildCache - 编译缓存

**线程安全的编译去重机制**：

```cpp
class JitBuildCache {
    std::unordered_map<size_t, State> entries_;  // hash -> Building/Built
    std::mutex mutex_;
    std::condition_variable cv_;

    enum class State { Building, Built };
};
```

**工作原理**：
1. 相同哈希的并发编译请求，只有一个线程执行编译
2. 其他线程等待编译完成（通过 condition_variable）
3. 编译失败时移除条目，允许后续重试

### 3.5 JitBuildSettings - 内核设置接口

**抽象基类**，由上层内核实现提供编译细节：

```cpp
class JitBuildSettings {
public:
    virtual const std::string& get_full_kernel_name() const = 0;
    virtual std::string_view get_compiler_opt_level() const = 0;
    virtual void process_defines(std::function<void(const std::string&, const std::string&)>) const = 0;
    virtual void process_compile_time_args(std::function<void(const std::vector<uint32_t>&)>) const = 0;
    virtual void process_named_compile_time_args(...) const = 0;
    virtual void process_include_paths(const std::function<void(const std::string&)>&) const = 0;
};
```

### 3.6 HalJitBuildQueryInterface - HAL 查询接口

HAL 提供的**架构特定编译参数查询接口**：

```cpp
class HalJitBuildQueryInterface {
public:
    struct Params {
        bool is_fw;
        HalProgrammableCoreType core_type;
        HalProcessorClassType processor_class;
        uint32_t processor_id;
        const llrt::RunTimeOptions& rtoptions;
    };

    virtual std::vector<std::string> link_objs(const Params& params) const = 0;
    virtual std::vector<std::string> includes(const Params& params) const = 0;
    virtual std::vector<std::string> defines(const Params& params) const = 0;
    virtual std::vector<std::string> srcs(const Params& params) const = 0;
    virtual std::string common_flags(const Params& params) const = 0;
    virtual std::string linker_script(const Params& params) const = 0;
    virtual std::string linker_flags(const Params& params) const = 0;
    virtual bool firmware_is_kernel_object(const Params& params) const = 0;
    virtual std::string target_name(const Params& params) const = 0;
    virtual std::string weakened_firmware_target_name(const Params& params) const = 0;
};
```

---

## 4. 编译流程

### 4.1 固件编译流程

```
build_firmware(device_id)
    │
    ├─→ 检查预编译固件（precompiled::find_precompiled_dir）
    │   └─→ 存在则直接使用，跳过编译
    │
    └─→ jit_build_once(build_key, lambda)
        │
        └─→ jit_build_subset(firmware_build_states)
            │
            └─→ 对每个 JitBuildState 并行执行 build(nullptr)
                │
                ├─→ compile()     # 编译源文件
                ├─→ link()        # 链接生成 ELF
                ├─→ weaken()      # 弱化符号（固件特殊处理）
                └─→ write_build_state_hash()  # 写入状态哈希
```

### 4.2 内核编译流程

```
jit_build(build_state, settings)
    │
    └─→ build.build(settings)
        │
        ├─→ 创建输出目录
        ├─→ build_state_matches()   # 检查编译状态是否匹配
        │
        ├─→ compile()               # 编译阶段
        │   ├─→ need_compile()      # 检查是否需要编译（缓存判断）
        │   ├─→ compile_one()       # 单文件编译
        │   │   ├─→ 构造 g++ 命令行
        │   │   ├─→ 处理用户 defines
        │   │   ├─→ 处理编译时参数
        │   │   ├─→ run_command()   # 执行编译
        │   │   └─→ write_dependency_hashes()  # 写入依赖哈希
        │   └─→ sync_build_steps()  # 等待所有编译完成
        │
        ├─→ link()                  # 链接阶段
        │   ├─→ need_link()         # 检查是否需要链接
        │   ├─→ 构造链接命令行
        │   ├─→ 添加弱化固件依赖
        │   └─→ run_command()       # 执行链接
        │
        ├─→ weaken() [仅固件]       # 弱化符号供内核链接
        ├─→ write_build_state_hash() # 写入构建状态
        └─→ extract_zone_src_locations() [仅 profiler]
```

### 4.3 代码生成流程（genfiles）

```
jit_build_genfiles_kernel_include()
    └─→ 生成 kernel_includes.hpp（包含用户内核源码）

jit_build_genfiles_triscs_src()
    ├─→ 检测内核语法类型（简化语法 vs 传统语法）
    ├─→ 生成 chlkc_unpack.cpp（UNPACK TRISC）
    ├─→ 生成 chlkc_math.cpp（MATH TRISC）
    ├─→ 生成 chlkc_pack.cpp（PACK TRISC）
    ├─→ 生成 chlkc_isolate_sfpu.cpp（SFPU TRISC）
    └─→ 生成 defines_generated.h（用户 defines）

jit_build_genfiles_descriptors()
    └─→ 生成 chlkc_descriptors.h
        ├─→ 计算数据格式（unpack/pack src/dst）
        ├─→ 计算 tile 维度
        ├─→ 生成数学精度配置
        └─→ 生成目标累积模式配置
```

### 4.4 依赖追踪流程

```
编译时：
    g++ -MMD -c ...           # 生成 .d 依赖文件
        ↓
    write_dependency_hashes()  # 解析 .d 文件，计算依赖哈希
        ↓
    写入 .dephash 文件

下次编译时：
    need_compile()
        ↓
    dependencies_up_to_date()  # 读取 .dephash
        ↓
    比较当前依赖文件哈希与存储值
        ↓
    哈希一致 → 跳过编译（缓存命中）
    哈希不一致 → 重新编译
```

---

## 5. 设计模式与实现技巧

### 5.1 单例模式

```cpp
// BuildEnvManager 单例
static BuildEnvManager& get_instance() {
    static BuildEnvManager instance(MetalContext::instance().hal());
    return instance;
}

// JitBuildCache 单例
static JitBuildCache& inst() {
    static JitBuildCache instance;
    return instance;
}
```

### 5.2 RAII 文件操作（FileRenamer）

**解决多进程并发写入问题**：

```cpp
class FileRenamer {
    std::string temp_path_;   // 临时文件名（含随机ID）
    std::string target_path_; // 目标文件名

public:
    FileRenamer(const std::string& target_path);
    ~FileRenamer() {
        // 析构时原子重命名
        std::filesystem::rename(temp_path_, target_path_);
    }
};

// 使用示例
{
    FileRenamer tmp("output.o");
    std::ofstream file(tmp.path());  // 写入临时文件
    // 析构时自动重命名为 output.o
}
```

### 5.3 缓存对齐的容器

```cpp
static constexpr uint32_t CACHE_LINE_ALIGNMENT = 64;

template <typename T>
using vector_cache_aligned = std::vector<T, tt::stl::aligned_allocator<T, CACHE_LINE_ALIGNMENT>>;

// 用于存储源文件和对象文件路径，优化多线程访问
vector_cache_aligned<std::string> srcs_;
vector_cache_aligned<std::string> objs_;
```

### 5.4 构建状态哈希

**防止编译选项变更后使用陈旧缓存**：

```cpp
// 计算所有影响编译的参数哈希
FNV1a hasher;
hasher.update(env_.gpp_);
hasher.update(cflags_);
hasher.update(defines_);
hasher.update(includes_);
hasher.update(lflags_);
hasher.update(linker_script_);
hasher.update(extra_link_objs_);
for (const auto& src : srcs_) { hasher.update(src); }
build_state_hash_ = hasher.digest();

// 编译前后比较哈希，不匹配则强制重新编译
bool state_changed = !build_state_matches(out_dir);
```

### 5.5 多处理器共享编译（jit_build_for_processors）

**多个处理器共享相同源码时的优化**：

```cpp
void jit_build_for_processors(std::span<const JitBuildState* const> targets,
                              const JitBuildSettings* settings) {
    // 第一个目标执行编译
    const JitBuildState& primary = *targets[0];
    primary.build(settings, targets);  // 编译一次，链接多次

    // 所有目标（包括第一个）分别链接
    // 生成各自的处理特定 ELF
}
```

### 5.6 简化内核语法转换

**支持用户友好的 kernel_main() 语法**：

```cpp
// 用户编写的简化语法
void kernel_main() {
    // 内核代码
}

// 自动转换为传统语法（针对 TRISC）
namespace chlkc_unpack {
    void unpack_main() { /* 原 kernel_main 内容 */ }
}
namespace chlkc_math {
    void math_main() { /* 原 kernel_main 内容 */ }
}
namespace chlkc_pack {
    void pack_main() { /* 原 kernel_main 内容 */ }
}
```

### 5.7 条件编译与预编译固件

```cpp
void add_build_env_locked(...) {
    // 检查是否存在预编译固件
    auto precompiled_dir = precompiled::find_precompiled_dir(...);
    if (precompiled_dir.has_value()) {
        dev_build_env.build_env.set_firmware_binary_root(*precompiled_dir);
        dev_build_env.firmware_precompiled = true;
    }
}

void build_firmware(ChipId device_id, bool ignore_precompiled) {
    if (!ignore_precompiled && build_env.firmware_precompiled) {
        // 直接使用预编译固件，跳过编译
        return;
    }
    // 执行 JIT 编译
}
```

---

## 6. 源码注释摘录

### 6.1 build.hpp

```cpp
// The build environment
// Includes the path to the src/output and global defines, flags, etc
// Device specific
class JitBuildEnv {
    // ...
};

// All the state used for a build in an abstract base class
// Contains everything needed to do a build (all settings, methods, etc)
class alignas(CACHE_LINE_ALIGNMENT) JitBuildState {
    // ...
};

// Execute build_fn exactly once for a given hash.
// Concurrent callers with the same hash block until the build completes.
// Returns immediately if hash was already built.
// If build_fn throws, subsequent callers will retry.
void jit_build_once(size_t hash, const std::function<void()>& build_fn);
```

### 6.2 build.cpp

```cpp
// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B.
// B imports A's weakened symbols, B's symbols of the same name don't
// result in duplicate symbols but B can reference A's symbols.
// Force the fw_export symbols to remain strong so to propagate link addresses
void JitBuildState::weaken(const string& out_dir) const {
    std::string pathname_in = out_dir + target_name_ + ".elf";
    jit_build::utils::FileRenamer out_file(this->weakened_firmware_name_);

    ll_api::ElfFile elf;
    elf.ReadImage(pathname_in);
    static const std::string_view strong_names[] = {"__fw_export_*", "__global_pointer$"};
    elf.WeakenDataSymbols(strong_names);
    if (this->firmware_is_kernel_object_) {
        elf.ObjectifyExecutable();
    }
    elf.WriteImage(out_file.path());
}
```

### 6.3 jit_build_cache.hpp

```cpp
// Thread-safe build-once cache for JIT compilation.
//
// Ensures that for a given hash (representing a build target), the build function
// is executed exactly once. Concurrent callers with the same hash block until the
// build completes. Callers arriving after the build is done return immediately.
//
// Used to deduplicate both kernel and firmware JIT builds across threads.
class JitBuildCache {
    // ...
};
```

### 6.4 genfiles.cpp

```cpp
// Simple kernel syntax refers to declaring kernel entry point as just "void kernel_main()"
// This is in contrast to legacy syntax: "namespace NAMESPACE { void MAIN() { ... } }"
// Eventually we may want to deprecate legacy syntax, but for now we support both.
namespace simple_kernel_syntax {
    const std::regex kernel_main_pattern(R"(\bvoid\s+kernel_main\s*\(\s*\)\s*\{)");

    // Transforms simplified kernel to legacy format:
    //   - Splits at "void kernel_main()"
    //   - Preamble (#includes) stays outside namespace
    //   - Function body wrapped in namespace, renamed to func_name
    string transform_to_legacy_syntax(const string& source, const char* ns_name, const char* func_name);
}
```

### 6.5 jit_device_config.hpp

```cpp
// Device-specific configuration snapshot consumed by the JIT build system.
//
// Captures the hardware topology, dispatch layout, and memory parameters that
// influence kernel compilation. Instances are intended to be created once per
// device via `create_jit_device_config` and then treated as read-only; the
// build pipeline uses these values to produce compiler defines and to compute
// a cache key that uniquely identifies a build configuration.
struct JitDeviceConfig {
    const Hal* hal = nullptr;
    tt::ARCH arch = tt::ARCH::Invalid;
    size_t num_dram_banks = 0;
    size_t num_l1_banks = 0;
    CoreCoord pcie_core{0, 0};
    uint32_t harvesting_mask = 0;
    // ...
};
```

### 6.6 depend.hpp

```cpp
// Parses a Makefile-style dependency file (generated by gcc -MMD)
ParsedDependencies parse_dependency_file(std::istream& file);

// Returns true if all dependencies' hashes match those stored in the .hash file.
bool dependencies_up_to_date(const std::string& out_dir, const std::string& obj);
```

### 6.7 build_env_manager.hpp

```cpp
// Singleton class to generate and hold build environments, build keys, and build states.
class BuildEnvManager {
public:
    // Add a new build environment for the corresponding device id and num_hw_cqs.
    // Also generates the build key and build states.
    // This requires a live device to be available at device_id.
    void add_build_env(ChipId device_id, uint8_t num_hw_cqs);

    // Build firmware for the device
    void build_firmware(ChipId device_id, bool ignore_precompiled = false);
};
```

---

## 7. 关键数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              JIT Build 数据流                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  用户内核代码                    编译流程                      输出产物
  ┌──────────────┐              ┌──────────────┐            ┌──────────────┐
  │ kernel.cpp   │─────────────→│ 语法检测/转换 │───────────→│ chlkc_*.cpp  │
  └──────────────┘              └──────────────┘            └──────────────┘
                                        │                          │
                                        ↓                          ↓
                              ┌──────────────┐            ┌──────────────┐
                              │ JitBuildEnv  │            │   g++ 编译    │
                              │  (环境配置)   │───────────→│  -c 编译选项  │
                              └──────────────┘            └──────────────┘
                                     │                              │
                                     ↓                              ↓
                              ┌──────────────┐            ┌──────────────┐
                              │JitBuildState │            │    .o 文件   │
                              │  (编译状态)   │            └──────────────┘
                              └──────────────┘                     │
                                     │                             │
                                     ↓                             ↓
                              ┌──────────────┐            ┌──────────────┐
                              │   compile()  │            │    link()    │
                              │  (并行编译)   │───────────→│  (链接ELF)   │
                              └──────────────┘            └──────────────┘
                                                                     │
                                                                     ↓
                                                              ┌──────────────┐
                                                              │  kernel.elf  │
                                                              │  (可执行文件) │
                                                              └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              缓存机制                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  源文件/头文件              依赖追踪                    缓存判断
  ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
  │   *.cpp      │────────→│ g++ -MMD     │─────────→│   .d 文件    │
  │   *.hpp      │         │ (生成依赖)    │          └──────────────┘
  └──────────────┘         └──────────────┘                 │
                                                            ↓
                                                     ┌──────────────┐
                                                     │parse_dependency
                                                     │   _file()    │
                                                     └──────────────┘
                                                            │
                                                            ↓
                                                     ┌──────────────┐
                                                     │ 计算文件哈希  │
                                                     │ (FNV1a)      │
                                                     └──────────────┘
                                                            │
                                                            ↓
                                                     ┌──────────────┐
                                                     │  .dephash    │←──── 下次编译对比
                                                     │ (存储哈希)    │      哈希值
                                                     └──────────────┘
```

---

## 8. 总结

`jit_build/` 模块是 TT-Metal 框架中负责**运行时内核编译**的核心组件，其设计亮点包括：

1. **分层架构**：JitBuildEnv（全局环境）→ JitBuildState（单个目标）→ BuildEnvManager（多设备管理）
2. **高效缓存**：多级缓存机制（build_key、build_state_hash、dependency_hash）避免重复编译
3. **并发安全**：JitBuildCache 确保相同目标的并发编译只执行一次
4. **架构抽象**：通过 HAL 查询接口支持不同芯片架构（Grayskull、Wormhole、Blackhole）
5. **语法兼容**：支持简化的 `kernel_main()` 语法和传统命名空间语法
6. **依赖追踪**：基于 GCC -MMD 的精确依赖追踪，实现真正的增量编译
