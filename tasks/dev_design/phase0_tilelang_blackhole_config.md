# Phase 0 TileLang + Blackhole 配置

## 文档定位

- **状态**: 历史记录
- **注意**: 本文档中的早期架构设想已经被 `final_blackhole_backend_redesign.md` 取代

## 保留原因

本文件只保留一个历史事实：

- Blackhole 后端最初是以“新增 codegen / runtime 模块框架、先把 target 接进 TileLang”为目标启动的

## 当前不再采用的早期结论

以下内容已不再作为当前设计依据：

- 以单个 kernel 字符串为后端主产物
- 让 `SplitBlackholeKernel` 成为主路径前置
- 把 Blackhole 后端视为绕开 TileLang / TVM 主线 pass 的独立 target pipeline

## 当前应参考

- 总体架构：`final_blackhole_backend_redesign.md`
- 当前阶段：`stage2_pass_reuse_matrix.md`
