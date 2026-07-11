# Transformer RTL 分阶段实现总说明

## 1. 项目目标

本项目目标是在本地 Synopsys EDA 工具链下，完成一个可综合、可验证、可进行版图实现的 Transformer 推理 RTL 核心。

主线范围：

1. 可重构 GEMV/PE 计算核心；
2. 单头生成阶段 Attention；
3. 动态 KV Cache；
4. 多头 Attention 与 QKV/输出投影；
5. 完整 Transformer Layer；
6. 最终综合、STA、功耗分析与版图实现。

当前主线**不实现 voting-based KV cache eviction**，但模块边界应保留未来接入 Voting Engine 的旁路接口。

参考论文：

- `2507.00797v1.pdf`
- 重点参考 Section IV：Flexible-product Dataflow
- 重点参考 Section IV-B：Element-serial Scheduling
- 重点参考 Figure 5、Figure 6、Figure 7

## 2. 阶段文件

| 文件 | 阶段 |
|---|---|
| `00_spec_and_reference_model.md` | 规格、数值模型与验收体系 |
| `01_arithmetic_and_memory_primitives.md` | 算术、FIFO、SRAM Wrapper |
| `02_reconfigurable_pe_core.md` | 可重构 PE 阵列与 GEMV 核心 |
| `03_single_head_attention.md` | 单头 Attention 完整数据通路 |
| `04_dynamic_kv_and_generation.md` | 动态 KV Cache 与连续生成 |
| `05_multihead_and_projection.md` | 多头、QKV 投影、输出投影 |
| `06_full_transformer_layer.md` | Norm、Residual、FFN、完整层 |
| `07_physical_implementation_and_signoff.md` | 最终物理实现与签核 |

阶段必须按顺序推进。每个阶段只有在满足“退出条件”后才能进入下一阶段。

## 3. 全局架构原则

### 3.1 计算资源共享

QKV Projection、`q × K^T`、`s' × V`、输出投影、FFN1、FFN2 应尽可能复用同一个 `shared_gemv_core`。

禁止在第一版中分别实例化多套大规模 MAC 阵列。

### 3.2 Attention 数据流

生成阶段单头 Attention 采用：

```text
q
→ qK^T inner-product
→ score buffer
→ softmax reduction
→ softmax normalization
→ s'V outer-product
→ head output
```

允许的并行：

```text
qK^T PE 运算
|| softmax reduction
```

以及：

```text
softmax normalization
|| s'V outer-product PE 运算
```

不允许假设同一套 PE 阵列同时执行 `qK^T` 和 `s'V`。

### 3.3 存储格式

K、V 统一采用 token-major：

```text
K_cache[token_index][dimension]
V_cache[token_index][dimension]
```

禁止为了 `qK^T` 单独保存物理转置 K。

### 3.4 数值验证

必须同时维护：

1. 浮点参考模型；
2. bit-accurate 模型；
3. RTL scoreboard；
4. 周期级性能计数器。

任何 RTL 数值差异必须能够定位到具体模块和具体舍入点。

### 3.5 流接口

模块间优先使用 ready/valid 流接口。

所有长流水模块必须支持：

- backpressure；
- flush；
- transaction 边界；
- token/head 元数据同步；
- FIFO overflow/underflow assertion。

### 3.6 PPA 控制

Stage 0 不执行 PDK 相关综合、STA、功耗或版图流程。后续 RTL 阶段在工具和库可用且阶段文件要求时执行：

```text
VCS / Verdi
Design Compiler
PrimeTime STA
基于 SAIF/VCD 的功耗估算
```

关键阶段执行 P&R：

1. Stage 2：PE + SFU 子系统；
2. Stage 4：单头 Attention + KV/Score Buffer；
3. Stage 6：完整 Transformer Layer；
4. Stage 7：最终版图和签核。

## 4. 推荐工程目录

```text
project/
├── docs/
├── model/
│   ├── float_model/
│   ├── bit_model/
│   └── cycle_model/
├── rtl/
│   ├── arithmetic/
│   ├── memory/
│   ├── pe/
│   ├── sfu/
│   ├── attention/
│   ├── transformer/
│   └── top/
├── tb/
│   ├── unit/
│   ├── block/
│   ├── integration/
│   └── vectors/
├── constraints/
├── scripts/
│   ├── vcs/
│   ├── dc/
│   ├── pt/
│   ├── power/
│   └── pnr/
├── reports/
└── Makefile
```

## 5. Codex 工作规则

Codex 每次只处理一个明确阶段或一个明确模块。

每次提交必须包含：

- RTL；
- testbench；
- assertions；
- reference vectors；
- regression command；
- synthesis script；
- timing/power report parser；
- 当前阶段的验收报告；
- 修改说明。

禁止：

- 只写 RTL 不写测试；
- 使用不可综合 `real/shortreal` 完成正式数据通路；
- 用触发器数组替代最终 SRAM 宏并据此声称面积有效；
- 隐藏 warning；
- 在没有 bit-accurate 模型时调整 RTL 数值路径；
- 在功能尚未闭环时直接进入完整顶层 P&R。

## 6. 全局完成标准

最终设计至少应给出：

- 单头和多头功能正确性；
- 连续 token 生成验证；
- 端到端 Transformer Layer 误差；
- 各阶段周期数；
- PE 利用率；
- SRAM/外存带宽；
- 综合面积；
- 时序裕量；
- 动态功耗、泄漏功耗；
- post-route 频率；
- 每 token 延迟与能耗；
- 关键路径与拥塞分析；
- 门级或 SDF 回归结果。
