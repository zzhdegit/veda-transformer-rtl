# Stage 7：最终物理实现、门级验证与签核

## 1. 阶段目标

冻结最终 RTL 和参数，完成正式综合、STA、功耗分析、P&R、门级验证和可交付报告。

本阶段原则上不再新增架构功能。

## 2. 冻结内容

进入本阶段前冻结：

```text
RTL commit
parameter set
memory macro set
clock target
IO constraints
operating corners
power intent
test mode
reset strategy
```

任何功能修改都必须重新经过 Stage 6 回归。

## 3. 正式综合

使用 Design Compiler 或 Fusion Compiler 完成：

```text
elaboration
link
uniquify
check_design
compile
clock gating insertion
DFT-related preparation if required
write netlist
write SDC
write reports
```

必须检查：

- unconstrained paths；
- inferred latch；
- multi-driven net；
- combinational loop；
- excessive fanout；
- ungroup/flatten 策略；
- clock gating coverage；
- memory black box 一致性。

## 4. Formal Equivalence

若许可证支持，执行：

```text
RTL ↔ post-synthesis netlist
post-synthesis ↔ post-route netlist
```

任何不等价必须关闭后才能进入最终签核。

## 5. Floorplan

重点：

- PE 阵列规整布局；
- SRAM 宏靠近主要消费者；
- K/V SRAM 与 PE 的带宽路径；
- score buffer 靠近 SFU；
- clock trunk；
- power ring/strap；
- congestion margin；
- 宏间 channel；
- pin placement。

禁止只依赖自动 floorplan 后直接 route。

## 6. Placement 与优化

检查：

```text
global congestion
local congestion
cell density
high fanout
buffer insertion
critical path location
clock enable placement
adder tree placement
```

对于广播信号和 mode 控制，可使用局部复制寄存器。

## 7. CTS

目标：

```text
skew
insertion delay
clock transition
clock power
gating checks
```

需检查所有 generated clock、gated clock 和 test clock。

## 8. Routing

完成：

```text
global route
detailed route
antenna fix
crosstalk-aware optimization
SI-aware timing
```

关注：

- PE 数据总线；
- SRAM 宏边缘拥塞；
- global reset；
- broadcast scalar；
- mode control；
- long ready/valid path。

## 9. STA

至少覆盖：

```text
setup corners
hold corners
max transition
max capacitance
max fanout
clock gating checks
recovery/removal
min pulse width
```

要求：

```text
WNS >= 0
TNS = 0
hold violations = 0
unconstrained paths = 0
```

不得只看单一 typical corner。

## 10. 功耗分析

必须使用真实活动率。

建议场景：

1. QKV Projection；
2. 短序列 Attention；
3. 长序列 Attention；
4. FFN；
5. 完整 Layer；
6. Idle；
7. 最大吞吐。

报告：

```text
dynamic power
leakage power
clock power
memory power
PE power
SFU power
control power
power per token
energy per layer
```

若工具链支持，加入 vectorless 与 vector-based 对照。

## 11. 门级仿真

至少执行：

```text
post-synthesis GLS
post-route SDF GLS
```

覆盖：

- reset；
- 单次 operation；
- 连续 token；
- backpressure；
- mode switching；
- 最大长度；
- error path；
- SRAM 时序模型。

对于大型设计，可使用小型参数配置做完整 SDF，最终配置做关键路径和代表性场景。

## 12. DRC/LVS 与版图检查

若本地 PDK 和工具许可支持：

```text
DRC
LVS
ERC
antenna
density
metal fill
```

若无法完成 foundry signoff，应在报告中明确“完成到哪一级”，不能声称完整流片签核。

## 13. 最终性能报告

必须包含：

```text
technology
voltage
temperature
clock
area
macro area
standard-cell area
frequency
WNS/TNS
dynamic power
leakage power
energy per token
latency per token
throughput
PE utilization
memory bandwidth
critical path
congestion summary
```

同时给出不同序列长度下：

```text
latency
energy
bandwidth
utilization
```

## 14. 最终对比

至少对比：

- 纯 inner-product 基线；
- flexible inner/outer dataflow；
- 无 element-serial overlap；
- 有 element-serial overlap；
- 不同 PE 数量；
- 不同 softmax 实现；
- pre-route 与 post-route PPA。

Voting 未实现，不得在结论中声称完整复现 VEDA。

准确表述：

```text
基于 VEDA flexible-product dataflow 与 element-serial scheduling
实现的 Transformer RTL 加速器。
```

## 15. 最终交付物

```text
rtl_release/
netlist/
sdc/
lef_def_gds/  # 视工具链许可
sdf/
reports/final/
waveforms/
scripts/reproducible_flow/
docs/final_report.md
docs/reproduction_guide.md
```

## 16. 最终完成条件

- 所有 regression 通过；
- RTL/netlist 等价；
- setup/hold 全部关闭；
- 无未约束路径；
- 功耗使用真实活动率；
- post-route 结果可复现；
- 门级关键场景通过；
- PPA 数据完整；
- 结论不超出实际实现范围；
- 任意第三方可按 reproduction guide 重跑主要结果。
