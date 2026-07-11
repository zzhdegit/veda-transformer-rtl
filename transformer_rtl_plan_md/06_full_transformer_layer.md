# Stage 6：完整 Transformer Layer

## 1. 阶段目标

在多头 Attention 基础上加入：

```text
Norm
Residual
FFN1
Activation
FFN2
第二次 Residual/Norm
```

形成可端到端验证和进行完整 P&R 的 Transformer Layer。

## 2. 结构冻结

进入本阶段前必须明确最终层结构。

### 方案 A：Pre-Norm 标准 Transformer

```text
x
→ Norm
→ MHA
→ Residual Add
→ Norm
→ FFN1
→ Activation
→ FFN2
→ Residual Add
```

### 方案 B：Post-Norm

```text
x
→ MHA
→ Residual Add
→ Norm
→ FFN
→ Residual Add
→ Norm
```

不得在 RTL 编写过程中随意切换。

## 3. 顶层模块

```text
transformer_layer
├── layer_scheduler
├── norm_engine
├── multihead_attention_block
├── residual_buffer
├── ffn_controller
├── activation_engine
├── shared_gemv_core
├── layer_input_buffer
└── layer_output_buffer
```

## 4. Norm Engine

参考论文的 reduction + normalization 思路。

### LayerNorm

Reduction：

```text
sum_x
sum_x2
mean
variance
```

Normalization：

```text
y_i = gamma_i * (x_i - mean) / sqrt(variance + eps) + beta_i
```

### RMSNorm

Reduction：

```text
sum_x2
rms
```

Normalization：

```text
y_i = gamma_i * x_i / sqrt(mean(x^2) + eps)
```

RMSNorm 更简单，若目标偏 Llama 风格可优先采用。

## 5. Norm 数据流

建议：

```text
第一遍读取 x
→ 计算 sum / square sum
→ 保存 x
→ 得到 mean/variance 或 rms
→ 第二遍读取 x
→ normalization
```

需要输入 buffer，除非允许重新读取上一级输出。

## 6. Residual

Residual 加法需要明确：

- 输入保存位置；
- 位宽；
- 加法顺序；
- 饱和/舍入；
- 与 MHA/FFN 输出对齐；
- 是否支持 bypass。

Residual buffer 不应默认用大量触发器，应使用 SRAM 或 banked register file。

## 7. FFN

标准 FFN：

```text
h = activation(x W1)
y = h W2
```

复用 shared GEMV core。

若采用 SwiGLU：

```text
a = x W_gate
b = x W_up
h = SiLU(a) * b
y = h W_down
```

第一版建议使用更简单的 ReLU 或 GELU，除非模型结构已经冻结为 Llama 风格。

## 8. Activation Engine

候选：

- ReLU：最简单；
- GELU：需要近似；
- SiLU：需要 sigmoid/exp；
- SwiGLU：需要额外投影和逐元素乘。

任何近似函数必须：

- 有 bit 模型；
- 有最大误差；
- 有平均误差；
- 有综合 PPA 报告。

## 9. Layer Scheduler

建议命令序列写成微程序或清晰 FSM：

```text
LOAD_INPUT
NORM1_REDUCE
NORM1_APPLY
QKV_PROJ
HEAD_LOOP
OUTPUT_PROJ
RESIDUAL1
NORM2_REDUCE
NORM2_APPLY
FFN1
ACTIVATION
FFN2
RESIDUAL2
WRITE_OUTPUT
DONE
```

若采用不同 Norm 位置，状态顺序相应调整。

必须避免一个巨型组合 FSM 控制所有细节。建议层级控制：

```text
layer scheduler
→ block controller
→ local datapath controller
```

## 10. 端到端验证

从小型模型开始：

```text
D_MODEL   = 32
NUM_HEADS = 4
D_HEAD    = 8
FFN_DIM   = 128
SEQ_LEN   = 小规模
```

比较：

```text
Norm1 output
Q/K/V
每个 head output
MHA output
Residual1 output
Norm2 output
FFN1 output
Activation output
FFN2 output
Final output
```

不能只比较最终输出，否则难以定位误差。

## 11. 性能计数

输出：

```text
norm1_cycles
qkv_cycles
attention_cycles
output_projection_cycles
residual1_cycles
norm2_cycles
ffn1_cycles
activation_cycles
ffn2_cycles
residual2_cycles
total_layer_cycles
shared_pe_utilization
sfu_utilization
memory_stall_cycles
weight_stall_cycles
```

## 12. 第三次完整 P&R

本阶段完成完整 Transformer Layer P&R。

必须使用：

- 真实或代表性 SRAM 宏；
- 冻结参数；
- 正式 SDC；
- IO delay；
- clock uncertainty；
- multi-corner 条件；
- SAIF/VCD 活动率。

检查：

```text
floorplan
macro placement
placement congestion
CTS skew
route congestion
post-route STA
hold fixing
power
IR/EM 条件若工具链支持
```

## 13. PPA 优化顺序

出现问题时按以下顺序优化：

1. 流水级；
2. 高扇出控制；
3. memory banking；
4. 数据复用；
5. clock enable；
6. 算术近似；
7. 降低并行度；
8. 调整目标频率。

不要先通过大幅降低频率掩盖结构问题。

## 14. Stage 6 交付物

```text
rtl/transformer/layer/*
rtl/transformer/norm/*
rtl/transformer/ffn/*
tb/integration/transformer_layer/*
model/bit_model/transformer_layer.py
model/cycle_model/transformer_layer.py
scripts/pnr/transformer_layer/*
reports/stage6/*
docs/stage6_report.md
```

## 15. 退出条件

- 完整层端到端通过；
- 各中间节点可追踪；
- 无死锁和数据错位；
- cycle model 与 RTL 周期一致；
- post-route WNS ≥ 0；
- hold violation 清零；
- 面积、功耗、时序、带宽和利用率报告完整；
- 能明确说明最大瓶颈及其原因。
