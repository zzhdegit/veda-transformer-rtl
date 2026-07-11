# Stage 0：规格冻结、参考模型与验收体系

> 2026-07-11 audit note: this file is the original broad Stage 0 plan.
> The executable Stage 0 implementation contract for the current project line
> is `docs/stage_00_spec.md`. The bit-accurate model, cycle model, and PPA
> budget items listed below remain required for later closure work, but are
> explicitly deferred to Stage 1+ by the current scoped Stage 0 spec and are
> not claimed as completed Stage 0 artifacts.

## 1. 阶段目标

在开始大规模 RTL 编写前，冻结数学定义、数值格式、模块接口、流水原则、存储布局和验证方法。

本阶段不实现大规模硬件功能。输出是后续所有阶段的唯一规格来源。

## 2. 必须冻结的模型定义

至少明确以下内容：

```text
MODEL_STYLE       = Standard MHA / Llama-like
NORM_STYLE        = LayerNorm / RMSNorm
NORM_POSITION     = Pre-Norm / Post-Norm
ACTIVATION        = GELU / ReLU / SiLU / SwiGLU
POSITION_ENCODING = None / RoPE
ATTENTION_STYLE   = MHA / GQA / MQA
BIAS_ENABLE       = 0 / 1
```

第一版建议：

```text
MODEL_STYLE       = Standard MHA
NORM_STYLE        = LayerNorm 或 RMSNorm，二选一
NORM_POSITION     = Pre-Norm
ACTIVATION        = GELU 或 ReLU
POSITION_ENCODING = None
ATTENTION_STYLE   = MHA
BIAS_ENABLE       = 0
```

后续扩展到 Llama 风格时，再加入 RoPE、RMSNorm、GQA、SwiGLU。

## 3. 必须冻结的硬件参数

参数应分为 elaboration-time parameter 和 runtime configuration。

建议 elaboration-time：

```text
DATA_W
ACC_W
EXP_W
D_HEAD
PE_ROWS
PE_COLS
PE_BANKS
MAX_SEQ_LEN
MAX_D_MODEL
MAX_FFN_DIM
```

建议 runtime：

```text
active_seq_len
active_head_num
active_d_model
active_ffn_dim
operation_mode
base_address
stride
```

第一版建议建立小型验证配置与最终综合配置：

### 小型验证配置

```text
D_MODEL     = 32
NUM_HEADS   = 4
D_HEAD      = 8
FFN_DIM     = 128
MAX_SEQ_LEN = 32 或 64
```

### 目标配置

```text
D_HEAD      = 128
MAX_SEQ_LEN = 4096
PE_NUM      = 128，或采用更小 PE 数量进行 tiling
```

最终参数由可用工艺库、SRAM 宏和设计周期决定。

## 4. 数值格式决策

必须明确：

- 输入和权重格式；
- 乘法输出格式；
- 累加器格式；
- softmax score 格式；
- exponent 输入输出格式；
- reciprocal/division 格式；
- Norm 中平方和格式；
- residual 累加格式；
- 舍入模式；
- 饱和模式；
- 非规格数、NaN、Inf 的处理策略。

推荐建立统一的数值规格表：

| 信号 | 格式 | 舍入 | 饱和 | 备注 |
|---|---|---|---|---|
| activation | 待定 | 待定 | 待定 | 输入激活 |
| weight | 待定 | 待定 | 待定 | 权重 |
| MAC product | 待定 | 待定 | 待定 | 乘法结果 |
| MAC accumulator | 待定 | 待定 | 待定 | 点积累加 |
| attention score | 待定 | 待定 | 待定 | 缩放后 |
| exp result | 待定 | 待定 | 待定 | softmax |
| softmax probability | 待定 | 待定 | 待定 | 送入 s'V |

若采用 FP16 输入，建议累加器至少采用更高精度格式；不要默认 FP16 直接长链累加能够满足误差要求。

## 5. Attention 数学定义

生成阶段单头 Attention：

```text
q: [1, d]
K: [l, d]
V: [l, d]
```

```text
score_i = dot(q, K_i)
scaled_score_i = score_i / sqrt(d)
p_i = exp(scaled_score_i - max_score) / exp_sum
o = sum_i p_i * V_i
```

必须明确当前 token 的 K/V 是在 Attention 前还是后加入 Cache。

推荐：

```text
生成当前 q/k/v
→ append 当前 k/v
→ 当前 q 对更新后的 K/V 做 causal attention
```

## 6. Softmax 在线归约模型

在线最大值和指数和：

```text
m_new = max(m_old, x)
z_new = z_old * exp(m_old - m_new) + exp(x - m_new)
```

最终 normalization：

```text
p_i = exp(x_i - m_final) / z_final
```

score 必须保存到 score buffer，除非后续明确实现重计算方案。

## 7. 接口规范

至少定义：

```text
stream_data
stream_valid
stream_ready
stream_last
token_id
head_id
dimension_id
operation_id
error_flag
```

命令接口至少包括：

```text
op_mode
input_base
weight_base
output_base
length_k
length_n
clear_acc
start
done
```

接口文档必须明确：

- valid 保持规则；
- ready 拉低时数据是否保持；
- last 的语义；
- reset 后状态；
- flush 行为；
- 跨 head、跨 token 元数据对齐规则。

## 8. 参考模型

必须建立三个模型：

### 8.1 浮点模型

用于验证算法正确性，不模拟硬件舍入。

### 8.2 Bit-accurate 模型

必须与 RTL 使用完全一致的：

- 位宽；
- 舍入；
- 截断；
- 饱和；
- exponent 近似；
- reciprocal 近似；
- accumulator 顺序。

### 8.3 Cycle model

模拟：

- 每级 pipeline latency；
- FIFO 深度；
- ready/valid；
- PE 模式切换；
- memory latency；
- head 切换；
- 各阶段 bubble。

## 9. 性能预算

建立初始预算表：

| 指标 | 预算 |
|---|---|
| 目标时钟 | 根据工艺库确定 |
| 单头 Attention 最大周期 | 待定 |
| PE 利用率 | 待定 |
| Score FIFO 深度 | 至少覆盖最大活动窗口或 tile |
| SRAM 带宽 | 根据 PE 每周期吞吐反推 |
| 面积上限 | 待定 |
| 动态功耗上限 | 待定 |
| WNS | ≥ 0 |
| TNS | = 0 |

## 10. Stage 0 交付物

```text
docs/spec.md
docs/interface.md
docs/microarchitecture.md
docs/verification.md
model/float_model/
model/bit_model/
model/cycle_model/
```

## 11. 退出条件

只有满足以下条件才能进入 Stage 1：

- 模型结构已冻结；
- 数值格式已冻结；
- Attention 插入 KV 的顺序已冻结；
- ready/valid 规则已冻结；
- 小型配置能在浮点模型中跑通；
- bit 模型能输出测试向量；
- cycle model 给出初始周期估计；
- PPA 预算已形成表格；
- 所有待决策项均有 owner 和截止时间。
