# Stage 5：多头 Attention、QKV Projection 与输出投影

## 1. 阶段目标

将单头生成子系统扩展为多头 Attention，并加入：

```text
Q Projection
K Projection
V Projection
Multi-head Attention
Concat
Output Projection
```

优先使用时间复用共享 PE，不在第一版中复制多个完整 Attention 引擎。

## 2. 顶层结构

```text
multihead_attention_block
├── qkv_projection_controller
├── shared_gemv_core
├── q_buffer
├── per_head_kv_cache
├── head_scheduler
├── single_head_attention_engine
├── head_output_buffer
├── concat_mapper
├── output_projection_controller
└── block_output_buffer
```

## 3. 资源共享策略

推荐：

```text
一个 shared_gemv_core
一个 softmax SFU
一个 single_head_attention_engine
多个 head 的 K/V 地址空间
```

执行顺序：

```text
QKV Projection
→ Head 0 Attention
→ Head 1 Attention
→ ...
→ Head H-1 Attention
→ Concat
→ Output Projection
```

后续若性能不足，再评估双引擎或多引擎并行。

## 4. QKV Projection

对当前 token 输入 `x`：

```text
q_all = x W_Q
k_all = x W_K
v_all = x W_V
```

输出按 head 切分：

```text
q_h = q_all[h*d : (h+1)*d-1]
k_h = k_all[h*d : (h+1)*d-1]
v_h = v_all[h*d : (h+1)*d-1]
```

必须明确：

- 三个投影串行还是权重预取重叠；
- 权重布局；
- bias；
- 输出切分顺序；
- 每个 head K/V append 时机。

## 5. Head Scheduler

职责：

```text
选择当前 head
配置 K/V base address
装载 q_h
触发单头 Attention
保存 head output
推进 head_id
```

head 切换必须无旧 accumulator 和旧 softmax 状态残留。

## 6. Head Output Buffer 与 Concat

Concat 不应通过大规模数据移动实现。

推荐逻辑布局本身即为：

```text
head_output_buffer[head_id][dimension]
```

Output Projection 读取时，通过地址生成器按 concat 顺序流出。

## 7. Output Projection

```text
y = concat(head_outputs) W_O
```

复用 `MODE_GEMV`。

需考虑：

- 输入维度 D_MODEL；
- 输出维度 D_MODEL；
- tiling；
- 权重带宽；
- 累加器精度；
- 输出 buffer。

## 8. 小型端到端配置

优先验证：

```text
D_MODEL   = 32
NUM_HEADS = 4
D_HEAD    = 8
SEQ_LEN   = 1..32
```

软件模型计算：

```text
x
→ Q/K/V
→ per-head attention
→ concat
→ W_O
```

RTL 与 bit 模型逐级比较。

## 9. 功能验证

覆盖：

- NUM_HEADS = 1；
- NUM_HEADS > 1；
- D_MODEL 非 PE_NUM 整数倍；
- head 切换；
- head 输出顺序；
- 不同 head 使用不同 K/V；
- 不同 head 序列长度一致性；
- QKV 权重边界；
- Output Projection tiling；
- 随机 backpressure；
- 连续 token。

## 10. 多头状态隔离

必须保证每个 head 的：

```text
K cache
V cache
valid length
q vector
head output
```

互不污染。

建议仿真中为每个 head 使用独立随机分布，避免错误共享仍然碰巧通过。

## 11. 性能分析

记录：

```text
q_projection_cycles
k_projection_cycles
v_projection_cycles
attention_cycles_per_head
head_switch_cycles
concat_cycles
output_projection_cycles
total_mha_cycles
weight_stall_cycles
kv_stall_cycles
```

分析：

- 时间是否主要花在 projection；
- Attention 是否随 seq_len 主导；
- head 切换 bubble；
- 权重带宽是否成为瓶颈；
- 共享 PE 利用率。

## 12. 面积与功耗

重点比较：

```text
单引擎时间复用
双引擎
多 head 完全复制
```

第一版只实现单引擎，但 cycle model 应能估算不同并行度，避免后续盲目复制硬件。

## 13. 未来 Voting 接口预留

虽然当前不实现 Voting，建议在 softmax probability 流旁路预留：

```text
prob_valid
prob_ready
prob_data
prob_token_index
prob_head_id
prob_last
```

默认 tie-off，不影响主数据通路。

## 14. Stage 5 交付物

```text
rtl/attention/multihead/*
rtl/transformer/projection/*
tb/integration/multihead/*
model/bit_model/multihead.py
model/cycle_model/multihead.py
reports/stage5/*
docs/stage5_report.md
```

## 15. 退出条件

- 小型 MHA 端到端 bit-exact；
- 所有 head 状态隔离；
- QKV 和 W_O tiling 正确；
- 连续 token 正确；
- 无 head 切换 bubble 异常；
- 综合 WNS ≥ 0；
- 给出多头周期分解、带宽和资源利用率。
