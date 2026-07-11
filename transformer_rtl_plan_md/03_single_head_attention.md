# Stage 3：单头 Generation Attention

## 1. 阶段目标

在共享 PE 核心上完成一个可连续运行的单头生成阶段 Attention：

```text
qK^T
→ scale
→ softmax
→ s'V
```

本阶段暂不加入多头、QKV Projection、完整 Transformer，也不加入 Voting。

## 2. 顶层边界

```text
single_head_attention
├── attention_controller
├── q_buffer
├── kv_read_interface
├── shared_gemv_core
├── score_buffer
├── softmax_reduction
├── softmax_normalization
├── v_stream_aligner
├── head_output_buffer
└── performance_counter
```

## 3. 数据流

### Phase A：QK Inner Product + Softmax Reduction

```text
读取 K_i
→ PE 计算 score_i
→ scale by 1/sqrt(d)
→ 写 score_buffer
→ 同时送 softmax_reduction
```

Softmax reduction 在线维护：

```text
max_score
exp_sum
```

### Phase B：Softmax Normalization + SV Outer Product

```text
读取 score_i
→ exp(score_i - max_score)
→ divide by exp_sum
→ 得到 p_i
→ 读取 V_i
→ PE 执行 outer-product accumulation
```

最终输出 head vector。

## 4. Score Buffer

Score buffer 是必要模块，即使不做 Voting 也不能删除。

必须明确：

- 深度；
- 端口数；
- 写入顺序；
- 读取顺序；
- 读延迟；
- score 格式；
- 是否按 tile 存储；
- 是否允许与下一 token 重叠。

第一版建议单 token 串行运行，先不做双缓冲跨 token overlap。

## 5. Softmax Reduction

在线算法：

```text
m_new = max(m_old, x)
z_new = z_old * exp(m_old - m_new) + exp(x - m_new)
```

需要考虑：

- 第一元素初始化；
- 所有 score 相等；
- 极大负数；
- exp 下溢；
- max 更新；
- exp_sum 饱和；
- 最后一个元素后的流水排空。

## 6. Softmax Normalization

```text
p_i = exp(score_i - m_final) / z_final
```

输出 `p_i` 必须与对应 `V_i` 严格对齐。

若 EXP 与 DIV 有不同延迟，需要 metadata pipeline 保存：

```text
token_index
head_id
last
valid
```

## 7. Attention 控制器

建议状态：

```text
IDLE
LOAD_Q
QK_STREAM
QK_DRAIN
SOFTMAX_FINALIZE
SV_STREAM
SV_DRAIN
OUTPUT
DONE
ERROR
```

状态机必须通过真实 valid/ready 和 pipeline empty 状态切换，不能只依赖固定等待周期。

## 8. 功能验证

序列长度至少覆盖：

```text
1, 2, 3, 7, 8,
31, 32, 63, 64,
127, 128, 129,
MAX_SEQ_LEN-1,
MAX_SEQ_LEN
```

数值场景：

- 所有 score 相同；
- 一个 score 远大于其他；
- score 全负；
- score 动态范围很大；
- 接近均匀分布；
- 接近 one-hot；
- q 或 K 为零；
- V 为单位基向量；
- V 为常量；
- 随机数。

流控场景：

- K 读取停顿；
- V 读取停顿；
- EXP 停顿；
- DIV 停顿；
- 输出 backpressure；
- reset 中断；
- 最大 FIFO occupancy。

## 9. 验证指标

同时检查：

1. `score_i` 与 bit 模型一致；
2. `max_score` 一致；
3. `exp_sum` 一致；
4. 每个 `p_i` 一致；
5. `sum(p_i)` 在规定误差内接近 1；
6. 最终 head output 一致；
7. 无 score/V 地址错位；
8. 无 deadlock；
9. 无 overflow/underflow。

## 10. 性能指标

至少输出：

```text
qk_cycles
qk_drain_cycles
softmax_finalize_cycles
sv_cycles
sv_drain_cycles
total_cycles
pe_active_cycles
sfu_active_cycles
pe_idle_cycles
sfu_idle_cycles
score_buffer_peak_occupancy
stall_cycles
```

需要验证论文式 element-serial scheduling 是否真正减少了 PE/SFU 空闲，而不是只在结构图上并行。

## 11. 综合与 PPA

本阶段执行完整 block-level synthesis。

重点检查：

- score buffer 面积；
- EXP/DIV 是否限制频率；
- PE 到 reduction 的时序；
- normalization 到 outer-product 的对齐；
- ready 组合路径；
- control 高扇出。

若 `softmax_finalize_cycles` 占比过高，应评估：

- EXP pipeline；
- reciprocal 预计算；
- 双 EXP 单元；
- 分 tile online softmax；
- 更高吞吐 normalization。

## 12. Stage 3 交付物

```text
rtl/attention/single_head/*
rtl/sfu/softmax/*
tb/block/single_head/*
model/bit_model/attention_head.py
model/cycle_model/attention_head.py
reports/stage3/*
docs/stage3_report.md
```

## 13. 退出条件

- 所有指定序列长度通过；
- softmax 各中间量与 bit 模型一致；
- 随机 backpressure 无死锁；
- 输出无 token/index 错位；
- 性能计数器与 cycle model 一致；
- 综合无负裕量；
- 能给出单头不同序列长度下的 latency 曲线和利用率曲线。
