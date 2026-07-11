# Stage 2：可重构 PE 阵列与共享 GEMV 核心

## 1. 阶段目标

实现一套可在多种运算模式间切换的共享计算核心，至少支持：

```text
MODE_GEMV
MODE_QK_INNER
MODE_SV_OUTER
```

这一阶段只验证线性计算，不接完整 Softmax。

## 2. 架构目标

参考论文的 flexible-product dataflow：

- `q × K^T` 使用 inner-product；
- `s' × V` 使用 outer-product；
- K/V 统一按 `(sequence, dimension)` 存储；
- 动态序列长度映射为时间维；
- `d` 映射为空间维或分块空间维。

## 3. 模块边界

```text
shared_gemv_core
├── input_loader
├── weight_loader
├── pe_array
├── lane_mask_generator
├── local_reduction
├── hierarchical_adder_tree
├── accumulator_bank
├── output_formatter
├── mode_controller
└── perf_counter
```

PE 内建议包含：

```text
input register
weight register
multiplier
local adder/mux
accumulator register
partial-sum forwarding
mode decode
clock enable
```

## 4. 运行模式

### 4.1 MODE_GEMV

```text
y = xW
```

适用于：

- QKV Projection；
- Output Projection；
- FFN1；
- FFN2。

必须支持 k/n 维分块。

### 4.2 MODE_QK_INNER

每次读取一个 K token：

```text
score_i = dot(q, K_i)
```

输出按 token 顺序逐元素产生。

### 4.3 MODE_SV_OUTER

每周期输入一个 softmax 标量 `p_i`，广播到 PE，读取 `V_i`：

```text
o_j += p_i * V_i,j
```

所有输出维度在本地 accumulator 中持续累加。

## 5. Tiling

若 `D_HEAD > PE_NUM`，必须支持 dimension tiling：

```text
num_tiles = ceil(D_HEAD / PE_NUM)
```

最后一个 tile 使用 lane mask。

必须明确：

- 每个 tile 的 partial sum 保存位置；
- accumulator 清零时机；
- 最终结果有效时机；
- tile 之间是否有 bubble；
- 读地址变化；
- mode 切换延迟。

## 6. 加法树

禁止单周期形成超深大树。

建议：

```text
multiplier output
→ local pair reduction
→ row reduction
→ bank reduction
→ global result register
```

每级均可寄存。

加法树级数由目标工艺库和时钟约束决定。

## 7. 高扇出控制

重点信号：

```text
mode
clear_acc
lane_mask
broadcast_scalar
reset
clock_enable
```

必须采用：

- 分层 mode register；
- 局部 enable；
- broadcast tree；
- 物理阶段可复制控制寄存器。

不要让一个顶层寄存器直接驱动全部 PE 内部多级逻辑。

## 8. 流水与流控

推荐端口：

```text
cmd_valid/cmd_ready
input_valid/input_ready
weight_valid/weight_ready
output_valid/output_ready
```

必须允许：

- 输入暂时停止；
- 输出 backpressure；
- pipeline drain；
- operation flush；
- mode 间安全切换。

任何 mode 切换只能发生在前一 operation 完全完成后，除非明确实现 context buffering。

## 9. 功能验证

分别验证：

### 9.1 GEMV

随机矩阵与向量，覆盖：

```text
k < PE_NUM
k = PE_NUM
k = PE_NUM + 1
k = 2*PE_NUM - 1
n 非 PE_NUM 整数倍
```

### 9.2 QK Inner Product

连续输出多个 token score，检查顺序和数值。

### 9.3 SV Outer Product

连续输入多个 `(p_i, V_i)`，检查最终 accumulator。

### 9.4 Backpressure

在输入、权重和输出端随机拉低 ready。

### 9.5 Mode Switching

随机在三个 mode 间切换，检查无旧 accumulator 残留。

## 10. 性能计数器

至少记录：

```text
total_cycles
active_pe_cycles
idle_cycles
stall_input_cycles
stall_output_cycles
tile_count
mode_switch_cycles
valid_lane_count
```

PE 利用率建议定义为：

```text
valid MAC operations / available MAC slots
```

## 11. 第一次物理验证

Stage 2 建议执行第一次 P&R。

范围：

```text
pe_array
adder_tree
accumulator_bank
minimal input/output register slice
```

检查：

- adder tree 布线；
- scalar broadcast；
- mode 高扇出；
- accumulator feedback；
- 局部拥塞；
- 时钟树；
- post-route WNS/TNS；
- post-route power。

若布线明显恶化，应在进入 Attention 前调整：

- PE 分区；
- adder tree 层级；
- pipeline；
- SRAM/寄存器位置；
- broadcast 结构。

## 12. Stage 2 交付物

```text
rtl/pe/*
tb/block/pe_core/*
model/bit_model/gemv_core.py
scripts/pnr/pe_core/*
reports/stage2/*
docs/stage2_report.md
```

## 13. 退出条件

- 三种 mode 均 bit-exact；
- 所有 tiling 边界通过；
- 随机 backpressure 无死锁；
- mode 切换无数据污染；
- WNS ≥ 0，TNS = 0；
- P&R 无不可接受拥塞；
- PE 利用率和周期数与 cycle model 一致；
- 关键路径有清晰归属。
