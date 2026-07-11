# Stage 1：算术单元、FIFO 与 SRAM Wrapper

## 1. 阶段目标

建立后续 PE、Softmax、Norm 和 FFN 所需的可综合基础模块，并完成单元级功能、时序、面积和功耗评估。

本阶段不实现完整 Attention。

## 2. 模块范围

建议模块：

```text
rtl/arithmetic/
├── fp_or_fixed_mul.sv
├── fp_or_fixed_add.sv
├── mac_unit.sv
├── accumulator.sv
├── exp_unit.sv
├── reciprocal_unit.sv
├── divider_unit.sv
├── sqrt_unit.sv
├── round_sat.sv
└── compare_max.sv

rtl/memory/
├── sync_fifo.sv
├── skid_buffer.sv
├── sram_1p_wrapper.sv
├── sram_2p_wrapper.sv
└── banked_sram_wrapper.sv
```

若采用 Synopsys DesignWare 或其他可综合算术 IP，应统一包在自定义 wrapper 中，禁止上层直接依赖 IP 特定端口。

## 3. 算术单元流水

每个长延迟单元必须定义：

```text
LATENCY
THROUGHPUT
READY_VALID_BEHAVIOR
PIPELINE_FLUSH
RESET_BEHAVIOR
```

目标优先级：

1. 吞吐率可持续为每周期一个数据；
2. 延迟可以大于 1；
3. 支持 backpressure；
4. 元数据与结果严格对齐。

### 3.1 MAC

建议拆分：

```text
input register
→ multiply
→ product register
→ add/accumulate
→ output register
```

避免 multiplier、large mux、adder、accumulator feedback 处于同一组合路径。

### 3.2 EXP

候选实现：

- DesignWare/IP；
- LUT + piecewise approximation；
- range reduction + polynomial；
- base-2 exponential approximation。

第一版先保证正确和可综合，再以综合结果决定是否替换。

### 3.3 Reciprocal / Divider

候选实现：

- IP；
- LUT 初值 + Newton-Raphson；
- 固定迭代次数；
- 组合近似仅用于很小位宽。

### 3.4 SQRT

主要用于 LayerNorm 标准差路径。若最终选择 RMSNorm，也仍可能需要 reciprocal sqrt。

## 4. FIFO 与流控

FIFO 必须支持：

- 同周期读写；
- full/empty；
- almost_full；
- occupancy counter；
- overflow/underflow assertion；
- reset 清空；
- 可选 fall-through；
- 参数化深度；
- 非 2 的幂深度检查。

Skid buffer 用于切断 ready 组合路径。

禁止形成长距离 combinational ready chain。

## 5. SRAM Wrapper

行为模型与综合模型必须分离。

Wrapper 需统一描述：

```text
read_latency
write_first/read_first/no_change
byte_enable
bank_select
collision_behavior
clock_enable
```

综合时应能够替换为：

- SRAM macro；
- memory compiler 输出；
- black-box memory；
- FPGA/仿真行为模型。

禁止直接将大容量 K/V/Score 数组展开为触发器后用于正式面积评估。

## 6. 功能验证

### 6.1 算术

每个模块至少覆盖：

- 0；
- 最大正数；
- 最小负数；
- 极小数；
- 正负混合；
- 连续最大吞吐；
- 随机 backpressure；
- reset 插入；
- pipeline flush；
- 舍入边界；
- 饱和边界。

若采用浮点，还应明确：

- NaN；
- Inf；
- denormal；
- signed zero。

若第一版不支持这些情况，必须在规格中明确钳位或屏蔽规则。

### 6.2 FIFO

随机执行：

```text
push
pop
push+pop
stall
reset
near-full
near-empty
```

使用 reference queue 比较数据顺序。

## 7. Assertions

至少包括：

```text
no_write_when_full
no_read_when_empty
valid_stable_until_ready
metadata_stable_until_ready
occupancy_in_range
no_unknown_output_when_valid
pipeline_transaction_count_conserved
```

## 8. EDA 检查

每个算术模块单独执行：

```text
lint
VCS regression
DC synthesis
PrimeTime STA
SAIF-based power
```

报告：

```text
area
cell count
critical path
latency
throughput
dynamic power
leakage power
clock power
```

重点检查 EXP、DIV、SQRT 是否成为后续不可接受的面积或关键路径来源。

## 9. PPA 优化决策点

完成第一轮综合后，对每个高成本模块做三选一：

1. 保持精确实现；
2. 改为更深流水；
3. 改为近似实现。

任何近似实现必须同步更新 bit-accurate 模型和误差预算。

## 10. Stage 1 交付物

```text
rtl/arithmetic/*
rtl/memory/*
tb/unit/*
scripts/dc/*
scripts/pt/*
scripts/power/*
reports/stage1/*
docs/stage1_report.md
```

## 11. 退出条件

- 所有基础模块 regression 通过；
- 无 FIFO overflow/underflow；
- 无 latch、多驱动、未约束路径；
- 所有长延迟单元能持续吞吐；
- WNS ≥ 0，TNS = 0；
- 面积与功耗报告完整；
- bit-accurate 模型与 RTL 对齐；
- 选定 EXP、DIV、SQRT 的第一版实现。
