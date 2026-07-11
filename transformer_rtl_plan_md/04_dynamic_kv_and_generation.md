# Stage 4：动态 KV Cache 与连续 Generation

## 1. 阶段目标

在单头 Attention 基础上加入动态 K/V 写入、有效长度管理和连续多 token 生成控制，使设计从“单次算子”升级为“连续生成子系统”。

本阶段仍不加入 Voting。

## 2. 模块边界

```text
generation_attention_subsystem
├── token_input_controller
├── qkv_input_interface
├── kv_cache_manager
├── k_cache_banks
├── v_cache_banks
├── kv_address_generator
├── single_head_attention
├── output_queue
└── generation_scheduler
```

## 3. KV Cache 语义

推荐流程：

```text
接收当前 q/k/v
→ 将当前 k/v append 到 cache
→ valid_seq_len += 1
→ 当前 q 对 cache[0 : valid_seq_len-1] 做 attention
→ 输出当前 token 的 head output
```

必须固定：

- reset 后 cache 长度；
- cache 满时行为；
- 超过 MAX_SEQ_LEN 时错误或停止；
- 当前 token 是否能关注自己；
- 写入与读取冲突时的 SRAM 语义。

## 4. 存储布局

统一 token-major：

```text
K_cache[token][dimension]
V_cache[token][dimension]
```

推荐银行化：

```text
bank_id = dimension group
row     = token index
```

每周期读取足够多 dimension lane 以匹配 PE 吞吐。

若 SRAM 单端口无法支持同周期 append 与 attention 读取，应采用：

- 双端口 SRAM；
- K/V 分离；
- bank；
- 写入阶段与读取阶段错开；
- 小型 write buffer。

## 5. 地址生成

地址生成器必须支持：

```text
token_index
dimension_tile
bank_index
head_base
cache_base
```

第一版不做 eviction，因此 logical token index 与 physical row 可以相同。

仍建议将地址生成和物理存储解耦，以便未来加入：

- sliding window；
- circular buffer；
- voting eviction；
- logical-to-physical map。

## 6. 连续 Token 调度

建议状态：

```text
WAIT_TOKEN
APPEND_KV
START_ATTENTION
RUN_ATTENTION
PUSH_OUTPUT
UPDATE_STATE
WAIT_NEXT
```

后续优化可加入双缓冲：

```text
当前 token Attention
|| 下一 token q/k/v 预取
```

第一版先确保严格串行正确。

## 7. Cache 满处理

至少选择一种并写入规格：

1. 返回 error；
2. 停止接受新 token；
3. circular overwrite；
4. sliding window。

主线建议第一版使用“停止并报错”，避免无意引入算法变化。

## 8. 功能验证

覆盖：

- 从长度 0 增长到 1；
- 连续生成多个 token；
- 长度跨过 PE/tile 边界；
- 长度跨过 SRAM bank 边界；
- MAX_SEQ_LEN；
- append 与 read 冲突；
- 随机输入间隔；
- 输出 backpressure；
- reset 后重新开始；
- cache 满错误；
- 多次独立 sequence。

对每个 token，软件模型必须使用相同历史 K/V 计算输出。

## 9. 数据一致性检查

必须检查：

```text
K 写入值正确
V 写入值正确
valid_seq_len 正确
K/V 地址同步
score token index 正确
p_i 与 V_i 对齐
sequence reset 不残留旧值
```

建议加入 debug-only tag RAM，在仿真中保存 token_id 并随数据传播。

## 10. 性能与带宽

记录：

```text
append_cycles
attention_cycles_per_token
memory_read_bytes_per_token
memory_write_bytes_per_token
peak_bandwidth
average_bandwidth
bank_conflict_cycles
memory_stall_cycles
```

生成阶段 Attention 理论周期随序列长度增长，需要给出：

```text
latency vs seq_len
bandwidth vs seq_len
PE utilization vs seq_len
```

## 11. 第二次 P&R

Stage 4 建议进行第二次关键 P&R。

范围：

```text
PE core
SFU
score buffer
K/V SRAM wrappers or representative macros
controllers
```

重点检查：

- SRAM 宏位置；
- K/V 到 PE 的长总线；
- score buffer 到 SFU；
- bank 端口拥塞；
- clock tree；
- SRAM 周边逻辑；
- memory enable 功耗；
- post-route timing；
- post-route power。

若使用真实 SRAM 宏，必须在 floorplan 中固定宏位置后再评估。

## 12. Stage 4 交付物

```text
rtl/attention/generation/*
rtl/memory/kv_cache/*
tb/integration/generation_attention/*
model/bit_model/generation.py
model/cycle_model/generation.py
scripts/pnr/generation_attention/*
reports/stage4/*
docs/stage4_report.md
```

## 13. 退出条件

- 连续多 token 结果正确；
- 所有 KV 地址边界通过；
- cache 满行为正确；
- 无 K/V 数据错位；
- SRAM 带宽满足当前 PE 配置；
- P&R 无不可接受拥塞；
- post-route WNS ≥ 0；
- 能给出序列长度增长下的延迟、带宽、功耗趋势。
