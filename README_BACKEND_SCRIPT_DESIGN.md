# 后端脚本设计思路与关键方法

> Legacy note: this file is not a VEDA Stage 0 or Stage 1 specification.
> It comes from earlier SFU/backend workflow notes and contains unverified
> local paths, tool assumptions, and process references. Do not use it as
> evidence that Synopsys tools, TSMC28 PDK files, reports, or PPA results are
> available for this repository. VEDA stage work must follow `AGENTS.md`,
> `PROJECT_STATE.md`, `HANDOFF.md`, `docs/stage_00_spec.md`, and the active
> stage file under `transformer_rtl_plan_md/`.

本文说明当前 TSMC28 ICC 后端脚本的整体逻辑、关键方法和调试思路。重点不是逐条背命令，而是理解脚本为什么这样组织，以及遇到 DRC/PG/时序问题时应该怎么定位。

## 1. 当前目标

当前 SFU 后端目标是：

- 使用 TSMC28 HPC+ 标准单元库完成 ASIC 后端布局布线。
- 保持 `create_floorplan -core_utilization 0.60` 的 floorplan 设定。
- 让 ICC in-design DRC 收敛到 0。
- 保证 short/open 为 0，VDD/VSS PG 不再 open。
- 只有 clean 时才导出 FINAL DEF 和 post-PnR Verilog。

需要注意：这里的 DRC clean 指 ICC 内部 `verify_zrt_route` clean，不等价于 Calibre/ICV signoff clean。

## 2. 脚本文件分工

### `run_icc_tsmc28_full.tcl`

完整后端入口脚本。

作用：

- 设置 `RUN_FULL_PNR=1`
- source 真正的主流程 `run_icc_tsmc28.tcl`

这样设计是为了防止误运行主脚本。因为完整后端会删除并重建 Milkyway 设计库。

### `run_icc_tsmc28.tcl`

完整 ICC PnR 主脚本。

它负责：

- 建立 Milkyway library
- 读入 DC 门级网表
- link 设计
- 设置 TLU+ RC 规则
- 建立时钟和 IO 约束
- floorplan
- placement
- PG rail/core ring
- CTS
- routing
- filler 插入
- DRC closure
- LVS/DRC 最终检查
- clean 时导出 FINAL DEF 和 post-PnR Verilog

### `icc_drc_repair_lib.tcl`

动态 DRC 修复库。

它不负责从 RTL 开始跑完整后端，而是被主脚本调用，用于 routing 后的 DRC closure。

核心思想：

```text
verify_zrt_route
-> get_drc_errors
-> 提取 type / bbox / net
-> 按错误类型扩大修复窗口
-> 合并相近窗口
-> 局部 detailed routing 修复
-> 相关 net ECO route
-> 增量 DRC-only detailed routing 修复
-> 再 verify_zrt_route
-> 重复直到 DRC = 0 或达到迭代上限
```

### `repair_icc_drc_from_saved.tcl`

从已保存 Milkyway layout 继续修复。

它不会重新做：

- create Milkyway library
- read_verilog
- place_opt
- clock_opt
- initial route

它适合在已有 post-route 结果已经很接近 clean 时继续做 ECO DRC closure。

### `probe_icc_drc.tcl`

只探测当前保存版图的 DRC object。

作用：

- 打开已保存 Milkyway cell
- 运行 `verify_zrt_route`
- 打印当前 DRC 的 type、bbox、layer、net 等信息

### `verify_saved_icc_final.tcl`

只做最终复核。

作用：

- 打开已保存 Milkyway cell
- 重新跑 DRC/LVS
- 重新输出最终 QoR/timing/power/constraint 报告

## 3. 主流程整体逻辑

当前主流程可以概括为：

```text
1. 建库与读网表
2. 显式连接 VDD/VSS
3. 设置 RC 规则和 route 选项
4. 建立时序约束
5. 60% floorplan
6. placement
7. PG rail + M2/M3 core ring
8. CTS
9. initial route + detailed route
10. filler 插入并接 VDD/VSS
11. DRC-only 增量修复
12. 动态 DRC closure
13. verify_zrt_route + verify_lvs
14. DRC 为 0 才导出 FINAL
```

这套流程的关键点是：PG 必须在正式信号布线前规划好，DRC closure 必须基于当前版图动态读取 violation，而不是写死上一次的坐标。

## 4. PG 连接方法

PG 的目标是让标准单元的 VDD/VSS pin、top-level VDD/VSS port、std-cell rails 和 core ring 形成连通网络。

当前脚本里有三个关键过程。

### 1. `icc_pg_ensure_ports_and_nets`

作用：

- 如果没有 VDD/VSS net，就创建 power/ground net。
- 用 `derive_pg_connection -create_ports top` 创建顶层 VDD/VSS port。
- 用 `derive_pg_connection -tie` 处理 tie-high/tie-low 相关连接。
- 用 `derive_pg_connection -reconnect` 重新连接标准单元 PG pin。

为什么这样做：

如果只依赖库里的隐式 PG pin，ICC 可能能布线，但 LVS 会看到 VDD/VSS 没有明确 top port 或 logical connection，导致 `Logical Net VDD is open` 或 `Logical Net VSS is open`。

### 2. `icc_pg_preroute_stdcell_rails`

作用：

- 使用 `preroute_standard_cells -mode rail`
- 横向连接标准单元 row 里的 VDD/VSS rail
- 生成 boundary pin
- 删除 floating PG 小碎片

为什么这样做：

标准单元内部有 VDD/VSS pin，但不同 row、不同 cell 之间需要通过 rail 连接起来。没有 rail，信号布线可以跑完，但 PG 可能还是 open。

### 3. `icc_pg_create_core_ring`

作用：

- 在 core 周围创建 VDD/VSS core ring
- 垂直边使用 M2
- 水平边使用 M3
- 宽度 `0.10`
- absolute offset `0.85`

为什么这样设置：

之前尝试过内部 vertical straps，但在当前高密度布线下会制造大量 via spacing、short 和低层拥塞。最后采用比较保守的 core ring，让 PG 有全局闭合边界，同时不在 core 内部强行占用大量 M2/M3 资源。

## 5. 为什么不全局绕开 M1/M2

当前脚本没有全局禁止 M1/M2，也没有把 M1/M2 完全绕开。

原因：

- M1/M2 对标准单元 pin access 很重要。
- 如果全局禁止低层金属，router 可能找不到从 cell pin 接出的合法路径。
- 如果大幅惩罚 M1/M2，可能减少一部分低层 spacing，但会把压力转移到 via 和上层金属，导致新的 DRC。

当前策略是：

```tcl
set_route_zrt_common_options \
    -route_soft_rule_effort_level high \
    -post_detail_route_fix_soft_violations true \
    -post_incremental_detail_route_fix_soft_violations true \
    -post_eco_route_fix_soft_violations true
```

也就是不粗暴禁层，而是让 ZRoute 在 detail route、incremental route 和 ECO route 阶段更积极地修低层 soft-rule/spacing 问题。

## 6. DRC closure 的核心方法

### 第一步：验证当前版图

```tcl
verify_zrt_route
```

它检查当前 ICC/ZRoute 数据库中的 routing DRC，例如：

- short
- spacing
- via spacing
- end-of-line spacing
- fat contact extension
- open net

### 第二步：读取当前 violation object

```tcl
get_drc_errors
```

每个 DRC object 重点提取三类信息：

- `type`: 错误类型
- `bbox`: 几何坐标窗口
- `net`: 相关信号线

这样做的原因是：每次从头 place/route 后，DRC 坐标可能不同。如果把上一次的坐标写死到脚本里，下一次重跑很可能失效。动态读取当前 object 才能保证脚本可复现。

### 第三步：按错误类型扩大窗口

不同错误需要不同 margin。

当前思路：

- short: 窗口略大，因为短路通常涉及两条线或一组 via。
- via/contact: 窗口可以小一些，因为问题通常集中在 via cut 附近。
- fat contact extension: 窗口更大，因为它经常和接触孔周围金属延伸相关。
- 普通 spacing/EOL: 使用中等窗口。

### 第四步：合并相近窗口

如果两个 DRC 很近，单独修可能会互相影响。脚本会把相近 bbox 合并成一个更大的 repair window。

这样 ICC 在局部 detailed routing 时可以同时处理一小片拥塞区，而不是修 A 破坏 B。

### 第五步：局部 detailed routing 修复

```tcl
route_zrt_detail -coordinates <window>
```

含义：

- 只在指定窗口内重新做 detail route。
- 不推翻整个布局布线结果。
- 适合修 spacing、EOL、局部 short 等几何问题。

这一步解决的是“这个区域几何上不合法”的问题。

### 第六步：相关 net ECO route

```tcl
route_zrt_eco -nets <problem_nets>
```

含义：

- 针对参与 violation 的 net 做 ECO reroute。
- 它不是只看矩形窗口，而是看具体哪几根 net 需要重新走。

这一步解决的是“这几根线本身需要换走法”的问题。

### 第七步：增量 DRC-only 修复

```tcl
route_opt -incremental -only_design_rule -effort high
```

含义：

- 在当前 routing 基础上做增量合法化。
- 目标只放在 design rule，不重新追求完整时序/面积优化。
- 它不是 global route，也不是从头布线。

为什么最后还要跑一轮：

局部 window 修复和 net ECO 修复之后，可能产生新的边界小问题。最后的 DRC-only pass 用于全局范围内收尾，把局部修复造成的次生 DRC 再合法化。

## 7. 为什么设置 DRC 数量阈值

脚本里有一个思路：如果当前 DRC 很多，就先不做精细局部窗口，而是先跑增量 DRC-only 修复。

原因：

- DRC 数量很大时，说明不是一两个热点，而是整体 routing 还没有合法化。
- 这时逐个 bbox 修复效率低，且容易互相破坏。
- 先用 `route_opt -incremental -only_design_rule` 降低整体错误量，再进入局部修复，收敛更稳定。

当前阈值思路是：DRC 数量大于约 160 时，先做整体 detailed-routing 合法化；降到较低数量后，再做局部 object-driven repair。

## 8. Filler 的作用

当前脚本插入 filler：

```tcl
insert_stdcell_filler \
    -cell_with_metal {FILL64... FILL2...} \
    -connect_to_power VDD \
    -connect_to_ground VSS
```

作用：

- 填补标准单元 row 里的空 site。
- 保证 well/implant/metal continuity。
- 让 PG rail 更完整。
- 减少 LVS/DRC 中由于 row 空洞造成的问题。

为什么不能忽略 filler：

一个版图即使信号线 DRC 为 0，如果没有 filler，可能仍然在 well continuity、implant continuity、PG rail continuity 或 LVS 规则上出问题。

## 9. FINAL 和 LAST_ATTEMPT 的保护机制

当前脚本只有在：

```text
FINAL_VERIFY_ZRT_DRC_COUNT = 0
```

时才导出：

```text
sfu_top_28nm_FINAL.def
sfu_top_28nm_post_pnr.v
icc_qor_FINAL.rpt
icc_timing_FINAL.rpt
```

如果最终 DRC 不是 0，只导出：

```text
LAST_ATTEMPT
```

并报错退出。

这样设计的原因是防止一个不 clean 的新结果覆盖之前 clean 的 FINAL 结果。

## 10. 目前流程的边界

当前流程已经能证明：

- DC + ICC 后端链路能跑通。
- ICC in-design DRC 可以收敛到 0。
- short/open 可以为 0。
- VDD/VSS PG open 问题已经解决。
- FINAL DEF 和 post-PnR Verilog 可以导出。

但它还不等于完整流片 signoff。

真正 tapeout 前还需要：

- Calibre 或 ICV signoff DRC
- Calibre 或 ICV signoff LVS
- antenna check
- parasitic extraction, PEX
- post-route STA with extracted RC
- IR drop
- EM
- formal equivalence check
- 多 corner 多 mode 验证

## 11. 向老师解释时的主线

可以按下面逻辑讲：

1. 我先用 DC 生成 TSMC28 标准单元门级网表。
2. 后端用 ICC，从 Milkyway/TF/TLU+ 建立真实物理环境。
3. floorplan 设定为 60%，然后做 placement、CTS、routing。
4. 普通 routing 后 DRC 不一定为 0，所以我把 DRC closure 写成动态闭环。
5. 每一轮不是用固定坐标，而是从当前 layout 重新读取 DRC object。
6. 根据 DRC type、bbox、net 生成修复窗口和 ECO net list。
7. 先局部 detail route，再对相关 net 做 ECO route，最后用 DRC-only incremental route 收尾。
8. 同时补完整 PG：VDD/VSS top port、std-cell rail、M2/M3 core ring、filler power connection。
9. 最终只有 ICC DRC 为 0 时才导出 FINAL。
10. 我会明确说明这是 ICC in-design clean，不是 signoff clean。

## 12. 常见提问与回答

### Q1: 为什么不能直接说 DRC clean 就可以流片？

不能。当前 DRC 0 是 ICC `verify_zrt_route` 的结果。流片需要 foundry signoff deck，通过 Calibre/ICV 重新检查。

### Q2: 为什么不用固定坐标修复？

因为重新 place/route 后 DRC 位置会变。固定坐标只能修某一次 layout，不能保证下一次重跑仍然有效。当前方法每轮读取当前 violation object，所以更可复现。

### Q3: 局部 detailed routing 和 ECO route 有什么区别？

局部 detailed routing 是按窗口修几何区域；ECO route 是按相关 net 修具体线。前者解决局部区域不合法，后者解决某些 net 走法不合理。实际流程里两者可以连续使用。

### Q4: 为什么最后还要 DRC-only route_opt？

局部修复可能在窗口边界产生新的 spacing/EOL/via 问题。最后一轮 DRC-only 增量修复用于收尾，目标只放在 design rule 合法化。

### Q5: 为什么不用内部 vertical straps？

当前密度下内部 strap 会占用低层/过孔资源，实测会显著增加 DRC。最终采用 std-cell rail + M2/M3 core ring 的保守 PG 方案，先保证 PG 连通和 DRC clean。

### Q6: 60% 利用率为什么报告里看到 74% 左右？

`core_utilization 0.60` 是 floorplan 阶段用于估算 core 大小的目标值。ICC 报告的 std-cell utilization 是标准单元面积除以实际可用 placement site，统计口径不同，所以数值会更高。

### Q7: DRC 和时序违例有关系吗？

两者不是同一种违例。DRC 是几何/工艺规则问题，时序违例是路径延迟问题。但它们会互相影响：修 DRC 可能绕线变长，导致时序变差；修时序可能插 buffer 或改线，也可能制造新的 DRC。因此后端通常需要 DRC closure 和 timing ECO 迭代。
