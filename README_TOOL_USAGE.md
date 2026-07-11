# 后端工具与 Vivado 调用方法

> Legacy note: this file is not a VEDA Stage 0 or Stage 1 specification.
> It comes from earlier SFU/backend workflow notes and contains unverified
> local paths, tool assumptions, and process references. Do not use it as
> evidence that Synopsys tools, TSMC28 PDK files, reports, or PPA results are
> available for this repository. VEDA stage work must follow `AGENTS.md`,
> `PROJECT_STATE.md`, `HANDOFF.md`, `docs/stage_00_spec.md`, and the active
> stage file under `transformer_rtl_plan_md/`.

本文记录当前工作区里 SFU/ASIC 后端工具链和 Vivado FPGA 流程的调用方式。默认假设 Windows 工作区 `D:\IC_Workspace` 在 Docker 内映射为 `/workspace`。

## 1. 工具分工

### Design Compiler, DC

DC 用于前端综合，把 RTL 转成 TSMC28 标准单元门级网表。

主要输入：

- RTL: `/workspace/SFU/rtl/src/sfu_top.v`
- RTL: `/workspace/SFU/rtl/src/sfu_lane.v`
- 标准单元时序库 `.db`: `/workspace/TSMC28_PDK/.../tcbn28hpcplusbwp7t40p140tt0p9v25c.db`
- 时序约束由综合脚本内部创建，例如 `create_clock`、`set_input_delay`、`set_output_delay`

主要输出：

- 门级网表: `/workspace/SFU/syn/results/sfu_top_netlist_FIXED.v`
- 综合报告: `/workspace/SFU/syn/reports/*.rpt`

### IC Compiler, ICC

ICC 用于 TSMC28 后端布局布线，也就是 floorplan、placement、CTS、routing、DRC closure、LVS 检查和 DEF/post-PnR netlist 导出。

主要输入：

- DC 输出的门级网表: `/workspace/SFU/syn/results/sfu_top_netlist_FIXED.v`
- TSMC28 Milkyway reference library
- TSMC28 technology file `.tf`
- TLU+ RC 文件和 tech-to-ITF mapping 文件

主要输出：

- Milkyway 设计库: `/workspace/SFU/pnr/work/sfu_mw_lib_28nm`
- FINAL DEF: `/workspace/SFU/syn/results/sfu_top_28nm_FINAL.def`
- post-PnR Verilog: `/workspace/SFU/syn/results/sfu_top_28nm_post_pnr.v`
- ICC QoR/timing/power/constraint 报告: `/workspace/SFU/syn/reports/icc_*_FINAL.rpt`

### Vivado

Vivado 用于 FPGA 侧验证和资源/时序评估，不参与 TSMC28 ASIC 后端实现。当前脚本主要支持：

- Zynq-7020: `xc7z020clg400-1`
- Alveo U50/CU50: `xcu50-fsvh2104-2-e`

Vivado 输出：

- FPGA utilization report
- FPGA timing summary
- Vivado project directory

## 2. 进入 Docker 环境

在 Windows PowerShell 中：

```powershell
cd D:\IC_Workspace
docker ps -a
docker start nailong
docker exec -it nailong bash
```

进入容器后：

```bash
cd /workspace/SFU
pwd
which dc_shell
which icc_shell
dc_shell -version
icc_shell -version
```

如果 `which dc_shell` 或 `which icc_shell` 找不到工具，说明当前容器环境没有加载 Synopsys 工具路径或 license 环境。

## 3. 运行 DC 综合

推荐先运行严格映射版本，因为当前 ICC 后端脚本默认读取它生成的 `sfu_top_netlist_FIXED.v`。

```bash
cd /workspace/SFU
dc_shell -f /workspace/SFU/syn/scripts/run_syn_tsmc28_strict.tcl \
  | tee /workspace/SFU/reports/dc_syn_$(date +%Y%m%d_%H%M%S).log
```

普通版本也可以运行：

```bash
cd /workspace/SFU
dc_shell -f /workspace/SFU/syn/scripts/run_syn_tsmc28.tcl \
  | tee /workspace/SFU/reports/dc_syn_tsmc28_$(date +%Y%m%d_%H%M%S).log
```

综合后重点检查：

```bash
ls -lh /workspace/SFU/syn/results
ls -lh /workspace/SFU/syn/reports
grep -E "slack|VIOLATED|Total cell area|Design Area" /workspace/SFU/syn/reports/*.rpt
```

## 4. 运行 ICC 完整后端

完整重跑入口：

```bash
cd /workspace/SFU
icc_shell -f /workspace/SFU/pnr/scripts/run_icc_tsmc28_full.tcl \
  | tee /workspace/SFU/reports/icc_full_$(date +%Y%m%d_%H%M%S).log
```

这个入口会设置 `RUN_FULL_PNR=1`，然后 source 主脚本：

```tcl
/workspace/SFU/pnr/scripts/run_icc_tsmc28.tcl
```

注意：完整后端会删除并重建：

```text
/workspace/SFU/pnr/work/sfu_mw_lib_28nm
```

所以它适合从网表重新实现，不适合在已有 post-route 版图上继续 ECO。

## 5. 检查 ICC 最终结果

完整跑完后，在日志中检查这些关键字段：

```bash
grep -E "FINAL_VERIFY_ZRT_DRC_COUNT|DYNAMIC_DRC_FINAL_COUNT|FINAL_EXPORT_STATUS" /workspace/SFU/reports/icc_full_*.log
grep -E "Total SHORT Nets|Total OPEN Nets|Floating ports|Logical Net VDD|Logical Net VSS" /workspace/SFU/reports/icc_full_*.log
grep -E "Std cell utilization|Critical Path Slack|Total Negative Slack|WNS|TNS" /workspace/SFU/reports/icc_full_*.log
```

关键接受标准：

- `FINAL_VERIFY_ZRT_DRC_COUNT 0`
- `Total SHORT Nets are 0`
- `Total OPEN Nets are 0`
- 日志中没有 `Logical Net VDD is open` 或 `Logical Net VSS is open`
- `FINAL_EXPORT_STATUS clean_drc=0 exported=FINAL`
- FINAL DEF 和 post-PnR Verilog 文件非空

检查导出文件：

```bash
ls -lh /workspace/SFU/syn/results/sfu_top_28nm_FINAL.def
ls -lh /workspace/SFU/syn/results/sfu_top_28nm_post_pnr.v
ls -lh /workspace/SFU/syn/reports/icc_qor_FINAL.rpt
ls -lh /workspace/SFU/syn/reports/icc_timing_FINAL.rpt
```

## 6. 只检查或修复已保存 ICC 版图

如果已经有保存好的 Milkyway layout，不想重新 place/route，可以使用以下脚本。

只探测当前 DRC：

```bash
cd /workspace/SFU
icc_shell -f /workspace/SFU/pnr/scripts/probe_icc_drc.tcl
```

从已保存 layout 继续 DRC ECO 修复：

```bash
cd /workspace/SFU
icc_shell -f /workspace/SFU/pnr/scripts/repair_icc_drc_from_saved.tcl \
  | tee /workspace/SFU/reports/icc_repair_$(date +%Y%m%d_%H%M%S).log
```

最终复核已保存 layout：

```bash
cd /workspace/SFU
icc_shell -f /workspace/SFU/pnr/scripts/verify_saved_icc_final.tcl \
  | tee /workspace/SFU/reports/icc_verify_saved_$(date +%Y%m%d_%H%M%S).log
```

使用原则：

- `run_icc_tsmc28_full.tcl`: 从头重建，适合改了网表、floorplan、PG 策略或主流程。
- `repair_icc_drc_from_saved.tcl`: 不重新布局布线，只在已有 Milkyway cell 上继续 ECO 修复。
- `probe_icc_drc.tcl`: 只读当前 DRC object，用于定位问题。
- `verify_saved_icc_final.tcl`: 最终确认已保存版图是否仍然 clean。

## 7. 运行 Vivado FPGA 实现

Vivado 一般在 Windows 主机上运行，不依赖 Synopsys Docker。先确认 Vivado 在 PATH 中：

```powershell
vivado -version
```

推荐从 `rtl\scripts` 目录运行，因为脚本内部使用了 `../src` 和 `../xdc` 相对路径。

Zynq-7020：

```powershell
cd D:\IC_Workspace\SFU\rtl\scripts
vivado -mode batch -source run_impl_z7020.tcl
```

U50/CU50：

```powershell
cd D:\IC_Workspace\SFU\rtl\scripts
vivado -mode batch -source run_impl_cu50.tcl
```

如果需要打开 GUI，可以用：

```powershell
cd D:\IC_Workspace\SFU\rtl\scripts
vivado -mode gui -source run_impl_z7020.tcl
```

Vivado 输出报告在：

```text
D:\IC_Workspace\SFU\rtl\vivadoreport\impl_utilization_z7020.txt
D:\IC_Workspace\SFU\rtl\vivadoreport\impl_timing_z7020.txt
D:\IC_Workspace\SFU\rtl\vivadoreport\impl_utilization_cu50.txt
D:\IC_Workspace\SFU\rtl\vivadoreport\impl_timing_cu50.txt
```

## 8. 常见问题

### 1. 为什么 ASIC 后端不用 Vivado？

Vivado 是 FPGA 工具，目标是 Xilinx FPGA fabric；ICC 是 ASIC 后端工具，目标是 TSMC28 标准单元和金属层。两者的输入 RTL 可以相同，但物理实现目标完全不同。

### 2. 为什么完整 ICC 脚本要单独有一个 full 入口？

`run_icc_tsmc28.tcl` 会删除并重建 Milkyway 库。为了避免误操作，脚本要求必须由 `run_icc_tsmc28_full.tcl` 设置 `RUN_FULL_PNR=1` 后才能执行。

### 3. DRC 0 是否等于可以流片？

不是。当前结果是 ICC `verify_zrt_route` 的 in-design DRC clean。真正 tapeout 前还需要 foundry signoff deck 下的 Calibre/ICV DRC/LVS，以及 antenna、PEX、STA、IR drop、EM 等 signoff 检查。

### 4. Vivado 脚本为什么要在 `rtl\scripts` 目录运行？

因为脚本中写的是相对路径：

```tcl
add_files {../src/sfu_top.v ../src/sfu_lane.v}
add_files -fileset constrs_1 ../xdc/sfu_timing_z7020.xdc
```

如果从其他目录运行，Vivado 会找不到 RTL 或 XDC。
