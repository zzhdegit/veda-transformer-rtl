#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage1_dw_probe"
REPORT_DIR="$ROOT_DIR/reports/stage_01"
SUMMARY="$REPORT_DIR/dw_probe.txt"

mkdir -p "$BUILD_DIR" "$REPORT_DIR"

if [ -z "${DW_SIM_DIR:-}" ]; then
  {
    echo "Stage 1 DesignWare probe"
    echo "DW sim dir: NOT SET"
    echo "result=SKIPPED"
    echo "reason=DW_SIM_DIR environment variable is required."
  } > "$SUMMARY"
  cat "$SUMMARY"
  exit 10
fi

if [ ! -d "$DW_SIM_DIR" ]; then
  {
    echo "Stage 1 DesignWare probe"
    echo "DW sim dir: INVALID"
    echo "result=FAIL"
    echo "reason=DW_SIM_DIR does not name a readable directory."
  } > "$SUMMARY"
  cat "$SUMMARY"
  exit 11
fi

cat > "$BUILD_DIR/dw_probe_tb.sv" <<'SV'
`timescale 1ns/1ps
module dw_probe_tb;
  localparam [2:0] RND_NEAREST_EVEN = 3'b000;

  reg  [15:0] h_a, h_b;
  wire [15:0] h_mul_z;
  wire [7:0]  h_mul_status;

  reg  [31:0] f_a, f_b, f_c;
  wire [31:0] f_add_z;
  wire [7:0]  f_add_status;
  wire [31:0] f_mac_z;
  wire [7:0]  f_mac_status;

  wire [39:0] h_to_ifp;
  wire [31:0] h_to_f_z;
  wire [7:0]  h_to_f_status;
  wire [39:0] f_to_ifp;
  wire [15:0] f_to_h_z;
  wire [7:0]  f_to_h_status;

  wire [31:0] exp_z;
  wire [7:0]  exp_status;
  wire [31:0] div_z;
  wire [7:0]  div_status;
  wire [31:0] recip_z;
  wire [7:0]  recip_status;
  wire [31:0] sqrt_z;
  wire [7:0]  sqrt_status;
  wire [31:0] invsqrt_z;
  wire [7:0]  invsqrt_status;

  DW_fp_mult #(10, 5, 1) u_fp16_mult (
    .a(h_a), .b(h_b), .rnd(RND_NEAREST_EVEN), .z(h_mul_z), .status(h_mul_status)
  );

  DW_fp_add #(23, 8, 1) u_fp32_add (
    .a(f_a), .b(f_b), .rnd(RND_NEAREST_EVEN), .z(f_add_z), .status(f_add_status)
  );

  DW_fp_mac #(23, 8, 1) u_fp32_mac (
    .a(f_a), .b(f_b), .c(f_c), .rnd(RND_NEAREST_EVEN), .z(f_mac_z), .status(f_mac_status)
  );

  DW_fp_ifp_conv #(10, 5, 25, 8, 1, 0) u_h_to_ifp (.a(h_a), .z(h_to_ifp));
  DW_ifp_fp_conv #(25, 8, 23, 8, 1) u_ifp_to_f (
    .a(h_to_ifp), .rnd(RND_NEAREST_EVEN), .z(h_to_f_z), .status(h_to_f_status)
  );

  DW_fp_ifp_conv #(23, 8, 25, 8, 1, 0) u_f_to_ifp (.a(f_a), .z(f_to_ifp));
  DW_ifp_fp_conv #(25, 8, 10, 5, 1) u_ifp_to_h (
    .a(f_to_ifp), .rnd(RND_NEAREST_EVEN), .z(f_to_h_z), .status(f_to_h_status)
  );

  DW_fp_exp #(23, 8, 1, 2) u_fp32_exp (.a(32'h00000000), .z(exp_z), .status(exp_status));
  DW_fp_div #(23, 8, 1, 0) u_fp32_div (
    .a(32'h40800000), .b(32'h40000000), .rnd(RND_NEAREST_EVEN), .z(div_z), .status(div_status)
  );
  DW_fp_recip #(23, 8, 1, 0) u_fp32_recip (
    .a(32'h40000000), .rnd(RND_NEAREST_EVEN), .z(recip_z), .status(recip_status)
  );
  DW_fp_sqrt #(23, 8, 1) u_fp32_sqrt (
    .a(32'h40800000), .rnd(RND_NEAREST_EVEN), .z(sqrt_z), .status(sqrt_status)
  );
  DW_fp_invsqrt #(23, 8, 1) u_fp32_invsqrt (
    .a(32'h40800000), .rnd(RND_NEAREST_EVEN), .z(invsqrt_z), .status(invsqrt_status)
  );

  task check16(input [127:0] name, input [15:0] got, input [15:0] exp);
    begin
      if (got !== exp) begin
        $display("DW_PROBE_FAIL %0s got=%h expected=%h", name, got, exp);
        $fatal(1);
      end
    end
  endtask

  task check32(input [127:0] name, input [31:0] got, input [31:0] exp);
    begin
      if (got !== exp) begin
        $display("DW_PROBE_FAIL %0s got=%h expected=%h", name, got, exp);
        $fatal(1);
      end
    end
  endtask

  initial begin
    h_a = 16'h3e00; h_b = 16'hc000; f_a = 32'h3fc00000; f_b = 32'h40100000; f_c = 32'h3f800000;
    #5;
    check16("fp16_mult_1p5_x_minus2", h_mul_z, 16'hc200);
    check32("fp32_add_1p5_plus_2p25", f_add_z, 32'h40700000);
    check32("fp32_mac_1p5_x_2p25_plus_1", f_mac_z, 32'h408c0000);
    $display("DW_PROBE_CONVERSION_UNCONFIRMED h_to_f_z=%h h_to_f_status=%h f_to_h_z=%h f_to_h_status=%h",
             h_to_f_z, h_to_f_status, f_to_h_z, f_to_h_status);
    check32("exp_0", exp_z, 32'h3f800000);
    check32("div_4_by_2", div_z, 32'h40000000);
    check32("recip_2", recip_z, 32'h3f000000);
    check32("sqrt_4", sqrt_z, 32'h40000000);
    check32("invsqrt_4", invsqrt_z, 32'h3f000000);

    h_a = 16'h0000; h_b = 16'hbc00; f_a = 32'h3f800000; f_b = 32'h3f800000; f_c = 32'h00000000;
    #5;
    check16("fp16_mult_zero", h_mul_z, 16'h8000);
    check32("fp32_add_1_plus_1", f_add_z, 32'h40000000);
    check32("fp32_mac_1_x_1", f_mac_z, 32'h3f800000);

    $display("DW_PROBE_PASS");
    $finish;
  end
endmodule
SV

DW_FILES=(
  "$DW_SIM_DIR/DW_fp_addsub.v"
  "$DW_SIM_DIR/DW_fp_dp2.v"
  "$DW_SIM_DIR/DW_exp2.v"
  "$DW_SIM_DIR/DW_inv_sqrt.v"
  "$DW_SIM_DIR/DW_ifp_mult.v"
  "$DW_SIM_DIR/DW_ifp_addsub.v"
  "$DW_SIM_DIR/DW_fp_mult.v"
  "$DW_SIM_DIR/DW_fp_add.v"
  "$DW_SIM_DIR/DW_fp_mac.v"
  "$DW_SIM_DIR/DW_fp_ifp_conv.v"
  "$DW_SIM_DIR/DW_ifp_fp_conv.v"
  "$DW_SIM_DIR/DW_fp_exp.v"
  "$DW_SIM_DIR/DW_fp_div.v"
  "$DW_SIM_DIR/DW_fp_recip.v"
  "$DW_SIM_DIR/DW_fp_sqrt.v"
  "$DW_SIM_DIR/DW_fp_invsqrt.v"
)

{
  echo "Stage 1 DesignWare probe"
  echo "DW sim dir: provided by DW_SIM_DIR"
  echo "Probe dir: build/stage1_dw_probe"
} > "$SUMMARY"

cd "$BUILD_DIR" || exit 2

vcs -full64 -sverilog -timescale=1ns/1ps \
  +incdir+"$DW_SIM_DIR" \
  -Mdir="$BUILD_DIR/csrc" \
  -o "$BUILD_DIR/dw_probe_simv" \
  "${DW_FILES[@]}" "$BUILD_DIR/dw_probe_tb.sv" \
  -top dw_probe_tb \
  -l "$BUILD_DIR/vcs_compile.log"
compile_code=$?

if [ "$compile_code" -ne 0 ]; then
  echo "vcs_compile_exit_code=$compile_code" >> "$SUMMARY"
  echo "result=FAIL" >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$compile_code"
fi

timeout 120s "$BUILD_DIR/dw_probe_simv" -l "$BUILD_DIR/vcs_run.log"
run_code=$?

errors=0
if grep -E "(DW_PROBE_FAIL|Error:|Fatal:|invalid parameter)" "$BUILD_DIR/vcs_run.log" >/dev/null 2>&1; then
  errors=1
fi

if grep -q "DW_PROBE_PASS" "$BUILD_DIR/vcs_run.log" && [ "$run_code" -eq 0 ] && [ "$errors" -eq 0 ]; then
  result=PASS
  exit_code=0
else
  result=FAIL
  exit_code=1
fi

{
  echo "vcs_compile_exit_code=$compile_code"
  echo "vcs_run_exit_code=$run_code"
  echo "probe_errors=$errors"
  echo "result=$result"
  echo
  echo "Run markers:"
  grep -E "^(DW_PROBE_)" "$BUILD_DIR/vcs_run.log" || true
  echo "Full logs: build/stage1_dw_probe/vcs_compile.log and build/stage1_dw_probe/vcs_run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
exit "$exit_code"
