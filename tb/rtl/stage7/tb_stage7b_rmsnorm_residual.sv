`timescale 1ns/1ps
`default_nettype none

module tb_stage7b_rmsnorm_residual;
`ifndef STAGE7_D_MODEL
    localparam int D_MODEL = 8;
`else
    localparam int D_MODEL = `STAGE7_D_MODEL;
`endif
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int DIM_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);

    logic clk;
    logic rst_n;

    logic rms_clear;
    logic rms_gamma_valid;
    logic rms_gamma_ready;
    logic [DIM_W-1:0] rms_gamma_dim;
    logic [15:0] rms_gamma_data_fp16;
    logic rms_gamma_commit;
    logic rms_input_valid;
    logic rms_input_ready;
    logic [DIM_W-1:0] rms_input_dim;
    logic [31:0] rms_input_data_fp32;
    logic rms_input_last;
    logic [META_W-1:0] rms_input_meta;
    logic rms_input_commit;
    logic rms_start_valid;
    logic rms_start_ready;
    logic [META_W-1:0] rms_start_meta;
    logic rms_output_valid;
    logic rms_output_ready;
    logic [DIM_W-1:0] rms_output_dim;
    logic [15:0] rms_output_data_fp16;
    logic [7:0] rms_output_status;
    logic rms_output_invalid;
    logic [META_W-1:0] rms_output_meta;
    logic rms_output_last;
    logic rms_done_valid;
    logic rms_done_ready;
    logic [7:0] rms_done_status;
    logic rms_done_invalid;
    logic [META_W-1:0] rms_done_meta;
    logic [31:0] rms_debug_sum_sq;
    logic [31:0] rms_debug_inv_rms;
    logic [COUNTER_W-1:0] rms_perf_reduce_cycles;
    logic [COUNTER_W-1:0] rms_perf_apply_cycles;
    logic [COUNTER_W-1:0] rms_perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] rms_perf_output_stall_cycles;

    logic res_clear;
    logic res_input_valid;
    logic res_input_ready;
    logic [DIM_W-1:0] res_input_dim;
    logic [31:0] res_input_lhs_fp32;
    logic [31:0] res_input_rhs_fp32;
    logic res_input_last;
    logic [META_W-1:0] res_input_meta;
    logic res_input_commit;
    logic res_start_valid;
    logic res_start_ready;
    logic [META_W-1:0] res_start_meta;
    logic res_output_valid;
    logic res_output_ready;
    logic [DIM_W-1:0] res_output_dim;
    logic [31:0] res_output_data_fp32;
    logic [7:0] res_output_status;
    logic res_output_invalid;
    logic [META_W-1:0] res_output_meta;
    logic res_output_last;
    logic res_done_valid;
    logic res_done_ready;
    logic [7:0] res_done_status;
    logic res_done_invalid;
    logic [META_W-1:0] res_done_meta;
    logic [COUNTER_W-1:0] res_perf_add_cycles;
    logic [COUNTER_W-1:0] res_perf_output_stall_cycles;

    logic [31:0] norm_input_fp32 [0:D_MODEL-1];
    logic [15:0] norm_gamma_fp16 [0:D_MODEL-1];
    logic [15:0] norm_expected_fp16 [0:D_MODEL-1];
    logic [31:0] residual_lhs_fp32 [0:D_MODEL-1];
    logic [31:0] residual_rhs_fp32 [0:D_MODEL-1];
    logic [31:0] residual_expected_fp32 [0:D_MODEL-1];
    logic [31:0] expected_sum_sq;
    logic [31:0] expected_inv_rms;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    rmsnorm_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_rmsnorm (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .clear                     (rms_clear),
        .gamma_valid               (rms_gamma_valid),
        .gamma_ready               (rms_gamma_ready),
        .gamma_dim                 (rms_gamma_dim),
        .gamma_data_fp16           (rms_gamma_data_fp16),
        .gamma_commit              (rms_gamma_commit),
        .input_valid               (rms_input_valid),
        .input_ready               (rms_input_ready),
        .input_dim                 (rms_input_dim),
        .input_data_fp32           (rms_input_data_fp32),
        .input_last                (rms_input_last),
        .input_meta                (rms_input_meta),
        .input_commit              (rms_input_commit),
        .start_valid               (rms_start_valid),
        .start_ready               (rms_start_ready),
        .start_meta                (rms_start_meta),
        .output_valid              (rms_output_valid),
        .output_ready              (rms_output_ready),
        .output_dim                (rms_output_dim),
        .output_data_fp16          (rms_output_data_fp16),
        .output_status             (rms_output_status),
        .output_invalid            (rms_output_invalid),
        .output_meta               (rms_output_meta),
        .output_last               (rms_output_last),
        .done_valid                (rms_done_valid),
        .done_ready                (rms_done_ready),
        .done_status               (rms_done_status),
        .done_invalid              (rms_done_invalid),
        .done_meta                 (rms_done_meta),
        .debug_sum_sq              (rms_debug_sum_sq),
        .debug_inv_rms             (rms_debug_inv_rms),
        .perf_reduce_cycles        (rms_perf_reduce_cycles),
        .perf_apply_cycles         (rms_perf_apply_cycles),
        .perf_sfu_stall_cycles     (rms_perf_sfu_stall_cycles),
        .perf_output_stall_cycles  (rms_perf_output_stall_cycles)
    );

    residual_add_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_residual (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .clear                     (res_clear),
        .input_valid               (res_input_valid),
        .input_ready               (res_input_ready),
        .input_dim                 (res_input_dim),
        .input_lhs_fp32            (res_input_lhs_fp32),
        .input_rhs_fp32            (res_input_rhs_fp32),
        .input_last                (res_input_last),
        .input_meta                (res_input_meta),
        .input_commit              (res_input_commit),
        .start_valid               (res_start_valid),
        .start_ready               (res_start_ready),
        .start_meta                (res_start_meta),
        .output_valid              (res_output_valid),
        .output_ready              (res_output_ready),
        .output_dim                (res_output_dim),
        .output_data_fp32          (res_output_data_fp32),
        .output_status             (res_output_status),
        .output_invalid            (res_output_invalid),
        .output_meta               (res_output_meta),
        .output_last               (res_output_last),
        .done_valid                (res_done_valid),
        .done_ready                (res_done_ready),
        .done_status               (res_done_status),
        .done_invalid              (res_done_invalid),
        .done_meta                 (res_done_meta),
        .perf_add_cycles           (res_perf_add_cycles),
        .perf_output_stall_cycles  (res_perf_output_stall_cycles)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE7B_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        string tag;
        int d_value;
        int dim;
        logic [31:0] a;
        logic [31:0] b;
        logic [31:0] c;
        logic [15:0] h;
        begin
            if (!$value$plusargs("STAGE7B_VECTOR_FILE=%s", path)) begin
                tb_fail("missing +STAGE7B_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) begin
                tb_fail("could not open vector file");
            end
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code == 1) begin
                    if (tag == "D") begin
                        code = $fscanf(fd, "%d\n", d_value);
                        if (d_value != D_MODEL) tb_fail("D_MODEL mismatch");
                    end else if (tag == "R") begin
                        code = $fscanf(fd, "%h %h\n", expected_sum_sq, expected_inv_rms);
                    end else if (tag == "N") begin
                        code = $fscanf(fd, "%h %h %h %h\n", dim, a, h, norm_expected_fp16[dim]);
                        norm_input_fp32[dim] = a;
                        norm_gamma_fp16[dim] = h;
                    end else if (tag == "A") begin
                        code = $fscanf(fd, "%h %h %h %h\n", dim, a, b, c);
                        residual_lhs_fp32[dim] = a;
                        residual_rhs_fp32[dim] = b;
                        residual_expected_fp32[dim] = c;
                    end else begin
                        tb_fail("unknown vector tag");
                    end
                end
            end
            $fclose(fd);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            rms_clear = 1'b0;
            rms_gamma_valid = 1'b0;
            rms_gamma_dim = '0;
            rms_gamma_data_fp16 = '0;
            rms_gamma_commit = 1'b0;
            rms_input_valid = 1'b0;
            rms_input_dim = '0;
            rms_input_data_fp32 = '0;
            rms_input_last = 1'b0;
            rms_input_meta = '0;
            rms_input_commit = 1'b0;
            rms_start_valid = 1'b0;
            rms_start_meta = '0;
            rms_output_ready = 1'b0;
            rms_done_ready = 1'b0;
            res_clear = 1'b0;
            res_input_valid = 1'b0;
            res_input_dim = '0;
            res_input_lhs_fp32 = '0;
            res_input_rhs_fp32 = '0;
            res_input_last = 1'b0;
            res_input_meta = '0;
            res_input_commit = 1'b0;
            res_start_valid = 1'b0;
            res_start_meta = '0;
            res_output_ready = 1'b0;
            res_done_ready = 1'b0;
            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic load_rmsnorm;
        begin
            for (int dim = 0; dim < D_MODEL; dim++) begin
                @(negedge clk);
                rms_gamma_valid = 1'b1;
                rms_gamma_dim = DIM_W'(dim);
                rms_gamma_data_fp16 = norm_gamma_fp16[dim];
                rms_gamma_commit = dim == D_MODEL - 1;
                rms_input_valid = 1'b1;
                rms_input_dim = DIM_W'(dim);
                rms_input_data_fp32 = norm_input_fp32[dim];
                rms_input_last = dim == D_MODEL - 1;
                rms_input_meta = 16'h7001;
                rms_input_commit = dim == D_MODEL - 1;
                #1;
                if (!rms_gamma_ready || !rms_input_ready) begin
                    tb_fail("rmsnorm load not ready");
                end
                @(posedge clk);
            end
            @(negedge clk);
            rms_gamma_valid = 1'b0;
            rms_gamma_commit = 1'b0;
            rms_input_valid = 1'b0;
            rms_input_last = 1'b0;
            rms_input_commit = 1'b0;
        end
    endtask

    task automatic run_rmsnorm;
        int received;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        begin
            @(negedge clk);
            rms_start_valid = 1'b1;
            rms_start_meta = 16'h7001;
            #1;
            if (!rms_start_ready) tb_fail("rmsnorm start not ready");
            @(posedge clk);
            @(negedge clk);
            rms_start_valid = 1'b0;
            received = 0;
            cycle = 0;
            while (received < D_MODEL || !rms_done_valid) begin
                @(negedge clk);
                rms_output_ready = (cycle % 5) != 2;
                rms_done_ready = 1'b1;
                #1;
                pre_out_fire = rms_output_valid && rms_output_ready;
                pre_done_fire = rms_done_valid && rms_done_ready;
                if (pre_out_fire) begin
                    if (rms_output_dim !== DIM_W'(received)) tb_fail("rmsnorm output dimension mismatch");
                    if (rms_output_data_fp16 !== norm_expected_fp16[received]) begin
                        $display("CHECK_FAIL rmsnorm dim=%0d got=%04h expected=%04h",
                                 received, rms_output_data_fp16, norm_expected_fp16[received]);
                        $fatal(1);
                    end
                    if (rms_output_invalid) tb_fail("rmsnorm output invalid");
                    if (rms_output_meta !== 16'h7001) tb_fail("rmsnorm metadata mismatch");
                    if (rms_output_last !== (received == D_MODEL - 1)) tb_fail("rmsnorm last mismatch");
                    received++;
                end
                if (pre_done_fire) begin
                    if (received != D_MODEL) tb_fail("rmsnorm done before all outputs");
                    if (rms_done_invalid) tb_fail("rmsnorm done invalid");
                    if (rms_debug_sum_sq !== expected_sum_sq) begin
                        $display("CHECK_FAIL rmsnorm sum_sq got=%08h expected=%08h inv=%08h expected_inv=%08h",
                                 rms_debug_sum_sq, expected_sum_sq, rms_debug_inv_rms, expected_inv_rms);
                        $fatal(1);
                    end
                end
                @(posedge clk);
                cycle++;
                if (cycle > 20000) tb_fail("rmsnorm timeout");
            end
            @(negedge clk);
            rms_output_ready = 1'b0;
            rms_done_ready = 1'b0;
        end
    endtask

    task automatic load_residual;
        begin
            for (int dim = 0; dim < D_MODEL; dim++) begin
                @(negedge clk);
                res_input_valid = 1'b1;
                res_input_dim = DIM_W'(dim);
                res_input_lhs_fp32 = residual_lhs_fp32[dim];
                res_input_rhs_fp32 = residual_rhs_fp32[dim];
                res_input_last = dim == D_MODEL - 1;
                res_input_meta = 16'h7002;
                res_input_commit = dim == D_MODEL - 1;
                #1;
                if (!res_input_ready) tb_fail("residual load not ready");
                @(posedge clk);
            end
            @(negedge clk);
            res_input_valid = 1'b0;
            res_input_last = 1'b0;
            res_input_commit = 1'b0;
        end
    endtask

    task automatic run_residual;
        int received;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        begin
            @(negedge clk);
            res_start_valid = 1'b1;
            res_start_meta = 16'h7002;
            #1;
            if (!res_start_ready) tb_fail("residual start not ready");
            @(posedge clk);
            @(negedge clk);
            res_start_valid = 1'b0;
            received = 0;
            cycle = 0;
            while (received < D_MODEL || !res_done_valid) begin
                @(negedge clk);
                res_output_ready = (cycle % 4) != 1;
                res_done_ready = 1'b1;
                #1;
                pre_out_fire = res_output_valid && res_output_ready;
                pre_done_fire = res_done_valid && res_done_ready;
                if (pre_out_fire) begin
                    if (res_output_dim !== DIM_W'(received)) tb_fail("residual output dimension mismatch");
                    if (res_output_data_fp32 !== residual_expected_fp32[received]) begin
                        $display("CHECK_FAIL residual dim=%0d got=%08h expected=%08h",
                                 received, res_output_data_fp32, residual_expected_fp32[received]);
                        $fatal(1);
                    end
                    if (res_output_invalid) tb_fail("residual output invalid");
                    if (res_output_meta !== 16'h7002) tb_fail("residual metadata mismatch");
                    if (res_output_last !== (received == D_MODEL - 1)) tb_fail("residual last mismatch");
                    received++;
                end
                if (pre_done_fire) begin
                    if (received != D_MODEL) tb_fail("residual done before all outputs");
                    if (res_done_invalid) tb_fail("residual done invalid");
                end
                @(posedge clk);
                cycle++;
                if (cycle > 20000) tb_fail("residual timeout");
            end
            @(negedge clk);
            res_output_ready = 1'b0;
            res_done_ready = 1'b0;
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        load_rmsnorm();
        run_rmsnorm();
        load_residual();
        run_residual();
        $display("STAGE7B_RMSNORM_RESIDUAL_PASS d_model=%0d", D_MODEL);
        $finish;
    end
endmodule

`default_nettype wire
