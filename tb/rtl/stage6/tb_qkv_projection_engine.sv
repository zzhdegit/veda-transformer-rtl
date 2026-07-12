`timescale 1ns/1ps
`default_nettype none

module tb_qkv_projection_engine;
`ifndef STAGE6_N_HEAD
    localparam int N_HEAD = 2;
`else
    localparam int N_HEAD = `STAGE6_N_HEAD;
`endif
`ifndef STAGE6_D_HEAD
    localparam int D_HEAD = 8;
`else
    localparam int D_HEAD = `STAGE6_D_HEAD;
`endif
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int D_MODEL = N_HEAD * D_HEAD;
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);

    logic clk;
    logic rst_n;
    logic input_valid;
    logic input_ready;
    logic [MODEL_W-1:0] input_dim;
    logic [15:0] input_data_fp16;
    logic input_last;
    logic [META_W-1:0] input_meta;
    logic weight_valid;
    logic weight_ready;
    logic [1:0] weight_kind;
    logic [MODEL_W-1:0] weight_output_index;
    logic [MODEL_W-1:0] weight_input_index;
    logic [15:0] weight_data_fp16;
    logic weight_last;
    logic weight_commit;
    logic start_valid;
    logic start_ready;
    logic [META_W-1:0] start_meta;
    logic qkv_valid;
    logic qkv_ready;
    logic [HEAD_W-1:0] qkv_head;
    logic [DIM_W-1:0] qkv_dim;
    logic [15:0] qkv_q_fp16;
    logic [15:0] qkv_k_fp16;
    logic [15:0] qkv_v_fp16;
    logic qkv_last_dim;
    logic qkv_last_head;
    logic [META_W-1:0] qkv_meta;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [COUNTER_W-1:0] perf_q_projection_cycles;
    logic [COUNTER_W-1:0] perf_k_projection_cycles;
    logic [COUNTER_W-1:0] perf_v_projection_cycles;
    logic [COUNTER_W-1:0] perf_qkv_quantization_cycles;
    logic [COUNTER_W-1:0] perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;

    logic [15:0] hidden [0:D_MODEL-1];
    logic [15:0] weights [0:2][0:D_MODEL-1][0:D_MODEL-1];
    logic [HEAD_W-1:0] exp_head [0:D_MODEL-1];
    logic [DIM_W-1:0] exp_dim [0:D_MODEL-1];
    logic [15:0] exp_q [0:D_MODEL-1];
    logic [15:0] exp_k [0:D_MODEL-1];
    logic [15:0] exp_v [0:D_MODEL-1];
    logic exp_last_dim [0:D_MODEL-1];
    logic exp_last_head [0:D_MODEL-1];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    qkv_projection_engine #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(1'b0)
    ) u_dut (
        .clk                           (clk),
        .rst_n                         (rst_n),
        .input_valid                   (input_valid),
        .input_ready                   (input_ready),
        .input_dim                     (input_dim),
        .input_data_fp16               (input_data_fp16),
        .input_last                    (input_last),
        .input_meta                    (input_meta),
        .weight_valid                  (weight_valid),
        .weight_ready                  (weight_ready),
        .weight_kind                   (weight_kind),
        .weight_output_index           (weight_output_index),
        .weight_input_index            (weight_input_index),
        .weight_data_fp16              (weight_data_fp16),
        .weight_last                   (weight_last),
        .weight_commit                 (weight_commit),
        .start_valid                   (start_valid),
        .start_ready                   (start_ready),
        .start_meta                    (start_meta),
        .qkv_valid                     (qkv_valid),
        .qkv_ready                     (qkv_ready),
        .qkv_head                      (qkv_head),
        .qkv_dim                       (qkv_dim),
        .qkv_q_fp16                    (qkv_q_fp16),
        .qkv_k_fp16                    (qkv_k_fp16),
        .qkv_v_fp16                    (qkv_v_fp16),
        .qkv_last_dim                  (qkv_last_dim),
        .qkv_last_head                 (qkv_last_head),
        .qkv_meta                      (qkv_meta),
        .done_valid                    (done_valid),
        .done_ready                    (done_ready),
        .done_status                   (done_status),
        .done_invalid                  (done_invalid),
        .done_meta                     (done_meta),
        .perf_q_projection_cycles      (perf_q_projection_cycles),
        .perf_k_projection_cycles      (perf_k_projection_cycles),
        .perf_v_projection_cycles      (perf_v_projection_cycles),
        .perf_qkv_quantization_cycles  (perf_qkv_quantization_cycles),
        .perf_weight_stall_cycles      (perf_weight_stall_cycles),
        .perf_pe_stall_cycles          (perf_pe_stall_cycles),
        .perf_output_stall_cycles      (perf_output_stall_cycles)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE6C_QKV_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic expect_token(input string got, input string expected);
        begin
            if (got != expected) begin
                $display("CHECK_FAIL token got=%s expected=%s", got, expected);
                $fatal(1);
            end
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        string token;
        int idx;
        int got_index;
        int got_head;
        int got_dim;
        int got_last_dim;
        int got_last_head;
        begin
            if (!$value$plusargs("QKV_VECTOR_FILE=%s", path)) tb_fail("missing QKV_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open QKV vector file");
            code = $fscanf(fd, "%s", token);
            expect_token(token, "HIDDEN");
            for (idx = 0; idx < D_MODEL; idx++) begin
                code = $fscanf(fd, "%h", hidden[idx]);
                if (code != 1) tb_fail("hidden parse failed");
            end
            for (int kind = 0; kind < 3; kind++) begin
                code = $fscanf(fd, "%s", token);
                if (kind == 0) expect_token(token, "WQ");
                if (kind == 1) expect_token(token, "WK");
                if (kind == 2) expect_token(token, "WV");
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        code = $fscanf(fd, "%h", weights[kind][out_idx][in_idx]);
                        if (code != 1) tb_fail("weight parse failed");
                    end
                end
            end
            code = $fscanf(fd, "%s", token);
            expect_token(token, "EXPECTED");
            for (idx = 0; idx < D_MODEL; idx++) begin
                code = $fscanf(fd, "%h %h %h %h %h %h %d %d",
                               got_index, got_head, got_dim,
                               exp_q[idx], exp_k[idx], exp_v[idx],
                               got_last_dim, got_last_head);
                if (code != 8) tb_fail("expected parse failed");
                if (got_index != idx) tb_fail("expected index mismatch");
                exp_head[idx] = HEAD_W'(got_head);
                exp_dim[idx] = DIM_W'(got_dim);
                exp_last_dim[idx] = got_last_dim != 0;
                exp_last_head[idx] = got_last_head != 0;
            end
            $fclose(fd);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            input_valid = 1'b0;
            input_dim = '0;
            input_data_fp16 = '0;
            input_last = 1'b0;
            input_meta = '0;
            weight_valid = 1'b0;
            weight_kind = 2'd0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = '0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            start_valid = 1'b0;
            start_meta = '0;
            qkv_ready = 1'b0;
            done_ready = 1'b0;
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic load_weights;
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            for (int kind = 0; kind < 3; kind++) begin
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        drive_valid = 1'b1;
                        wait_cycles = 0;
                        while (drive_valid) begin
                            @(negedge clk);
                            weight_valid = drive_valid;
                            weight_kind = kind[1:0];
                            weight_output_index = MODEL_W'(out_idx);
                            weight_input_index = MODEL_W'(in_idx);
                            weight_data_fp16 = weights[kind][out_idx][in_idx];
                            weight_last = (out_idx == D_MODEL - 1) && (in_idx == D_MODEL - 1);
                            weight_commit = 1'b0;
                            #1;
                            pre_fire = weight_valid && weight_ready;
                            @(posedge clk); #1;
                            if (pre_fire) begin
                                drive_valid = 1'b0;
                                weight_valid = 1'b0;
                            end
                            wait_cycles++;
                            if (wait_cycles > 1000) begin
                                tb_fail($sformatf("weight load handshake timeout kind=%0d out=%0d in=%0d ready=%0b dut_state=%0d proj_state=%0d",
                                                  kind, out_idx, in_idx, weight_ready, u_dut.state_q,
                                                  u_dut.u_projection_controller.state_q));
                            end
                        end
                    end
                end
                @(negedge clk);
                weight_valid = 1'b0;
                weight_kind = kind[1:0];
                weight_commit = 1'b1;
                @(posedge clk); #1;
                weight_commit = 1'b0;
            end
        end
    endtask

    task automatic load_hidden(input logic [META_W-1:0] meta);
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            for (int idx = 0; idx < D_MODEL; idx++) begin
                drive_valid = 1'b1;
                wait_cycles = 0;
                while (drive_valid) begin
                    @(negedge clk);
                    input_valid = drive_valid;
                    input_dim = MODEL_W'(idx);
                    input_data_fp16 = hidden[idx];
                    input_last = idx == D_MODEL - 1;
                    input_meta = meta;
                    #1;
                    pre_fire = input_valid && input_ready;
                    @(posedge clk); #1;
                    if (pre_fire) begin
                        drive_valid = 1'b0;
                        input_valid = 1'b0;
                    end
                    wait_cycles++;
                    if (wait_cycles > 1000) begin
                        tb_fail($sformatf("hidden load handshake timeout idx=%0d input_ready=%0b weight_ready=%0b done_valid=%0b dut_state=%0d proj_state=%0d",
                                          idx, input_ready, weight_ready, done_valid, u_dut.state_q,
                                          u_dut.u_projection_controller.state_q));
                    end
                end
            end
        end
    endtask

    task automatic start_projection(input logic [META_W-1:0] meta);
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            drive_valid = 1'b1;
            wait_cycles = 0;
            while (drive_valid) begin
                @(negedge clk);
                start_valid = drive_valid;
                start_meta = meta;
                #1;
                pre_fire = start_valid && start_ready;
                @(posedge clk); #1;
                if (pre_fire) begin
                    drive_valid = 1'b0;
                    start_valid = 1'b0;
                end
                wait_cycles++;
                if (wait_cycles > 1000) begin
                    tb_fail($sformatf("start handshake timeout start_ready=%0b done_valid=%0b dut_state=%0d proj_state=%0d",
                                      start_ready, done_valid, u_dut.state_q, u_dut.u_projection_controller.state_q));
                end
            end
        end
    endtask

    task automatic drain_qkv(input logic [META_W-1:0] meta);
        int received;
        int cycle;
        logic pre_fire;
        logic [HEAD_W-1:0] pre_head;
        logic [DIM_W-1:0] pre_dim;
        logic [15:0] pre_q;
        logic [15:0] pre_k;
        logic [15:0] pre_v;
        logic pre_last_dim;
        logic pre_last_head;
        logic [META_W-1:0] pre_meta;
        begin
            received = 0;
            cycle = 0;
            while (received < D_MODEL) begin
                @(negedge clk);
                qkv_ready = ((cycle % 5) != 1) && ((cycle % 7) != 3);
                done_ready = 1'b0;
                #1;
                pre_fire = qkv_valid && qkv_ready;
                pre_head = qkv_head;
                pre_dim = qkv_dim;
                pre_q = qkv_q_fp16;
                pre_k = qkv_k_fp16;
                pre_v = qkv_v_fp16;
                pre_last_dim = qkv_last_dim;
                pre_last_head = qkv_last_head;
                pre_meta = qkv_meta;
                @(posedge clk); #1;
                if (pre_fire) begin
                    if (pre_head !== exp_head[received]) tb_fail("qkv head mismatch");
                    if (pre_dim !== exp_dim[received]) tb_fail("qkv dim mismatch");
                    if (pre_q !== exp_q[received]) tb_fail("q mismatch");
                    if (pre_k !== exp_k[received]) tb_fail("k mismatch");
                    if (pre_v !== exp_v[received]) tb_fail("v mismatch");
                    if (pre_last_dim !== exp_last_dim[received]) tb_fail("last_dim mismatch");
                    if (pre_last_head !== exp_last_head[received]) tb_fail("last_head mismatch");
                    if (pre_meta !== meta) tb_fail("meta mismatch");
                    received++;
                end
                cycle++;
                if (cycle > 200000) tb_fail("qkv drain timeout");
            end
            qkv_ready = 1'b0;
        end
    endtask

    task automatic drain_done(input logic [META_W-1:0] meta);
        int cycle;
        begin
            cycle = 0;
            while (!done_valid) begin
                @(negedge clk);
                done_ready = 1'b0;
                @(posedge clk); #1;
                cycle++;
                if (cycle > 1000) tb_fail("done timeout");
            end
            if (done_invalid) tb_fail("done invalid");
            if (done_status[7]) tb_fail("unexpected done status");
            if (done_meta !== meta) tb_fail("done meta mismatch");
            @(negedge clk);
            done_ready = 1'b1;
            @(posedge clk); #1;
            done_ready = 1'b0;
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        load_weights();
        load_hidden(16'h6C01);
        start_projection(16'h6C02);
        drain_qkv(16'h6C02);
        drain_done(16'h6C02);
        load_hidden(16'h6C11);
        start_projection(16'h6C12);
        drain_qkv(16'h6C12);
        drain_done(16'h6C12);
        $display("STAGE6C_QKV_PROJECTION_PASS N_HEAD=%0d D_HEAD=%0d q=%0d k=%0d v=%0d quant=%0d pe_stall=%0d output_stall=%0d",
                 N_HEAD, D_HEAD, perf_q_projection_cycles, perf_k_projection_cycles,
                 perf_v_projection_cycles, perf_qkv_quantization_cycles,
                 perf_pe_stall_cycles, perf_output_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
