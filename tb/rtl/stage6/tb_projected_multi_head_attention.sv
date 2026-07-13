`timescale 1ns/1ps
`default_nettype none

module tb_projected_multi_head_attention;
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
    localparam int MAX_SEQ_LEN = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int D_MODEL = N_HEAD * D_HEAD;
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int MAX_TILES = (D_HEAD + PE_NUM - 1) / PE_NUM;
    localparam int MAX_OUTPUTS = N_HEAD * MAX_TILES;

    logic clk;
    logic rst_n;
    logic hidden_valid;
    logic hidden_ready;
    logic [MODEL_W-1:0] hidden_dim;
    logic [15:0] hidden_data_fp16;
    logic hidden_last;
    logic [META_W-1:0] hidden_meta;
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
    logic output_valid;
    logic output_ready;
    logic [HEAD_W-1:0] output_head;
    logic [DIM_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last_tile;
    logic output_last_head;
    logic output_last_token;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;
    logic [COUNTER_W-1:0] perf_q_projection_cycles;
    logic [COUNTER_W-1:0] perf_k_projection_cycles;
    logic [COUNTER_W-1:0] perf_v_projection_cycles;
    logic [COUNTER_W-1:0] perf_qkv_quantization_cycles;
    logic [COUNTER_W-1:0] perf_attention_cycles;
    logic [COUNTER_W-1:0] perf_generation_steps;
    logic [COUNTER_W-1:0] perf_total_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    logic [15:0] weights [0:2][0:D_MODEL-1][0:D_MODEL-1];
    logic [15:0] current_hidden [0:D_MODEL-1];
    logic [HEAD_W-1:0] exp_head [0:MAX_OUTPUTS-1];
    logic [DIM_W-1:0] exp_base [0:MAX_OUTPUTS-1];
    logic [PE_NUM-1:0] exp_mask [0:MAX_OUTPUTS-1];
    logic [PE_NUM*32-1:0] exp_vector [0:MAX_OUTPUTS-1];
    logic exp_last_tile [0:MAX_OUTPUTS-1];
    logic exp_last_head [0:MAX_OUTPUTS-1];
    logic exp_last_token [0:MAX_OUTPUTS-1];
    string current_name;
    logic [META_W-1:0] current_meta;
    int current_seq_before;
    int current_seq_after;
    bit current_expect_invalid;
    logic [7:0] current_expect_status;
    int exp_count;
    int step_run_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    projected_multi_head_attention #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(1'b0)
    ) u_dut (
        .clk                           (clk),
        .rst_n                         (rst_n),
        .hidden_valid                  (hidden_valid),
        .hidden_ready                  (hidden_ready),
        .hidden_dim                    (hidden_dim),
        .hidden_data_fp16              (hidden_data_fp16),
        .hidden_last                   (hidden_last),
        .hidden_meta                   (hidden_meta),
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
        .output_valid                  (output_valid),
        .output_ready                  (output_ready),
        .output_head                   (output_head),
        .output_base_dim               (output_base_dim),
        .output_vector_fp32            (output_vector_fp32),
        .output_lane_mask              (output_lane_mask),
        .output_status                 (output_status),
        .output_invalid                (output_invalid),
        .output_meta                   (output_meta),
        .output_last_tile              (output_last_tile),
        .output_last_head              (output_last_head),
        .output_last_token             (output_last_token),
        .done_valid                    (done_valid),
        .done_ready                    (done_ready),
        .done_status                   (done_status),
        .done_invalid                  (done_invalid),
        .done_meta                     (done_meta),
        .done_valid_seq_len            (done_valid_seq_len),
        .current_valid_seq_len         (current_valid_seq_len),
        .perf_q_projection_cycles      (perf_q_projection_cycles),
        .perf_k_projection_cycles      (perf_k_projection_cycles),
        .perf_v_projection_cycles      (perf_v_projection_cycles),
        .perf_qkv_quantization_cycles  (perf_qkv_quantization_cycles),
        .perf_attention_cycles         (perf_attention_cycles),
        .perf_generation_steps         (perf_generation_steps),
        .perf_total_cycles             (perf_total_cycles),
        .perf_pe_stall_cycles          (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles         (perf_sfu_stall_cycles),
        .perf_weight_stall_cycles      (perf_weight_stall_cycles),
        .perf_buffer_stall_cycles      (perf_buffer_stall_cycles),
        .perf_output_stall_cycles      (perf_output_stall_cycles),
        .perf_peak_valid_seq_len       (perf_peak_valid_seq_len)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE6D_PROJECTED_MHA_FAIL N_HEAD=%0d D_HEAD=%0d step=%s: %s",
                     N_HEAD, D_HEAD, current_name, message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            hidden_valid = 1'b0;
            hidden_dim = '0;
            hidden_data_fp16 = 16'd0;
            hidden_last = 1'b0;
            hidden_meta = '0;
            weight_valid = 1'b0;
            weight_kind = 2'd0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = 16'd0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            start_valid = 1'b0;
            start_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic load_weights_to_dut;
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
                            weight_valid = 1'b1;
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
                            if (wait_cycles > 1000) tb_fail("weight handshake timeout");
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

    task automatic drive_hidden;
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            for (int idx = 0; idx < D_MODEL; idx++) begin
                drive_valid = 1'b1;
                wait_cycles = 0;
                while (drive_valid) begin
                    @(negedge clk);
                    hidden_valid = 1'b1;
                    hidden_dim = MODEL_W'(idx);
                    hidden_data_fp16 = current_hidden[idx];
                    hidden_last = idx == D_MODEL - 1;
                    hidden_meta = current_meta;
                    #1;
                    pre_fire = hidden_valid && hidden_ready;
                    @(posedge clk); #1;
                    if (pre_fire) begin
                        drive_valid = 1'b0;
                        hidden_valid = 1'b0;
                    end
                    wait_cycles++;
                    if (wait_cycles > 1000) tb_fail("hidden handshake timeout");
                end
            end
        end
    endtask

    task automatic start_projection;
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            drive_valid = 1'b1;
            wait_cycles = 0;
            while (drive_valid) begin
                @(negedge clk);
                start_valid = 1'b1;
                start_meta = current_meta;
                #1;
                pre_fire = start_valid && start_ready;
                @(posedge clk); #1;
                if (pre_fire) begin
                    drive_valid = 1'b0;
                    start_valid = 1'b0;
                end
                wait_cycles++;
                if (wait_cycles > 1000) tb_fail("start handshake timeout");
            end
        end
    endtask

    task automatic run_current_step;
        int out_idx;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        logic [HEAD_W-1:0] pre_head;
        logic [DIM_W-1:0] pre_base;
        logic [PE_NUM-1:0] pre_mask;
        logic [PE_NUM*32-1:0] pre_vector;
        logic [7:0] pre_output_status;
        logic pre_output_invalid;
        logic [META_W-1:0] pre_output_meta;
        logic pre_last_tile;
        logic pre_last_head;
        logic pre_last_token;
        logic [7:0] pre_done_status;
        logic pre_done_invalid;
        logic [META_W-1:0] pre_done_meta;
        logic [SEQ_LEN_W-1:0] pre_done_seq_len;
        bit done_seen;
        begin
            out_idx = 0;
            cycle = 0;
            done_seen = 1'b0;
            drive_hidden();
            start_projection();
            while (!done_seen) begin
                @(negedge clk);
                output_ready = ((cycle % 5) != 1) && ((cycle % 11) != 7);
                done_ready = ((cycle % 7) != 3);
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                pre_head = output_head;
                pre_base = output_base_dim;
                pre_mask = output_lane_mask;
                pre_vector = output_vector_fp32;
                pre_output_status = output_status;
                pre_output_invalid = output_invalid;
                pre_output_meta = output_meta;
                pre_last_tile = output_last_tile;
                pre_last_head = output_last_head;
                pre_last_token = output_last_token;
                pre_done_status = done_status;
                pre_done_invalid = done_invalid;
                pre_done_meta = done_meta;
                pre_done_seq_len = done_valid_seq_len;
                @(posedge clk); #1;

                if (pre_out_fire) begin
                    if (out_idx >= exp_count) tb_fail("too many output tiles");
                    if (pre_head !== exp_head[out_idx]) tb_fail("output head mismatch");
                    if (pre_base !== exp_base[out_idx]) tb_fail("output base mismatch");
                    if (pre_mask !== exp_mask[out_idx]) tb_fail("output mask mismatch");
                    if (pre_vector !== exp_vector[out_idx]) begin
                        $display("CHECK_FAIL stage6d N_HEAD=%0d D_HEAD=%0d step=%s out=%0d got=%h expected=%h",
                                 N_HEAD, D_HEAD, current_name, out_idx, pre_vector, exp_vector[out_idx]);
                        $fatal(1);
                    end
                    if (pre_output_invalid !== current_expect_invalid) tb_fail("output invalid mismatch");
                    if (pre_output_meta !== current_meta) tb_fail("output metadata mismatch");
                    if (pre_last_tile !== exp_last_tile[out_idx]) tb_fail("last_tile mismatch");
                    if (pre_last_head !== exp_last_head[out_idx]) tb_fail("last_head mismatch");
                    if (pre_last_token !== exp_last_token[out_idx]) tb_fail("last_token mismatch");
                    if (^pre_output_status === 1'bx) tb_fail("unknown output status");
                    out_idx++;
                end

                if (pre_done_fire) begin
                    if (out_idx != exp_count) tb_fail("done before all output tiles");
                    if (pre_done_invalid !== current_expect_invalid) tb_fail("done invalid mismatch");
                    if (current_expect_invalid && (pre_done_status !== current_expect_status)) tb_fail("done status mismatch");
                    if (pre_done_meta !== current_meta) tb_fail("done metadata mismatch");
                    if (pre_done_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("done valid_seq_len mismatch");
                    if (current_valid_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("current valid_seq_len mismatch");
                    $display("STAGE6D_PROJECTED_MHA_PERF N_HEAD=%0d D_HEAD=%0d step=%s seq_before=%0d seq_after=%0d q=%0d k=%0d v=%0d qkv_quant=%0d attention=%0d total=%0d pe_stall=%0d sfu_stall=%0d weight_stall=%0d buffer_stall=%0d output_stall=%0d peak_seq=%0d",
                             N_HEAD, D_HEAD, current_name, current_seq_before, current_seq_after,
                             perf_q_projection_cycles, perf_k_projection_cycles, perf_v_projection_cycles,
                             perf_qkv_quantization_cycles, perf_attention_cycles, perf_total_cycles,
                             perf_pe_stall_cycles, perf_sfu_stall_cycles, perf_weight_stall_cycles,
                             perf_buffer_stall_cycles, perf_output_stall_cycles, perf_peak_valid_seq_len);
                    done_seen = 1'b1;
                    done_ready = 1'b0;
                    output_ready = 1'b0;
                end

                cycle++;
                if (cycle > 4000000) tb_fail("projected MHA timeout");
            end
        end
    endtask

    task automatic parse_and_run_file;
        string path;
        int fd;
        string tag;
        string name;
        int code;
        int seq_before;
        int seq_after;
        int expect_invalid;
        int head;
        int base;
        int last_tile;
        int out_last_head;
        int out_last_token;
        logic [META_W-1:0] meta;
        logic [7:0] status;
        logic [PE_NUM-1:0] mask;
        logic [31:0] values [0:PE_NUM-1];
        begin
            if (!$value$plusargs("PROJECTED_MHA_VECTOR_FILE=%s", path)) tb_fail("missing +PROJECTED_MHA_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open projected MHA vector file");

            for (int kind = 0; kind < 3; kind++) begin
                code = $fscanf(fd, "%s", tag);
                if (kind == 0 && tag != "WQ") tb_fail("expected WQ");
                if (kind == 1 && tag != "WK") tb_fail("expected WK");
                if (kind == 2 && tag != "WV") tb_fail("expected WV");
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        code = $fscanf(fd, "%h", weights[kind][out_idx][in_idx]);
                        if (code != 1) tb_fail("weight parse failed");
                    end
                end
            end

            load_weights_to_dut();
            step_run_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code != 1) begin
                    void'($fgets(tag, fd));
                end else if (tag == "STEP") begin
                    code = $fscanf(fd, "%s %h %d %d %d %h\n", name, meta, seq_before, seq_after, expect_invalid, status);
                    if (code != 6) tb_fail("bad STEP line");
                    current_name = name;
                    current_meta = meta;
                    current_seq_before = seq_before;
                    current_seq_after = seq_after;
                    current_expect_invalid = expect_invalid[0];
                    current_expect_status = status;
                    exp_count = 0;
                    if (current_valid_seq_len !== SEQ_LEN_W'(seq_before)) tb_fail("pre-step valid_seq_len mismatch");
                end else if (tag == "H") begin
                    for (int idx = 0; idx < D_MODEL; idx++) begin
                        code = $fscanf(fd, "%h", current_hidden[idx]);
                        if (code != 1) tb_fail("hidden parse failed");
                    end
                end else if (tag == "O") begin
                    code = $fscanf(fd, "%d %d %h %h %h %h %h %h %h %h %h %d %d %d\n",
                        head, base, mask,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        last_tile, out_last_head, out_last_token);
                    if (code != 14) tb_fail("bad O line");
                    if (exp_count >= MAX_OUTPUTS) tb_fail("too many expected output tiles");
                    exp_head[exp_count] = HEAD_W'(head);
                    exp_base[exp_count] = DIM_W'(base);
                    exp_mask[exp_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        exp_vector[exp_count][lane*32 +: 32] = values[lane];
                    end
                    exp_last_tile[exp_count] = last_tile[0];
                    exp_last_head[exp_count] = out_last_head[0];
                    exp_last_token[exp_count] = out_last_token[0];
                    exp_count++;
                end else if (tag == "RUN") begin
                    run_current_step();
                    step_run_count++;
                end else if (tag == "END") begin
                    // Step boundary.
                end else begin
                    tb_fail({"unknown vector tag ", tag});
                end
            end
            $fclose(fd);
            if (step_run_count != (MAX_SEQ_LEN + 1)) tb_fail("did not execute all projected MHA steps");
        end
    endtask

    initial begin
        current_name = "none";
        step_run_count = 0;
        apply_reset();
        parse_and_run_file();
        $display("STAGE6D_PROJECTED_MHA_PASS N_HEAD=%0d D_HEAD=%0d generation_steps=%0d",
                 N_HEAD, D_HEAD, perf_generation_steps);
        $finish;
    end
endmodule

`default_nettype wire
