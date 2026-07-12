`timescale 1ns/1ps
`default_nettype none

`ifndef STAGE4_D_HEAD
`define STAGE4_D_HEAD 8
`endif

module tb_generation_attention_engine;
    localparam int PE_NUM = 8;
    localparam int D_HEAD = `STAGE4_D_HEAD;
    localparam int MAX_SEQ_LEN = 8;
    localparam int META_W = 16;
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int MAX_TILES = (D_HEAD + PE_NUM - 1) / PE_NUM;

    logic clk;
    logic rst_n;
    logic token_valid;
    logic token_ready;
    logic [DIM_W-1:0] token_dim;
    logic [15:0] token_q_fp16;
    logic [15:0] token_k_fp16;
    logic [15:0] token_v_fp16;
    logic token_last_dim;
    logic [META_W-1:0] token_meta;
    logic output_valid;
    logic output_ready;
    logic [DIM_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;

    logic [63:0] perf_generation_steps;
    logic [63:0] perf_provisional_append_cycles;
    logic [63:0] perf_attention_cycles;
    logic [63:0] perf_commit_cycles;
    logic [63:0] perf_cache_read_cycles;
    logic [63:0] perf_cache_write_cycles;
    logic [63:0] perf_cache_stall_cycles;
    logic [63:0] perf_pe_stall_cycles;
    logic [63:0] perf_sfu_stall_cycles;
    logic [63:0] perf_total_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    string current_name;
    logic [META_W-1:0] current_meta;
    int current_seq_before;
    int current_seq_after;
    bit current_expect_invalid;
    logic [7:0] current_expect_status;
    logic [DIM_W-1:0] exp_base [0:MAX_TILES-1];
    logic [PE_NUM-1:0] exp_mask [0:MAX_TILES-1];
    logic [PE_NUM*32-1:0] exp_vector [0:MAX_TILES-1];
    logic exp_last [0:MAX_TILES-1];
    int exp_count;
    int step_run_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    generation_attention_engine #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W)
    ) u_dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .token_valid             (token_valid),
        .token_ready             (token_ready),
        .token_dim               (token_dim),
        .token_q_fp16            (token_q_fp16),
        .token_k_fp16            (token_k_fp16),
        .token_v_fp16            (token_v_fp16),
        .token_last_dim          (token_last_dim),
        .token_meta              (token_meta),
        .output_valid            (output_valid),
        .output_ready            (output_ready),
        .output_base_dim         (output_base_dim),
        .output_vector_fp32      (output_vector_fp32),
        .output_lane_mask        (output_lane_mask),
        .output_status           (output_status),
        .output_invalid          (output_invalid),
        .output_meta             (output_meta),
        .output_last             (output_last),
        .done_valid              (done_valid),
        .done_ready              (done_ready),
        .done_status             (done_status),
        .done_invalid            (done_invalid),
        .done_meta               (done_meta),
        .done_valid_seq_len      (done_valid_seq_len),
        .current_valid_seq_len   (current_valid_seq_len),
        .perf_generation_steps   (perf_generation_steps),
        .perf_provisional_append_cycles (perf_provisional_append_cycles),
        .perf_attention_cycles   (perf_attention_cycles),
        .perf_commit_cycles      (perf_commit_cycles),
        .perf_cache_read_cycles  (perf_cache_read_cycles),
        .perf_cache_write_cycles (perf_cache_write_cycles),
        .perf_cache_stall_cycles (perf_cache_stall_cycles),
        .perf_pe_stall_cycles    (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles   (perf_sfu_stall_cycles),
        .perf_total_cycles       (perf_total_cycles),
        .perf_peak_valid_seq_len (perf_peak_valid_seq_len)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE4_GENERATION_TB_FAIL D_HEAD=%0d step=%s: %s", D_HEAD, current_name, message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            token_valid = 1'b0;
            token_dim = '0;
            token_q_fp16 = 16'd0;
            token_k_fp16 = 16'd0;
            token_v_fp16 = 16'd0;
            token_last_dim = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic drive_token_dim(
        input int dim,
        input logic [15:0] q_data,
        input logic [15:0] k_data,
        input logic [15:0] v_data,
        input bit last_dim
    );
        logic pre_fire;
        begin
            if ((dim % 3) == 1) begin
                @(negedge clk);
                token_valid = 1'b0;
            end
            @(negedge clk);
            token_valid = 1'b1;
            token_dim = DIM_W'(dim);
            token_q_fp16 = q_data;
            token_k_fp16 = k_data;
            token_v_fp16 = v_data;
            token_last_dim = last_dim;
            token_meta = current_meta;
            do begin
                #1;
                pre_fire = token_valid && token_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            token_valid = 1'b0;
        end
    endtask

    task automatic drive_probe_token(input logic [META_W-1:0] meta);
        begin
            current_meta = meta;
            for (int dim = 0; dim < D_HEAD; dim++) begin
                drive_token_dim(
                    dim,
                    16'h3c00,
                    (dim[0] ? 16'hbc00 : 16'h4000),
                    16'h3800 + dim[15:0],
                    dim == D_HEAD - 1
                );
            end
        end
    endtask

    task automatic wait_for_internal_high(input string what, input int max_cycles);
        int cycles;
        begin
            cycles = 0;
            if (what == "append") begin
                while (u_dut.cache_append_valid !== 1'b1) begin
                    @(posedge clk); #1;
                    cycles++;
                    if (cycles > max_cycles) tb_fail("timeout waiting for internal append");
                end
            end else if (what == "sha_start") begin
                while (u_dut.sha_start_valid !== 1'b1) begin
                    @(posedge clk); #1;
                    cycles++;
                    if (cycles > max_cycles) tb_fail("timeout waiting for internal attention start");
                end
            end else begin
                tb_fail("unknown internal wait selector");
            end
        end
    endtask

    task automatic reset_during_provisional_append;
        begin
            current_name = "reset_provisional_append";
            apply_reset();
            drive_probe_token(16'h5a01);
            wait_for_internal_high("append", 1000);
            apply_reset();
            if (current_valid_seq_len !== '0) tb_fail("reset during provisional append left valid_seq_len nonzero");
            if (!token_ready) tb_fail("token_ready not restored after provisional reset");
        end
    endtask

    task automatic reset_during_attention;
        begin
            current_name = "reset_attention";
            apply_reset();
            drive_probe_token(16'h5a02);
            wait_for_internal_high("sha_start", 100000);
            @(posedge clk); #1;
            apply_reset();
            if (current_valid_seq_len !== '0) tb_fail("reset during attention left valid_seq_len nonzero");
            if (!token_ready) tb_fail("token_ready not restored after attention reset");
        end
    endtask

    task automatic run_current_step;
        int out_idx;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        logic [DIM_W-1:0] pre_base;
        logic [PE_NUM-1:0] pre_mask;
        logic [PE_NUM*32-1:0] pre_vector;
        logic [7:0] pre_output_status;
        logic pre_output_invalid;
        logic [META_W-1:0] pre_output_meta;
        logic pre_last;
        logic [7:0] pre_done_status;
        logic pre_done_invalid;
        logic [META_W-1:0] pre_done_meta;
        logic [SEQ_LEN_W-1:0] pre_done_seq_len;
        bit done_seen;
        begin
            out_idx = 0;
            cycle = 0;
            done_seen = 1'b0;
            while (!done_seen) begin
                @(negedge clk);
                output_ready = ((cycle % 5) != 2) && ((cycle % 13) != 9);
                done_ready = ((cycle % 7) != 4);
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                pre_base = output_base_dim;
                pre_mask = output_lane_mask;
                pre_vector = output_vector_fp32;
                pre_output_status = output_status;
                pre_output_invalid = output_invalid;
                pre_output_meta = output_meta;
                pre_last = output_last;
                pre_done_status = done_status;
                pre_done_invalid = done_invalid;
                pre_done_meta = done_meta;
                pre_done_seq_len = done_valid_seq_len;
                @(posedge clk); #1;

                if (pre_out_fire) begin
                    if (out_idx >= exp_count) tb_fail("too many output tiles");
                    if (pre_base !== exp_base[out_idx]) tb_fail("output base dim mismatch");
                    if (pre_mask !== exp_mask[out_idx]) tb_fail("output lane mask mismatch");
                    if (pre_vector !== exp_vector[out_idx]) begin
                        $display("CHECK_FAIL generation D_HEAD=%0d step=%s tile=%0d got=%h expected=%h",
                                 D_HEAD, current_name, out_idx, pre_vector, exp_vector[out_idx]);
                        $fatal(1);
                    end
                    if (pre_output_invalid !== current_expect_invalid) tb_fail("output invalid mismatch");
                    if (pre_output_meta !== current_meta) tb_fail("output metadata mismatch");
                    if (pre_last !== exp_last[out_idx]) tb_fail("output last mismatch");
                    if (^pre_output_status === 1'bx) tb_fail("unknown output status");
                    out_idx++;
                end

                if (pre_done_fire) begin
                    if (out_idx != exp_count) tb_fail("done before all output tiles");
                    if (pre_done_invalid !== current_expect_invalid) tb_fail("done invalid mismatch");
                    if (^pre_done_status === 1'bx) tb_fail("unknown done status");
                    if (current_expect_invalid && (pre_done_status !== current_expect_status)) tb_fail("done status mismatch");
                    if (pre_done_meta !== current_meta) tb_fail("done metadata mismatch");
                    if (pre_done_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("done valid_seq_len mismatch");
                    if (current_valid_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("current valid_seq_len mismatch");
                    $display("STAGE4_GENERATION_PERF D_HEAD=%0d step=%s seq_before=%0d seq_after=%0d total=%0d attention=%0d provisional_append=%0d commit=%0d cache_read=%0d cache_write=%0d cache_stall=%0d pe_stall=%0d sfu_stall=%0d peak_seq=%0d",
                             D_HEAD, current_name, current_seq_before, current_seq_after,
                             perf_total_cycles, perf_attention_cycles, perf_provisional_append_cycles,
                             perf_commit_cycles,
                             perf_cache_read_cycles, perf_cache_write_cycles, perf_cache_stall_cycles,
                             perf_pe_stall_cycles, perf_sfu_stall_cycles, perf_peak_valid_seq_len);
                    done_seen = 1'b1;
                    done_ready = 1'b0;
                    output_ready = 1'b0;
                end

                cycle++;
                if (cycle > 1000000) tb_fail("generation timeout");
            end
        end
    endtask

    task automatic parse_and_run_file;
        string path;
        int fd;
        string tag;
        string name;
        int seq_before;
        int seq_after;
        int expect_invalid;
        int dim;
        int base;
        int last;
        int code;
        logic [META_W-1:0] meta;
        logic [7:0] status;
        logic [15:0] q_data;
        logic [15:0] k_data;
        logic [15:0] v_data;
        logic [PE_NUM-1:0] mask;
        logic [31:0] values [0:PE_NUM-1];
        begin
            if (!$value$plusargs("GENERATION_VECTOR_FILE=%s", path)) tb_fail("missing +GENERATION_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open generation vector file");
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
                end else if (tag == "T") begin
                    code = $fscanf(fd, "%d %h %h %h %d\n", dim, q_data, k_data, v_data, last);
                    if (code != 5) tb_fail("bad T line");
                    drive_token_dim(dim, q_data, k_data, v_data, last[0]);
                end else if (tag == "O") begin
                    code = $fscanf(fd, "%d %h %h %h %h %h %h %h %h %h %d\n",
                        base, mask,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        last);
                    if (code != 11) tb_fail("bad O line");
                    exp_base[exp_count] = DIM_W'(base);
                    exp_mask[exp_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        exp_vector[exp_count][lane*32 +: 32] = values[lane];
                    end
                    exp_last[exp_count] = last[0];
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
            if (step_run_count != (MAX_SEQ_LEN + 1)) tb_fail("did not execute all generation steps");
        end
    endtask

    initial begin
        current_name = "none";
        step_run_count = 0;
        apply_reset();
        reset_during_provisional_append();
        reset_during_attention();
        current_name = "none";
        parse_and_run_file();
        $display("STAGE4_GENERATION_ATTENTION_PASS D_HEAD=%0d", D_HEAD);
        $finish;
    end
endmodule

`default_nettype wire
