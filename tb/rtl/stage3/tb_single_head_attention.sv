`timescale 1ns/1ps
`default_nettype none

`ifndef STAGE3_D_HEAD
`define STAGE3_D_HEAD 8
`endif

`ifndef STAGE3_ATTENTION_PE_ARCH
`define STAGE3_ATTENTION_PE_ARCH 0
`endif

module tb_single_head_attention;
    localparam int PE_NUM = 8;
    localparam int D_HEAD = `STAGE3_D_HEAD;
    localparam int MAX_SEQ_LEN = 32;
    localparam int META_W = 16;
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int D_ADDR_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);
    localparam int MAX_TILES = (D_HEAD + PE_NUM - 1) / PE_NUM;

    logic clk;
    logic rst_n;
    logic load_valid;
    logic load_ready;
    logic [1:0] load_kind;
    logic [TOKEN_W-1:0] load_token;
    logic [D_ADDR_W-1:0] load_dim;
    logic [15:0] load_data;
    logic start_valid;
    logic start_ready;
    logic [SEQ_LEN_W-1:0] start_seq_len;
    logic [META_W-1:0] start_meta;
    logic output_valid;
    logic output_ready;
    logic [D_ADDR_W-1:0] output_base_dim;
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

    logic [63:0] perf_total_attention_cycles;
    logic [63:0] perf_qk_cycles;
    logic [63:0] perf_qk_pe_busy_cycles;
    logic [63:0] perf_scale_cycles;
    logic [63:0] perf_reduction_cycles;
    logic [63:0] perf_reduction_finalize_cycles;
    logic [63:0] perf_normalization_cycles;
    logic [63:0] perf_sv_cycles;
    logic [63:0] perf_pe_stall_cycles;
    logic [63:0] perf_sfu_stall_cycles;
    logic [63:0] perf_buffer_stall_cycles;
    logic [63:0] perf_output_stall_cycles;
    logic [63:0] perf_score_buffer_peak_occupancy;
    logic [63:0] perf_paper_array_active_cycles;
    logic [63:0] perf_paper_array_idle_cycles;
    logic [63:0] perf_inner_mode_cycles;
    logic [63:0] perf_outer_mode_cycles;
    logic [63:0] perf_group0_active_cycles;
    logic [63:0] perf_group1_active_cycles;
    logic [63:0] perf_tail_masked_pe_cycles;
    logic [63:0] perf_mode_switch_cycles;
    logic [63:0] perf_array_input_stall_cycles;
    logic [63:0] perf_array_output_stall_cycles;

    int current_seq_len;
    logic [META_W-1:0] current_meta;
    string current_name;
    logic [D_ADDR_W-1:0] exp_base [0:MAX_TILES-1];
    logic [PE_NUM-1:0] exp_mask [0:MAX_TILES-1];
    logic [PE_NUM*32-1:0] exp_vector [0:MAX_TILES-1];
    logic exp_last [0:MAX_TILES-1];
    int exp_count;
    int case_run_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    single_head_attention #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .ATTENTION_PE_ARCH(`STAGE3_ATTENTION_PE_ARCH)
    ) u_dut (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .load_valid                       (load_valid),
        .load_ready                       (load_ready),
        .load_kind                        (load_kind),
        .load_token                       (load_token),
        .load_dim                         (load_dim),
        .load_data                        (load_data),
        .start_valid                      (start_valid),
        .start_ready                      (start_ready),
        .start_seq_len                    (start_seq_len),
        .start_meta                       (start_meta),
        .output_valid                     (output_valid),
        .output_ready                     (output_ready),
        .output_base_dim                  (output_base_dim),
        .output_vector_fp32               (output_vector_fp32),
        .output_lane_mask                 (output_lane_mask),
        .output_status                    (output_status),
        .output_invalid                   (output_invalid),
        .output_meta                      (output_meta),
        .output_last                      (output_last),
        .done_valid                       (done_valid),
        .done_ready                       (done_ready),
        .done_status                      (done_status),
        .done_invalid                     (done_invalid),
        .done_meta                        (done_meta),
        .perf_total_attention_cycles      (perf_total_attention_cycles),
        .perf_qk_cycles                   (perf_qk_cycles),
        .perf_qk_pe_busy_cycles           (perf_qk_pe_busy_cycles),
        .perf_scale_cycles                (perf_scale_cycles),
        .perf_reduction_cycles            (perf_reduction_cycles),
        .perf_reduction_finalize_cycles   (perf_reduction_finalize_cycles),
        .perf_normalization_cycles        (perf_normalization_cycles),
        .perf_sv_cycles                   (perf_sv_cycles),
        .perf_pe_stall_cycles             (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles            (perf_sfu_stall_cycles),
        .perf_buffer_stall_cycles         (perf_buffer_stall_cycles),
        .perf_output_stall_cycles         (perf_output_stall_cycles),
        .perf_score_buffer_peak_occupancy (perf_score_buffer_peak_occupancy),
        .perf_paper_array_active_cycles   (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles     (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles           (perf_inner_mode_cycles),
        .perf_outer_mode_cycles           (perf_outer_mode_cycles),
        .perf_group0_active_cycles        (perf_group0_active_cycles),
        .perf_group1_active_cycles        (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles       (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles          (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles    (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles   (perf_array_output_stall_cycles)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE3_ATTENTION_TB_FAIL D_HEAD=%0d case=%s: %s", D_HEAD, current_name, message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            load_valid = 1'b0;
            load_kind = 2'd0;
            load_token = '0;
            load_dim = '0;
            load_data = '0;
            start_valid = 1'b0;
            start_seq_len = '0;
            start_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic drive_load(input logic [1:0] kind, input int token, input int dim, input logic [15:0] data);
        logic pre_fire;
        begin
            @(negedge clk);
            load_valid = 1'b1;
            load_kind = kind;
            load_token = TOKEN_W'(token);
            load_dim = D_ADDR_W'(dim);
            load_data = data;
            do begin
                #1;
                pre_fire = load_valid && load_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            load_valid = 1'b0;
        end
    endtask

    task automatic drive_start(input int seq_len, input logic [META_W-1:0] meta);
        logic pre_fire;
        begin
            @(negedge clk);
            start_valid = 1'b1;
            start_seq_len = SEQ_LEN_W'(seq_len);
            start_meta = meta;
            do begin
                #1;
                pre_fire = start_valid && start_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task automatic run_current_case;
        int out_idx;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        logic [D_ADDR_W-1:0] pre_base;
        logic [PE_NUM-1:0] pre_mask;
        logic [PE_NUM*32-1:0] pre_vector;
        logic [7:0] pre_status;
        logic pre_invalid;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        logic done_seen;
        begin
            drive_start(current_seq_len, current_meta);
            out_idx = 0;
            cycle = 0;
            done_seen = 1'b0;
            while (!done_seen) begin
                @(negedge clk);
                output_ready = ((cycle % 5) != 1) && ((cycle % 11) != 7);
                done_ready = ((cycle % 7) != 3);
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                pre_base = output_base_dim;
                pre_mask = output_lane_mask;
                pre_vector = output_vector_fp32;
                pre_status = output_status;
                pre_invalid = output_invalid;
                pre_meta = output_meta;
                pre_last = output_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (out_idx >= exp_count) tb_fail("too many output tiles");
                    if (pre_base !== exp_base[out_idx]) tb_fail("output base dim mismatch");
                    if (pre_mask !== exp_mask[out_idx]) tb_fail("output lane mask mismatch");
                    if (pre_vector !== exp_vector[out_idx]) begin
                        $display("CHECK_FAIL attention D_HEAD=%0d case=%s tile=%0d got=%h expected=%h",
                                 D_HEAD, current_name, out_idx, pre_vector, exp_vector[out_idx]);
                        $fatal(1);
                    end
                    if (pre_invalid) tb_fail("unexpected output invalid");
                    if (^pre_status === 1'bx) tb_fail("unknown output status");
                    if (pre_meta !== current_meta) tb_fail("output metadata mismatch");
                    if (pre_last !== exp_last[out_idx]) tb_fail("output last mismatch");
                    out_idx++;
                end
                if (pre_done_fire) begin
                    if (out_idx != exp_count) tb_fail("done before all output tiles");
                    if (done_invalid) tb_fail("unexpected done invalid");
                    if (^done_status === 1'bx) tb_fail("unknown done status");
                    if (done_meta !== current_meta) tb_fail("done metadata mismatch");
                    $display("STAGE3_ATTENTION_PERF arch=%0d D_HEAD=%0d case=%s seq_len=%0d total=%0d qk=%0d qk_busy=%0d scale=%0d reduction=%0d reduction_finalize=%0d normalization=%0d sv=%0d pe_stall=%0d sfu_stall=%0d buffer_stall=%0d output_stall=%0d score_peak=%0d paper_active=%0d paper_inner=%0d paper_outer=%0d paper_tail=%0d paper_mode_switch=%0d paper_out_stall=%0d",
                             `STAGE3_ATTENTION_PE_ARCH, D_HEAD, current_name, current_seq_len,
                             perf_total_attention_cycles, perf_qk_cycles, perf_qk_pe_busy_cycles,
                             perf_scale_cycles, perf_reduction_cycles, perf_reduction_finalize_cycles,
                             perf_normalization_cycles, perf_sv_cycles, perf_pe_stall_cycles,
                             perf_sfu_stall_cycles, perf_buffer_stall_cycles, perf_output_stall_cycles,
                             perf_score_buffer_peak_occupancy, perf_paper_array_active_cycles,
                             perf_inner_mode_cycles, perf_outer_mode_cycles,
                             perf_tail_masked_pe_cycles, perf_mode_switch_cycles,
                             perf_array_output_stall_cycles);
                    done_ready = 1'b0;
                    output_ready = 1'b0;
                    done_seen = 1'b1;
                end
                cycle++;
                if (cycle > 500000) tb_fail("attention timeout");
            end
        end
    endtask

    task automatic parse_and_run_file;
        string path;
        int fd;
        string tag;
        string name;
        int seq_len;
        int token;
        int dim;
        int base;
        logic [META_W-1:0] meta;
        logic [15:0] data16;
        logic [PE_NUM-1:0] mask;
        logic [31:0] values [0:PE_NUM-1];
        int last;
        int code;
        begin
            if (!$value$plusargs("ATTENTION_VECTOR_FILE=%s", path)) tb_fail("missing +ATTENTION_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open attention vector file");
            case_run_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code != 1) begin
                    void'($fgets(tag, fd));
                end else if (tag == "CASE") begin
                    code = $fscanf(fd, "%s %d %h\n", name, seq_len, meta);
                    if (code != 3) tb_fail("bad CASE line");
                    current_name = name;
                    current_seq_len = seq_len;
                    current_meta = meta;
                    exp_count = 0;
                end else if (tag == "Q") begin
                    code = $fscanf(fd, "%d %h\n", dim, data16);
                    if (code != 2) tb_fail("bad Q line");
                    drive_load(2'd0, 0, dim, data16);
                end else if (tag == "K") begin
                    code = $fscanf(fd, "%d %d %h\n", token, dim, data16);
                    if (code != 3) tb_fail("bad K line");
                    drive_load(2'd1, token, dim, data16);
                end else if (tag == "V") begin
                    code = $fscanf(fd, "%d %d %h\n", token, dim, data16);
                    if (code != 3) tb_fail("bad V line");
                    drive_load(2'd2, token, dim, data16);
                end else if (tag == "O") begin
                    code = $fscanf(fd, "%d %h %h %h %h %h %h %h %h %h %d\n",
                        base, mask,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        last);
                    if (code != 11) tb_fail("bad O line");
                    exp_base[exp_count] = D_ADDR_W'(base);
                    exp_mask[exp_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        exp_vector[exp_count][lane*32 +: 32] = values[lane];
                    end
                    exp_last[exp_count] = last[0];
                    exp_count++;
                end else if (tag == "RUN") begin
                    if (exp_count == 0) tb_fail("RUN without expected outputs");
                    run_current_case();
                    case_run_count++;
                end else if (tag == "END") begin
                    // Case boundary; the next CASE line overwrites case-local state.
                end else begin
                    tb_fail({"unknown vector tag ", tag});
                end
            end
            $fclose(fd);
            if (case_run_count != 4) tb_fail("did not execute all vector cases");
        end
    endtask

    initial begin
        current_name = "none";
        case_run_count = 0;
        apply_reset();
        parse_and_run_file();
        $display("STAGE3_SINGLE_HEAD_ATTENTION_PASS D_HEAD=%0d", D_HEAD);
        $finish;
    end
endmodule

`default_nettype wire
