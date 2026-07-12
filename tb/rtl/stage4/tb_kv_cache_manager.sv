`timescale 1ns/1ps
`default_nettype none

module tb_kv_cache_manager;
    localparam int D_HEAD = 4;
    localparam int MAX_SEQ_LEN = 3;
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);

    logic clk;
    logic rst_n;
    logic rd_valid;
    logic rd_ready;
    logic [TOKEN_W-1:0] rd_token;
    logic [DIM_W-1:0] rd_dim;
    logic provisional_read_enable;
    logic rd_rsp_valid;
    logic rd_rsp_ready;
    logic [TOKEN_W-1:0] rd_rsp_token;
    logic [DIM_W-1:0] rd_rsp_dim;
    logic [15:0] rd_rsp_k_fp16;
    logic [15:0] rd_rsp_v_fp16;
    logic append_valid;
    logic append_ready;
    logic [TOKEN_W-1:0] append_token_index;
    logic [DIM_W-1:0] append_dim;
    logic [15:0] append_k_fp16;
    logic [15:0] append_v_fp16;
    logic append_last_dim;
    logic append_complete;
    logic commit_valid;
    logic commit_ready;
    logic [TOKEN_W-1:0] commit_token_index;
    logic abort_valid;
    logic [SEQ_LEN_W-1:0] valid_seq_len;
    logic append_incomplete;
    logic provisional_valid;
    logic [TOKEN_W-1:0] provisional_token_index;
    logic cache_full;
    logic error_valid;
    logic error_ready;
    logic [7:0] error_code;
    logic [63:0] perf_cache_read_cycles;
    logic [63:0] perf_cache_write_cycles;
    logic [63:0] perf_cache_stall_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    kv_cache_manager #(
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) u_dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .rd_valid                (rd_valid),
        .rd_ready                (rd_ready),
        .rd_token                (rd_token),
        .rd_dim                  (rd_dim),
        .provisional_read_enable (provisional_read_enable),
        .rd_rsp_valid            (rd_rsp_valid),
        .rd_rsp_ready            (rd_rsp_ready),
        .rd_rsp_token            (rd_rsp_token),
        .rd_rsp_dim              (rd_rsp_dim),
        .rd_rsp_k_fp16           (rd_rsp_k_fp16),
        .rd_rsp_v_fp16           (rd_rsp_v_fp16),
        .append_valid            (append_valid),
        .append_ready            (append_ready),
        .append_token_index      (append_token_index),
        .append_dim              (append_dim),
        .append_k_fp16           (append_k_fp16),
        .append_v_fp16           (append_v_fp16),
        .append_last_dim         (append_last_dim),
        .append_complete         (append_complete),
        .commit_valid            (commit_valid),
        .commit_ready            (commit_ready),
        .commit_token_index      (commit_token_index),
        .abort_valid             (abort_valid),
        .valid_seq_len           (valid_seq_len),
        .append_incomplete       (append_incomplete),
        .provisional_valid       (provisional_valid),
        .provisional_token_index (provisional_token_index),
        .cache_full              (cache_full),
        .error_valid             (error_valid),
        .error_ready             (error_ready),
        .error_code              (error_code),
        .perf_cache_read_cycles  (perf_cache_read_cycles),
        .perf_cache_write_cycles (perf_cache_write_cycles),
        .perf_cache_stall_cycles (perf_cache_stall_cycles),
        .perf_peak_valid_seq_len (perf_peak_valid_seq_len)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE4_KV_CACHE_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            rd_valid = 1'b0;
            rd_token = '0;
            rd_dim = '0;
            provisional_read_enable = 1'b0;
            rd_rsp_ready = 1'b0;
            append_valid = 1'b0;
            append_token_index = '0;
            append_dim = '0;
            append_k_fp16 = 16'd0;
            append_v_fp16 = 16'd0;
            append_last_dim = 1'b0;
            append_complete = 1'b0;
            commit_valid = 1'b0;
            commit_token_index = '0;
            abort_valid = 1'b0;
            error_ready = 1'b1;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic append_one_dim(input int token, input int dim, input logic [15:0] k_data, input logic [15:0] v_data);
        logic pre_fire;
        begin
            @(negedge clk);
            append_valid = 1'b1;
            append_token_index = TOKEN_W'(token);
            append_dim = DIM_W'(dim);
            append_k_fp16 = k_data;
            append_v_fp16 = v_data;
            append_last_dim = (dim == D_HEAD - 1);
            append_complete = (dim == D_HEAD - 1);
            do begin
                #1;
                pre_fire = append_valid && append_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            append_valid = 1'b0;
        end
    endtask

    task automatic append_one_token_provisional(input int token);
        begin
            for (int dim = 0; dim < D_HEAD; dim++) begin
                append_one_dim(token, dim, 16'h1000 + token * 16 + dim, 16'h2000 + token * 16 + dim);
            end
            if (valid_seq_len !== SEQ_LEN_W'(token)) tb_fail("valid_seq_len changed before commit");
            if (!provisional_valid) tb_fail("provisional token not complete");
            if (provisional_token_index !== TOKEN_W'(token)) tb_fail("provisional token index mismatch");
        end
    endtask

    task automatic commit_one_token(input int token);
        logic pre_fire;
        begin
            @(negedge clk);
            commit_valid = 1'b1;
            commit_token_index = TOKEN_W'(token);
            do begin
                #1;
                pre_fire = commit_valid && commit_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            commit_valid = 1'b0;
            if (valid_seq_len !== SEQ_LEN_W'(token + 1)) tb_fail("valid_seq_len did not increment on commit");
            if (provisional_valid || append_incomplete) tb_fail("provisional state not cleared on commit");
        end
    endtask

    task automatic append_and_commit_token(input int token);
        begin
            append_one_token_provisional(token);
            commit_one_token(token);
        end
    endtask

    task automatic read_one_dim(
        input int token,
        input int dim,
        input logic [15:0] exp_k,
        input logic [15:0] exp_v,
        input bit include_provisional
    );
        logic pre_fire;
        logic pre_rsp_fire;
        int cycles;
        begin
            @(negedge clk);
            rd_valid = 1'b1;
            rd_token = TOKEN_W'(token);
            rd_dim = DIM_W'(dim);
            provisional_read_enable = include_provisional;
            rd_rsp_ready = 1'b1;
            do begin
                #1;
                pre_fire = rd_valid && rd_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            rd_valid = 1'b0;
            cycles = 0;
            do begin
                #1;
                pre_rsp_fire = rd_rsp_valid && rd_rsp_ready;
                if (pre_rsp_fire) begin
                    if (rd_rsp_token !== TOKEN_W'(token)) tb_fail("read token mismatch");
                    if (rd_rsp_dim !== DIM_W'(dim)) tb_fail("read dim mismatch");
                    if (rd_rsp_k_fp16 !== exp_k) tb_fail("read K mismatch");
                    if (rd_rsp_v_fp16 !== exp_v) tb_fail("read V mismatch");
                end
                @(posedge clk); #1;
                cycles++;
                if (cycles > 16) tb_fail("read response timeout");
            end while (!pre_rsp_fire);
            @(negedge clk);
            rd_rsp_ready = 1'b0;
            provisional_read_enable = 1'b0;
        end
    endtask

    task automatic test_append_stall_during_pending_read;
        logic pre_fire;
        logic pre_rsp_fire;
        int cycles;
        begin
            @(negedge clk);
            rd_valid = 1'b1;
            rd_token = '0;
            rd_dim = '0;
            rd_rsp_ready = 1'b0;
            provisional_read_enable = 1'b0;
            do begin
                #1;
                pre_fire = rd_valid && rd_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            rd_valid = 1'b0;
            cycles = 0;
            while (!rd_rsp_valid) begin
                @(posedge clk); #1;
                cycles++;
                if (cycles > 16) tb_fail("pending read response timeout");
            end

            @(negedge clk);
            append_valid = 1'b1;
            append_token_index = TOKEN_W'(1);
            append_dim = '0;
            append_last_dim = 1'b0;
            append_complete = 1'b0;
            #1;
            if (append_ready) tb_fail("append_ready asserted while read response pending");
            @(negedge clk);
            append_valid = 1'b0;
            rd_rsp_ready = 1'b1;
            do begin
                #1;
                pre_rsp_fire = rd_rsp_valid && rd_rsp_ready;
                @(posedge clk); #1;
            end while (!pre_rsp_fire);
            @(negedge clk);
            rd_rsp_ready = 1'b0;
        end
    endtask

    initial begin
        apply_reset();
        if (valid_seq_len !== '0) tb_fail("reset valid_seq_len not zero");

        append_one_dim(0, 0, 16'h1111, 16'h2222);
        if (!append_incomplete) tb_fail("append_incomplete not set after partial token");
        if (valid_seq_len !== '0) tb_fail("valid_seq_len changed during partial append");
        @(negedge clk);
        rd_valid = 1'b1;
        rd_token = '0;
        rd_dim = '0;
        provisional_read_enable = 1'b1;
        #1;
        if (rd_ready) tb_fail("read was not stalled during incomplete provisional append");
        @(negedge clk);
        rd_valid = 1'b0;
        apply_reset();
        if (valid_seq_len !== '0 || append_incomplete || provisional_valid) tb_fail("reset did not clear provisional state");

        append_one_dim(0, 0, 16'h1111, 16'h2222);
        append_one_dim(0, 1, 16'h1112, 16'h2223);
        append_one_dim(0, 2, 16'h1113, 16'h2224);
        append_one_dim(0, 3, 16'h1114, 16'h2225);
        if (valid_seq_len !== '0) tb_fail("valid_seq_len incremented before explicit commit");
        if (!provisional_valid) tb_fail("final append did not complete provisional token");
        read_one_dim(0, 0, 16'h1111, 16'h2222, 1'b1);
        commit_one_token(0);
        read_one_dim(0, 0, 16'h1111, 16'h2222, 1'b0);
        read_one_dim(0, 3, 16'h1114, 16'h2225, 1'b0);

        test_append_stall_during_pending_read();

        append_one_dim(1, 0, 16'h3333, 16'h4444);
        if (!append_incomplete) tb_fail("append_incomplete not set before abort");
        @(negedge clk);
        abort_valid = 1'b1;
        @(posedge clk); #1;
        @(negedge clk);
        abort_valid = 1'b0;
        if (valid_seq_len !== SEQ_LEN_W'(1)) tb_fail("abort changed valid_seq_len");
        if (append_incomplete || provisional_valid) tb_fail("abort did not clear provisional state");

        append_and_commit_token(1);
        append_and_commit_token(2);
        if (!cache_full) tb_fail("cache_full not set");
        @(negedge clk);
        append_valid = 1'b1;
        append_token_index = TOKEN_W'(3);
        append_dim = '0;
        append_last_dim = 1'b0;
        append_complete = 1'b0;
        #1;
        if (append_ready) tb_fail("append_ready asserted while full");
        @(negedge clk);
        append_valid = 1'b0;
        if (perf_peak_valid_seq_len !== SEQ_LEN_W'(MAX_SEQ_LEN)) tb_fail("peak valid_seq_len mismatch");
        $display("STAGE4_KV_CACHE_MANAGER_PASS read_cycles=%0d write_cycles=%0d stall_cycles=%0d peak_seq=%0d",
                 perf_cache_read_cycles, perf_cache_write_cycles, perf_cache_stall_cycles, perf_peak_valid_seq_len);
        $finish;
    end
endmodule

`default_nettype wire
