`timescale 1ns/1ps
`default_nettype none

module tb_multi_head_kv_cache_manager;
    localparam int N_HEAD = 2;
    localparam int D_HEAD = 4;
    localparam int MAX_SEQ_LEN = 2;
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD);
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD);

    logic clk;
    logic rst_n;
    logic rd_valid;
    logic rd_ready;
    logic [HEAD_W-1:0] rd_head;
    logic [TOKEN_W-1:0] rd_token;
    logic [DIM_W-1:0] rd_dim;
    logic provisional_read_enable;
    logic rd_rsp_valid;
    logic rd_rsp_ready;
    logic [HEAD_W-1:0] rd_rsp_head;
    logic [TOKEN_W-1:0] rd_rsp_token;
    logic [DIM_W-1:0] rd_rsp_dim;
    logic [15:0] rd_rsp_k_fp16;
    logic [15:0] rd_rsp_v_fp16;
    logic append_valid;
    logic append_ready;
    logic [HEAD_W-1:0] append_head;
    logic [TOKEN_W-1:0] append_token_index;
    logic [DIM_W-1:0] append_dim;
    logic [15:0] append_k_fp16;
    logic [15:0] append_v_fp16;
    logic append_last_dim;
    logic append_last_head;
    logic append_complete;
    logic commit_valid;
    logic commit_ready;
    logic [TOKEN_W-1:0] commit_token_index;
    logic abort_valid;
    logic [SEQ_LEN_W-1:0] valid_seq_len;
    logic append_incomplete;
    logic provisional_valid;
    logic [N_HEAD-1:0] provisional_head_valid;
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

    multi_head_kv_cache_manager #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) u_dut (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .rd_valid                (rd_valid),
        .rd_ready                (rd_ready),
        .rd_head                 (rd_head),
        .rd_token                (rd_token),
        .rd_dim                  (rd_dim),
        .provisional_read_enable (provisional_read_enable),
        .rd_rsp_valid            (rd_rsp_valid),
        .rd_rsp_ready            (rd_rsp_ready),
        .rd_rsp_head             (rd_rsp_head),
        .rd_rsp_token            (rd_rsp_token),
        .rd_rsp_dim              (rd_rsp_dim),
        .rd_rsp_k_fp16           (rd_rsp_k_fp16),
        .rd_rsp_v_fp16           (rd_rsp_v_fp16),
        .append_valid            (append_valid),
        .append_ready            (append_ready),
        .append_head             (append_head),
        .append_token_index      (append_token_index),
        .append_dim              (append_dim),
        .append_k_fp16           (append_k_fp16),
        .append_v_fp16           (append_v_fp16),
        .append_last_dim         (append_last_dim),
        .append_last_head        (append_last_head),
        .append_complete         (append_complete),
        .commit_valid            (commit_valid),
        .commit_ready            (commit_ready),
        .commit_token_index      (commit_token_index),
        .abort_valid             (abort_valid),
        .valid_seq_len           (valid_seq_len),
        .append_incomplete       (append_incomplete),
        .provisional_valid       (provisional_valid),
        .provisional_head_valid  (provisional_head_valid),
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
            $display("STAGE5_KV_CACHE_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            rd_valid = 1'b0;
            rd_head = '0;
            rd_token = '0;
            rd_dim = '0;
            provisional_read_enable = 1'b0;
            rd_rsp_ready = 1'b0;
            append_valid = 1'b0;
            append_head = '0;
            append_token_index = '0;
            append_dim = '0;
            append_k_fp16 = 16'd0;
            append_v_fp16 = 16'd0;
            append_last_dim = 1'b0;
            append_last_head = 1'b0;
            append_complete = 1'b0;
            commit_valid = 1'b0;
            commit_token_index = '0;
            abort_valid = 1'b0;
            error_ready = 1'b1;
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic append_one_dim(input int head, input int token, input int dim);
        logic pre_fire;
        begin
            @(negedge clk);
            append_valid = 1'b1;
            append_head = HEAD_W'(head);
            append_token_index = TOKEN_W'(token);
            append_dim = DIM_W'(dim);
            append_k_fp16 = 16'h4100 + head[15:0] * 16'h10 + dim[15:0];
            append_v_fp16 = 16'h5100 + head[15:0] * 16'h10 + dim[15:0];
            append_last_dim = dim == D_HEAD - 1;
            append_last_head = head == N_HEAD - 1 && dim == D_HEAD - 1;
            append_complete = head == N_HEAD - 1 && dim == D_HEAD - 1;
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

    task automatic append_token(input int token);
        begin
            for (int head = 0; head < N_HEAD; head++) begin
                for (int dim = 0; dim < D_HEAD; dim++) begin
                    append_one_dim(head, token, dim);
                end
            end
        end
    endtask

    task automatic read_one_dim(input int head, input int token, input int dim, input bit provisional);
        logic pre_fire;
        begin
            @(negedge clk);
            rd_valid = 1'b1;
            rd_head = HEAD_W'(head);
            rd_token = TOKEN_W'(token);
            rd_dim = DIM_W'(dim);
            provisional_read_enable = provisional;
            rd_rsp_ready = 1'b0;
            do begin
                #1;
                pre_fire = rd_valid && rd_ready;
                @(posedge clk); #1;
                if (!pre_fire) @(negedge clk);
            end while (!pre_fire);
            @(negedge clk);
            rd_valid = 1'b0;
            repeat (2) @(posedge clk);
            if (!rd_rsp_valid) tb_fail("read response not held under backpressure");
            if (rd_rsp_head !== HEAD_W'(head)) tb_fail("read response head mismatch");
            if (rd_rsp_token !== TOKEN_W'(token)) tb_fail("read response token mismatch");
            if (rd_rsp_dim !== DIM_W'(dim)) tb_fail("read response dim mismatch");
            @(negedge clk);
            rd_rsp_ready = 1'b1;
            @(posedge clk); #1;
            @(negedge clk);
            rd_rsp_ready = 1'b0;
        end
    endtask

    task automatic commit_token(input int token);
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
        end
    endtask

    initial begin
        apply_reset();
        if (valid_seq_len !== '0) tb_fail("reset valid_seq_len not zero");

        for (int dim = 0; dim < D_HEAD; dim++) begin
            append_one_dim(0, 0, dim);
        end
        if (valid_seq_len !== '0) tb_fail("valid_seq_len incremented before commit");
        if (provisional_valid) tb_fail("all-head provisional completed after only head0");
        if (provisional_head_valid !== 2'b01) tb_fail("head0 provisional complete not visible");
        if (rd_ready) tb_fail("cache accepted reads before all-head provisional token completed");

        for (int dim = 0; dim < D_HEAD; dim++) begin
            append_one_dim(1, 0, dim);
        end
        if (!provisional_valid) tb_fail("all-head provisional not complete");
        if (provisional_token_index !== '0) tb_fail("provisional token index mismatch");
        read_one_dim(0, 0, 2, 1'b1);
        read_one_dim(1, 0, 1, 1'b1);
        if (valid_seq_len !== '0) tb_fail("valid_seq_len incremented before commit after all heads");
        commit_token(0);
        if (valid_seq_len !== SEQ_LEN_W'(1)) tb_fail("valid_seq_len did not increment on commit");

        append_one_dim(0, 1, 0);
        if (!append_incomplete) tb_fail("append_incomplete not set for partial token");
        @(negedge clk);
        abort_valid = 1'b1;
        @(posedge clk); #1;
        @(negedge clk);
        abort_valid = 1'b0;
        if (append_incomplete || provisional_valid || (provisional_head_valid != '0)) tb_fail("abort did not clear provisional state");
        if (valid_seq_len !== SEQ_LEN_W'(1)) tb_fail("abort changed committed valid_seq_len");

        append_token(1);
        commit_token(1);
        if (!cache_full) tb_fail("cache_full not set at MAX_SEQ_LEN");
        if (valid_seq_len !== SEQ_LEN_W'(2)) tb_fail("valid_seq_len mismatch at full");
        $display("STAGE5_KV_CACHE_MANAGER_PASS read_cycles=%0d write_cycles=%0d stall_cycles=%0d peak_seq=%0d",
                 perf_cache_read_cycles, perf_cache_write_cycles, perf_cache_stall_cycles, perf_peak_valid_seq_len);
        $finish;
    end
endmodule

`default_nettype wire
