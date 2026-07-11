`timescale 1ns/1ps
`default_nettype none

module tb_stage1_all;
    localparam int STREAM_DATA_W = 16;
    localparam int STREAM_META_W = 4;
    localparam int FIFO_DATA_W = 16;
    localparam int FIFO_META_W = 4;

    logic clk;
    logic rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task automatic tb_fail(input string message);
        begin
            $display("STAGE1_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic check_eq(input string name, input int got, input int expected);
        begin
            if (got !== expected) begin
                $display("CHECK_FAIL %s got=%0d expected=%0d", name, got, expected);
                $fatal(1);
            end
        end
    endtask

    task automatic check_bits(input string name, input logic [63:0] got, input logic [63:0] expected);
        begin
            if (got !== expected) begin
                $display("CHECK_FAIL %s got=0x%0h expected=0x%0h", name, got, expected);
                $fatal(1);
            end
        end
    endtask

    function automatic logic signed [7:0] sat8(input integer value);
        begin
            if (value > 127) begin
                sat8 = 8'sd127;
            end else if (value < -128) begin
                sat8 = -8'sd128;
            end else begin
                sat8 = value[7:0];
            end
        end
    endfunction

    function automatic integer rne_shift(input integer value, input integer frac_drop);
        integer mag;
        integer trunc_mag;
        integer rem;
        integer half;
        integer inc;
        begin
            mag = (value < 0) ? -value : value;
            trunc_mag = mag >>> frac_drop;
            rem = mag & ((1 << frac_drop) - 1);
            half = 1 << (frac_drop - 1);
            inc = ((rem > half) || ((rem == half) && (trunc_mag & 1))) ? 1 : 0;
            rne_shift = (value < 0) ? -(trunc_mag + inc) : (trunc_mag + inc);
        end
    endfunction

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    // stream_reg
    logic stream_in_valid;
    logic stream_in_ready;
    logic [STREAM_DATA_W-1:0] stream_in_data;
    logic [STREAM_META_W-1:0] stream_in_meta;
    logic stream_in_last;
    logic stream_out_valid;
    logic stream_out_ready;
    logic [STREAM_DATA_W-1:0] stream_out_data;
    logic [STREAM_META_W-1:0] stream_out_meta;
    logic stream_out_last;

    stream_reg #(
        .DATA_W(STREAM_DATA_W),
        .META_W(STREAM_META_W)
    ) u_stream_reg (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (stream_in_valid),
        .in_ready  (stream_in_ready),
        .in_data   (stream_in_data),
        .in_meta   (stream_in_meta),
        .in_last   (stream_in_last),
        .out_valid (stream_out_valid),
        .out_ready (stream_out_ready),
        .out_data  (stream_out_data),
        .out_meta  (stream_out_meta),
        .out_last  (stream_out_last)
    );

    // skid_buffer
    logic skid_in_valid;
    logic skid_in_ready;
    logic [STREAM_DATA_W-1:0] skid_in_data;
    logic [STREAM_META_W-1:0] skid_in_meta;
    logic skid_in_last;
    logic skid_out_valid;
    logic skid_out_ready;
    logic [STREAM_DATA_W-1:0] skid_out_data;
    logic [STREAM_META_W-1:0] skid_out_meta;
    logic skid_out_last;

    skid_buffer #(
        .DATA_W(STREAM_DATA_W),
        .META_W(STREAM_META_W)
    ) u_skid_buffer (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (skid_in_valid),
        .in_ready  (skid_in_ready),
        .in_data   (skid_in_data),
        .in_meta   (skid_in_meta),
        .in_last   (skid_in_last),
        .out_valid (skid_out_valid),
        .out_ready (skid_out_ready),
        .out_data  (skid_out_data),
        .out_meta  (skid_out_meta),
        .out_last  (skid_out_last)
    );

    // FIFO, non-power-of-two depth.
    logic fifo_wr_valid;
    logic fifo_wr_ready;
    logic [FIFO_DATA_W-1:0] fifo_wr_data;
    logic [FIFO_META_W-1:0] fifo_wr_meta;
    logic fifo_wr_last;
    logic fifo_rd_valid;
    logic fifo_rd_ready;
    logic [FIFO_DATA_W-1:0] fifo_rd_data;
    logic [FIFO_META_W-1:0] fifo_rd_meta;
    logic fifo_rd_last;
    logic fifo_full;
    logic fifo_empty;
    logic fifo_almost_full;
    logic [2:0] fifo_occupancy;

    sync_fifo #(
        .DATA_W(FIFO_DATA_W),
        .META_W(FIFO_META_W),
        .DEPTH(5),
        .ALMOST_FULL_THRESHOLD(4)
    ) u_fifo5 (
        .clk         (clk),
        .rst_n       (rst_n),
        .wr_valid    (fifo_wr_valid),
        .wr_ready    (fifo_wr_ready),
        .wr_data     (fifo_wr_data),
        .wr_meta     (fifo_wr_meta),
        .wr_last     (fifo_wr_last),
        .rd_valid    (fifo_rd_valid),
        .rd_ready    (fifo_rd_ready),
        .rd_data     (fifo_rd_data),
        .rd_meta     (fifo_rd_meta),
        .rd_last     (fifo_rd_last),
        .full        (fifo_full),
        .empty       (fifo_empty),
        .almost_full (fifo_almost_full),
        .occupancy   (fifo_occupancy)
    );

    logic fifo3_wr_valid;
    logic fifo3_wr_ready;
    logic [FIFO_DATA_W-1:0] fifo3_wr_data;
    logic [FIFO_META_W-1:0] fifo3_wr_meta;
    logic fifo3_wr_last;
    logic fifo3_rd_valid;
    logic fifo3_rd_ready;
    logic [FIFO_DATA_W-1:0] fifo3_rd_data;
    logic [FIFO_META_W-1:0] fifo3_rd_meta;
    logic fifo3_rd_last;
    logic fifo3_full;
    logic fifo3_empty;
    logic fifo3_almost_full;
    logic [1:0] fifo3_occupancy;

    sync_fifo #(
        .DATA_W(FIFO_DATA_W),
        .META_W(FIFO_META_W),
        .DEPTH(3),
        .ALMOST_FULL_THRESHOLD(2)
    ) u_fifo3 (
        .clk         (clk),
        .rst_n       (rst_n),
        .wr_valid    (fifo3_wr_valid),
        .wr_ready    (fifo3_wr_ready),
        .wr_data     (fifo3_wr_data),
        .wr_meta     (fifo3_wr_meta),
        .wr_last     (fifo3_wr_last),
        .rd_valid    (fifo3_rd_valid),
        .rd_ready    (fifo3_rd_ready),
        .rd_data     (fifo3_rd_data),
        .rd_meta     (fifo3_rd_meta),
        .rd_last     (fifo3_rd_last),
        .full        (fifo3_full),
        .empty       (fifo3_empty),
        .almost_full (fifo3_almost_full),
        .occupancy   (fifo3_occupancy)
    );

    // SRAM wrappers.
    logic s1_clk_en;
    logic s1_req_valid;
    logic s1_req_ready;
    logic s1_req_write;
    logic [2:0] s1_req_addr;
    logic [15:0] s1_req_wdata;
    logic [1:0] s1_req_wstrb;
    logic s1_rsp_valid;
    logic s1_rsp_ready;
    logic [15:0] s1_rsp_rdata;

    sram_1p_wrapper #(
        .DATA_W(16),
        .DEPTH(8)
    ) u_sram1p (
        .clk        (clk),
        .rst_n      (rst_n),
        .clk_en     (s1_clk_en),
        .req_valid  (s1_req_valid),
        .req_ready  (s1_req_ready),
        .req_write  (s1_req_write),
        .req_addr   (s1_req_addr),
        .req_wdata  (s1_req_wdata),
        .req_wstrb  (s1_req_wstrb),
        .rsp_valid  (s1_rsp_valid),
        .rsp_ready  (s1_rsp_ready),
        .rsp_rdata  (s1_rsp_rdata)
    );

    logic s2_clk_en;
    logic s2_wr_valid;
    logic s2_wr_ready;
    logic [2:0] s2_wr_addr;
    logic [15:0] s2_wr_data;
    logic [1:0] s2_wr_wstrb;
    logic s2_rd_valid;
    logic s2_rd_ready;
    logic [2:0] s2_rd_addr;
    logic s2_rsp_valid;
    logic s2_rsp_ready;
    logic [15:0] s2_rsp_rdata;

    sram_2p_wrapper #(
        .DATA_W(16),
        .DEPTH(8)
    ) u_sram2p (
        .clk        (clk),
        .rst_n      (rst_n),
        .clk_en     (s2_clk_en),
        .wr_valid   (s2_wr_valid),
        .wr_ready   (s2_wr_ready),
        .wr_addr    (s2_wr_addr),
        .wr_data    (s2_wr_data),
        .wr_wstrb   (s2_wr_wstrb),
        .rd_valid   (s2_rd_valid),
        .rd_ready   (s2_rd_ready),
        .rd_addr    (s2_rd_addr),
        .rsp_valid  (s2_rsp_valid),
        .rsp_ready  (s2_rsp_ready),
        .rsp_rdata  (s2_rsp_rdata)
    );

    // Arithmetic wrappers.
    logic mul_in_valid;
    logic mul_in_ready;
    logic signed [7:0] mul_in_a;
    logic signed [7:0] mul_in_b;
    logic [3:0] mul_in_meta;
    logic mul_in_last;
    logic mul_out_valid;
    logic mul_out_ready;
    logic signed [15:0] mul_out_result;
    logic mul_out_overflow;
    logic [3:0] mul_out_meta;
    logic mul_out_last;

    mul_unit #(
        .A_W(8),
        .B_W(8),
        .OUT_W(16),
        .META_W(4)
    ) u_mul (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (mul_in_valid),
        .in_ready     (mul_in_ready),
        .in_a         (mul_in_a),
        .in_b         (mul_in_b),
        .in_meta      (mul_in_meta),
        .in_last      (mul_in_last),
        .out_valid    (mul_out_valid),
        .out_ready    (mul_out_ready),
        .out_result   (mul_out_result),
        .out_overflow (mul_out_overflow),
        .out_meta     (mul_out_meta),
        .out_last     (mul_out_last)
    );

    logic add_in_valid;
    logic add_in_ready;
    logic signed [7:0] add_in_a;
    logic signed [7:0] add_in_b;
    logic [3:0] add_in_meta;
    logic add_in_last;
    logic add_out_valid;
    logic add_out_ready;
    logic signed [7:0] add_out_result;
    logic add_out_overflow;
    logic [3:0] add_out_meta;
    logic add_out_last;

    add_unit #(
        .A_W(8),
        .B_W(8),
        .OUT_W(8),
        .META_W(4)
    ) u_add (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (add_in_valid),
        .in_ready     (add_in_ready),
        .in_a         (add_in_a),
        .in_b         (add_in_b),
        .in_meta      (add_in_meta),
        .in_last      (add_in_last),
        .out_valid    (add_out_valid),
        .out_ready    (add_out_ready),
        .out_result   (add_out_result),
        .out_overflow (add_out_overflow),
        .out_meta     (add_out_meta),
        .out_last     (add_out_last)
    );

    logic mac_in_valid;
    logic mac_in_ready;
    logic signed [7:0] mac_in_a;
    logic signed [7:0] mac_in_b;
    logic signed [15:0] mac_in_acc;
    logic mac_in_clear;
    logic [3:0] mac_in_meta;
    logic mac_in_last;
    logic mac_out_valid;
    logic mac_out_ready;
    logic signed [15:0] mac_out_acc;
    logic mac_out_overflow;
    logic [3:0] mac_out_meta;
    logic mac_out_last;

    mac_unit #(
        .A_W(8),
        .B_W(8),
        .ACC_W(16),
        .META_W(4)
    ) u_mac (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (mac_in_valid),
        .in_ready     (mac_in_ready),
        .in_a         (mac_in_a),
        .in_b         (mac_in_b),
        .in_acc       (mac_in_acc),
        .in_clear     (mac_in_clear),
        .in_meta      (mac_in_meta),
        .in_last      (mac_in_last),
        .out_valid    (mac_out_valid),
        .out_ready    (mac_out_ready),
        .out_acc      (mac_out_acc),
        .out_overflow (mac_out_overflow),
        .out_meta     (mac_out_meta),
        .out_last     (mac_out_last)
    );

    logic cmp_in_valid;
    logic cmp_in_ready;
    logic signed [7:0] cmp_in_a;
    logic signed [7:0] cmp_in_b;
    logic [3:0] cmp_in_meta;
    logic cmp_in_last;
    logic cmp_out_valid;
    logic cmp_out_ready;
    logic signed [7:0] cmp_out_max;
    logic cmp_out_take_b;
    logic [3:0] cmp_out_meta;
    logic cmp_out_last;

    compare_max #(
        .DATA_W(8),
        .META_W(4)
    ) u_cmp (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (cmp_in_valid),
        .in_ready   (cmp_in_ready),
        .in_a       (cmp_in_a),
        .in_b       (cmp_in_b),
        .in_meta    (cmp_in_meta),
        .in_last    (cmp_in_last),
        .out_valid  (cmp_out_valid),
        .out_ready  (cmp_out_ready),
        .out_max    (cmp_out_max),
        .out_take_b (cmp_out_take_b),
        .out_meta   (cmp_out_meta),
        .out_last   (cmp_out_last)
    );

    logic rs_in_valid;
    logic rs_in_ready;
    logic signed [15:0] rs_in_data;
    logic [3:0] rs_in_meta;
    logic rs_in_last;
    logic rs_out_valid;
    logic rs_out_ready;
    logic signed [7:0] rs_out_data;
    logic rs_out_overflow;
    logic rs_out_underflow;
    logic rs_out_inexact;
    logic [3:0] rs_out_meta;
    logic rs_out_last;

    round_sat #(
        .IN_W(16),
        .OUT_W(8),
        .FRAC_DROP(1),
        .ROUND_MODE(1),
        .SATURATE(1),
        .META_W(4)
    ) u_round_sat (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (rs_in_valid),
        .in_ready      (rs_in_ready),
        .in_data       (rs_in_data),
        .in_meta       (rs_in_meta),
        .in_last       (rs_in_last),
        .out_valid     (rs_out_valid),
        .out_ready     (rs_out_ready),
        .out_data      (rs_out_data),
        .out_overflow  (rs_out_overflow),
        .out_underflow (rs_out_underflow),
        .out_inexact   (rs_out_inexact),
        .out_meta      (rs_out_meta),
        .out_last      (rs_out_last)
    );

    task automatic init_inputs;
        begin
            stream_in_valid = 1'b0;
            stream_in_data = '0;
            stream_in_meta = '0;
            stream_in_last = 1'b0;
            stream_out_ready = 1'b0;

            skid_in_valid = 1'b0;
            skid_in_data = '0;
            skid_in_meta = '0;
            skid_in_last = 1'b0;
            skid_out_ready = 1'b0;

            fifo_wr_valid = 1'b0;
            fifo_wr_data = '0;
            fifo_wr_meta = '0;
            fifo_wr_last = 1'b0;
            fifo_rd_ready = 1'b0;
            fifo3_wr_valid = 1'b0;
            fifo3_wr_data = '0;
            fifo3_wr_meta = '0;
            fifo3_wr_last = 1'b0;
            fifo3_rd_ready = 1'b0;

            s1_clk_en = 1'b1;
            s1_req_valid = 1'b0;
            s1_req_write = 1'b0;
            s1_req_addr = '0;
            s1_req_wdata = '0;
            s1_req_wstrb = 2'b11;
            s1_rsp_ready = 1'b1;

            s2_clk_en = 1'b1;
            s2_wr_valid = 1'b0;
            s2_wr_addr = '0;
            s2_wr_data = '0;
            s2_wr_wstrb = 2'b11;
            s2_rd_valid = 1'b0;
            s2_rd_addr = '0;
            s2_rsp_ready = 1'b1;

            mul_in_valid = 1'b0;
            mul_in_a = '0;
            mul_in_b = '0;
            mul_in_meta = '0;
            mul_in_last = 1'b0;
            mul_out_ready = 1'b1;

            add_in_valid = 1'b0;
            add_in_a = '0;
            add_in_b = '0;
            add_in_meta = '0;
            add_in_last = 1'b0;
            add_out_ready = 1'b1;

            mac_in_valid = 1'b0;
            mac_in_a = '0;
            mac_in_b = '0;
            mac_in_acc = '0;
            mac_in_clear = 1'b0;
            mac_in_meta = '0;
            mac_in_last = 1'b0;
            mac_out_ready = 1'b1;

            cmp_in_valid = 1'b0;
            cmp_in_a = '0;
            cmp_in_b = '0;
            cmp_in_meta = '0;
            cmp_in_last = 1'b0;
            cmp_out_ready = 1'b1;

            rs_in_valid = 1'b0;
            rs_in_data = '0;
            rs_in_meta = '0;
            rs_in_last = 1'b0;
            rs_out_ready = 1'b1;
        end
    endtask

    task automatic send_stream_reg_item(input int idx, input int stall_cycles);
        logic [STREAM_DATA_W-1:0] held_data;
        logic [STREAM_META_W-1:0] held_meta;
        logic held_last;
        int wait_count;
        begin
            held_data = 16'h1000 + idx[15:0];
            held_meta = idx[3:0];
            held_last = (idx == 31);
            stream_out_ready = 1'b1;
            stream_in_valid = 1'b1;
            stream_in_data = held_data;
            stream_in_meta = held_meta;
            stream_in_last = held_last;
            wait_count = 0;
            do begin
                @(posedge clk); #1;
                wait_count = wait_count + 1;
                if (wait_count > 20) tb_fail("stream_reg input wait timeout");
            end while (!stream_in_ready);
            stream_in_valid = 1'b0;
            stream_out_ready = 1'b0;

            if (!stream_out_valid) tb_fail("stream_reg output missing before backpressure");
            check_bits("stream_reg data", stream_out_data, held_data);
            check_bits("stream_reg meta", stream_out_meta, held_meta);
            check_eq("stream_reg last", stream_out_last, held_last);

            repeat (stall_cycles) begin
                @(posedge clk); #1;
                if (!stream_out_valid) tb_fail("stream_reg output valid missing under backpressure");
                check_bits("stream_reg stable data", stream_out_data, held_data);
                check_bits("stream_reg stable meta", stream_out_meta, held_meta);
                check_eq("stream_reg stable last", stream_out_last, held_last);
            end

            stream_out_ready = 1'b1;
            @(posedge clk); #1;
            @(posedge clk); #1;
        end
    endtask

    task automatic test_stream_reg;
        int idx;
        begin
            $display("TEST stream_reg");
            apply_reset();
            if (stream_out_valid) tb_fail("stream_reg valid not clear after reset");

            for (idx = 0; idx < 32; idx = idx + 1) begin
                send_stream_reg_item(idx, (idx % 5 == 0) ? 4 : (idx % 3));
            end
        end
    endtask

    task automatic test_skid_buffer;
        int idx;
        logic [STREAM_DATA_W-1:0] held_data;
        logic [STREAM_META_W-1:0] held_meta;
        logic held_last;
        int wait_count;
        begin
            $display("TEST skid_buffer");
            apply_reset();
            if (skid_out_valid) tb_fail("skid_buffer valid not clear after reset");

            for (idx = 0; idx < 48; idx = idx + 1) begin
                held_data = 16'h2000 + idx[15:0];
                held_meta = idx[3:0] ^ 4'h5;
                held_last = (idx == 47);
                skid_out_ready = 1'b1;
                skid_in_valid = 1'b1;
                skid_in_data = held_data;
                skid_in_meta = held_meta;
                skid_in_last = held_last;
                wait_count = 0;
                @(posedge clk); #1;
                if (!skid_in_ready) begin
                    do begin
                        @(posedge clk); #1;
                        wait_count = wait_count + 1;
                        if (wait_count > 20) tb_fail("skid input wait timeout");
                    end while (!skid_in_ready);
                end
                skid_in_valid = 1'b0;
                skid_out_ready = 1'b0;
                if (!skid_out_valid) tb_fail("skid output missing before backpressure");
                check_bits("skid data", skid_out_data, held_data);
                check_bits("skid meta", skid_out_meta, held_meta);
                check_eq("skid last", skid_out_last, held_last);

                repeat ((idx % 6 == 0) ? 4 : (idx % 2)) begin
                    @(posedge clk); #1;
                    if (!skid_out_valid) tb_fail("skid output valid missing under backpressure");
                    check_bits("skid stable data", skid_out_data, held_data);
                    check_bits("skid stable meta", skid_out_meta, held_meta);
                    check_eq("skid stable last", skid_out_last, held_last);
                end

                skid_out_ready = 1'b1;
                @(posedge clk); #1;
                @(posedge clk); #1;
            end
        end
    endtask

    task automatic test_fifo5;
        int cycle;
        int sent;
        int seen;
        logic pre_wr_fire;
        logic pre_rd_fire;
        logic [FIFO_DATA_W-1:0] pre_rd_data;
        logic [FIFO_META_W-1:0] pre_rd_meta;
        logic pre_rd_last;
        begin
            $display("TEST sync_fifo depth5");
            apply_reset();
            if (!fifo_empty || fifo_occupancy !== 0) tb_fail("fifo5 not empty after reset");
            sent = 0;
            seen = 0;
            for (cycle = 0; cycle < 240; cycle = cycle + 1) begin
                @(negedge clk);
                fifo_wr_valid = (sent < 64) && ((cycle % 3) != 1);
                fifo_wr_data = 16'h3000 + sent[15:0];
                fifo_wr_meta = sent[3:0] ^ 4'h9;
                fifo_wr_last = (sent == 63);
                fifo_rd_ready = ((cycle % 5) != 2);
                #1;
                pre_wr_fire = fifo_wr_valid && fifo_wr_ready;
                pre_rd_fire = fifo_rd_valid && fifo_rd_ready;
                pre_rd_data = fifo_rd_data;
                pre_rd_meta = fifo_rd_meta;
                pre_rd_last = fifo_rd_last;
                @(posedge clk); #1;
                if (pre_wr_fire) sent = sent + 1;
                if (pre_rd_fire) begin
                    check_bits("fifo5 data", pre_rd_data, 16'h3000 + seen);
                    check_bits("fifo5 meta", pre_rd_meta, (seen[3:0] ^ 4'h9));
                    check_eq("fifo5 last", pre_rd_last, (seen == 63));
                    seen = seen + 1;
                end
                if (fifo_occupancy > 5) tb_fail("fifo5 occupancy out of range");
                if (fifo_almost_full !== (fifo_occupancy >= 4)) tb_fail("fifo5 almost_full mismatch");
            end
            fifo_wr_valid = 1'b0;
            fifo_rd_ready = 1'b1;
            repeat (20) begin
                @(negedge clk);
                #1;
                pre_rd_fire = fifo_rd_valid && fifo_rd_ready;
                pre_rd_data = fifo_rd_data;
                @(posedge clk); #1;
                if (pre_rd_fire) begin
                    check_bits("fifo5 drain data", pre_rd_data, 16'h3000 + seen);
                    seen = seen + 1;
                end
            end
            check_eq("fifo5 sent", sent, 64);
            check_eq("fifo5 seen", seen, 64);
            if (!fifo_empty) tb_fail("fifo5 not empty after drain");
        end
    endtask

    task automatic test_fifo3_full_pop_push;
        logic pre_wr_ready;
        logic pre_rd_valid;
        logic [FIFO_DATA_W-1:0] pre_rd_data;
        begin
            $display("TEST sync_fifo depth3 full pop+push");
            apply_reset();
            fifo3_rd_ready = 1'b0;
            for (int i = 0; i < 3; i = i + 1) begin
                @(negedge clk);
                fifo3_wr_valid = 1'b1;
                fifo3_wr_data = 16'h4000 + i[15:0];
                fifo3_wr_meta = i[3:0];
                fifo3_wr_last = 1'b0;
                #1;
                pre_wr_ready = fifo3_wr_ready;
                @(posedge clk); #1;
                if (!pre_wr_ready) tb_fail("fifo3 did not accept fill item");
            end
            fifo3_wr_valid = 1'b0;
            @(posedge clk); #1;
            if (!fifo3_full || fifo3_occupancy !== 3) tb_fail("fifo3 not full");

            @(negedge clk);
            fifo3_wr_valid = 1'b1;
            fifo3_wr_data = 16'h4003;
            fifo3_wr_meta = 4'h3;
            fifo3_rd_ready = 1'b1;
            #1;
            pre_wr_ready = fifo3_wr_ready;
            pre_rd_valid = fifo3_rd_valid;
            pre_rd_data = fifo3_rd_data;
            @(posedge clk); #1;
            if (!pre_wr_ready) tb_fail("fifo3 did not allow full pop+push");
            if (!pre_rd_valid || pre_rd_data !== 16'h4000) tb_fail("fifo3 first output mismatch");
            fifo3_wr_valid = 1'b0;
            repeat (4) begin
                @(posedge clk); #1;
            end
        end
    endtask

    task automatic sram1_write(input [2:0] addr, input [15:0] data);
        begin
            s1_req_valid = 1'b1;
            s1_req_write = 1'b1;
            s1_req_addr = addr;
            s1_req_wdata = data;
            s1_req_wstrb = 2'b11;
            @(posedge clk); #1;
            if (!s1_req_ready) tb_fail("sram1 write not ready");
            s1_req_valid = 1'b0;
        end
    endtask

    task automatic sram1_read_check(input [2:0] addr, input [15:0] data);
        begin
            s1_req_valid = 1'b1;
            s1_req_write = 1'b0;
            s1_req_addr = addr;
            @(posedge clk); #1;
            if (!s1_req_ready) tb_fail("sram1 read not ready");
            s1_req_valid = 1'b0;
            if (!s1_rsp_valid) tb_fail("sram1 missing read response");
            check_bits("sram1 read data", s1_rsp_rdata, data);
            @(posedge clk); #1;
        end
    endtask

    task automatic test_sram1p;
        begin
            $display("TEST sram_1p_wrapper");
            apply_reset();
            s1_rsp_ready = 1'b1;
            sram1_write(3, 16'h1234);
            sram1_read_check(3, 16'h1234);
            s1_clk_en = 1'b0;
            s1_req_valid = 1'b1;
            s1_req_write = 1'b1;
            s1_req_addr = 4;
            s1_req_wdata = 16'hDEAD;
            @(posedge clk); #1;
            if (s1_req_ready) tb_fail("sram1 ready high when clk_en=0");
            s1_clk_en = 1'b1;
            s1_req_valid = 1'b0;
            sram1_write(4, 16'hBEEF);
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            sram1_read_check(4, 16'hBEEF);
        end
    endtask

    task automatic test_sram2p;
        begin
            $display("TEST sram_2p_wrapper");
            apply_reset();
            s2_wr_valid = 1'b1;
            s2_wr_addr = 2;
            s2_wr_data = 16'hAAAA;
            s2_wr_wstrb = 2'b11;
            @(posedge clk); #1;
            if (!s2_wr_ready) tb_fail("sram2 write not ready");
            s2_wr_valid = 1'b0;

            s2_rd_valid = 1'b1;
            s2_rd_addr = 2;
            @(posedge clk); #1;
            if (!s2_rd_ready) tb_fail("sram2 read not ready");
            s2_rd_valid = 1'b0;
            if (!s2_rsp_valid) tb_fail("sram2 missing response");
            check_bits("sram2 read old", s2_rsp_rdata, 16'hAAAA);
            @(posedge clk); #1;

            s2_wr_valid = 1'b1;
            s2_wr_addr = 2;
            s2_wr_data = 16'h5555;
            s2_rd_valid = 1'b1;
            s2_rd_addr = 2;
            @(posedge clk); #1;
            s2_wr_valid = 1'b0;
            s2_rd_valid = 1'b0;
            if (!s2_rsp_valid) tb_fail("sram2 missing collision response");
            check_bits("sram2 read-first collision", s2_rsp_rdata, 16'hAAAA);
            @(posedge clk); #1;

            s2_clk_en = 1'b0;
            s2_rd_valid = 1'b1;
            @(posedge clk); #1;
            if (s2_rd_ready) tb_fail("sram2 read ready high when clk_en=0");
            s2_clk_en = 1'b1;
            s2_rd_valid = 1'b0;
        end
    endtask

    task automatic test_arithmetic;
        integer expected;
        integer rounded;
        begin
            $display("TEST arithmetic wrappers");
            apply_reset();

            mul_out_ready = 1'b1;
            mul_in_valid = 1'b1;
            mul_in_a = -8'sd7;
            mul_in_b = 8'sd6;
            mul_in_meta = 4'hA;
            mul_in_last = 1'b1;
            @(posedge clk); #1;
            mul_in_valid = 1'b0;
            mul_out_ready = 1'b0;
            if (!mul_out_valid) tb_fail("mul output missing");
            check_eq("mul result", mul_out_result, -42);
            check_eq("mul overflow", mul_out_overflow, 0);
            check_bits("mul meta", mul_out_meta, 4'hA);
            check_eq("mul last", mul_out_last, 1);
            repeat (3) @(posedge clk);
            check_eq("mul stable under backpressure", mul_out_result, -42);
            @(negedge clk);
            mul_out_ready = 1'b1;
            @(posedge clk); #1;

            add_out_ready = 1'b1;
            add_in_valid = 1'b1;
            add_in_a = 8'sd127;
            add_in_b = 8'sd1;
            add_in_meta = 4'hB;
            add_in_last = 1'b1;
            @(posedge clk); #1;
            add_in_valid = 1'b0;
            if (!add_out_valid) tb_fail("add output missing");
            check_eq("add wrapped result", add_out_result, -128);
            check_eq("add overflow", add_out_overflow, 1);
            check_bits("add meta", add_out_meta, 4'hB);
            @(posedge clk); #1;

            mac_out_ready = 1'b1;
            mac_in_valid = 1'b1;
            mac_in_a = 8'sd3;
            mac_in_b = -8'sd5;
            mac_in_acc = 16'sd20;
            mac_in_clear = 1'b0;
            mac_in_meta = 4'hC;
            mac_in_last = 1'b0;
            @(posedge clk); #1;
            mac_in_valid = 1'b0;
            if (!mac_out_valid) tb_fail("mac output missing");
            check_eq("mac result", mac_out_acc, 5);
            check_eq("mac overflow", mac_out_overflow, 0);
            check_bits("mac meta", mac_out_meta, 4'hC);
            @(posedge clk); #1;

            cmp_out_ready = 1'b1;
            cmp_in_valid = 1'b1;
            cmp_in_a = -8'sd2;
            cmp_in_b = 8'sd5;
            cmp_in_meta = 4'hD;
            cmp_in_last = 1'b1;
            @(posedge clk); #1;
            cmp_in_valid = 1'b0;
            if (!cmp_out_valid) tb_fail("compare output missing");
            check_eq("compare max", cmp_out_max, 5);
            check_eq("compare take_b", cmp_out_take_b, 1);
            check_bits("compare meta", cmp_out_meta, 4'hD);
            @(posedge clk); #1;

            rs_out_ready = 1'b1;
            rs_in_valid = 1'b1;
            rs_in_data = -16'sd5;
            rs_in_meta = 4'hE;
            rs_in_last = 1'b1;
            @(posedge clk); #1;
            rs_in_valid = 1'b0;
            rs_out_ready = 1'b0;
            if (!rs_out_valid) tb_fail("round_sat output missing");
            rounded = rne_shift(-5, 1);
            expected = sat8(rounded);
            check_eq("round_sat -2.5 tie even", rs_out_data, expected);
            check_eq("round_sat inexact", rs_out_inexact, 1);
            check_bits("round_sat meta", rs_out_meta, 4'hE);
            repeat (2) @(posedge clk);
            check_eq("round_sat stable under backpressure", rs_out_data, expected);
            @(negedge clk);
            rs_out_ready = 1'b1;
            @(posedge clk); #1;

            rs_out_ready = 1'b1;
            rs_in_valid = 1'b1;
            rs_in_data = 16'sd300;
            rs_in_meta = 4'hF;
            rs_in_last = 1'b0;
            @(posedge clk); #1;
            rs_in_valid = 1'b0;
            if (!rs_out_valid) tb_fail("round_sat saturation output missing");
            check_eq("round_sat overflow data", rs_out_data, 127);
            check_eq("round_sat overflow flag", rs_out_overflow, 1);
            @(posedge clk); #1;
        end
    endtask

    initial begin
        init_inputs();
        apply_reset();
        test_stream_reg();
        init_inputs();
        test_skid_buffer();
        init_inputs();
        test_fifo5();
        init_inputs();
        test_fifo3_full_pop_push();
        init_inputs();
        test_sram1p();
        init_inputs();
        test_sram2p();
        init_inputs();
        test_arithmetic();
        $display("STAGE1_RTL_SIM_PASS");
        $finish;
    end
endmodule

`default_nettype wire
