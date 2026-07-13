`timescale 1ns/1ps
`default_nettype none

import paper_score_packet_pkg::*;

module tb_h9_probability_fifo;
    logic clk;
    logic rst_n;
    logic clear;
    logic in_valid;
    logic in_ready;
    h9_probability_packet_t in_packet;
    logic out_valid;
    logic out_ready;
    h9_probability_packet_t out_packet;
    logic [2:0] occupancy;
    logic [2:0] peak_occupancy;
    logic full;
    logic empty;
    logic [1:0] read_pointer;
    logic [1:0] write_pointer;
    logic [63:0] full_stall_cycles;
    logic [63:0] empty_stall_cycles;
    logic [63:0] producer_stall_cycles;
    logic [63:0] consumer_stall_cycles;

    paper_probability_fifo #(
        .DEPTH(4),
        .COUNTER_W(64)
    ) dut (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .clear                 (clear),
        .in_valid              (in_valid),
        .in_ready              (in_ready),
        .in_packet             (in_packet),
        .out_valid             (out_valid),
        .out_ready             (out_ready),
        .out_packet            (out_packet),
        .occupancy             (occupancy),
        .peak_occupancy        (peak_occupancy),
        .full                  (full),
        .empty                 (empty),
        .read_pointer          (read_pointer),
        .write_pointer         (write_pointer),
        .full_stall_cycles     (full_stall_cycles),
        .empty_stall_cycles    (empty_stall_cycles),
        .producer_stall_cycles (producer_stall_cycles),
        .consumer_stall_cycles (consumer_stall_cycles)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic fail(input string msg);
        begin
            $display("HW_H9_PROBABILITY_FIFO_FAIL: %s", msg);
            $finish;
        end
    endtask

    task automatic push_probability(input int index, input logic last);
        begin
            in_packet = '0;
            in_packet.token_meta = 16'hBEEF;
            in_packet.head_id = 8'd2;
            in_packet.logical_token_index = index[15:0];
            in_packet.cache_slot = index[15:0];
            in_packet.probability_index = index[15:0];
            in_packet.probability_fp32 = 32'h3F00_0000;
            in_packet.last_probability = last;
            do @(negedge clk); while (!in_ready);
            in_valid = 1'b1;
            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    initial begin
        rst_n = 1'b0;
        clear = 1'b0;
        in_valid = 1'b0;
        out_ready = 1'b0;
        in_packet = '0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        push_probability(0, 1'b0);
        push_probability(1, 1'b1);
        repeat (3) @(posedge clk);
        if (!out_valid) fail("first probability did not become valid");
        if (out_packet.probability_index != 16'd0) fail("first probability index mismatch");
        repeat (2) @(posedge clk);
        if (!out_valid || out_packet.probability_index != 16'd0) fail("probability not stable during stall");
        @(negedge clk);
        out_ready = 1'b1;
        @(negedge clk);
        out_ready = 1'b0;
        repeat (3) @(posedge clk);
        if (!out_valid) fail("second probability did not become valid");
        if (out_packet.probability_index != 16'd1 || !out_packet.last_probability) begin
            $display("HW_H9_PROBABILITY_FIFO_DEBUG second_valid=%0b index=%0d last=%0b occupancy=%0d empty=%0b rd=%0d wr=%0d",
                     out_valid, out_packet.probability_index, out_packet.last_probability,
                     occupancy, empty, read_pointer, write_pointer);
            fail("second probability packet mismatch");
        end
        @(negedge clk);
        out_ready = 1'b1;
        @(negedge clk);
        out_ready = 1'b0;

        clear = 1'b1;
        @(posedge clk);
        clear = 1'b0;
        @(posedge clk);
        if (!empty) fail("clear did not empty FIFO");
        $display("HW_H9_PROBABILITY_FIFO_PASS peak=%0d empty_stall=%0d", peak_occupancy, empty_stall_cycles);
        $finish;
    end
endmodule

`default_nettype wire
