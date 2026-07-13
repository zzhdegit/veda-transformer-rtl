`default_nettype none

import paper_score_packet_pkg::*;

module paper_probability_fifo #(
    parameter int DEPTH = 32,
    parameter int COUNTER_W = 64,
    localparam int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    localparam int COUNT_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH + 1)
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,

    input  logic                     in_valid,
    output logic                     in_ready,
    input  h9_probability_packet_t   in_packet,

    output logic                     out_valid,
    input  logic                     out_ready,
    output h9_probability_packet_t   out_packet,

    output logic [COUNT_W-1:0]       occupancy,
    output logic [COUNT_W-1:0]       peak_occupancy,
    output logic                     full,
    output logic                     empty,
    output logic [PTR_W-1:0]         read_pointer,
    output logic [PTR_W-1:0]         write_pointer,
    output logic [COUNTER_W-1:0]     full_stall_cycles,
    output logic [COUNTER_W-1:0]     empty_stall_cycles,
    output logic [COUNTER_W-1:0]     producer_stall_cycles,
    output logic [COUNTER_W-1:0]     consumer_stall_cycles
);
    h9_probability_packet_t mem [0:DEPTH-1];
    logic [PTR_W-1:0] rd_ptr_q;
    logic [PTR_W-1:0] wr_ptr_q;
    logic [COUNT_W-1:0] count_q;
    logic pop_mem;
    logic [COUNT_W-1:0] count_next;
    logic [COUNT_W-1:0] total_count;
    h9_probability_packet_t out_packet_q;
    logic out_valid_q;

    wire in_fire = in_valid && in_ready;
    wire out_fire = out_valid && out_ready;

    initial begin
        if (DEPTH <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "paper_probability_fifo parameters must be positive");
        end
    end

    assign total_count = count_q + COUNT_W'(out_valid_q);
    assign full = total_count == COUNT_W'(DEPTH);
    assign empty = total_count == '0;
    assign in_ready = !full || out_fire;
    assign out_valid = out_valid_q;
    assign out_packet = out_packet_q;
    assign occupancy = total_count;
    assign read_pointer = rd_ptr_q;
    assign write_pointer = wr_ptr_q;
    assign pop_mem = (count_q != '0) && (!out_valid_q || out_fire);

    always_comb begin
        count_next = count_q;
        if (in_fire && !pop_mem) begin
            count_next = count_q + COUNT_W'(1);
        end else if (!in_fire && pop_mem) begin
            count_next = count_q - COUNT_W'(1);
        end
    end

    function automatic [PTR_W-1:0] ptr_next(input logic [PTR_W-1:0] ptr);
        begin
            if (ptr == PTR_W'(DEPTH - 1)) begin
                ptr_next = '0;
            end else begin
                ptr_next = ptr + PTR_W'(1);
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr_q <= '0;
            wr_ptr_q <= '0;
            count_q <= '0;
            out_packet_q <= '0;
            out_valid_q <= 1'b0;
            peak_occupancy <= '0;
            full_stall_cycles <= '0;
            empty_stall_cycles <= '0;
            producer_stall_cycles <= '0;
            consumer_stall_cycles <= '0;
        end else begin
            if (clear) begin
                rd_ptr_q <= '0;
                wr_ptr_q <= '0;
                count_q <= '0;
                out_packet_q <= '0;
                out_valid_q <= 1'b0;
                peak_occupancy <= '0;
            end else begin
                if (in_valid && !in_ready) begin
                    full_stall_cycles <= full_stall_cycles + COUNTER_W'(1);
                    producer_stall_cycles <= producer_stall_cycles + COUNTER_W'(1);
                end
                if (out_ready && !out_valid) begin
                    empty_stall_cycles <= empty_stall_cycles + COUNTER_W'(1);
                    consumer_stall_cycles <= consumer_stall_cycles + COUNTER_W'(1);
                end

                if (out_fire && !pop_mem) begin
                    out_valid_q <= 1'b0;
                end

                if (in_fire) begin
                    mem[int'(wr_ptr_q)] <= in_packet;
                    wr_ptr_q <= ptr_next(wr_ptr_q);
                end

                if (pop_mem) begin
                    out_packet_q <= mem[int'(rd_ptr_q)];
                    out_valid_q <= 1'b1;
                    rd_ptr_q <= ptr_next(rd_ptr_q);
                end
                count_q <= count_next;

                if (total_count > peak_occupancy) begin
                    peak_occupancy <= total_count;
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && !clear) begin
            assert (!(in_fire && full))
                else $error("paper_probability_fifo no_probability_overflow failed");
            assert (!(out_fire && empty && !out_valid_q))
                else $error("paper_probability_fifo no_probability_underflow failed");
            if ($past(rst_n) && $past(in_valid && !in_ready)) begin
                assert (in_valid)
                    else $error("paper_probability_fifo probability_payload_stable_until_ready valid failed");
                assert ($stable(in_packet))
                    else $error("paper_probability_fifo probability_payload_stable_until_ready payload failed");
            end
            if ($past(rst_n) && $past(out_valid && !out_ready)) begin
                assert (out_valid)
                    else $error("paper_probability_fifo output valid dropped under backpressure");
                assert ($stable(out_packet))
                    else $error("paper_probability_fifo output payload changed under backpressure");
            end
            assert (!(out_valid && $isunknown(out_packet)))
                else $error("paper_probability_fifo no_unknown_output_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
