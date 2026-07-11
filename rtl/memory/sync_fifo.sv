`default_nettype none

module sync_fifo #(
    parameter int DATA_W = 32,
    parameter int META_W = 1,
    parameter int DEPTH = 4,
    parameter int ALMOST_FULL_THRESHOLD = (DEPTH > 0) ? (DEPTH - 1) : 0,
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    localparam int COUNT_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH + 1)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  wr_valid,
    output logic                  wr_ready,
    input  logic [DATA_W-1:0]     wr_data,
    input  logic [META_W-1:0]     wr_meta,
    input  logic                  wr_last,

    output logic                  rd_valid,
    input  logic                  rd_ready,
    output logic [DATA_W-1:0]     rd_data,
    output logic [META_W-1:0]     rd_meta,
    output logic                  rd_last,

    output logic                  full,
    output logic                  empty,
    output logic                  almost_full,
    output logic [COUNT_W-1:0]    occupancy
);
    logic [DATA_W-1:0]  data_mem [0:DEPTH-1];
    logic [META_W-1:0]  meta_mem [0:DEPTH-1];
    logic               last_mem [0:DEPTH-1];
    logic [ADDR_W-1:0]  wr_ptr_q;
    logic [ADDR_W-1:0]  rd_ptr_q;
    logic [COUNT_W-1:0] count_q;

    initial begin
        if (DATA_W <= 0) begin
            $fatal(1, "sync_fifo DATA_W must be positive");
        end
        if (META_W <= 0) begin
            $fatal(1, "sync_fifo META_W must be positive");
        end
        if (DEPTH <= 0) begin
            $fatal(1, "sync_fifo DEPTH must be positive");
        end
        if (ALMOST_FULL_THRESHOLD < 0 || ALMOST_FULL_THRESHOLD > DEPTH) begin
            $fatal(1, "sync_fifo ALMOST_FULL_THRESHOLD out of range");
        end
    end

    function automatic logic [ADDR_W-1:0] ptr_inc(input logic [ADDR_W-1:0] ptr);
        if (ptr == ADDR_W'(DEPTH - 1)) begin
            ptr_inc = '0;
        end else begin
            ptr_inc = ptr + {{(ADDR_W-1){1'b0}}, 1'b1};
        end
    endfunction

    assign full        = (count_q == COUNT_W'(DEPTH));
    assign empty       = (count_q == '0);
    assign occupancy   = count_q;
    assign rd_valid    = !empty;
    assign wr_ready    = !full || (rd_valid && rd_ready);
    assign almost_full = (count_q >= COUNT_W'(ALMOST_FULL_THRESHOLD));

    assign rd_data = data_mem[rd_ptr_q];
    assign rd_meta = meta_mem[rd_ptr_q];
    assign rd_last = last_mem[rd_ptr_q];

    wire push = wr_valid && wr_ready;
    wire pop  = rd_valid && rd_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr_q <= '0;
            rd_ptr_q <= '0;
            count_q  <= '0;
        end else begin
            if (push) begin
                data_mem[wr_ptr_q] <= wr_data;
                meta_mem[wr_ptr_q] <= wr_meta;
                last_mem[wr_ptr_q] <= wr_last;
                wr_ptr_q <= ptr_inc(wr_ptr_q);
            end

            if (pop) begin
                rd_ptr_q <= ptr_inc(rd_ptr_q);
            end

            unique case ({push, pop})
                2'b10: count_q <= count_q + COUNT_W'(1);
                2'b01: count_q <= count_q - COUNT_W'(1);
                default: count_q <= count_q;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (count_q <= COUNT_W'(DEPTH))
                else $error("sync_fifo fifo_occupancy_in_range failed");
            assert (!(full && push && !pop))
                else $error("sync_fifo no_fifo_write_when_full failed");
            assert (!(empty && pop))
                else $error("sync_fifo no_fifo_read_when_empty failed");
            assert (!(rd_valid && $isunknown({rd_data, rd_meta, rd_last})))
                else $error("sync_fifo no_unknown_output_when_valid failed");

            if ($past(rst_n)) begin
                if ($past(wr_valid && !wr_ready)) begin
                    assert (wr_valid)
                        else $error("sync_fifo write valid dropped under backpressure");
                    assert ($stable(wr_data))
                        else $error("sync_fifo write data changed under backpressure");
                    assert ($stable(wr_meta))
                        else $error("sync_fifo write metadata changed under backpressure");
                    assert ($stable(wr_last))
                        else $error("sync_fifo write last changed under backpressure");
                end

                if ($past(rd_valid && !rd_ready)) begin
                    assert (rd_valid)
                        else $error("sync_fifo read valid dropped under backpressure");
                    assert ($stable(rd_data))
                        else $error("sync_fifo read data changed under backpressure");
                    assert ($stable(rd_meta))
                        else $error("sync_fifo read metadata changed under backpressure");
                    assert ($stable(rd_last))
                        else $error("sync_fifo read last changed under backpressure");
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
