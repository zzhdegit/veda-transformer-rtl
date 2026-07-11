`default_nettype none

module sram_2p_wrapper #(
    parameter int DATA_W = 32,
    parameter int DEPTH = 1024,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int READ_LATENCY = 1,
    parameter int RDW_MODE = 0,
    localparam int WSTRB_W = (DATA_W + 7) / 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clk_en,

    input  logic                    wr_valid,
    output logic                    wr_ready,
    input  logic [ADDR_W-1:0]       wr_addr,
    input  logic [DATA_W-1:0]       wr_data,
    input  logic [WSTRB_W-1:0]      wr_wstrb,

    input  logic                    rd_valid,
    output logic                    rd_ready,
    input  logic [ADDR_W-1:0]       rd_addr,

    output logic                    rsp_valid,
    input  logic                    rsp_ready,
    output logic [DATA_W-1:0]       rsp_rdata
);
    localparam int RDW_READ_FIRST = 0;

    logic [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (DATA_W <= 0) begin
            $fatal(1, "sram_2p_wrapper DATA_W must be positive");
        end
        if ((DATA_W % 8) != 0) begin
            $fatal(1, "sram_2p_wrapper DATA_W must be byte-addressable");
        end
        if (DEPTH <= 0) begin
            $fatal(1, "sram_2p_wrapper DEPTH must be positive");
        end
        if (READ_LATENCY != 1) begin
            $fatal(1, "sram_2p_wrapper only supports explicit READ_LATENCY=1 in Stage 1");
        end
        if (RDW_MODE != RDW_READ_FIRST) begin
            $fatal(1, "sram_2p_wrapper only supports RDW_READ_FIRST in Stage 1");
        end
    end

    assign wr_ready = clk_en;
    assign rd_ready = clk_en && (rsp_ready || !rsp_valid);

    wire wr_fire = wr_valid && wr_ready;
    wire rd_fire = rd_valid && rd_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rsp_valid <= 1'b0;
            rsp_rdata <= '0;
        end else if (clk_en) begin
            if (rsp_valid && !rsp_ready) begin
                rsp_valid <= rsp_valid;
                rsp_rdata <= rsp_rdata;
            end else begin
                rsp_valid <= rd_fire;
                if (rd_fire) begin
                    rsp_rdata <= mem[rd_addr];
                end
            end

            if (wr_fire) begin
                for (int byte_idx = 0; byte_idx < WSTRB_W; byte_idx++) begin
                    if (wr_wstrb[byte_idx]) begin
                        mem[wr_addr][byte_idx*8 +: 8] <= wr_data[byte_idx*8 +: 8];
                    end
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(wr_fire && (int'(wr_addr) >= DEPTH)))
                else $error("sram_2p_wrapper write address out of range");
            assert (!(rd_fire && (int'(rd_addr) >= DEPTH)))
                else $error("sram_2p_wrapper read address out of range");
            assert (!(rsp_valid && $isunknown(rsp_rdata)))
                else $error("sram_2p_wrapper no_unknown_output_when_valid failed");

            if ($past(rst_n) && $past(rsp_valid && !rsp_ready)) begin
                assert (rsp_valid)
                    else $error("sram_2p_wrapper response valid dropped under backpressure");
                assert ($stable(rsp_rdata))
                    else $error("sram_2p_wrapper response data changed under backpressure");
            end
        end
    end
`endif
endmodule

`default_nettype wire
