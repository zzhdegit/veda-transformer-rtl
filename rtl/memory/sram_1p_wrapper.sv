`default_nettype none

module sram_1p_wrapper #(
    parameter int DATA_W = 32,
    parameter int DEPTH = 1024,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int READ_LATENCY = 1,
    localparam int WSTRB_W = (DATA_W + 7) / 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clk_en,

    input  logic                    req_valid,
    output logic                    req_ready,
    input  logic                    req_write,
    input  logic [ADDR_W-1:0]       req_addr,
    input  logic [DATA_W-1:0]       req_wdata,
    input  logic [WSTRB_W-1:0]      req_wstrb,

    output logic                    rsp_valid,
    input  logic                    rsp_ready,
    output logic [DATA_W-1:0]       rsp_rdata
);
    logic [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (DATA_W <= 0) begin
            $fatal(1, "sram_1p_wrapper DATA_W must be positive");
        end
        if ((DATA_W % 8) != 0) begin
            $fatal(1, "sram_1p_wrapper DATA_W must be byte-addressable");
        end
        if (DEPTH <= 0) begin
            $fatal(1, "sram_1p_wrapper DEPTH must be positive");
        end
        if (READ_LATENCY != 1) begin
            $fatal(1, "sram_1p_wrapper only supports explicit READ_LATENCY=1 in Stage 1");
        end
    end

    assign req_ready = clk_en && (rsp_ready || !rsp_valid);

    wire req_fire = req_valid && req_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rsp_valid <= 1'b0;
            rsp_rdata <= '0;
        end else if (clk_en) begin
            if (rsp_valid && !rsp_ready) begin
                rsp_valid <= rsp_valid;
                rsp_rdata <= rsp_rdata;
            end else begin
                rsp_valid <= req_fire && !req_write;
                if (req_fire && !req_write) begin
                    rsp_rdata <= mem[req_addr];
                end
            end

            if (req_fire && req_write) begin
                for (int byte_idx = 0; byte_idx < WSTRB_W; byte_idx++) begin
                    if (req_wstrb[byte_idx]) begin
                        mem[req_addr][byte_idx*8 +: 8] <= req_wdata[byte_idx*8 +: 8];
                    end
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(req_fire && (int'(req_addr) >= DEPTH)))
                else $error("sram_1p_wrapper address out of range");
            assert (!(rsp_valid && $isunknown(rsp_rdata)))
                else $error("sram_1p_wrapper no_unknown_output_when_valid failed");

            if ($past(rst_n) && $past(rsp_valid && !rsp_ready)) begin
                assert (rsp_valid)
                    else $error("sram_1p_wrapper response valid dropped under backpressure");
                assert ($stable(rsp_rdata))
                    else $error("sram_1p_wrapper response data changed under backpressure");
            end
        end
    end
`endif
endmodule

`default_nettype wire
