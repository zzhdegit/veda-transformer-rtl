`default_nettype none

module score_buffer #(
    parameter int DEPTH = 32,
    parameter int COUNTER_W = 64,
    parameter int READ_LATENCY = 1,
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    localparam int COUNT_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH + 1)
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic                    read_rewind,

    input  logic                    wr_valid,
    output logic                    wr_ready,
    input  logic [ADDR_W-1:0]       wr_addr,
    input  logic [31:0]             wr_score,

    input  logic                    rd_valid,
    output logic                    rd_ready,
    input  logic [ADDR_W-1:0]       rd_addr,

    output logic                    rsp_valid,
    input  logic                    rsp_ready,
    output logic [ADDR_W-1:0]       rsp_addr,
    output logic [31:0]             rsp_score,

    output logic [COUNT_W-1:0]      valid_length,
    output logic [COUNT_W-1:0]      occupancy,
    output logic [COUNT_W-1:0]      peak_occupancy
);
    localparam int WSTRB_W = 4;

    logic mem_wr_ready;
    logic mem_rd_ready;
    logic mem_rsp_valid;
    logic mem_rsp_ready;
    logic [31:0] mem_rsp_rdata;
    logic [ADDR_W-1:0] rsp_addr_q;
    logic [COUNT_W-1:0] valid_count_q;
    logic [COUNT_W-1:0] read_count_q;
    logic [COUNT_W-1:0] occupancy_next;
    logic wr_fire;
    logic rd_fire;
    logic rsp_fire;

    initial begin
        if (DEPTH <= 0) begin
            $fatal(1, "score_buffer DEPTH must be positive");
        end
        if (READ_LATENCY != 1) begin
            $fatal(1, "score_buffer explicitly supports READ_LATENCY=1");
        end
    end

    assign wr_ready = mem_wr_ready && (valid_count_q < COUNT_W'(DEPTH));
    assign rd_ready = mem_rd_ready;
    assign wr_fire = wr_valid && wr_ready;
    assign rd_fire = rd_valid && rd_ready;
    assign rsp_fire = rsp_valid && rsp_ready;
    assign mem_rsp_ready = rsp_ready;
    assign rsp_valid = mem_rsp_valid;
    assign rsp_score = mem_rsp_rdata;
    assign rsp_addr = rsp_addr_q;
    assign valid_length = valid_count_q;
    assign occupancy = (valid_count_q >= read_count_q) ? (valid_count_q - read_count_q) : '0;

    always_comb begin
        occupancy_next = occupancy;
        unique case ({wr_fire, rsp_fire})
            2'b10: occupancy_next = occupancy + COUNT_W'(1);
            2'b01: occupancy_next = (occupancy == '0) ? '0 : (occupancy - COUNT_W'(1));
            default: occupancy_next = occupancy;
        endcase
    end

    sram_2p_wrapper #(
        .DATA_W(32),
        .DEPTH(DEPTH),
        .READ_LATENCY(READ_LATENCY)
    ) u_mem (
        .clk       (clk),
        .rst_n     (rst_n),
        .clk_en    (1'b1),
        .wr_valid  (wr_valid && (valid_count_q < COUNT_W'(DEPTH))),
        .wr_ready  (mem_wr_ready),
        .wr_addr   (wr_addr),
        .wr_data   (wr_score),
        .wr_wstrb  ({WSTRB_W{1'b1}}),
        .rd_valid  (rd_valid),
        .rd_ready  (mem_rd_ready),
        .rd_addr   (rd_addr),
        .rsp_valid (mem_rsp_valid),
        .rsp_ready (mem_rsp_ready),
        .rsp_rdata (mem_rsp_rdata)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_count_q <= '0;
            read_count_q <= '0;
            rsp_addr_q <= '0;
            peak_occupancy <= '0;
        end else begin
            if (clear) begin
                valid_count_q <= '0;
                read_count_q <= '0;
                peak_occupancy <= '0;
            end else begin
                if (read_rewind) begin
                    read_count_q <= '0;
                end
                if (wr_fire) begin
                    valid_count_q <= valid_count_q + COUNT_W'(1);
                end
                if (rsp_fire && !read_rewind) begin
                    read_count_q <= read_count_q + COUNT_W'(1);
                end
                if (occupancy_next > peak_occupancy) begin
                    peak_occupancy <= occupancy_next;
                end
            end

            if (rd_fire) begin
                rsp_addr_q <= rd_addr;
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && !clear) begin
            assert (!(wr_fire && (wr_addr != valid_count_q[ADDR_W-1:0])))
                else $error("score_buffer score_write_order failed");
            assert (!(wr_fire && (valid_count_q >= COUNT_W'(DEPTH))))
                else $error("score_buffer no buffer overflow failed");
            assert (!(rd_fire && (COUNT_W'(rd_addr) >= valid_count_q)))
                else $error("score_buffer no_score_read_before_written failed");
            assert (!(rsp_valid && $isunknown({rsp_addr, rsp_score})))
                else $error("score_buffer no_unknown_output_when_valid failed");
            assert (valid_count_q <= COUNT_W'(DEPTH))
                else $error("score_buffer valid length out of range");

            if ($past(rst_n) && $past(rsp_valid && !rsp_ready)) begin
                assert (rsp_valid)
                    else $error("score_buffer response valid dropped under backpressure");
                assert ($stable({rsp_addr, rsp_score}))
                    else $error("score_buffer response payload changed under backpressure");
            end
        end
    end
`endif
endmodule

`default_nettype wire
