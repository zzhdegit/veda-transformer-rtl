`default_nettype none

module stream_reg #(
    parameter int DATA_W = 32,
    parameter int META_W = 1
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [DATA_W-1:0]     in_data,
    input  logic [META_W-1:0]     in_meta,
    input  logic                  in_last,

    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [DATA_W-1:0]     out_data,
    output logic [META_W-1:0]     out_meta,
    output logic                  out_last
);
    initial begin
        if (DATA_W <= 0) begin
            $fatal(1, "stream_reg DATA_W must be positive");
        end
        if (META_W <= 0) begin
            $fatal(1, "stream_reg META_W must be positive");
        end
    end

    assign in_ready = out_ready || !out_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= '0;
            out_meta  <= '0;
            out_last  <= 1'b0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            out_data  <= in_data;
            out_meta  <= in_meta;
            out_last  <= in_last;
        end
    end

`ifndef SYNTHESIS
    logic [31:0] accepted_count;
    logic [31:0] emitted_count;

    wire input_fire  = in_valid && in_ready;
    wire output_fire = out_valid && out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accepted_count <= '0;
            emitted_count  <= '0;
        end else begin
            if (input_fire) begin
                accepted_count <= accepted_count + 32'd1;
            end
            if (output_fire) begin
                emitted_count <= emitted_count + 32'd1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(out_valid && $isunknown({out_data, out_meta, out_last})))
                else $error("stream_reg no_unknown_output_when_valid failed");

            assert (accepted_count >= emitted_count)
                else $error("stream_reg transaction_count_conserved failed");
            assert ((accepted_count - emitted_count) <= 32'd1)
                else $error("stream_reg occupancy range failed");

            if ($past(rst_n)) begin
                if ($past(in_valid && !in_ready)) begin
                    assert (in_valid)
                        else $error("stream_reg valid_stable_until_ready failed");
                    assert ($stable(in_data))
                        else $error("stream_reg data_stable_until_ready failed");
                    assert ($stable(in_meta))
                        else $error("stream_reg metadata_stable_until_ready failed");
                    assert ($stable(in_last))
                        else $error("stream_reg last_stable_until_ready failed");
                end

                if ($past(out_valid && !out_ready)) begin
                    assert (out_valid)
                        else $error("stream_reg output valid dropped under backpressure");
                    assert ($stable(out_data))
                        else $error("stream_reg output data changed under backpressure");
                    assert ($stable(out_meta))
                        else $error("stream_reg output metadata changed under backpressure");
                    assert ($stable(out_last))
                        else $error("stream_reg output last changed under backpressure");
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
