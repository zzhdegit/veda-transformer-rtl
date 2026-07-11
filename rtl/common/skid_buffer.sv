`default_nettype none

module skid_buffer #(
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
    logic                  skid_valid;
    logic [DATA_W-1:0]     skid_data;
    logic [META_W-1:0]     skid_meta;
    logic                  skid_last;

    initial begin
        if (DATA_W <= 0) begin
            $fatal(1, "skid_buffer DATA_W must be positive");
        end
        if (META_W <= 0) begin
            $fatal(1, "skid_buffer META_W must be positive");
        end
    end

    // This ready only depends on local state, so the buffer cuts ready paths.
    assign in_ready = !skid_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid  <= 1'b0;
            out_data   <= '0;
            out_meta   <= '0;
            out_last   <= 1'b0;
            skid_valid <= 1'b0;
            skid_data  <= '0;
            skid_meta  <= '0;
            skid_last  <= 1'b0;
        end else begin
            if (out_valid && !out_ready) begin
                if (in_valid && in_ready) begin
                    skid_valid <= 1'b1;
                    skid_data  <= in_data;
                    skid_meta  <= in_meta;
                    skid_last  <= in_last;
                end
            end else begin
                if (skid_valid) begin
                    out_valid  <= 1'b1;
                    out_data   <= skid_data;
                    out_meta   <= skid_meta;
                    out_last   <= skid_last;
                    skid_valid <= 1'b0;
                end else begin
                    out_valid <= in_valid;
                    out_data  <= in_data;
                    out_meta  <= in_meta;
                    out_last  <= in_last;
                end
            end
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
                else $error("skid_buffer no_unknown_output_when_valid failed");
            assert (accepted_count >= emitted_count)
                else $error("skid_buffer transaction_count_conserved failed");
            assert ((accepted_count - emitted_count) <= 32'd2)
                else $error("skid_buffer occupancy range failed");

            if ($past(rst_n)) begin
                if ($past(in_valid && !in_ready)) begin
                    assert (in_valid)
                        else $error("skid_buffer valid_stable_until_ready failed");
                    assert ($stable(in_data))
                        else $error("skid_buffer data_stable_until_ready failed");
                    assert ($stable(in_meta))
                        else $error("skid_buffer metadata_stable_until_ready failed");
                    assert ($stable(in_last))
                        else $error("skid_buffer last_stable_until_ready failed");
                end

                if ($past(out_valid && !out_ready)) begin
                    assert (out_valid)
                        else $error("skid_buffer output valid dropped under backpressure");
                    assert ($stable(out_data))
                        else $error("skid_buffer output data changed under backpressure");
                    assert ($stable(out_meta))
                        else $error("skid_buffer output metadata changed under backpressure");
                    assert ($stable(out_last))
                        else $error("skid_buffer output last changed under backpressure");
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
