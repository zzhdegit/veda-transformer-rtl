`default_nettype none

module accumulator_bank #(
    parameter int PE_NUM = 8,
    parameter bit RESET_TO_ZERO = 1'b1
) (
    input  logic                   clk,
    input  logic                   rst_n,

    input  logic                   clear_valid,
    input  logic [PE_NUM-1:0]      clear_mask,

    input  logic                   update_valid,
    input  logic [PE_NUM-1:0]      update_mask,
    input  logic [PE_NUM*32-1:0]   update_values,

    output logic [PE_NUM*32-1:0]   read_values
);
    logic [31:0] acc_q [0:PE_NUM-1];

    initial begin
        if (PE_NUM <= 0) begin
            $fatal(1, "accumulator_bank PE_NUM must be positive");
        end
    end

    genvar lane_g;
    generate
        for (lane_g = 0; lane_g < PE_NUM; lane_g++) begin : g_read_pack
            assign read_values[lane_g*32 +: 32] = acc_q[lane_g];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if (RESET_TO_ZERO) begin
                for (int lane = 0; lane < PE_NUM; lane++) begin
                    acc_q[lane] <= 32'd0;
                end
            end
        end else begin
            if (clear_valid) begin
                for (int lane = 0; lane < PE_NUM; lane++) begin
                    if (clear_mask[lane]) begin
                        acc_q[lane] <= 32'd0;
                    end
                end
            end

            if (update_valid) begin
                for (int lane = 0; lane < PE_NUM; lane++) begin
                    if (update_mask[lane]) begin
                        acc_q[lane] <= update_values[lane*32 +: 32];
                    end
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(update_valid && $isunknown(update_mask)))
                else $error("accumulator_bank update mask unknown");
            assert (!(clear_valid && $isunknown(clear_mask)))
                else $error("accumulator_bank clear mask unknown");
            assert (!(update_valid && $isunknown(update_values)))
                else $error("accumulator_bank accumulator_write_only_on_valid_result failed");
            assert (!(|clear_mask && !clear_valid && update_valid && $isunknown(update_values)))
                else $error("accumulator_bank unknown update payload");
        end
    end
`endif
endmodule

`default_nettype wire
