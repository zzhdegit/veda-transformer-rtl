`default_nettype none

module lane_mask_generator #(
    parameter int PE_NUM = 8,
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1)
) (
    input  logic                         use_explicit_mask,
    input  logic [PE_NUM-1:0]            explicit_lane_mask,
    input  logic [LANE_COUNT_W-1:0]      active_lanes,
    output logic [PE_NUM-1:0]            lane_mask
);
    initial begin
        if (PE_NUM <= 0) begin
            $fatal(1, "lane_mask_generator PE_NUM must be positive");
        end
    end

    always_comb begin
        lane_mask = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            lane_mask[lane] = (active_lanes > LANE_COUNT_W'(lane));
        end
        if (use_explicit_mask) begin
            lane_mask = explicit_lane_mask;
        end
    end

`ifndef SYNTHESIS
    always_comb begin
        assert (active_lanes <= LANE_COUNT_W'(PE_NUM))
            else $error("lane_mask_generator lane_mask_legal failed");
    end
`endif
endmodule

`default_nettype wire
