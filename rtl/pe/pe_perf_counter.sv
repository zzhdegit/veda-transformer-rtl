`default_nettype none

module pe_perf_counter #(
    parameter int PE_NUM = 8,
    parameter int COUNTER_W = 64,
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         clear,

    input  logic                         busy,
    input  logic                         lane_op_fire,
    input  logic [LANE_COUNT_W-1:0]      active_lanes,
    input  logic                         input_stall,
    input  logic                         output_stall,
    input  logic                         mode_switch,
    input  logic                         tile_fire,
    input  logic                         operation_fire,
    input  logic                         invalid_fire,

    output logic [COUNTER_W-1:0]         total_cycles,
    output logic [COUNTER_W-1:0]         busy_cycles,
    output logic [COUNTER_W-1:0]         active_lane_cycles,
    output logic [COUNTER_W-1:0]         available_lane_cycles,
    output logic [COUNTER_W-1:0]         input_stall_cycles,
    output logic [COUNTER_W-1:0]         output_stall_cycles,
    output logic [COUNTER_W-1:0]         mode_switch_cycles,
    output logic [COUNTER_W-1:0]         tile_count,
    output logic [COUNTER_W-1:0]         operation_count,
    output logic [COUNTER_W-1:0]         invalid_count
);
    initial begin
        if (PE_NUM <= 0) begin
            $fatal(1, "pe_perf_counter PE_NUM must be positive");
        end
        if (COUNTER_W <= 0) begin
            $fatal(1, "pe_perf_counter COUNTER_W must be positive");
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_cycles         <= '0;
            busy_cycles          <= '0;
            active_lane_cycles   <= '0;
            available_lane_cycles <= '0;
            input_stall_cycles   <= '0;
            output_stall_cycles  <= '0;
            mode_switch_cycles   <= '0;
            tile_count           <= '0;
            operation_count      <= '0;
            invalid_count        <= '0;
        end else if (clear) begin
            total_cycles         <= '0;
            busy_cycles          <= '0;
            active_lane_cycles   <= '0;
            available_lane_cycles <= '0;
            input_stall_cycles   <= '0;
            output_stall_cycles  <= '0;
            mode_switch_cycles   <= '0;
            tile_count           <= '0;
            operation_count      <= '0;
            invalid_count        <= '0;
        end else begin
            total_cycles <= total_cycles + COUNTER_W'(1);
            if (busy) begin
                busy_cycles <= busy_cycles + COUNTER_W'(1);
            end
            if (lane_op_fire) begin
                active_lane_cycles <= active_lane_cycles + COUNTER_W'(active_lanes);
                available_lane_cycles <= available_lane_cycles + COUNTER_W'(PE_NUM);
            end
            if (input_stall) begin
                input_stall_cycles <= input_stall_cycles + COUNTER_W'(1);
            end
            if (output_stall) begin
                output_stall_cycles <= output_stall_cycles + COUNTER_W'(1);
            end
            if (mode_switch) begin
                mode_switch_cycles <= mode_switch_cycles + COUNTER_W'(1);
            end
            if (tile_fire) begin
                tile_count <= tile_count + COUNTER_W'(1);
            end
            if (operation_fire) begin
                operation_count <= operation_count + COUNTER_W'(1);
            end
            if (invalid_fire) begin
                invalid_count <= invalid_count + COUNTER_W'(1);
            end
        end
    end
endmodule

`default_nettype wire
