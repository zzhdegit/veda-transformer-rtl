`default_nettype none

module tb_paper_array_inner;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam logic [1:0] MODE_INNER_PRODUCT = 2'd1;
    localparam logic [1:0] MODE_OUTER_PRODUCT = 2'd2;

    logic clk;
    logic rst_n;

    logic cmd_valid;
    logic cmd_ready;
    logic [1:0] cmd_mode;
    logic [15:0] cmd_k_size;
    logic [15:0] cmd_m_size;
    logic [15:0] cmd_n_size;
    logic [15:0] cmd_tile_id;
    logic [META_W-1:0] cmd_meta;
    logic cmd_clear_acc;
    logic cmd_tile_last;
    logic [1:0] cmd_group_mask;
    logic [127:0] cmd_lane_mask;
    logic [31:0] cmd_scalar_fp32;
    logic [128*16-1:0] cmd_operand_a_fp16;
    logic [128*16-1:0] cmd_operand_b_fp16;
    logic cmd_last;

    logic result_valid;
    logic result_ready;
    logic [1:0] result_mode;
    logic [15:0] result_tile_id;
    logic [31:0] result_scalar_fp32;
    logic [128*32-1:0] result_vector_fp32;
    logic [127:0] result_lane_mask;
    logic [7:0] result_status;
    logic result_invalid;
    logic [META_W-1:0] result_meta;
    logic result_last;

    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;

    logic [COUNTER_W-1:0] perf_paper_array_active_cycles;
    logic [COUNTER_W-1:0] perf_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] perf_inner_mode_cycles;
    logic [COUNTER_W-1:0] perf_outer_mode_cycles;
    logic [COUNTER_W-1:0] perf_group0_active_cycles;
    logic [COUNTER_W-1:0] perf_group1_active_cycles;
    logic [COUNTER_W-1:0] perf_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] perf_array_input_stall_cycles;
    logic [COUNTER_W-1:0] perf_array_output_stall_cycles;
    logic [COUNTER_W-1:0] mode_switch_before_reset;

    paper_array_8x8x2 #(
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(1'b1)
    ) dut (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .cmd_valid                      (cmd_valid),
        .cmd_ready                      (cmd_ready),
        .cmd_mode                       (cmd_mode),
        .cmd_k_size                     (cmd_k_size),
        .cmd_m_size                     (cmd_m_size),
        .cmd_n_size                     (cmd_n_size),
        .cmd_tile_id                    (cmd_tile_id),
        .cmd_meta                       (cmd_meta),
        .cmd_clear_acc                  (cmd_clear_acc),
        .cmd_tile_last                  (cmd_tile_last),
        .cmd_group_mask                 (cmd_group_mask),
        .cmd_lane_mask                  (cmd_lane_mask),
        .cmd_scalar_fp32                (cmd_scalar_fp32),
        .cmd_operand_a_fp16             (cmd_operand_a_fp16),
        .cmd_operand_b_fp16             (cmd_operand_b_fp16),
        .cmd_last                       (cmd_last),
        .result_valid                   (result_valid),
        .result_ready                   (result_ready),
        .result_mode                    (result_mode),
        .result_tile_id                 (result_tile_id),
        .result_scalar_fp32             (result_scalar_fp32),
        .result_vector_fp32             (result_vector_fp32),
        .result_lane_mask               (result_lane_mask),
        .result_status                  (result_status),
        .result_invalid                 (result_invalid),
        .result_meta                    (result_meta),
        .result_last                    (result_last),
        .done_valid                     (done_valid),
        .done_ready                     (done_ready),
        .done_status                    (done_status),
        .done_invalid                   (done_invalid),
        .done_meta                      (done_meta),
        .perf_paper_array_active_cycles (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles   (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles         (perf_inner_mode_cycles),
        .perf_outer_mode_cycles         (perf_outer_mode_cycles),
        .perf_group0_active_cycles      (perf_group0_active_cycles),
        .perf_group1_active_cycles      (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles     (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles        (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles  (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles (perf_array_output_stall_cycles)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function automatic [31:0] fp32_for_count(input int count);
        begin
            unique case (count)
                1: fp32_for_count = 32'h3F800000;
                7: fp32_for_count = 32'h40E00000;
                8: fp32_for_count = 32'h41000000;
                9: fp32_for_count = 32'h41100000;
                15: fp32_for_count = 32'h41700000;
                16: fp32_for_count = 32'h41800000;
                31: fp32_for_count = 32'h41F80000;
                32: fp32_for_count = 32'h42000000;
                64: fp32_for_count = 32'h42800000;
                128: fp32_for_count = 32'h43000000;
                default: fp32_for_count = 32'd0;
            endcase
        end
    endfunction

    function automatic [127:0] mask_for_count(input int count);
        begin
            mask_for_count = 128'd0;
            for (int i = 0; i < 128; i++) begin
                if (i < count) begin
                    mask_for_count[i] = 1'b1;
                end
            end
        end
    endfunction

    task automatic clear_inputs;
        begin
            cmd_valid = 1'b0;
            cmd_mode = MODE_INNER_PRODUCT;
            cmd_k_size = 16'd0;
            cmd_m_size = 16'd0;
            cmd_n_size = 16'd0;
            cmd_tile_id = 16'd0;
            cmd_meta = 16'd0;
            cmd_clear_acc = 1'b0;
            cmd_tile_last = 1'b1;
            cmd_group_mask = 2'b11;
            cmd_lane_mask = 128'd0;
            cmd_scalar_fp32 = 32'd0;
            cmd_operand_a_fp16 = '0;
            cmd_operand_b_fp16 = '0;
            cmd_last = 1'b1;
            result_ready = 1'b1;
            done_ready = 1'b1;
        end
    endtask

    task automatic set_lane(input int lane, input logic [15:0] a, input logic [15:0] b);
        begin
            cmd_operand_a_fp16[lane*16 +: 16] = a;
            cmd_operand_b_fp16[lane*16 +: 16] = b;
        end
    endtask

    task automatic send_inner_ones(input int length, input logic [1:0] group_mask, input [15:0] tile_id);
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            cmd_mode = MODE_INNER_PRODUCT;
            cmd_k_size = length[15:0];
            cmd_m_size = 16'd1;
            cmd_n_size = 16'd1;
            cmd_tile_id = tile_id;
            cmd_meta = 16'h8100 | tile_id;
            cmd_clear_acc = 1'b1;
            cmd_tile_last = 1'b1;
            cmd_group_mask = group_mask;
            cmd_lane_mask = mask_for_count(length);
            cmd_scalar_fp32 = 32'd0;
            cmd_operand_a_fp16 = '0;
            cmd_operand_b_fp16 = '0;
            cmd_last = 1'b1;
            for (int i = 0; i < length; i++) begin
                set_lane(i, 16'h3C00, 16'h3C00);
            end
            while (!cmd_ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic send_cancellation;
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            cmd_mode = MODE_INNER_PRODUCT;
            cmd_k_size = 16'd2;
            cmd_m_size = 16'd1;
            cmd_n_size = 16'd1;
            cmd_tile_id = 16'h00CA;
            cmd_meta = 16'h80CA;
            cmd_clear_acc = 1'b1;
            cmd_tile_last = 1'b1;
            cmd_group_mask = 2'b01;
            cmd_lane_mask = mask_for_count(2);
            cmd_scalar_fp32 = 32'd0;
            cmd_operand_a_fp16 = '0;
            cmd_operand_b_fp16 = '0;
            set_lane(0, 16'h3C00, 16'h3C00);
            set_lane(1, 16'h3C00, 16'hBC00);
            cmd_last = 1'b1;
            while (!cmd_ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic send_outer(input bit clear_acc, input bit tile_last, input [15:0] tile_id);
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            cmd_mode = MODE_OUTER_PRODUCT;
            cmd_k_size = 16'd4;
            cmd_m_size = 16'd1;
            cmd_n_size = 16'd4;
            cmd_tile_id = tile_id;
            cmd_meta = 16'h8200 | tile_id;
            cmd_clear_acc = clear_acc;
            cmd_tile_last = tile_last;
            cmd_group_mask = 2'b01;
            cmd_lane_mask = mask_for_count(4);
            cmd_scalar_fp32 = 32'h3F800000;
            cmd_operand_a_fp16 = '0;
            cmd_operand_b_fp16 = '0;
            set_lane(0, 16'h3C00, 16'h3C00);
            set_lane(1, 16'h3C00, 16'h4000);
            set_lane(2, 16'h3C00, 16'hBC00);
            set_lane(3, 16'h3C00, 16'h3800);
            cmd_last = tile_last;
            while (!cmd_ready) begin
                @(posedge clk);
            end
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic expect_inner(input [31:0] expected, input [15:0] tile_id);
        begin
            done_ready <= 1'b0;
            wait (result_valid);
            result_ready <= 1'b0;
            repeat (3) @(posedge clk);
            if (result_scalar_fp32 !== expected) begin
                $error("CHECK_FAIL inner expected=%h got=%h", expected, result_scalar_fp32);
                $finish;
            end
            if (result_mode !== MODE_INNER_PRODUCT || result_tile_id !== tile_id || result_invalid) begin
                $error("CHECK_FAIL inner metadata/status mode=%0d tile=%h invalid=%0b", result_mode, result_tile_id, result_invalid);
                $finish;
            end
            result_ready <= 1'b1;
            wait (done_valid);
            if (done_invalid) begin
                $error("CHECK_FAIL inner done invalid");
                $finish;
            end
            done_ready <= 1'b1;
            @(posedge clk);
        end
    endtask

    task automatic expect_outer_vector;
        begin
            done_ready <= 1'b0;
            wait (result_valid);
            if (result_mode !== MODE_OUTER_PRODUCT || result_invalid) begin
                $error("CHECK_FAIL outer status");
                $finish;
            end
            if (result_vector_fp32[0*32 +: 32] !== 32'h3F800000 ||
                result_vector_fp32[1*32 +: 32] !== 32'h40000000 ||
                result_vector_fp32[2*32 +: 32] !== 32'hBF800000 ||
                result_vector_fp32[3*32 +: 32] !== 32'h3F000000) begin
                $error("CHECK_FAIL outer vector got %h %h %h %h",
                       result_vector_fp32[0*32 +: 32],
                       result_vector_fp32[1*32 +: 32],
                       result_vector_fp32[2*32 +: 32],
                       result_vector_fp32[3*32 +: 32]);
                $finish;
            end
            @(posedge clk);
            wait (done_valid);
            done_ready <= 1'b1;
            @(posedge clk);
        end
    endtask

    task automatic reset_dut;
        begin
            rst_n = 1'b0;
            clear_inputs();
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (3) @(posedge clk);
        end
    endtask

    initial begin
        reset_dut();

        send_inner_ones(1, 2'b01, 16'd1);     expect_inner(fp32_for_count(1), 16'd1);
        send_inner_ones(7, 2'b01, 16'd7);     expect_inner(fp32_for_count(7), 16'd7);
        send_inner_ones(8, 2'b01, 16'd8);     expect_inner(fp32_for_count(8), 16'd8);
        send_inner_ones(9, 2'b01, 16'd9);     expect_inner(fp32_for_count(9), 16'd9);
        send_inner_ones(15, 2'b01, 16'd15);   expect_inner(fp32_for_count(15), 16'd15);
        send_inner_ones(16, 2'b01, 16'd16);   expect_inner(fp32_for_count(16), 16'd16);
        send_inner_ones(31, 2'b01, 16'd31);   expect_inner(fp32_for_count(31), 16'd31);
        send_inner_ones(32, 2'b01, 16'd32);   expect_inner(fp32_for_count(32), 16'd32);
        send_inner_ones(64, 2'b01, 16'd64);   expect_inner(fp32_for_count(64), 16'd64);
        send_inner_ones(128, 2'b11, 16'd128); expect_inner(fp32_for_count(128), 16'd128);

        send_cancellation();
        expect_inner(32'h00000000, 16'h00CA);

        send_outer(1'b1, 1'b1, 16'd201);
        expect_outer_vector();
        mode_switch_before_reset = perf_mode_switch_cycles;
        if (mode_switch_before_reset == 0) begin
            $error("CHECK_FAIL expected mode switch counter to increment");
            $finish;
        end

        send_inner_ones(128, 2'b11, 16'd301);
        repeat (4) @(posedge clk);
        rst_n <= 1'b0;
        repeat (5) @(posedge clk);
        rst_n <= 1'b1;
        repeat (3) @(posedge clk);

        send_inner_ones(1, 2'b01, 16'd302);
        expect_inner(fp32_for_count(1), 16'd302);

        $display("STAGE8C_PAPER_ARRAY_PASS active=%0d inner=%0d outer=%0d group0=%0d group1=%0d tail=%0d mode_switch=%0d output_stall=%0d",
                 perf_paper_array_active_cycles,
                 perf_inner_mode_cycles,
                 perf_outer_mode_cycles,
                 perf_group0_active_cycles,
                 perf_group1_active_cycles,
                 perf_tail_masked_pe_cycles,
                 mode_switch_before_reset,
                 perf_array_output_stall_cycles);
        $finish;
    end

    initial begin
        #20000000;
        $error("CHECK_FAIL timeout");
        $finish;
    end
endmodule

`default_nettype wire
