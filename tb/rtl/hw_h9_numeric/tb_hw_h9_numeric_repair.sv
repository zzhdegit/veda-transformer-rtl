`timescale 1ns/1ps
`default_nettype none

module tb_hw_h9_numeric_repair;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int MAX_ADD_VECTORS = 64;
    localparam int MAX_REDUCTION_VECTORS = 160;
    localparam int MAX_CORE_VECTORS = 96;

    logic clk;
    logic rst_n;

    logic add_in_valid;
    logic add_in_ready;
    logic [31:0] add_in_a;
    logic [31:0] add_in_b;
    logic [META_W-1:0] add_in_meta;
    logic add_in_last;
    logic add_out_valid;
    logic add_out_ready;
    logic [31:0] add_out_result;
    logic [7:0] add_out_status;
    logic add_out_invalid;
    logic [META_W-1:0] add_out_meta;
    logic add_out_last;

    logic red_in_valid;
    logic red_in_ready;
    logic [PE_NUM*32-1:0] red_in_values;
    logic [PE_NUM-1:0] red_in_mask;
    logic [META_W-1:0] red_in_meta;
    logic red_in_last;
    logic red_out_valid;
    logic red_out_ready;
    logic [31:0] red_out_sum;
    logic [7:0] red_out_status;
    logic red_out_invalid;
    logic [META_W-1:0] red_out_meta;
    logic red_out_last;
    logic red_busy;

    logic core_in_valid;
    logic core_in_ready;
    logic [1:0] core_in_mode;
    logic core_in_clear;
    logic core_in_tile_first;
    logic core_in_tile_last;
    logic core_in_use_explicit_mask;
    logic [$clog2(PE_NUM+1)-1:0] core_in_active_lanes;
    logic [PE_NUM-1:0] core_in_lane_mask;
    logic [31:0] core_in_scalar;
    logic [PE_NUM*16-1:0] core_in_a;
    logic [PE_NUM*16-1:0] core_in_b;
    logic [META_W-1:0] core_in_meta;
    logic core_in_last;
    logic core_out_valid;
    logic core_out_ready;
    logic [1:0] core_out_mode;
    logic [31:0] core_out_scalar;
    logic [PE_NUM*32-1:0] core_out_vector;
    logic [PE_NUM-1:0] core_out_mask;
    logic [7:0] core_out_status;
    logic core_out_invalid;
    logic [META_W-1:0] core_out_meta;
    logic core_out_last;
    logic [63:0] core_perf_total_cycles;
    logic [63:0] core_perf_busy_cycles;
    logic [63:0] core_perf_active_lane_cycles;
    logic [63:0] core_perf_available_lane_cycles;
    logic [63:0] core_perf_input_stall_cycles;
    logic [63:0] core_perf_output_stall_cycles;
    logic [63:0] core_perf_mode_switch_cycles;
    logic [63:0] core_perf_tile_count;
    logic [63:0] core_perf_operation_count;
    logic [63:0] core_perf_invalid_count;

    logic [31:0] add_vec_a [0:MAX_ADD_VECTORS-1];
    logic [31:0] add_vec_b [0:MAX_ADD_VECTORS-1];
    logic [31:0] add_vec_expected [0:MAX_ADD_VECTORS-1];
    logic [META_W-1:0] add_vec_meta [0:MAX_ADD_VECTORS-1];
    logic add_vec_last [0:MAX_ADD_VECTORS-1];
    int add_count;

    logic [PE_NUM-1:0] red_vec_mask [0:MAX_REDUCTION_VECTORS-1];
    logic [PE_NUM*32-1:0] red_vec_values [0:MAX_REDUCTION_VECTORS-1];
    logic [31:0] red_vec_expected [0:MAX_REDUCTION_VECTORS-1];
    logic [META_W-1:0] red_vec_meta [0:MAX_REDUCTION_VECTORS-1];
    logic red_vec_last [0:MAX_REDUCTION_VECTORS-1];
    int red_count;

    logic [1:0] core_vec_mode [0:MAX_CORE_VECTORS-1];
    logic core_vec_clear [0:MAX_CORE_VECTORS-1];
    logic core_vec_first [0:MAX_CORE_VECTORS-1];
    logic core_vec_last_tile [0:MAX_CORE_VECTORS-1];
    logic [PE_NUM-1:0] core_vec_mask [0:MAX_CORE_VECTORS-1];
    logic [31:0] core_vec_scalar [0:MAX_CORE_VECTORS-1];
    logic [PE_NUM*16-1:0] core_vec_a [0:MAX_CORE_VECTORS-1];
    logic [PE_NUM*16-1:0] core_vec_b [0:MAX_CORE_VECTORS-1];
    logic core_vec_expect [0:MAX_CORE_VECTORS-1];
    logic [31:0] core_vec_exp_scalar [0:MAX_CORE_VECTORS-1];
    logic [PE_NUM*32-1:0] core_vec_exp_vector [0:MAX_CORE_VECTORS-1];
    logic [META_W-1:0] core_vec_meta [0:MAX_CORE_VECTORS-1];
    logic core_vec_last [0:MAX_CORE_VECTORS-1];
    int core_count;

    logic [31:0] add_expected_q[$];
    logic [META_W-1:0] add_meta_q[$];
    logic add_last_q[$];
    logic [31:0] red_expected_q[$];
    logic [META_W-1:0] red_meta_q[$];
    logic red_last_q[$];
    logic [31:0] core_expected_q[$];
    logic [META_W-1:0] core_meta_q[$];
    logic core_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_add_wrapper #(.META_W(META_W)) u_add (
        .clk(clk), .rst_n(rst_n),
        .in_valid(add_in_valid), .in_ready(add_in_ready),
        .in_a(add_in_a), .in_b(add_in_b), .in_meta(add_in_meta), .in_last(add_in_last),
        .out_valid(add_out_valid), .out_ready(add_out_ready),
        .out_result(add_out_result), .out_status(add_out_status),
        .out_invalid(add_out_invalid), .out_meta(add_out_meta), .out_last(add_out_last)
    );

    fp32_reduction_tree #(.PE_NUM(PE_NUM), .META_W(META_W)) u_reduction (
        .clk(clk), .rst_n(rst_n),
        .in_valid(red_in_valid), .in_ready(red_in_ready),
        .in_values(red_in_values), .in_lane_mask(red_in_mask), .in_meta(red_in_meta), .in_last(red_in_last),
        .out_valid(red_out_valid), .out_ready(red_out_ready),
        .out_sum(red_out_sum), .out_status(red_out_status),
        .out_invalid(red_out_invalid), .out_meta(red_out_meta), .out_last(red_out_last), .busy(red_busy)
    );

    reconfigurable_pe_core #(.PE_NUM(PE_NUM), .META_W(META_W)) u_core (
        .clk(clk), .rst_n(rst_n),
        .in_valid(core_in_valid), .in_ready(core_in_ready), .in_mode(core_in_mode),
        .in_clear(core_in_clear), .in_tile_first(core_in_tile_first), .in_tile_last(core_in_tile_last),
        .in_use_explicit_mask(core_in_use_explicit_mask), .in_active_lanes(core_in_active_lanes),
        .in_lane_mask(core_in_lane_mask), .in_scalar_fp32(core_in_scalar),
        .in_vector_a_fp16(core_in_a), .in_vector_b_fp16(core_in_b),
        .in_meta(core_in_meta), .in_last(core_in_last),
        .out_valid(core_out_valid), .out_ready(core_out_ready), .out_mode(core_out_mode),
        .out_scalar_fp32(core_out_scalar), .out_vector_fp32(core_out_vector),
        .out_lane_mask(core_out_mask), .out_status(core_out_status),
        .out_invalid(core_out_invalid), .out_meta(core_out_meta), .out_last(core_out_last),
        .perf_total_cycles(core_perf_total_cycles), .perf_busy_cycles(core_perf_busy_cycles),
        .perf_active_lane_cycles(core_perf_active_lane_cycles),
        .perf_available_lane_cycles(core_perf_available_lane_cycles),
        .perf_input_stall_cycles(core_perf_input_stall_cycles),
        .perf_output_stall_cycles(core_perf_output_stall_cycles),
        .perf_mode_switch_cycles(core_perf_mode_switch_cycles),
        .perf_tile_count(core_perf_tile_count), .perf_operation_count(core_perf_operation_count),
        .perf_invalid_count(core_perf_invalid_count)
    );

    task automatic tb_fail(input string message);
        begin
            $display("HW_H9_NUMERIC_REPAIR_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            add_in_valid = 1'b0; add_in_a = '0; add_in_b = '0; add_in_meta = '0; add_in_last = 1'b0; add_out_ready = 1'b0;
            red_in_valid = 1'b0; red_in_values = '0; red_in_mask = '0; red_in_meta = '0; red_in_last = 1'b0; red_out_ready = 1'b0;
            core_in_valid = 1'b0; core_in_mode = 2'd0; core_in_clear = 1'b0; core_in_tile_first = 1'b0;
            core_in_tile_last = 1'b0; core_in_use_explicit_mask = 1'b1; core_in_active_lanes = '0;
            core_in_lane_mask = '0; core_in_scalar = '0; core_in_a = '0; core_in_b = '0;
            core_in_meta = '0; core_in_last = 1'b0; core_out_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
            if (add_out_valid || red_out_valid || core_out_valid) tb_fail("valid output after reset");
        end
    endtask

    task automatic load_vectors;
        string add_path;
        string red_path;
        string core_path;
        int fd;
        int code;
        logic [31:0] a32;
        logic [31:0] b32;
        logic [31:0] expected32;
        logic [META_W-1:0] meta;
        logic last;
        logic [PE_NUM-1:0] mask;
        logic [31:0] values [0:PE_NUM-1];
        logic [1:0] mode;
        logic clear;
        logic first;
        logic last_tile;
        logic [31:0] scalar;
        logic [15:0] a16 [0:PE_NUM-1];
        logic [15:0] b16 [0:PE_NUM-1];
        logic expect_flag;
        logic [31:0] exp_vector [0:PE_NUM-1];
        begin
            if (!$value$plusargs("ADD_VECTOR_FILE=%s", add_path)) tb_fail("missing +ADD_VECTOR_FILE");
            if (!$value$plusargs("REDUCTION_VECTOR_FILE=%s", red_path)) tb_fail("missing +REDUCTION_VECTOR_FILE");
            if (!$value$plusargs("CORE_VECTOR_FILE=%s", core_path)) tb_fail("missing +CORE_VECTOR_FILE");

            fd = $fopen(add_path, "r");
            if (fd == 0) tb_fail("could not open add vector file");
            add_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %h %b\n", a32, b32, expected32, meta, last);
                if (code == 5) begin
                    if (add_count >= MAX_ADD_VECTORS) tb_fail("too many add vectors");
                    add_vec_a[add_count] = a32;
                    add_vec_b[add_count] = b32;
                    add_vec_expected[add_count] = expected32;
                    add_vec_meta[add_count] = meta;
                    add_vec_last[add_count] = last;
                    add_count++;
                end
            end
            $fclose(fd);

            fd = $fopen(red_path, "r");
            if (fd == 0) tb_fail("could not open reduction vector file");
            red_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %b\n",
                               mask, values[0], values[1], values[2], values[3], values[4],
                               values[5], values[6], values[7], expected32, meta, last);
                if (code == 12) begin
                    if (red_count >= MAX_REDUCTION_VECTORS) tb_fail("too many reduction vectors");
                    red_vec_mask[red_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        red_vec_values[red_count][lane*32 +: 32] = values[lane];
                    end
                    red_vec_expected[red_count] = expected32;
                    red_vec_meta[red_count] = meta;
                    red_vec_last[red_count] = last;
                    red_count++;
                end
            end
            $fclose(fd);

            fd = $fopen(core_path, "r");
            if (fd == 0) tb_fail("could not open core vector file");
            core_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd,
                    "%h %b %b %b %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %b %h %h %h %h %h %h %h %h %h %h %b\n",
                    mode, clear, first, last_tile, mask, scalar,
                    a16[0], a16[1], a16[2], a16[3], a16[4], a16[5], a16[6], a16[7],
                    b16[0], b16[1], b16[2], b16[3], b16[4], b16[5], b16[6], b16[7],
                    expect_flag, expected32,
                    exp_vector[0], exp_vector[1], exp_vector[2], exp_vector[3],
                    exp_vector[4], exp_vector[5], exp_vector[6], exp_vector[7],
                    meta, last);
                if (code == 34) begin
                    if (core_count >= MAX_CORE_VECTORS) tb_fail("too many core vectors");
                    core_vec_mode[core_count] = mode;
                    core_vec_clear[core_count] = clear;
                    core_vec_first[core_count] = first;
                    core_vec_last_tile[core_count] = last_tile;
                    core_vec_mask[core_count] = mask;
                    core_vec_scalar[core_count] = scalar;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        core_vec_a[core_count][lane*16 +: 16] = a16[lane];
                        core_vec_b[core_count][lane*16 +: 16] = b16[lane];
                        core_vec_exp_vector[core_count][lane*32 +: 32] = exp_vector[lane];
                    end
                    core_vec_expect[core_count] = expect_flag;
                    core_vec_exp_scalar[core_count] = expected32;
                    core_vec_meta[core_count] = meta;
                    core_vec_last[core_count] = last;
                    core_count++;
                end
            end
            $fclose(fd);
            if (add_count == 0 || red_count == 0 || core_count == 0) tb_fail("empty vector set");
            $display("HW_H9_NUMERIC_VECTOR_COUNTS add=%0d reduction=%0d core=%0d", add_count, red_count, core_count);
        end
    endtask

    task automatic run_add_vectors;
        int sent;
        int received;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        begin
            sent = 0; received = 0; cycle = 0; drive_valid = 1'b0; drive_index = 0;
            while (received < add_count) begin
                @(negedge clk);
                if (!drive_valid && sent < add_count && ((cycle % 5) != 2)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                add_in_valid = drive_valid;
                add_in_a = add_vec_a[drive_index];
                add_in_b = add_vec_b[drive_index];
                add_in_meta = add_vec_meta[drive_index];
                add_in_last = add_vec_last[drive_index];
                add_out_ready = ((cycle % 7) != 3);
                #1;
                pre_in_fire = add_in_valid && add_in_ready;
                pre_out_fire = add_out_valid && add_out_ready;
                if (pre_out_fire) begin
                    if (add_expected_q.size() == 0) tb_fail("unexpected add output");
                    if (add_out_result !== add_expected_q.pop_front()) begin
                        $display("CHECK_FAIL hw_h9_known_add got=%08h expected=%08h", add_out_result, add_vec_expected[received]);
                        $fatal(1);
                    end
                    if (add_out_invalid) tb_fail("add invalid");
                    if (add_out_meta !== add_meta_q.pop_front()) tb_fail("add meta mismatch");
                    if (add_out_last !== add_last_q.pop_front()) tb_fail("add last mismatch");
                    received++;
                end
                @(posedge clk); #1;
                if (pre_in_fire) begin
                    add_expected_q.push_back(add_vec_expected[drive_index]);
                    add_meta_q.push_back(add_vec_meta[drive_index]);
                    add_last_q.push_back(add_vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 10000) tb_fail("add timeout");
            end
            add_in_valid = 1'b0;
            add_out_ready = 1'b0;
            $display("HW_H9_NUMERIC_KNOWN_ADD_PASS count=%0d", add_count);
        end
    endtask

    task automatic run_reduction_reset_probe;
        begin
            @(negedge clk);
            red_in_valid = 1'b1;
            red_in_values = red_vec_values[0];
            red_in_mask = red_vec_mask[0];
            red_in_meta = red_vec_meta[0];
            red_in_last = red_vec_last[0];
            red_out_ready = 1'b1;
            #1;
            if (!red_in_ready) tb_fail("reset probe reduction not ready");
            @(posedge clk);
            @(negedge clk);
            red_in_valid = 1'b0;
            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
            if (red_out_valid || red_busy) tb_fail("reduction reset did not clear in-flight operation");
        end
    endtask

    task automatic run_reduction_vectors;
        int sent;
        int received;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        begin
            sent = 0; received = 0; cycle = 0; drive_valid = 1'b0; drive_index = 0;
            while (received < red_count) begin
                @(negedge clk);
                if (!drive_valid && sent < red_count && ((cycle % 6) != 1) && ((cycle % 11) != 5)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                red_in_valid = drive_valid;
                red_in_values = red_vec_values[drive_index];
                red_in_mask = red_vec_mask[drive_index];
                red_in_meta = red_vec_meta[drive_index];
                red_in_last = red_vec_last[drive_index];
                red_out_ready = ((cycle % 8) != 4) && ((cycle % 13) != 6);
                #1;
                pre_in_fire = red_in_valid && red_in_ready;
                pre_out_fire = red_out_valid && red_out_ready;
                if (pre_out_fire) begin
                    if (red_expected_q.size() == 0) tb_fail("unexpected reduction output");
                    if (red_out_sum !== red_expected_q.pop_front()) begin
                        $display("CHECK_FAIL hw_h9_reduction got=%08h expected=%08h", red_out_sum, red_vec_expected[received]);
                        $fatal(1);
                    end
                    if (red_out_invalid) tb_fail("reduction invalid");
                    if (red_out_meta !== red_meta_q.pop_front()) tb_fail("reduction meta mismatch");
                    if (red_out_last !== red_last_q.pop_front()) tb_fail("reduction last mismatch");
                    received++;
                end
                @(posedge clk); #1;
                if (pre_in_fire) begin
                    red_expected_q.push_back(red_vec_expected[drive_index]);
                    red_meta_q.push_back(red_vec_meta[drive_index]);
                    red_last_q.push_back(red_vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 100000) tb_fail("reduction timeout");
            end
            red_in_valid = 1'b0;
            red_out_ready = 1'b0;
            $display("HW_H9_NUMERIC_REDUCTION_PASS count=%0d", red_count);
        end
    endtask

    task automatic run_core_vectors;
        int sent;
        int received;
        int expected_outputs;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        begin
            sent = 0; received = 0; expected_outputs = 0; cycle = 0; drive_valid = 1'b0; drive_index = 0;
            for (int idx = 0; idx < core_count; idx++) begin
                expected_outputs += core_vec_expect[idx];
            end
            while (received < expected_outputs) begin
                @(negedge clk);
                if (!drive_valid && sent < core_count && ((cycle % 5) != 1) && ((cycle % 17) != 9)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                core_in_valid = drive_valid;
                core_in_mode = core_vec_mode[drive_index];
                core_in_clear = core_vec_clear[drive_index];
                core_in_tile_first = core_vec_first[drive_index];
                core_in_tile_last = core_vec_last_tile[drive_index];
                core_in_use_explicit_mask = 1'b1;
                core_in_active_lanes = '0;
                core_in_lane_mask = core_vec_mask[drive_index];
                core_in_scalar = core_vec_scalar[drive_index];
                core_in_a = core_vec_a[drive_index];
                core_in_b = core_vec_b[drive_index];
                core_in_meta = core_vec_meta[drive_index];
                core_in_last = core_vec_last[drive_index];
                core_out_ready = ((cycle % 6) != 2) && ((cycle % 19) != 11);
                #1;
                pre_in_fire = core_in_valid && core_in_ready;
                pre_out_fire = core_out_valid && core_out_ready;
                if (pre_out_fire) begin
                    if (core_expected_q.size() == 0) tb_fail("unexpected core output");
                    if (core_out_scalar !== core_expected_q.pop_front()) begin
                        $display("CHECK_FAIL hw_h9_core got=%08h expected=%08h", core_out_scalar, core_vec_exp_scalar[received]);
                        $fatal(1);
                    end
                    if (core_out_invalid) tb_fail("core invalid");
                    if (core_out_meta !== core_meta_q.pop_front()) tb_fail("core meta mismatch");
                    if (core_out_last !== core_last_q.pop_front()) tb_fail("core last mismatch");
                    received++;
                end
                @(posedge clk); #1;
                if (pre_in_fire) begin
                    if (core_vec_expect[drive_index]) begin
                        core_expected_q.push_back(core_vec_exp_scalar[drive_index]);
                        core_meta_q.push_back(core_vec_meta[drive_index]);
                        core_last_q.push_back(core_vec_last[drive_index]);
                    end
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 400000) tb_fail("core timeout");
            end
            if (sent != core_count) tb_fail("not all core vectors sent");
            core_in_valid = 1'b0;
            core_out_ready = 1'b0;
            $display("HW_H9_NUMERIC_PE_CORE_PASS vectors=%0d outputs=%0d", core_count, expected_outputs);
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_add_vectors();
        run_reduction_reset_probe();
        run_reduction_vectors();
        run_core_vectors();
        $display("HW_H9_NUMERIC_REPAIR_PASS known_operand=3c837d4a random_reductions=%0d core_vectors=%0d",
                 red_count - 6, core_count);
        $finish;
    end
endmodule

`default_nettype wire
