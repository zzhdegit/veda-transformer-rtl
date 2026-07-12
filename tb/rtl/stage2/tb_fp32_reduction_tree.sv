`timescale 1ns/1ps
`default_nettype none

module tb_fp32_reduction_tree;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 32;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic [PE_NUM*32-1:0] in_values;
    logic [PE_NUM-1:0] in_lane_mask;
    logic [META_W-1:0] in_meta;
    logic in_last;
    logic out_valid;
    logic out_ready;
    logic [31:0] out_sum;
    logic [7:0] out_status;
    logic out_invalid;
    logic [META_W-1:0] out_meta;
    logic out_last;
    logic busy;

    logic [PE_NUM-1:0] vec_mask [0:MAX_VECTORS-1];
    logic [PE_NUM*32-1:0] vec_values [0:MAX_VECTORS-1];
    logic [31:0] vec_expected [0:MAX_VECTORS-1];
    logic [META_W-1:0] vec_meta [0:MAX_VECTORS-1];
    logic vec_last [0:MAX_VECTORS-1];
    int vector_count;

    logic [31:0] exp_sum_q[$];
    logic [META_W-1:0] exp_meta_q[$];
    logic exp_last_q[$];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_reduction_tree #(
        .PE_NUM(PE_NUM),
        .META_W(META_W)
    ) u_dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (in_valid),
        .in_ready     (in_ready),
        .in_values    (in_values),
        .in_lane_mask (in_lane_mask),
        .in_meta      (in_meta),
        .in_last      (in_last),
        .out_valid    (out_valid),
        .out_ready    (out_ready),
        .out_sum      (out_sum),
        .out_status   (out_status),
        .out_invalid  (out_invalid),
        .out_meta     (out_meta),
        .out_last     (out_last),
        .busy         (busy)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE2_REDUCTION_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic [PE_NUM-1:0] mask;
        logic [31:0] value [0:PE_NUM-1];
        logic [31:0] expected;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("REDUCTION_VECTOR_FILE=%s", path)) tb_fail("missing +REDUCTION_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open reduction vector file");
            vector_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %b\n",
                               mask, value[0], value[1], value[2], value[3],
                               value[4], value[5], value[6], value[7], expected, meta, last);
                if (code == 12) begin
                    vec_mask[vector_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        vec_values[vector_count][lane*32 +: 32] = value[lane];
                    end
                    vec_expected[vector_count] = expected;
                    vec_meta[vector_count] = meta;
                    vec_last[vector_count] = last;
                    vector_count++;
                end
            end
            $fclose(fd);
            if (vector_count == 0) tb_fail("no reduction vectors loaded");
            $display("STAGE2_REDUCTION_VECTORS count=%0d", vector_count);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            in_valid = 1'b0;
            in_values = '0;
            in_lane_mask = '0;
            in_meta = '0;
            in_last = 1'b0;
            out_ready = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            if (out_valid) tb_fail("out_valid not clear after reset");
        end
    endtask

    task automatic run_vectors;
        int sent;
        int received;
        int cycle;
        int drive_index;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [31:0] pre_sum;
        logic [7:0] pre_status;
        logic pre_invalid;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        logic [31:0] expected;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            drive_index = 0;
            while (received < vector_count) begin
                @(negedge clk);
                if (!drive_valid && (sent < vector_count) && ((cycle % 6) != 4)) begin
                    drive_valid = 1'b1;
                    drive_index = sent;
                end
                in_valid = drive_valid;
                in_values = vec_values[drive_index];
                in_lane_mask = vec_mask[drive_index];
                in_meta = vec_meta[drive_index];
                in_last = vec_last[drive_index];
                out_ready = ((cycle % 7) != 3);
                #1;
                pre_in_fire = in_valid && in_ready;
                pre_out_fire = out_valid && out_ready;
                pre_sum = out_sum;
                pre_status = out_status;
                pre_invalid = out_invalid;
                pre_meta = out_meta;
                pre_last = out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (exp_sum_q.size() == 0) tb_fail("unexpected reduction output");
                    expected = exp_sum_q.pop_front();
                    if (pre_sum !== expected) begin
                        $display("CHECK_FAIL reduction got=%08h expected=%08h", pre_sum, expected);
                        $fatal(1);
                    end
                    if (pre_invalid) tb_fail("unexpected reduction invalid");
                    if (^pre_status === 1'bx) tb_fail("unknown reduction status");
                    if (pre_meta !== exp_meta_q.pop_front()) tb_fail("metadata mismatch");
                    if (pre_last !== exp_last_q.pop_front()) tb_fail("last mismatch");
                    received++;
                end
                if (pre_in_fire) begin
                    exp_sum_q.push_back(vec_expected[drive_index]);
                    exp_meta_q.push_back(vec_meta[drive_index]);
                    exp_last_q.push_back(vec_last[drive_index]);
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 20000) tb_fail("reduction timeout");
            end
            if (exp_sum_q.size() != 0) tb_fail("reduction expected queue not empty");
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_vectors();
        $display("STAGE2_REDUCTION_TREE_PASS");
        $finish;
    end
endmodule

`default_nettype wire
