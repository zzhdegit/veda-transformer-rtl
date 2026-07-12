`timescale 1ns/1ps
`default_nettype none

module tb_fp32_exp_recip_wrappers;
    localparam int META_W = 16;
    localparam int MAX_VECTORS = 64;

    logic clk;
    logic rst_n;

    logic exp_in_valid;
    logic exp_in_ready;
    logic [31:0] exp_in_a;
    logic [META_W-1:0] exp_in_meta;
    logic exp_in_last;
    logic exp_out_valid;
    logic exp_out_ready;
    logic [31:0] exp_out_result;
    logic [7:0] exp_out_status;
    logic exp_out_invalid;
    logic [META_W-1:0] exp_out_meta;
    logic exp_out_last;

    logic recip_in_valid;
    logic recip_in_ready;
    logic [31:0] recip_in_a;
    logic [META_W-1:0] recip_in_meta;
    logic recip_in_last;
    logic recip_out_valid;
    logic recip_out_ready;
    logic [31:0] recip_out_result;
    logic [7:0] recip_out_status;
    logic recip_out_invalid;
    logic [META_W-1:0] recip_out_meta;
    logic recip_out_last;

    logic [31:0] exp_vec_in [0:MAX_VECTORS-1];
    logic [31:0] exp_vec_out [0:MAX_VECTORS-1];
    logic [META_W-1:0] exp_vec_meta [0:MAX_VECTORS-1];
    logic exp_vec_last [0:MAX_VECTORS-1];
    int exp_count;

    logic [31:0] recip_vec_in [0:MAX_VECTORS-1];
    logic [31:0] recip_vec_out [0:MAX_VECTORS-1];
    logic [META_W-1:0] recip_vec_meta [0:MAX_VECTORS-1];
    logic recip_vec_last [0:MAX_VECTORS-1];
    int recip_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    fp32_exp_wrapper #(
        .META_W(META_W)
    ) u_exp (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (exp_in_valid),
        .in_ready    (exp_in_ready),
        .in_a        (exp_in_a),
        .in_meta     (exp_in_meta),
        .in_last     (exp_in_last),
        .out_valid   (exp_out_valid),
        .out_ready   (exp_out_ready),
        .out_result  (exp_out_result),
        .out_status  (exp_out_status),
        .out_invalid (exp_out_invalid),
        .out_meta    (exp_out_meta),
        .out_last    (exp_out_last)
    );

    fp32_recip_wrapper #(
        .META_W(META_W)
    ) u_recip (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (recip_in_valid),
        .in_ready    (recip_in_ready),
        .in_a        (recip_in_a),
        .in_meta     (recip_in_meta),
        .in_last     (recip_in_last),
        .out_valid   (recip_out_valid),
        .out_ready   (recip_out_ready),
        .out_result  (recip_out_result),
        .out_status  (recip_out_status),
        .out_invalid (recip_out_invalid),
        .out_meta    (recip_out_meta),
        .out_last    (recip_out_last)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE3_SFU_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        logic [31:0] a;
        logic [31:0] z;
        logic [META_W-1:0] meta;
        logic last;
        begin
            if (!$value$plusargs("EXP_VECTOR_FILE=%s", path)) tb_fail("missing +EXP_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open exp vector file");
            exp_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %b\n", a, z, meta, last);
                if (code == 4) begin
                    exp_vec_in[exp_count] = a;
                    exp_vec_out[exp_count] = z;
                    exp_vec_meta[exp_count] = meta;
                    exp_vec_last[exp_count] = last;
                    exp_count++;
                end
            end
            $fclose(fd);
            if (!$value$plusargs("RECIP_VECTOR_FILE=%s", path)) tb_fail("missing +RECIP_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open recip vector file");
            recip_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%h %h %h %b\n", a, z, meta, last);
                if (code == 4) begin
                    recip_vec_in[recip_count] = a;
                    recip_vec_out[recip_count] = z;
                    recip_vec_meta[recip_count] = meta;
                    recip_vec_last[recip_count] = last;
                    recip_count++;
                end
            end
            $fclose(fd);
            $display("STAGE3_EXP_VECTORS count=%0d", exp_count);
            $display("STAGE3_RECIP_VECTORS count=%0d", recip_count);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            exp_in_valid = 1'b0;
            exp_in_a = '0;
            exp_in_meta = '0;
            exp_in_last = 1'b0;
            exp_out_ready = 1'b0;
            recip_in_valid = 1'b0;
            recip_in_a = '0;
            recip_in_meta = '0;
            recip_in_last = 1'b0;
            recip_out_ready = 1'b0;
            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic run_exp;
        int sent;
        int received;
        int cycle;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [31:0] pre_result;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            while (received < exp_count) begin
                @(negedge clk);
                if (!drive_valid && sent < exp_count && ((cycle % 3) != 1)) drive_valid = 1'b1;
                exp_in_valid = drive_valid;
                exp_in_a = exp_vec_in[sent];
                exp_in_meta = exp_vec_meta[sent];
                exp_in_last = exp_vec_last[sent];
                exp_out_ready = ((cycle % 5) != 2);
                #1;
                pre_in_fire = exp_in_valid && exp_in_ready;
                pre_out_fire = exp_out_valid && exp_out_ready;
                pre_result = exp_out_result;
                pre_meta = exp_out_meta;
                pre_last = exp_out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (pre_result !== exp_vec_out[received]) begin
                        $display("CHECK_FAIL exp got=%08h expected=%08h index=%0d", pre_result, exp_vec_out[received], received);
                        $fatal(1);
                    end
                    if (exp_out_invalid) tb_fail("unexpected exp invalid");
                    if (pre_meta !== exp_vec_meta[received]) tb_fail("exp metadata mismatch");
                    if (pre_last !== exp_vec_last[received]) tb_fail("exp last mismatch");
                    received++;
                end
                if (pre_in_fire) begin
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 5000) tb_fail("exp timeout");
            end
            exp_in_valid = 1'b0;
            exp_out_ready = 1'b0;
        end
    endtask

    task automatic run_recip;
        int sent;
        int received;
        int cycle;
        logic drive_valid;
        logic pre_in_fire;
        logic pre_out_fire;
        logic [31:0] pre_result;
        logic [META_W-1:0] pre_meta;
        logic pre_last;
        begin
            sent = 0;
            received = 0;
            cycle = 0;
            drive_valid = 1'b0;
            while (received < recip_count) begin
                @(negedge clk);
                if (!drive_valid && sent < recip_count && ((cycle % 4) != 1)) drive_valid = 1'b1;
                recip_in_valid = drive_valid;
                recip_in_a = recip_vec_in[sent];
                recip_in_meta = recip_vec_meta[sent];
                recip_in_last = recip_vec_last[sent];
                recip_out_ready = ((cycle % 6) != 3);
                #1;
                pre_in_fire = recip_in_valid && recip_in_ready;
                pre_out_fire = recip_out_valid && recip_out_ready;
                pre_result = recip_out_result;
                pre_meta = recip_out_meta;
                pre_last = recip_out_last;
                @(posedge clk); #1;
                if (pre_out_fire) begin
                    if (pre_result !== recip_vec_out[received]) begin
                        $display("CHECK_FAIL recip got=%08h expected=%08h index=%0d", pre_result, recip_vec_out[received], received);
                        $fatal(1);
                    end
                    if (recip_out_invalid) tb_fail("unexpected recip invalid");
                    if (pre_meta !== recip_vec_meta[received]) tb_fail("recip metadata mismatch");
                    if (pre_last !== recip_vec_last[received]) tb_fail("recip last mismatch");
                    received++;
                end
                if (pre_in_fire) begin
                    sent++;
                    drive_valid = 1'b0;
                end
                cycle++;
                if (cycle > 5000) tb_fail("recip timeout");
            end
            recip_in_valid = 1'b0;
            recip_out_ready = 1'b0;
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        run_exp();
        run_recip();
        $display("STAGE3_FP32_EXP_RECIP_WRAPPERS_PASS");
        $finish;
    end
endmodule

`default_nettype wire

