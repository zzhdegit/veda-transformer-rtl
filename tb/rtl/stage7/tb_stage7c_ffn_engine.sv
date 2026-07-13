`timescale 1ns/1ps
`default_nettype none

module tb_stage7c_ffn_engine;
`ifndef STAGE7_D_MODEL
    localparam int D_MODEL = 8;
`else
    localparam int D_MODEL = `STAGE7_D_MODEL;
`endif
    localparam int D_FFN = 4 * D_MODEL;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam int FFN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN);

    logic clk;
    logic rst_n;

    logic clear;
    logic weight_valid;
    logic weight_ready;
    logic weight_kind;
    logic [FFN_W-1:0] weight_output_index;
    logic [FFN_W-1:0] weight_input_index;
    logic [15:0] weight_data_fp16;
    logic weight_commit;
    logic input_valid;
    logic input_ready;
    logic [MODEL_W-1:0] input_dim;
    logic [15:0] input_data_fp16;
    logic input_last;
    logic [META_W-1:0] input_meta;
    logic input_commit;
    logic start_valid;
    logic start_ready;
    logic [META_W-1:0] start_meta;
    logic output_valid;
    logic output_ready;
    logic [MODEL_W-1:0] output_dim;
    logic [31:0] output_data_fp32;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [COUNTER_W-1:0] perf_ffn1_cycles;
    logic [COUNTER_W-1:0] perf_relu_cycles;
    logic [COUNTER_W-1:0] perf_activation_quantization_cycles;
    logic [COUNTER_W-1:0] perf_ffn2_cycles;
    logic [COUNTER_W-1:0] perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;

    logic [15:0] input_vec [0:D_MODEL-1];
    logic [15:0] w1_mem [0:D_FFN-1][0:D_MODEL-1];
    logic [15:0] w2_mem [0:D_MODEL-1][0:D_FFN-1];
    logic [31:0] expected_output [0:D_MODEL-1];

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    ffn_engine #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_ffn (
        .clk                                (clk),
        .rst_n                              (rst_n),
        .clear                              (clear),
        .weight_valid                       (weight_valid),
        .weight_ready                       (weight_ready),
        .weight_kind                        (weight_kind),
        .weight_output_index                (weight_output_index),
        .weight_input_index                 (weight_input_index),
        .weight_data_fp16                   (weight_data_fp16),
        .weight_commit                      (weight_commit),
        .input_valid                        (input_valid),
        .input_ready                        (input_ready),
        .input_dim                          (input_dim),
        .input_data_fp16                    (input_data_fp16),
        .input_last                         (input_last),
        .input_meta                         (input_meta),
        .input_commit                       (input_commit),
        .start_valid                        (start_valid),
        .start_ready                        (start_ready),
        .start_meta                         (start_meta),
        .output_valid                       (output_valid),
        .output_ready                       (output_ready),
        .output_dim                         (output_dim),
        .output_data_fp32                   (output_data_fp32),
        .output_status                      (output_status),
        .output_invalid                     (output_invalid),
        .output_meta                        (output_meta),
        .output_last                        (output_last),
        .done_valid                         (done_valid),
        .done_ready                         (done_ready),
        .done_status                        (done_status),
        .done_invalid                       (done_invalid),
        .done_meta                          (done_meta),
        .perf_ffn1_cycles                   (perf_ffn1_cycles),
        .perf_relu_cycles                   (perf_relu_cycles),
        .perf_activation_quantization_cycles(perf_activation_quantization_cycles),
        .perf_ffn2_cycles                   (perf_ffn2_cycles),
        .perf_pe_stall_cycles               (perf_pe_stall_cycles),
        .perf_output_stall_cycles           (perf_output_stall_cycles)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE7C_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        string tag;
        int d_model_value;
        int d_ffn_value;
        int kind;
        int row;
        int col;
        int dim;
        logic [15:0] h;
        logic [31:0] f;
        begin
            if (!$value$plusargs("STAGE7C_VECTOR_FILE=%s", path)) begin
                tb_fail("missing +STAGE7C_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) begin
                tb_fail("could not open vector file");
            end
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code == 1) begin
                    if (tag == "D") begin
                        code = $fscanf(fd, "%d %d\n", d_model_value, d_ffn_value);
                        if (d_model_value != D_MODEL || d_ffn_value != D_FFN) tb_fail("dimension mismatch");
                    end else if (tag == "I") begin
                        code = $fscanf(fd, "%h %h\n", dim, h);
                        input_vec[dim] = h;
                    end else if (tag == "W") begin
                        code = $fscanf(fd, "%d %h %h %h\n", kind, row, col, h);
                        if (kind == 0) begin
                            w1_mem[row][col] = h;
                        end else begin
                            w2_mem[row][col] = h;
                        end
                    end else if (tag == "O") begin
                        code = $fscanf(fd, "%h %h\n", dim, f);
                        expected_output[dim] = f;
                    end else begin
                        tb_fail("unknown vector tag");
                    end
                end
            end
            $fclose(fd);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            clear = 1'b0;
            weight_valid = 1'b0;
            weight_kind = 1'b0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = '0;
            weight_commit = 1'b0;
            input_valid = 1'b0;
            input_dim = '0;
            input_data_fp16 = '0;
            input_last = 1'b0;
            input_meta = '0;
            input_commit = 1'b0;
            start_valid = 1'b0;
            start_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (6) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic drive_weight(
        input logic kind,
        input int row,
        input int col,
        input logic [15:0] data,
        input logic commit
    );
        begin
            @(negedge clk);
            weight_valid = 1'b1;
            weight_kind = kind;
            weight_output_index = FFN_W'(row);
            weight_input_index = FFN_W'(col);
            weight_data_fp16 = data;
            weight_commit = commit;
            #1;
            if (!weight_ready) tb_fail("weight load not ready");
            @(posedge clk);
            @(negedge clk);
            weight_valid = 1'b0;
            weight_commit = 1'b0;
        end
    endtask

    task automatic load_weights;
        begin
            for (int row = 0; row < D_FFN; row++) begin
                for (int col = 0; col < D_MODEL; col++) begin
                    drive_weight(1'b0, row, col, w1_mem[row][col],
                                 (row == D_FFN - 1) && (col == D_MODEL - 1));
                end
            end
            for (int row = 0; row < D_MODEL; row++) begin
                for (int col = 0; col < D_FFN; col++) begin
                    drive_weight(1'b1, row, col, w2_mem[row][col],
                                 (row == D_MODEL - 1) && (col == D_FFN - 1));
                end
            end
        end
    endtask

    task automatic load_input;
        begin
            for (int dim = 0; dim < D_MODEL; dim++) begin
                @(negedge clk);
                input_valid = 1'b1;
                input_dim = MODEL_W'(dim);
                input_data_fp16 = input_vec[dim];
                input_last = dim == D_MODEL - 1;
                input_meta = 16'h7C01;
                input_commit = dim == D_MODEL - 1;
                #1;
                if (!input_ready) tb_fail("input load not ready");
                @(posedge clk);
            end
            @(negedge clk);
            input_valid = 1'b0;
            input_last = 1'b0;
            input_commit = 1'b0;
        end
    endtask

    task automatic run_ffn;
        int received;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        begin
            @(negedge clk);
            start_valid = 1'b1;
            start_meta = 16'h7C01;
            #1;
            if (!start_ready) tb_fail("start not ready");
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            received = 0;
            cycle = 0;
            while (received < D_MODEL || !done_valid) begin
                @(negedge clk);
                output_ready = (cycle % 5) != 1;
                done_ready = 1'b1;
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                if (pre_out_fire) begin
                    if (output_dim !== MODEL_W'(received)) tb_fail("output dimension mismatch");
                    if (output_data_fp32 !== expected_output[received]) begin
                        $display("CHECK_FAIL ffn dim=%0d got=%08h expected=%08h",
                                 received, output_data_fp32, expected_output[received]);
                        $fatal(1);
                    end
                    if (output_invalid) tb_fail("output invalid");
                    if (output_meta !== 16'h7C01) tb_fail("metadata mismatch");
                    if (output_last !== (received == D_MODEL - 1)) tb_fail("last mismatch");
                    received++;
                end
                if (pre_done_fire) begin
                    if (received != D_MODEL) tb_fail("done before all outputs");
                    if (done_invalid) tb_fail("done invalid");
                end
                @(posedge clk);
                cycle++;
                if (cycle > 80000) tb_fail("ffn timeout");
            end
            @(negedge clk);
            output_ready = 1'b0;
            done_ready = 1'b0;
        end
    endtask

    initial begin
        load_vectors();
        apply_reset();
        load_weights();
        load_input();
        run_ffn();
        $display("STAGE7C_FFN_ENGINE_PASS d_model=%0d d_ffn=%0d", D_MODEL, D_FFN);
        $finish;
    end
endmodule

`default_nettype wire
