`timescale 1ns/1ps
`default_nettype none

module tb_projection_integrated_mha_stage6e;
`ifndef STAGE6_N_HEAD
    localparam int N_HEAD = 2;
`else
    localparam int N_HEAD = `STAGE6_N_HEAD;
`endif
`ifndef STAGE6_D_HEAD
    localparam int D_HEAD = 8;
`else
    localparam int D_HEAD = `STAGE6_D_HEAD;
`endif
    localparam int PE_NUM = 8;
    localparam int MAX_SEQ_LEN = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int D_MODEL = N_HEAD * D_HEAD;
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int MAX_OUTPUTS = (D_MODEL + PE_NUM - 1) / PE_NUM;
    localparam logic [3:0] DUT_ST_LOAD_HIDDEN       = 4'd0;
    localparam logic [3:0] DUT_ST_RUN_Q             = 4'd2;
    localparam logic [3:0] DUT_ST_RUN_K             = 4'd4;
    localparam logic [3:0] DUT_ST_RUN_V             = 4'd6;
    localparam logic [3:0] DUT_ST_QKV_STREAM        = 4'd7;
    localparam logic [3:0] DUT_ST_ATTENTION         = 4'd8;
    localparam logic [3:0] DUT_ST_OUTPUT_PROJECTION = 4'd10;

    logic clk;
    logic rst_n;
    logic weight_valid;
    logic weight_ready;
    logic [1:0] weight_kind;
    logic [MODEL_W-1:0] weight_output_index;
    logic [MODEL_W-1:0] weight_input_index;
    logic [15:0] weight_data_fp16;
    logic weight_last;
    logic weight_commit;
    logic token_valid;
    logic token_ready;
    logic [MODEL_W-1:0] token_dim;
    logic [15:0] token_hidden_fp16;
    logic token_last_dim;
    logic [META_W-1:0] token_meta;
    logic output_valid;
    logic output_ready;
    logic [MODEL_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;
    logic [COUNTER_W-1:0] perf_generation_steps;
    logic [COUNTER_W-1:0] perf_total_cycles;
    logic [COUNTER_W-1:0] perf_q_projection_cycles;
    logic [COUNTER_W-1:0] perf_k_projection_cycles;
    logic [COUNTER_W-1:0] perf_v_projection_cycles;
    logic [COUNTER_W-1:0] perf_qkv_quantization_cycles;
    logic [COUNTER_W-1:0] perf_attention_cycles;
    logic [COUNTER_W-1:0] perf_concat_quantization_cycles;
    logic [COUNTER_W-1:0] perf_output_projection_cycles;
    logic [COUNTER_W-1:0] perf_projection_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_attention_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    logic [15:0] weights [0:3][0:D_MODEL-1][0:D_MODEL-1];
    logic [15:0] current_hidden [0:D_MODEL-1];
    logic [MODEL_W-1:0] exp_base [0:MAX_OUTPUTS-1];
    logic [PE_NUM-1:0] exp_mask [0:MAX_OUTPUTS-1];
    logic [PE_NUM*32-1:0] exp_vector [0:MAX_OUTPUTS-1];
    logic exp_last [0:MAX_OUTPUTS-1];
    logic [15:0] reset_hidden [0:D_MODEL-1];
    logic [MODEL_W-1:0] reset_exp_base [0:MAX_OUTPUTS-1];
    logic [PE_NUM-1:0] reset_exp_mask [0:MAX_OUTPUTS-1];
    logic [PE_NUM*32-1:0] reset_exp_vector [0:MAX_OUTPUTS-1];
    logic reset_exp_last [0:MAX_OUTPUTS-1];
    string reset_baseline_name;
    logic [META_W-1:0] reset_baseline_meta;
    int reset_baseline_seq_before;
    int reset_baseline_seq_after;
    bit reset_baseline_expect_invalid;
    logic [7:0] reset_baseline_expect_status;
    int reset_exp_count;
    bit reset_baseline_valid;
    string current_name;
    logic [META_W-1:0] current_meta;
    int current_seq_before;
    int current_seq_after;
    bit current_expect_invalid;
    logic [7:0] current_expect_status;
    int exp_count;
    int step_run_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    projection_integrated_mha #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(1'b0)
    ) u_dut (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .weight_valid                     (weight_valid),
        .weight_ready                     (weight_ready),
        .weight_kind                      (weight_kind),
        .weight_output_index              (weight_output_index),
        .weight_input_index               (weight_input_index),
        .weight_data_fp16                 (weight_data_fp16),
        .weight_last                      (weight_last),
        .weight_commit                    (weight_commit),
        .token_valid                      (token_valid),
        .token_ready                      (token_ready),
        .token_dim                        (token_dim),
        .token_hidden_fp16                (token_hidden_fp16),
        .token_last_dim                   (token_last_dim),
        .token_meta                       (token_meta),
        .output_valid                     (output_valid),
        .output_ready                     (output_ready),
        .output_base_dim                  (output_base_dim),
        .output_vector_fp32               (output_vector_fp32),
        .output_lane_mask                 (output_lane_mask),
        .output_status                    (output_status),
        .output_invalid                   (output_invalid),
        .output_meta                      (output_meta),
        .output_last                      (output_last),
        .done_valid                       (done_valid),
        .done_ready                       (done_ready),
        .done_status                      (done_status),
        .done_invalid                     (done_invalid),
        .done_meta                        (done_meta),
        .done_valid_seq_len               (done_valid_seq_len),
        .current_valid_seq_len            (current_valid_seq_len),
        .perf_generation_steps            (perf_generation_steps),
        .perf_total_cycles                (perf_total_cycles),
        .perf_q_projection_cycles         (perf_q_projection_cycles),
        .perf_k_projection_cycles         (perf_k_projection_cycles),
        .perf_v_projection_cycles         (perf_v_projection_cycles),
        .perf_qkv_quantization_cycles     (perf_qkv_quantization_cycles),
        .perf_attention_cycles            (perf_attention_cycles),
        .perf_concat_quantization_cycles  (perf_concat_quantization_cycles),
        .perf_output_projection_cycles    (perf_output_projection_cycles),
        .perf_projection_pe_stall_cycles  (perf_projection_pe_stall_cycles),
        .perf_attention_pe_stall_cycles   (perf_attention_pe_stall_cycles),
        .perf_sfu_stall_cycles            (perf_sfu_stall_cycles),
        .perf_weight_stall_cycles         (perf_weight_stall_cycles),
        .perf_buffer_stall_cycles         (perf_buffer_stall_cycles),
        .perf_output_stall_cycles         (perf_output_stall_cycles),
        .perf_peak_valid_seq_len          (perf_peak_valid_seq_len)
    );

    task automatic tb_fail(input string message);
        begin
            $display("STAGE6E_INTEGRATED_MHA_FAIL N_HEAD=%0d D_HEAD=%0d step=%s: %s",
                     N_HEAD, D_HEAD, current_name, message);
            $display("DEBUG state=%0d proj_state=%0d wo_state=%0d proj_start=%0b/%0b proj_done=%0b/%0b qkv_all=%0b stream=%0d stage5_done=%0b/%0b seen=%0b inv=%0b concat_complete=%0b concat_read=%0b idx=%0d concat_invalid=%0b concat_error=%0b wo_start=%0b/%0b wo_in=%0b/%0b dim=%0d wo_proj_start=%0b/%0b wo_done=%0b/%0b final_done=%0b/%0b token_ready=%0b current_seq=%0d",
                     u_dut.state_q,
                     u_dut.u_shared_projection_controller.state_q,
                     u_dut.u_output_projection_controller.state_q,
                     u_dut.proj_start_valid, u_dut.proj_start_ready,
                     u_dut.proj_done_valid, u_dut.proj_done_ready,
                     u_dut.qkv_all_complete, u_dut.stream_index_q,
                     u_dut.stage5_done_valid, u_dut.stage5_done_ready,
                     u_dut.stage5_done_seen_q, u_dut.stage5_done_invalid_q,
                     u_dut.concat_complete, u_dut.concat_read_valid, u_dut.concat_read_index,
                     u_dut.concat_invalid, u_dut.concat_buffer_error,
                     u_dut.wo_start_valid, u_dut.wo_start_ready,
                     u_dut.wo_proj_input_valid, u_dut.wo_proj_input_ready, u_dut.wo_proj_input_dim,
                     u_dut.wo_proj_start_valid, u_dut.wo_proj_start_ready,
                     u_dut.wo_done_valid, u_dut.wo_done_ready,
                     done_valid, done_ready, token_ready, current_valid_seq_len);
            $fatal(1);
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            weight_valid = 1'b0;
            weight_kind = 2'd0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = 16'd0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            token_valid = 1'b0;
            token_dim = '0;
            token_hidden_fp16 = 16'd0;
            token_last_dim = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic check_reset_cleared(input string scenario);
        begin
            #1;
            if (output_valid !== 1'b0) tb_fail({scenario, " output_valid not cleared by reset"});
            if (done_valid !== 1'b0) tb_fail({scenario, " done_valid not cleared by reset"});
            if (u_dut.state_q !== DUT_ST_LOAD_HIDDEN) tb_fail({scenario, " active transaction not cleared"});
            if (u_dut.expected_dim_q !== '0) tb_fail({scenario, " hidden dimension state not cleared"});
            if (u_dut.stage5_done_seen_q !== 1'b0) tb_fail({scenario, " stage5 done state not cleared"});
            if (u_dut.final_done_valid_q !== 1'b0) tb_fail({scenario, " final done state not cleared"});
            if (u_dut.concat_loaded_mask !== '0) tb_fail({scenario, " concat buffer valid state not cleared"});
            if (u_dut.q_complete || u_dut.k_complete || u_dut.v_complete || u_dut.qkv_all_complete) begin
                tb_fail({scenario, " qkv buffer valid state not cleared"});
            end
            if (current_valid_seq_len !== '0) tb_fail({scenario, " cache valid_seq_len not cleared"});
            if ($isunknown({output_valid, done_valid, token_ready, weight_ready, current_valid_seq_len})) begin
                tb_fail({scenario, " reset-visible output contains X"});
            end
        end
    endtask

    task automatic pulse_mid_transaction_reset(input string scenario);
        begin
            @(negedge clk);
            rst_n = 1'b0;
            weight_valid = 1'b0;
            weight_commit = 1'b0;
            token_valid = 1'b0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            check_reset_cleared(scenario);
            repeat (8) begin
                @(posedge clk);
                check_reset_cleared(scenario);
            end
            @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
            #1;
            if (output_valid !== 1'b0) tb_fail({scenario, " output_valid set after reset release"});
            if (done_valid !== 1'b0) tb_fail({scenario, " done_valid set after reset release"});
            if (current_valid_seq_len !== '0) tb_fail({scenario, " cache valid_seq_len nonzero after reset release"});
            if (token_ready !== 1'b1) tb_fail({scenario, " next token cannot start after reset release"});
        end
    endtask

    task automatic save_reset_baseline;
        begin
            for (int idx = 0; idx < D_MODEL; idx++) begin
                reset_hidden[idx] = current_hidden[idx];
            end
            reset_exp_count = exp_count;
            reset_baseline_name = current_name;
            reset_baseline_meta = current_meta;
            reset_baseline_seq_before = current_seq_before;
            reset_baseline_seq_after = current_seq_after;
            reset_baseline_expect_invalid = current_expect_invalid;
            reset_baseline_expect_status = current_expect_status;
            for (int out_idx = 0; out_idx < MAX_OUTPUTS; out_idx++) begin
                reset_exp_base[out_idx] = exp_base[out_idx];
                reset_exp_mask[out_idx] = exp_mask[out_idx];
                reset_exp_vector[out_idx] = exp_vector[out_idx];
                reset_exp_last[out_idx] = exp_last[out_idx];
            end
            reset_baseline_valid = 1'b1;
        end
    endtask

    task automatic restore_reset_baseline_step;
        begin
            current_name = reset_baseline_name;
            current_meta = reset_baseline_meta;
            current_seq_before = reset_baseline_seq_before;
            current_seq_after = reset_baseline_seq_after;
            current_expect_invalid = reset_baseline_expect_invalid;
            current_expect_status = reset_baseline_expect_status;
            exp_count = reset_exp_count;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                current_hidden[idx] = reset_hidden[idx];
            end
            for (int out_idx = 0; out_idx < MAX_OUTPUTS; out_idx++) begin
                exp_base[out_idx] = reset_exp_base[out_idx];
                exp_mask[out_idx] = reset_exp_mask[out_idx];
                exp_vector[out_idx] = reset_exp_vector[out_idx];
                exp_last[out_idx] = reset_exp_last[out_idx];
            end
        end
    endtask

    task automatic setup_reset_recovery_step(input string scenario, input int scenario_id);
        begin
            current_name = {scenario, "_recovery"};
            current_meta = META_W'(16'h7800 + scenario_id);
            current_seq_before = 0;
            current_seq_after = 1;
            current_expect_invalid = 1'b0;
            current_expect_status = 8'h00;
            exp_count = reset_exp_count;
            for (int idx = 0; idx < D_MODEL; idx++) begin
                current_hidden[idx] = reset_hidden[idx];
            end
            for (int out_idx = 0; out_idx < MAX_OUTPUTS; out_idx++) begin
                exp_base[out_idx] = reset_exp_base[out_idx];
                exp_mask[out_idx] = reset_exp_mask[out_idx];
                exp_vector[out_idx] = reset_exp_vector[out_idx];
                exp_last[out_idx] = reset_exp_last[out_idx];
            end
        end
    endtask

    task automatic start_reset_interrupt_token(input string scenario, input int scenario_id);
        begin
            current_name = {scenario, "_interrupt"};
            current_meta = META_W'(16'h7300 + scenario_id);
            for (int idx = 0; idx < D_MODEL; idx++) begin
                current_hidden[idx] = reset_hidden[idx];
            end
            output_ready = 1'b1;
            done_ready = 1'b1;
            drive_hidden();
        end
    endtask

    task automatic wait_for_state(input logic [3:0] target_state, input string scenario);
        int cycle;
        begin
            cycle = 0;
            while (u_dut.state_q !== target_state) begin
                @(negedge clk);
                output_ready = 1'b1;
                done_ready = 1'b1;
                #1;
                if (done_valid) tb_fail({scenario, " completed before reaching reset target"});
                cycle++;
                if (cycle > 6000000) tb_fail({scenario, " timed out waiting for reset target state"});
            end
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic wait_for_concat_activity(input string scenario);
        int cycle;
        begin
            cycle = 0;
            while (!(u_dut.concat_busy || u_dut.concat_write_valid)) begin
                @(negedge clk);
                output_ready = 1'b1;
                done_ready = 1'b1;
                #1;
                if (done_valid) tb_fail({scenario, " completed before concat reset target"});
                cycle++;
                if (cycle > 6000000) tb_fail({scenario, " timed out waiting for concat activity"});
            end
            @(posedge clk);
        end
    endtask

    task automatic wait_for_final_output_stall(input string scenario);
        int cycle;
        begin
            cycle = 0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            while (!output_valid) begin
                @(negedge clk);
                output_ready = 1'b0;
                done_ready = 1'b0;
                #1;
                if (done_valid) tb_fail({scenario, " done before stalled output"});
                cycle++;
                if (cycle > 6000000) tb_fail({scenario, " timed out waiting for stalled output"});
            end
            #1;
            if (output_valid !== 1'b1) tb_fail({scenario, " output did not remain valid while stalled"});
            if ($isunknown({output_base_dim, output_vector_fp32, output_lane_mask,
                            output_status, output_invalid, output_meta, output_last})) begin
                tb_fail({scenario, " stalled output contains X"});
            end
        end
    endtask

    task automatic wait_for_final_done_stall(input string scenario);
        int cycle;
        begin
            cycle = 0;
            output_ready = 1'b1;
            done_ready = 1'b0;
            while (!done_valid) begin
                @(negedge clk);
                output_ready = 1'b1;
                done_ready = 1'b0;
                #1;
                cycle++;
                if (cycle > 6000000) tb_fail({scenario, " timed out waiting for stalled done"});
            end
            #1;
            if (done_valid !== 1'b1) tb_fail({scenario, " done did not remain valid while stalled"});
            if ($isunknown({done_status, done_invalid, done_meta, done_valid_seq_len})) begin
                tb_fail({scenario, " stalled done contains X"});
            end
        end
    endtask

    task automatic recover_after_reset(input string scenario, input int scenario_id);
        begin
            load_weights_to_dut();
            #1;
            if (u_dut.u_shared_projection_controller.matrix_complete !== 4'hf) begin
                tb_fail({scenario, " recovery weights are not complete after reload"});
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[0][0][0] !== weights[0][0][0]) begin
                tb_fail({scenario, " recovery weight data mismatch after reload"});
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[2][0][0] !== weights[2][0][0]) begin
                tb_fail({scenario, " recovery V weight data mismatch after reload"});
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[3][0][0] !== weights[3][0][0]) begin
                tb_fail({scenario, " recovery WO weight data mismatch after reload"});
            end
            setup_reset_recovery_step(scenario, scenario_id);
            run_current_step();
            if (perf_generation_steps !== COUNTER_W'(1)) tb_fail({scenario, " duplicate or missing commit after reset recovery"});
            if (current_valid_seq_len !== SEQ_LEN_W'(1)) tb_fail({scenario, " recovery valid_seq_len mismatch"});
        end
    endtask

    task automatic run_reset_state_scenario(
        input string scenario,
        input int scenario_id,
        input logic [3:0] target_state
    );
        begin
            apply_reset();
            load_weights_to_dut();
            start_reset_interrupt_token(scenario, scenario_id);
            wait_for_state(target_state, scenario);
            pulse_mid_transaction_reset(scenario);
            recover_after_reset(scenario, scenario_id);
        end
    endtask

    task automatic run_reset_concat_scenario(input string scenario, input int scenario_id);
        begin
            apply_reset();
            load_weights_to_dut();
            start_reset_interrupt_token(scenario, scenario_id);
            wait_for_concat_activity(scenario);
            pulse_mid_transaction_reset(scenario);
            recover_after_reset(scenario, scenario_id);
        end
    endtask

    task automatic run_reset_final_output_scenario(input string scenario, input int scenario_id);
        begin
            apply_reset();
            load_weights_to_dut();
            start_reset_interrupt_token(scenario, scenario_id);
            wait_for_final_output_stall(scenario);
            pulse_mid_transaction_reset(scenario);
            recover_after_reset(scenario, scenario_id);
        end
    endtask

    task automatic run_reset_final_done_scenario(input string scenario, input int scenario_id);
        begin
            apply_reset();
            load_weights_to_dut();
            start_reset_interrupt_token(scenario, scenario_id);
            wait_for_final_done_stall(scenario);
            pulse_mid_transaction_reset(scenario);
            recover_after_reset(scenario, scenario_id);
        end
    endtask

    task automatic run_directed_reset_tests;
        begin
            if (!reset_baseline_valid) tb_fail("reset baseline was not captured");
            apply_reset();
            load_weights_to_dut();
            restore_reset_baseline_step();
            run_current_step();
            run_reset_state_scenario("reset_during_q_projection", 1, DUT_ST_RUN_Q);
            run_reset_state_scenario("reset_during_k_projection", 2, DUT_ST_RUN_K);
            run_reset_state_scenario("reset_during_v_projection", 3, DUT_ST_RUN_V);
            run_reset_state_scenario("reset_during_qkv_stream", 4, DUT_ST_QKV_STREAM);
            run_reset_state_scenario("reset_during_attention", 5, DUT_ST_ATTENTION);
            run_reset_concat_scenario("reset_during_concat_quantization", 6);
            run_reset_state_scenario("reset_during_wo_projection", 7, DUT_ST_OUTPUT_PROJECTION);
            run_reset_final_output_scenario("reset_during_final_output_stall", 8);
            run_reset_final_done_scenario("reset_during_final_done_stall", 9);
            $display("STAGE6E_DIRECTED_RESET_PASS N_HEAD=%0d D_HEAD=%0d scenarios=9", N_HEAD, D_HEAD);
        end
    endtask

    task automatic load_weights_to_dut;
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            for (int kind = 0; kind < 4; kind++) begin
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        drive_valid = 1'b1;
                        wait_cycles = 0;
                        while (drive_valid) begin
                            @(negedge clk);
                            weight_valid = 1'b1;
                            weight_kind = kind[1:0];
                            weight_output_index = MODEL_W'(out_idx);
                            weight_input_index = MODEL_W'(in_idx);
                            weight_data_fp16 = weights[kind][out_idx][in_idx];
                            weight_last = (out_idx == D_MODEL - 1) && (in_idx == D_MODEL - 1);
                            weight_commit = 1'b0;
                            #1;
                            pre_fire = weight_valid && weight_ready;
                            @(posedge clk); #1;
                            if (pre_fire) begin
                                drive_valid = 1'b0;
                                weight_valid = 1'b0;
                            end
                            wait_cycles++;
                            if (wait_cycles > 2000) tb_fail("weight handshake timeout");
                        end
                    end
                end
                @(negedge clk);
                weight_valid = 1'b0;
                weight_kind = kind[1:0];
                weight_commit = 1'b1;
                @(posedge clk); #1;
                weight_commit = 1'b0;
            end
            #1;
            if (u_dut.u_shared_projection_controller.matrix_complete !== 4'hf) begin
                tb_fail("projection weight buffer incomplete after load");
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[0][0][0] !== weights[0][0][0]) begin
                tb_fail("projection weight buffer WQ sample mismatch after load");
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[2][0][0] !== weights[2][0][0]) begin
                tb_fail("projection weight buffer WV sample mismatch after load");
            end
            if (u_dut.u_shared_projection_controller.u_weight_buffer.data_q[3][0][0] !== weights[3][0][0]) begin
                tb_fail("projection weight buffer WO sample mismatch after load");
            end
        end
    endtask

    task automatic drive_hidden;
        logic drive_valid;
        logic pre_fire;
        int wait_cycles;
        begin
            for (int idx = 0; idx < D_MODEL; idx++) begin
                drive_valid = 1'b1;
                wait_cycles = 0;
                while (drive_valid) begin
                    @(negedge clk);
                    token_valid = 1'b1;
                    token_dim = MODEL_W'(idx);
                    token_hidden_fp16 = current_hidden[idx];
                    token_last_dim = idx == D_MODEL - 1;
                    token_meta = current_meta;
                    #1;
                    pre_fire = token_valid && token_ready;
                    @(posedge clk); #1;
                    if (pre_fire) begin
                        drive_valid = 1'b0;
                        token_valid = 1'b0;
                    end
                    wait_cycles++;
                    if (wait_cycles > 2000) tb_fail("hidden handshake timeout");
                end
            end
            #1;
            if (u_dut.u_shared_projection_controller.input_complete !== 1'b1) begin
                tb_fail("hidden input buffer incomplete after load");
            end
            if (u_dut.u_shared_projection_controller.u_input_buffer.data_q[0] !== current_hidden[0]) begin
                tb_fail("hidden input buffer data mismatch after load");
            end
        end
    endtask

    task automatic run_current_step;
        int out_idx;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        logic [MODEL_W-1:0] pre_base;
        logic [PE_NUM-1:0] pre_mask;
        logic [PE_NUM*32-1:0] pre_vector;
        logic [7:0] pre_output_status;
        logic pre_output_invalid;
        logic [META_W-1:0] pre_output_meta;
        logic pre_output_last;
        logic [7:0] pre_done_status;
        logic pre_done_invalid;
        logic [META_W-1:0] pre_done_meta;
        logic [SEQ_LEN_W-1:0] pre_done_seq_len;
        bit done_seen;
        begin
            out_idx = 0;
            cycle = 0;
            done_seen = 1'b0;
            drive_hidden();
            while (!done_seen) begin
                @(negedge clk);
                output_ready = ((cycle % 5) != 1) && ((cycle % 11) != 7);
                done_ready = ((cycle % 7) != 3);
                if (token_ready) tb_fail("accepted next token before final done");
                #1;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                pre_base = output_base_dim;
                pre_mask = output_lane_mask;
                pre_vector = output_vector_fp32;
                pre_output_status = output_status;
                pre_output_invalid = output_invalid;
                pre_output_meta = output_meta;
                pre_output_last = output_last;
                pre_done_status = done_status;
                pre_done_invalid = done_invalid;
                pre_done_meta = done_meta;
                pre_done_seq_len = done_valid_seq_len;
                @(posedge clk); #1;

                if (pre_out_fire) begin
                    if (out_idx >= exp_count) tb_fail("too many output tiles");
                    if (pre_base !== exp_base[out_idx]) tb_fail("output base mismatch");
                    if (pre_mask !== exp_mask[out_idx]) tb_fail("output mask mismatch");
                    if (pre_vector !== exp_vector[out_idx]) begin
                        $display("CHECK_FAIL stage6e N_HEAD=%0d D_HEAD=%0d step=%s out=%0d got=%h expected=%h",
                                 N_HEAD, D_HEAD, current_name, out_idx, pre_vector, exp_vector[out_idx]);
                        $fatal(1);
                    end
                    if (pre_output_invalid !== 1'b0) tb_fail("valid output marked invalid");
                    if (pre_output_meta !== current_meta) tb_fail("output metadata mismatch");
                    if (pre_output_last !== exp_last[out_idx]) tb_fail("output last mismatch");
                    if (^pre_output_status === 1'bx) tb_fail("unknown output status");
                    out_idx++;
                end

                if (pre_done_fire) begin
                    if (out_idx != exp_count) tb_fail("done before all output tiles");
                    if (pre_done_invalid !== current_expect_invalid) tb_fail("done invalid mismatch");
                    if (current_expect_invalid && (pre_done_status !== current_expect_status)) tb_fail("done status mismatch");
                    if (pre_done_meta !== current_meta) tb_fail("done metadata mismatch");
                    if (pre_done_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("done valid_seq_len mismatch");
                    if (current_valid_seq_len !== SEQ_LEN_W'(current_seq_after)) tb_fail("current valid_seq_len mismatch");
                    $display("STAGE6E_INTEGRATED_MHA_PERF N_HEAD=%0d D_HEAD=%0d step=%s seq_before=%0d seq_after=%0d q=%0d k=%0d v=%0d qkv_quant=%0d attention=%0d concat=%0d wo=%0d total=%0d proj_pe_stall=%0d attn_pe_stall=%0d sfu_stall=%0d weight_stall=%0d buffer_stall=%0d output_stall=%0d peak_seq=%0d",
                             N_HEAD, D_HEAD, current_name, current_seq_before, current_seq_after,
                             perf_q_projection_cycles, perf_k_projection_cycles, perf_v_projection_cycles,
                             perf_qkv_quantization_cycles, perf_attention_cycles,
                             perf_concat_quantization_cycles, perf_output_projection_cycles,
                             perf_total_cycles, perf_projection_pe_stall_cycles,
                             perf_attention_pe_stall_cycles, perf_sfu_stall_cycles,
                             perf_weight_stall_cycles, perf_buffer_stall_cycles,
                             perf_output_stall_cycles, perf_peak_valid_seq_len);
                    done_seen = 1'b1;
                    done_ready = 1'b0;
                    output_ready = 1'b0;
                end

                cycle++;
                if (cycle > 6000000) tb_fail("projection integrated MHA timeout");
            end
        end
    endtask

    task automatic parse_and_run_file;
        string path;
        int fd;
        string tag;
        string name;
        int code;
        int seq_before;
        int seq_after;
        int expect_invalid;
        int base;
        int last;
        logic [META_W-1:0] meta;
        logic [7:0] status;
        logic [PE_NUM-1:0] mask;
        logic [31:0] values [0:PE_NUM-1];
        begin
            if (!$value$plusargs("INTEGRATED_MHA_VECTOR_FILE=%s", path)) tb_fail("missing +INTEGRATED_MHA_VECTOR_FILE");
            fd = $fopen(path, "r");
            if (fd == 0) tb_fail("could not open integrated MHA vector file");

            for (int kind = 0; kind < 4; kind++) begin
                code = $fscanf(fd, "%s", tag);
                if (kind == 0 && tag != "WQ") tb_fail("expected WQ");
                if (kind == 1 && tag != "WK") tb_fail("expected WK");
                if (kind == 2 && tag != "WV") tb_fail("expected WV");
                if (kind == 3 && tag != "WO") tb_fail("expected WO");
                for (int out_idx = 0; out_idx < D_MODEL; out_idx++) begin
                    for (int in_idx = 0; in_idx < D_MODEL; in_idx++) begin
                        code = $fscanf(fd, "%h", weights[kind][out_idx][in_idx]);
                        if (code != 1) tb_fail("weight parse failed");
                    end
                end
            end

            load_weights_to_dut();
            step_run_count = 0;
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code != 1) begin
                    void'($fgets(tag, fd));
                end else if (tag == "STEP") begin
                    code = $fscanf(fd, "%s %h %d %d %d %h\n", name, meta, seq_before, seq_after, expect_invalid, status);
                    if (code != 6) tb_fail("bad STEP line");
                    current_name = name;
                    current_meta = meta;
                    current_seq_before = seq_before;
                    current_seq_after = seq_after;
                    current_expect_invalid = expect_invalid[0];
                    current_expect_status = status;
                    exp_count = 0;
                    if (current_valid_seq_len !== SEQ_LEN_W'(seq_before)) tb_fail("pre-step valid_seq_len mismatch");
                end else if (tag == "H") begin
                    for (int idx = 0; idx < D_MODEL; idx++) begin
                        code = $fscanf(fd, "%h", current_hidden[idx]);
                        if (code != 1) tb_fail("hidden parse failed");
                    end
                end else if (tag == "O") begin
                    code = $fscanf(fd, "%d %h %h %h %h %h %h %h %h %h %d\n",
                        base, mask,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7],
                        last);
                    if (code != 11) tb_fail("bad O line");
                    if (exp_count >= MAX_OUTPUTS) tb_fail("too many expected output tiles");
                    exp_base[exp_count] = MODEL_W'(base);
                    exp_mask[exp_count] = mask;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        exp_vector[exp_count][lane*32 +: 32] = values[lane];
                    end
                    exp_last[exp_count] = last[0];
                    exp_count++;
                end else if (tag == "RUN") begin
                    if (step_run_count == 0) begin
                        save_reset_baseline();
                        run_directed_reset_tests();
                        apply_reset();
                        load_weights_to_dut();
                        restore_reset_baseline_step();
                    end
                    run_current_step();
                    step_run_count++;
                end else if (tag == "END") begin
                    // Step boundary.
                end else begin
                    tb_fail({"unknown vector tag ", tag});
                end
            end
            $fclose(fd);
            if (step_run_count != (MAX_SEQ_LEN + 1)) tb_fail("did not execute all integrated MHA steps");
        end
    endtask

    initial begin
        current_name = "none";
        step_run_count = 0;
        reset_baseline_valid = 1'b0;
        apply_reset();
        parse_and_run_file();
        $display("STAGE6E_INTEGRATED_MHA_PASS N_HEAD=%0d D_HEAD=%0d generation_steps=%0d",
                 N_HEAD, D_HEAD, perf_generation_steps);
        $finish;
    end
endmodule

`default_nettype wire
