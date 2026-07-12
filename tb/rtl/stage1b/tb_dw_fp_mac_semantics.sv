`timescale 1ns/1ps
`default_nettype none

module tb_dw_fp_mac_semantics;
    localparam [31:0] DISC_A = 32'h3f2d72e9;
    localparam [31:0] DISC_B = 32'h3f7e9531;
    localparam [31:0] DISC_C = 32'hbf40153d;
    localparam [31:0] DISC_FUSED = 32'hbd9cc126;
    localparam [31:0] DISC_NON_FUSED = 32'hbd9cc128;
    localparam [31:0] RNE_A = 32'hc00f6617;
    localparam [31:0] RNE_B = 32'hc07cab26;
    localparam [31:0] RNE_C = 32'hc3147987;
    localparam [31:0] RNE_EXPECTED = 32'hc30ba101;

    reg [2:0] rnd;
    wire [31:0] z;
    wire [7:0] status;
    wire [31:0] z_rne;
    wire [7:0] status_rne;

    DW_fp_mac #(23, 8, 1) u_dw_fp_mac (
        .a      (DISC_A),
        .b      (DISC_B),
        .c      (DISC_C),
        .rnd    (rnd),
        .z      (z),
        .status (status)
    );

    DW_fp_mac #(23, 8, 1) u_dw_fp_mac_rne (
        .a      (RNE_A),
        .b      (RNE_B),
        .c      (RNE_C),
        .rnd    (rnd),
        .z      (z_rne),
        .status (status_rne)
    );

    initial begin
        for (int i = 0; i < 8; i = i + 1) begin
            rnd = i[2:0];
            #10;
            $display("DW_FP_MAC_RND_%0d result=%08h status=%02h rne_result=%08h rne_status=%02h",
                     i, z, status, z_rne, status_rne);
            if (z === DISC_FUSED) begin
                $display("DW_FP_MAC_SEMANTICS_FUSED rnd=%0d result=%08h status=%02h", i, z, status);
            end
            if (z === DISC_NON_FUSED) begin
                $display("DW_FP_MAC_SEMANTICS_NON_FUSED rnd=%0d result=%08h status=%02h", i, z, status);
            end
            if ((z === DISC_FUSED) && (z_rne === RNE_EXPECTED)) begin
                $display("DW_FP_MAC_RNE_RND=%0d result=%08h rne_result=%08h status=%02h rne_status=%02h",
                         i, z, z_rne, status, status_rne);
            end
        end
        if (1'b1) begin
            $display("DW_FP_MAC_SEMANTICS_PROBE_DONE fused=%08h non_fused=%08h", DISC_FUSED, DISC_NON_FUSED);
        end
        $finish;
    end
endmodule

`default_nettype wire
