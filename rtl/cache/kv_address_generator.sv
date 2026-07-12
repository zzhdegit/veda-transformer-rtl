`default_nettype none

module kv_address_generator #(
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int DEPTH = MAX_SEQ_LEN * D_HEAD,
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic [TOKEN_W-1:0] token_index,
    input  logic [DIM_W-1:0]   dimension,
    output logic [ADDR_W-1:0]  address,
    output logic               token_in_range,
    output logic               dim_in_range,
    output logic               address_in_range
);
    int unsigned address_int;

    initial begin
        if (D_HEAD <= 0 || MAX_SEQ_LEN <= 0) begin
            $fatal(1, "kv_address_generator parameters must be positive");
        end
    end

    always_comb begin
        address_int = (int'(token_index) * D_HEAD) + int'(dimension);
        address = ADDR_W'(address_int);
        token_in_range = (int'(token_index) < MAX_SEQ_LEN);
        dim_in_range = (int'(dimension) < D_HEAD);
        address_in_range = token_in_range && dim_in_range && (address_int < DEPTH);
    end
endmodule

`default_nettype wire
