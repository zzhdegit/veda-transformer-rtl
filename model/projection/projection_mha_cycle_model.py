"""Simple Stage 6 projection-integrated MHA cycle estimate helpers."""


def ceil_div(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def no_stall_single_token_estimate(n_head, d_head, pe_num, seq_len_before):
    d_model = n_head * d_head
    tiles_model = ceil_div(d_model, pe_num)
    tiles_head = ceil_div(d_head, pe_num)
    projection_row = tiles_model + 2
    q_projection = d_model * projection_row
    k_projection = d_model * projection_row
    v_projection = d_model * projection_row
    qkv_quantization = 3 * d_model
    attention = n_head * (d_head + (seq_len_before + 1) * d_head * 2 + tiles_head * 6)
    concat_quantization = d_model
    output_projection = d_model + d_model * projection_row
    final_output = tiles_model
    control_overhead = 12 + n_head
    total = (
        d_model
        + q_projection
        + k_projection
        + v_projection
        + qkv_quantization
        + attention
        + concat_quantization
        + output_projection
        + final_output
        + control_overhead
    )
    return {
        "hidden_load": d_model,
        "q_projection": q_projection,
        "k_projection": k_projection,
        "v_projection": v_projection,
        "qkv_quantization": qkv_quantization,
        "attention": attention,
        "concat_quantization": concat_quantization,
        "output_projection": output_projection,
        "final_output": final_output,
        "control_overhead": control_overhead,
        "total_cycles": total,
    }


def example_table():
    return {
        "H1D8": no_stall_single_token_estimate(1, 8, 8, 0),
        "H2D8": no_stall_single_token_estimate(2, 8, 8, 0),
        "H4D8": no_stall_single_token_estimate(4, 8, 8, 0),
        "H2D16": no_stall_single_token_estimate(2, 16, 8, 0),
    }


def main():
    print("Stage 6 no-stall single-token cycle estimate")
    print(
        "total_cycles = hidden_load + q_projection + k_projection + v_projection "
        "+ qkv_quantization + attention + concat_quantization + output_projection "
        "+ final_output + control_overhead"
    )
    for name, row in example_table().items():
        print(
            "%s total=%d hidden=%d q=%d k=%d v=%d qkv_quant=%d attention=%d "
            "concat=%d wo=%d final_output=%d overhead=%d"
            % (
                name,
                row["total_cycles"],
                row["hidden_load"],
                row["q_projection"],
                row["k_projection"],
                row["v_projection"],
                row["qkv_quantization"],
                row["attention"],
                row["concat_quantization"],
                row["output_projection"],
                row["final_output"],
                row["control_overhead"],
            )
        )


if __name__ == "__main__":
    main()
