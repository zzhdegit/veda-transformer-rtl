"""Stage 7 no-stall cycle model scaffold."""


def estimate_no_stall_cycles(n_head, d_head, pe_num=8):
    d_model = n_head * d_head
    d_ffn = 4 * d_model
    input_load = d_model
    norm_reduce = d_model
    norm_apply = d_model
    mha = d_model * d_model * 4 // max(pe_num, 1) + n_head * d_head
    residual = d_model
    ffn1 = d_ffn * ((d_model + pe_num - 1) // pe_num)
    relu = d_ffn
    activation_quantization = d_ffn
    ffn2 = d_model * ((d_ffn + pe_num - 1) // pe_num)
    final_output = (d_model + pe_num - 1) // pe_num
    control_overhead = 24
    total = (
        input_load
        + norm_reduce
        + norm_apply
        + mha
        + residual
        + norm_reduce
        + norm_apply
        + ffn1
        + relu
        + activation_quantization
        + ffn2
        + residual
        + final_output
        + control_overhead
    )
    return {
        "input_load": input_load,
        "norm1_reduce": norm_reduce,
        "norm1_apply": norm_apply,
        "mha": mha,
        "residual1": residual,
        "norm2_reduce": norm_reduce,
        "norm2_apply": norm_apply,
        "ffn1": ffn1,
        "relu": relu,
        "activation_quantization": activation_quantization,
        "ffn2": ffn2,
        "residual2": residual,
        "final_output": final_output,
        "control_overhead": control_overhead,
        "total": total,
    }


def main():
    print("Stage 7 no-stall single-token cycle estimate")
    print("total_layer_cycles = input_load + norm1 + MHA + residual1 + norm2 + FFN + residual2 + final_output + control_overhead")
    for name, n_head, d_head in (("H1D8", 1, 8), ("H2D8", 2, 8), ("H4D8", 4, 8), ("H2D16", 2, 16)):
        row = estimate_no_stall_cycles(n_head, d_head)
        print(
            "%s total=%d input=%d norm1_reduce=%d norm1_apply=%d mha=%d residual1=%d norm2_reduce=%d norm2_apply=%d ffn1=%d relu=%d act_quant=%d ffn2=%d residual2=%d final=%d overhead=%d"
            % (
                name,
                row["total"],
                row["input_load"],
                row["norm1_reduce"],
                row["norm1_apply"],
                row["mha"],
                row["residual1"],
                row["norm2_reduce"],
                row["norm2_apply"],
                row["ffn1"],
                row["relu"],
                row["activation_quantization"],
                row["ffn2"],
                row["residual2"],
                row["final_output"],
                row["control_overhead"],
            )
        )


if __name__ == "__main__":
    main()
