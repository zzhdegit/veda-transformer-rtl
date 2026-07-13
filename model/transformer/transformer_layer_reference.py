"""Stage 7 Pre-Norm Transformer layer reference."""

from collections import namedtuple

from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.projection.projection_mha_reference import ProjectionMhaReference
from model.transformer.ffn_reference import ffn_forward
from model.transformer.residual_reference import residual_add
from model.transformer.rmsnorm_reference import EPS_FP32_DEFAULT, rmsnorm


NORM1_GAMMA = 4
NORM2_GAMMA = 5
FFN_W1 = 6
FFN_W2 = 7

TransformerLayerStepTrace = namedtuple(
    "TransformerLayerStepTrace",
    [
        "meta",
        "input_fp32",
        "norm1",
        "norm1_sum_sq",
        "norm1_inv_rms",
        "norm1_fp32",
        "norm1_fp16",
        "mha",
        "mha_fp32",
        "residual1",
        "residual1_fp32",
        "norm2",
        "norm2_sum_sq",
        "norm2_inv_rms",
        "norm2_fp32",
        "norm2_fp16",
        "ffn",
        "ffn1_fp32",
        "relu_fp32",
        "activation_fp16",
        "ffn2_fp32",
        "residual2",
        "final_fp32",
        "status",
        "invalid",
    ],
)


class TransformerLayerReference(object):
    def __init__(self, n_head, d_head, max_seq_len, pe_num, mha_weights, gamma1_fp16, gamma2_fp16, w1, w2, eps_fp32=EPS_FP32_DEFAULT):
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = n_head * d_head
        self.d_ffn = 4 * self.d_model
        self.max_seq_len = max_seq_len
        self.pe_num = pe_num
        self.gamma1_fp16 = gamma1_fp16
        self.gamma2_fp16 = gamma2_fp16
        self.w1 = w1
        self.w2 = w2
        self.eps_fp32 = eps_fp32
        if len(gamma1_fp16) != self.d_model or len(gamma2_fp16) != self.d_model:
            raise ValueError("gamma length must equal D_MODEL")
        if len(w1) != self.d_ffn or any(len(row) != self.d_model for row in w1):
            raise ValueError("W1 shape must be D_FFN x D_MODEL")
        if len(w2) != self.d_model or any(len(row) != self.d_ffn for row in w2):
            raise ValueError("W2 shape must be D_MODEL x D_FFN")
        self.mha = ProjectionMhaReference(n_head, d_head, max_seq_len, pe_num, mha_weights)

    def reset(self):
        self.mha.reset()

    def _expand_input(self, hidden_fp16):
        if len(hidden_fp16) != self.d_model:
            raise ValueError("hidden length must equal D_MODEL")
        input_fp32 = []
        invalid = False
        for value in hidden_fp16:
            conv = fp16_to_fp32_bits(value)
            input_fp32.append(conv["output_bits"])
            invalid = invalid or bool(conv["invalid"])
        return input_fp32, invalid

    def run_token(self, hidden_fp16, meta=0):
        input_fp32, input_invalid = self._expand_input(hidden_fp16)
        norm1_trace = rmsnorm(input_fp32, self.gamma1_fp16, self.eps_fp32)
        mha_trace = self.mha.run_token(norm1_trace.norm_fp16, meta)

        if input_invalid or norm1_trace.invalid or mha_trace.invalid:
            return TransformerLayerStepTrace(
                meta=meta,
                input_fp32=input_fp32,
                norm1=norm1_trace,
                norm1_sum_sq=norm1_trace.sum_sq,
                norm1_inv_rms=norm1_trace.inv_rms,
                norm1_fp32=norm1_trace.norm_fp32,
                norm1_fp16=norm1_trace.norm_fp16,
                mha=mha_trace,
                mha_fp32=mha_trace.final_output_fp32,
                residual1=None,
                residual1_fp32=[],
                norm2=None,
                norm2_sum_sq=0,
                norm2_inv_rms=0,
                norm2_fp32=[],
                norm2_fp16=[],
                ffn=None,
                ffn1_fp32=[],
                relu_fp32=[],
                activation_fp16=[],
                ffn2_fp32=[],
                residual2=None,
                final_fp32=[],
                status=mha_trace.status,
                invalid=True,
            )

        residual1_trace = residual_add(input_fp32, mha_trace.final_output_fp32)
        norm2_trace = rmsnorm(residual1_trace.output_fp32, self.gamma2_fp16, self.eps_fp32)
        ffn_trace = ffn_forward(norm2_trace.norm_fp16, self.w1, self.w2, self.pe_num)
        residual2_trace = residual_add(residual1_trace.output_fp32, ffn_trace.ffn2_fp32)
        invalid = residual1_trace.invalid or norm2_trace.invalid or ffn_trace.invalid or residual2_trace.invalid

        return TransformerLayerStepTrace(
            meta=meta,
            input_fp32=input_fp32,
            norm1=norm1_trace,
            norm1_sum_sq=norm1_trace.sum_sq,
            norm1_inv_rms=norm1_trace.inv_rms,
            norm1_fp32=norm1_trace.norm_fp32,
            norm1_fp16=norm1_trace.norm_fp16,
            mha=mha_trace,
            mha_fp32=mha_trace.final_output_fp32,
            residual1=residual1_trace,
            residual1_fp32=residual1_trace.output_fp32,
            norm2=norm2_trace,
            norm2_sum_sq=norm2_trace.sum_sq,
            norm2_inv_rms=norm2_trace.inv_rms,
            norm2_fp32=norm2_trace.norm_fp32,
            norm2_fp16=norm2_trace.norm_fp16,
            ffn=ffn_trace,
            ffn1_fp32=ffn_trace.ffn1_fp32,
            relu_fp32=ffn_trace.relu_fp32,
            activation_fp16=ffn_trace.activation_fp16,
            ffn2_fp32=ffn_trace.ffn2_fp32,
            residual2=residual2_trace,
            final_fp32=residual2_trace.output_fp32,
            status=mha_trace.status,
            invalid=invalid,
        )
