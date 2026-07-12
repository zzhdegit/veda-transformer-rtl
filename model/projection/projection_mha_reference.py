"""Stage 6 projection-integrated MHA reference framework."""

from collections import namedtuple

from model.cache.multi_head_generation_reference import MultiHeadGenerationReference
from model.projection.projection_reference import WO, output_projection, qkv_projection


ProjectionMhaStepTrace = namedtuple(
    "ProjectionMhaStepTrace",
    [
        "meta",
        "qkv",
        "attention",
        "output_projection",
        "final_output_fp32",
        "status",
        "invalid",
    ],
)


class ProjectionMhaReference(object):
    def __init__(self, n_head, d_head, max_seq_len, pe_num, weights):
        self.n_head = n_head
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.pe_num = pe_num
        self.weights = weights
        self.stage5 = MultiHeadGenerationReference(n_head, d_head, max_seq_len, pe_num)

    def reset(self):
        self.stage5.reset()

    def run_token(self, hidden_fp16, meta=0):
        qkv = qkv_projection(hidden_fp16, self.weights, self.n_head, self.d_head, self.pe_num)
        attention = self.stage5.run_token(qkv.q_heads, qkv.k_heads, qkv.v_heads, meta)
        if attention.invalid:
            output = None
            final_output = []
        else:
            output = output_projection(attention.outputs, self.weights[WO], self.n_head, self.d_head, self.pe_num)
            final_output = output.output_fp32
        return ProjectionMhaStepTrace(
            meta=meta,
            qkv=qkv,
            attention=attention,
            output_projection=output,
            final_output_fp32=final_output,
            status=attention.status,
            invalid=attention.invalid,
        )
