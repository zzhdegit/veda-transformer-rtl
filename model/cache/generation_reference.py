"""Stage 4 continuous single-head generation reference model."""

from collections import namedtuple

from model.attention.single_head_reference import single_head_attention_bit_model
from model.cache.kv_cache_reference import KVCacheError, KVCacheReference


STATUS_OK = 0x00
STATUS_CACHE_FULL = 0x82


GenerationStepTrace = namedtuple(
    "GenerationStepTrace",
    [
        "meta",
        "seq_len_before",
        "seq_len_after",
        "output",
        "status",
        "invalid",
        "cache_k",
        "cache_v",
        "attention_trace",
    ],
)


class GenerationReference(object):
    def __init__(self, d_head, max_seq_len, pe_num):
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.pe_num = pe_num
        self.cache = KVCacheReference(d_head, max_seq_len)

    def reset(self):
        self.cache.reset()

    def prefill(self, k_rows, v_rows):
        if len(k_rows) != len(v_rows):
            raise ValueError("K/V prefill length mismatch")
        for k_row, v_row in zip(k_rows, v_rows):
            self.cache.append_token(k_row, v_row)

    def run_token(self, q_row, k_row, v_row, meta=0):
        if len(q_row) != self.d_head or len(k_row) != self.d_head or len(v_row) != self.d_head:
            raise ValueError("Q/K/V row dimension mismatch")

        seq_len_before = self.cache.valid_seq_len
        committed_k, committed_v = self.cache.snapshot()

        if self.cache.cache_full:
            return GenerationStepTrace(
                meta=meta,
                seq_len_before=seq_len_before,
                seq_len_after=seq_len_before,
                output=[],
                status=STATUS_CACHE_FULL,
                invalid=True,
                cache_k=committed_k,
                cache_v=committed_v,
                attention_trace=None,
            )

        self.cache.append_provisional_token(k_row, v_row)
        visible_k, visible_v = self.cache.visible_snapshot(include_provisional=True)
        attention_trace = single_head_attention_bit_model(q_row, visible_k, visible_v, self.pe_num)
        output = list(attention_trace.output)
        self.cache.commit_provisional()
        cache_k, cache_v = self.cache.snapshot()
        return GenerationStepTrace(
            meta=meta,
            seq_len_before=seq_len_before,
            seq_len_after=self.cache.valid_seq_len,
            output=output,
            status=STATUS_OK,
            invalid=False,
            cache_k=cache_k,
            cache_v=cache_v,
            attention_trace=attention_trace,
        )


def run_generation(tokens, d_head, max_seq_len, pe_num):
    ref = GenerationReference(d_head, max_seq_len, pe_num)
    traces = []
    for token in tokens:
        traces.append(ref.run_token(token["q"], token["k"], token["v"], token.get("meta", 0)))
    return traces


def require_no_error(trace):
    if trace.invalid:
        raise KVCacheError("generation step failed with status 0x%02x" % trace.status)
