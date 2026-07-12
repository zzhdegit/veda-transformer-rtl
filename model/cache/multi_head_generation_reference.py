"""Stage 5 shared multi-head generation attention reference model."""

from collections import namedtuple

from model.attention.single_head_reference import single_head_attention_bit_model
from model.cache.multi_head_kv_cache_reference import (
    MultiHeadKVCacheError,
    MultiHeadKVCacheReference,
)


STATUS_OK = 0x00
STATUS_CACHE_FULL = 0x82
STATUS_HEAD_FAILED = 0x84


MultiHeadGenerationStepTrace = namedtuple(
    "MultiHeadGenerationStepTrace",
    [
        "meta",
        "seq_len_before",
        "seq_len_after",
        "outputs",
        "status",
        "invalid",
        "cache_k",
        "cache_v",
        "attention_traces",
        "failed_head",
    ],
)


class MultiHeadGenerationReference(object):
    def __init__(self, n_head, d_head, max_seq_len, pe_num):
        self.n_head = n_head
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.pe_num = pe_num
        self.cache = MultiHeadKVCacheReference(n_head, d_head, max_seq_len)

    def reset(self):
        self.cache.reset()

    def prefill(self, k_tokens, v_tokens):
        if len(k_tokens) != len(v_tokens):
            raise ValueError("K/V prefill length mismatch")
        for k_heads, v_heads in zip(k_tokens, v_tokens):
            self.cache.append_token(k_heads, v_heads)

    def _validate_heads(self, q_heads, k_heads, v_heads):
        if len(q_heads) != self.n_head or len(k_heads) != self.n_head or len(v_heads) != self.n_head:
            raise ValueError("Q/K/V head count mismatch")
        for head in range(self.n_head):
            if (
                len(q_heads[head]) != self.d_head
                or len(k_heads[head]) != self.d_head
                or len(v_heads[head]) != self.d_head
            ):
                raise ValueError("Q/K/V row dimension mismatch")

    def run_token(self, q_heads, k_heads, v_heads, meta=0, fail_head=None):
        self._validate_heads(q_heads, k_heads, v_heads)
        if fail_head is not None and (fail_head < 0 or fail_head >= self.n_head):
            raise ValueError("fail_head out of range")

        seq_len_before = self.cache.valid_seq_len
        committed_k, committed_v = self.cache.snapshot()

        if self.cache.cache_full:
            return MultiHeadGenerationStepTrace(
                meta=meta,
                seq_len_before=seq_len_before,
                seq_len_after=seq_len_before,
                outputs=[],
                status=STATUS_CACHE_FULL,
                invalid=True,
                cache_k=committed_k,
                cache_v=committed_v,
                attention_traces=[],
                failed_head=None,
            )

        self.cache.append_provisional_token(k_heads, v_heads)
        outputs = []
        attention_traces = []
        for head in range(self.n_head):
            if fail_head is not None and head == fail_head:
                self.cache.abort_provisional()
                cache_k, cache_v = self.cache.snapshot()
                return MultiHeadGenerationStepTrace(
                    meta=meta,
                    seq_len_before=seq_len_before,
                    seq_len_after=seq_len_before,
                    outputs=outputs,
                    status=STATUS_HEAD_FAILED,
                    invalid=True,
                    cache_k=cache_k,
                    cache_v=cache_v,
                    attention_traces=attention_traces,
                    failed_head=head,
                )
            visible_k, visible_v = self.cache.visible_snapshot(head, include_provisional=True)
            attention_trace = single_head_attention_bit_model(q_heads[head], visible_k, visible_v, self.pe_num)
            outputs.append(list(attention_trace.output))
            attention_traces.append(attention_trace)

        self.cache.commit_provisional()
        cache_k, cache_v = self.cache.snapshot()
        return MultiHeadGenerationStepTrace(
            meta=meta,
            seq_len_before=seq_len_before,
            seq_len_after=self.cache.valid_seq_len,
            outputs=outputs,
            status=STATUS_OK,
            invalid=False,
            cache_k=cache_k,
            cache_v=cache_v,
            attention_traces=attention_traces,
            failed_head=None,
        )


def run_multi_head_generation(tokens, n_head, d_head, max_seq_len, pe_num):
    ref = MultiHeadGenerationReference(n_head, d_head, max_seq_len, pe_num)
    traces = []
    for token in tokens:
        traces.append(
            ref.run_token(
                token["q"],
                token["k"],
                token["v"],
                token.get("meta", 0),
                token.get("fail_head"),
            )
        )
    return traces


def require_no_error(trace):
    if trace.invalid:
        raise MultiHeadKVCacheError("generation step failed with status 0x%02x" % trace.status)
