from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.attention.single_head_reference import single_head_attention_bit_model
from model.cache.generation_reference import GenerationReference
from model.cache.multi_head_generation_reference import (
    MultiHeadGenerationReference,
    STATUS_CACHE_FULL,
    STATUS_HEAD_FAILED,
)
from model.cache.multi_head_kv_cache_reference import (
    MultiHeadKVCacheError,
    MultiHeadKVCacheReference,
    multi_head_linear_address,
)


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_TWO = 0xC000
FP16_NEG_HALF = 0xB800


def row(pattern, d_head, offset=0):
    return [pattern[(offset + idx) % len(pattern)] for idx in range(d_head)]


def multi_head_token_stream(n_head, d_head, count):
    q_pattern = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF, FP16_ZERO]
    k_pattern = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_NEG_HALF]
    v_pattern = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE, FP16_ZERO]
    tokens = []
    for step in range(count):
        q_heads = []
        k_heads = []
        v_heads = []
        for head in range(n_head):
            offset = step * 5 + head * 7
            q_heads.append(row(q_pattern, d_head, offset))
            k_heads.append(row(k_pattern, d_head, offset + 2))
            v_heads.append(row(v_pattern, d_head, offset + 4))
        tokens.append({"q": q_heads, "k": k_heads, "v": v_heads, "meta": 0x850 + step})
    return tokens


def assert_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError("expected %s" % exc_type.__name__)


def test_multi_head_linear_address_boundaries():
    assert multi_head_linear_address(0, 0, 0, 4, 8, 32) == 0
    assert multi_head_linear_address(0, 0, 7, 4, 8, 32) == 7
    assert multi_head_linear_address(1, 0, 0, 4, 8, 32) == 256
    assert multi_head_linear_address(3, 31, 7, 4, 8, 32) == 1023
    assert_raises(MultiHeadKVCacheError, multi_head_linear_address, 4, 0, 0, 4, 8, 32)
    assert_raises(MultiHeadKVCacheError, multi_head_linear_address, 0, 32, 0, 4, 8, 32)
    assert_raises(MultiHeadKVCacheError, multi_head_linear_address, 0, 0, 8, 4, 8, 32)


def test_multi_head_provisional_visibility_commit_abort_and_order():
    cache = MultiHeadKVCacheReference(n_head=2, d_head=4, max_seq_len=2)
    cache.append_dim(0, 0, 0, FP16_ONE, FP16_HALF, False, False, False)
    cache.append_dim(0, 0, 1, FP16_ONE, FP16_HALF, False, False, False)
    cache.append_dim(0, 0, 2, FP16_ONE, FP16_HALF, False, False, False)
    cache.append_dim(0, 0, 3, FP16_ONE, FP16_HALF, True, False, False)
    assert cache.valid_seq_len == 0
    assert cache.provisional_head_valid == [True, False]
    assert cache.read(0, 0, 0, include_provisional=True) == (FP16_ONE, FP16_HALF)
    assert_raises(MultiHeadKVCacheError, cache.read, 1, 0, 0, True)
    assert_raises(MultiHeadKVCacheError, cache.commit_provisional)

    assert_raises(
        MultiHeadKVCacheError,
        cache.append_dim,
        1,
        0,
        2,
        FP16_TWO,
        FP16_ONE,
        False,
        False,
        False,
    )
    assert cache.valid_seq_len == 0
    assert not cache.append_incomplete

    cache.append_provisional_token(
        [[FP16_ONE] * 4, [FP16_TWO] * 4],
        [[FP16_HALF] * 4, [FP16_NEG_HALF] * 4],
    )
    assert cache.provisional_valid
    assert cache.provisional_token_index == 0
    cache.abort_provisional()
    assert cache.valid_seq_len == 0
    assert not cache.provisional_valid

    cache.append_token([[FP16_ONE] * 4, [FP16_TWO] * 4], [[FP16_HALF] * 4, [FP16_NEG_HALF] * 4])
    assert cache.valid_seq_len == 1
    assert cache.read(1, 0, 3) == (FP16_TWO, FP16_NEG_HALF)


def test_first_token_each_head_attends_to_current_token():
    ref = MultiHeadGenerationReference(n_head=2, d_head=8, max_seq_len=8, pe_num=8)
    token = multi_head_token_stream(2, 8, 1)[0]
    trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
    assert trace.seq_len_before == 0
    assert trace.seq_len_after == 1
    assert not trace.invalid
    for head in range(2):
        expected = single_head_attention_bit_model(token["q"][head], [token["k"][head]], [token["v"][head]], 8)
        expected_v_fp32 = [fp16_to_fp32_bits(value)["output_bits"] for value in token["v"][head]]
        assert trace.outputs[head] != [0] * 8
        assert trace.outputs[head] == expected.output
        assert trace.outputs[head] == expected_v_fp32
        assert trace.attention_traces[head].probabilities == [0x3F800000]


def test_second_token_uses_history_and_current_per_head():
    d_head = 8
    ref = MultiHeadGenerationReference(n_head=2, d_head=d_head, max_seq_len=8, pe_num=8)
    first = {
        "q": [[FP16_TWO] * d_head, [FP16_TWO] * d_head],
        "k": [[FP16_NEG_TWO] * d_head, [FP16_NEG_TWO] * d_head],
        "v": [[FP16_HALF] * d_head, [FP16_NEG_HALF] * d_head],
        "meta": 0x900,
    }
    second = {
        "q": [[FP16_TWO] * d_head, [FP16_TWO] * d_head],
        "k": [[FP16_TWO] * d_head, [FP16_TWO] * d_head],
        "v": [[FP16_ONE] * d_head, [FP16_TWO] * d_head],
        "meta": 0x901,
    }
    ref.run_token(first["q"], first["k"], first["v"], first["meta"])
    trace = ref.run_token(second["q"], second["k"], second["v"], second["meta"])
    assert trace.seq_len_before == 1
    assert trace.seq_len_after == 2
    for head in range(2):
        expected = single_head_attention_bit_model(
            second["q"][head],
            [first["k"][head], second["k"][head]],
            [first["v"][head], second["v"][head]],
            8,
        )
        old_only = single_head_attention_bit_model(second["q"][head], [first["k"][head]], [first["v"][head]], 8)
        assert trace.outputs[head] == expected.output
        assert trace.outputs[head] != old_only.output


def test_multi_head_steps_match_per_head_bit_model():
    for n_head in [1, 2, 4]:
        for d_head in [8, 16]:
            for steps in [1, 2, 3, 8]:
                ref = MultiHeadGenerationReference(n_head=n_head, d_head=d_head, max_seq_len=8, pe_num=8)
                tokens = multi_head_token_stream(n_head, d_head, steps)
                for step, token in enumerate(tokens):
                    trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
                    assert trace.meta == token["meta"]
                    assert trace.seq_len_before == step
                    assert trace.seq_len_after == step + 1
                    assert len(trace.outputs) == n_head
                    for head in range(n_head):
                        expected = single_head_attention_bit_model(
                            token["q"][head],
                            [prev["k"][head] for prev in tokens[: step + 1]],
                            [prev["v"][head] for prev in tokens[: step + 1]],
                            8,
                        )
                        assert len(trace.attention_traces[head].probabilities) == step + 1
                        assert trace.outputs[head] == expected.output
                        assert trace.cache_k[head][step] == token["k"][head]
                        assert trace.cache_v[head][step] == token["v"][head]


def test_n_head_one_matches_stage4p1_reference():
    tokens = multi_head_token_stream(1, 8, 4)
    stage4 = GenerationReference(d_head=8, max_seq_len=8, pe_num=8)
    stage5 = MultiHeadGenerationReference(n_head=1, d_head=8, max_seq_len=8, pe_num=8)
    for token in tokens:
        trace4 = stage4.run_token(token["q"][0], token["k"][0], token["v"][0], token["meta"])
        trace5 = stage5.run_token(token["q"], token["k"], token["v"], token["meta"])
        assert trace5.seq_len_before == trace4.seq_len_before
        assert trace5.seq_len_after == trace4.seq_len_after
        assert trace5.status == trace4.status
        assert trace5.invalid == trace4.invalid
        assert trace5.outputs == [trace4.output]


def test_cache_full_does_not_overwrite_or_emit_outputs():
    ref = MultiHeadGenerationReference(n_head=2, d_head=8, max_seq_len=3, pe_num=8)
    tokens = multi_head_token_stream(2, 8, 4)
    for token in tokens[:3]:
        trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
        assert not trace.invalid
    before_k, before_v = ref.cache.snapshot()
    trace = ref.run_token(tokens[3]["q"], tokens[3]["k"], tokens[3]["v"], tokens[3]["meta"])
    after_k, after_v = ref.cache.snapshot()
    assert trace.invalid
    assert trace.status == STATUS_CACHE_FULL
    assert trace.seq_len_before == 3
    assert trace.seq_len_after == 3
    assert trace.outputs == []
    assert before_k == after_k
    assert before_v == after_v


def test_failed_head_aborts_all_provisional_heads_atomically():
    ref = MultiHeadGenerationReference(n_head=4, d_head=8, max_seq_len=8, pe_num=8)
    tokens = multi_head_token_stream(4, 8, 3)
    first = ref.run_token(tokens[0]["q"], tokens[0]["k"], tokens[0]["v"], tokens[0]["meta"])
    assert not first.invalid
    before_k, before_v = ref.cache.snapshot()
    failed = ref.run_token(tokens[1]["q"], tokens[1]["k"], tokens[1]["v"], tokens[1]["meta"], fail_head=2)
    after_k, after_v = ref.cache.snapshot()
    assert failed.invalid
    assert failed.status == STATUS_HEAD_FAILED
    assert failed.failed_head == 2
    assert failed.seq_len_before == 1
    assert failed.seq_len_after == 1
    assert before_k == after_k
    assert before_v == after_v

    recovered = ref.run_token(tokens[2]["q"], tokens[2]["k"], tokens[2]["v"], tokens[2]["meta"])
    assert not recovered.invalid
    assert recovered.seq_len_before == 1
    assert recovered.seq_len_after == 2
