from model.arithmetic.fp16_fp32_reference import fp16_to_fp32_bits
from model.attention.single_head_reference import single_head_attention_bit_model
from model.cache.generation_reference import GenerationReference, STATUS_CACHE_FULL
from model.cache.kv_cache_reference import KVCacheError, KVCacheReference, linear_address


FP16_ZERO = 0x0000
FP16_ONE = 0x3C00
FP16_TWO = 0x4000
FP16_HALF = 0x3800
FP16_NEG_ONE = 0xBC00
FP16_NEG_HALF = 0xB800


def row(pattern, d_head, offset=0):
    return [pattern[(offset + idx) % len(pattern)] for idx in range(d_head)]


def token_stream(d_head, count):
    q_pattern = [FP16_ONE, FP16_HALF, FP16_NEG_ONE, FP16_TWO, FP16_NEG_HALF, FP16_ZERO]
    k_pattern = [FP16_TWO, FP16_NEG_ONE, FP16_HALF, FP16_ONE, FP16_ZERO, FP16_NEG_HALF]
    v_pattern = [FP16_HALF, FP16_ONE, FP16_NEG_HALF, FP16_TWO, FP16_NEG_ONE, FP16_ZERO]
    tokens = []
    for step in range(count):
        tokens.append(
            {
                "q": row(q_pattern, d_head, step),
                "k": row(k_pattern, d_head, step * 3),
                "v": row(v_pattern, d_head, step * 5),
                "meta": 0x700 + step,
            }
        )
    return tokens


def assert_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError("expected %s" % exc_type.__name__)


def test_linear_address_boundaries():
    assert linear_address(0, 0, 8, 32) == 0
    assert linear_address(0, 7, 8, 32) == 7
    assert linear_address(31, 0, 8, 32) == 248
    assert linear_address(31, 7, 8, 32) == 255
    assert_raises(KVCacheError, linear_address, 32, 0, 8, 32)
    assert_raises(KVCacheError, linear_address, 0, 8, 8, 32)


def test_kv_cache_provisional_visibility_commit_abort_and_full():
    cache = KVCacheReference(d_head=4, max_seq_len=2)
    cache.append_dim(0, 0, FP16_ONE, FP16_TWO, False, False)
    assert cache.append_incomplete
    assert cache.valid_seq_len == 0
    assert_raises(KVCacheError, cache.read, 0, 0)
    assert_raises(KVCacheError, cache.read, 0, 0, True)

    cache.append_dim(0, 1, FP16_HALF, FP16_NEG_ONE, False, False)
    cache.append_dim(0, 2, FP16_ZERO, FP16_ONE, False, False)
    cache.append_dim(0, 3, FP16_TWO, FP16_HALF, True, True)
    assert cache.valid_seq_len == 0
    assert cache.provisional_valid
    assert cache.provisional_token_index == 0
    assert cache.read(0, 0, include_provisional=True) == (FP16_ONE, FP16_TWO)
    assert_raises(KVCacheError, cache.read, 0, 0)

    cache.commit_provisional()
    assert cache.valid_seq_len == 1
    assert not cache.provisional_valid
    assert cache.read(0, 0) == (FP16_ONE, FP16_TWO)

    cache.append_dim(1, 0, FP16_NEG_ONE, FP16_HALF, False, False)
    assert cache.append_incomplete
    cache.abort_provisional()
    assert cache.valid_seq_len == 1
    assert not cache.append_incomplete
    assert not cache.provisional_valid

    cache.append_token([FP16_ZERO] * 4, [FP16_ONE] * 4)
    assert cache.cache_full
    assert_raises(KVCacheError, cache.append_token, [FP16_ONE] * 4, [FP16_ONE] * 4)


def test_generation_first_token_attends_to_itself():
    ref = GenerationReference(d_head=8, max_seq_len=8, pe_num=8)
    token = token_stream(8, 1)[0]
    trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
    expected = single_head_attention_bit_model(token["q"], [token["k"]], [token["v"]], 8)
    expected_v_fp32 = [fp16_to_fp32_bits(value)["output_bits"] for value in token["v"]]
    assert trace.seq_len_before == 0
    assert trace.seq_len_after == 1
    assert trace.output != [0] * 8
    assert trace.output == expected.output
    assert trace.output == expected_v_fp32
    assert trace.attention_trace.probabilities == [0x3F800000]
    assert trace.cache_k[0] == token["k"]
    assert trace.cache_v[0] == token["v"]


def test_generation_second_token_uses_history_and_current_token():
    d_head = 8
    tokens = token_stream(d_head, 3)
    ref = GenerationReference(d_head=d_head, max_seq_len=8, pe_num=8)

    first = ref.run_token(tokens[0]["q"], tokens[0]["k"], tokens[0]["v"], tokens[0]["meta"])
    assert first.output == single_head_attention_bit_model(
        tokens[0]["q"],
        [tokens[0]["k"]],
        [tokens[0]["v"]],
        8,
    ).output

    second = ref.run_token(tokens[1]["q"], tokens[1]["k"], tokens[1]["v"], tokens[1]["meta"])
    expected_new = single_head_attention_bit_model(
        tokens[1]["q"],
        [tokens[0]["k"], tokens[1]["k"]],
        [tokens[0]["v"], tokens[1]["v"]],
        8,
    )
    old_only = single_head_attention_bit_model(tokens[1]["q"], [tokens[0]["k"]], [tokens[0]["v"]], 8)
    assert second.seq_len_before == 1
    assert second.seq_len_after == 2
    assert second.output == expected_new.output
    assert second.output != old_only.output

    third = ref.run_token(tokens[2]["q"], tokens[2]["k"], tokens[2]["v"], tokens[2]["meta"])
    expected_third = single_head_attention_bit_model(
        tokens[2]["q"],
        [tokens[0]["k"], tokens[1]["k"], tokens[2]["k"]],
        [tokens[0]["v"], tokens[1]["v"], tokens[2]["v"]],
        8,
    )
    assert third.seq_len_before == 2
    assert third.seq_len_after == 3
    assert third.output == expected_third.output


def test_generation_dimensions_and_steps_use_t_plus_one_kv_rows():
    for d_head in [8, 16]:
        for steps in [1, 2, 3, 8]:
            ref = GenerationReference(d_head=d_head, max_seq_len=8, pe_num=8)
            tokens = token_stream(d_head, steps)
            for step, token in enumerate(tokens):
                trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
                assert trace.meta == token["meta"]
                assert trace.seq_len_before == step
                assert trace.seq_len_after == step + 1
                assert len(trace.cache_k) == step + 1
                assert trace.cache_k[step] == token["k"]
                assert trace.cache_v[step] == token["v"]
                expected = single_head_attention_bit_model(
                    token["q"],
                    [prev["k"] for prev in tokens[: step + 1]],
                    [prev["v"] for prev in tokens[: step + 1]],
                    8,
                )
                assert len(trace.attention_trace.probabilities) == step + 1
                assert trace.output == expected.output


def test_generation_cache_full_reports_error_without_overwrite():
    d_head = 8
    ref = GenerationReference(d_head=d_head, max_seq_len=3, pe_num=8)
    tokens = token_stream(d_head, 4)
    for token in tokens[:3]:
        trace = ref.run_token(token["q"], token["k"], token["v"], token["meta"])
        assert not trace.invalid
    before_k, before_v = ref.cache.snapshot()
    full_trace = ref.run_token(tokens[3]["q"], tokens[3]["k"], tokens[3]["v"], tokens[3]["meta"])
    after_k, after_v = ref.cache.snapshot()
    assert full_trace.invalid
    assert full_trace.status == STATUS_CACHE_FULL
    assert full_trace.seq_len_before == 3
    assert full_trace.seq_len_after == 3
    assert full_trace.output == []
    assert before_k == after_k
    assert before_v == after_v
