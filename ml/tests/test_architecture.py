import torch

from ml.architecture.attention import KVCache
from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.architecture.feed_forward import FeedForward
from ml.architecture.rmsnorm import RMSNorm
from ml.architecture.state_dict_mapping import required_state_dict_names, validate_state_dict_names
from ml.architecture.transformer_layer import TransformerLayer


def tiny_config(vocab_size=64, context_length=16):
    return HardwareMatchedConfig(vocab_size=vocab_size, context_length=context_length)


def test_config_contract():
    cfg = tiny_config()
    cfg.validate()
    assert cfg.d_model == cfg.num_attention_heads * cfg.d_head
    assert cfg.d_ffn == 4 * cfg.d_model
    assert cfg.num_key_value_heads == cfg.num_attention_heads


def test_model_shapes_and_no_bias():
    torch.manual_seed(1)
    cfg = tiny_config()
    model = HardwareMatchedCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    out = model(ids)
    assert out["logits"].shape == (2, 5, cfg.vocab_size)
    for module in [model.layers[0].attn.wq, model.layers[0].attn.wk, model.layers[0].attn.wv, model.layers[0].attn.wo, model.layers[0].ffn.w1, model.layers[0].ffn.w2, model.lm_head]:
        assert module.bias is None


def test_rmsnorm_matches_manual_formula():
    norm = RMSNorm(4, eps=1e-5)
    norm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * norm.weight
    assert torch.allclose(norm(x), expected)


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(2)
    cfg = tiny_config()
    model = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 5, 6, 7, 8]])
    ids_changed = ids.clone()
    ids_changed[:, 3:] = torch.tensor([[20, 21]])
    with torch.no_grad():
        logits_a = model(ids)["logits"]
        logits_b = model(ids_changed)["logits"]
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)


def test_current_token_is_visible():
    torch.manual_seed(3)
    cfg = tiny_config()
    layer = TransformerLayer(cfg).eval()
    x = torch.randn(1, 1, cfg.d_model)
    with torch.no_grad():
        full, _ = layer(x, use_cache=False)
        inc, cache = layer(x, use_cache=True)
    assert cache is not None
    assert cache.valid_seq_len == 1
    assert torch.allclose(full, inc, atol=1e-6)


def test_residual_order_with_zero_attention_and_ffn():
    cfg = tiny_config()
    layer = TransformerLayer(cfg).eval()
    for param in layer.attn.parameters():
        param.data.zero_()
    for param in layer.ffn.parameters():
        param.data.zero_()
    x = torch.randn(2, 3, cfg.d_model)
    with torch.no_grad():
        y, _ = layer(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_ffn_relu_clamps_negative_values():
    cfg = tiny_config()
    ffn = FeedForward(cfg).eval()
    ffn.w1.weight.data.fill_(-1.0)
    ffn.w2.weight.data.fill_(1.0)
    x = torch.ones(1, 1, cfg.d_model)
    with torch.no_grad():
        y = ffn(x)
    assert torch.count_nonzero(y) == 0


def test_state_dict_names_and_weight_tying():
    cfg = tiny_config()
    model = HardwareMatchedCausalLM(cfg)
    names = set(model.state_dict().keys())
    for name in required_state_dict_names():
        assert name in names
    validate_state_dict_names(model.state_dict())
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_deterministic_forward():
    cfg = tiny_config()
    torch.manual_seed(4)
    model_a = HardwareMatchedCausalLM(cfg).eval()
    torch.manual_seed(4)
    model_b = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(model_a(ids)["logits"], model_b(ids)["logits"])


def test_incremental_kv_matches_full_sequence():
    torch.manual_seed(5)
    cfg = tiny_config(context_length=16)
    model = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 10, 11, 12, 2]])
    with torch.no_grad():
        full = model(ids)["logits"]
        cache = None
        parts = []
        for pos in range(ids.shape[1]):
            out = model(ids[:, pos:pos + 1], past_kv=cache, use_cache=True, start_pos=pos)
            cache = out["past_kv"]
            parts.append(out["logits"])
        inc = torch.cat(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-5)
    assert cache[0].valid_seq_len == ids.shape[1]


def test_one_and_multi_token_greedy_decode_and_cache_reset():
    torch.manual_seed(6)
    cfg = tiny_config(context_length=16)
    model = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 5]])
    with torch.no_grad():
        generated = model.generate_greedy(ids, max_new_tokens=2, eos_token_id=None)
        out = model(ids[:, :1], use_cache=True)
    assert generated.shape[1] == 4
    cache: KVCache = out["past_kv"][0]
    assert cache.valid_seq_len == 1
    cache.reset()
    assert cache.valid_seq_len == 0

