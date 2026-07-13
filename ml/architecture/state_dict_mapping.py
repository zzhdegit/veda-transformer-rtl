"""State dict and export-name mapping for ML-M2."""

from __future__ import annotations


RTL_TENSOR_MAP = {
    "norm1_gamma": "layers.0.norm1.weight",
    "wq": "layers.0.attn.wq.weight",
    "wk": "layers.0.attn.wk.weight",
    "wv": "layers.0.attn.wv.weight",
    "wo": "layers.0.attn.wo.weight",
    "norm2_gamma": "layers.0.norm2.weight",
    "w1": "layers.0.ffn.w1.weight",
    "w2": "layers.0.ffn.w2.weight",
}

SOFTWARE_TENSOR_MAP = {
    "token_embedding": "token_embedding.weight",
    "position_embedding": "position_embedding.weight",
    "final_norm_gamma": "final_norm.weight",
    "lm_head": "lm_head.weight",
}


def required_state_dict_names() -> list[str]:
    return [
        "token_embedding.weight",
        "position_embedding.weight",
        "layers.0.norm1.weight",
        "layers.0.attn.wq.weight",
        "layers.0.attn.wk.weight",
        "layers.0.attn.wv.weight",
        "layers.0.attn.wo.weight",
        "layers.0.norm2.weight",
        "layers.0.ffn.w1.weight",
        "layers.0.ffn.w2.weight",
        "final_norm.weight",
        "lm_head.weight",
    ]


def validate_state_dict_names(state_dict: dict) -> None:
    missing = [name for name in required_state_dict_names() if name not in state_dict]
    if missing:
        raise ValueError(f"missing state_dict entries: {missing}")

