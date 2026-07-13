"""Configuration for the ML-M2 hardware-matched causal LM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareMatchedConfig:
    vocab_size: int = 2048
    context_length: int = 128
    num_layers: int = 1
    d_model: int = 64
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    d_head: int = 8
    d_ffn: int = 256
    rms_norm_eps: float = 1.0e-5
    activation: str = "relu"
    bias: bool = False
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    seed: int = 20260713

    def validate(self) -> None:
        if self.num_layers != 1:
            raise ValueError("ML-M2 accepts exactly one layer")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("ML-M2 does not implement GQA/MQA")
        if self.d_model != self.num_attention_heads * self.d_head:
            raise ValueError("D_MODEL must equal N_HEAD * D_HEAD")
        if self.d_ffn != 4 * self.d_model:
            raise ValueError("D_FFN must equal 4 * D_MODEL")
        if self.bias:
            raise ValueError("ML-M2 hardware-matched model must be bias-free")
        if self.activation != "relu":
            raise ValueError("ML-M2 activation must be ReLU")
        if self.context_length <= 0 or self.vocab_size <= 0:
            raise ValueError("context_length and vocab_size must be positive")

    def to_json_dict(self) -> dict:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json_dict(cls, data: dict) -> "HardwareMatchedConfig":
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "HardwareMatchedConfig":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))
