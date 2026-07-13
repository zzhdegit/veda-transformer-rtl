"""Deterministic pure-Python BPE tokenizer for ML-M2."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass(frozen=True)
class TokenizerStats:
    vocab_size: int
    merge_count: int
    document_count: int
    average_encoded_length: float
    unk_tokens: int
    total_tokens: int


class SimpleBPETokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.merges = list(merges)
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._piece_cache: dict[str, list[str]] = {}
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 2048, min_pair_frequency: int = 2) -> "SimpleBPETokenizer":
        if vocab_size < len(SPECIAL_TOKENS) + 1:
            raise ValueError("vocab_size too small")
        words_counter: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for piece in re.findall(r"\S+|\s+", text):
                if piece:
                    words_counter[tuple(piece)] += 1
        chars = sorted({ch for word in words_counter for ch in word})
        vocab: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        for ch in chars:
            if ch not in vocab and len(vocab) < vocab_size:
                vocab[ch] = len(vocab)
        merges: list[tuple[str, str]] = []

        while len(vocab) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in words_counter.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            if best_count < min_pair_frequency:
                break
            merged = "".join(best_pair)
            if merged in vocab:
                break
            vocab[merged] = len(vocab)
            merges.append(best_pair)
            new_counter: Counter[tuple[str, ...]] = Counter()
            for word, freq in words_counter.items():
                new_counter[tuple(cls._merge_word(list(word), best_pair, merged))] += freq
            words_counter = new_counter
        return cls(vocab, merges, SPECIAL_TOKENS)

    @staticmethod
    def _merge_word(word: list[str], pair: tuple[str, str], merged: str) -> list[str]:
        out: list[str] = []
        idx = 0
        while idx < len(word):
            if idx + 1 < len(word) and word[idx] == pair[0] and word[idx + 1] == pair[1]:
                out.append(merged)
                idx += 2
            else:
                out.append(word[idx])
                idx += 1
        return out

    def _tokenize_piece(self, piece: str) -> list[str]:
        cached = self._piece_cache.get(piece)
        if cached is not None:
            return list(cached)
        if piece in self.vocab:
            self._piece_cache[piece] = [piece]
            return [piece]
        pieces = list(piece)
        while len(pieces) > 1:
            pairs = list(zip(pieces, pieces[1:]))
            ranked = [(self._merge_ranks[pair], pair) for pair in pairs if pair in self._merge_ranks]
            if not ranked:
                break
            _, best = min(ranked, key=lambda item: item[0])
            pieces = self._merge_word(pieces, best, best[0] + best[1])
        self._piece_cache[piece] = list(pieces)
        return pieces

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for piece in re.findall(r"\S+|\s+", text):
            tokens.extend(self._tokenize_piece(piece))
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.vocab.get(piece, self.unk_id) for piece in self.tokenize(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if skip_special and token in self.special_tokens:
                continue
            pieces.append(token if token != "<unk>" else "")
        return "".join(pieces)

    def stats(self, texts: list[str]) -> TokenizerStats:
        lengths = []
        unk_tokens = 0
        total_tokens = 0
        for text in texts:
            encoded = self.encode(text, add_bos=True, add_eos=True)
            lengths.append(len(encoded))
            unk_tokens += sum(1 for token_id in encoded if token_id == self.unk_id)
            total_tokens += len(encoded)
        average = float(sum(lengths)) / len(lengths) if lengths else 0.0
        return TokenizerStats(
            vocab_size=len(self.vocab),
            merge_count=len(self.merges),
            document_count=len(texts),
            average_encoded_length=average,
            unk_tokens=unk_tokens,
            total_tokens=total_tokens,
        )

    def to_json_dict(self) -> dict:
        return {
            "type": "simple_bpe",
            "version": 1,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": [[left, right] for left, right in self.merges],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json_dict(cls, data: dict) -> "SimpleBPETokenizer":
        if data.get("type") != "simple_bpe":
            raise ValueError("unsupported tokenizer type")
        return cls(
            vocab={str(token): int(idx) for token, idx in data["vocab"].items()},
            merges=[(str(left), str(right)) for left, right in data["merges"]],
            special_tokens=list(data.get("special_tokens", SPECIAL_TOKENS)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleBPETokenizer":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_tokenizer(path: str | Path) -> SimpleBPETokenizer:
    return SimpleBPETokenizer.load(path)
