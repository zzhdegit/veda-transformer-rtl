from pathlib import Path

from ml.data.dataset_hash import sha256_text
from ml.data.fixtures import fixture_text
from ml.data.sequence_builder import build_lm_sequences, deterministic_split
from ml.data.tinystories_loader import make_tinystories_manifest, split_tinystories_text
from ml.data.tiny_shakespeare_loader import make_tiny_shakespeare_manifest
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer
from ml.tokenizer.train_bpe import train_tokenizer_from_texts
from ml.tokenizer.validate_tokenizer import validate_tokenizer


def test_tinystories_split_and_manifest_metadata():
    text = "one story<|endoftext|>two story<|endoftext|>"
    assert split_tinystories_text(text) == ["one story", "two story"]
    manifest = make_tinystories_manifest()
    assert manifest.license == "cdla-sharing-1.0"
    assert manifest.revision == "main"


def test_tiny_shakespeare_fixture_manifest():
    manifest = make_tiny_shakespeare_manifest()
    assert manifest.name == "builtin_fixture"
    assert manifest.sha256 == sha256_text(fixture_text())
    assert manifest.num_characters > 100


def test_sequence_builder_packs_next_token_labels():
    batch = build_lm_sequences([1, 10, 11, 12, 2], context_length=4, pad_id=0)
    assert batch.input_ids == [[1, 10, 11, 12]]
    assert batch.labels == [[10, 11, 12, 2]]


def test_deterministic_split_keeps_test_prompts_out_of_train():
    items = [f"doc{i}" for i in range(10)]
    train, val, test = deterministic_split(items, validation_fraction=0.2, test_count=3)
    assert test == ["doc0", "doc1", "doc2"]
    assert not set(test) & set(train)
    assert val


def test_simple_bpe_round_trip_and_determinism():
    texts = [fixture_text()]
    tok1 = SimpleBPETokenizer.train(texts, vocab_size=128)
    tok2 = SimpleBPETokenizer.train(texts, vocab_size=128)
    assert tok1.vocab == tok2.vocab
    assert tok1.merges == tok2.merges
    text = "Lina had a red kite."
    encoded = tok1.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == tok1.bos_id
    assert encoded[-1] == tok1.eos_id
    assert tok1.decode(encoded) == text


def test_tokenizer_save_load_manifest_and_validation(tmp_path: Path):
    tokenizer, manifest = train_tokenizer_from_texts([fixture_text()], tmp_path, vocab_size=128)
    loaded = SimpleBPETokenizer.load(tmp_path / "tokenizer.json")
    assert loaded.encode("A yellow bird") == tokenizer.encode("A yellow bird")
    assert manifest.vocab_size == len(tokenizer.vocab)
    result = validate_tokenizer(tmp_path / "tokenizer.json")
    assert result["vocab_size"] == manifest.vocab_size

