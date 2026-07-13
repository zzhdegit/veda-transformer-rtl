from pathlib import Path

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.cosim.hardware_aware_model import run_hardware_aware_model
from ml.cosim.trace_compare import compare_logits
from ml.cosim.rtl_vector_builder import write_small_rtl_fixture
from ml.export.export_fp16_weights import export_fp16_weights
from ml.export.export_trace import export_trace
from ml.export.validate_export import validate_export_manifest, validate_linear_weight_direction


def test_linear_weight_direction_is_output_row_major(tmp_path: Path):
    lines = validate_linear_weight_direction(tmp_path)
    assert len(lines) == 6
    assert lines[0] == "3c00"  # fp16(1.0), first output row first input
    assert lines[3] == "4400"  # fp16(4.0), second output row first input


def test_fp16_export_manifest_and_sha(tmp_path: Path):
    torch.manual_seed(10)
    cfg = HardwareMatchedConfig(vocab_size=64, context_length=16)
    model = HardwareMatchedCausalLM(cfg)
    manifest = export_fp16_weights(model, tmp_path)
    assert manifest.tensor_count == 12
    result = validate_export_manifest(tmp_path / "export_manifest.json")
    assert result["tensor_count"] == 12
    names = {record.logical_name for record in manifest.records}
    assert {"norm1_gamma", "wq", "wk", "wv", "wo", "norm2_gamma", "w1", "w2"} <= names


def test_hardware_aware_model_runs_and_reports_error(tmp_path: Path):
    torch.manual_seed(11)
    cfg = HardwareMatchedConfig(vocab_size=64, context_length=16)
    model = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 5]])
    with torch.no_grad():
        pt_logits = model(ids)["logits"]
        hw = run_hardware_aware_model(model, ids)
    assert hw["layer_output"].shape == (1, 2, cfg.d_model)
    assert len(hw["k_cache"]) == cfg.num_attention_heads
    metrics = compare_logits(pt_logits, hw["logits"])
    assert "max_abs_error" in metrics
    assert "top1_agreement" in metrics


def test_trace_manifest_and_small_rtl_fixture(tmp_path: Path):
    torch.manual_seed(12)
    cfg = HardwareMatchedConfig(vocab_size=64, context_length=16)
    model = HardwareMatchedCausalLM(cfg).eval()
    ids = torch.tensor([[1, 5]])
    manifest = export_trace(model, ids, tmp_path / "trace_manifest.json")
    assert manifest["trace_node_count"] >= 20
    assert manifest["cache"]["valid_seq_len"] == 2
    names = {record["name"] for record in manifest["records"]}
    assert {"token_ids", "layer_input", "rmsnorm1_output", "q_projection_fp32", "softmax_probability", "residual2"} <= names
    write_small_rtl_fixture(tmp_path / "rtl_fixture.json", manifest)
    assert (tmp_path / "rtl_fixture.json").exists()

