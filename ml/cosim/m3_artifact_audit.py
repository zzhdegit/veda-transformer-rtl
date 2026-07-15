"""Artifact and weight-mapping audit for ML-M3."""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ml.architecture.causal_lm import HardwareMatchedCausalLM
from ml.architecture.config import HardwareMatchedConfig
from ml.cosim.m3_trace_schema import (
    ARTIFACT_ROOT,
    EXPORT_TENSORS,
    H9_ACCEPTED_COMMIT,
    H9_ACCEPTED_TAG,
    HARDWARE_REPO,
    MODEL_CONFIG_EXPECTED,
    Q2_CHECKPOINT,
    Q2_CHECKPOINT_SHA256,
    Q2_EXPORT_DIR,
    Q2_TOKENIZER,
    Q2_TOKENIZER_SHA256,
    RTL_WEIGHT_SHAPES,
    RTL_WEIGHT_STATE,
    ensure_artifact_dirs,
    read_json,
    sha256_file,
    write_json,
)
from ml.export.formal_export import _torch_load
from ml.inference.incremental_decode import compare_full_vs_incremental
from ml.tokenizer.load_tokenizer import SimpleBPETokenizer


@dataclass(frozen=True)
class LoadedM3Model:
    model: HardwareMatchedCausalLM
    payload: dict[str, Any]
    tokenizer: SimpleBPETokenizer


def _run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def load_q2_model() -> LoadedM3Model:
    payload = _torch_load(Q2_CHECKPOINT)
    cfg = HardwareMatchedConfig.from_json_dict(payload["config"])
    model = HardwareMatchedCausalLM(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tokenizer = SimpleBPETokenizer.load(Q2_TOKENIZER)
    return LoadedM3Model(model=model, payload=payload, tokenizer=tokenizer)


def deterministic_ids(tokenizer: SimpleBPETokenizer, length: int) -> torch.Tensor:
    seed_text = "Once upon a time there was a small red bird who liked kind stories in the garden."
    ids = tokenizer.encode(seed_text, add_bos=True)
    filler = [idx for idx in ids if idx not in {tokenizer.pad_id, tokenizer.eos_id}]
    while len(filler) < length:
        filler.extend(filler[1:] or [tokenizer.bos_id])
    return torch.tensor([filler[:length]], dtype=torch.long)


def _tensor_finite_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    checked = {}
    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            continue
        finite = torch.isfinite(tensor.detach().float())
        checked[name] = {
            "shape": list(tensor.shape),
            "finite": bool(finite.all().item()),
            "nan_count": int(torch.isnan(tensor.detach().float()).sum().item()),
            "inf_count": int(torch.isinf(tensor.detach().float()).sum().item()),
            "min": float(tensor.detach().float().min().item()) if tensor.numel() else 0.0,
            "max": float(tensor.detach().float().max().item()) if tensor.numel() else 0.0,
        }
    return checked


def _export_records_by_name() -> dict[str, dict[str, Any]]:
    manifest = read_json(Q2_EXPORT_DIR / "export_manifest.json")
    return {record["logical_name"]: record for record in manifest["records"]}


def _weight_mapping(state: dict[str, torch.Tensor], export_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for logical_name, state_name in RTL_WEIGHT_STATE.items():
        tensor = state[state_name]
        export_record = export_records[logical_name]
        expected_shape = RTL_WEIGHT_SHAPES[logical_name]
        rows.append(
            {
                "tensor": logical_name,
                "source_state_dict_name": state_name,
                "pytorch_shape": list(tensor.shape),
                "rtl_shape": expected_shape,
                "transpose": bool(export_record["transpose_applied"]),
                "elements": int(tensor.numel()),
                "export_sha256": export_record["sha256"],
                "layout": export_record["rtl_layout"],
                "result": "PASS"
                if list(tensor.shape) == expected_shape
                and int(tensor.numel()) == int(export_record["element_count"])
                and not export_record["transpose_applied"]
                else "FAIL",
            }
        )
    return rows


def _config_check(cfg: HardwareMatchedConfig) -> dict[str, Any]:
    data = cfg.to_json_dict()
    mismatches = {}
    for key, expected in MODEL_CONFIG_EXPECTED.items():
        actual = data.get(key)
        if isinstance(expected, float):
            ok = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        else:
            ok = actual == expected
        if not ok:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {"config": data, "expected": MODEL_CONFIG_EXPECTED, "mismatches": mismatches, "result": "PASS" if not mismatches else "FAIL"}


def run_artifact_audit() -> dict[str, Any]:
    ensure_artifact_dirs()
    checkpoint_sha = sha256_file(Q2_CHECKPOINT)
    tokenizer_sha = sha256_file(Q2_TOKENIZER)
    loaded = load_q2_model()
    state = loaded.model.state_dict()
    export_records = _export_records_by_name()
    export_missing = [name for name in EXPORT_TENSORS if name not in export_records]
    export_extra = sorted(set(export_records) - set(EXPORT_TENSORS))
    finite = _tensor_finite_summary(state)
    finite_pass = all(row["finite"] for row in finite.values())
    mapping = _weight_mapping(state, export_records)
    ids = deterministic_ids(loaded.tokenizer, 16)
    incremental_atol = 5.0e-5
    inc = compare_full_vs_incremental(loaded.model, ids, atol=incremental_atol)
    hardware_status = {
        "worktree_short": _run_git(["status", "--short"], HARDWARE_REPO),
        "branch": _run_git(["branch", "--show-current"], HARDWARE_REPO),
        "head": _run_git(["rev-parse", "HEAD"], HARDWARE_REPO),
        "accepted_tag": H9_ACCEPTED_TAG,
        "accepted_commit": _run_git(["rev-parse", f"{H9_ACCEPTED_TAG}^{{commit}}"], HARDWARE_REPO),
    }
    hardware_status["matches_accepted_tag"] = (
        hardware_status["head"] == hardware_status["accepted_commit"] == H9_ACCEPTED_COMMIT
        and hardware_status["worktree_short"] == ""
    )
    result = {
        "stage": "ML-M3",
        "model_name": "VEDA-HWLM-1L64-Q2",
        "checkpoint": str(Q2_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha_expected": Q2_CHECKPOINT_SHA256,
        "checkpoint_sha_result": "PASS" if checkpoint_sha == Q2_CHECKPOINT_SHA256 else "FAIL",
        "tokenizer": str(Q2_TOKENIZER),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_sha_expected": Q2_TOKENIZER_SHA256,
        "tokenizer_sha_result": "PASS" if tokenizer_sha == Q2_TOKENIZER_SHA256 else "FAIL",
        "config_check": _config_check(loaded.model.config),
        "export_manifest": str(Q2_EXPORT_DIR / "export_manifest.json"),
        "export_tensor_count": len(export_records),
        "export_missing": export_missing,
        "export_extra": export_extra,
        "export_result": "PASS" if not export_missing and not export_extra and len(export_records) == 12 else "FAIL",
        "finite_tensor_summary": finite,
        "finite_result": "PASS" if finite_pass else "FAIL",
        "weight_mapping": mapping,
        "weight_mapping_result": "PASS" if all(row["result"] == "PASS" for row in mapping) else "FAIL",
        "incremental_reference": inc,
        "incremental_reference_atol": incremental_atol,
        "incremental_reference_result": "PASS" if inc["allclose"] and inc["valid_seq_len"] == 16 else "FAIL",
        "hardware_readonly_status": hardware_status,
        "hardware_readonly_result": "PASS" if hardware_status["matches_accepted_tag"] else "FAIL",
    }
    result["overall_result"] = "PASS" if all(
        result[key] == "PASS"
        for key in [
            "checkpoint_sha_result",
            "tokenizer_sha_result",
            "export_result",
            "finite_result",
            "weight_mapping_result",
            "incremental_reference_result",
            "hardware_readonly_result",
        ]
    ) and result["config_check"]["result"] == "PASS" else "FAIL"
    write_json(ARTIFACT_ROOT / "manifests" / "artifact_audit.json", result)
    _write_artifact_reports(result)
    return result


def _write_artifact_reports(result: dict[str, Any]) -> None:
    reports = Path("reports/ml_m3")
    reports.mkdir(parents=True, exist_ok=True)
    artifact_lines = [
        "# ML-M3 Artifact Audit",
        "",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Checkpoint SHA256: `{result['checkpoint_sha256']}` ({result['checkpoint_sha_result']})",
        f"- Tokenizer: `{result['tokenizer']}`",
        f"- Tokenizer SHA256: `{result['tokenizer_sha256']}` ({result['tokenizer_sha_result']})",
        f"- Export tensor count: {result['export_tensor_count']} ({result['export_result']})",
        f"- Hardware read-only branch: `{result['hardware_readonly_status']['branch']}`",
        f"- Hardware HEAD: `{result['hardware_readonly_status']['head']}`",
        f"- Hardware accepted tag commit: `{result['hardware_readonly_status']['accepted_commit']}`",
        f"- Incremental/full reference: {result['incremental_reference_result']} max_abs={result['incremental_reference']['max_abs_error']}",
        f"- Overall: **{result['overall_result']}**",
        "",
        "No checkpoint, tokenizer, dataset, trace, waveform, or hardware source file is written to Git by this audit.",
        "",
    ]
    (reports / "artifact_audit.md").write_text("\n".join(artifact_lines), encoding="utf-8")
    mapping_lines = [
        "# ML-M3 Weight Mapping Audit",
        "",
        "RTL layer weights use `weight[output_index][input_index]`. Embedding, learned position, final RMSNorm, and tied LM head remain software-side.",
        "",
        "| Tensor | PyTorch Shape | RTL Shape | Transpose | Elements | SHA256 | Result |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for row in result["weight_mapping"]:
        mapping_lines.append(
            f"| {row['tensor']} | {row['pytorch_shape']} | {row['rtl_shape']} | {row['transpose']} | "
            f"{row['elements']} | `{row['export_sha256']}` | {row['result']} |"
        )
    mapping_lines.extend(["", f"Overall: **{result['weight_mapping_result']}**", ""])
    (reports / "weight_mapping_audit.md").write_text("\n".join(mapping_lines), encoding="utf-8")


def main() -> None:
    payload = run_artifact_audit()
    print(payload["overall_result"])


if __name__ == "__main__":
    main()
