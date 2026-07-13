"""Audit the CUDA environment for ML-M2 Formal."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as exc:
        return f"ERROR: {exc}"


def cuda_smoke() -> dict:
    if not torch.cuda.is_available():
        return {"status": "fail", "reason": "torch.cuda.is_available() is False"}
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn(512, 512, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(512, 512, device=device, dtype=torch.float32)
    loss = (a @ b).float().pow(2).mean()
    loss.backward()
    opt = torch.optim.AdamW([a], lr=1.0e-4)
    opt.step()
    bf16_ok = False
    if torch.cuda.is_bf16_supported():
        c = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        d = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        e = c @ d
        bf16_ok = e.dtype == torch.bfloat16
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
    return {"status": "ok", "bf16_matmul": bf16_ok, "peak_allocated_bytes": peak}


def main() -> int:
    nvidia_smi = _run(["nvidia-smi"])
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader",
        ]
    )
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "nvidia_smi_query": query.strip(),
        "cuda_smoke": cuda_smoke(),
    }
    out_json = Path("reports/ml_m2/formal_gpu_environment.json")
    out_md = Path("reports/ml_m2/formal_gpu_environment.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# ML-M2 Formal GPU Environment",
                "",
                f"- Python: `{info['python'].splitlines()[0]}`",
                f"- OS: `{info['platform']}`",
                f"- PyTorch: `{info['torch']}`",
                f"- PyTorch CUDA runtime: `{info['torch_cuda_runtime']}`",
                f"- CUDA available: `{info['cuda_available']}`",
                f"- GPU: `{info['device_name']}`",
                f"- Compute capability: `{info['device_capability']}`",
                f"- BF16 supported: `{info['bf16_supported']}`",
                f"- nvidia-smi query: `{info['nvidia_smi_query']}`",
                f"- CUDA smoke: `{info['cuda_smoke']}`",
                "",
                "## nvidia-smi",
                "",
                "```text",
                nvidia_smi.rstrip(),
                "```",
                "",
                "Official PyTorch install pages list CUDA 12.8 as a supported binary compute platform for Windows/Pip; this run used the existing CUDA 12.8 PyTorch environment rather than the base CPU-only environment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not info["cuda_available"] or "RTX 5080" not in info["device_name"] or info["cuda_smoke"]["status"] != "ok":
        print(json.dumps(info, indent=2, sort_keys=True))
        return 1
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
