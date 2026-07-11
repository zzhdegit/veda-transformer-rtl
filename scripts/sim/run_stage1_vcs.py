import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    script = ROOT / "scripts" / "sim" / "run_stage1_vcs.sh"
    proc = subprocess.run(["bash", str(script)], cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
