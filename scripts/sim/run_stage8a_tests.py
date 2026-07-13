#!/usr/bin/env python3
"""Stage 8A documentation and paper-evidence checks."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_FILES = [
    ROOT / "docs/stage_08/paper_evidence.md",
    ROOT / "docs/stage_08/spec.md",
    ROOT / "reports/stage_08/phase_8a_spec.md",
]

REQUIRED_EVIDENCE_TERMS = [
    "8*8*2 Reconfigurable PEs",
    "Section IV-A",
    "Figure 4",
    "Figure 5",
    "Figure 7",
    "Table I",
    "repository design decisions",
]

REQUIRED_SPEC_TERMS = [
    "8 rows x 8 columns x 2 PE groups = 128 physical PE cells",
    "MODE_INNER_PRODUCT",
    "MODE_OUTER_PRODUCT",
    "QK complete -> existing Softmax/SFU -> sV",
    "SFU-PE interleaving is out of scope for Stage 8",
    "Projection PE and FFN PE remain legacy",
]


def check_file(path):
    if not path.exists():
        return ["missing file: %s" % path.relative_to(ROOT)]
    if path.stat().st_size == 0:
        return ["empty file: %s" % path.relative_to(ROOT)]
    return []


def check_terms(path, terms):
    text = path.read_text(encoding="utf-8")
    errors = []
    for term in terms:
        if term not in text:
            errors.append("%s missing term: %s" % (path.relative_to(ROOT), term))
    return errors


def main():
    errors = []
    for path in REQUIRED_FILES:
        errors.extend(check_file(path))

    if not errors:
        errors.extend(check_terms(ROOT / "docs/stage_08/paper_evidence.md", REQUIRED_EVIDENCE_TERMS))
        errors.extend(check_terms(ROOT / "docs/stage_08/spec.md", REQUIRED_SPEC_TERMS))

    if errors:
        print("Stage 8A paper/spec checks: FAIL")
        for error in errors:
            print("ERROR: %s" % error)
        return 1

    print("Stage 8A paper/spec checks: PASS")
    print("checked_files=%d" % len(REQUIRED_FILES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
