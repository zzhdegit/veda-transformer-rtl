from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


RTL_FILES = [
    ROOT / "rtl/common/stream_reg.sv",
    ROOT / "rtl/common/skid_buffer.sv",
    ROOT / "rtl/memory/sync_fifo.sv",
    ROOT / "rtl/memory/sram_1p_wrapper.sv",
    ROOT / "rtl/memory/sram_2p_wrapper.sv",
    ROOT / "rtl/arithmetic/mul_unit.sv",
    ROOT / "rtl/arithmetic/add_unit.sv",
    ROOT / "rtl/arithmetic/mac_unit.sv",
    ROOT / "rtl/arithmetic/compare_max.sv",
    ROOT / "rtl/arithmetic/round_sat.sv",
]


def read_all_rtl() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in RTL_FILES)


def test_expected_rtl_files_exist():
    for path in RTL_FILES:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_no_real_or_shortreal_in_synthesizable_rtl():
    text = read_all_rtl()
    assert "shortreal" not in text
    assert " real " not in f" {text} "


def test_required_modules_are_declared():
    text = read_all_rtl()
    for module in [
        "stream_reg",
        "skid_buffer",
        "sync_fifo",
        "sram_1p_wrapper",
        "sram_2p_wrapper",
        "mul_unit",
        "add_unit",
        "mac_unit",
        "compare_max",
        "round_sat",
    ]:
        assert f"module {module}" in text


def test_required_assertion_coverage_is_present():
    text = read_all_rtl()
    for phrase in [
        "valid_stable_until_ready",
        "data_stable_until_ready",
        "metadata_stable_until_ready",
        "no_fifo_write_when_full",
        "no_fifo_read_when_empty",
        "fifo_occupancy_in_range",
        "transaction_count_conserved",
        "no_unknown_output_when_valid",
        "$fatal",
    ]:
        assert phrase in text


def test_round_sat_nearest_even_uses_sign_extended_magnitude():
    text = (ROOT / "rtl/arithmetic/round_sat.sv").read_text(encoding="utf-8")
    assert "input_ext_bits = {in_data[IN_W-1], in_data}" in text
    assert "~input_ext_bits" in text
    assert "~{1'b0, in_data}" not in text


def test_no_pdk_or_macro_artifacts_added_to_repo():
    banned_suffixes = {".lib", ".lef", ".gds", ".gdsii", ".db", ".ndm", ".nlib"}
    for path in ROOT.rglob("*"):
        if any(part in {"build", ".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        assert path.suffix.lower() not in banned_suffixes, path
