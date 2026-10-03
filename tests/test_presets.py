"""Unit tests for contest preset summary helpers."""

from __future__ import annotations

from src.presets import (
    ContestFormat,
    ContestPreset,
    PRESET_CONFIGS,
    format_preset_summary,
    get_preset_config,
)


def test_format_preset_summary_classic_cash():
    cfg = get_preset_config(ContestPreset.CLASSIC_CASH)
    rows = dict(format_preset_summary(cfg))
    assert rows["Lineup count"].startswith("1 ")
    assert "deterministic" in rows["Projection jitter"].lower() or rows[
        "Projection jitter"
    ].startswith("None")
    assert rows["Stacks"] == "None"
    assert rows["Exclude vs DST"] == "No"
    assert rows["Max repeating players"] == "No limit"
    assert rows["Ownership cap"] == "None"
    assert "Strategy notes" in rows
    assert cfg.description in rows["Strategy notes"]


def test_format_preset_summary_classic_large_gpp():
    cfg = get_preset_config(ContestPreset.CLASSIC_LARGE_GPP)
    rows = dict(format_preset_summary(cfg))
    assert "20" in rows["Lineup count"]
    assert "Position SDs" in rows["Projection jitter"]
    assert "QB + 2" in rows["Stacks"]
    assert "bring-back" in rows["Stacks"]
    assert rows["Exclude vs DST"] == "Yes"
    assert rows["Max repeating players"] == "5"
    assert "125%" in rows["Ownership cap"]


def test_format_preset_summary_showdown_gpp():
    cfg = get_preset_config(ContestPreset.SHOWDOWN_GPP)
    rows = dict(format_preset_summary(cfg))
    assert cfg.format is ContestFormat.SHOWDOWN
    assert "$48,000" in rows["Salary window"] and "$49,800" in rows["Salary window"]
    assert rows["Min teams"] == "2"
    assert rows["Game-script correlation"] == "Yes"
    assert rows["CPT QB pass-catchers"] == "1"
    assert rows["Max K+DST combined"] == "2"
    assert rows["Max repeating players"] == "4"


def test_format_preset_summary_covers_every_preset():
    for preset, cfg in PRESET_CONFIGS.items():
        rows = format_preset_summary(cfg)
        assert rows, f"{preset} summary should not be empty"
        keys = [k for k, _ in rows]
        assert "Lineup count" in keys
        assert "Projection jitter" in keys
        assert "Max repeating players" in keys
        assert len(keys) == len(set(keys))
