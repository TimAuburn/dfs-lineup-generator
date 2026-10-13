"""Unit tests for contest preset summary helpers."""

from __future__ import annotations

from src.presets import (
    ContestFormat,
    ContestPreset,
    PRESET_CONFIGS,
    config_with_overrides,
    format_preset_summary,
    format_settings_comparison,
    get_preset_config,
    presets_for_format,
)


def test_showdown_presets_list_cash_before_gpp():
    presets = presets_for_format(ContestFormat.SHOWDOWN)
    assert presets[0] is ContestPreset.SHOWDOWN_CASH
    assert presets[1] is ContestPreset.SHOWDOWN_GPP


def test_format_preset_summary_classic_cash():
    cfg = get_preset_config(ContestPreset.CLASSIC_CASH)
    rows = dict(format_preset_summary(cfg))
    assert rows["Lineup count"].startswith("1 ")
    assert "deterministic" in rows["Projection jitter"].lower() or rows[
        "Projection jitter"
    ].startswith("None")
    assert rows["Stacks"] == "None"
    assert rows["Flex"] == "WR or RB"
    assert rows["Skill salary floor"] == "$3,800"
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
    assert rows["Stacks"] == "QB + WR + 1 WR/TE + skill bring-back"
    assert rows["Flex"] == "WR or RB"
    assert rows["Exclude vs DST"] == "Yes"
    assert rows["Max repeating players"] == "5"
    assert rows["Max player exposure"] == "50%"
    assert "125%" in rows["Ownership cap"]
    assert rows["Tiered salary"].startswith("12 at $49,800–$50,000")
    assert "$49,200–$49,700" in rows["Tiered salary"]


def test_format_preset_summary_showdown_cash_limits_same_team_rbs():
    cfg = get_preset_config(ContestPreset.SHOWDOWN_CASH)
    rows = dict(format_preset_summary(cfg))
    assert rows["Same-team RBs"] == "At most 1"
    gpp = dict(format_preset_summary(get_preset_config(ContestPreset.SHOWDOWN_GPP)))
    assert "Same-team RBs" not in gpp


def test_format_preset_summary_showdown_gpp():
    cfg = get_preset_config(ContestPreset.SHOWDOWN_GPP)
    rows = dict(format_preset_summary(cfg))
    assert cfg.format is ContestFormat.SHOWDOWN
    assert "$48,000" in rows["Salary window"] and "$49,800" in rows["Salary window"]
    assert rows["Min teams"] == "2"
    assert rows["Game-script correlation"] == "Yes"
    assert rows["CPT QB pass-catchers"] == "2"
    assert rows["Max K+DST combined"] == "2"
    assert rows["Max kickers"] == "1"
    assert rows["Max repeating players"] == "3"
    assert rows["Kicker captain"] == "No"


def test_classic_small_gpp_flex_defaults_to_any_and_can_be_rb_or_wr():
    cfg = get_preset_config(ContestPreset.CLASSIC_SMALL_GPP)
    rows = dict(format_preset_summary(cfg))
    assert rows["Flex"] == "Any"
    actual = config_with_overrides(
        cfg,
        lineup_count=cfg.default_lineup_count,
        overrides={"flex_positions": ("RB", "WR")},
    )
    compared = {
        setting: (default, value)
        for setting, default, value in format_settings_comparison(cfg, actual)
    }
    assert compared["Flex"] == ("Any", "WR or RB")


def test_settings_comparison_matches_when_unchanged():
    cfg = get_preset_config(ContestPreset.CLASSIC_SMALL_GPP)
    actual = config_with_overrides(cfg, lineup_count=cfg.default_lineup_count, overrides={})
    rows = format_settings_comparison(cfg, actual)
    assert rows
    assert all(default == actual_value for _, default, actual_value in rows)
    assert "Strategy notes" not in {setting for setting, _, _ in rows}


def test_settings_comparison_shows_lineup_and_stack_overrides():
    cfg = get_preset_config(ContestPreset.CLASSIC_SMALL_GPP)
    actual = config_with_overrides(
        cfg,
        lineup_count=5,
        overrides={
            "enable_bring_back": True,
            "bring_back_position": "TE",
            "jitter_pct": (0.0, 0.0),
            "position_jitter_sds": None,
            "not_a_field": "ignored",
        },
    )
    compared = {
        setting: (default, value)
        for setting, default, value in format_settings_comparison(
            cfg,
            actual,
            extra_rows=[("Max exposure", "Off", "40%")],
        )
    }
    assert compared["Lineup count"][0].startswith("3 ")
    assert compared["Lineup count"][1].startswith("5 ")
    assert "bring-back" not in compared["Stacks"][0]
    assert "TE bring-back" in compared["Stacks"][1]
    assert compared["Projection jitter"][1].startswith("None")
    assert compared["Max exposure"] == ("Off", "40%")
    assert actual.label == cfg.label


def test_settings_comparison_showdown_salary_window():
    cfg = get_preset_config(ContestPreset.SHOWDOWN_GPP)
    actual = config_with_overrides(
        cfg,
        lineup_count=20,
        overrides={"min_salary_cap": None, "max_salary_cap": None, "min_teams": None},
    )
    compared = dict(
        (setting, actual_value)
        for setting, _, actual_value in format_settings_comparison(cfg, actual)
    )
    assert compared["Salary window"] == "Full cap"
    assert compared["Min teams"] == "No minimum"
    assert compared["Lineup count"].startswith("20 ")


def test_format_preset_summary_covers_every_preset():
    for preset, cfg in PRESET_CONFIGS.items():
        rows = format_preset_summary(cfg)
        assert rows, f"{preset} summary should not be empty"
        keys = [k for k, _ in rows]
        assert "Lineup count" in keys
        assert "Projection jitter" in keys
        assert "Max repeating players" in keys
        assert len(keys) == len(set(keys))
