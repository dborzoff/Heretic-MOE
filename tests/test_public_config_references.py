from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_FILES = (
    ROOT / "README.md",
    ROOT / "src" / "heretic" / "main.py",
    ROOT / ".gemini" / "styleguide.md",
    ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-08-11-heretic-moe-multilingual-search-v3.md",
)
FORBIDDEN_PUBLIC_REFERENCES = (
    "config.default.toml",
    "--base-config",
    "--data-root",
    "--n-trials",
    "devices.selection",
    "64 x 1,024-token PPL windows",
    "64×1024",
    "direction_safe.jsonl",
    "direction_unsafe.jsonl",
    "search_unsafe.jsonl",
    "prototypes.jsonl",
    "five constraint-feasible candidates",
    "keyword refusal signal",
)


def test_public_docs_and_user_errors_do_not_reference_legacy_config() -> None:
    offenders: list[str] = []
    for path in PUBLIC_TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_PUBLIC_REFERENCES:
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {needle}")

    assert not offenders, chr(10).join(offenders)


def test_public_readme_uses_config_yaml_and_approved_overrides() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "hereticMOE --config config.yaml" in text
    assert "devices.mode: auto" in text
    assert "`map_*_1000.jsonl`" in text
    assert "trial_*_400.jsonl" in text
    assert "`final_*_200.jsonl`" in text
    assert "built-in cross-model SRG profile" in text
    assert "disjoint across the three pools" in text
    assert "TOP-6" in text
    for override in (
        "--model",
        "--run-root",
        "--devices",
        "--target-trials",
        "--exploration-trials",
        "--post-search",
        "--incompatible-contract",
        "--dry-run",
    ):
        assert override in text


def test_qwen3_8b_v4_profile_freezes_the_production_contract() -> None:
    path = ROOT / "config.heretic_moe_4lang_v4_qwen3_8b.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["model"]["path"].endswith("/Qwen__Qwen3-8B")
    assert config["run"]["target_trials"] == 600
    assert config["run"]["exploration_trials"] == 120
    assert config["run"]["post_search"] == "export"
    assert config["devices"]["mode"] == "auto"
    assert config["data"]["languages"] == ["en", "ru", "zh", "ja"]
    assert config["data"]["direction_rows_per_cell"] == 1000
    assert config["data"]["trial_rows_per_cell"] == 400
    assert config["data"]["final_rows_per_cell"] == 200
    assert config["generation"]["ordinary_max_new_tokens"] == 100
    assert config["finalists"]["top_n"] == 6
    assert config["finalists"]["export_roles"] == ["Balanced", "Max"]
