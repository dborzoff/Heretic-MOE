from __future__ import annotations

from pathlib import Path

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
    assert "direction_{lang}_{safe,unsafe}.jsonl" in text
    assert "trial_*_400.jsonl" in text
    assert "built-in cross-model SRG profile" in text
    assert "independent final holdout" in text
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
