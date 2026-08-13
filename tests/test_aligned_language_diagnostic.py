import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "scripts"
        / "analyze_aligned_language_responses.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_aligned_language_responses", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def response(index, canonical_id, language, *, capped=False, category="C01"):
    return {
        "id": index,
        "canonical_id": canonical_id,
        "row_id": f"{language.upper()}-{canonical_id}",
        "language": language,
        "direction_class": "unsafe",
        "category_id": category,
        "prompt": "hidden prompt",
        "answer": "hidden answer",
        "generated_tokens": 42,
        "hit_token_cap": capped,
    }


def score(index, *, fulfilled=0.1, substituted=0.1, refusal=0.1):
    return {
        "id": index,
        "p_fulfilled": fulfilled,
        "p_substituted": substituted,
        "p_direct_refusal": refusal,
    }


def test_pairs_by_canonical_id_and_counts_directional_transition():
    module = load_script()
    responses = [
        response(1, "U0001", "ru"),
        response(0, "U0001", "en"),
    ]
    scores = [score(0, fulfilled=0.9), score(1, refusal=0.9)]

    rows, report = module.analyze(responses, scores, ("en", "ru"))

    assert [row["row_id"] for row in rows] == ["EN-U0001", "RU-U0001"]
    assert report["overall"]["transitions"] == {
        "delivered_both": 0,
        "refusal_both": 0,
        "en_delivered_ru_refusal": 1,
        "en_refusal_ru_delivered": 0,
        "unclear": 0,
    }


def test_final_token_cap_forces_unclear_instead_of_false_refusal():
    module = load_script()
    responses = [
        response(0, "U0001", "en", capped=True),
        response(1, "U0001", "ru"),
    ]
    scores = [score(0, refusal=0.9), score(1, fulfilled=0.9)]

    rows, report = module.analyze(responses, scores, ("en", "ru"))

    assert rows[0]["radar_label"] == "unclear"
    assert report["overall"]["transitions"]["unclear"] == 1


def test_rejects_score_coverage_and_translation_metadata_drift():
    module = load_script()
    responses = [response(0, "U0001", "en"), response(1, "U0001", "ru")]
    with pytest.raises(ValueError, match="score id coverage"):
        module.analyze(responses, [score(0, fulfilled=0.9)], ("en", "ru"))

    responses[1]["category_id"] = "C02"
    with pytest.raises(ValueError, match="category_id"):
        module.analyze(
            responses,
            [score(0, fulfilled=0.9), score(1, fulfilled=0.9)],
            ("en", "ru"),
        )
