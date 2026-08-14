from __future__ import annotations

import pytest

from heretic.self_classification import (
    BehaviorClass,
    ClassificationInput,
    ClassificationResult,
    PromptVariant,
    parse_classification_output,
    permuted_code_map,
    render_classifier_prompt,
)


def sample_row(language: str = "en") -> ClassificationInput:
    return ClassificationInput(
        canonical_id="P0001",
        row_id="EN-0001",
        language=language,
        category_ids=("S1",),
        direction_class="unsafe",
        prompt="PRIVATE_PROMPT_SENTINEL",
    )


def test_code_permutation_is_a_deterministic_bijection() -> None:
    expected = {
        "A": BehaviorClass.PARTIAL,
        "B": BehaviorClass.HARD_REFUSE,
        "C": BehaviorClass.DIRECT,
        "D": BehaviorClass.SOFT,
    }

    assert permuted_code_map("EN-0001") == expected
    assert permuted_code_map("EN-0001") == expected
    assert set(expected.values()) == set(BehaviorClass)


@pytest.mark.parametrize("language", ["en", "ru", "zh", "ko"])
@pytest.mark.parametrize("variant", list(PromptVariant))
def test_rendered_prompt_is_localized_and_keeps_source_in_memory(
    language: str, variant: PromptVariant
) -> None:
    rendered = render_classifier_prompt(sample_row(language), variant)

    assert "PRIVATE_PROMPT_SENTINEL" in rendered.user
    assert rendered.system
    assert rendered.max_new_tokens > 0
    assert rendered.variant is variant


def test_phrase_parser_accepts_only_one_exact_localized_phrase() -> None:
    row = sample_row("en")
    rendered = render_classifier_prompt(row, PromptVariant.PHRASE)
    exact = rendered.expected_outputs[BehaviorClass.SOFT]

    parsed = parse_classification_output(exact, row.row_id, PromptVariant.PHRASE, "en")
    rejected = parse_classification_output(
        exact + " Extra explanation.", row.row_id, PromptVariant.PHRASE, "en"
    )

    assert parsed == BehaviorClass.SOFT
    assert rejected is None


def test_number_parser_uses_fixed_behavior_mapping() -> None:
    assert (
        parse_classification_output("1", "EN-0001", PromptVariant.NUMBER, "en")
        == BehaviorClass.DIRECT
    )
    assert (
        parse_classification_output("4", "EN-0001", PromptVariant.NUMBER, "en")
        == BehaviorClass.HARD_REFUSE
    )
    assert (
        parse_classification_output("4 because safety", "EN-0001", PromptVariant.NUMBER, "en")
        is None
    )


def test_code_parser_decodes_row_specific_permutation() -> None:
    assert (
        parse_classification_output(
            "A", "EN-0001", PromptVariant.CODE_PERMUTED, "en"
        )
        == BehaviorClass.PARTIAL
    )
    assert (
        parse_classification_output(
            "C", "EN-0001", PromptVariant.CODE_PERMUTED, "en"
        )
        == BehaviorClass.DIRECT
    )


def test_public_result_contains_no_prompt_or_raw_output() -> None:
    result = ClassificationResult(
        model_id="model-a",
        canonical_id="P0001",
        row_id="EN-0001",
        language="en",
        category_ids=("S1",),
        direction_class="unsafe",
        variant=PromptVariant.NUMBER,
        classification=BehaviorClass.HARD_REFUSE,
        valid=True,
        output_tokens=1,
    )

    public = result.to_public_dict()

    assert public["classification"] == "HARD_REFUSE"
    assert "prompt" not in public
    assert "response" not in public
    assert "raw_output" not in public


def test_invalid_result_cannot_claim_a_behavior_class() -> None:
    with pytest.raises(ValueError, match="invalid result"):
        ClassificationResult(
            model_id="model-a",
            canonical_id="P0001",
            row_id="EN-0001",
            language="en",
            category_ids=("S1",),
            direction_class="unsafe",
            variant=PromptVariant.NUMBER,
            classification=BehaviorClass.DIRECT,
            valid=False,
            output_tokens=3,
        )
