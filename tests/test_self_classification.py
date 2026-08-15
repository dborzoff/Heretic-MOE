from __future__ import annotations

import pytest

from heretic.self_classification import (
    BehaviorClass,
    ClassificationInput,
    ClassificationResult,
    PromptVariant,
    classify_output_shape,
    parse_classification_output,
    permuted_code_map,
    render_classifier_prompt,
    variant_option_order,
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


def test_four_code_shifts_assign_every_class_to_every_code_once() -> None:
    variants = (
        PromptVariant.CODE_PERMUTED,
        PromptVariant.CODE_SHIFT_1,
        PromptVariant.CODE_SHIFT_2,
        PromptVariant.CODE_SHIFT_3,
    )

    for behavior in BehaviorClass:
        outputs = {
            render_classifier_prompt(sample_row(), variant).expected_outputs[behavior]
            for variant in variants
        }
        assert outputs == {"A", "B", "C", "D"}


def test_word_order_variants_rotate_positions_without_changing_labels() -> None:
    variants = (
        PromptVariant.WORD_ORDER_0,
        PromptVariant.WORD_ORDER_1,
        PromptVariant.WORD_ORDER_2,
        PromptVariant.WORD_ORDER_3,
    )
    expected_orders = (
        tuple(BehaviorClass),
        (
            BehaviorClass.PARTIAL,
            BehaviorClass.SOFT,
            BehaviorClass.HARD_REFUSE,
            BehaviorClass.DIRECT,
        ),
        (
            BehaviorClass.SOFT,
            BehaviorClass.HARD_REFUSE,
            BehaviorClass.DIRECT,
            BehaviorClass.PARTIAL,
        ),
        (
            BehaviorClass.HARD_REFUSE,
            BehaviorClass.DIRECT,
            BehaviorClass.PARTIAL,
            BehaviorClass.SOFT,
        ),
    )

    for variant, expected_order in zip(variants, expected_orders, strict=True):
        rendered = render_classifier_prompt(sample_row(), variant)
        assert variant_option_order(variant) == expected_order
        assert rendered.expected_outputs == {
            behavior: behavior.value for behavior in BehaviorClass
        }


def test_word_parser_accepts_only_one_exact_class_word() -> None:
    assert (
        parse_classification_output(
            "HARD_REFUSE",
            "EN-0001",
            PromptVariant.WORD_ORDER_2,
            "en",
        )
        == BehaviorClass.HARD_REFUSE
    )
    assert (
        parse_classification_output(
            "HARD_REFUSE because",
            "EN-0001",
            PromptVariant.WORD_ORDER_2,
            "en",
        )
        is None
    )


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


def test_english_system_mode_keeps_ru_request_but_uses_shared_meta_instruction() -> None:
    ru = render_classifier_prompt(
        sample_row("ru"),
        PromptVariant.CODE_PERMUTED,
        system_mode="english",
        max_new_tokens=16,
    )
    en = render_classifier_prompt(
        sample_row("en"),
        PromptVariant.CODE_PERMUTED,
        system_mode="english",
    )

    assert ru.system == en.system
    assert "PRIVATE_PROMPT_SENTINEL" in ru.user
    assert ru.max_new_tokens == 16


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "exact"),
        ('"A"', "quoted"),
        ('{"label":"A"}', "json_single_label"),
        ("A.", "label_punctuation"),
        ("A because", "label_plus_text"),
        ("unknown", "unknown"),
    ],
)
def test_output_shape_classifies_format_without_retaining_text(
    text: str,
    expected: str,
) -> None:
    rendered = render_classifier_prompt(sample_row(), PromptVariant.CODE_PERMUTED)

    assert classify_output_shape(text, rendered.expected_outputs.values()) == expected
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
    assert public["output_shape"] is None


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
