import hashlib
import json
from pathlib import Path

import torch

from heretic.language_map_directions import (
    build_direction_map_profile,
    write_direction_map_package,
)


LANGUAGES = ("en", "ru", "zh", "es", "fr")


def _synthetic_map() -> tuple[list[dict[str, object]], torch.Tensor]:
    rows: list[dict[str, object]] = []
    values: list[torch.Tensor] = []
    language_offsets = {
        language: float(index - 2) for index, language in enumerate(LANGUAGES)
    }
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for number in range(8):
            canonical_id = f"{prefix}{number + 1:04d}"
            category = "C01" if number < 6 else "C02"
            for language in LANGUAGES:
                rows.append(
                    {
                        "canonical_id": canonical_id,
                        "row_id": f"{language.upper()}-{canonical_id}",
                        "language": language,
                        "direction_class": direction,
                        "category_id": category,
                    }
                )
                semantic = number * 0.01
                direction_shift = 2.0 if direction == "unsafe" else 0.0
                category_shift = 0.3 if direction == "unsafe" and category == "C02" else 0.0
                layer0 = torch.tensor(
                    [language_offsets[language], direction_shift, category_shift, semantic]
                )
                layer1 = torch.tensor(
                    [0.5 * language_offsets[language], 1.5 * direction_shift, 0.0, semantic]
                )
                values.append(torch.stack((layer0, layer1)))
    return rows, torch.stack(values).float()


def test_profile_separates_language_axis_from_direction_axis() -> None:
    index, residuals = _synthetic_map()

    profile = build_direction_map_profile(
        index,
        residuals,
        languages=LANGUAGES,
        explained_variance_target=0.90,
    )

    assert profile.consensus_refusal_direction.shape == (2, 4)
    assert abs(float(profile.consensus_refusal_direction[0, 0])) < 1e-5
    assert float(profile.consensus_refusal_direction[0, 1]) > 0.99
    assert float(profile.consensus_refusal_direction[1, 1]) > 0.99
    assert profile.language_subspace.shape == (2, 4, 4)
    assert profile.language_ranks.tolist() == [1, 1]
    assert bool(((profile.layer_reliability >= 0) & (profile.layer_reliability <= 1)).all())
    assert bool((profile.layer_reliability > 0).all())
    assert profile.recommended_layer_bounds == (0, 1)
    assert profile.category_ids == ("C01", "C02")
    assert profile.category_branch_directions.shape == (2, 2, 4)


def test_direction_package_is_atomic_hashed_and_text_free(tmp_path: Path) -> None:
    index, residuals = _synthetic_map()
    profile = build_direction_map_profile(index, residuals, languages=LANGUAGES)

    manifest = write_direction_map_package(profile, tmp_path / "directions")

    assert manifest["status"] == "PASS"
    assert manifest["recommended_layer_bounds"] == [0, 1]
    serialized = json.dumps(manifest, sort_keys=True)
    assert not any(word in serialized for word in ("prompt", "response", "answer"))
    for name in ("directions.safetensors", "manifest.json"):
        path = tmp_path / "directions" / name
        assert path.is_file()
    tensor_path = tmp_path / "directions" / "directions.safetensors"
    assert hashlib.sha256(tensor_path.read_bytes()).hexdigest() == manifest["files"][
        "directions.safetensors"
    ]["sha256"]
