from __future__ import annotations

import json
from pathlib import Path

from heretic.builtin_srg import (
    load_builtin_srg_contract,
    materialize_builtin_srg_runtime,
)
from heretic.multilingual_runtime import resolve_srg_runtime_contract


def test_builtin_srg_is_external_only_and_cross_model_calibrated() -> None:
    contract = load_builtin_srg_contract()

    assert contract["status"] == "PASS"
    assert contract["external_only"] is True
    assert contract["model_count"] >= 2
    assert contract["profile_path"].is_file()
    assert contract["prototype_path"].is_file()
    assert "prompt_path" not in contract
    assert "prompt_rows" not in contract


def test_materialized_builtin_srg_is_portable_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "runtime" / "srg_profile"

    first = materialize_builtin_srg_runtime(destination)
    second = materialize_builtin_srg_runtime(destination)
    resolved = resolve_srg_runtime_contract(destination)

    assert first == second
    assert resolved["external_only"] is True
    assert resolved["profile_path"].parent == destination.resolve()
    assert resolved["prototype_path"].parent == destination.resolve()
    serialized = json.dumps(first, sort_keys=True).lower()
    assert "prompt_path" not in serialized
    assert "prompt_rows" not in serialized

