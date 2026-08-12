# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare the clean R reference only after TOP-6 has been frozen."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE prepare-final-holdout")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--top-six-manifest", required=True, type=Path)
    parser.add_argument("--device", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if not args.config.is_file() or not args.top_six_manifest.is_file():
        raise FileNotFoundError("final-holdout preparation inputs are incomplete")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    from .config import Settings, generation_runtime_contract
    from .language_map_directions import load_direction_map_package
    from .model import Model
    from .multilingual_contract import load_multilingual_dataset_bundle
    from .multilingual_final_holdout import build_final_holdout_archive
    from .multilingual_prepare import fingerprint_local_model
    from .multilingual_runtime import (
        apply_multilingual_search_mode,
        build_multilingual_srg_scorer,
    )
    from .utils import Prompt

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    multilingual = config.get("multilingual_search")
    if not isinstance(multilingual, dict) or not multilingual.get("enabled"):
        raise ValueError("config does not enable multilingual search v3")
    runtime_root = args.runtime_root.resolve()
    multilingual["runtime_root"] = runtime_root.as_posix()
    config["device_map"] = "auto"
    settings = Settings.model_validate(config)
    apply_multilingual_search_mode(settings)
    contract = settings.multilingual_search
    bundle = load_multilingual_dataset_bundle(
        dataset_root=str(contract.dataset_root),
        split_root=contract.split_root,
        languages=tuple(contract.languages),
        direction_rows_per_cell=contract.direction_rows_per_cell,
        trial_rows_per_cell=contract.trial_rows_per_cell,
        calibration_rows_per_language=contract.calibration_rows_per_language,
    )
    top_six = json.loads(args.top_six_manifest.read_text(encoding="utf-8"))
    top_six_sha = str(top_six.get("shortlist_contract_sha256", ""))
    if top_six.get("status") != "FROZEN" or len(top_six_sha) != 64:
        raise ValueError("TOP-6 manifest is not frozen")
    model_path = Path(settings.model)
    if not model_path.is_dir():
        raise ValueError("final-holdout preparation requires a local model directory")
    model_fingerprint = fingerprint_local_model(model_path)["model_fingerprint"]
    model = Model(settings)
    model.prepare_prompt_cache(
        [Prompt(system="", user=row.prompt) for row in bundle.final_rows]
    )
    model.pin_prompt_cache()
    scorer = build_multilingual_srg_scorer(settings, model, runtime_root)
    profile, _ = load_direction_map_package(runtime_root / "clean_map" / "directions")
    srg_profile = json.loads(
        (runtime_root / "srg_calibration" / "calibration_profile.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_final_holdout_archive(
        model=model,
        rows=bundle.final_rows,
        refusal_direction=profile.consensus_refusal_direction,
        srg_scorer=scorer,
        srg_profile=srg_profile,
        output_dir=runtime_root / "final_holdout_reference",
        dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
        model_fingerprint=str(model_fingerprint),
        top_six_contract_sha256=top_six_sha,
        max_response_length=contract.final_max_new_tokens,
        generation_contract=generation_runtime_contract(settings),
    )
    print(
        json.dumps(
            {
                "event": "final_holdout_reference_ready",
                "status": "PASS",
                "rows": manifest["rows"],
                "archive_contract_sha256": manifest["archive_contract_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return manifest
