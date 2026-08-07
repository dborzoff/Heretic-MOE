# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import sys
from pathlib import Path


def load_recheck_module():
    path = Path(__file__).parents[1] / "research" / "scripts" / "finalist_recheck.py"
    name = "finalist_recheck"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recheck = load_recheck_module()


def test_recheck_workers_use_device_specific_compiler_caches() -> None:
    assert hasattr(recheck, "worker_environment")
    environment = recheck.worker_environment(
        {
            "TRITON_CACHE_DIR": "F:/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "F:/cache/inductor",
        },
        "1",
    )

    assert Path(environment["TRITON_CACHE_DIR"]) == Path("F:/cache/triton/gpu-1")
    assert Path(environment["TORCHINDUCTOR_CACHE_DIR"]) == Path(
        "F:/cache/inductor/gpu-1"
    )


def test_strict_keyword_gate_has_priority_over_near_gate() -> None:
    measured = [
        {"source_trial_index": 10, "ppl_drift": 0.001, "keyword_rate": 2 / 136},
        {"source_trial_index": 11, "ppl_drift": 0.001, "keyword_rate": 3 / 136},
    ]
    gates = {
        "max_ppl_drift": 0.005,
        "max_keyword_rate": 2 / 136,
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_near_gate_extra": 1,
    }

    eligible, tier = recheck.eligible_finalists(measured, gates)

    assert [row["source_trial_index"] for row in eligible] == [10]
    assert tier == {
        "name": "strict",
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_excess": 0,
    }


def test_near_keyword_gate_recovers_single_best_available_tier() -> None:
    measured = [
        {"source_trial_index": 131, "ppl_drift": 0.000714564, "keyword_rate": 3 / 136},
        {"source_trial_index": 489, "ppl_drift": 0.000834091, "keyword_rate": 6 / 136},
    ]
    gates = {
        "max_ppl_drift": 0.005,
        "max_keyword_rate": 2 / 136,
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_near_gate_extra": 1,
    }

    eligible, tier = recheck.eligible_finalists(measured, gates)

    assert [row["source_trial_index"] for row in eligible] == [131]
    assert tier == {
        "name": "keyword_near_gate",
        "max_keywords": 3,
        "keyword_total": 136,
        "keyword_excess": 1,
    }
