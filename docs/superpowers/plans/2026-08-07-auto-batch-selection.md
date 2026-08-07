# Automatic Batch Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the existing measured automatic selector through batch size 4096 while retaining at least 10% of total VRAM.

**Architecture:** Keep the current generation benchmark, VRAM-reserve check, and OOM fallback. Change only its default ceiling and the maintained profile inputs that currently bypass it.

**Tech Stack:** Python 3.12, Pydantic settings, TOML, pytest.

## Global Constraints

- Do not inspect prompt or response contents.
- Preserve the old fixed-batch Qwen journal and restart in a separate run root.
- Do not introduce new dependencies or a second batch-selection algorithm.

---

### Task 1: Restore automatic batching

**Files:**
- Modify: `src/heretic/config.py`
- Modify: `research/configs/adaptive_search/ministral3_sparse_geometry.toml`
- Modify: `research/configs/adaptive_search/gemma4_e4b_sparse_geometry.toml`
- Modify: `research/configs/adaptive_search/gemma2_sparse_geometry.toml`
- Create: `tests/test_auto_batch_selection.py`

**Interfaces:**
- Consumes: `Settings.batch_size == 0` as the existing automatic-selection sentinel.
- Produces: automatic search over powers of two through `Settings.max_batch_size == 4096` with `Settings.batch_size_vram_headroom_fraction == 0.10`.

- [x] **Step 1: Write the failing test**

```python
import tomllib
from pathlib import Path

from heretic.config import Settings


def test_auto_batch_defaults_and_search_profiles():
    settings = Settings(model="placeholder")
    assert settings.batch_size == 0
    assert settings.max_batch_size == 4096
    assert settings.batch_size_vram_headroom_fraction == 0.10

    root = Path(__file__).parents[1]
    for name in (
        "ministral3_sparse_geometry.toml",
        "gemma4_e4b_sparse_geometry.toml",
        "gemma2_sparse_geometry.toml",
    ):
        profile = tomllib.loads(
            (root / "research" / "configs" / "adaptive_search" / name).read_text(
                encoding="utf-8"
            )
        )
        assert profile["batch_size"] == 0
        assert profile["max_batch_size"] == 4096
        assert profile["batch_size_vram_headroom_fraction"] == 0.10
```

- [x] **Step 2: Run test to verify it fails**

Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest tests/test_auto_batch_selection.py -q`

Expected: FAIL because the current maximum is 512 and the VRAM reserve is 8%.

- [x] **Step 3: Write minimal implementation**

Set `Settings.max_batch_size` to 4096 and `Settings.batch_size_vram_headroom_fraction` to 0.10. Mirror both values in each maintained profile while retaining `batch_size = 0`.

- [x] **Step 4: Run focused and full tests**

Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest tests/test_auto_batch_selection.py -q`

Expected: PASS.

Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q`

Expected: all collected tests PASS.

- [x] **Step 5: Verify live-run isolation and commit**

Confirm that the active Qwen process still advances and that no current-run configuration file was modified. Commit only the source, maintained profiles, focused test, design, and plan.
