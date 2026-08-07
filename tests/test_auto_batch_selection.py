import importlib
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


def test_next_batch_memory_prediction_blocks_unsafe_doubling():
    main = importlib.import_module("heretic.main")
    predict = getattr(main, "_predict_next_batch_free_bytes", None)
    assert predict is not None

    # 16.4 -> 11.0 GiB consumed 5.4 GiB. Doubling the batch is expected
    # to consume about twice that increment, leaving only 0.2 GiB.
    assert predict(previous_free_bytes=164, current_free_bytes=110) == 2
