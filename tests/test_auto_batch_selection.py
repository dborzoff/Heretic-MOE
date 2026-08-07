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
