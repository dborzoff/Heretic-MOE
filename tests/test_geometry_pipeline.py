from __future__ import annotations

from pathlib import Path

from heretic.geometry_pipeline import geometry_project_command, geometry_render_command


def test_geometry_project_command_reuses_frozen_map_and_shared_journal() -> None:
    command = geometry_project_command(
        executable=Path("F:/bin/hereticMOE.exe"),
        split_root=Path("F:/data/split"),
        runtime_sources=Path("F:/run/runtime_sources/direction_map"),
        journal=Path("F:/run/shared/study.jsonl"),
        package_dir=Path("F:/run/runtime/geometry_3d"),
        languages=("en", "ru"),
        rows_per_cell=1000,
        seed=17,
    )

    assert command[:3] == ["F:\\bin\\hereticMOE.exe", "geometry-map", "project"]
    assert command[command.index("--cache-dir") + 1] == "F:\\run\\runtime_sources\\direction_map\\cache"
    assert command[command.index("--journal") + 1] == "F:\\run\\shared\\study.jsonl"
    assert command.count("--group-a") == 2
    assert command.count("--group-b") == 2


def test_geometry_render_command_targets_the_offline_report() -> None:
    command = geometry_render_command(
        executable=Path("F:/bin/hereticMOE.exe"),
        package_dir=Path("F:/run/runtime/geometry_3d"),
    )

    assert command == [
        "F:\\bin\\hereticMOE.exe",
        "geometry-map",
        "render",
        "--package-dir",
        "F:\\run\\runtime\\geometry_3d",
    ]

