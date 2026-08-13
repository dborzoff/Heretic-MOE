from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from heretic import supervisor


def test_supervisor_public_cli_is_config_plus_overrides() -> None:
    args = supervisor.parse_args(
        [
            "--config",
            "config.yaml",
            "--model",
            "example/override",
            "--run-root",
            "override-run",
            "--devices",
            "0,1",
            "--target-trials",
            "800",
            "--exploration-trials",
            "160",
            "--post-search",
            "recheck",
            "--incompatible-contract",
            "new_run",
            "--dry-run",
        ]
    )

    assert args.config == Path("config.yaml")
    assert args.model == "example/override"
    assert args.run_root == Path("override-run")
    assert args.devices == "0,1"
    assert args.target_trials == 800
    assert args.exploration_trials == 160
    assert args.post_search == "recheck"
    assert args.incompatible_contract == "new_run"
    assert args.dry_run is True


def test_supervisor_defaults_to_config_yaml_and_no_overrides() -> None:
    args = supervisor.parse_args([])

    assert args.config == Path("config.yaml")
    assert args.model is None
    assert args.run_root is None
    assert args.devices is None
    assert args.target_trials is None
    assert args.exploration_trials is None
    assert args.post_search is None
    assert args.incompatible_contract is None
    assert args.dry_run is False


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "--base-config",
        "--data-root",
        "--srg-calibration-source",
        "--max-workers",
        "--min-free-fraction",
        "--min-free-gib",
        "--n-trials",
        "--continue-shared-only",
        "--finalize",
        "--no-finalize",
        "--search-only",
        "--recheck-only",
        "--worker-executable",
    ],
)
def test_supervisor_rejects_legacy_public_flags(legacy_flag: str) -> None:
    with pytest.raises(SystemExit):
        supervisor.parse_args([legacy_flag])


def test_run_lock_lives_beside_run_root_and_does_not_create_it(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "planned-run"

    with supervisor.AdaptiveRunLock(run_root) as lock:
        assert lock.path == tmp_path / ".planned-run.hereticmoe-controller.lock"
        assert not run_root.exists()
        first_size = lock.path.stat().st_size

    with supervisor.AdaptiveRunLock(run_root) as lock:
        assert lock.path.stat().st_size == first_size == 64

    assert lock.path.read_bytes().startswith(b"pid=")


def test_gpu_probe_reports_unavailable_numeric_fields_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        stdout = "0, NVIDIA Test GPU, 24564, N/A, N/A\n"

    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="unavailable numeric field"):
        supervisor.detect_nvidia_gpus()


def _write_minimal_launch_config(path: Path, run_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "model": {"path": "example/model"},
                "run": {"root": str(run_root)},
                "data": {
                    "dataset_root": str(path.parent / "dataset"),
                    "split_root": str(path.parent / "dataset" / "split"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_supervisor_dry_run_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    _write_minimal_launch_config(config_path, run_root)
    monkeypatch.setattr(supervisor, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        supervisor, "executable_path", lambda _value: tmp_path / "hereticMOE.exe"
    )
    monkeypatch.setattr(
        supervisor,
        "detect_nvidia_gpus",
        lambda: [supervisor.GpuInfo("0", "GPU", 24576, 24000, 0)],
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run must not start the controller"),
    )

    supervisor.main(["--config", str(config_path), "--dry-run"])

    assert not run_root.exists()
    assert not (tmp_path / ".run.hereticmoe-controller.lock").exists()


def test_supervisor_locks_before_mutating_run_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    _write_minimal_launch_config(config_path, run_root)
    events: list[str] = []

    class RecordingLock:
        def __init__(self, root: Path):
            assert root == run_root

        def __enter__(self):
            events.append("lock-enter")
            return self

        def __exit__(self, *args):
            events.append("lock-exit")

    real_resolve = supervisor.resolve_run_root

    def recording_resolve(*args, **kwargs):
        events.append("resolve")
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(supervisor, "AdaptiveRunLock", RecordingLock)
    monkeypatch.setattr(supervisor, "resolve_run_root", recording_resolve)
    monkeypatch.setattr(supervisor, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        supervisor, "executable_path", lambda _value: tmp_path / "hereticMOE.exe"
    )
    monkeypatch.setattr(
        supervisor,
        "detect_nvidia_gpus",
        lambda: [supervisor.GpuInfo("0", "GPU", 24576, 24000, 0)],
    )

    class Result:
        returncode = 0

    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(SystemExit, match="0"):
        supervisor.main(["--config", str(config_path)])

    assert events == ["lock-enter", "resolve", "resolve", "lock-exit"]


def test_bundle_failure_preserves_existing_run_before_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    _write_minimal_launch_config(config_path, run_root)
    run_root.mkdir()
    marker = run_root / "old-journal.bin"
    marker.write_bytes(b"keep")
    monkeypatch.setattr(supervisor, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        supervisor,
        "executable_path",
        lambda _value: tmp_path / "hereticMOE.exe",
    )
    monkeypatch.setattr(
        supervisor,
        "detect_nvidia_gpus",
        lambda: [supervisor.GpuInfo("0", "GPU", 24576, 24000, 0)],
    )

    def fail_bundle(*args, **kwargs):
        raise OSError("simulated bundle write failure")

    monkeypatch.setattr(supervisor, "write_effective_config_bundle", fail_bundle)

    with pytest.raises(OSError, match="simulated bundle write failure"):
        supervisor.main(["--config", str(config_path)])

    assert marker.read_bytes() == b"keep"
    assert not tuple(tmp_path.glob("run_archive_*"))


def test_new_run_bundle_uses_the_selected_suffixed_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    _write_minimal_launch_config(config_path, run_root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["run"]["incompatible_contract"] = "new_run"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    run_root.mkdir()
    (run_root / "old-journal.bin").write_bytes(b"keep")
    monkeypatch.setattr(supervisor, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        supervisor,
        "executable_path",
        lambda _value: tmp_path / "hereticMOE.exe",
    )
    monkeypatch.setattr(
        supervisor,
        "detect_nvidia_gpus",
        lambda: [supervisor.GpuInfo("0", "GPU", 24576, 24000, 0)],
    )
    captured: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="0"):
        supervisor.main(["--config", str(config_path)])

    selected_root = tmp_path / "run_v2"
    effective = yaml.safe_load(
        (selected_root / "config.effective.yaml").read_text(encoding="utf-8")
    )
    assert Path(effective["run"]["root"]) == selected_root
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--run-root") + 1] == str(selected_root)


def test_supervisor_controller_uses_public_config_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "model": {"path": "example/model"},
                "run": {
                    "root": str(run_root),
                    "target_trials": 700,
                    "exploration_trials": 140,
                },
                "devices": {"mode": "include", "include": ["0", "1"]},
                "data": {
                    "dataset_root": str(tmp_path / "dataset"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        supervisor, "executable_path", lambda _value: tmp_path / "hereticMOE.exe"
    )
    monkeypatch.setattr(
        supervisor,
        "detect_nvidia_gpus",
        lambda: [
            supervisor.GpuInfo("0", "GPU0", 24576, 24000, 0),
            supervisor.GpuInfo("1", "GPU1", 24576, 24000, 0),
        ],
    )
    captured: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return Result()

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="0"):
        supervisor.main(["--config", str(config_path)])

    command = captured["command"]
    assert isinstance(command, list)
    assert "--model" in command
    assert command[command.index("--model") + 1] == "example/model"
    assert command[command.index("--target-trials") + 1] == "700"
    assert command[command.index("--exploration-trials") + 1] == "140"
    assert command[command.index("--devices") + 1] == "0,1"
    assert "--data-root" in command
    assert command[command.index("--data-root") + 1] == str(tmp_path / "dataset")
    assert "--srg-calibration-source" not in command
