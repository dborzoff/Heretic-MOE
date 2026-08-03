#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render text-free Heretic Adaptive search and finalist visualizations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState


COLORS = {
    "random_startup": "#58a6ff",
    "sobol_startup": "#bc8cff",
    "source_tpe": "#f2cc60",
    "shared_tpe": "#ff7b72",
    "front": "#3fb950",
    "base": "#8b949e",
    "max": "#3fb950",
    "balanced": "#39c5cf",
}

PHASE_LABELS = {
    "random_startup": "Random startup",
    "sobol_startup": "Scrambled Sobol startup",
    "source_tpe": "Source-study multivariate TPE",
    "shared_tpe": "Shared multivariate TPE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--semantic-summary", type=Path, required=True)
    parser.add_argument("--ppl-summary", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_study(journal: Path) -> optuna.study.Study:
    storage = JournalStorage(
        JournalFileBackend(
            str(journal),
            lock_obj=JournalFileOpenLock(str(journal)),
        )
    )
    summaries = optuna.get_all_study_summaries(storage=storage)
    if len(summaries) != 1:
        raise ValueError(f"Expected one study in {journal}, found {len(summaries)}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def trial_phase(trial: optuna.trial.FrozenTrial) -> str:
    source = trial.user_attrs.get("merged_source")
    source_trial = trial.user_attrs.get("merged_source_trial_number")
    if source in {"random", "sobol"} and isinstance(source_trial, int):
        if source_trial < 60:
            return f"{source}_startup"
        return "source_tpe"
    return "shared_tpe"


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    best_refusal = float("inf")
    for point in sorted(
        points,
        key=lambda row: (row["surrogate_ppl_change"], row["keyword_rate"]),
    ):
        refusal = float(point["keyword_rate"])
        if refusal < best_refusal:
            front.append(point)
            best_refusal = refusal
    return front


def build_data(
    study: optuna.study.Study,
    journal: Path,
    semantic_summary: dict[str, Any],
    ppl_summary: dict[str, Any],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE or not trial.values:
            continue
        trials.append(
            {
                "trial": trial.number,
                "display_trial": trial.number + 1,
                "phase": trial_phase(trial),
                "keyword_rate": trial.values[0],
                "surrogate_ppl_change": trial.values[1],
            }
        )
    trials.sort(key=lambda row: row["trial"])

    phase_counts: dict[str, int] = {}
    for row in trials:
        phase = str(row["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    exact_models = ppl_summary["models"]
    semantic_models = semantic_summary["systems"]
    finalists = {}
    for variant, trial_number in (("max", 260), ("balanced", 488)):
        semantics = semantic_models[variant]
        finalists[variant] = {
            "trial": trial_number,
            "exact_ppl": exact_models[variant],
            "semantic": semantics,
        }

    return {
        "schema_version": 1,
        "study": {
            "journal_sha256": sha256(journal),
            "completed_trials": len(trials),
            "phase_counts": phase_counts,
            "search_evaluation": {
                "refusal_proxy": "KeywordRate on 136 prompts",
                "response_tokens": 48,
                "perplexity_surrogate": "24 windows x 512 tokens on frozen local text",
            },
        },
        "trials": trials,
        "final_validation": {
            "semantic_rows_per_system": 136,
            "semantic_response_tokens": 2048,
            "exact_perplexity": "400 windows x 512 tokens on Wikitext-2 test",
            "base": {
                "exact_ppl": exact_models["original"],
                "semantic": semantic_models["original"],
            },
            "finalists": finalists,
        },
        "text_free": True,
    }


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#8b949e",
            "axes.labelcolor": "#e6edf3",
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9",
            "text.color": "#e6edf3",
            "grid.color": "#30363d",
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "heretic-adaptive-v1",
        }
    )


def render_finalists(data: dict[str, Any], output: Path) -> None:
    validation = data["final_validation"]
    systems = {
        "Base model": validation["base"],
        "Balanced": validation["finalists"]["balanced"],
        "Max": validation["finalists"]["max"],
    }
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.82)
    marker_for = {"Base model": "o", "Balanced": "D", "Max": "*"}
    color_for = {
        "Base model": COLORS["base"],
        "Balanced": COLORS["balanced"],
        "Max": COLORS["max"],
    }
    size_for = {"Base model": 150, "Balanced": 180, "Max": 300}

    for name, payload in systems.items():
        ppl_change = float(payload["exact_ppl"]["relative_change"]) * 100
        semantics = payload["semantic"]
        residual = (
            float(semantics["residual_refusal_or_evasion_total"]) / 136 * 100
        )
        ax.scatter(
            [ppl_change],
            [residual],
            marker=marker_for[name],
            s=size_for[name],
            color=color_for[name],
            edgecolor="#f0f6fc",
            linewidth=0.8,
            zorder=4,
        )
        offset = {
            "Base model": (10, 13),
            "Balanced": (10, 8),
            "Max": (-170, -24),
        }[name]
        ax.annotate(
            f"{name}\n{int(semantics['residual_refusal_or_evasion_total'])}/136 "
            f"residual • {ppl_change:+.5f}% PPL",
            (ppl_change, residual),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=color_for[name],
            weight="bold",
        )

    ax.annotate(
        "lower is better",
        xy=(0.004, 22.5),
        xytext=(0.018, 30),
        arrowprops={"arrowstyle": "->", "color": COLORS["front"], "lw": 1.6},
        color=COLORS["front"],
        fontsize=10,
    )
    fig.text(
        0.09,
        0.95,
        "Heretic Adaptive v1 — exact exported-model validation",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.905,
        "136 multilingual prompts at 2,048 tokens • PPL: 400 × 512-token windows",
        color="#8b949e",
        fontsize=9.5,
    )
    ax.set_xlabel("Exact perplexity change vs base (%)")
    ax.set_ylabel("Policy refusals + evasive responses (%)")
    ax.set_xlim(-0.004, 0.053)
    ax.set_ylim(20, 72)
    ax.grid(True, alpha=0.75, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="svg",
        facecolor=fig.get_facecolor(),
        metadata={"Date": None, "Creator": "Heretic Adaptive"},
    )
    # Matplotlib formats path commands with trailing spaces. Normalize them so
    # generated documentation also passes Git's whitespace checks.
    svg = output.read_text(encoding="utf-8")
    output.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        output.with_suffix(".png"),
        format="png",
        dpi=160,
        facecolor=fig.get_facecolor(),
        metadata={"Software": "Heretic Adaptive"},
    )
    plt.close(fig)


def phase_at(display_trial: int) -> str:
    if display_trial <= 60:
        return "random_startup"
    if display_trial <= 120:
        return "source_tpe"
    if display_trial <= 180:
        return "sobol_startup"
    if display_trial <= 240:
        return "source_tpe"
    return "shared_tpe"


def render_search_animation(data: dict[str, Any], output: Path, fps: int) -> None:
    trials = data["trials"]
    stops = set(range(10, 601, 10))
    stops.update({60, 120, 180, 240, 260, 488, 600})
    frames = sorted(stops)
    x_limit = 2.0

    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)

    def draw(display_trial: int) -> None:
        ax.clear()
        observed = [row for row in trials if row["display_trial"] <= display_trial]
        visible = [
            row
            for row in observed
            if float(row["surrogate_ppl_change"]) * 100 <= x_limit
        ]
        hidden = len(observed) - len(visible)
        for phase in PHASE_LABELS:
            phase_rows = [row for row in visible if row["phase"] == phase]
            if not phase_rows:
                continue
            ax.scatter(
                [float(row["surrogate_ppl_change"]) * 100 for row in phase_rows],
                [float(row["keyword_rate"]) * 100 for row in phase_rows],
                s=20,
                alpha=0.68,
                color=COLORS[phase],
                edgecolors="none",
                label=PHASE_LABELS[phase],
            )

        front = pareto_front(observed)
        visible_front = [
            row
            for row in front
            if float(row["surrogate_ppl_change"]) * 100 <= x_limit
        ]
        if visible_front:
            ax.plot(
                [float(row["surrogate_ppl_change"]) * 100 for row in visible_front],
                [float(row["keyword_rate"]) * 100 for row in visible_front],
                color=COLORS["front"],
                lw=2.1,
                marker="o",
                markersize=3.5,
                label="Current Pareto front",
                zorder=5,
            )

        for trial_number, name, marker, color in (
            (260, "Max", "*", COLORS["max"]),
            (488, "Balanced", "D", COLORS["balanced"]),
        ):
            if display_trial <= trial_number:
                continue
            row = trials[trial_number]
            x = float(row["surrogate_ppl_change"]) * 100
            y = float(row["keyword_rate"]) * 100
            ax.scatter(
                [x],
                [y],
                marker=marker,
                s=180 if marker == "*" else 95,
                color=color,
                edgecolor="#f0f6fc",
                linewidth=0.8,
                zorder=7,
            )
            ax.annotate(
                name,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                color=color,
                weight="bold",
                fontsize=9,
            )

        current_phase = phase_at(display_trial)
        ax.set_title(
            f"Heretic Adaptive search — {display_trial}/600 trials",
            loc="left",
            fontsize=16,
            weight="bold",
            pad=14,
        )
        ax.text(
            0,
            1.01,
            f"Current phase: {PHASE_LABELS[current_phase]}",
            transform=ax.transAxes,
            color=COLORS[current_phase],
            fontsize=10,
            weight="bold",
        )
        ax.text(
            0.99,
            1.01,
            f"{hidden} high-cost points outside the 2% view",
            transform=ax.transAxes,
            ha="right",
            color="#8b949e",
            fontsize=8.5,
        )
        ax.set_xlabel("Search-time PPL surrogate change (%)")
        ax.set_ylabel("Keyword refusal proxy (%)")
        ax.set_xlim(-0.1, x_limit)
        ax.set_ylim(-1, 40)
        ax.grid(True, alpha=0.75, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            frameon=False,
            fontsize=8.5,
            ncol=2,
        )

    movie = animation.FuncAnimation(
        fig,
        draw,
        frames=frames,
        interval=1000 / fps,
        repeat_delay=1400,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    study = load_study(args.journal.resolve())
    semantic = json.loads(args.semantic_summary.read_text(encoding="utf-8"))
    ppl = json.loads(args.ppl_summary.read_text(encoding="utf-8"))
    data = build_data(study, args.journal.resolve(), semantic, ppl)
    if data["study"]["completed_trials"] != 600:
        raise RuntimeError("The release visualization requires exactly 600 trials")

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    set_plot_style()
    render_finalists(data, args.asset_dir / "adaptive-finalists.svg")
    render_search_animation(data, args.asset_dir / "adaptive-search-progress.gif", args.fps)
    print(
        json.dumps(
            {
                "data": str(args.data_output.resolve()),
                "data_sha256": sha256(args.data_output),
                "static": str((args.asset_dir / "adaptive-finalists.svg").resolve()),
                "animation": str(
                    (args.asset_dir / "adaptive-search-progress.gif").resolve()
                ),
                "trials": data["study"]["completed_trials"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
