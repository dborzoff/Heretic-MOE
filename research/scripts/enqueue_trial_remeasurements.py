# SPDX-License-Identifier: AGPL-3.0-or-later

"""Queue completed Optuna trials for measurement under an updated scorer budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--trial-indices", type=int, nargs="+", required=True)
    parser.add_argument("--ppl-chunks", type=int, required=True)
    parser.add_argument("--close-running", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = JournalStorage(JournalFileBackend(str(args.journal)))
    summaries = optuna.study.get_all_study_summaries(storage)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one study in journal, found {len(summaries)}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    closed_running: list[int] = []
    if args.close_running:
        for trial in study.trials:
            if trial.state == TrialState.RUNNING:
                storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)  # noqa: SLF001
                closed_running.append(trial.number)

    queued: list[dict[str, int]] = []
    trials = study.get_trials(deepcopy=False)
    for display_index in args.trial_indices:
        matches = [
            trial
            for trial in trials
            if trial.user_attrs.get("index", trial.number + 1) == display_index
            and trial.state == TrialState.COMPLETE
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one completed trial for display index {display_index}, "
                f"found {len(matches)}"
            )
        source = matches[0]
        study.enqueue_trial(
            source.params,
            user_attrs={
                "remeasure_of_trial_index": display_index,
                "remeasure_ppl_chunks": args.ppl_chunks,
            },
            skip_if_exists=False,
        )
        queued.append(
            {
                "source_trial_number": source.number,
                "source_display_index": display_index,
            }
        )

    state_counts = {
        state.name: sum(trial.state == state for trial in study.trials)
        for state in TrialState
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "journal": str(args.journal),
                "study_name": study.study_name,
                "closed_running_trial_numbers": closed_running,
                "queued": queued,
                "ppl_chunks": args.ppl_chunks,
                "state_counts": state_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
