"""Optional live geometry capture for an already edited Heretic trial."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import Tensor

from .language_map_trajectory import TrialRecord, append_trial_projection
from .model import Model
from .utils import Prompt


def _prompt_hash(prompt: Prompt) -> str:
    return sha256((prompt.system + "\0" + prompt.user).encode("utf-8")).hexdigest()


class TrialGeometrySession:
    """Collect evaluation residuals and commit one frozen anchor control."""

    def __init__(self, package_dir: Path, *, trial_number: int) -> None:
        self.package_dir = Path(package_dir)
        self.trial_number = int(trial_number)
        anchor_path = self.package_dir / "private" / "anchors.jsonl"
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
        with anchor_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows:
            raise ValueError("private anchor panel is empty")
        self._anchor_prompts = [
            Prompt(system=str(row.get("system", "")), user=str(row["prompt"]))
            for row in rows
        ]
        self._evaluation_hashes: list[str] = []
        self._evaluation_residuals: list[Tensor] = []
        self._seen_evaluation_hashes: set[str] = set()
        self._finalized = False

    def capture_evaluation(self, prompts: list[Prompt], residuals: Tensor) -> None:
        """Collect one-token residuals from normal evaluation prompt batches."""

        if self._finalized:
            raise RuntimeError("trial geometry session is already finalized")
        if residuals.ndim != 3 or residuals.shape[0] != len(prompts):
            raise ValueError("evaluation prompt/residual coverage mismatch")
        keep: list[int] = []
        for index, prompt in enumerate(prompts):
            digest = _prompt_hash(prompt)
            if digest in self._seen_evaluation_hashes:
                continue
            self._seen_evaluation_hashes.add(digest)
            self._evaluation_hashes.append(digest)
            keep.append(index)
        if keep:
            self._evaluation_residuals.append(
                residuals[keep].detach().cpu().to(torch.float32).contiguous()
            )

    def finalize(self, model: Model, trial: TrialRecord) -> dict[str, object]:
        """Measure the frozen anchors and publish the complete trial atomically."""

        if self._finalized:
            raise RuntimeError("trial geometry session is already finalized")
        if trial.number != self.trial_number:
            raise ValueError("trial identity differs from geometry session")
        anchor_residuals = model.get_residuals_batched(self._anchor_prompts)
        evaluation = (
            torch.cat(self._evaluation_residuals, dim=0)
            if self._evaluation_residuals
            else None
        )
        entry = append_trial_projection(
            self.package_dir,
            trial,
            anchor_residuals,
            evaluation_residuals=evaluation,
            evaluation_prompt_hashes=(
                self._evaluation_hashes if evaluation is not None else None
            ),
        )
        self._finalized = True
        return entry
