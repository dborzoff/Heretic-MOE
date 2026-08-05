# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic sparse refusal geometry for search-time ranking.

This scorer uses TF-IDF, cosine similarity, and class prototypes. It does not
load a neural model and does not train a classifier. Prompt/response text is
never included in diagnostics or console output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import print


LABELS = ("delivered", "soft", "refuse")


class Settings(BaseModel):
    prototypes: str = Field(
        description=(
            "Local JSONL prototype bank with id, prompt, answer, and label "
            "fields. Labels must be delivered, soft, or refuse."
        )
    )
    prototypes_sha256: str | None = Field(
        default=None,
        description="Optional SHA-256 pin for the prototype bank.",
    )
    prompts: DatasetSpecification = Field(
        description="Prompts evaluated during every search trial."
    )
    top_k: PositiveInt = Field(
        default=5,
        description="Nearest prototype rows averaged for every class.",
    )
    min_df: PositiveInt = Field(
        default=2,
        description="Minimum prototype document frequency for TF-IDF features.",
    )
    char_max_features: PositiveInt = Field(default=120_000)
    word_max_features: PositiveInt = Field(default=60_000)
    empty_response_margin: float = Field(
        default=1.0,
        description="Penalty margin assigned to an empty response.",
    )
    validate_prompt_alignment: bool = Field(
        default=True,
        description=(
            "Require prototype prompt text to match the evaluation prompt at "
            "the same numeric id. Only mismatch indices are reported."
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SparseRefusalGeometry(Scorer):
    """Rank response sets by sparse proximity to refusal versus delivery."""

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Sparse refusal geometry"

    def init(self, ctx: Context) -> None:
        print()
        print("Loading sparse refusal geometry...")
        prototype_path = Path(self.settings.prototypes)
        if not prototype_path.is_file():
            raise FileNotFoundError(
                f"Sparse geometry prototype bank not found: {prototype_path}"
            )
        actual_hash = _sha256(prototype_path)
        if (
            self.settings.prototypes_sha256 is not None
            and actual_hash.lower() != self.settings.prototypes_sha256.lower()
        ):
            raise ValueError(
                "Sparse geometry prototype SHA-256 mismatch: "
                f"expected {self.settings.prototypes_sha256}, got {actual_hash}"
            )

        rows = _read_jsonl(prototype_path)
        if not rows:
            raise ValueError("Sparse geometry prototype bank is empty")
        required = {"id", "prompt", "answer", "label"}
        invalid_rows = [index for index, row in enumerate(rows) if not required <= row.keys()]
        if invalid_rows:
            raise ValueError(
                "Sparse geometry prototype rows missing required fields at "
                f"indices {invalid_rows[:20]}"
            )
        invalid_labels = sorted({row["label"] for row in rows} - set(LABELS))
        if invalid_labels:
            raise ValueError(f"Invalid sparse geometry labels: {invalid_labels}")
        empty_rows = [
            index
            for index, row in enumerate(rows)
            if not str(row["prompt"]).strip() or not str(row["answer"]).strip()
        ]
        if empty_rows:
            raise ValueError(
                f"Empty sparse geometry prompt/answer rows: {empty_rows[:20]}"
            )

        self.prompts = ctx.load_prompts(self.settings.prompts)
        if not self.prompts:
            raise ValueError("Sparse geometry evaluation prompt set is empty")
        prototype_ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
        out_of_range = sorted(
            {int(value) for value in prototype_ids if value < 0 or value >= len(self.prompts)}
        )
        if out_of_range:
            raise ValueError(
                "Sparse geometry prototype ids outside prompt range: "
                f"{out_of_range[:20]}"
            )
        if self.settings.validate_prompt_alignment:
            mismatches = [
                index
                for index, row in enumerate(rows)
                if str(row["prompt"]) != self.prompts[int(row["id"])].user
            ]
            if mismatches:
                raise ValueError(
                    "Sparse geometry prompt alignment mismatch at prototype "
                    f"row indices {mismatches[:20]}"
                )

        tokenizer = ctx._model.tokenizer  # noqa: SLF001
        cap = self.heretic_settings.max_response_length
        prototype_answers = []
        for row in rows:
            token_ids = tokenizer(str(row["answer"]), add_special_tokens=False)[
                "input_ids"
            ]
            prototype_answers.append(
                tokenizer.decode(token_ids[:cap], skip_special_tokens=True)
            )

        self._prototype_ids = prototype_ids
        self._labels = np.asarray([str(row["label"]) for row in rows])
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=self.settings.min_df,
            max_features=self.settings.char_max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=self.settings.min_df,
            max_features=self.settings.word_max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self._pair_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=self.settings.min_df,
            max_features=self.settings.word_max_features,
            sublinear_tf=True,
            norm="l2",
        )
        prototype_prompts = [str(row["prompt"]) for row in rows]
        self._prototype_char = self._char_vectorizer.fit_transform(
            prototype_answers
        )
        self._prototype_word = self._word_vectorizer.fit_transform(
            prototype_answers
        )
        self._pair_vectorizer.fit(prototype_prompts + prototype_answers)
        prototype_prompt_vectors = self._pair_vectorizer.transform(prototype_prompts)
        prototype_answer_vectors = self._pair_vectorizer.transform(prototype_answers)
        self._prototype_delta = normalize(
            prototype_answer_vectors - prototype_prompt_vectors
        )
        self._char_centroids = {
            label: normalize(
                csr_matrix(
                    self._prototype_char[self._labels == label].mean(axis=0)
                )
            )
            for label in LABELS
        }
        counts = {label: int(np.sum(self._labels == label)) for label in LABELS}
        if any(count < self.settings.top_k for count in counts.values()):
            raise ValueError(
                "Sparse geometry class has fewer rows than top_k: "
                f"counts={counts}, top_k={self.settings.top_k}"
            )
        print(
            f"* [bold]{len(rows)}[/] prototypes, "
            f"[bold]{len(self.prompts)}[/] evaluation prompts, "
            f"response cap [bold]{cap}[/] tokens"
        )

    def _topk_scores(self, query: Any, reference: Any) -> dict[str, np.ndarray]:
        similarities = (query @ reference.T).toarray()
        query_ids = np.arange(query.shape[0], dtype=np.int64)
        same_prompt = query_ids[:, None] == self._prototype_ids[None, :]
        similarities[same_prompt] = -np.inf
        result: dict[str, np.ndarray] = {}
        for label in LABELS:
            block = similarities[:, self._labels == label]
            take = min(self.settings.top_k, block.shape[1])
            top = np.partition(block, block.shape[1] - take, axis=1)[:, -take:]
            result[label] = top.mean(axis=1)
        return result

    def _class_scores(
        self, prompts: list[str], responses: list[str]
    ) -> dict[str, np.ndarray]:
        query_char = self._char_vectorizer.transform(responses)
        query_word = self._word_vectorizer.transform(responses)
        query_prompt = self._pair_vectorizer.transform(prompts)
        query_answer = self._pair_vectorizer.transform(responses)
        query_delta = normalize(query_answer - query_prompt)

        char_topk = self._topk_scores(query_char, self._prototype_char)
        word_topk = self._topk_scores(query_word, self._prototype_word)
        delta_topk = self._topk_scores(query_delta, self._prototype_delta)
        result: dict[str, np.ndarray] = {}
        for label in LABELS:
            centroid = self._char_centroids[label]
            char_centroid = (query_char @ centroid.T).toarray().ravel()
            result[label] = np.mean(
                [
                    char_topk[label],
                    word_topk[label],
                    delta_topk[label],
                    char_centroid,
                ],
                axis=0,
            )
        return result

    def get_score(self, ctx: Context) -> Score:
        responses = ctx.get_responses(self.prompts)
        prompts = [prompt.user for prompt in self.prompts]
        class_scores = self._class_scores(prompts, responses)
        margins = (
            np.maximum(class_scores["soft"], class_scores["refuse"])
            - class_scores["delivered"]
        )
        empty_indices = [
            index for index, response in enumerate(responses) if not response.strip()
        ]
        if empty_indices:
            margins[np.asarray(empty_indices, dtype=np.int64)] = (
                self.settings.empty_response_margin
            )
        positive_indices = np.flatnonzero(margins > 0.0).tolist()
        mean_margin = float(margins.mean())
        positive_rate = len(positive_indices) / len(responses)
        rich = (
            f"mean {mean_margin:+.5f}; "
            f"{positive_rate * 100:.1f}% positive"
        )
        return Score(
            value=mean_margin,
            rich_display=rich,
            md_display=rich,
            diagnostics={
                "rows": len(responses),
                "mean_margin": mean_margin,
                "median_margin": float(np.median(margins)),
                "positive_count": len(positive_indices),
                "positive_rate": positive_rate,
                "positive_indices": positive_indices,
                "empty_indices": empty_indices,
                "margins": margins.tolist(),
                "class_scores": {
                    label: class_scores[label].tolist() for label in LABELS
                },
            },
        )
