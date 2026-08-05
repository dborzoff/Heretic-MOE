r"""Perplexity scorer for Heretic: measure edit cost on text.

Why use this instead of KL.

The built-in KLDivergence compares the first-token distribution on 100 harmless
prompts with the baseline model. Perplexity covers 200,000 tokens on the full
wikitext test set. Scale is not the main difference. First-token KL sees whether
the opening changed and is nearly blind to what follows.

Our measurements exposed the gap. These are two points from the same front:

  build        KL (first token)   perplexity vs baseline
  balanced          0.0021                +3.4%
  max               0.0126               +18.1%

KL separates them by a factor of six. Perplexity says the aggressive edit costs
five times more, at much larger absolute damage. The search was optimizing a
proxy that could not see the real cost curve.

A full CPU pass takes fourteen minutes, too long for the search loop. The same
data takes seconds on a GPU, so the proxy is no longer needed.

This scorer averages negative log-likelihood over fixed text windows and
returns the absolute relative drift from the baseline model:

    signed_change = perplexity / perplexity_baseline - 1
    value = abs(signed_change)

Zero means the model predicts the text as well as before. A value of 0.03 means
that perplexity moved three percent in either direction. This is a preservation
cost: a large decrease is still a large behavioural change and must not receive
an optimization advantage over an unchanged model. The signed change remains
available in diagnostics. Diagnostics also include a paired window-level
confidence interval so small changes can be distinguished from sampling
variation. This keeps the result near the scale expected by Scorer.

Enable it in the scorer configuration:

    [[scorers]]
    plugin = "heretic.scorers.perplexity.Perplexity"
    optimization = "minimize"
"""
import hashlib
from importlib import resources
from math import expm1, sqrt
from pathlib import Path
from statistics import fmean, stdev

import torch
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import print

BUILTIN_PERPLEXITY_CORPORA = {
    "builtin://perplexity-reference-v1": (
        "perplexity_reference_v1.txt",
        "1d6f25ca80bd49255212d67d7eff96763ab01abbd472c04b916ec62318857a9d",
    ),
}


def paired_relative_perplexity_interval(
    window_nll: list[float],
    baseline_window_nll: list[float],
    confidence_z: float = 1.959963984540054,
) -> dict[str, float | bool | int]:
    """Estimate uncertainty of the paired relative perplexity change.

    Each edited window is paired with the same baseline window.  The interval is
    computed in mean-NLL space and transformed back to relative perplexity.  It
    describes uncertainty across the configured text windows; it is diagnostic
    only and deliberately does not clamp or replace the raw optimization value.
    """

    if len(window_nll) != len(baseline_window_nll):
        raise ValueError("Current and baseline perplexity windows must match")
    if not window_nll:
        raise ValueError("At least one perplexity window is required")

    deltas = [
        current - baseline
        for current, baseline in zip(window_nll, baseline_window_nll, strict=True)
    ]
    mean_delta = fmean(deltas)
    standard_error = stdev(deltas) / sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    lower = expm1(mean_delta - confidence_z * standard_error)
    upper = expm1(mean_delta + confidence_z * standard_error)
    return {
        "paired_window_count": len(deltas),
        "paired_nll_delta_mean": mean_delta,
        "paired_nll_delta_standard_error": standard_error,
        "relative_change_ci95_lower": lower,
        "relative_change_ci95_upper": upper,
        "statistically_distinguishable_from_baseline": lower > 0.0 or upper < 0.0,
    }


def load_perplexity_text(spec: DatasetSpecification) -> str:
    """Load a local UTF-8 text file or a configured Hugging Face dataset column."""

    builtin = BUILTIN_PERPLEXITY_CORPORA.get(spec.dataset)
    if builtin is not None:
        filename, expected_sha256 = builtin
        payload = (
            resources.files("heretic.data").joinpath(filename).read_bytes()
        )
        # Git stores the frozen corpus with LF line endings. Normalize a
        # Windows worktree's CRLF representation before validating and using
        # it so the same committed corpus is byte-identical on every host.
        payload = payload.replace(b"\r\n", b"\n")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Built-in perplexity corpus hash mismatch: {actual_sha256}"
            )
        return payload.decode("utf-8")

    local_text_path = Path(spec.dataset)
    if local_text_path.is_file():
        return local_text_path.read_text(encoding="utf-8")

    from datasets import load_dataset

    if spec.dataset.endswith("wikitext"):
        dataset = load_dataset(
            spec.dataset, "wikitext-2-raw-v1", split=spec.split
        )
    else:
        dataset = load_dataset(spec.dataset, split=spec.split)
    if spec.column is None:
        raise ValueError(
            "Perplexity dataset column is required for non-text-file inputs"
        )
    return "\n\n".join(text for text in dataset[spec.column] if text.strip())


class Settings(BaseModel):
    text: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="builtin://perplexity-reference-v1",
        ),
        description=(
            "Text corpus used to measure perplexity. The default is bundled "
            "with Heretic so offline and remote runs use identical bytes."
        ),
    )
    window: int = Field(
        default=512,
        description=(
            "Tokens per window. This matches llama-perplexity so results can be"
            " compared with GGUF measurements."
        ),
    )
    chunks: int = Field(
        default=24,
        description=(
            "Number of windows. The default covers 12,000 tokens and takes a few"
            " seconds on a GPU. Published figures need more data. Search trials"
            " do not: every trial uses the same windows."
        ),
    )


class Perplexity(Scorer):
    """
    Absolute perplexity drift on a fixed text corpus relative to baseline.
    Measures how much the edit changed language modelling in either direction.
    Lower is better (better preservation).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Perplexity drift"

    # Context withholds the model, and its public methods cannot score arbitrary
    # token sequences: get_logits accepts prompts and returns only the final
    # token. Keep the private access here so upstream API changes have one repair
    # point.
    @staticmethod
    def _model_and_tokenizer(ctx: Context):
        m = ctx._model
        return m.model, m.tokenizer

    def _windows(self, ctx: Context):
        """Tokenize and split the configured corpus once per run."""
        spec = self.settings.text
        text = load_perplexity_text(spec)

        _, tok = self._model_and_tokenizer(ctx)
        ids = tok(text, return_tensors="pt").input_ids[0]
        w = self.settings.window
        n = min(self.settings.chunks, len(ids) // w)
        return [ids[i * w:(i + 1) * w] for i in range(n)]

    @torch.no_grad()
    def _perplexity(self, ctx: Context) -> tuple[float, list[float], int]:
        model, _ = self._model_and_tokenizer(ctx)
        device = next(model.parameters()).device
        total, count = 0.0, 0
        window_nll: list[float] = []
        for w in self._windows_cached:
            ids = w.unsqueeze(0).to(device)
            # The model shifts labels by one token internally.
            out = model(ids, labels=ids)
            # out.loss is mean NLL over n-1 targets. Weight by that count so
            # windows of different lengths contribute per token.
            loss = float(out.loss)
            window_nll.append(loss)
            total += loss * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
        return (
            float(torch.exp(torch.tensor(total / max(count, 1)))),
            window_nll,
            count,
        )

    def init(self, ctx: Context) -> None:
        print()
        print(f"Loading Perplexity text from [bold]{self.settings.text.dataset}[/]...")
        self._windows_cached = self._windows(ctx)
        print(f"* [bold]{len(self._windows_cached)}[/] windows of "
              f"[bold]{self.settings.window}[/] tokens")
        print("* Measuring baseline perplexity...")
        self._baseline, self._baseline_window_nll, self._baseline_token_count = (
            self._perplexity(ctx)
        )
        print(f"* Baseline perplexity: [bold]{self._baseline:.4f}[/]")

    def get_score(self, ctx: Context) -> Score:
        ppl, window_nll, token_count = self._perplexity(ctx)
        rel = ppl / self._baseline - 1.0
        drift = abs(rel)
        uncertainty = paired_relative_perplexity_interval(
            window_nll, self._baseline_window_nll
        )
        # The front-selection menu prints this field without Rich parsing.
        # Markup would appear verbatim, as in "[bold]51.71[/]".
        return Score(
            value=drift,
            rich_display=f"{ppl:.4f} ({drift * 100:.2f}% drift)",
            md_display=f"{ppl:.4f} ({drift * 100:.2f}% drift)",
            diagnostics={
                "perplexity": ppl,
                "baseline_perplexity": self._baseline,
                "relative_change": rel,
                "absolute_relative_change": drift,
                "token_count": token_count,
                "window_nll": window_nll,
                "baseline_window_nll": self._baseline_window_nll,
                **uncertainty,
            },
        )
