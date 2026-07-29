# -*- coding: utf-8 -*-
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
returns the relative increase over the baseline model:

    value = perplexity / perplexity_baseline - 1

Zero means the model predicts the text as well as before. A value of 0.03 means
three percent worse. This keeps the result near the scale expected by Scorer.

Enable it in the scorer configuration:

    [[scorers]]
    plugin = "heretic.scorers.perplexity.Perplexity"
    optimization = "minimize"
"""
import torch
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import print


class Settings(BaseModel):
    text: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="Salesforce/wikitext",
            split="test",
            column="text",
        ),
        description="Text corpus used to measure perplexity.",
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
    Perplexity on a fixed text corpus, relative to the baseline model.
    Measures how much the edit degraded language modelling.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Perplexity increase"

    # Context withholds the model, and its public methods cannot score arbitrary
    # token sequences: get_logits accepts prompts and returns only the final
    # token. Keep the private access here so upstream API changes have one repair
    # point.
    @staticmethod
    def _model_and_tokenizer(ctx: Context):
        m = ctx._model            # noqa: SLF001
        return m.model, m.tokenizer

    def _windows(self, ctx: Context):
        """Tokenize and split the corpus once per run."""
        from datasets import load_dataset

        spec = self.settings.text
        # Wikitext also requires a configuration name. The Hub now rejects the
        # bare "wikitext" dataset ID, which used to fail at startup.
        if spec.dataset.endswith("wikitext"):
            ds = load_dataset(spec.dataset, "wikitext-2-raw-v1", split=spec.split)
        else:
            ds = load_dataset(spec.dataset, split=spec.split)
        text = "\n\n".join(t for t in ds[spec.column] if t.strip())

        _, tok = self._model_and_tokenizer(ctx)
        ids = tok(text, return_tensors="pt").input_ids[0]
        w = self.settings.window
        n = min(self.settings.chunks, len(ids) // w)
        return [ids[i * w:(i + 1) * w] for i in range(n)]

    @torch.no_grad()
    def _perplexity(self, ctx: Context) -> float:
        model, _ = self._model_and_tokenizer(ctx)
        device = next(model.parameters()).device
        total, count = 0.0, 0
        for w in self._windows_cached:
            ids = w.unsqueeze(0).to(device)
            # The model shifts labels by one token internally.
            out = model(ids, labels=ids)
            # out.loss is mean NLL over n-1 targets. Weight by that count so
            # windows of different lengths contribute per token.
            total += float(out.loss) * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
        return float(torch.exp(torch.tensor(total / max(count, 1))))

    def init(self, ctx: Context) -> None:
        print()
        print(f"Loading Perplexity text from [bold]{self.settings.text.dataset}[/]...")
        self._windows_cached = self._windows(ctx)
        print(f"* [bold]{len(self._windows_cached)}[/] windows of "
              f"[bold]{self.settings.window}[/] tokens")
        print("* Measuring baseline perplexity...")
        self._baseline = self._perplexity(ctx)
        print(f"* Baseline perplexity: [bold]{self._baseline:.4f}[/]")

    def get_score(self, ctx: Context) -> Score:
        ppl = self._perplexity(ctx)
        rel = ppl / self._baseline - 1.0
        # The front-selection menu prints this field without Rich parsing.
        # Markup would appear verbatim, as in "[bold]51.71[/]".
        return Score(
            value=rel,
            rich_display=f"{ppl:.4f} ({rel * 100:+.2f}% vs baseline)",
            md_display=f"{ppl:.4f} ({rel * 100:+.2f}%)",
        )
