# SPDX-License-Identifier: AGPL-3.0-or-later
"""Preserve the Gemma hidden-state stack consumed by LTX.

LTX does not condition on only the final Gemma layer.  ComfyUI's LTX text
encoder concatenates the embedding output and all 48 transformer-layer
outputs, normalizes every layer, and projects the resulting 49-way stack.
This scorer mirrors the normalization before that projection and measures
the cosine drift from the unedited model on a fixed prompt sample.
"""

from __future__ import annotations

from statistics import fmean

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import Prompt, print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        description="Fixed prompts used to preserve LTX conditioning."
    )
    max_prompts: int = Field(
        default=8,
        ge=1,
        description="Deterministic prefix of the configured prompt set to score.",
    )
    max_tokens: int = Field(
        default=128,
        ge=8,
        description="Maximum tokens retained from each conditioning prompt.",
    )
    normalization_epsilon: float = Field(
        default=1e-6,
        gt=0,
        description="Epsilon used by the LTX hidden-stack normalization.",
    )


class LTXConditioningDrift(Scorer):
    """Cosine drift of the normalized all-layer Gemma representation."""

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "LTX conditioning drift"

    @staticmethod
    def _model_and_tokenizer(ctx: Context):
        wrapper = ctx._model  # noqa: SLF001 - scorer needs hidden states
        return wrapper.model, wrapper.tokenizer

    @staticmethod
    def _input_device(model: torch.nn.Module) -> torch.device:
        root = model
        with torch.no_grad():
            # PeftModel -> wrapped Transformers model.
            if hasattr(root, "get_base_model"):
                root = root.get_base_model()

            candidates = (
                ("model", "language_model", "embed_tokens"),
                ("language_model", "model", "embed_tokens"),
                ("model", "embed_tokens"),
            )
            for path in candidates:
                module = root
                try:
                    for part in path:
                        module = getattr(module, part)
                    return module.weight.device
                except (AttributeError, TypeError):
                    continue
        return next(model.parameters()).device

    @staticmethod
    def _prompt_text(prompt: Prompt) -> str:
        if prompt.system:
            return f"{prompt.system}\n\n{prompt.user}"
        return prompt.user

    def _normalized_hidden_stack(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompt: Prompt,
    ) -> list[torch.Tensor]:
        encoded = tokenizer(
            self._prompt_text(prompt),
            add_special_tokens=True,
            max_length=self.settings.max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        device = self._input_device(model)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states required by LTXConditioningDrift"
            )

        valid = attention_mask[0].bool()
        normalized: list[torch.Tensor] = []
        eps = self.settings.normalization_epsilon
        for hidden in hidden_states:
            values = hidden[0, valid].float()
            # Mirrors comfy/text_encoders/lt.py: each of the 49 representations
            # is normalized across its token and hidden dimensions before the
            # LTX projection sees the concatenated stack.
            values = 8.0 * (values - values.mean()) / (
                values.amax() - values.amin() + eps
            )
            normalized.append(values.to(device="cpu", dtype=torch.float16))
        return normalized

    def init(self, ctx: Context) -> None:
        print()
        print(
            "Loading LTX conditioning prompts from "
            f"[bold]{self.settings.prompts.dataset}[/]..."
        )
        loaded = ctx.load_prompts(self.settings.prompts)
        self.prompts = loaded[: self.settings.max_prompts]
        if not self.prompts:
            raise ValueError("LTXConditioningDrift requires at least one prompt")
        print(f"* [bold]{len(self.prompts)}[/] prompts selected")

        model, tokenizer = self._model_and_tokenizer(ctx)
        print("* Measuring baseline all-layer LTX conditioning...")
        self._baseline = [
            self._normalized_hidden_stack(model, tokenizer, prompt)
            for prompt in self.prompts
        ]
        layer_counts = {len(stack) for stack in self._baseline}
        if len(layer_counts) != 1:
            raise RuntimeError("Inconsistent hidden-state count across prompts")
        self._layer_count = next(iter(layer_counts))
        print(f"* [bold]{self._layer_count}[/] representations per prompt")

    def get_score(self, ctx: Context) -> Score:
        model, tokenizer = self._model_and_tokenizer(ctx)
        cosine_drifts: list[float] = []
        relative_rms: list[float] = []
        per_layer: list[list[float]] = [[] for _ in range(self._layer_count)]

        for prompt, baseline_stack in zip(
            self.prompts, self._baseline, strict=True
        ):
            current_stack = self._normalized_hidden_stack(model, tokenizer, prompt)
            if len(current_stack) != self._layer_count:
                raise RuntimeError("Hidden-state count changed during optimization")

            for layer_index, (current_cpu, baseline_cpu) in enumerate(
                zip(current_stack, baseline_stack, strict=True)
            ):
                current = current_cpu.float().reshape(-1)
                baseline = baseline_cpu.float().reshape(-1)
                drift = float(
                    1.0 - F.cosine_similarity(current, baseline, dim=0).item()
                )
                # Numerical rounding can produce a tiny negative value at the
                # unedited baseline; a drift metric must remain non-negative.
                drift = max(0.0, drift)
                cosine_drifts.append(drift)
                per_layer[layer_index].append(drift)
                relative_rms.append(
                    float(
                        torch.sqrt(torch.mean((current - baseline) ** 2))
                        / (torch.sqrt(torch.mean(baseline**2)) + 1e-12)
                    )
                )

        value = fmean(cosine_drifts)
        layer_means = [fmean(values) for values in per_layer]
        third = max(1, self._layer_count // 3)
        diagnostics = {
            "prompt_count": len(self.prompts),
            "representation_count": self._layer_count,
            "mean_cosine_drift": value,
            "max_layer_cosine_drift": max(layer_means),
            "mean_relative_rms": fmean(relative_rms),
            "early_mean_cosine_drift": fmean(layer_means[:third]),
            "middle_mean_cosine_drift": fmean(layer_means[third : 2 * third]),
            "late_mean_cosine_drift": fmean(layer_means[2 * third :]),
        }
        return Score(
            value=value,
            rich_display=f"{value:.6f}",
            md_display=f"{value:.6f}",
            diagnostics=diagnostics,
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0.0,
            rich_display="0 (by definition)",
            md_display="0 *(by definition)*",
            diagnostics={
                "prompt_count": len(self.prompts),
                "representation_count": self._layer_count,
            },
        )
