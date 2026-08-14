# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

from .config import GenerationBackend, QuantizationMethod, RowNormalization, Settings
from .generation_batch_selection import (
    GenerationBatchProbe,
    next_batch_candidate,
)
from .system import empty_cache
from .teacher_forced import per_row_conditional_nll
from .utils import Prompt, batchify, format_exception, print


class ModelLoadError(RuntimeError):
    """All configured dtypes failed to load the model."""


def get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    local_path = Path(model)
    if local_path.is_file():
        raise ValueError(
            "Heretic-MOE expects a Hugging Face model directory or repository "
            f"ID, not a standalone checkpoint file: {local_path}. Encoder-only "
            "safetensors such as ByT5 conditioning checkpoints require a "
            "separate conditioning-preservation workflow."
        )

    config, _ = PretrainedConfig.get_config_dict(model)

    if config.get("is_encoder_decoder"):
        architectures = config.get("architectures") or []
        architecture = ", ".join(architectures) or "unknown"
        raise ValueError(
            "Heretic-MOE requires a decoder-only causal language model; "
            f"{architecture} is an encoder-decoder architecture. Text encoders "
            "such as UMT5 and ByT5 do not produce chat refusals and need a "
            "separate conditioning-preservation workflow."
        )

    if "vision_config" in config:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


def project_fused_expert_chunk(
    original: Tensor,
    direction: Tensor,
    weight: float,
    normalization: RowNormalization,
) -> Tensor:
    """Apply the dense-path projection mechanics to a fused expert chunk.

    ``original`` has shape ``[experts, output_rows, input_features]``. The
    operation is exact because fused tensors are edited directly rather than
    represented through a low-rank adapter.
    """

    if original.dim() != 3:
        raise ValueError("Fused expert chunks must be three-dimensional")
    if direction.dim() != 1 or direction.shape[0] != original.shape[1]:
        raise ValueError("Direction size must match fused output rows")

    W = original.to(torch.float32)
    v = F.normalize(direction.to(device=W.device, dtype=torch.float32), dim=0)
    if normalization == RowNormalization.NONE:
        projection = torch.einsum("h,ehi->ei", v, W)
        return W - weight * v.view(1, -1, 1) * projection.unsqueeze(1)

    row_norms = LA.vector_norm(W, dim=2, keepdim=True)
    normalized = F.normalize(W, p=2, dim=2)
    projection = torch.einsum("h,ehi->ei", v, normalized)
    normalized_delta = weight * v.view(1, -1, 1) * projection.unsqueeze(1)

    if normalization == RowNormalization.PRE:
        return W - row_norms * normalized_delta
    if normalization == RowNormalization.FULL:
        adjusted = F.normalize(normalized - normalized_delta, p=2, dim=2)
        return adjusted * row_norms
    raise ValueError(f"Unsupported row normalization: {normalization}")


def low_rank_frobenius_squared(left: Tensor, right: Tensor) -> float:
    """Return ||left @ right||_F^2 without materializing the full matrix."""

    left_gram = left.T.to(torch.float32) @ left.to(torch.float32)
    right_gram = right.to(torch.float32) @ right.T.to(torch.float32)
    return float(torch.sum(left_gram * right_gram.T))


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype
    # Original weights of fused-expert MoE tensors, cached to keep abliteration reversible.
    _fused_experts_cache: dict[int, tuple[torch.nn.Parameter, Tensor]]

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or (
            isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
        )

    def _release_failed_cuda_batch(self) -> None:
        """Release retained generation state while keeping weights resident."""

        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                if "_cache" in vars(module):
                    module._cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _clear_prompt_runtime_cache(self, *, reset_role_probe: bool = False) -> None:
        """Discard tokenized prompts and tokenizer-specific runtime decisions."""

        self._prompt_token_cache = {}
        self._prompt_token_arena = None
        self._prompt_token_cache_signature = None
        if reset_role_probe:
            self._no_system_role = False

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self._batch_event_sink: Callable[[dict[str, object]], None] | None = None
        self._fused_experts_cache = {}
        self._last_edit_telemetry: dict[str, Any] = {"layers": [], "total": {}}
        self._edit_telemetry_accumulator: dict[
            tuple[str, int, str], dict[str, Any]
        ] = {}

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # Resolve and validate the architecture once, before loading tokenizer
        # or weights. This gives encoder-only/encoder-decoder checkpoints a
        # precise error instead of several misleading dtype load failures.
        self.model_class = get_model_class(settings.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            **self.revision_kwargs,
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if self.model_class == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        self.trusted_models: set[str] = set()

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = self.model_class.from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                )
            except Exception as error:  # noqa: BLE001 - dtype fallback must intercept any model-load failure
                self.model = None  # ty:ignore[invalid-assignment]
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise ModelLoadError("Failed to load model with all configured dtypes.")

        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Abliterable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def set_batch_event_sink(
        self, sink: Callable[[dict[str, object]], None] | None
    ) -> None:
        """Attach text-free batch-selection telemetry for a controller/worker."""

        self._batch_event_sink = sink

    def _emit_batch_event(self, event: str, mode: str, **values: object) -> None:
        sink = getattr(self, "_batch_event_sink", None)
        if sink is not None:
            sink({"event": event, "mode": mode, **values})

    def _apply_lora(self):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            lora_rank = self.settings.full_normalization_lora_rank

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        display_targets = sorted({name.rsplit(".", 1)[-1] for name in target_modules})
        print(
            f"* LoRA adapters initialized (target types: {', '.join(display_targets)})"
        )

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = self.model_class.from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            for name, module in self.model.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            for fused, original in self._fused_experts_cache.values():
                fused.data.copy_(original)
            return

        # Purge existing model object from memory to make space.
        self._clear_prompt_runtime_cache(reset_role_probe=True)
        self.model = None  # ty:ignore[invalid-assignment]
        self._fused_experts_cache = {}
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = self.model_class.from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self._apply_lora()

        self.needs_reload = False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers

        # Text-only models.
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules: dict[str, list[Module]] = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.linear.out_proj", layer.linear_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        with suppress(Exception):
            try_add("mlp.shared.down_proj", layer.mlp.shared_expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty:ignore[possibly-missing-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty:ignore[possibly-missing-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        # Fused experts are batched parameters, not modules, so get_layer_modules
        # cannot report them - it only collects nn.Module instances. They still
        # need a weight schedule of their own, and without this entry the search
        # never creates one and 92% of the weights go untouched.
        #
        # This used to work by accident: the shared expert registered under the
        # same key. Once it got a key of its own, the routed experts silently
        # lost theirs.
        if self._has_fused_experts():
            components.add("mlp.experts.down_proj")

        return sorted(components)

    def _has_fused_experts(self) -> bool:
        """Return whether any layer stores experts as a batched 3D parameter."""
        return next(self._iter_fused_expert_parameters(), None) is not None

    def _iter_fused_expert_parameters(self):
        """Yield supported fused routed-expert output projections once."""

        seen: set[int] = set()
        for layer_index, layer in enumerate(self.get_layers()):
            for block_name in ("mlp", "block_sparse_moe", "feed_forward", "moe"):
                block = getattr(layer, block_name, None)
                experts = getattr(block, "experts", None)
                if experts is None:
                    continue
                for param_name in ("down_proj", "w2", "output_linear"):
                    fused = getattr(experts, param_name, None)
                    if (
                        isinstance(fused, torch.nn.Parameter)
                        and fused.dim() == 3
                        and id(fused) not in seen
                    ):
                        seen.add(id(fused))
                        yield layer_index, fused
                        break

    def abliterate(
        self,
        residual_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        self._edit_telemetry_accumulator = {}
        self._last_edit_telemetry = {"layers": [], "total": {}}
        if direction_index is None:
            residual_direction = None
        else:
            residual_direction = self._interpolate_residual_direction(
                residual_directions,
                direction_index,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = cast(float, abs(layer_index - params.max_weight_position))

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                # A weight of 0 disables this component's ablation. reset_model() has
                # already left the adapter at identity, so abort before the otherwise
                # wasteful decomposition (which would also be operating on a zero matrix).
                if weight == 0:
                    continue

                if residual_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of residual_directions is the direction for the embeddings.
                    layer_residual_direction = residual_directions[layer_index + 1]
                else:
                    layer_residual_direction = residual_direction

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 residual direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_residual_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        # This cast is always valid. Type inference fails here because the
                        # bnb.functional module is not found by ty for some reason.
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                                base_weight.data,
                                quant_state,
                            ).to(torch.float32),
                        )

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)
                    W_base = W

                    if self.settings.row_normalization == RowNormalization.FULL:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = self.peft_config.r

                        # svd_lowrank is randomized:
                        # https://github.com/pytorch/pytorch/blob/20919052303c0b5ba87f8bf7e19237dc33ab09d3/torch/_lowrank.py#L108-L109
                        # Reseed immediately before the call so restoring a trial is independent of RNG history.
                        torch.manual_seed(self.settings.seed)
                        # "It's safe to call this function if CUDA is not available;
                        # in that case, it is silently ignored."
                        torch.cuda.manual_seed_all(self.settings.seed)  # ty:ignore[invalid-argument-type]
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)

                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

                    if self.settings.record_edit_telemetry:
                        self._accumulate_edit_telemetry(
                            component=component,
                            layer_index=layer_index,
                            path="dense_lora",
                            scheduled_weight=weight,
                            delta_squared=low_rank_frobenius_squared(lora_B, lora_A),
                            base_squared=float(
                                torch.sum(W_base.to(torch.float32) ** 2)
                            ),
                            parameter_count=W_base.numel(),
                        )

        # Fused-expert MoE blocks (e.g. Qwen3.5-MoE) are not reached by the loop above.
        self._abliterate_fused_experts(
            residual_directions, residual_direction, parameters
        )
        self._finalize_edit_telemetry()

    def _accumulate_edit_telemetry(
        self,
        *,
        component: str,
        layer_index: int,
        path: str,
        scheduled_weight: float,
        delta_squared: float,
        base_squared: float,
        parameter_count: int,
    ) -> None:
        key = (component, layer_index, path)
        entry = self._edit_telemetry_accumulator.setdefault(
            key,
            {
                "component": component,
                "layer": layer_index,
                "path": path,
                "row_normalization": self.settings.row_normalization.value,
                "scheduled_weight": float(scheduled_weight),
                "delta_squared": 0.0,
                "base_squared": 0.0,
                "parameter_count": 0,
                "module_count": 0,
            },
        )
        entry["delta_squared"] += max(0.0, float(delta_squared))
        entry["base_squared"] += max(0.0, float(base_squared))
        entry["parameter_count"] += int(parameter_count)
        entry["module_count"] += 1

    def _finalize_edit_telemetry(self) -> None:
        if not self.settings.record_edit_telemetry:
            return
        layers: list[dict[str, Any]] = []
        total_delta_squared = 0.0
        total_base_squared = 0.0
        total_parameters = 0
        for entry in sorted(
            self._edit_telemetry_accumulator.values(),
            key=lambda item: (item["component"], item["layer"], item["path"]),
        ):
            delta_squared = entry.pop("delta_squared")
            base_squared = entry.pop("base_squared")
            delta_fro = math.sqrt(delta_squared)
            base_fro = math.sqrt(base_squared)
            entry["delta_fro"] = delta_fro
            entry["base_fro"] = base_fro
            entry["relative_edit_fro"] = delta_fro / base_fro if base_fro > 0 else 0.0
            layers.append(entry)
            total_delta_squared += delta_squared
            total_base_squared += base_squared
            total_parameters += entry["parameter_count"]
        total_delta = math.sqrt(total_delta_squared)
        total_base = math.sqrt(total_base_squared)
        self._last_edit_telemetry = {
            "layers": layers,
            "total": {
                "delta_fro": total_delta,
                "base_fro": total_base,
                "relative_edit_fro": total_delta / total_base
                if total_base > 0
                else 0.0,
                "edited_parameter_count": total_parameters,
            },
        }

    def get_last_edit_telemetry(self) -> dict[str, Any]:
        """Return the dry numeric edit report produced by the latest trial."""

        return self._last_edit_telemetry

    def _abliterate_fused_experts(
        self,
        refusal_directions: Tensor,
        refusal_direction: Tensor | None,
        parameters: dict[str, AbliterationParameters],
    ) -> None:
        """Orthogonalize fused-expert MoE blocks.

        Some MoE implementations (e.g. transformers' ``Qwen3_5MoeExperts``) pack all experts
        into batched tensors instead of a ``ModuleList`` of Linear experts, so
        ``get_layer_modules()`` (which does ``for expert in layer.mlp.experts``) never reaches
        them and every routed expert is silently skipped. Since most of a MoE's refusal
        behaviour lives in the routed experts, abliteration then plateaus far above zero.

        Here the batched ``experts.down_proj`` of shape ``[num_experts, hidden, inter]`` is
        orthogonalized per expert against the same per-layer refusal direction and weight
        schedule as the dense path. The base tensor is edited directly (reversibly, via a
        cached original), so no LoRA adapter is involved and ``get_merged_model()`` needs no
        change.
        """
        if "mlp.experts.down_proj" not in parameters:
            return
        params = parameters["mlp.experts.down_proj"]
        try:
            hidden = self.model.config.get_text_config().hidden_size
        except (AttributeError, TypeError):
            hidden = getattr(self.model.config, "hidden_size", None)
        for layer_index, fused in self._iter_fused_expert_parameters():
            # Expect [num_experts, out=hidden, in=inter]; skip on unexpected orientation.
            if hidden is not None and fused.shape[1] != hidden:
                continue
            cache = self._fused_experts_cache
            if id(fused) not in cache:
                cache[id(fused)] = (fused, fused.data.clone())
            _, original = cache[id(fused)]
            # Restore the original before (re-)abliterating, so trials are independent.
            fused.data.copy_(original)
            distance = abs(layer_index - params.max_weight_position)
            if distance > params.min_weight_distance:
                continue
            weight = params.max_weight + (distance / params.min_weight_distance) * (
                params.min_weight - params.max_weight
            )
            if weight == 0:
                continue
            if refusal_direction is None:
                v = refusal_directions[layer_index + 1]
            else:
                v = refusal_direction
            v = F.normalize(v.to(torch.float32).to(fused.device), dim=0)
            chunk_size = self.settings.fused_expert_chunk_size
            for start in range(0, original.shape[0], chunk_size):
                stop = min(start + chunk_size, original.shape[0])
                original_chunk = original[start:stop]
                edited_chunk = project_fused_expert_chunk(
                    original_chunk,
                    v,
                    weight,
                    self.settings.row_normalization,
                )
                if self.settings.record_edit_telemetry:
                    original_fp32 = original_chunk.to(torch.float32)
                    delta = edited_chunk - original_fp32
                    self._accumulate_edit_telemetry(
                        component="mlp.experts.down_proj",
                        layer_index=layer_index,
                        path="fused_exact",
                        scheduled_weight=weight,
                        delta_squared=float(torch.sum(delta * delta)),
                        base_squared=float(torch.sum(original_fp32 * original_fp32)),
                        parameter_count=original_chunk.numel(),
                    )
                fused.data[start:stop].copy_(edited_chunk.to(fused.dtype))

    def _apply_template_safe(self, chats, **kwargs):
        """Retry without a system role when the template rejects it."""
        try:
            return self.tokenizer.apply_chat_template(chats, **kwargs)
        except Exception as error:
            if "system" not in str(error).lower():
                raise
            self._no_system_role = True
            plain = []
            for chat in chats:
                parts = [m["content"] for m in chat if m["content"]]
                plain.append([{"role": "user", "content": "\n\n".join(parts)}])
            return self.tokenizer.apply_chat_template(plain, **kwargs)

    def _render_chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        def build(with_system: bool):
            chats = []
            for prompt in prompts:
                if with_system and prompt.system:
                    chats.append(
                        [
                            {"role": "system", "content": prompt.system},
                            {"role": "user", "content": prompt.user},
                        ]
                    )
                else:
                    text = (
                        f"{prompt.system}\n\n{prompt.user}"
                        if prompt.system
                        else prompt.user
                    )
                    chats.append([{"role": "user", "content": text}])
            return chats

        rendered = cast(
            list[str],
            self._apply_template_safe(
                build(with_system=not getattr(self, "_no_system_role", False)),
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        if self.settings.response_prefix:
            rendered = [value + self.settings.response_prefix for value in rendered]
        return rendered

    @staticmethod
    def _interpolate_residual_direction(
        residual_directions: Tensor,
        direction_index: float,
    ) -> Tensor:
        """Interpolate one layer direction without reading past an endpoint."""

        shifted = float(direction_index) + 1.0
        weight, integral = math.modf(shifted)
        index = int(integral)
        if index < 0 or index >= len(residual_directions):
            raise IndexError(
                f"direction index {direction_index} is outside residual map "
                f"with {len(residual_directions)} rows"
            )
        direction = residual_directions[index]
        if weight != 0.0:
            if index + 1 >= len(residual_directions):
                raise IndexError(
                    f"fractional direction index {direction_index} exceeds "
                    "the residual map endpoint"
                )
            direction = direction.lerp(residual_directions[index + 1], weight)
        return F.normalize(direction, p=2, dim=0)

    def _prompt_cache_signature(self) -> tuple[str, bool, str, str]:
        return (
            str(getattr(self.settings, "response_prefix", None) or ""),
            bool(getattr(self, "_no_system_role", False)),
            str(getattr(self.tokenizer, "name_or_path", "")),
            str(getattr(self.tokenizer, "chat_template", "")),
        )

    @staticmethod
    def _prompt_cache_key(prompt: Prompt) -> tuple[str, str]:
        return prompt.system, prompt.user

    def prepare_prompt_cache(self, prompts: Sequence[Prompt]) -> dict[str, int]:
        """Tokenize fixed prompts once and retain compact CPU token IDs.

        The persistent cache intentionally stays in ordinary RAM. Generation
        collates only the current batch into one pinned CPU allocation and copies
        it asynchronously, leaving VRAM available for KV cache and larger batches.
        """

        signature = self._prompt_cache_signature()
        cache_signature = getattr(self, "_prompt_token_cache_signature", None)
        if cache_signature != signature:
            self._prompt_token_cache = {}
            self._prompt_token_cache_signature = signature
            self._prompt_token_arena = None
        cache: dict[tuple[str, str], Tensor] = getattr(self, "_prompt_token_cache", {})
        self._prompt_token_cache = cache

        unique: dict[tuple[str, str], Prompt] = {}
        for prompt in prompts:
            unique.setdefault(self._prompt_cache_key(prompt), prompt)
        missing = [prompt for key, prompt in unique.items() if key not in cache]
        if missing:
            self._prompt_token_arena = None
        new_rows = 0
        new_tokens = 0
        for start in range(0, len(missing), 4096):
            chunk = missing[start : start + 4096]
            rendered = self._render_chat_prompts(chunk)
            # A tokenizer can discover that the model's chat template rejects a
            # system role. That changes the rendered form and invalidates any
            # entries produced under the previous template mode.
            final_signature = self._prompt_cache_signature()
            if final_signature != self._prompt_token_cache_signature:
                cache.clear()
                self._prompt_token_cache_signature = final_signature
            encoded = self.tokenizer(
                rendered,
                padding=False,
                return_token_type_ids=False,
            )
            rows = encoded["input_ids"]
            if isinstance(rows, Tensor):
                rows = rows.tolist()
            if len(rows) != len(chunk):
                raise ValueError("tokenizer returned an unaligned prompt batch")
            for prompt, raw_ids in zip(chunk, rows, strict=True):
                ids = torch.tensor(raw_ids, dtype=torch.int32, device="cpu")
                if ids.ndim != 1 or ids.numel() == 0:
                    raise ValueError("tokenizer returned an empty prompt")
                cache[self._prompt_cache_key(prompt)] = ids.contiguous()
                new_rows += 1
                new_tokens += int(ids.numel())
        return {
            "rows": len(prompts),
            "unique": len(unique),
            "new": new_rows,
            "tokens": new_tokens,
        }

    def pin_prompt_cache(self) -> dict[str, int | bool]:
        """Pack cached IDs into one contiguous, optionally page-locked arena."""

        cache: dict[tuple[str, str], Tensor] = getattr(self, "_prompt_token_cache", {})
        total = sum(int(row.numel()) for row in cache.values())
        use_pin = bool(
            torch.cuda.is_available() and torch.device(self.model.device).type == "cuda"
        )
        arena = torch.empty(
            total,
            dtype=torch.int32,
            device="cpu",
            pin_memory=use_pin,
        )
        offset = 0
        for key, row in tuple(cache.items()):
            stop = offset + int(row.numel())
            arena[offset:stop].copy_(row)
            cache[key] = arena[offset:stop]
            offset = stop
        self._prompt_token_arena = arena
        return {"rows": len(cache), "tokens": total, "pinned": use_pin}

    def _cached_prompt_token_ids(self, prompts: Sequence[Prompt]) -> list[Tensor]:
        self.prepare_prompt_cache(prompts)
        cache = self._prompt_token_cache
        return [cache[self._prompt_cache_key(prompt)] for prompt in prompts]

    def _collate_cached_prompts(self, prompts: Sequence[Prompt]) -> BatchEncoding:
        rows = self._cached_prompt_token_ids(prompts)
        if not rows:
            raise ValueError("prompts must not be empty")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
        maximum = max(int(row.numel()) for row in rows)
        bucket_multiple = int(
            getattr(self.settings, "generation_prompt_bucket_multiple", 0)
        )
        if bucket_multiple > 0:
            maximum = math.ceil(maximum / bucket_multiple) * bucket_multiple
        device = self.model.device
        pin_memory = bool(
            torch.cuda.is_available() and torch.device(device).type == "cuda"
        )
        input_ids = torch.full(
            (len(rows), maximum),
            int(pad_token_id),
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        attention_mask = torch.zeros(
            (len(rows), maximum),
            dtype=torch.bool,
            device="cpu",
            pin_memory=pin_memory,
        )
        for index, row in enumerate(rows):
            offset = maximum - int(row.numel())
            input_ids[index, offset:] = row
            attention_mask[index, offset:] = True
        return BatchEncoding(
            {
                "input_ids": input_ids.to(device, non_blocking=pin_memory),
                "attention_mask": attention_mask.to(device, non_blocking=pin_memory),
            }
        )

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        inputs = self._collate_cached_prompts(prompts)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        generation_kwargs = dict(kwargs)
        backend = getattr(
            self.settings,
            "generation_backend",
            GenerationBackend.DYNAMIC_EAGER,
        )
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 0) or 0)
        if (
            backend == GenerationBackend.COMPILED_STATIC
            or backend == GenerationBackend.COMPILED_STATIC.value
        ) and max_new_tokens > 1:
            from transformers.generation.configuration_utils import CompileConfig

            mode = str(getattr(self.settings, "generation_compile_mode", "default"))
            compile_config = getattr(self, "_generation_compile_config", None)
            if compile_config is None or compile_config.mode != mode:
                compile_config = CompileConfig(mode=mode)
                self._generation_compile_config = compile_config
            generation_kwargs.setdefault("cache_implementation", "static")
            generation_kwargs.setdefault("compile_config", compile_config)

        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses: list[str] = []
        for batch in batchify(prompts, self.settings.batch_size):
            responses.extend(
                self.get_responses(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                )
            )

        return responses

    def get_response_artifacts_with_prefill_residuals(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], list[list[int]], Tensor]:
        """Generate once and return text, exact generated IDs, and prefill residuals.

        Forward hooks retain only the first invocation, which is the prompt prefill.
        Token-by-token decode activations are neither stored nor returned. The compact
        ``[prompt, embedding+layers, hidden]`` tensor is always float32 on CPU so the
        hooks do not hold GPU storage after generation.
        """

        if not prompts:
            raise ValueError("prompts must not be empty")
        modules: list[Module] = [self.model.get_input_embeddings(), *self.get_layers()]
        captured: list[Tensor | None] = [None] * len(modules)
        handles = []

        def capture(index: int):
            def hook(_module: Module, _inputs: Any, output: Any) -> None:
                if captured[index] is not None:
                    return
                value = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(value, Tensor) or value.ndim != 3:
                    raise ValueError(
                        f"prefill hook {index} did not receive [batch,sequence,hidden]"
                    )
                # Keep only one position per layer on-device. Moving each hook
                # result to CPU here serializes the forward pass behind dozens
                # of tiny device synchronizations. Stack first and transfer once.
                captured[index] = value[:, -1, :].detach().clone()

            return hook

        for index, module in enumerate(modules):
            handles.append(module.register_forward_hook(capture(index)))
        try:
            inputs, outputs = self.generate(
                prompts,
                max_new_tokens=self.settings.max_response_length,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [index for index, value in enumerate(captured) if value is None]
        if missing:
            raise RuntimeError(f"prefill residual hooks did not fire: {missing}")
        residuals = torch.stack(
            [cast(Tensor, value) for value in captured],
            dim=1,
        ).to(torch.float32)
        if 0 <= self.settings.winsorization_quantile < 1:
            thresholds = torch.quantile(
                torch.abs(residuals),
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)
        residuals = residuals.cpu().contiguous()

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        sequences = cast(Tensor, sequences)
        generated = sequences[:, cast(Tensor, inputs["input_ids"]).shape[1] :]
        responses = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=skip_special_tokens,
        )
        eos_value = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = (
            {int(value) for value in eos_value}
            if isinstance(eos_value, (tuple, list, set))
            else ({int(eos_value)} if eos_value is not None else set())
        )
        pad_value = getattr(self.tokenizer, "pad_token_id", None)
        token_ids: list[list[int]] = []
        for raw_row in generated.detach().cpu().tolist():
            row = [int(value) for value in raw_row]
            stop = next(
                (index + 1 for index, value in enumerate(row) if value in eos_ids),
                None,
            )
            if stop is not None:
                row = row[:stop]
            elif pad_value is not None:
                while row and row[-1] == int(pad_value):
                    row.pop()
            token_ids.append(row)
        return responses, token_ids, residuals

    def get_responses_with_prefill_residuals(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], Tensor]:
        """Compatibility wrapper returning generated text and prefill residuals."""

        responses, _, residuals = self.get_response_artifacts_with_prefill_residuals(
            prompts,
            skip_special_tokens=skip_special_tokens,
        )
        return responses, residuals

    def get_response_artifacts_with_prefill_residuals_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        progress: Callable[[int, int, int], None] | None = None,
    ) -> tuple[list[str], list[list[int]], Tensor]:
        """Batched text, generated token IDs, and prefill residual capture."""

        if not prompts:
            raise ValueError("prompts must not be empty")
        configured = int(self.settings.batch_size)
        automatic = configured == 0
        tuning = automatic and int(
            getattr(self, "_adaptive_generation_batch_size", 0)
        ) <= 0
        batch_size = (
            int(getattr(self, "_adaptive_generation_batch_size", 0))
            if automatic
            else configured
        )
        if batch_size <= 0:
            batch_size = min(int(self.settings.max_batch_size), len(prompts))
        if tuning:
            self._emit_batch_event(
                "batch_probe", "generation", batch_size=batch_size
            )
        # Similar lengths reduce left-padding without changing the row contract.
        if hasattr(self, "tokenizer"):
            prompt_lengths = [
                int(row.numel()) for row in self._cached_prompt_token_ids(prompts)
            ]
        else:
            # Some narrow unit-test doubles replace the artifact capture method
            # and intentionally have no tokenizer. Production Models always do.
            prompt_lengths = [
                len(prompt.system) + len(prompt.user) for prompt in prompts
            ]
        order = sorted(range(len(prompts)), key=prompt_lengths.__getitem__)
        responses: list[str | None] = [None] * len(prompts)
        token_ids: list[list[int] | None] = [None] * len(prompts)
        residuals: list[Tensor | None] = [None] * len(prompts)
        position = 0
        while position < len(order):
            selected = order[position : position + batch_size]
            batch = [prompts[index] for index in selected]
            try:
                batch_responses, batch_token_ids, batch_residuals = (
                    self.get_response_artifacts_with_prefill_residuals(
                        batch,
                        skip_special_tokens=skip_special_tokens,
                    )
                )
            except BaseException as error:
                if not automatic or batch_size == 1 or not self._is_cuda_oom(error):
                    raise
                next_batch_size = max(1, batch_size // 2)
                if tuning:
                    self._emit_batch_event(
                        "batch_backoff",
                        "generation",
                        batch_size=batch_size,
                        next_batch_size=next_batch_size,
                        reason="OOM",
                    )
                batch_size = next_batch_size
                self._adaptive_generation_batch_size = batch_size
                self._release_failed_cuda_batch()
                continue
            if tuning:
                self._emit_batch_event(
                    "batch_selected", "generation", batch_size=batch_size
                )
                tuning = False
            for local, original in enumerate(selected):
                responses[original] = batch_responses[local]
                token_ids[original] = batch_token_ids[local]
                residuals[original] = batch_residuals[local]
            position += len(selected)
            if progress is not None:
                progress(position, len(order), batch_size)
        if automatic:
            self._adaptive_generation_batch_size = batch_size
        if any(value is None for value in responses + token_ids + residuals):
            raise RuntimeError("adaptive generation batch lost row coverage")
        return (
            cast(list[str], responses),
            cast(list[list[int]], token_ids),
            torch.stack(cast(list[Tensor], residuals), dim=0),
        )

    def prewarm_generation_backend(
        self,
        prompts: Sequence[Prompt],
        *,
        expected_rows: int | Sequence[int] | None = None,
    ) -> dict[str, object]:
        """Compile each resident generation shape once before timed trials."""

        backend = getattr(
            self.settings,
            "generation_backend",
            GenerationBackend.DYNAMIC_EAGER,
        )
        if (
            backend != GenerationBackend.COMPILED_STATIC
            and backend != GenerationBackend.COMPILED_STATIC.value
        ) or int(self.settings.max_response_length) <= 1:
            return {"status": "DISABLED", "batch_size": 0, "shapes": []}
        if not prompts:
            raise ValueError("generation prewarm requires at least one prompt")
        row_counts = (
            (len(prompts),)
            if expected_rows is None
            else (
                (int(expected_rows),)
                if isinstance(expected_rows, int)
                else tuple(int(value) for value in expected_rows)
            )
        )
        if not row_counts or any(row_count <= 0 for row_count in row_counts):
            raise ValueError("expected_rows must contain positive row counts")
        row_count = max(row_counts)
        configured = int(self.settings.batch_size)
        automatic = configured == 0
        batch_size = configured or int(
            getattr(self, "_adaptive_generation_batch_size", 0)
        )
        if batch_size <= 0:
            batch_size = min(int(self.settings.max_batch_size), row_count)
        token_rows = self._cached_prompt_token_ids(prompts)
        bucket_multiple = int(
            getattr(self.settings, "generation_prompt_bucket_multiple", 0)
        )

        def padded_width(row: Tensor) -> int:
            width = int(row.numel())
            if bucket_multiple > 0:
                width = math.ceil(width / bucket_multiple) * bucket_multiple
            return width

        by_width: dict[int, list[Prompt]] = {}
        for prompt, row in zip(prompts, token_rows, strict=True):
            by_width.setdefault(padded_width(row), []).append(prompt)

        while True:
            tail_sizes = sorted(
                {count % batch_size for count in row_counts if count % batch_size}
            )
            shapes: list[tuple[int, int, list[Prompt]]] = []
            for width, candidates in sorted(by_width.items()):
                sample = [
                    candidates[index % len(candidates)] for index in range(batch_size)
                ]
                shapes.append((batch_size, width, sample))
            for tail_size in tail_sizes:
                width = max(by_width)
                candidates = by_width[width]
                sample = [
                    candidates[index % len(candidates)] for index in range(tail_size)
                ]
                shapes.append((tail_size, width, sample))
            try:
                for _size, _width, sample in shapes:
                    self.get_response_artifacts_with_prefill_residuals(
                        sample,
                        skip_special_tokens=True,
                    )
                break
            except BaseException as error:
                if not automatic or batch_size == 1 or not self._is_cuda_oom(error):
                    raise
                batch_size = max(1, batch_size // 2)
                self._release_failed_cuda_batch()
        if automatic:
            self._adaptive_generation_batch_size = batch_size
        return {
            "status": "PASS",
            "batch_size": batch_size,
            "shapes": [[size, width] for size, width, _sample in shapes],
        }

    def validate_generation_batch_size(
        self,
        prompts: Sequence[Prompt],
        *,
        batch_size: int,
        expected_rows: int,
    ) -> dict[str, object]:
        """Run one short real 100-token batch to prove the tuned shape fits."""

        if not prompts or expected_rows <= 0:
            raise ValueError("generation batch validation requires prompts and rows")
        validation_size = min(int(batch_size), int(expected_rows), len(prompts))
        if validation_size <= 0:
            raise ValueError("validation batch size must be positive")
        # Keep this probe conservative and stable across runtime response limits:
        # it proves that the selected batch can sustain the v3 100-token contract.
        max_new_tokens = 100
        prompt_rows = self._cached_prompt_token_ids(prompts)
        longest = sorted(
            range(len(prompts)),
            key=lambda index: int(prompt_rows[index].numel()),
            reverse=True,
        )
        sample = [prompts[longest[index % len(longest)]] for index in range(validation_size)]
        widths = [int(row.numel()) for row in self._cached_prompt_token_ids(sample)]
        width = max(widths)
        bucket_multiple = int(
            getattr(self.settings, "generation_prompt_bucket_multiple", 0)
        )
        if bucket_multiple > 0:
            width = math.ceil(width / bucket_multiple) * bucket_multiple
        baseline_free, baseline_total, _baseline_peak = self._cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
        inputs = outputs = None
        measured_free = measured_total = measured_peak = 0
        validation_error: BaseException | None = None
        try:
            inputs, outputs = self.generate(
                sample,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
            )
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            if (
                not isinstance(sequences, Tensor)
                or sequences.ndim != 2
                or int(sequences.shape[0]) != validation_size
            ):
                raise ValueError("validation generation returned an invalid batch shape")
            if input_ids is not None and (
                not isinstance(input_ids, Tensor)
                or input_ids.ndim != 2
                or int(input_ids.shape[0]) != validation_size
            ):
                raise ValueError("validation collation returned an invalid batch shape")
            if input_ids is not None:
                generated_width = int(sequences.shape[1]) - int(input_ids.shape[1])
                if generated_width != max_new_tokens:
                    raise ValueError(
                        "validation generation returned "
                        f"{generated_width} new tokens; expected {max_new_tokens} new tokens"
                    )
            torch.cuda.synchronize()
            measured_free, measured_total, measured_peak = self._cuda_memory_snapshot()
        except BaseException as error:  # noqa: BLE001 - restore CUDA cache before OOM propagation
            validation_error = error
        finally:
            del inputs, outputs
            self._release_generation_probe_cache()
        recovered_free, recovered_total, _recovered_peak = self._cuda_memory_snapshot()
        if baseline_total != recovered_total or (
            validation_error is None and baseline_total != measured_total
        ):
            raise RuntimeError("CUDA device memory total changed during batch validation")
        tolerance = (
            int(getattr(self.settings, "generation_batch_recovery_tolerance_mib", 256))
            * 1024**2
        )
        if recovered_free + tolerance < baseline_free:
            leaked = (baseline_free - recovered_free) / 1024**2
            raise RuntimeError(
                "generation batch validation did not release its CUDA cache "
                f"({leaked:.0f} MiB still resident)"
            )
        if validation_error is not None:
            raise validation_error.with_traceback(validation_error.__traceback__)
        required_free = max(
            int(float(self.settings.batch_size_vram_headroom_gib) * 1024**3),
            int(
                float(self.settings.batch_size_vram_headroom_fraction)
                * measured_total
            ),
        )
        status = "PASS" if measured_free >= required_free else "INSUFFICIENT_HEADROOM"
        return {
            "status": status,
            "batch_size": validation_size,
            "rows": validation_size,
            "expected_rows": int(expected_rows),
            "max_new_tokens": max_new_tokens,
            "prompt_width": int(width),
            "baseline_free_bytes": int(baseline_free),
            "min_free_bytes": int(measured_free),
            "working_set_bytes": int(max(0, baseline_free - measured_free)),
            "required_free_bytes": int(required_free),
            "recovered_free_bytes": int(recovered_free),
            "peak_allocated_bytes": int(measured_peak),
        }

    def _probe_generation_batch(
        self,
        prompts: Sequence[Prompt],
        batch_size: int,
    ) -> GenerationBatchProbe:
        prompt_rows = self._cached_prompt_token_ids(prompts)
        longest = sorted(
            range(len(prompts)),
            key=lambda index: int(prompt_rows[index].numel()),
            reverse=True,
        )
        sample = [prompts[longest[index % len(longest)]] for index in range(batch_size)]
        widths = [int(row.numel()) for row in self._cached_prompt_token_ids(sample)]
        width = max(widths)
        bucket_multiple = int(
            getattr(self.settings, "generation_prompt_bucket_multiple", 0)
        )
        if bucket_multiple > 0:
            width = math.ceil(width / bucket_multiple) * bucket_multiple
        self._release_generation_probe_cache()
        baseline_free, baseline_total, _baseline_peak = self._cuda_memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
        inputs = outputs = None
        measured_free = measured_total = measured_peak = 0
        generation_config = getattr(self.model, "generation_config", None)
        configured_max_length = (
            getattr(generation_config, "max_length", None)
            if generation_config is not None
            else None
        )
        probe_error: BaseException | None = None
        try:
            if generation_config is not None:
                generation_config.max_length = None
            inputs, outputs = self.generate(
                sample,
                max_new_tokens=1,
                cache_implementation="static",
                max_cache_len=width + int(self.settings.max_response_length),
                disable_compile=True,
            )
            torch.cuda.synchronize()
            measured_free, measured_total, measured_peak = self._cuda_memory_snapshot()
        except BaseException as error:  # noqa: BLE001 - restore CUDA state before re-raising OOM/interrupts
            probe_error = error
        finally:
            if generation_config is not None:
                generation_config.max_length = configured_max_length
            del inputs, outputs
            self._release_generation_probe_cache()
        recovered_free, recovered_total, _recovered_peak = self._cuda_memory_snapshot()
        if baseline_total != recovered_total or (
            probe_error is None and baseline_total != measured_total
        ):
            raise RuntimeError("CUDA device memory total changed during batch probe")
        tolerance = (
            int(getattr(self.settings, "generation_batch_recovery_tolerance_mib", 256))
            * 1024**2
        )
        if recovered_free + tolerance < baseline_free:
            leaked = (baseline_free - recovered_free) / 1024**2
            raise RuntimeError(
                "generation batch probe did not release its CUDA cache "
                f"({leaked:.0f} MiB still resident)"
            )
        if probe_error is not None:
            raise probe_error.with_traceback(probe_error.__traceback__)
        return GenerationBatchProbe(
            batch_size=batch_size,
            free_bytes=int(measured_free),
            total_bytes=int(measured_total),
            peak_allocated_bytes=int(measured_peak),
            baseline_free_bytes=int(baseline_free),
            recovered_free_bytes=int(recovered_free),
        )

    def _release_generation_probe_cache(self) -> None:
        for module in self.model.modules():
            if "_cache" in vars(module):
                module._cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _cuda_memory_snapshot() -> tuple[int, int, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes), int(total_bytes), int(torch.cuda.max_memory_allocated())

    def autotune_generation_batch_size(
        self,
        prompts: Sequence[Prompt],
        *,
        expected_rows: int,
    ) -> dict[str, object]:
        """Choose a resident batch with isolated fixed-step VRAM probes."""

        if not prompts or expected_rows <= 0:
            raise ValueError("generation autotune requires prompts and rows")
        maximum = min(int(self.settings.max_batch_size), expected_rows)
        granularity = int(self.settings.generation_batch_granularity)
        candidate = min(int(self.settings.generation_batch_probe_start), maximum)
        candidate = max(1, candidate)
        probes: list[GenerationBatchProbe] = []
        best: GenerationBatchProbe | None = None
        self._emit_batch_event("batch_probe", "generation", batch_size=candidate)
        while candidate <= maximum:
            try:
                probe = self._probe_generation_batch(prompts, candidate)
            except BaseException as error:
                if not self._is_cuda_oom(error):
                    raise
                self._release_failed_cuda_batch()
                if best is None and candidate > 1:
                    next_candidate = max(1, candidate // 2)
                    self._emit_batch_event(
                        "batch_backoff",
                        "generation",
                        batch_size=candidate,
                        next_batch_size=next_candidate,
                        reason="OOM",
                    )
                    candidate = next_candidate
                    continue
                break
            probes.append(probe)
            self._emit_batch_event(
                "batch_probe_result",
                "generation",
                batch_size=probe.batch_size,
                free_gib=round(probe.free_bytes / 1024**3, 3),
            )
            hard_reserve = max(
                int(float(self.settings.batch_size_vram_headroom_gib) * 1024**3),
                int(
                    float(self.settings.batch_size_vram_headroom_fraction)
                    * probe.total_bytes
                ),
            )
            if probe.free_bytes < hard_reserve:
                if best is None and candidate > 1:
                    candidate = max(1, candidate // 2)
                    continue
                break
            best = probe
            if candidate < int(self.settings.generation_batch_probe_start):
                break
            preferred_reserve = max(
                hard_reserve,
                int(
                    float(self.settings.generation_batch_target_headroom_fraction)
                    * probe.total_bytes
                ),
            )
            next_candidate = next_batch_candidate(
                current_batch_size=probe.batch_size,
                current_free_bytes=probe.free_bytes,
                required_free_bytes=preferred_reserve,
                maximum_batch_size=maximum,
                granularity=granularity,
            )
            if next_candidate is None or next_candidate <= candidate:
                break
            candidate = next_candidate
        if best is None:
            raise RuntimeError(
                "no generation batch candidate satisfies the CUDA VRAM reserve"
            )
        candidate = best.batch_size
        validation: dict[str, object] | None = None
        validation_attempts: list[dict[str, object]] = []
        while True:
            self._emit_batch_event(
                "batch_validation",
                "generation",
                batch_size=candidate,
                max_new_tokens=100,
            )
            try:
                validation = self.validate_generation_batch_size(
                    prompts,
                    batch_size=candidate,
                    expected_rows=expected_rows,
                )
            except BaseException as error:
                if not self._is_cuda_oom(error):
                    raise
                validation_attempts.append(
                    {"batch_size": candidate, "status": "OOM"}
                )
                self._release_failed_cuda_batch()
                if candidate == 1:
                    raise RuntimeError(
                        "real 100-token validation OOM at batch size 1"
                    ) from error
                candidate = max(1, candidate // 2)
                continue
            validation_attempts.append(dict(validation))
            self._emit_batch_event(
                "batch_validation_result",
                "generation",
                batch_size=candidate,
                status=str(validation["status"]),
                free_gib=round(int(validation["min_free_bytes"]) / 1024**3, 3),
                recovered_gib=round(
                    int(validation["recovered_free_bytes"]) / 1024**3,
                    3,
                ),
            )
            if validation["status"] == "PASS":
                break
            if candidate == 1:
                raise RuntimeError(
                    "real 100-token validation misses the CUDA VRAM reserve at batch size 1"
                )
            candidate = max(1, candidate // 2)
        self._adaptive_generation_batch_size = candidate
        self._emit_batch_event(
            "batch_selected", "generation", batch_size=candidate
        )
        return {
            "status": "PASS",
            "batch_size": candidate,
            "probes": [
                {
                    "batch_size": probe.batch_size,
                    "free_gib": probe.free_bytes / 1024**3,
                    "baseline_free_gib": probe.baseline_free_bytes / 1024**3,
                    "recovered_free_gib": probe.recovered_free_bytes / 1024**3,
                    "peak_allocated_gib": probe.peak_allocated_bytes / 1024**3,
                }
                for probe in probes
            ],
            "validation": validation,
            "validation_attempts": validation_attempts,
        }

    def get_responses_with_prefill_residuals_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], Tensor]:
        """Batched form of :meth:`get_responses_with_prefill_residuals`."""

        if not prompts:
            raise ValueError("prompts must not be empty")
        responses: list[str] = []
        residuals: list[Tensor] = []
        for batch in batchify(prompts, self.settings.batch_size):
            batch_responses, batch_residuals = (
                self.get_responses_with_prefill_residuals(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                )
            )
            responses.extend(batch_responses)
            residuals.append(batch_residuals)
        return responses, torch.cat(residuals, dim=0)

    def get_conditional_nll(
        self,
        prompts: list[Prompt],
        target_token_ids: Sequence[Sequence[int]],
    ) -> list[float]:
        """Score fixed clean targets without autoregressive generation."""

        if not prompts or len(prompts) != len(target_token_ids):
            raise ValueError(
                "prompts and target token IDs must be non-empty and aligned"
            )
        prompt_token_ids = self._cached_prompt_token_ids(prompts)
        if len(prompt_token_ids) != len(prompts):
            raise ValueError("tokenizer returned an unaligned prompt batch")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

        pairs = list(zip(prompt_token_ids, target_token_ids, strict=True))
        order = sorted(
            range(len(pairs)),
            key=lambda index: len(pairs[index][0]) + len(pairs[index][1]),
        )
        configured = int(
            getattr(
                self.settings,
                "conditional_nll_batch_size",
                self.settings.batch_size,
            )
        )
        automatic = configured == 0
        cached_batch_size = (
            int(getattr(self, "_adaptive_nll_batch_size", 0)) if automatic else 0
        )
        tuning = automatic and cached_batch_size <= 0
        batch_size = cached_batch_size if automatic else configured
        if batch_size <= 0:
            generation_batch_size = int(
                getattr(self, "_adaptive_generation_batch_size", 0)
            )
            if generation_batch_size > 0:
                # Teacher-forced NLL materializes full-vocabulary logits for
                # every token.  It therefore needs a substantially smaller
                # starting point than autoregressive generation, even though
                # both operate over the same prompt pool.
                batch_size = max(1, generation_batch_size // 4)
            else:
                batch_size = int(
                    getattr(self.settings, "generation_batch_probe_start", 8)
                )
            batch_size = min(
                int(self.settings.max_batch_size), len(pairs), batch_size
            )
        if tuning:
            self._emit_batch_event("batch_probe", "conditional NLL", batch_size=batch_size)
        largest_passing_batch = 0
        smallest_failing_batch: int | None = None
        values: list[float | None] = [None] * len(pairs)
        position = 0
        while position < len(order):
            selected = order[position : position + batch_size]
            batch = [pairs[index] for index in selected]
            sequences: list[tuple[Tensor, Tensor]] = []
            prompt_lengths: list[int] = []
            for raw_prompt_ids, raw_target_ids in batch:
                prompt_ids = raw_prompt_ids.to(dtype=torch.long, device="cpu")
                target_ids = torch.as_tensor(
                    raw_target_ids, dtype=torch.long, device="cpu"
                )
                if prompt_ids.numel() == 0 or target_ids.numel() == 0:
                    raise ValueError(
                        "prompt and target token sequences must be non-empty"
                    )
                sequences.append((prompt_ids, target_ids))
                prompt_lengths.append(int(prompt_ids.numel()))
            maximum = max(
                int(prompt_ids.numel() + target_ids.numel())
                for prompt_ids, target_ids in sequences
            )
            device = self.model.device
            pin_memory = bool(
                torch.cuda.is_available() and torch.device(device).type == "cuda"
            )
            cpu_input_ids = torch.full(
                (len(batch), maximum),
                int(pad_token_id),
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_attention_mask = torch.zeros(
                (len(batch), maximum),
                dtype=torch.bool,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_labels = torch.full(
                (len(batch), maximum),
                -100,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
            for row, ((prompt_ids, target_ids), prompt_length) in enumerate(
                zip(sequences, prompt_lengths, strict=True)
            ):
                target_length = int(target_ids.numel())
                length = prompt_length + target_length
                cpu_input_ids[row, :prompt_length].copy_(prompt_ids)
                cpu_input_ids[row, prompt_length:length].copy_(target_ids)
                cpu_attention_mask[row, :length] = True
                cpu_labels[row, prompt_length:length].copy_(target_ids)
            input_ids = cpu_input_ids.to(device, non_blocking=pin_memory)
            attention_mask = cpu_attention_mask.to(device, non_blocking=pin_memory)
            labels = cpu_labels.to(device, non_blocking=pin_memory)
            outputs = None
            measured_free = measured_total = 0
            try:
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                batch_values = per_row_conditional_nll(outputs.logits, labels)
                if automatic and torch.cuda.is_available():
                    measured_free, measured_total, _ = self._cuda_memory_snapshot()
            except BaseException as error:
                del input_ids, attention_mask, labels, outputs
                if not automatic or batch_size == 1 or not self._is_cuda_oom(error):
                    raise
                smallest_failing_batch = (
                    batch_size
                    if smallest_failing_batch is None
                    else min(smallest_failing_batch, batch_size)
                )
                next_batch_size = (
                    max(
                        largest_passing_batch,
                        (largest_passing_batch + smallest_failing_batch) // 2,
                    )
                    if largest_passing_batch > 0
                    else max(1, batch_size // 2)
                )
                if tuning:
                    self._emit_batch_event(
                        "batch_backoff",
                        "conditional NLL",
                        batch_size=batch_size,
                        next_batch_size=next_batch_size,
                        reason="OOM",
                    )
                batch_size = next_batch_size
                self._adaptive_nll_batch_size = batch_size
                self._release_failed_cuda_batch()
                continue
            required_free = max(
                int(
                    float(
                        getattr(
                            self.settings,
                            "batch_size_vram_headroom_gib",
                            0.0,
                        )
                    )
                    * 1024**3
                ),
                int(
                    float(
                        getattr(
                            self.settings,
                            "batch_size_vram_headroom_fraction",
                            0.10,
                        )
                    )
                    * measured_total
                ),
            )
            if automatic and measured_free < required_free:
                del input_ids, attention_mask, labels, outputs, batch_values
                if batch_size == 1:
                    self._release_failed_cuda_batch()
                    raise RuntimeError(
                        "conditional NLL cannot preserve the configured VRAM reserve "
                        "at batch size 1"
                    )
                smallest_failing_batch = (
                    batch_size
                    if smallest_failing_batch is None
                    else min(smallest_failing_batch, batch_size)
                )
                next_batch_size = (
                    max(
                        largest_passing_batch,
                        (largest_passing_batch + smallest_failing_batch) // 2,
                    )
                    if largest_passing_batch > 0
                    else max(1, batch_size // 2)
                )
                if tuning:
                    self._emit_batch_event(
                        "batch_backoff",
                        "conditional NLL",
                        batch_size=batch_size,
                        next_batch_size=next_batch_size,
                        reason="VRAM reserve",
                    )
                batch_size = next_batch_size
                self._adaptive_nll_batch_size = batch_size
                self._release_failed_cuda_batch()
                continue
            if automatic and tuning and smallest_failing_batch is not None:
                largest_passing_batch = max(largest_passing_batch, batch_size)
                if smallest_failing_batch - largest_passing_batch > 1:
                    next_batch_size = (
                        largest_passing_batch + smallest_failing_batch
                    ) // 2
                    del input_ids, attention_mask, labels, outputs, batch_values
                    self._emit_batch_event(
                        "batch_probe",
                        "conditional NLL",
                        batch_size=next_batch_size,
                    )
                    batch_size = next_batch_size
                    self._adaptive_nll_batch_size = batch_size
                    self._release_failed_cuda_batch()
                    continue
            del input_ids, attention_mask, labels, outputs
            if automatic:
                # NLL logits can leave a multi-GiB CUDA allocator cache (and a
                # matching WDDM system-memory backing store) after each part.
                # Release only the transient cache; model weights stay resident.
                self._release_failed_cuda_batch()
            if tuning:
                self._emit_batch_event(
                    "batch_selected", "conditional NLL", batch_size=batch_size
                )
                tuning = False
            for original, value in zip(selected, batch_values, strict=True):
                values[original] = float(value)
            position += len(selected)
        if automatic:
            self._adaptive_nll_batch_size = batch_size
        if any(value is None for value in values):
            raise RuntimeError("adaptive conditional NLL batch lost row coverage")
        return cast(list[float], values)

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # Capture only the final prefill position. Asking generate() to return
        # all hidden states retains every prompt position for every layer.
        modules: list[Module] = [self.model.get_input_embeddings(), *self.get_layers()]
        captured: list[Tensor | None] = [None] * len(modules)
        handles = []

        def capture(index: int):
            def hook(_module: Module, _inputs: Any, output: Any) -> None:
                if captured[index] is not None:
                    return
                value = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(value, Tensor) or value.ndim != 3:
                    raise ValueError(
                        f"residual hook {index} did not receive [batch,sequence,hidden]"
                    )
                captured[index] = value[:, -1, :].detach().clone()

            return hook

        for index, module in enumerate(modules):
            handles.append(module.register_forward_hook(capture(index)))
        try:
            self.generate(
                prompts,
                max_new_tokens=1,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        missing = [index for index, value in enumerate(captured) if value is None]
        if missing:
            raise RuntimeError(f"residual hooks did not fire: {missing}")
        residuals = torch.stack([cast(Tensor, value) for value in captured], dim=1).to(
            torch.float32
        )

        if 0 <= self.settings.winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                self.settings.winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            # One consolidated transfer. The caching allocator is intentionally
            # retained between successful batches and is cleared only after OOM
            # or an explicit phase boundary.
            residuals = residuals.cpu().contiguous()

        return residuals

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        batch_size = self.settings.residual_batch_size or self.settings.batch_size
        return torch.cat(list(self.iter_residual_batches(prompts, batch_size)), dim=0)

    def iter_residual_batches(self, prompts: list[Prompt], batch_size: int):
        """Yield residual tensors without materializing the complete corpus."""

        automatic = batch_size == 0
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if automatic:
            cached_batch_size = int(
                getattr(self, "_adaptive_residual_batch_size", 0)
            )
            tuning = cached_batch_size <= 0
            batch_size = cached_batch_size
            if batch_size <= 0:
                batch_size = min(int(self.settings.max_batch_size), len(prompts))
            if tuning:
                self._emit_batch_event(
                    "batch_probe", "residual", batch_size=batch_size
                )
        else:
            tuning = False
        token_lengths = None
        if hasattr(self, "tokenizer"):
            token_lengths = [
                int(row.numel()) for row in self._cached_prompt_token_ids(prompts)
            ]
        position = 0
        while position < len(prompts):
            batch = prompts[position : position + batch_size]
            restore_order = None
            if token_lengths is not None:
                local_lengths = token_lengths[position : position + len(batch)]
                order = sorted(range(len(batch)), key=local_lengths.__getitem__)
                if order != list(range(len(batch))):
                    batch = [batch[index] for index in order]
                    restore_order = [0] * len(order)
                    for sorted_index, original_index in enumerate(order):
                        restore_order[original_index] = sorted_index
            try:
                residuals = self.get_residuals(batch)
            except BaseException as error:
                if not automatic or batch_size == 1 or not self._is_cuda_oom(error):
                    raise
                next_batch_size = max(1, batch_size // 2)
                if tuning:
                    self._emit_batch_event(
                        "batch_backoff",
                        "residual",
                        batch_size=batch_size,
                        next_batch_size=next_batch_size,
                        reason="OOM",
                    )
                batch_size = next_batch_size
                self._adaptive_residual_batch_size = batch_size
                self._release_failed_cuda_batch()
                continue
            if tuning:
                self._emit_batch_event(
                    "batch_selected", "residual", batch_size=batch_size
                )
                tuning = False
            if restore_order is not None:
                residuals = residuals[restore_order]
            position += len(batch)
            yield residuals
        if automatic:
            self._adaptive_residual_batch_size = batch_size

    def get_residuals_mean(self, prompts: list[Prompt]) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0
        batch_size = self.settings.residual_batch_size or self.settings.batch_size

        for batch in batchify(prompts, batch_size):
            batch_residuals = self.get_residuals(batch)

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
