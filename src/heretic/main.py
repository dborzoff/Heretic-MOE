# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# ruff: noqa: E402

import sys

# Ensure standard output/error use UTF-8 instead of system default charmap (e.g. cp1252 on Windows).
for stream in (sys.stdout, sys.stderr):
    if (
        hasattr(stream, "reconfigure")
        and (getattr(stream, "encoding", "") or "").lower() != "utf-8"
    ):
        stream.reconfigure(encoding="utf-8")  # type: ignore

from .config import Settings


def _is_help_invocation() -> bool:
    args = sys.argv[1:]
    return "-h" in args or "--help" in args


# Parse and handle CLI help before importing heavyweight ML/runtime dependencies.
if _is_help_invocation():
    Settings()  # ty:ignore[missing-argument]

# FIXME: Rich progress bars are currently disabled because of rendering issues
#        when used from multiple threads in parallel (e.g. by huggingface_hub).
"""
from .progress import patch_tqdm

# This patches tqdm class definitions, which must happen
# before any other module imports tqdm.
patch_tqdm()
"""

import logging
import math
import os
import random
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path
from typing import Any

import huggingface_hub
import lm_eval
import numpy as np
import optuna
import questionary
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import HfApi, ModelCard, ModelCardData
from lm_eval.models.huggingface import HFLM
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState, create_trial
from pydantic import ValidationError
from questionary import Choice, Style
from rich.table import Table
from rich.traceback import install

from .analyzer import Analyzer
from .config import ExportStrategy, QuantizationMethod
from .evaluator import Evaluator
from .model import AbliterationParameters, Model, get_model_class
from .promotion import load_seed_parameters
from .reproduce import (
    check_environment,
    collect_reproducibles,
    load_reproduction_information,
)
from .search import OptimizationRunner
from .study_diagnostics import make_parameter_importance_callbacks
from .system import empty_cache, get_accelerator_info
from .trial_selection import candidate_trials
from .utils import (
    ask_if_unset,
    format_duration,
    format_exception,
    get_file_sha256,
    get_readme_intro,
    get_trial_parameters,
    is_hf_path,
    load_prompts,
    print,
    print_memory_usage,
    upload_reproduce_folder,
)


def obtain_export_strategy(
    settings: Settings,
    model: Model,
) -> ExportStrategy | None:
    """
    Gets the export strategy from settings or prompts the user.
    Provides info to the user if the model is quantized on memory use.
    Returns an export strategy, or None if cancelled.
    """

    if (
        settings.quantization == QuantizationMethod.BNB_4BIT
        and settings.export_strategy is None
    ):
        print()
        print(
            "The model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                    if settings.model in model.trusted_models
                    else None,
                    **model.revision_kwargs,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )

        print()

    return ask_if_unset(
        settings.export_strategy,
        questionary.select(
            "How do you want to export the model?",
            choices=[
                Choice(
                    title="Merge the abliteration LoRA and export the full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value=ExportStrategy.MERGE,
                ),
                Choice(
                    title="Export the abliteration LoRA only (can be merged later)",
                    value=ExportStrategy.ADAPTER,
                ),
            ],
            style=Style([("highlighted", "reverse")]),
        ),
    )


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print(
        "[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]  [blue underline]https://heretic-project.org[/]"
    )
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # Heretic is being invoked in standard (model processing) mode.
        and "--collect-reproducibles" not in sys.argv
        and "--reproduce" not in sys.argv
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    # Work around the "model" argument being required
    # when Heretic is invoked in a non-processing mode.
    if (
        "--collect-reproducibles" in sys.argv or "--reproduce" in sys.argv
    ) and "--model" not in sys.argv:
        sys.argv.extend(["--model", ""])

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error_details in error.errors():
            print(
                f"[bold]{error_details['loc'][0]}[/]: [yellow]{error_details['msg']}[/]"
            )

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    if settings.collect_reproducibles is not None:
        collect_reproducibles(settings.collect_reproducibles)
        return

    reproduction_mode = settings.reproduce is not None

    if settings.reproduce is not None:
        print(f"Loading reproduction information from [bold]{settings.reproduce}[/]...")
        # FIXME: "Reproduction"/"reproducibility" name inconsistency!
        reproduction_information = load_reproduction_information(settings.reproduce)

        # Version 3 is the plugin-era schema, which stores generic scorer
        # `scores`/`baseline_scores`. It is intentionally NOT compatible with the
        # pre-plugin v1/v2 schema (hardcoded refusals/KL `metrics`), so those are
        # rejected rather than silently failing on a missing key later.
        if reproduction_information["version"] != "3":
            print(
                (
                    f"[red]Unsupported file format version: [bold]{reproduction_information['version']}[/].[/] "
                    "This version of Heretic reads version 3 (plugin scorer) reproduce.json files. "
                    "Older files were produced before the scorer-plugin refactor and are not supported. "
                    "Please install Heretic 1.4 to use these files."
                )
            )
            return

        if not check_environment(settings, reproduction_information):
            return

        print()

        settings = Settings.model_validate(reproduction_information["settings"])

    if settings.seed is None:
        settings.seed = random.randint(0, 2**32 - 1)

    transformers.set_seed(settings.seed)

    print(get_accelerator_info())

    if settings.print_debug_information:
        print()
        print(torch.__config__.show().strip())
        print()
        print(
            f"torch.backends.mkldnn.enabled = [bold]{torch.backends.mkldnn.enabled}[/]"
        )
        print(f"torch.get_num_threads() = [bold]{torch.get_num_threads()}[/]")
        print(
            f"torch.get_num_interop_threads() = [bold]{torch.get_num_interop_threads()}[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # Another library that generates warning spam.
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]
        )
        + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if (
        existing_study is not None
        and settings.evaluate_model is None
        and not reproduction_mode
    ):
        choices = []

        if existing_study.user_attrs["finished"]:
            if settings.checkpoint_action is None:
                print()
                print(
                    (
                        "[green]You have already processed this model.[/] "
                        "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                        "Alternatively, you can ignore the previous run and start from scratch. "
                        "This will delete the checkpoint file and all results from the previous run."
                    )
                )

            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            if settings.checkpoint_action is None:
                print()
                print(
                    (
                        "[yellow]You have already processed this model, but the run was interrupted.[/] "
                        "You can continue the previous run from where it stopped. This will override any specified settings. "
                        "Alternatively, you can ignore the previous run and start from scratch. "
                        "This will delete the checkpoint file and all results from the previous run."
                    )
                )

            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        if settings.checkpoint_action is None:
            print()

        if settings.checkpoint_action is None:
            action = questionary.select(
                "How would you like to proceed?",
                choices=choices,
                style=Style([("highlighted", "reverse")]),
            ).ask()
        else:
            action = settings.checkpoint_action

        if action is None or action == "":
            return

        if action == "continue":
            _cli = settings
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
            # Restoring stored settings discards command-line values. Some fields,
            # including save_directory and upload_repo_id, are excluded from the
            # journal entirely; losing them turns an unattended run interactive.
            _always_runtime_fields = (
                "save_directory",
                "model_action",
                "upload_repo_id",
                "trial_index",
                "restore_trial_number",
                "batch_size",
                "export_strategy",
                "parallel_workers",
                "worker_trial_budget",
                "seed",
                "save_trial_responses",
                "trial_responses_file",
                "trial_response_number_offset",
                "trial_response_number_stride",
            )
            # These fields are archived in the journal, but an explicitly supplied
            # value is a legitimate continuation control. In particular,
            # ``--n-trials 1000`` must extend a finished 600-trial study instead of
            # silently restoring the old target. Defaults are deliberately not
            # copied, so an ordinary ``checkpoint_action=continue`` remains exactly
            # reproducible.
            _explicit_resume_fields = (
                "n_trials",
                "n_additional_trials",
                "optimization_only",
                "parameter_importance_interval",
                "selection_policy",
                "primary_objective",
            )
            for _f in _always_runtime_fields + _explicit_resume_fields:
                _v = getattr(_cli, _f, None)
                if _f in _explicit_resume_fields and _f not in _cli.model_fields_set:
                    continue
                if _v is not None:
                    setattr(settings, _f, _v)
                    print(f"From command line: {_f}={_v}")
        elif action == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"[red]Failed:\n{formatted}[/]")
                else:
                    print(f"[red]Failed ({formatted})[/]")

                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            headroom_ok = True
            headroom_description = ""
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                gib = 1024**3
                required_bytes = max(
                    settings.batch_size_vram_headroom_gib * gib,
                    settings.batch_size_vram_headroom_fraction * total_bytes,
                )
                headroom_ok = free_bytes >= required_bytes
                headroom_description = (
                    f"; [bold]{free_bytes / gib:.1f}[/] GiB free, "
                    f"[bold]{required_bytes / gib:.1f}[/] GiB required"
                )

            status = "[green]Ok[/]" if headroom_ok else "[yellow]Insufficient headroom[/]"
            print(
                f"{status} ([bold]{performance:.0f}[/] tokens/s"
                f"{headroom_description})"
            )

            if not headroom_ok:
                # Larger batches cannot restore memory headroom. Keep batch 1
                # as a last-resort fallback if even the smallest batch misses
                # the configured reserve, otherwise retain the previous safe
                # and fastest candidate.
                if best_batch_size == -1:
                    best_batch_size = batch_size
                    best_performance = performance
                break

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    if settings.response_prefix is None:
        print()
        print("Checking for common response prefix...")
        prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
        responses = model.get_responses_batched(prefix_check_prompts)

        # Despite being located in os.path, commonprefix actually performs
        # a naive string operation without any path-specific logic,
        # which is exactly what we need here. Trailing spaces are removed
        # to avoid issues where multiple different tokens that all start
        # with a space character lead to the common prefix ending with
        # a space, which would result in an uncommon tokenization.
        settings.response_prefix = commonprefix(responses).rstrip(" ")

        if settings.response_prefix:
            print(f"* Prefix found: [bold]{settings.response_prefix!r}[/]")

            for cot_initializer, closed_cot_block in settings.chain_of_thought_skips:
                if settings.response_prefix.startswith(cot_initializer):
                    settings.response_prefix = closed_cot_block
                    print(
                        f"* Closed Chain-of-Thought block: [bold]{settings.response_prefix!r}[/]"
                    )

                    # When using a Chain-of-Thought skip, we need to check that the prefix
                    # is actually complete (e.g. not missing a trailing newline).
                    print("* Rechecking with prefix...")
                    responses = model.get_responses_batched(prefix_check_prompts)
                    additional_prefix = commonprefix(responses).rstrip(" ")
                    if additional_prefix:
                        settings.response_prefix += additional_prefix
                        print(
                            f"* Extended prefix found: [bold]{settings.response_prefix!r}[/]"
                        )

                    break
        else:
            print("* None found")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        print()
        print("[bold]Metrics:[/]")
        for score_name, score in evaluator.get_scores():
            print(f"  * {score_name}: [bold]{score.rich_display}[/]")
        return

    if not reproduction_mode and not evaluator.get_objective_names():
        print()
        print(
            "[red]No optimization objectives configured.[/] At least one scorer "
            'must set [bold]optimization[/] to "maximize" or "minimize". '
            "See [bold]config.default.toml[/] for details."
        )
        return

    print()
    print("Calculating per-layer residual directions...")

    needs_full_residuals = settings.print_residual_geometry or settings.plot_residuals

    if needs_full_residuals:
        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)

        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)

        analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

        if settings.print_residual_geometry:
            analyzer.print_residual_geometry()

        if settings.plot_residuals:
            analyzer.plot_residuals()

        # We don't need the full residuals after computing their means and analyzing geometry.
        del good_residuals, bad_residuals, analyzer
    else:
        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(good_prompts)
        print("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(bad_prompts)

    residual_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Implements https://huggingface.co/blog/grimjim/projected-abliteration
        # Adjust the residual directions so that only the component that is
        # orthogonal to the good direction is subtracted during abliteration.
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(residual_directions * good_directions, dim=1)
        residual_directions = (
            residual_directions - projection_vector.unsqueeze(1) * good_directions
        )
        residual_directions = F.normalize(residual_directions, p=2, dim=1)
        del good_directions, projection_vector

    del good_means, bad_means

    # Clear cache before starting the optimization study.
    # This should free up memory from the objects released with the del statements above.
    empty_cache()

    trial_index = 0
    start_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, ...]:
        nonlocal trial_index
        trial_started = time.perf_counter()
        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)
        # Optuna allocates trial numbers atomically in shared storage. Deriving the
        # display index from that number keeps it unique across parallel workers.
        trial_index = trial.number + 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}
        component_enabled: dict[str, bool] = {}

        for component in model.get_abliterable_components():
            enabled = (
                trial.suggest_categorical(f"{component}.enabled", [True, False])
                if settings.conditional_components
                else True
            )
            component_enabled[component] = enabled
            if not enabled:
                parameters[component] = AbliterationParameters(
                    max_weight=0.0,
                    max_weight_position=0.0,
                    min_weight=0.0,
                    min_weight_distance=1.0,
                )
                continue
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            #
            # The MLP gets a negative lower bound that is then clamped to 0, so the
            # optimizer can fully disable its ablation. The clamp puts a positive
            # probability mass on exactly 0 (the continuous sampler would otherwise
            # reach 0 with probability zero). Ablating the MLP is often unnecessary for
            # removing refusals and tends to damage model intelligence more than
            # ablating the attention output, so on many models the optimum is to leave
            # it (mostly) untouched. See issue #202.
            # Full attention keeps the upstream 0.8 floor; it is the established
            # ablation target. MLP and linear attention may clamp to zero because
            # their value depends on the architecture. A continuous distribution
            # would otherwise never disable them exactly.
            #
            # Match component meaning rather than exact keys. Splitting components
            # here must not change the bounds of unchanged keys on other models.
            max_weight_lower_bound = (
                -0.25
                if component.startswith("mlp.") or "linear" in component
                else 0.8
            )
            max_weight = max(
                0.0,
                trial.suggest_float(
                    f"{component}.max_weight",
                    max_weight_lower_bound,
                    2.5,
                ),
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.0,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(1.5 * last_layer_index, 1.0),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})
        trial.set_user_attr("component_enabled", component_enabled)

        worker_label = os.environ.get("HERETIC_WORKER_LABEL", "").strip()
        worker_prefix = f"[bold cyan]{worker_label}[/] | " if worker_label else ""
        print()
        print(
            f"{worker_prefix}Running trial [bold]{trial_index}[/] "
            f"of [bold]{settings.n_trials}[/]..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...")
        model.reset_model()
        print("* Abliterating...")
        model.abliterate(residual_directions, direction_index, parameters)
        edit_telemetry = model.get_last_edit_telemetry()
        print("* Evaluating...")
        scores = evaluator.get_scores(response_archive_id=trial.number)
        objective_values = evaluator.get_objective_values(scores)
        constraint_values = evaluator.get_constraint_values(scores)
        trial.set_user_attr("constraints", list(constraint_values))
        trial.set_user_attr(
            "feasible", all(value <= 0 for value in constraint_values)
        )

        print("  * Metrics:")
        for name, score in scores:
            print(f"    * {name}: [bold]{score.rich_display}[/]")

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / (trial_index - start_index)) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            worker_count = max(settings.parallel_workers, 1)
            worker_word = "GPU" if worker_count == 1 else "GPUs"
            print(
                "[grey50]Estimated study remaining time "
                f"([bold]{worker_count} {worker_word}[/]): "
                f"[bold]{format_duration(remaining_time)}[/][/]"
            )
        trial.set_user_attr(
            "scores",
            evaluator.get_paired_score_records(scores),
        )
        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_index)
        cuda_peaks = []
        if torch.cuda.is_available():
            cuda_peaks = [
                {
                    "device": device_index,
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(
                        device_index
                    ),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
                }
                for device_index in range(torch.cuda.device_count())
            ]
        trial.set_user_attr(
            "telemetry",
            {
                "schema_version": 1,
                "runtime_seconds": time.perf_counter() - trial_started,
                "cuda_peaks": cuda_peaks,
                "edit": edit_telemetry,
            },
        )
        print_memory_usage()

        return objective_values

    def objective_wrapper(trial: Trial) -> tuple[float, ...]:
        try:
            return objective(trial)
        except KeyboardInterrupt:
            # Stop the study gracefully on Ctrl+C.
            trial.study.stop()
            raise TrialPruned()

    # Derive objective info from the configured scorers.
    objective_names = evaluator.get_objective_names()
    directions = evaluator.get_objective_directions()
    constraint_names = evaluator.get_constraint_names()
    study_callbacks = make_parameter_importance_callbacks(
        interval=settings.parameter_importance_interval,
        checkpoint_path=study_checkpoint_file,
        objective_names=objective_names,
        seed=settings.seed,
    )

    if not reproduction_mode:
        optimization_runner = OptimizationRunner(
            startup_design=settings.startup_design,
            n_startup_trials=settings.n_startup_trials,
            seed=settings.seed,
            parallel_workers=settings.parallel_workers,
            constraint_count=len(constraint_names),
            tpe_group=settings.tpe_group,
        )
        study = optuna.create_study(
            sampler=optimization_runner.initial_sampler,
            storage=storage,
            directions=directions,
            study_name="heretic",
            load_if_exists=True,
        )

        study.set_user_attr("settings", settings.model_dump_json())
        study.set_user_attr("constraint_names", constraint_names)
        study.set_user_attr("finished", False)

        start_index = trial_index = len(study.trials)
        if start_index > 0:
            print()
            print("Resuming existing study.")
        elif settings.seed_trials_from:
            # A changed objective invalidates stored scores, but the parameters
            # that reached the old front are still the best starting points.
            #
            # Drop parameters absent from the new space. Optuna samples any new
            # ones when it evaluates the queued trial.
            seeds = load_seed_parameters(
                settings.seed_trials_from,
                settings.seed_trials_count,
                model.get_abliterable_components(),
                settings.seed_selection,
                settings.seed_trials_additional_numbers,
            )
            for params in seeds:
                study.enqueue_trial(
                    params,
                    skip_if_exists=not settings.seed_trials_preserve_duplicates,
                )
            print()
            print(f"Enqueued [bold]{len(seeds)}[/] seed trials from a previous study.")

        try:
            if settings.worker_trial_budget is None:
                optimization_runner.optimize_to(
                    study,
                    objective_wrapper,
                    target_trial_count=settings.n_trials,
                    callbacks=study_callbacks,
                )
            else:
                optimization_runner.optimize_budget(
                    study,
                    objective_wrapper,
                    trial_budget=settings.worker_trial_budget,
                    callbacks=study_callbacks,
                )
        except KeyboardInterrupt:
            # This additional handler takes care of the small chance that KeyboardInterrupt
            # is raised just between trials, which wouldn't be caught by the handler
            # defined in objective_wrapper above.
            pass

        if len(study.trials) == settings.n_trials:
            study.set_user_attr("finished", True)

        report_bound_pressure(study)

        if settings.optimization_only:
            print()
            print("Optimization-only run completed; journal and diagnostics are saved.")
            return

    trial_loop_active = True

    while trial_loop_active:
        if not reproduction_mode:
            # If no trials at all have been evaluated, the study must have been stopped
            # by pressing Ctrl+C while the first trial was running. In this case, we just
            # re-raise the interrupt to invoke the standard handler defined below.
            completed_trials = [
                t for t in study.trials if t.state == TrialState.COMPLETE
            ]
            if not completed_trials:
                raise KeyboardInterrupt

            primary_objective_index = 0
            if settings.primary_objective is not None:
                try:
                    primary_objective_index = objective_names.index(
                        settings.primary_objective
                    )
                except ValueError as error:
                    raise ValueError(
                        "primary_objective must match one configured objective name; "
                        f"got {settings.primary_objective!r}, available={objective_names!r}"
                    ) from error
            sorted_trials = candidate_trials(
                completed_trials,
                directions,
                policy=settings.selection_policy,
                constraint_count=len(constraint_names),
                primary_objective_index=primary_objective_index,
            )

            def format_trial_title(trial: FrozenTrial) -> str:
                feasibility = (
                    " feasible"
                    if trial.user_attrs.get("feasible", not constraint_names)
                    else " INFEASIBLE"
                )
                prefix = f"[Trial {trial.user_attrs['index']:>3} ·{feasibility}]"

                # We don't directly use the trial.values here since we need to show the
                # CLI-formatted versions, which are stored in the trial's user attributes.
                score_parts: list[str] = []
                for score in trial.user_attrs["scores"]:
                    name = score["name"]
                    value = score["score"]["rich_display"]
                    score_parts.append(f"{name}: {value}")

                return f"{prefix} " + ", ".join(score_parts)

            choices = [
                Choice(title=format_trial_title(trial), value=trial)
                for trial in sorted_trials
            ]

            choices.append(
                Choice(
                    title="Run additional trials",
                    value="continue",
                )
            )

            choices.append(
                Choice(
                    title="Exit program",
                    value="",
                )
            )

            print()
            print("[bold green]Optimization finished![/]")

            if settings.trial_index is None:
                print()
                print(
                    (
                        "The following trials resulted in Pareto optimal combinations of the optimization objectives. "
                        "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                        "chat with it to test how well it works, or run standard benchmarks on it. "
                        "You can return to this menu later to select a different trial. "
                        "[yellow]Note that KL divergence values above 0.5 usually indicate significant damage to the original model's capabilities.[/]"
                    )
                )

        while trial_loop_active:
            # Ensure a predefined trial is only processed once.
            if (
                settings.trial_index is not None
                or settings.restore_trial_number is not None
            ):
                trial_loop_active = False

            if reproduction_mode:
                parameters = reproduction_information["parameters"]

                trial = create_trial(
                    values=[],
                    user_attrs={
                        "direction_index": parameters["direction_index"],
                        "parameters": parameters["abliteration_parameters"],
                        "scores": reproduction_information["scores"],
                    },
                )

                print()
                print("Restoring model from reproduction information...")
            else:
                if settings.trial_index is None:
                    print()

                if settings.restore_trial_number is not None:
                    selected_trial = next(
                        (
                            candidate
                            for candidate in completed_trials
                            if candidate.number == settings.restore_trial_number
                        ),
                        None,
                    )
                    if selected_trial is None:
                        raise ValueError(
                            "restore_trial_number "
                            f"{settings.restore_trial_number} does not name a "
                            "completed trial"
                        )
                elif settings.trial_index is not None:
                    selected_trial = sorted_trials[settings.trial_index]
                else:
                    selected_trial = None

                trial = ask_if_unset(
                    selected_trial,
                    questionary.select(
                        "Which trial do you want to use?",
                        choices=choices,
                        style=Style([("highlighted", "reverse")]),
                    ),
                )

                if trial is None or trial == "":
                    return

                if trial == "continue":
                    while True:
                        try:
                            n_additional_trials = ask_if_unset(
                                settings.n_additional_trials,
                                questionary.text(
                                    "How many additional trials do you want to run?"
                                ),
                            )
                            if n_additional_trials is None or n_additional_trials == "":
                                n_additional_trials = 0
                                break
                            n_additional_trials = int(n_additional_trials)
                            if n_additional_trials > 0:
                                break
                            print("[red]Please enter a number greater than 0.[/]")
                        except ValueError:
                            print("[red]Please enter a number.[/]")

                    if n_additional_trials == 0:
                        continue

                    settings.n_trials = len(study.trials) + n_additional_trials
                    study.set_user_attr("settings", settings.model_dump_json())
                    study.set_user_attr("finished", False)

                    try:
                        optimization_runner.optimize_to(
                            study,
                            objective_wrapper,
                            target_trial_count=settings.n_trials,
                            callbacks=study_callbacks,
                        )
                    except KeyboardInterrupt:
                        pass

                    if len(study.trials) == settings.n_trials:
                        study.set_user_attr("finished", True)

                    break

                print()
                print(
                    f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]..."
                )

            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")

            # Per https://github.com/huggingface/peft/issues/868#issuecomment-1820642893
            # once a LoRA is merged it's expected to be empty. Provide a utility function
            # to restore the previous LoRA-ified state.
            def reset_trial_model():
                print("* Resetting model...")
                model.reset_model()
                print("* Abliterating...")
                model.abliterate(
                    residual_directions,
                    trial.user_attrs["direction_index"],
                    {
                        k: AbliterationParameters(**v)
                        for k, v in trial.user_attrs["parameters"].items()
                    },
                )

            reset_trial_model()

            action_loop_active = True

            while action_loop_active:
                # Ensure a predefined action is only executed once.
                if settings.model_action is not None:
                    action_loop_active = False

                if settings.model_action is None:
                    print()

                action = ask_if_unset(
                    settings.model_action,
                    questionary.select(
                        "What do you want to do with the decensored model?",
                        choices=[
                            Choice(
                                title="Save the model to a local folder",
                                value="save",
                            ),
                            Choice(
                                title="Upload the model to Hugging Face",
                                value="upload",
                            ),
                            Choice(
                                title="Chat with the model",
                                value="chat",
                            ),
                            Choice(
                                title="Benchmark the model",
                                value="benchmark",
                            ),
                            Choice(
                                title="Exit program"
                                if reproduction_mode
                                else "Return to the trial selection menu",
                                value="",
                            ),
                        ],
                        style=Style([("highlighted", "reverse")]),
                    ),
                )

                if action is None or action == "":
                    if reproduction_mode:
                        return
                    else:
                        break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "save":
                            save_directory = ask_if_unset(
                                settings.save_directory,
                                questionary.path(
                                    "Path to the folder:",
                                    only_directories=True,
                                ),
                            )
                            if not save_directory:
                                continue

                            strategy = obtain_export_strategy(settings, model)
                            if strategy is None:
                                continue

                            if strategy == ExportStrategy.ADAPTER:
                                print("Saving LoRA adapter...")
                                model.model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                            else:
                                print("Saving merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(
                                    save_directory,
                                    max_shard_size=settings.max_shard_size,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.save_pretrained(save_directory)
                                if model.processor is not None:
                                    model.processor.save_pretrained(save_directory)
                                reset_trial_model()

                            print(f"Model saved to [bold]{save_directory}[/].")

                            if reproduction_mode:
                                print("Verifying hashes of weight files...")

                                for (
                                    filename,
                                    original_sha256,
                                ) in reproduction_information["hashes"].items():
                                    file_path = Path(save_directory) / filename

                                    if file_path.exists():
                                        sha256 = get_file_sha256(file_path)

                                        if sha256.lower() == original_sha256.lower():
                                            print(
                                                f"[bold]{filename}:[/] [green]Hash matches[/]"
                                            )
                                        else:
                                            print(
                                                f"[bold]{filename}:[/] [yellow]Hash doesn't match[/]"
                                            )
                                    else:
                                        print(
                                            f"[bold]{filename}:[/] [red]File not found[/]"
                                        )

                        case "upload":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                # NOTE: Unlike for most other values obtained from interactive inputs, it is
                                #       not possible to set the token via the settings. This is a security
                                #       precaution to prevent exporting the token under all circumstances.
                                #       For scripting, the correct way to set the token is through the HF_TOKEN
                                #       environment variable, or through the HF token file.
                                token = questionary.password(
                                    "Hugging Face access token:"
                                ).ask()
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = ask_if_unset(
                                settings.upload_repo_id,
                                questionary.text(
                                    "Name of repository:",
                                    default=f"{user['name']}/{Path(settings.model).name}-heretic",
                                ),
                            )
                            if not repo_id:
                                continue

                            visibility = ask_if_unset(
                                None
                                if settings.upload_repo_private is None
                                else (
                                    "Private"
                                    if settings.upload_repo_private
                                    else "Public"
                                ),
                                questionary.select(
                                    "Should the repository be public or private?",
                                    choices=[
                                        "Public",
                                        "Private",
                                    ],
                                    style=Style([("highlighted", "reverse")]),
                                ),
                            )
                            if visibility is None:
                                continue
                            private = visibility == "Private"

                            strategy = obtain_export_strategy(settings, model)
                            if strategy is None:
                                continue

                            # Reproducibility requires that the model and all datasets
                            # are available on the Hugging Face Hub (not local paths),
                            # that all datasets are pinned to a commit (an unpinned
                            # dataset was likely loaded from a local cache), and that
                            # only built-in scorer plugins are used (external plugins
                            # cannot be resolved when reproducing).
                            dataset_specifications = [
                                settings.good_prompts,
                                settings.bad_prompts,
                                *evaluator.get_dataset_specifications(),
                            ]
                            is_reproducible = (
                                is_hf_path(settings.model)
                                and all(
                                    is_hf_path(specification.dataset)
                                    and specification.commit is not None
                                    for specification in dataset_specifications
                                )
                                and evaluator.all_scorers_reproducible()
                                and evaluator.all_scorers_builtin()
                                and not reproduction_mode
                            )

                            if is_reproducible:
                                if settings.upload_reproducibility_information is None:
                                    print(
                                        (
                                            "Heretic can add information to the repository that allows others to reproduce the model. "
                                            "This is optional, but valuable to the community as both a learning tool and to preserve computational work already done. "
                                            "Guaranteeing reproducibility requires basic system information (Python and OS version, CPU and GPU/accelerator info) "
                                            "as tensor operations can give different results in different system environments. "
                                            "[bold]The information does not include any file system paths or other private data.[/]"
                                        )
                                    )

                                reproducibility_information = ask_if_unset(
                                    settings.upload_reproducibility_information,
                                    questionary.select(
                                        "Which reproducibility information do you want to add?",
                                        choices=[
                                            Choice(
                                                title="Full: Settings, package versions, and system information",
                                                value="full",
                                            ),
                                            Choice(
                                                title="Basic: Settings and package versions",
                                                value="basic",
                                            ),
                                            Choice(
                                                title="Don't add any reproducibility information",
                                                value="none",
                                            ),
                                        ],
                                        style=Style([("highlighted", "reverse")]),
                                    ),
                                )
                                if reproducibility_information is None:
                                    continue
                            else:
                                reproducibility_information = "none"

                            if strategy == ExportStrategy.ADAPTER:
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    max_shard_size=settings.max_shard_size,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                if model.processor is not None:
                                    model.processor.push_to_hub(
                                        repo_id,
                                        private=private,
                                        token=token,
                                    )
                                reset_trial_model()

                            if is_hf_path(settings.model):
                                card = ModelCard.load(settings.model)
                            else:
                                card_path = (
                                    Path(settings.model)
                                    / huggingface_hub.constants.REPOCARD_NAME
                                )
                                if card_path.exists():
                                    card = ModelCard.load(card_path)
                                else:
                                    card = None

                            if card is not None:
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                if reproducibility_information != "none":
                                    card.data.tags.append("reproducible")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        reproducibility_information != "none",
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            if reproducibility_information != "none":
                                # Set the number of trials to the number of actual completed trials
                                # for the reproduction configuration.
                                settings.n_trials = len(study.trials)
                                current_export_strategy = settings.export_strategy
                                settings.export_strategy = strategy

                                try:
                                    upload_reproduce_folder(
                                        repo_id,
                                        settings,
                                        token,
                                        checkpoint_path=study_checkpoint_file,
                                        trial=trial,
                                        include_system_information=(
                                            reproducibility_information == "full"
                                        ),
                                    )
                                finally:
                                    settings.export_strategy = current_export_strategy

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                            if reproduction_mode:
                                print("Verifying hashes of weight files...")

                                api = HfApi()
                                model_info = api.model_info(
                                    repo_id,
                                    files_metadata=True,
                                    token=token,
                                )

                                if not model_info.siblings:
                                    raise RuntimeError(
                                        "Could not fetch uploaded model hashes."
                                    )

                                for (
                                    filename,
                                    original_sha256,
                                ) in reproduction_information["hashes"].items():
                                    file_found = False

                                    for file in model_info.siblings:
                                        if file.rfilename == filename:
                                            sha256 = getattr(file, "lfs", {}).get(
                                                "sha256"
                                            )
                                            if not sha256:
                                                raise RuntimeError(
                                                    "Could not fetch uploaded model hashes."
                                                )

                                            if (
                                                sha256.lower()
                                                == original_sha256.lower()
                                            ):
                                                print(
                                                    f"[bold]{filename}:[/] [green]Hash matches[/]"
                                                )
                                            else:
                                                print(
                                                    f"[bold]{filename}:[/] [yellow]Hash doesn't match[/]"
                                                )

                                            file_found = True
                                            break

                                    if not file_found:
                                        print(
                                            f"[bold]{filename}:[/] [red]File not found[/]"
                                        )

                        case "chat":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = questionary.text(
                                        "User:",
                                        qmark=">",
                                    ).unsafe_ask()
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                        case "benchmark":
                            benchmarks = questionary.checkbox(
                                "Which benchmarks do you want to run?",
                                [
                                    Choice(
                                        title=f"{benchmark.name}: {benchmark.description}",
                                        value=benchmark,
                                    )
                                    for benchmark in settings.benchmarks
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if not benchmarks:
                                continue

                            scope = questionary.select(
                                (
                                    "Do you want to benchmark the original model along with the decensored model? "
                                    "Benchmarking both models allows you to compare the scores, but it takes twice as much time."
                                ),
                                choices=[
                                    "Benchmark only the decensored model",
                                    "Benchmark both models",
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if scope is None:
                                continue
                            benchmark_original_model = scope == "Benchmark both models"

                            hflm = HFLM(
                                pretrained=model.model,  # ty:ignore[invalid-argument-type]
                                tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
                                batch_size="auto",
                            )

                            table = Table()
                            table.add_column("Benchmark")
                            table.add_column("Metric")
                            if benchmark_original_model:
                                table.add_column("This model", justify="right")
                                table.add_column("Original model", justify="right")
                            else:
                                table.add_column("Value", justify="right")

                            try:
                                first_benchmark = True

                                for benchmark in benchmarks:
                                    print(
                                        f"Running benchmark [bold]{benchmark.name}[/]..."
                                    )

                                    def get_results() -> dict[str, Any]:
                                        results = lm_eval.simple_evaluate(
                                            model=hflm,
                                            tasks=[benchmark.task],
                                        )
                                        return results["results"][benchmark.task]

                                    results = get_results()
                                    if benchmark_original_model:
                                        with model.model.disable_adapter():  # ty:ignore[call-non-callable]
                                            original_results = get_results()

                                    first_row = True

                                    for metric, value in results.items():
                                        if metric != "alias":
                                            if first_row and not first_benchmark:
                                                if benchmark_original_model:
                                                    table.add_row("", "", "", "")
                                                else:
                                                    table.add_row("", "", "")

                                            def format_value(value: Any) -> str:
                                                if isinstance(
                                                    value,
                                                    (float, np.floating),
                                                ):
                                                    return f"{value:.4f}"
                                                else:
                                                    return f"{value}"

                                            cells = [
                                                benchmark.name if first_row else "",
                                                metric,
                                                format_value(value),
                                            ]
                                            if benchmark_original_model:
                                                cells.append(
                                                    format_value(
                                                        original_results[metric]
                                                    )
                                                )
                                            table.add_row(*cells)

                                            first_row = False
                                            first_benchmark = False
                            except KeyboardInterrupt:
                                pass

                            # The benchmark run might have been cancelled by the user
                            # before any benchmark was completed, so we only print results
                            # if there actually are some.
                            if table.rows:
                                print(table)

                except Exception as error:
                    formatted = format_exception(error)
                    if "\n" in formatted:
                        print(f"[red]Error:\n{formatted}[/]")
                    else:
                        print(f"[red]Error: {formatted}[/]")



def report_bound_pressure(study, threshold: float = 0.05) -> None:
    """Warn when Pareto-front trials crowd a search bound.

    Bounds are assumptions about where the answer can lie. A wrong bound makes
    the search return the best allowed point, not the best point. Objective
    values alone cannot distinguish the two.

    This cost us a full Qwen3.6 search. Winners crowded three of four bounds.
    Widening them moved the record from 0.15 to 0.01 refusals, with the edit
    peaking at layer 15 of 40, far earlier than the old 0.6-depth floor allowed.

    Bounds cannot change mid-study. TPE builds its model against distributions
    fixed when the study starts. This function only warns; restarting with wider
    bounds and the old front is cheap.
    """
    from optuna.distributions import FloatDistribution, IntDistribution

    front = study.best_trials
    if len(front) < 3:
        return

    pressed = []
    for name, dist in (front[0].distributions or {}).items():
        if not isinstance(dist, (FloatDistribution, IntDistribution)):
            continue
        span = dist.high - dist.low
        if span <= 0:
            continue
        values = [t.params[name] for t in front if name in t.params]
        if not values:
            continue
        at_low = sum(1 for v in values if v - dist.low <= span * threshold)
        at_high = sum(1 for v in values if dist.high - v <= span * threshold)
        if at_high > len(values) / 2:
            pressed.append((name, "upper", dist.high, at_high, len(values)))
        elif at_low > len(values) / 2:
            pressed.append((name, "lower", dist.low, at_low, len(values)))

    if not pressed:
        return
    print()
    print("[bold yellow]Pareto-front trials are crowding the search bounds:[/]")
    for name, side, bound, n, total in pressed:
        print(f"  * [bold]{name}[/]: {n} of {total} front trials are near the "
              f"{side} bound ({bound:.3f})")
    print("  The optimum may lie outside the search space.")
    print("  Widen the bounds and restart from these points "
          "(--seed-trials-from).")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
