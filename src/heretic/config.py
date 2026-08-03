# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from enum import Enum
from typing import Dict, Literal

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# !!!IMPORTANT!!!
#
# Any settings added to the classes defined in this module
# must be evaluated for privacy implications and have
# exclude=True set in their field definitions if appropriate.


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically possible, but provides no advantage.
    FULL = "full"


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class StartupDesign(str, Enum):
    RANDOM = "random"
    SOBOL = "sobol"
    HYBRID = "hybrid"


class SeedSelection(str, Enum):
    FIRST_OBJECTIVE = "first_objective"
    SPREAD = "spread"
    ALL = "all"


class SelectionPolicy(str, Enum):
    PARETO = "pareto"
    FEASIBLE_LEXICOGRAPHIC = "feasible_lexicographic"


class DatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    split: str | None = Field(
        default=None,
        description="Portion of the dataset to use. Required for datasets, optional for plain text files.",
    )

    column: str | None = Field(
        default=None,
        description="Column in the dataset that contains the prompts. Required for datasets, ignored for plain text files.",
    )

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )

    residual_plot_label: str | None = Field(
        default=None,
        description="Label to use for the dataset in plots of residual vectors.",
        exclude=True,
    )

    residual_plot_color: str | None = Field(
        default=None,
        description="Matplotlib color to use for the dataset in plots of residual vectors.",
        exclude=True,
    )


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer plugin.

    TOML format:
    - { plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    optimization: Literal["minimize", "maximize", "none"] = Field(
        description=(
            "Optimization direction for this scorer. "
            '"minimize" / "maximize" to include the scorer as an objective, '
            '"none" to compute the score without optimizing for it.'
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`."
        ),
    )

    constraint_lower: float | None = Field(
        default=None,
        description=(
            "Optional lower feasibility bound for this score. The trial is "
            "infeasible when score < constraint_lower."
        ),
    )

    constraint_upper: float | None = Field(
        default=None,
        description=(
            "Optional upper feasibility bound for this score. The trial is "
            "infeasible when score > constraint_upper."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value

    @model_validator(mode="after")
    def validate_constraint_interval(self) -> "ScorerConfig":
        if (
            self.constraint_lower is not None
            and self.constraint_upper is not None
            and self.constraint_lower > self.constraint_upper
        ):
            raise ValueError("constraint_lower cannot exceed constraint_upper")
        return self


class BenchmarkSpecification(BaseModel):
    task: str = Field(
        description="Task ID of the benchmark in the Language Model Evaluation Harness."
    )

    name: str = Field(description="Name of the benchmark for presentation purposes.")

    description: str = Field(
        description="Description of the benchmark for presentation purposes."
    )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the model.",
    )

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
        exclude=True,
    )

    collect_reproducibles: str | None = Field(
        default=None,
        description=(
            "If this directory path is set, then instead of abliterating a model, "
            "download all reproduce.json files from public Heretic model repositories "
            "on Hugging Face, and store them in that directory for archival purposes."
        ),
        exclude=True,
    )

    reproduce: str | None = Field(
        default=None,
        description=(
            "If this path or URL to a reproduce.json file is set, load reproduction information "
            "from that file, and attempt to reproduce the abliterated model it originated from."
        ),
        exclude=True,
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If "auto" resolves to float32, and that fails because it is too large,
            # and float16 fails due to range issues, try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of PyTorch dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantization method to use when loading the model. Options: "
            '"none" (no quantization), '
            '"bnb_4bit" (4-bit quantization using bitsandbytes).'
        ),
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: Dict[str, str] | None = Field(
        default=None,
        description='Maximum memory to allocate per device (e.g., { "0" = "20GB", "cpu" = "64GB" }).',
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Whether to move intermediate analysis tensors (such as residuals and logprobs) "
            "to CPU memory as soon as possible to reduce peak VRAM usage. "
            "This lowers peak VRAM usage during residual analysis and evaluation, "
            "but may slightly reduce performance due to host/device transfers."
        ),
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: PositiveInt = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    residual_batch_size: NonNegativeInt = Field(
        default=0,
        description=(
            "Batch size for per-layer residual extraction (0 = use batch_size). "
            "Set this below the generation batch size when hidden-state capture "
            "needs additional VRAM headroom."
        ),
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    chain_of_thought_skips: list[tuple[str, str]] = Field(
        default=[
            # Most thinking models.
            (
                "<think>",
                "<think></think>",
            ),
            # gpt-oss.
            (
                "<|channel|>analysis<|message|>",
                "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            # Unknown, suggested by user.
            (
                "<thought>",
                "<thought></thought>",
            ),
            # Unknown, suggested by user.
            (
                "[THINK]",
                "[THINK][/THINK]",
            ),
        ],
        description=(
            "List of pairs of the form (cot_initializer, closed_cot_block) used to skip "
            "the Chain-of-Thought block in responses, so that evaluation happens "
            "at the start of the actual response."
        ),
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    print_debug_information: bool = Field(
        default=False,
        description="Whether to print additional information that can help with debugging.",
        exclude=True,
    )

    print_residual_geometry: bool = Field(
        default=False,
        description="Whether to print detailed information about residuals and residual directions.",
        exclude=True,
    )

    plot_residuals: bool = Field(
        default=False,
        description="Whether to generate plots showing PaCMAP projections of residual vectors.",
        exclude=True,
    )

    residual_plot_path: str = Field(
        default="plots",
        description="Base path to save plots of residual vectors to.",
        exclude=True,
    )

    residual_plot_title: str = Field(
        default='PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts',
        description="Title placed above plots of residual vectors.",
        exclude=True,
    )

    residual_plot_style: str = Field(
        default="dark_background",
        description="Matplotlib style sheet to use for plots of residual vectors.",
        exclude=True,
    )

    scorers: list[ScorerConfig] = Field(
        default_factory=lambda: [
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
            ),
            ScorerConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ],
        description=(
            "List of scorer plugin configs. Each entry is an object"
            " { plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }."
            " <optimization> is one of 'minimize', 'maximize', 'none' (do not optimize)."
        ),
    )

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust the residual directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    fused_expert_chunk_size: PositiveInt = Field(
        default=8,
        description=(
            "Number of fused routed experts edited in FP32 at once. Smaller "
            "chunks reduce peak VRAM during exact fused-expert normalization."
        ),
    )

    record_edit_telemetry: bool = Field(
        default=True,
        description=(
            "Record text-free per-component and per-layer realized edit norms "
            "in every completed trial."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )

    n_trials: PositiveInt = Field(
        default=200,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=60,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    startup_design: StartupDesign = Field(
        default=StartupDesign.RANDOM,
        description=(
            'Exploration design for the first n_startup_trials: "random" keeps '
            'the legacy multivariate-TPE startup; "sobol" uses a scrambled Sobol '
            'sequence; "hybrid" alternates Random and scrambled Sobol trials in '
            "one shared study. Every non-random design then switches to "
            "multivariate TPE with the complete exploration history."
        ),
    )

    tpe_group: bool = Field(
        default=False,
        description=(
            "Use Optuna's group-decomposed multivariate TPE. Required when "
            "conditional_components enables a dynamic search space."
        ),
    )

    conditional_components: bool = Field(
        default=False,
        description=(
            "Sample an explicit enabled flag per editable component and omit "
            "curve parameters for disabled components."
        ),
    )

    selection_policy: SelectionPolicy = Field(
        default=SelectionPolicy.FEASIBLE_LEXICOGRAPHIC,
        description=(
            'Trial selection policy. "pareto" preserves the legacy menu; '
            '"feasible_lexicographic" filters by scorer constraints and then '
            "orders the feasible Pareto front by the primary objective."
        ),
    )

    primary_objective: str | None = Field(
        default=None,
        description=(
            "Objective display name used first by feasible lexicographic "
            "selection. Unset means the first configured optimization objective."
        ),
    )

    parameter_importance_interval: NonNegativeInt = Field(
        default=0,
        description=(
            "Write a text-free fANOVA parameter-importance report after every N "
            "completed trials (0 disables it). The report is diagnostic only and "
            "does not alter Optuna sampling."
        ),
    )

    optimization_only: bool = Field(
        default=False,
        description=(
            "Exit after optimization and journal diagnostics, without prompting "
            "to restore, export, or benchmark a selected trial. Useful for a "
            "discovery stage whose Pareto points seed a separate full-fidelity study."
        ),
    )

    parallel_workers: PositiveInt = Field(
        default=1,
        description=(
            "Number of concurrent Optuna workers sharing this study. Values above "
            "1 enable TPE constant-liar handling for running trials."
        ),
    )

    worker_trial_budget: PositiveInt | None = Field(
        default=None,
        description=(
            "Exact number of trials assigned to this process. Intended for a "
            "parallel continuation whose per-worker budgets sum to the remaining "
            "global trial target."
        ),
        exclude=True,
    )

    seed_trials_from: str | None = Field(
        default=None,
        description=(
            "Journal file from an earlier study. Its Pareto-front parameters are"
            " enqueued when a changed objective makes stored scores unusable."
            " Removed parameters are dropped; new ones are sampled."
        ),
    )

    seed_trials_count: NonNegativeInt = Field(
        default=12,
        description=(
            "Number of old front points to enqueue. Each consumes one trial under"
            " the new objective. Keep this modest after changing the search space:"
            " a large seed re-evaluates half-sampled points instead of exploring."
        ),
    )

    seed_selection: SeedSelection = Field(
        default=SeedSelection.FIRST_OBJECTIVE,
        description=(
            'How to choose seeds: "first_objective" preserves the legacy '
            'preference for the lowest first objective; "spread" keeps diverse '
            'trade-offs across normalized objective space; "all" replays completed '
            "source trials in their original order."
        ),
    )

    seed_trials_preserve_duplicates: bool = Field(
        default=False,
        description=(
            "Whether exact duplicate parameter sets from a seed journal should be "
            "measured again. Useful for a full objective-change replay and its "
            "repeatability check; leave disabled for ordinary front seeding."
        ),
    )

    seed_trials_additional_numbers: list[NonNegativeInt] = Field(
        default=[],
        description=(
            "Specific source trial numbers to append after the selected seeds. "
            "This keeps known reference candidates in a partial replay without "
            "replaying the objective-biased tail of the source study."
        ),
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible optimization. "
            "Applies to Python's random module, NumPy, PyTorch, and Optuna."
        ),
    )

    @model_validator(mode="after")
    def validate_adaptive_search_settings(self) -> "Settings":
        if self.conditional_components and not self.tpe_group:
            raise ValueError(
                "conditional_components requires tpe_group=true so Optuna can "
                "model the dynamic component subspaces"
            )
        return self

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
        exclude=True,
    )

    save_trial_responses: bool = Field(
        default=False,
        description=(
            "Save every evaluation response with its prompt and trial number. "
            "This makes later scorer changes replayable without regenerating text."
        ),
        exclude=True,
    )

    trial_responses_file: str = Field(
        default="trial-responses.sqlite3",
        description=(
            "SQLite file used when save_trial_responses=true. Prompts are stored "
            "once and linked to all answers produced across trials."
        ),
        exclude=True,
    )

    trial_response_number_offset: NonNegativeInt = Field(
        default=0,
        description=(
            "Offset applied to integer trial numbers in the response archive. "
            "The adaptive two-branch controller uses offsets 0 and 1."
        ),
        exclude=True,
    )

    trial_response_number_stride: PositiveInt = Field(
        default=1,
        description=(
            "Stride applied to integer trial numbers in the response archive. "
            "The adaptive two-branch controller uses stride 2 so Random and "
            "Sobol answers are numbered even and odd before journal merge."
        ),
        exclude=True,
    )

    benchmarks: list[BenchmarkSpecification] = Field(
        default=[
            BenchmarkSpecification(
                task="agieval",
                name="AGIEval",
                description="A Human-Centric Benchmark for Evaluating Foundation Models",
            ),
            BenchmarkSpecification(
                task="bbh",
                name="BIG-Bench Hard (BBH)",
                description="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            ),
            BenchmarkSpecification(
                task="commonsense_qa",
                name="CommonsenseQA",
                description="A Question Answering Challenge Targeting Commonsense Knowledge",
            ),
            BenchmarkSpecification(
                task="eq_bench",
                name="EQ-Bench",
                description="An Emotional Intelligence Benchmark for Large Language Models",
            ),
            BenchmarkSpecification(
                task="gsm8k",
                name="GSM8K",
                description="Training Verifiers to Solve Math Word Problems",
            ),
            BenchmarkSpecification(
                task="hellaswag",
                name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
            ),
            BenchmarkSpecification(
                task="ifeval",
                name="IFEval",
                description="Instruction-Following Evaluation for Large Language Models",
            ),
            BenchmarkSpecification(
                task="mmlu",
                name="MMLU",
                description="Measuring Massive Multitask Language Understanding",
            ),
            BenchmarkSpecification(
                task="mmlu_pro",
                name="MMLU-Pro",
                description="A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            ),
            BenchmarkSpecification(
                task="piqa",
                name="PIQA",
                description="Reasoning about Physical Commonsense in Natural Language",
            ),
            BenchmarkSpecification(
                task="winogrande",
                name="WinoGrande",
                description="An Adversarial Winograd Schema Challenge at Scale",
            ),
        ],
        description="Benchmarks to offer to the user for evaluating abliterated models.",
        exclude=True,
    )

    max_shard_size: PositiveInt | str = Field(
        default="5GB",
        description="Maximum size for individual safetensors files generated when exporting a model.",
    )

    export_strategy: ExportStrategy | None = Field(
        default=None,
        description='How to export the model: "merge", "adapter", or unset to prompt the user.',
    )

    checkpoint_action: str | None = Field(
        default=None,
        description='Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.',
    )

    trial_index: NonNegativeInt | None = Field(
        default=None,
        description="Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.",
    )

    restore_trial_number: NonNegativeInt | None = Field(
        default=None,
        description=(
            "Exact Optuna trial number to restore for unattended export. Unlike "
            "trial_index, this remains stable when another study changes the "
            "combined Pareto ordering."
        ),
        exclude=True,
    )

    n_additional_trials: PositiveInt | None = Field(
        default=None,
        description="Number of additional trials to run, or unset to prompt the user.",
    )

    model_action: str | None = Field(
        default=None,
        description='Action to take with the decensored model: "save", "upload", or unset to prompt the user.',
    )

    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_id: str | None = Field(
        default=None,
        description="Name of the Hugging Face repository to upload the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_private: bool | None = Field(
        default=None,
        description="Whether the Hugging Face repository to upload the model to should be private, or unset to prompt the user.",
    )

    upload_reproducibility_information: str | None = Field(
        default=None,
        description='Which reproducibility information to add to the Hugging Face repository: "full", "basic", "none", or unset to prompt the user.',
    )

    ignore_mismatches: bool | None = Field(
        default=None,
        description="Whether to attempt to reproduce the model even if there are environment mismatches, or unset to prompt the user.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for calculating refusal directions).",
    )

    bad_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmful" prompts',
            residual_plot_color="darkorange",
        ),
        description="Dataset of prompts that tend to result in refusals (used for calculating refusal directions).",
    )

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `Evaluator._get_plugin_namespace`).
    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Used during resume - should override *all* other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )
