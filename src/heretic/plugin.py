# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import importlib
import importlib.util
import hashlib
import inspect
import sqlite3
import sys
import types
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from torch import Tensor

from heretic.utils import Prompt, load_prompts

from .config import DatasetSpecification
from .config import Settings as HereticSettings
from .model import Model

T = TypeVar("T")


def get_plugin_namespace(
    model_extra: dict[str, Any] | None, namespace: str
) -> dict[str, Any]:
    """
    Returns the config dict from the `[<namespace>]` TOML table.
    """
    cur: Any = model_extra
    for part in namespace.split("."):
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(part)

    if cur is None:
        return {}
    if not isinstance(cur, dict):
        raise TypeError(
            f"Plugin namespace [{namespace}] must be a table/object, got {type(cur).__name__}"
        )
    return cur


def is_builtin_plugin(name: str) -> bool:
    """
    Whether the plugin name refers to a plugin that ships with Heretic.

    Only built-in plugins can be resolved when reproducing a model, so external
    plugins (file paths or third-party import paths) disable the reproducibility
    offer during upload.
    """
    return name.startswith("heretic.scorers.")


def load_plugin(
    name: str,
    base_class: type[T],
) -> type[T]:
    """
    Load a plugin class from either a filesystem `.py` file or a fully-qualified Python import path.
    Also checks that the class exists in the module and that it
    subclasses the correct Plugin subclass (e.g Scorer).

    Accepted forms:
    - `path/to/plugin.py:MyPluginClass` (relative or absolute): load `MyPluginClass`
      from that file.
    - `fully.qualified.module.MyPluginClass`: import the module and load the class.
    """

    def validate_class(module: ModuleType, class_name: str) -> type[Any]:
        """
        Checks that the module actually exports the class as claimed and returns the class.
        """
        obj = getattr(module, class_name, None)
        if not inspect.isclass(obj):
            raise ValueError(
                f"Plugin '{name}' does not export a class named '{class_name}'"
            )
        return obj

    # Common user trap with filepath imports.
    if name.endswith(".py"):
        raise ValueError(
            "You must append the plugin class name to the filepath like this: path/to/plugin.py:ClassName"
        )

    # File path with explicit class name, e.g. "C:\\path\\plugin.py:MyPlugin".
    if ":" in name:
        file_path, class_name = name.rsplit(":", 1)
        if not file_path.endswith(".py") or not class_name:
            raise ValueError(
                "File-based plugin must use the form 'path/to/plugin.py:ClassName'"
            )

        plugin_path = Path(file_path)
        if not plugin_path.is_absolute():
            plugin_path = Path.cwd() / plugin_path
        plugin_path = plugin_path.resolve()

        if not plugin_path.is_file():
            raise ImportError(f"Plugin file '{plugin_path}' does not exist")

        # We're writing directly to the sys.modules dict,
        # so the typical restrictions on module names
        # (no dots, slashes, etc.) don't apply.
        module_name = f"heretic_plugin_{plugin_path}"

        # Reuse already-loaded modules to avoid re-executing the plugin on repeated loads.
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load plugin '{name}' (invalid module spec)"
                )

            module = importlib.util.module_from_spec(spec)

            # Cache before executing to match normal import semantics and allow
            # circular imports. If execution fails, remove the entry.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise

        plugin_cls = validate_class(module, class_name)
    # Fully-qualified import path, e.g "heretic.scorers.keyword_rate.KeywordRate".
    else:
        if "." not in name:
            raise ValueError(
                "Import-based plugin must use the form 'fully.qualified.module.ClassName'"
            )
        module_name, class_name = name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Error loading plugin '{name}': {e}") from e
        plugin_cls = validate_class(module, class_name)

    if not issubclass(plugin_cls, base_class):
        raise TypeError(f"Plugin '{name}' must subclass {base_class.__name__}")

    return plugin_cls


class Context:
    """
    Runtime context passed to plugins

    Provides plugin-safe access to the model.

    Plugins must use `get_responses(...)`, `get_logits(...)`, etc.
    Direct access to the underlying Model is intentionally not exposed.
    """

    def __init__(
        self,
        settings: HereticSettings,
        model: Model,
        response_archive_id: str | int | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._response_archive_id = response_archive_id
        self._responses_cache: dict[tuple[tuple[str, str], ...], list[str]] = {}

    def _cache_key(self, prompts: list[Prompt]) -> tuple[tuple[str, str], ...]:
        return tuple((p.system, p.user) for p in prompts)

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        """Get model responses (cached within this context)."""
        key = self._cache_key(prompts)
        if key not in self._responses_cache:
            self._responses_cache[key] = self._model.get_responses_batched(
                prompts, skip_special_tokens=True
            )
            self._archive_responses(prompts, self._responses_cache[key])
        return self._responses_cache[key]

    def _archive_responses(
        self, prompts: list[Prompt], responses: list[str]
    ) -> None:
        if not self._settings.save_trial_responses:
            return
        archive_file = self._settings.trial_responses_file
        archive_id = self._response_archive_id
        if archive_id is None:
            return
        if len(prompts) != len(responses):
            raise ValueError("Prompt and response counts differ during archiving")

        prompt_hashes = [
            hashlib.sha256(
                (prompt.system + "\0" + prompt.user).encode("utf-8")
            ).hexdigest()
            for prompt in prompts
        ]
        batch_hash = hashlib.sha256("".join(prompt_hashes).encode("ascii")).hexdigest()
        destination = Path(archive_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(archive_id, int):
            trial_number = (
                getattr(self._settings, "trial_response_number_offset", 0)
                + archive_id
                * getattr(self._settings, "trial_response_number_stride", 1)
            )
            trial_id = f"trial:{trial_number}"
        else:
            trial_id = str(archive_id)
            trial_number = None

        with closing(sqlite3.connect(destination, timeout=60.0)) as database:
            with database:
                database.execute("PRAGMA journal_mode=WAL")
                database.execute("PRAGMA synchronous=NORMAL")
                database.execute("PRAGMA foreign_keys=ON")
                database.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_batches (
                        batch_hash TEXT PRIMARY KEY,
                        prompt_count INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS prompts (
                        batch_hash TEXT NOT NULL,
                        prompt_index INTEGER NOT NULL,
                        system_prompt TEXT NOT NULL,
                        question TEXT NOT NULL,
                        prompt_sha256 TEXT NOT NULL,
                        PRIMARY KEY (batch_hash, prompt_index),
                        FOREIGN KEY (batch_hash) REFERENCES prompt_batches(batch_hash)
                    );
                    CREATE TABLE IF NOT EXISTS trial_answers (
                        batch_hash TEXT NOT NULL,
                        prompt_index INTEGER NOT NULL,
                        trial_id TEXT NOT NULL,
                        trial_number INTEGER,
                        answer TEXT NOT NULL,
                        answer_sha256 TEXT NOT NULL,
                        PRIMARY KEY (batch_hash, prompt_index, trial_id),
                        FOREIGN KEY (batch_hash, prompt_index)
                            REFERENCES prompts(batch_hash, prompt_index)
                    );
                    CREATE INDEX IF NOT EXISTS trial_answers_by_trial
                        ON trial_answers(trial_number, batch_hash, prompt_index);
                    CREATE VIEW IF NOT EXISTS answers_by_question AS
                        SELECT
                            prompts.batch_hash,
                            prompts.prompt_index,
                            prompts.system_prompt,
                            prompts.question,
                            prompts.prompt_sha256,
                            COUNT(trial_answers.trial_id) AS answer_count,
                            json_group_array(
                                json_object(
                                    'trial_id', trial_answers.trial_id,
                                    'trial_number', trial_answers.trial_number,
                                    'answer', trial_answers.answer,
                                    'answer_sha256', trial_answers.answer_sha256
                                )
                            ) AS answers_json
                        FROM prompts
                        JOIN trial_answers USING (batch_hash, prompt_index)
                        GROUP BY prompts.batch_hash, prompts.prompt_index;
                    """
                )
                database.execute(
                    "INSERT OR IGNORE INTO prompt_batches VALUES (?, ?)",
                    (batch_hash, len(prompts)),
                )
                database.executemany(
                    """
                    INSERT OR IGNORE INTO prompts (
                        batch_hash, prompt_index, system_prompt, question, prompt_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            batch_hash,
                            index,
                            prompt.system,
                            prompt.user,
                            prompt_hashes[index],
                        )
                        for index, prompt in enumerate(prompts)
                    ],
                )
                database.executemany(
                    """
                    INSERT OR IGNORE INTO trial_answers (
                        batch_hash, prompt_index, trial_id, trial_number,
                        answer, answer_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            batch_hash,
                            index,
                            trial_id,
                            trial_number,
                            response,
                            hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        )
                        for index, response in enumerate(responses)
                    ],
                )

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        return self._model.get_logits_batched(prompts)

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        return self._model.get_residuals_batched(prompts)

    def load_prompts(self, specification: DatasetSpecification) -> list[Prompt]:
        return load_prompts(self._settings, specification)


class Plugin:
    """
    Base class for Heretic plugins.

    Plugins may define:
    - `settings: <BaseModelSubclass>` type annotation (recommended)
      Heretic will validate the corresponding config table against it and pass
      an instance as `settings`.
    """

    @property
    def reproducible(self) -> bool:
        """
        Whether runs using this plugin can be reproduced bit-for-bit.

        Set to False when the plugin's behavior is not deterministic or depends on
        state outside the pinned config, for example:
        - It calls an external service (e.g. an LLM judge over the OpenAI API).
        - It reads credentials or config from the environment (env vars, files).
        - It is otherwise non-deterministic (network, wall-clock, unseeded RNG).

        Defaults to False; override to True in your plugin class if any of the
        above DO NOT apply.
        """
        return False

    def __init__(
        self, *, heretic_settings: HereticSettings, settings: BaseModel | None = None
    ):
        # Plugins that declare a settings schema should always receive
        # validated plugin settings from the evaluator.
        settings_model = self.__class__.get_settings_model()
        if settings_model is not None:
            if settings is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires settings to be validated"
                )
            if not isinstance(settings, settings_model):
                raise TypeError(
                    f"{self.__class__.__name__}.settings must be an instance of "
                    f"{settings_model.__name__}"
                )
        self.settings = settings
        self.heretic_settings = heretic_settings

    @classmethod
    def validate_contract(cls) -> None:
        """
        Validate the plugin contract.

        - Plugins must not define a constructor (`__init__`). Initialization is
          handled by `Plugin.__init__` and an optional `init(ctx)` method.
        - Plugin subclasses may define `settings: <BaseModelSubclass>` to declare a settings schema.
        """
        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define __init__(). "
                "Use an optional init(ctx) method for plugin-specific initialization."
            )

    @classmethod
    def get_settings_model(cls) -> type[BaseModel] | None:
        """
        Return the plugin settings model, if present.
        - If the plugin has a `settings: <BaseModelSubclass>` type annotation,
          that type is used as the settings schema.
        - Otherwise: no settings schema.
        """

        def unwrap_settings_type(tp: Any) -> Any:
            """Unwrap `Annotated[T, ...]`."""
            while True:
                origin = get_origin(tp)
                if origin is Annotated:
                    tp = get_args(tp)[0]
                    continue
                return tp

        hints = get_type_hints(cls, include_extras=True)
        annotated = hints.get("settings")
        if annotated is None:
            return None

        model = unwrap_settings_type(annotated)
        origin = get_origin(model)
        if origin in (Union, types.UnionType) and type(None) in get_args(model):
            raise TypeError(
                f"{cls.__name__}.settings must not be Optional; "
                "use a non-optional pydantic.BaseModel subclass (e.g. `settings: Settings`)."
            )
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.settings must be annotated with a pydantic.BaseModel subclass"
            )
        return model

    @classmethod
    def validate_settings(
        cls, raw_namespace: dict[str, Any] | None
    ) -> BaseModel | None:
        """
        Validates plugin settings for this plugin class.

        - If a settings model is present: returns an instance of that model.
        - Otherwise returns None.
        """
        settings_model = cls.get_settings_model()
        if settings_model is None:
            return None
        return settings_model.model_validate(raw_namespace or {})

    def init(self, ctx: Context) -> None:
        """
        Runs before the plugin's main functionality.

        Override this in subclasses to do one-time setup (e.g. load prompts, compute
        baselines).
        """
        return None
