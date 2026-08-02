# Frozen Gemma t190 provenance

Дата аудита: 2026-08-02. Текстовые поля тестовых корпусов не читались.

## Вывод

Поиск, создавший внутренний Gemma-2 кандидат `t190`, был запущен кодом,
соответствующим Git-ревизии `4cbca43f3ea7d7704e4c1b6d374afcb5ee708f24`
(`Warn when the front presses against the search bounds`). Это последний коммит,
существовавший до старта неизменившегося search worker.

Модель была затем экспортирована после коммита
`613e230d4f5988f7f7ba46122b9a0e27bc6ec737`. Между `4cbca43` и `613e230`
изменены только `FORK.md` и Rich-формат отображения perplexity. Формулы правки
весов, пространство Optuna, нормировка и значения scorer не изменялись.
Поэтому `613e230` является удобной последней pre-localization базой разработки,
а `4cbca43` — точной доказанной search-базой.

Журнал того времени не сохранял Git SHA, поэтому абсолютное доказательство
чистоты worktree в момент запуска невозможно. Однако время одного worker,
сохранённые Settings и Optuna distributions полностью совпадают с `4cbca43`, а
последующие два коммита не меняют вычисления. Это максимально сильный доступный
provenance без повторного запуска.

## Временная линия

- `2026-07-29 17:05:35 +03:00`: commit `4cbca43`.
- `2026-07-29 17:15:02`: создан frozen Optuna journal, worker
  `06a18494-bd94-4b80-ad02-011a93da5cca-48792`.
- `2026-07-29 17:27:22 +03:00`: commit `8b241f0`, меняющий только Rich display
  perplexity; уже запущенный worker его не загружал.
- `2026-07-29 17:32:18 +03:00`: commit `613e230`, меняющий только `FORK.md`;
  уже запущенный worker его не загружал.
- `2026-07-29 17:52:03` — `17:52:12`: Optuna trial number `189`, сохранённый
  display index `190`, выполнен тем же исходным worker.
- `2026-07-29 19:05:56`: журнал завершён, 400/400 trials `COMPLETE`.
- `2026-07-29 19:12:41` — `19:12:53`: экспортированы файлы `out/t190`.
- Первый commit локализационной исследовательской серии — `cfb6ea1` в
  `2026-07-29 23:17:07`, то есть уже после создания t190.

## Settings исследования

- model: `F:\AI\hf_originals\google__gemma-2-2b-it`;
- direction data: frozen upstream defaults, `400` good + `400` bad;
- scorers: `KeywordRate` и GPU `Perplexity`, оба `minimize`;
- `row_normalization = full`;
- `full_normalization_lora_rank = 3`;
- `n_startup_trials = 60`;
- sampler: Optuna multivariate TPE, `n_ei_candidates = 128`;
- seed: `3149143241`;
- journal: 400 completed trials;
- model reset: чистая исходная модель перед каждой recipe.

Первичная конфигурация study сначала содержала `n_trials = 30`, затем была
продолжена до `200` и в итоге до `400`. Это один и тот же journal и тот же
search worker, а не накопительная правка весов.

## Пространство поиска, сохранённое у trial 189

- `direction_scope`: `global | per layer`;
- `direction_index`: `10.0 .. 22.5`;
- attention maximum: `0.8 .. 2.5`;
- MLP maximum before zero clamp: `-0.25 .. 2.5`;
- обе позиции пика: `0.0 .. 25.0`;
- обе доли minimum: `0.0 .. 1.0`;
- оба радиуса: `1.0 .. 37.5`.

Эти distributions побайтно по смыслу совпадают с расширением, введённым
коммитом `f78e37c` и присутствующим в `4cbca43`.

## Frozen t190 trial

- Optuna number: `189`;
- display index: `190`;
- Pareto: да;
- search values: `KeywordRate = 0.05`, quick relative PPL
  `-0.0032220547892540807`;
- exact later PPL 400x512: около `-0.0107%` к базе, то есть практически ноль.

Параметры:

```text
direction_scope = per layer
direction_index = sampled 10.7241034596, ignored by per-layer scope

attn.o_proj.max_weight = 1.5241593254
attn.o_proj.max_weight_position = 12.7984603190
attn.o_proj.min_weight = 0.9075196337
attn.o_proj.min_weight_distance = 7.6269308569

mlp.down_proj.max_weight = 1.3478876042
mlp.down_proj.max_weight_position = 21.4536535217
mlp.down_proj.min_weight = 0.8510247865
mlp.down_proj.min_weight_distance = 19.3214052980
```

## Frozen artifact hashes

```text
config.toml
  fba40a1d0428b4991104a61806d094a7970135999b53f7fe038af4b54bcf2e4a
save/config.toml
  a7335925f484ba368f2584c2e8c61976a5dcd0ab8b98e8710d1bebc047bf3754
save/save4.bat
  ec62421c47cfebbcb68c435cec102abf1342a857a8c974d62ee3da2ff941d5e0
Optuna journal
  e61ff6f6fe18fb9e590e4b87b4183e673fa489d7d463330b4a6d71e709399642
model shard 1
  1ff1438fd5dae0ba7db6dcdd23f581dfc5d027d84024d3cc4e374c84077eadef
model shard 2
  16c2023c27bc5fa059b591793482ac86c3b1241716adbf4edc5e0dd3922688af
model index
  363dac338d37ef2d4cda788d6c76e449e677d295fe59c3dec55077c6f085fd30
```

## Branching decision

For development, use `613e230` as the clean pre-localization source baseline:
it preserves the exact t190 numerical behavior and includes the final fork
description. Retain `4cbca43` in this document as the exact search-process
revision. Everything beginning with `cfb6ea1` belongs to the later research
line and must remain reachable through the local `test` branch.
