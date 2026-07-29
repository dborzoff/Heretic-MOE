# Обзор локальных моделей для поиска отказной зоны

Дата: 2026-07-29.

Статус: исследование конфигов, safetensors headers, локального Transformers-кода
и существующих Optuna-журналов. Веса целиком в RAM не загружались, модели не
изменялись.

Связанные документы:

- [LOCALIZED_INVERSION_DESIGN.md](LOCALIZED_INVERSION_DESIGN.md)
- [LAYER_LOCALIZATION_SEARCH.md](LAYER_LOCALIZATION_SEARCH.md)
- [CLAUDE_LOCALIZED_INVERSION_HANDOFF.md](CLAUDE_LOCALIZED_INVERSION_HANDOFF.md)

## 1. Инвентаризация плотных текстовых стеков

| Модель | Текстовых слоёв | Hidden | MLP intermediate | Attention |
|---|---:|---:|---:|---|
| Gemma-2-2B-it | 26 | 2304 | 9216 | 13 sliding + 13 full |
| Gemma-3-12B-it | 48 | 3840 | 15360 | 40 sliding + 8 full |
| Gemma-4-E4B-it | 42 | 2560 | 10240 | 35 sliding + 7 full |
| Ministral-3-3B | 26 | 3072 | 9216 | full attention |
| Qwen2.5-3B-Instruct | 36 | 2048 | 11008 | full attention |
| Qwen2.5-VL-7B-Instruct | 28 | 3584 | 18944 | full attention |
| Qwen3-4B | 36 | 2560 | 9728 | full attention |
| Qwen3-8B | 36 | 4096 | 12288 | full attention |
| Qwen3-VL-4B/8B | 36 | 2560/4096 | 9728/12288 | full attention |
| Qwen3.5-9B | 32 | 4096 | 12288 | 24 linear + 8 full |

Ministral-3-3B-Instruct-2512 в локальном каталоге FP8-квантован, поэтому для
первого чистого сравнения кривых он хуже BF16-моделей.

## 2. Реальные формы основных выходных проекций

| Модель | Attention output | MLP output |
|---|---|---|
| Gemma-2 | `[2304, 2048]` | `[2304, 9216]` |
| Gemma-3 | `[3840, 4096]` | `[3840, 15360]` |
| Gemma-4 | sliding `[2560, 2048]`, full `[2560, 4096]` | `[2560, 10240]` |
| Ministral-3 | `[3072, 4096]` | `[3072, 9216]` |
| Qwen2.5-3B | `[2048, 2048]` | `[2048, 11008]` |
| Qwen2.5-VL-7B | `[3584, 3584]` | `[3584, 18944]` |
| Qwen3-4B | `[2560, 4096]` | `[2560, 9728]` |
| Qwen3-8B | `[4096, 4096]` | `[4096, 12288]` |
| Qwen3.5-9B linear | `out_proj [4096, 4096]` | `[4096, 12288]` |
| Qwen3.5-9B full | `o_proj [4096, 4096]` | `[4096, 12288]` |

У всех этих матриц первая размерность равна residual hidden size, поэтому
направленная правка выходного пространства алгебраически применима.

## 3. Доля хранимых текстовых параметров в текущих output targets

Это число параметров `o_proj/out_proj/down_proj`, а не доля FLOPs и не доля
реально активных MoE-весов.

| Модель | Text parameters | Текущие output targets | Доля |
|---|---:|---:|---:|
| Gemma-2-2B-it | 2.614B | 0.675B | 25.8% |
| Gemma-3-12B-it | 11.766B | 3.586B | 30.5% |
| Gemma-4-E4B-it | 7.518B | 1.358B | 18.1% |
| Ministral-3-3B | 3.429B | 1.063B | 31.0% |
| Qwen2.5-3B | 3.086B | 0.963B | 31.2% |
| Qwen2.5-VL-7B | 7.616B | 2.261B | 29.7% |
| Qwen3-4B | 4.022B | 1.274B | 31.7% |
| Qwen3-8B | 8.191B | 2.416B | 29.5% |
| Qwen3.5-9B | 8.954B | 2.147B | 24.0% |

Низкая доля Gemma-4 объясняется прежде всего отдельной per-layer embedding
веткой, а Qwen3.5 — MTP и gate/input projections.

## 4. Gemma-2 и Gemma-3: скрытое смешение типов attention

Gemma-2 по умолчанию чередует:

```text
layer 0 sliding
layer 1 full
layer 2 sliding
layer 3 full
...
```

Gemma-3 использует full attention в каждом шестом слое, остальные слои
sliding. Во всех случаях модуль называется `self_attn.o_proj`, поэтому Heretic
кладёт их в один `attn.o_proj`.

Это та же проблема, ради которой в Qwen3.5 были разделены linear и full
attention, только имя класса здесь одинаковое.

### Следствие для локального окна

Гладкое окно по глубине одновременно меняет два качественно разных механизма.
Наблюдаемый выигрыш нельзя приписать конкретному типу attention.

Предлагаемый диагностический split:

```text
attn.o_proj          — full attention, старое имя
attn.sliding.o_proj  — sliding attention, новый ключ
```

Однако для Gemma это изменит смысл старого `attn.o_proj`, поэтому старую study
продолжать нельзя. Новый curve family всё равно требует новой study.

## 5. Gemma-4: третья запись в residual stream

После attention и MLP каждый слой Gemma-4 выполняет:

```text
residual = hidden_states
gate = activation(per_layer_input_gate(hidden_states))
x = gate * per_layer_input[layer]
x = per_layer_projection(x)
x = post_per_layer_input_norm(x)
hidden_states = residual + x
hidden_states *= layer_scalar
```

Локальная реализация:

```text
transformers/models/gemma4/modeling_gemma4.py:1452-1461
```

Формы на каждом из 42 слоёв:

```text
per_layer_input_gate.weight   [256, 2560]
per_layer_projection.weight   [2560, 256]
post_per_layer_input_norm     [2560]
layer_scalar                  [1]
```

Источники `per_layer_input`:

```text
embed_tokens_per_layer.weight        [262144, 10752]
per_layer_model_projection.weight    [10752, 2560]
```

`10752 = 42 × 256`: вход содержит отдельный 256-мерный блок для каждого слоя.

### Размеры ветки

| Часть | Параметры |
|---|---:|
| `embed_tokens_per_layer` | 2.819B |
| общий `per_layer_model_projection` | 27.53M |
| 42 `per_layer_input_gate` | 27.53M |
| 42 `per_layer_projection` | 27.53M |

То есть всего 27.5M выходных весов управляют записью в residual stream сигнала,
который питается из таблицы на 2.82B параметров.

### Проверка original против heretic build

В `google__gemma-4-E4B-it-heretic` выборочные фрагменты:

- `embed_tokens_per_layer`;
- `per_layer_model_projection`;
- `per_layer_input_gate`;
- `per_layer_projection`;

совпали с оригиналом бит-в-бит. Фрагменты `self_attn.o_proj` и
`mlp.down_proj` в тех же слоях отличаются.

Это согласуется с кодом Heretic: PLE-ветка не является target и остаётся
нетронутой.

### Наиболее интересный новый target

`per_layer_projection [2560, 256]` уже является выходной проекцией в residual
space и подходит под существующую математику без редактирования огромной
embedding-таблицы.

Возможный компонент:

```text
ple.per_layer_projection
```

Это более обоснованная первая проверка, чем правка `layer_scalar`: scalar
масштабирует весь residual state и почти наверняка будет менее хирургическим.

До изменения весов полезно причинно отключить PLE-выход только в выбранных
слоях и измерить, меняется ли отказ.

### Послойные нормы PLE

Проверены:

- четыре разнесённых блока словарных строк `embed_tokens_per_layer`;
- весь `per_layer_model_projection`;
- все 42 `per_layer_input_gate`;
- все 42 `per_layer_projection`;
- все 42 `layer_scalar`.

Результат:

| Величина | Вариация по слоям |
|---|---:|
| sampled embedding block RMS | CV 0.7%, max/min 1.04 |
| model projection block RMS | CV 5.0%, max/min 1.21 |
| output projection RMS | CV 4.0%, max/min 1.16 |
| input gate RMS | CV 35.9%, max/min 16.23 |

`layer_scalar` также не равен единице: он меняется примерно от `0.061` до
`0.887`. Корреляция между RMS `per_layer_input_gate` и `layer_scalar`:

```text
Pearson r = 0.981
```

Средняя gate-норма особенно высока на слоях 30–41; заметный провал находится
примерно на 18–23. Самые большие индивидуальные значения — на слоях 36–40.

Это не доказывает локализацию отказа: после projection применяется RMSNorm, а
`layer_scalar` масштабирует весь hidden state. Но структурный градиент ясно
находится в gate/scalar, а не в самой огромной embedding-таблице.

Практический приоритет для hooks:

1. норма и направление PLE-ветки до `post_per_layer_input_norm`;
2. после norm, непосредственно перед residual addition;
3. gate activations на harmful/harmless и по категориям;
4. причинное отключение PLE на слоях 18–23 и 30–41 как контрастных зон.

## 6. Qwen3.5-9B: две attention-ветки и два gate-механизма

Слои:

```text
24 × GatedDeltaNet linear attention
8 × full attention
```

Linear attention:

```text
in_proj_z [4096, 4096]
core = gated_rms_norm(core, z)
out_proj [4096, 4096]
residual += out_proj(core)
```

Full attention:

```text
q_proj [8192, 4096]
query, gate = split(q_proj(hidden))
attention_output *= sigmoid(gate)
o_proj [4096, 4096]
residual += o_proj(attention_output)
```

Размеры gate weights:

| Gate | Параметры |
|---|---:|
| 24 linear `in_proj_z` | 402.65M |
| gate-половина восьми full `q_proj` | 134.22M |

Gate не пишет в residual самостоятельно, а токен-зависимо регулирует сигнал до
выходной проекции. Поэтому для локализации сначала надо записать его активации
по слоям и вердиктам, а не сразу редактировать сотни миллионов входных весов.

Текущий fork уже правильно разделяет:

```text
attn.linear.out_proj
attn.o_proj
```

## 7. MTP: локальный экспорт не сохраняет ветку

В оригинальном `Qwen__Qwen3.5-9B` есть 15 MTP tensors, всего:

```text
243,290,624 parameters
0.453 GiB в BF16
```

В локальном `Qwen__Qwen3.5-9B-heretic`:

```text
MTP tensors: 0
```

При этом в обоих config остаётся:

```text
mtp_num_hidden_layers = 1
```

Локальный Transformers-класс содержит:

```text
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

а Heretic не содержит отдельного копирования MTP tensors после
`save_pretrained`. Поэтому для проверенной Qwen3.5-сборки формулировка
«MTP восстановлен из base verbatim» неверна: ветка не сохранена вообще.

Для Qwen3.6 оригинальный MTP действительно имеет 844,640,768 параметров, но
локальной heretic-сборки для сравнения ключей нет. При том же export path
ожидается аналогичная потеря; это надо проверить на фактическом артефакте, а не
утверждать заранее.

Практические последствия:

- обычная генерация основного CausalLM не использует эти tensors;
- speculative/MTP runtime не сможет использовать ветку, которой нет в
  сохранённом checkpoint;
- перед обсуждением «цензурированного MTP» сначала нужно решить сохранение и
  загрузку MTP как артефакта.

## 8. Что показывают старые Optuna-журналы

Ниже выбран trial с минимальным KeywordRate; это геометрия, а не честная
семантическая оценка.

| Модель | KeywordRate | Attention support | MLP support |
|---|---:|---|---|
| Gemma-2-2B-it | 0.04 | 2–25 | 7–25 |
| Gemma-3-12B-it | 0.57 | 15–47 | 16–47 |
| Gemma-4-E4B-it | 0.27 | 9–41 | 7–41 |
| Ministral-3-3B-text | 0.02 | 8–25 | 3–25 |
| Qwen2.5-3B | 0.03 | 12–35 | 10–35 |
| Qwen2.5-VL-7B | 0.04 | 4–27 | 11–23 |
| Qwen3-4B | 0.05 | 6–35 | 11–35 |
| Qwen3-8B | 0.11 | 4–35 | 9–35 |
| Qwen3-VL-4B | 0.10 | 8–35 | 17–25 |
| Qwen3-VL-8B | 0.06 | 7–35 | 13–35 |
| Qwen3.5-9B | 0.79 | 18–27 | 14–24 |

Повторяющийся паттерн:

- многие профили не имеют локального пика;
- они резко включаются в средней части;
- затем остаются почти плоскими до последнего слоя;
- Qwen2.5-VL, Qwen3-VL-4B и Qwen3.5 дают примеры действительно ограниченного
  MLP-окна.

Это поддерживает поиск границ и радиуса. Но старые trial использовали маркеры и
first-token KL, поэтому не доказывают, что эти интервалы причинно оптимальны.

Особенно подозрительны:

- Gemma-3: 57 отказов и KL 0.086 даже в агрессивной точке;
- Gemma-4: 27 отказов и KL 0.157.

Для Gemma-4 нетронутая PLE-ветка является конкретным альтернативным объяснением.

## 9. Приоритет экспериментальных моделей

### Приоритет 1: Qwen2.5-3B-Instruct

Почему:

- BF16;
- 36 однородных full-attention слоёв;
- нет PLE, linear attention, MTP и output gates;
- чистые `o_proj/down_proj`;
- существующий журнал и heretic build;
- достаточно мал для многошкального скана.

Это лучший контроль вопроса: помогает ли локальное окно само по себе.

### Приоритет 2: Gemma-2-2B-it

Почему:

- самая дешёвая instruction-модель;
- уже есть 400-trial study, четыре сохранённые точки и семантическая разметка
  одного прогона;
- 26 слоёв сокращают coarse-to-fine budget.

Ограничение: обязательно учитывать alternating sliding/full attention. Сначала
можно сканировать общий attention output, затем разложить лучший интервал по
типам.

### Приоритет 3: Gemma-4-E4B-it

Цель другая: не общая проверка кривой, а тест PLE re-injection.

Минимальный факторный эксперимент в одном найденном окне:

```text
attention only
MLP only
PLE projection only
attention + MLP
attention + MLP + PLE
```

Если добавление PLE резко снижает остаточные отказы без большого роста
perplexity, найден новый дешёвый target.

### Приоритет 4: Qwen3.5-9B

Использовать после чистого dense-контроля:

- проверить разделённые linear/full schedules;
- логировать gate activations;
- проверить локальные интервалы 14–27, предложенные старым журналом;
- отдельно решить сохранение MTP.

Его старый best KeywordRate 0.79 показывает, что два прежних output targets
почти не снимают отказ; это хорошая модель для поиска пропущенных веток, но
плохая первая модель для проверки формы кривой.

## 10. Конкретный следующий диагностический пакет

Без изменения основного Optuna:

1. Qwen2.5-3B:
   - attention и MLP;
   - окна шириной 8, затем 4 и 2;
   - `lambda = 1` и `lambda = 2`.
2. Gemma-2:
   - тот же скан;
   - для top windows повторить full-only/sliding-only.
3. Gemma-4:
   - только после нахождения общего интервала;
   - добавить PLE projection как третий фактор.
4. На каждой модели:
   - semantic prompt-level flips;
   - perplexity;
   - harmless/tech-sensitive;
   - несколько prompt splits.

Такой порядок отделяет три гипотезы:

- полезна ли локализация по глубине;
- зависит ли она от типа attention;
- существует ли повторная запись отказа через пропущенную residual branch.

## 11. Вывод

Плотных моделей достаточно, чтобы проверить гипотезу до 35B MoE.

Наиболее важные новые факты:

1. Старые фронты чаще находят границу включения, а не пик.
2. Gemma-2/3/4 смешивают full и sliding attention под одним ключом.
3. Gemma-4 имеет крупную PLE-ветку и дешёвый выходной target на 27.5M весов.
4. Qwen3.5 имеет более полумиллиарда gate weights, которые надо сначала
   диагностировать по активациям.
5. MTP не «возвращается неизменным» в проверенную Qwen3.5 heretic-сборку — он
   исчезает при export.
6. В Gemma output targets находятся до post-RMSNorm. Для точной операции над
   residual direction нужен pre-norm вектор `normalize(norm_scale * v)`;
   текущая проекция по `v` этого не учитывает.
7. Router Qwen3.6 является input-side target, а не ещё одной output projection:
   его rank-1 правка имеет вид `W - lambda (Wv)v^T`.

Алгебраическая проверка и точная карта локальных checkpoint вынесены в
[STATIC_MECHANICS_AUDIT.md](STATIC_MECHANICS_AUDIT.md).
