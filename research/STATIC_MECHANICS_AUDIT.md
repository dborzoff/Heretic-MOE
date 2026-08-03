# Статическая проверка механики Heretic на локальных моделях

Дата: 2026-07-29.

Статус: проверены кодовые пути, реальные config и safetensors headers. Модели
не запускались, промпты и ответы не читались, основной код Heretic не изменён.

Вспомогательные артефакты:

- `scripts/audit_model_topology.py` — читает только config и заголовки
  safetensors;
- `scripts/verify_projection_mechanics.py` — алгебраический dry-run;
- `results/model_topology_audit.json` — машинный результат по девяти локальным
  checkpoint.

## 1. Главный результат

Гладкое окно по глубине реализуемо, но одной заменой линейной палатки выигрыш
не гарантирован. В нынешней реализации смешаны три разных операции:

1. прямая output-side проекция в Qwen-подобных residual blocks;
2. проекция **до** post-RMSNorm в Gemma;
3. прямая правка слитых MoE-экспертов без той нормировки, которая применяется
   к плотным модулям.

Поэтому одинаковый `lambda` сейчас не означает одинаковую интервенцию даже до
учёта формы кривой.

Самая конкретная поправка: для Gemma направление надо переносить через
post-RMSNorm, а режим `FULL` нельзя использовать, если `lambda=2` должен
означать настоящую инверсию.

## 2. Где операция точна, а где нет

Для прямого выхода в residual stream:

```text
y = W x
h_next = h + y
```

сырая правка

```text
W' = (I - lambda v v^T) W
```

имеет точный смысл:

- `lambda=1` удаляет выходную компоненту вдоль `v`;
- `lambda=2` выполняет Householder reflection и меняет её знак;
- при `lambda=2` норма `W x` сохраняется.

Так устроены проверенные Qwen output targets: `self_attn.o_proj`,
`linear_attn.out_proj` и `mlp.down_proj` прибавляются к residual без
post-нормировки ветки.

### Gemma: post-RMSNorm меняет нужное направление

В Gemma-2/3/4 ветка имеет вид:

```text
y = W x
z = RMSNorm_g(y)
h_next = h + z
```

Это видно в локальном Transformers:

- Gemma-2: `modeling_gemma2.py:324-344`;
- Gemma-3: `modeling_gemma3.py:415-434`;
- Gemma-4: `modeling_gemma4.py:1416-1450`.

Пусть

```text
RMSNorm_g(y) = diag(g) y / rms(y).
```

Чтобы удалить именно `v` из **нормированного** выхода, проектировать исходную
матрицу надо не по `v`, а по

```text
u = normalize(g * v)
W' = (I - lambda u u^T) W.
```

Тогда:

```text
lambda=1  =>  v^T RMSNorm_g(W'x) = 0
lambda=2  =>  v^T RMSNorm_g(W'x) = -v^T RMSNorm_g(Wx)
```

для любого `x`. При `lambda=2` знаменатель RMSNorm не меняется, потому что
Householder reflection сохраняет норму `W x`.

Важно правильно получить `g`:

- Gemma-2/3: `g = 1 + norm.weight`;
- Gemma-4: `g = norm.weight`.

Для attention используется `post_attention_layernorm`, для MLP —
`post_feedforward_layernorm`, для PLE —
`post_per_layer_input_norm`.

### Проверка dry-run

`verify_projection_mechanics.py` проверяет эти тождества в `float64`:

- ошибка raw removal: `4.44e-16`;
- ошибка raw sign flip: `1.78e-15`;
- ошибка norm-aware removal: `4.44e-16`;
- ошибка norm-aware sign flip: `8.88e-16`.

На тех же детерминированных тензорах наивная проекция по `v` перед
неравномерным RMS scale оставляет компоненту `0.472`, то есть сама
алгебраическая гарантия исчезает.

Это не метрика качества модели, а проверка механики операции.

## 3. Почему `row_normalization=FULL` несовместима с точной инверсией

Путь `FULL` в `src/heretic/model.py:589-642` делает:

1. нормировку каждой строки `W`;
2. rank-1 projector;
3. повторную построчную нормировку;
4. восстановление старых row norms;
5. SVD-аппроксимацию дельты.

Householder reflection сохраняет общую евклидову норму выхода, но не нормы
отдельных строк матрицы. Поэтому шаг 3 меняет сам reflection. В dry-run ошибка
смены знака уже **до SVD** равна `0.481`.

Следствие:

- старые исследования должны оставаться в `legacy_full` для
  воспроизводимости;
- новая локальная инверсия должна иметь отдельный режим `raw_exact`;
- сравнивать старый и новый журнал как одну шкалу `max_weight` нельзя, их надо
  переоценивать на одинаковом frozen eval.

Для `lambda=1` norm-preserving вариант всё ещё может быть полезной
регуляризацией, но называться точной ортогональной проекцией он не должен.

## 4. Практическая Bézier-форма

Предложению пользователя лучше соответствует не одиночный треугольный пик, а
компактное окно с центральной зоной и гладкими краями:

```text
center        — середина зоны;
inner_radius  — половина неизменного центрального плато;
outer_radius  — полный радиус влияния;
amplitude     — lambda в центральной зоне.
```

Для края:

```text
S(t) = 3t^2 - 2t^3
```

Это cubic Hermite smoothstep, эквивалентный Bézier-краю с нулевой производной
на обеих опорных точках.

```text
d = abs(layer - center)

lambda = amplitude,                                      d <= inner_radius
lambda = amplitude * S((outer_radius-d) /
                       (outer_radius-inner_radius)),      inner < d < outer
lambda = 0,                                              d >= outer_radius
```

Свойства:

- начало и конец зоны остаются ровно в нуле;
- центральная зона остаётся ровно на выбранной амплитуде;
- края непрерывны и имеют нулевой наклон;
- `inner_radius=0` даёт гладкий bump;
- `inner_radius>0` даёт гладкую «шторку»;
- `amplitude=1` удаляет, `amplitude=2` инвертирует только в режиме
  `raw_exact`.

В dry-run для `center=12`, `inner_radius=2`, `outer_radius=6`,
`amplitude=2` получился дискретный профиль:

```text
слои 0-6:     0
слой 7:       0.3125
слой 8:       1.0
слой 9:       1.6875
слои 10-14:   2.0
слой 15:      1.6875
слой 16:      1.0
слой 17:      0.3125
слои 18-24:   0
```

Это прямо реализует «сохранить начальную и конечную точки, двигать центр и
радиус влияния».

## 5. Как провести окно через код

Сейчас одна и та же линейная формула скопирована в:

- плотный путь: `src/heretic/model.py:520-531`;
- fused-expert path: `src/heretic/model.py:712-717`.

Нужен единый helper:

```text
get_layer_weight(layer_index, schedule) -> float
```

и два семейства:

```text
legacy_tent
smooth_compact
```

Старые поля `max_weight_position/min_weight_distance` нельзя молча
переопределять: это сломает восстановление старых trial. Новое семейство должно
иметь отдельную сериализацию:

```text
family
center
inner_radius
outer_radius
amplitude
```

Только helper должен решать, какой `lambda` получает слой. Плотный и fused путь
обязаны вызывать один helper; иначе шкалы снова разойдутся.

Следующий уровень plumbing — описание target:

```text
ProjectionTarget(
    module_or_parameter,
    side="output" | "input",
    post_norm_scale=None | tensor,
)
```

Это требуется не ради абстракции, а потому что router математически правится с
другой стороны матрицы.

## 6. Router — реализуемый, но другой projector

Для Qwen3.6 router:

```text
W_router [256, 2048]
logits = W_router h
```

Residual direction имеет размер `2048`, то есть живёт на **входе** router.
Корректная rank-1 операция:

```text
W_router' = W_router (I - lambda v v^T)
          = W_router - lambda (W_router v) v^T.
```

Она означает:

- `lambda=1`: logits router перестают зависеть от компоненты `h` вдоль `v`;
- `lambda=2`: эта зависимость меняет знак.

Дельта по-прежнему rank-1 и представима LoRA:

```text
B = -lambda (W_router v)   # [experts, 1]
A = v^T                    # [1, hidden]
```

Это подтверждает реализуемость идеи с 20,971,520 весами router вместо правки
10,737,418,240 весов routed `down_proj`.

Но это не подтверждает выигрыш:

- top-k делает малую правку logits дискретной сменой маршрута;
- residual direction может кодировать отказ в выходе, но не быть причиной
  выбора эксперта;
- изменение router меняет, **какие функции вызываются**, а не удаляет отказную
  компоненту из их результата.

Поэтому router — сильный изолированный рычаг, но не замена output projector по
одной лишь экономии параметров.

## 7. Gates: почему их нельзя править тем же `v`

Статический аудит подтвердил:

| Модель | Gate | Параметры |
|---|---|---:|
| Qwen3.5-9B | 24 × `linear_attn.in_proj_z` | 402,653,184 |
| Qwen3.5-9B | gate-половина 8 × `self_attn.q_proj` | 134,217,728 |
| Qwen3.6-35B-A3B | 30 × `linear_attn.in_proj_z` | 251,658,240 |
| Qwen3.6-35B-A3B | gate-половина 10 × `self_attn.q_proj` | 83,886,080 |
| Qwen3.6-35B-A3B | 40 × `shared_expert_gate` | 81,920 |

Эти gate не пишут в residual напрямую. Они умножают сигнал в другом,
внутреннем базисе:

- `in_proj_z` — head-wise latent gate GatedDeltaNet;
- в `q_proj` query и gate упакованы в одну матрицу;
- `shared_expert_gate` выдаёт один sigmoid-скаляр на токен.

Совпадение размерности gate с `hidden_size` не означает совпадение координат.
Применять к ним residual `v` как output direction математически
необоснованно.

Безопасная последовательность:

1. сначала оставить gate неизменными и править находящийся после них
   `out_proj/o_proj`;
2. если gate станет отдельным target, считать его собственное activation
   direction;
3. для packed `q_proj` менять только gate-половину, не query rows;
4. для scalar shared gate проверять input-side удаление чувствительности к `v`,
   а не output-side projector.

## 8. Gemma-4 PLE

Аудит реального `google__gemma-4-E4B-it` подтвердил:

- `enable_moe_block=false`; предположение о пропущенных routed experts для
  этого checkpoint неверно;
- во всех 42 слоях есть `per_layer_input_gate` и
  `per_layer_projection`;
- каждый класс содержит по 27,525,120 весов;
- `per_layer_projection` пишет в residual через
  `post_per_layer_input_norm`.

Поэтому самый чистый PLE target:

```text
layer.per_layer_projection
```

с norm-aware направлением:

```text
u = normalize(layer.post_per_layer_input_norm.weight * v).
```

`per_layer_input_gate` — latent gate; к нему residual `v` напрямую применять
не следует.

## 9. Per-expert gamma

Для fused expert `e`:

```text
p_e = ||v^T W_e||
lambda_e = lambda * (p_e / p_max)^gamma
```

реализуется векторно в существующем `einsum`. Но `p_e` измеряет только
геометрическую доступность направления в весах. Он не учитывает:

- вызывается ли эксперт на нужных токенах;
- какие его входные признаки реально активны;
- является ли его вклад причиной отказа.

Кроме того, при одинаковом глобальном `lambda` gamma автоматически меняет
общий бюджет дельты. Честное сравнение требует нормировать масштабы так, чтобы

```text
sum_e ||Delta W_e||_F^2
```

совпадала с uniform baseline, либо сравнивать на равном routed-output delta.
Иначе улучшение может снова оказаться артефактом меньшей/большей правки.

Вывод: gamma технически прост, но по силе обоснования стоит после router и
пропущенных residual outputs.

## 10. Проверенное покрытие локальных checkpoint

| Модель | Слои | Текущие output targets | Нетронутые кандидаты |
|---|---:|---|---|
| Gemma-2-2B-it | 26 | 26 attention, 26 MLP | split sliding/full, norm-aware direction |
| Gemma-3-12B-it | 48 | 48 attention, 48 MLP | split sliding/full, norm-aware direction |
| Gemma-4-E4B-it | 42 | 42 attention, 42 MLP | 42 PLE projection/gate |
| Ministral-3-3B | 26 | 26 attention, 26 MLP | явной третьей ветки нет |
| Qwen2.5-3B | 36 | 36 attention, 36 MLP | чистый контроль |
| Qwen3-4B/8B | 36 | 36 attention, 36 MLP | явной третьей ветки нет |
| Qwen3.5-9B | 32 | 8 full, 24 linear, 32 MLP | два gate-механизма, MTP |
| Qwen3.6-35B-A3B | 40 | 10 full, 30 linear, 40 routed + 40 shared expert | router, gates, MTP |

Для Qwen3.6 точные размеры:

```text
routed expert down_proj: 10,737,418,240
shared expert down_proj:     41,943,040
router:                      20,971,520
shared expert gate:              81,920
MTP:                        844,640,768
```

## 11. MTP

Аудит headers подтвердил:

- Qwen3.5 original: 15 tensors, 243,290,624 parameters;
- Qwen3.6 original: 19 tensors, 844,640,768 parameters.

В проверенном Qwen3.5 heretic export MTP отсутствует полностью, хотя config
сохраняет `mtp_num_hidden_layers=1`. Поэтому первый MTP fix — сохранение
артефакта, а не abliteration.

## 12. Реальный порядок улучшений

По проверенной механике, а не по привлекательности идеи:

1. Вынести единый `legacy_tent/smooth_compact` helper для dense и fused путей.
2. Добавить `raw_exact` и не смешивать его шкалу со старым `FULL`.
3. Для Gemma использовать post-RMSNorm-aware direction.
4. Разделить full/sliding attention schedules, не меняя старый ключ по
   умолчанию.
5. Добавить Gemma-4 `per_layer_projection` как output target.
6. Добавить router как отдельный **input-side** target.
7. Сначала сохранить MTP, затем решать, нужно ли его править.
8. Gate edits и per-expert gamma оставить последующими, потому что им нужны
   отдельные направления или equal-budget контроль.

Таким образом, Bézier-окно имеет смысл, но наиболее вероятный чистый выигрыш
даст связка:

```text
compact smooth window
+ exact raw projector
+ norm-aware Gemma directions
+ architecture-specific residual outputs
```

а не одна более красивая кривая поверх прежней неоднородной операции.

