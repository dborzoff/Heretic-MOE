# Пакет для обсуждения локальной инверсии с Claude

Дата: 2026-07-29.

Подробности:

- [LOCALIZED_INVERSION_DESIGN.md](LOCALIZED_INVERSION_DESIGN.md)
- [LAYER_LOCALIZATION_SEARCH.md](LAYER_LOCALIZATION_SEARCH.md)
- [PROBLEM_ZONE_LOCALIZATION.md](PROBLEM_ZONE_LOCALIZATION.md)
- [MODEL_ARCHITECTURE_SURVEY.md](MODEL_ARCHITECTURE_SURVEY.md)
- [STATIC_MECHANICS_AUDIT.md](STATIC_MECHANICS_AUDIT.md)

## Гипотеза пользователя

Вместо широкой линейной правки по глубине:

1. найти слой или компактную зону, где отказная компонента возникает или
   усиливается;
2. оставить остальные слои без изменений;
3. внутри зоны плавно поднять `λ` до режима удаления или инверсии;
4. управлять центром, радиусом и формой краёв как Bézier/logarithmic handles.

Целевая форма:

```text
λ(layer): 0 → smooth rise → approximately 1 or 2 → smooth fall → 0
```

## Что уже подтверждено по коду

1. Текущий профиль — линейная палатка с жёстким обнулением за радиусом:
   `src/heretic/model.py:520-530` и `src/heretic/model.py:712-717`.
2. При положительном `min_weight` на границе возникает разрыв.
3. `max_weight_position` двигает одновременно пик и обе границы; это не чистый
   поиск слоя.
4. Направления нормализуются по каждому слою в `src/heretic/main.py:577`, а
   исходная величина `bad_mean-good_mean` удаляется. По текущим сохранённым
   данным нельзя построить честный график силы сырого отказного сигнала.
5. В голом проекторе `λ=1` стирает компоненту, `λ=2` меняет её знак.
6. `FULL` row normalization нарушает точную интерпретацию `λ`, слитый путь пока
   ближе к голой алгебре.

## Что показывает старый журнал

`F:\AI\checkpoints\qwen36-run2\journal-final.jsonl`, 524 complete trials.

Сильные точки нового фронта используют:

- attention как почти плоскую раннюю интервенцию с cutoff около слоя 30–33;
- MLP как позднее включающееся окно или широкую интервенцию до конца;
- веса около `2.0` для attention и до `2.5` для MLP.

Это согласуется с инверсией, но не доказывает её пользу: objective был
KeywordRate + first-token KL, а KeywordRate теперь известен как ненадёжный.
Кроме того, журнал получен до разделения full/linear attention и
routed/shared MLP, поэтому его геометрию нельзя прямо назначить новым четырём
компонентам.

Также это означает, что текущий оптимизатор использует `position + distance`
главным образом как координаты границ, а не как пик.

## Предлагаемая минимальная функция

```text
S_p(t) = t^p / (t^p + (1-t)^p)

B(x) = 0                                      outside [c-r, c+r]
B(x) = S_p((x-(c-r))/r)                      left side
B(x) = S_p(((c+r)-x)/r)                      right side

λ(x) = amplitude × B(x)
```

Первая версия:

```text
center
radius
amplitude
edge_power
```

Вторая версия только после выигрыша:

```text
left_radius/right_radius
left_power/right_power
optional plateau_fraction
```

## Предлагаемый порядок работы

1. Не менять основной Optuna немедленно.
2. Сделать отдельный diagnostic scan на Gemma.
3. Сохранять raw per-layer signal до нормализации.
4. Coarse-to-fine проверить локальные окна на attention/MLP.
5. В каждом окне сравнить `λ=1` и `λ=2`.
6. Оценивать prompt-level semantic flips и perplexity.
7. Проверить top windows на полном frozen eval.
8. Только при устойчивом эффекте добавить новый schedule family.
9. Запустить новую study с одинаковым бюджетом против старой формы.

## Вопросы Claude

1. Согласен ли он, что старые front trials идентифицируют скорее границы
   включения/выключения, чем положение пика?
2. Что честнее сделать первым: компактный bump или smooth plateau/shutter?
3. Нужно ли диагностическое вмешательство выполнять через тот же weight-edit
   path, или допустим residual-stream hook как быстрый предфильтр?
4. Как выровнять смысл `amplitude` между dense `FULL` и fused expert path до
   теста формы?
5. Можно ли переиспользовать текущий reset/cache path для быстрого window scan
   без merge модели?
6. Какой differentiable proxy использовать для attribution prefilter, не
   возвращаясь к 33 маркерам?
7. Следует ли искать одно окно на компонент или сначала общий центр и отдельные
   амплитуды?
8. Какой тест неаддитивности достаточен, прежде чем разрешать второй bump?
9. Нужно ли сразу разделять направления harm/restricted, или сначала изолировать
   эффект формы на прежнем направлении?

## Моя текущая оценка

Гипотеза достойна дешёвого причинного эксперимента и лучше мотивирована, чем
простое расширение диапазонов старой палатки.

Главный риск — перепутать локализацию доступного отказного сигнала с
локализацией причинного bottleneck. Второй риск — сгладить полезный cutoff,
который нынешняя формула случайно умеет создавать.

Поэтому правильный первый результат — не новая Bézier-реализация, а карта
`component × center × radius × amplitude`, подтверждённая смысловыми переходами
ответов. Если компактная зона не воспроизводится, усложнять кривую бессмысленно.

## Дополнение после просмотра локальных весов

1. Gemma-2/3/4 смешивают sliding и full attention под одним
   `attn.o_proj`. Для честной локализации их следует диагностически разделить.
2. Gemma-4 после MLP пишет в residual третью ветку
   `per_layer_input_gate → per_layer_projection`; последняя содержит всего
   27.5M весов и управляет сигналом из 2.82B per-layer embedding. Проверенные
   фрагменты этой ветки в original/heretic совпадают бит-в-бит.
3. В оригинальном Qwen3.5-9B присутствуют 243.29M MTP weights, а в локальном
   heretic export нет ни одного MTP tensor, хотя config всё ещё объявляет один
   MTP layer. Следовательно, перед обсуждением правки MTP надо сначала исправить
   или осознанно определить его сохранение.
4. Чистым первым контролем выбран Qwen2.5-3B-Instruct; Gemma-2 — вторым дешёвым
   сканом, Gemma-4 — отдельным тестом PLE re-injection.

## Дополнение после статической проверки механики

1. В Gemma-2/3/4 `o_proj/down_proj` проходят через отдельный post-RMSNorm до
   residual addition. Поэтому текущая проекция по residual `v` применяется в
   неверной метрике. Для norm scale `g` корректное направление до norm:
   `u = normalize(g * v)`.
2. При raw projector это даёт точное удаление при `lambda=1` и точную смену
   знака post-RMSNorm-компоненты при `lambda=2`. Алгебраический dry-run прошёл
   с ошибкой порядка `1e-15`.
3. `row_normalization=FULL` разрушает точный Householder reflection ещё до
   SVD. Для исследования инверсии нужен отдельный `raw_exact`, а старый режим
   надо сохранить как `legacy_full`.
4. Практическая Bézier-форма уточнена до компактного smoothstep-окна с
   `center`, `inner_radius`, `outer_radius`, `amplitude`: центральное плато
   сохраняется, оба края фиксированы в нуле.
5. Router Qwen3.6 требует input-side операции:
   `W' = W - lambda (Wv)v^T`. Это rank-1 LoRA и технически реализуемо на
   20,971,520 весах, но top-k делает эффект дискретным.
6. Qwen gate нельзя править residual `v` только из-за совпадающей размерности:
   они живут во внутренних head/gate координатах. Им нужны собственные
   направления.
7. Для Gemma-4 E4B `enable_moe_block=false`; routed experts там нет.
   Подтверждённый дополнительный residual target — 42
   `per_layer_projection`, всего 27,525,120 весов, также с norm-aware
   направлением.
8. Машинный аудит config и safetensors headers лежит в
   `research/results/model_topology_audit.json`; основной код и модели не
   изменялись.

## Дополнение: улучшенный поиск проблемной зоны

Новый разбор в `PROBLEM_ZONE_LOCALIZATION.md` меняет единый layer peak на
четыре карты:

```text
detection → residual writing → policy routing → actual weight-edit leverage
```

Ключевые предложения:

1. строить matched sensitive/control pairs и отдельно использовать фактические
   `REFUSAL/SOFT/COMPLY`, чтобы не смешивать harmfulness с refusal;
2. искать не максимум накопленного residual score, а его приращение и
   post-norm contribution каждой ветки;
3. делать малый signed `+eps/-eps` scan тем же weight projector, получать
   sensitivity и curvature;
4. проверять top candidates direction-specific necessity/sufficiency
   interchange, не перенося между запросами весь residual;
5. считать edit leverage по semantic benefit, harmless damage и perplexity,
   потому что causal localization не обязана совпадать с лучшим edit layer;
6. получать границы total-variation/change-point segmentation, затем
   причинно проверять целые окна;
7. для MoE раздельно фиксировать routing и expert outputs и проверять expert
   coalitions, а не только singleton experts;
8. требовать cross-fitting, leave-one-topic-out и bootstrap selection
   probability вместо одного argmax.

Минимальный контроль предлагается делать без SAE/CLT на Qwen2.5-3B, затем
перенести на Gemma-2, Gemma-4 PLE, Qwen3.5 gates и только после этого на
Qwen3.6 router/expert coalitions.
