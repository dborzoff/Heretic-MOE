# Как надёжнее находить проблемные зоны Heretic

Дата: 2026-07-29.

Статус: исследовательский дизайн после проверки кода Heretic, forward-путей
локальных моделей и современной литературы. Основной код и модели не изменены.

Связанные документы:

- [LAYER_LOCALIZATION_SEARCH.md](LAYER_LOCALIZATION_SEARCH.md) — исходный
  coarse-to-fine план;
- [STATIC_MECHANICS_AUDIT.md](STATIC_MECHANICS_AUDIT.md) — точная механика
  projector и карта residual writers;
- [LOCALIZED_INVERSION_DESIGN.md](LOCALIZED_INVERSION_DESIGN.md) — форма
  гладкого окна.

## 1. Главный вывод

Один график `||mean_bad - mean_good||` не находит «слой отказа». Он смешивает
как минимум четыре разных явления:

1. **Detection** — модель распознала тему или вредность.
2. **Writing** — конкретный attention/MLP/PLE записал связанный сигнал в
   residual stream.
3. **Routing** — небольшой gate решил, в какую политику направить дальнейшее
   вычисление.
4. **Edit leverage** — изменение именно этих весов дало полезный поведенческий
   эффект с малым побочным ущербом.

Для Heretic нужен прежде всего четвёртый объект. Он может не совпасть ни с
максимальной разделимостью активаций, ни с причинным gate. Исследования model
editing уже показывали, что causality-based localization не обязательно
предсказывает лучший слой для weight edit.

Правильный результат локализации — не один номер слоя, а:

```text
(компонент, диапазон слоёв, роль, направление/подпространство,
 causal effect, edit leverage, uncertainty)
```

## 2. Почему текущий поиск может выбирать не ту зону

### 2.1 Harmful против harmless смешивает вредность и отказ

Текущий direction:

```text
v_l = normalize(mean(harmful_l) - mean(harmless_l))
```

обучен на разных распределениях запросов. Он может описывать:

- тему;
- стиль AdvBench;
- длину и chat-template;
- распознанную вредность;
- решение отказать;
- уже сформированную формулировку ответа.

Работа *LLMs Encode Harmfulness and Refusal Separately* причинно разделяет
harmfulness direction и refusal direction. Значит, один contrast не должен
использоваться одновременно как detector и target для удаления.

### 2.2 Residual накапливает прошлые записи

Если сигнал записан в слое 12 и просто переносится до слоя 25, probe будет
высоким во всех слоях 12–25. Максимум на слое 25 не означает, что править надо
слой 25.

Нужно отдельно считать:

```text
state_score_l       = readout(h_l)
write_increment_l   = state_score_{l+1} - state_score_l
```

и вклад каждой ветки:

```text
attn_write_l
mlp_write_l
ple_write_l
expert_write_l
```

### 2.3 Direct contribution не видит gates

Direct logit attribution хорошо находит компоненты, которые несут сигнал к
выходу. Но gate может записывать очень мало напрямую и при этом быть причинно
необходимым: работа *How Alignment Routes* сообщает gate с вкладом менее 1%
в output DLA, который проходит necessity/sufficiency interchange tests.

Поэтому `projection magnitude` и DLA нельзя использовать как единственный
ранжирующий показатель.

### 2.4 Activation patching не гарантирует лучший weight-edit layer

Патч активации отвечает на вопрос «где вычисление причинно проходит». Heretic
задаёт другой вопрос: «в каком output matrix rank-1 изменение наиболее выгодно».
Это надо проверять тем же projector, который попадёт в итоговую модель.

### 2.5 Один token position недостаточен

`Model.get_residuals()` вызывает `generate(max_new_tokens=1)` и берёт
`hidden_states[0][:, -1]`. Практически это состояние последнего prompt token,
по которому предсказывается первый response token. Комментарий
«first generated token» неточен: сгенерированный token ещё не прошёл следующий
forward.

Для обычной chat-модели prompt boundary важен. Но при response prefix, скрытом
reasoning/CoT или длинном шаблоне решение может смещаться по token axis.

Локализация должна быть минимум двумерной:

```text
layer × token_position
```

а для MoE:

```text
layer × token_position × expert/route
```

## 3. Четыре карты вместо одного пика

| Карта | Вопрос | Основная метрика | Что она не доказывает |
|---|---|---|---|
| Availability | Где информация читается? | cross-validated probe, separation | причинность |
| Write | Где ветка добавляет сигнал? | post-norm component contribution, increment | управление downstream |
| Route | Где меняется политика? | necessity/sufficiency interchange, cascade | лучший слой для weight edit |
| Edit leverage | Где правка выгодна? | paired semantic benefit / collateral damage | универсальность вне eval |

Кандидат становится «проблемной зоной Heretic» только после четвёртой карты.

## 4. Данные: самое важное улучшение до любых hooks

### 4.1 Matched contrast pairs

Нельзя сравнивать произвольный AdvBench с произвольным Alpaca. Нужны пары с
максимально одинаковыми:

- темой;
- синтаксисом;
- длиной;
- форматом;
- технической лексикой;

но разным policy-relevant признаком.

Для каждой пары:

```text
sensitive_prompt_id
matched_control_id
topic
pair_id
```

Текст в diagnostic artifacts не нужен; достаточно IDs.

Если идеальных пар мало, разность считается сначала внутри темы/источника, а
затем агрегируется:

```text
delta_l = mean_topic(mean_sensitive_l - mean_control_l)
```

Это не даёт крупным классам и шаблону AdvBench захватить direction.

### 4.2 Разделить три набора меток

1. `input_sensitive`: свойство запроса.
2. `response_verdict`: фактические `REFUSAL/SOFT/COMPLY`.
3. `topic/source`: nuisance variables.

Тогда можно построить:

```text
detection_direction: sensitive vs matched control
refusal_direction:   refused vs complied, только внутри sensitive
routing_direction:   refusal после вычитания topic/source effects
```

Это прямо проверяет, не удаляет ли Heretic распознавание вредности вместо
механизма отказа.

### 4.3 Boundary-critical prompts

Сильные отказы полезны для финальной валидации, но плохо показывают, какая
ветка перевела модель через decision boundary. CRaFT предлагает отбирать
промпты, где competing refusal/compliance paths близки.

Практический диагностический набор делится на три страты:

```text
firm_refusal
boundary / SOFT / high uncertainty
firm_comply
```

Boundary-набор используется для причинного ранжирования, firm-наборы — для
проверки диапазона применимости.

### 4.4 Cross-fitting и leave-one-topic-out

Для каждого fold:

1. direction/probe строятся только на train IDs;
2. layer ranking считается на validation IDs;
3. causal edit проверяется на held-out IDs;
4. отдельный прогон оставляет целую тему вне обучения.

Высокая случайная probe accuracy не считается доказательством: современные
работы показывают, что даже permutation/null controls могут выглядеть
идеальными, а информативен перенос на невиданные категории.

## 5. Какие активации действительно записывать

### 5.1 Residual boundary states

Для каждого слоя:

```text
residual_pre_layer
residual_after_attention
residual_after_mlp
residual_after_extra_branch
```

### 5.2 Фактические residual writers

Записывать надо выход, который реально прибавляется к residual:

- Qwen: `o_proj/out_proj/down_proj` output;
- Gemma-2/3/4: output **после** соответствующего post-RMSNorm;
- Gemma-4: дополнительно PLE после `post_per_layer_input_norm`;
- Qwen3.5/3.6: раздельно full attention и linear attention;
- Qwen3.6 MoE: routed coalition output и shared expert output.

Pre-norm output тоже полезен, но только для связи с будущим weight edit.

### 5.3 Gate и route states

Отдельными полями:

```text
full_attn_gate
linear_attn_z_gate
router_logits
router_probabilities
selected_expert_ids
shared_expert_gate
```

Gate activations не надо проецировать на residual `v`: это другой базис.

### 5.4 Token positions

Минимум:

```text
last_user_content_token
last_prompt_token
mean(last 8 user tokens)
first response-step state
```

Для reasoning-моделей добавить несколько response/CoT positions. Если зоны
различаются, weight edit должен оптимизироваться по тому моменту, где он реально
будет работать при генерации, а не только по prompt boundary.

## 6. Этап A: дешёвая observational localization

### 6.1 Raw и standardized signal

Сохранять:

```text
||delta_l||
||delta_l|| / residual_rms_l
heldout projection effect
cross-validated AUROC
balanced accuracy
split-half direction cosine
leave-topic-out score
```

### 6.2 Производная по глубине

Вместо поиска максимума `state_score_l` считать:

```text
emergence_l = state_score_{l+1} - state_score_l
```

Для component output:

```text
write_l,c = readout(component_output_l,c)
```

Это отличает появление сигнала от его переноса.

### 6.3 Direct readout с двумя осями

Нужны как минимум два независимых readout:

```text
harmfulness_readout
refusal_or_policy_readout
```

Слой, высокий только по harmfulness, является detector candidate, но не
abliteration target.

### 6.4 Нулевые контроли

Обязательны:

- shuffled verdict внутри topic;
- случайные ортогональные directions той же нормы;
- matched benign против matched benign;
- prompt-length-only baseline;
- source/template classifier.

Если layer peak сохраняется в source/template control, это не refusal zone.

## 7. Этап B: signed causal sensitivity тем же оператором

Вместо немедленного перебора больших `lambda` каждый
`layer × component` сначала получает малую двустороннюю интервенцию:

```text
J(+eps)
J(0)
J(-eps)
```

где используется тот же raw/norm-aware weight projector, что и в финальной
модели.

Локальная чувствительность:

```text
g_j = (J(+eps) - J(-eps)) / (2 eps)
```

Нелинейность:

```text
q_j = (J(+eps) - 2J(0) + J(-eps)) / eps^2
```

Интерпретация:

- большой устойчивый `g_j` — хороший edit-leverage candidate;
- большой `q_j` — малый линейный proxy ненадёжен, нужен настоящий dose sweep;
- разные знаки между folds/topics — единого окна нет.

Отрицательный `lambda` нужен только диагностически: он усиливает исходную
компоненту и даёт симметричную finite difference. В сохранённую модель он не
попадает.

### 7.1 Continuous proxy

33 substring markers не подходят. Для быстрого этапа использовать ансамбль:

1. contrastive log-prob нескольких коротких refusal/compliance continuation
   templates;
2. cross-fitted response-verdict probe;
3. refusal-direction readout;
4. KL/perplexity на harmless control.

Кандидат проходит дальше только при согласованном знаке нескольких proxy.
Финальное решение всё равно принимает semantic verdict полного ответа.

### 7.2 Signed structured masks

Если `layer × component` слишком много, применить 24–64 гладких или
Hadamard/Rademacher masks:

```text
lambda_j = eps * mask_j
```

Каждую mask запускать парой `+mask/-mask`. Разность подавляет чётные
нелинейности и baseline drift. Послойные коэффициенты восстанавливать
total-variation/group-lasso regression:

```text
min ||y - M beta||^2
    + alpha ||beta||_1
    + tau TV(beta_by_depth)
```

Это предфильтр. Верхние коэффициенты обязательно проверяются одиночными edits.

AtP* можно использовать вместо mask regression, но только как ranking:
метод специально исправляет два класса false negatives обычного attribution
patching, однако всё равно остаётся приближением.

## 8. Этап C: direction-specific interchange

Полная замена residual между разными запросами переносит содержание и создаёт
артефакты. Лучше менять только коэффициент исследуемого направления.

Для matched pair:

```text
a_sensitive = u^T h_sensitive
a_control   = u^T h_control

h_sensitive' =
    h_sensitive + (a_control - a_sensitive) u

h_control' =
    h_control + (a_sensitive - a_control) u
```

### Necessity

Уменьшается ли отказ, когда sensitive run получает control coefficient?

```text
N_j = refusal(sensitive)
    - refusal(sensitive <- control)
```

### Sufficiency

Появляется ли отказ/steering, когда control run получает sensitive
coefficient?

```text
S_j = refusal(control <- sensitive)
    - refusal(control)
```

### Cascade

После interchange измерить, изменились ли downstream:

- amplifier component writes;
- refusal readout;
- router choices;
- response policy.

Классификация:

| Наблюдение | Роль |
|---|---|
| высокий probe, низкие N/S | detector или коррелят |
| низкий direct write, высокие N/S и cascade | gate |
| высокий direct write, knockout снижает refusal | amplifier/carrier |
| высокий late DLA, малая semantic change | formatter/readout |

Для верхних candidates direction-specific interchange повторяется полной
component-output заменой на matched pairs.

## 9. Этап D: настоящий weight-edit leverage scan

Даже подтверждённый gate не обязан быть лучшим местом для Heretic. Поэтому
после circuit localization выполняется отдельный scan тем же кодом weight edit:

```text
один слой
один компонент
lambda in {0.5, 1.0, 1.5, 2.0}
raw_exact / norm-aware where required
```

Для каждого candidate:

```text
semantic_benefit
soft_to_comply
refusal_to_comply
new_failures
harmless_damage
tech_sensitive_damage
perplexity_delta
KL_delta
```

Robust edit leverage:

```text
LE_j =
  lower_confidence_bound(semantic_benefit_j)
  - beta * upper_confidence_bound(harmless_damage_j)
  - gamma * upper_confidence_bound(perplexity_delta_j)
```

Лучшим считается не максимальный средний эффект, а максимальная нижняя граница
`LE` на held-out prompts.

## 10. Как из послойной карты получить окно

### 10.1 Не отправлять raw curve сразу в Optuna

Сначала для каждой component curve:

1. bootstrap по prompt IDs и блоками по topic;
2. total-variation denoising;
3. change-point detection;
4. selection probability каждого слоя;
5. объединение соседних стабильных слоёв.

### 10.2 Поиск интервала через cumulative score

Если одиночные эффекты приблизительно аддитивны:

```text
window_score(a,b) =
    sum_{l=a..b} LE_l - width_penalty * (b-a+1)
```

Лучшие интервалы находятся prefix sums/dynamic programming, без слепого
`O(L^2)` generation sweep.

Затем top-K интервалов обязательно применяются целиком, потому что
взаимодействия нарушают аддитивность.

### 10.3 Второе окно

Разрешать только если:

```text
synergy(A,B) =
  effect(A+B) - effect(A) - effect(B)
```

устойчива между folds и категориями. Иначе два пика — переобучение маленького
eval.

### 10.4 Форма края после границ

Сначала находятся support boundaries. Только затем сравниваются:

```text
hard window
linear edge
cubic smoothstep edge
```

Если одновременно искать center, radius, edge и amplitude, оптимизатор
компенсирует неправильные границы формой кривой.

## 11. MoE: разделить route и expert content

Ассоциация expert ID с отказом может быть всего лишь тематической. Для
Qwen3.6 нужна факторная проверка:

| Routing | Expert outputs | Вопрос |
|---|---|---|
| natural | natural | baseline |
| patched/fixed | natural | причинен ли router |
| natural | patched | несут ли experts отказной выход |
| patched/fixed | patched | взаимодействуют ли route и content |

### 11.1 Coalition, а не только singleton expert

Свежая expert-aware causal tracing работа получила:

- устойчивый singleton expert в одной Qwen MoE;
- отсутствие singleton localization в Mixtral;
- восстановление эффекта только коалицией активных experts.

Поэтому проверять:

```text
top-1 expert
all selected experts
union(selected_sensitive, selected_control)
shared expert
routed + shared coalition
```

### 11.2 Что логировать

```text
route JSD sensitive/control
top-k overlap
router entropy
expert coalition effect
effect with routing held fixed
effect with expert outputs held fixed
bootstrap selection stability
```

Per-expert gamma имеет смысл только после того, как экспертный вклад отделён от
маршрутизации и темы.

## 12. Архитектурные развилки

### Gemma-2/3/4

- record post-RMSNorm writer output;
- actual edit использует `normalize(norm_scale * v)`;
- full/sliding attention анализируются отдельно;
- Gemma-4 PLE — отдельный writer.

### Qwen2.5/Qwen3

- лучший чистый контроль;
- прямые output writes;
- сначала проверить саму локализацию без gates/MoE.

### Qwen3.5

- separate full/linear maps;
- `in_proj_z` и packed attention gate — route candidates;
- `out_proj/o_proj` — edit-leverage candidates;
- prompt boundary сравнить с response-prefix positions.

### Qwen3.6 MoE

- отдельные router, routed coalition, shared expert, shared gate;
- MTP сначала должен корректно сохраняться в export.

## 13. Что добавить в Heretic

Не в основной Optuna path, а отдельным research entrypoint:

```text
heretic-localize
```

### 13.1 Recorder

```text
ComponentRecorder:
  residual_pre
  residual_post
  attention_postnorm_write
  mlp_postnorm_write
  extra_branch_write
  gate_states
  router_states
  expert_outputs
```

### 13.2 Intervention API

```text
ActivationIntervention:
  component
  layer
  token_selector
  mode = coefficient_swap | mean_ablate | scale

WeightIntervention:
  target
  layer_mask
  direction
  lambda
  projection_mode
```

### 13.3 Numeric artifacts

Без текстов:

```text
run_manifest.json
prompt_metrics.jsonl
layer_component_metrics.jsonl
bootstrap_selection.json
candidate_windows.json
```

На prompt level:

```text
prompt_id
pair_id
topic
base_verdict
intervention_verdict
continuous_proxy_delta
KL_delta
```

## 14. Минимальная версия без SAE и transcoders

SAE/CLT могут дать более тонкие features, но для первого улучшения не нужны.

Практический первый пакет:

1. Qwen2.5-3B как чистый контроль.
2. 40–80 matched sensitive/control pairs.
3. Фактические response verdicts.
4. Отдельные detection/refusal directions.
5. Hooks на residual, attention output, MLP output.
6. Последний content token и prompt boundary.
7. Signed `+eps/-eps` single-layer scan.
8. Necessity/sufficiency coefficient interchange для top-8 candidates.
9. Exact weight-edit scan `lambda={1,2}` для top-4.
10. TV/change-point segmentation и проверка top-3 windows.
11. Leave-one-topic-out повтор.
12. Только после этого smooth-window Optuna.

После контрольной модели:

- Gemma-2 — проверить norm-aware и sliding/full;
- Gemma-4 — добавить PLE;
- Qwen3.5 — добавить gates;
- Qwen3.6 — router/expert coalitions.

## 15. Критерий подтверждённой проблемной зоны

Зона принимается, если одновременно:

1. direction и знак эффекта устойчивы между folds;
2. есть перенос на held-out topic;
3. causal necessity либо sufficiency выше permutation null;
4. actual weight edit улучшает semantic verdicts;
5. нижняя граница edit leverage положительна;
6. harmless/tech-sensitive damage ограничен;
7. найденные границы воспроизводятся bootstrap stability selection;
8. smooth profile не хуже hard-window baseline при сопоставимом бюджете.

Если probe peak есть, а weight-edit leverage отсутствует, это не неудача
поиска: найден detector, carrier или readout, но не хирургическая точка для
Heretic.

## 16. Источники и что именно из них использовано

- Refusal direction baseline:
  https://arxiv.org/abs/2406.11717
- Activation patching/AtP* и false-negative-aware ranking:
  https://arxiv.org/abs/2403.00745
- Causal localization не обязательно выбирает лучший edit layer:
  https://arxiv.org/abs/2301.04213
- Разделение harmfulness и refusal:
  https://arxiv.org/abs/2507.11878
- Несколько независимых refusal directions/concept cones:
  https://arxiv.org/abs/2502.17420
- SAE refusal features и причинное scaling:
  https://arxiv.org/abs/2505.23556
- Boundary-critical sampling и influence вместо activation magnitude:
  https://arxiv.org/abs/2604.01604
- Detection → routing → output, gate/interchange/cascade:
  https://arxiv.org/abs/2604.04385
- Почему held-out category важнее простой probe accuracy:
  https://arxiv.org/abs/2603.18280
- Layer-selective distributional intervention:
  https://arxiv.org/abs/2603.04355
- Expert-aware tracing и необходимость coalition checks:
  https://arxiv.org/abs/2606.03780

