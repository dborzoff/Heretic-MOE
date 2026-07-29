# -*- coding: utf-8 -*-
r"""Развести компоненты, которые heretic правит одной ручкой, по своим.

Зачем. Heretic заводит один набор из четырёх чисел (пик, его место, минимум,
ширина) на КЛЮЧ КОМПОНЕНТА. А в `get_layer_modules` под одним ключом лежат
разные по устройству вещи:

  attn.o_proj    = self_attn.o_proj (10 слоёв) + linear_attn.out_proj (30 слоёв)
  mlp.down_proj  = маршрутизируемые эксперты (сплав) + общий эксперт

Разбор весов Qwen3.6-35B-A3B показывает, насколько это разные вещи:

  эксперты (сплав)     32.21 млрд   89.6%   срабатывают 8 из 256
  линейное внимание     1.01 млрд    2.8%   30 слоёв
  полное внимание       0.27 млрд    0.8%   10 слоёв
  общий эксперт         0.13 млрд    0.4%   срабатывает ВСЕГДА

Линейное внимание весит вчетверо больше полного, а правится тем же весом.
Общий эксперт в 250 раз легче маршрутизируемых, но работает на каждом токене -
и цена его правки совсем другая. Одна ручка на всё означает, что поиск не может
оставить полезную часть правки и убрать дорогую: они двигаются вместе.

Наши замеры показывают, что цена есть и она немаленькая: агрессивная сборка
стоит +18% перплексии против +3.4% у сбалансированной. Если хотя бы часть этой
цены платится за правку не тех тензоров, разделение её вернёт.

Что меняется:
  1. ключи компонентов разводятся на четыре;
  2. нижняя граница веса перестаёт зависеть от одной строки "mlp.down_proj";
  3. границы расширяются - прежние обрезали лучшие точки (мы это уже видели:
     после расширения рекорд сдвинулся с 0.15 до 0.01 отказов).

Запуск:
  python split-components.patch.py <путь к heretic/src/heretic>
"""
import io
import os
import re
import sys

MODEL_OLD = '''            try_add("attn.o_proj", layer.self_attn.o_proj)'''
MODEL_NEW = '''            # Разведено на свои ключи: полное и линейное внимание - разные
            # операторы, и линейных втрое больше по числу слоёв и вчетверо по
            # весу. Под общим ключом они делили одну кривую по глубине.
            try_add("attn.self.o_proj", layer.self_attn.o_proj)'''

MODEL_OLD2 = '''            try_add("attn.o_proj", layer.linear_attn.out_proj)'''
MODEL_NEW2 = '''            try_add("attn.linear.out_proj", layer.linear_attn.out_proj)'''


def patch_model(path):
    s = io.open(path, encoding="utf-8").read()
    n = 0

    if MODEL_OLD in s:
        s = s.replace(MODEL_OLD, MODEL_NEW, 1); n += 1
    if MODEL_OLD2 in s:
        s = s.replace(MODEL_OLD2, MODEL_NEW2, 1); n += 1

    # Общий эксперт - на свой ключ. Он крошечный по весу, но активен на каждом
    # токене, поэтому его правка стоит дороже за единицу веса, чем правка
    # редких маршрутизируемых.
    s = s.replace(
        'try_add("mlp.down_proj", layer.mlp.shared_expert.down_proj)',
        'try_add("mlp.shared.down_proj", layer.mlp.shared_expert.down_proj)')
    if 'mlp.shared.down_proj' in s:
        n += 1

    # Маршрутизируемые эксперты: и обычный список модулей, и наш сплав.
    for old in ('try_add("mlp.down_proj", expert.down_proj)',
                'try_add("mlp.down_proj", expert.w2)',
                'try_add("mlp.down_proj", layer.mlp.down_proj)'):
        new = old.replace('"mlp.down_proj"', '"mlp.experts.down_proj"')
        if old in s:
            s = s.replace(old, new); n += 1

    # Наш патч для сплава экспертов кладёт их под тем же ключом - переименуем и там.
    s = re.sub(r'(_fused_experts_cache\[[^\]]+\]\s*=\s*)"mlp\.down_proj"',
               r'\1"mlp.experts.down_proj"', s)
    s = s.replace('"mlp.down_proj", fused', '"mlp.experts.down_proj", fused')

    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    return n


MAIN_OLD = '''            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8'''
MAIN_NEW = '''            # Нижняя граница по смыслу компонента, а не по одной строке имени.
            # Отрицательная нижняя граница с обрезкой в ноль даёт поиску
            # возможность полностью выключить правку этого компонента: у
            # непрерывного распределения ноль иначе недостижим.
            #
            # Полному вниманию оставляем прежний порог 0.8 - это классическая
            # цель абляции, и на ней метод заведомо работает. Всему остальному
            # разрешаем выключаться: про линейное внимание и про раздельную
            # правку экспертов мы ничего заранее не знаем, пусть решает поиск.
            max_weight_lower_bound = 0.8 if component == "attn.self.o_proj" else -0.25'''

# Прежние границы обрезали лучшие точки: победители упирались в них по трём
# параметрам из четырёх. После расширения рекорд сдвинулся с 0.15 до 0.01
# отказов, а пик правки встал на 15-й слой из 40 - вдвое раньше, чем разрешал
# исходный нижний предел 0.6 от глубины.
BOUNDS = [
    ('''                    max_weight_lower_bound,
                    1.5,''',
     '''                    max_weight_lower_bound,
                    2.5,'''),
    ('''                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,''',
     '''                f"{component}.max_weight_position",
                0.0,
                1.0 * last_layer_index,'''),
    ('''                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),''',
     '''                f"{component}.min_weight_distance",
                1.0,
                max(1.5 * last_layer_index, 1.0),'''),
]


def patch_main(path):
    s = io.open(path, encoding="utf-8").read()
    n = 0
    if MAIN_OLD in s:
        s = s.replace(MAIN_OLD, MAIN_NEW, 1); n += 1
    for old, new in BOUNDS:
        if old in s:
            s = s.replace(old, new, 1); n += 1
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    return n


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/heretic/src/heretic"
    m = patch_model(os.path.join(root, "model.py"))
    k = patch_main(os.path.join(root, "main.py"))
    print(f"  model.py: правок {m}")
    print(f"  main.py:  правок {k}")

    # Проверяем, что после правки файлы вообще разбираются.
    import ast
    for f in ("model.py", "main.py"):
        ast.parse(io.open(os.path.join(root, f), encoding="utf-8").read())
    print("  синтаксис обоих файлов в порядке")

    s = io.open(os.path.join(root, "model.py"), encoding="utf-8").read()
    keys = sorted(set(re.findall(r'try_add\("([^"]+)"', s)))
    print(f"  компоненты теперь: {keys}")


if __name__ == "__main__":
    main()
