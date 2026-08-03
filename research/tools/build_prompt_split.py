# -*- coding: utf-8 -*-
r"""Собрать наборы вредных промптов: на построение направления и на проверку.

Зачем это отдельно и почему детерминированно. Направление отказа считается по
одним промптам, а отказы меряются по другим. Если они пересекутся, числа станут
красивыми и бессмысленными: мы будем проверять модель на том же, подо что её и
правили. Разбиение задаётся здесь раз и сохраняется файлами, чтобы поиск и
замеры пользовались ровно одними и теми же наборами.

Что откуда:

  AdvBench (walledai/AdvBench)          520   классический набор
      mlabonne/harmful_behaviors - это он же, разрезанный на 416 и 104.
      Наши прежние замеры отказов сделаны ровно на тех 104, поэтому они
      остаются в проверочном наборе целиком - иначе новые числа не с чем
      будет сравнить.

  ForbiddenQuestions (walledai/...)     390   13 категорий по 30
      Совсем другие промпты, пересечение с AdvBench нулевое. И другая форма:
      у AdvBench почти всё "напиши руководство по X", здесь вопросы от первого
      лица. Направление, снятое на обеих формах, меньше цепляется за
      формулировку.

  harmless_alpaca (mlabonne/...)      25058   безобидные

Безобидные нужны не для полноты картины, а по устройству метода: направление
отказа - это разность средних между вредными и безобидными остаточными
состояниями. Одной половины мало, и число их должно совпадать, иначе разность
считается по выборкам разного размера и шум с одной стороны перевешивает.

Разбиение по категориям, а не вперемешку: тогда отказы можно мерить в разрезе
тем и видеть, равномерно ли снята цензура.

Запуск:
  python build_prompt_split.py [папка вывода]
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "prompt_split"
SEED = 20260729          # фиксируем, чтобы разбиение воспроизводилось
EVAL_PER_CATEGORY = 10   # 13 категорий -> 130 проверочных из ForbiddenQuestions


def main():
    from datasets import load_dataset

    adv = load_dataset("walledai/AdvBench", split="train")
    adv_col = "prompt" if "prompt" in adv.column_names else adv.column_names[0]
    adv_all = [r[adv_col] for r in adv]

    # Те самые 416/104: наши прежние замеры сделаны на тестовой половине,
    # и её состав менять нельзя, иначе сравнивать будет не с чем.
    mb_train = set(load_dataset("mlabonne/harmful_behaviors", split="train")["text"])
    mb_test = set(load_dataset("mlabonne/harmful_behaviors", split="test")["text"])

    fq = load_dataset("walledai/ForbiddenQuestions", split="train")
    by_cat = defaultdict(list)
    for r in fq:
        by_cat[r["category"]].append(r["prompt"])

    rng = random.Random(SEED)
    fq_eval, fq_train = [], []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat])
        rng.shuffle(items)
        fq_eval += [(cat, p) for p in items[:EVAL_PER_CATEGORY]]
        fq_train += [(cat, p) for p in items[EVAL_PER_CATEGORY:]]

    # На направление: обучающая половина AdvBench плюс оставшиеся вопросы.
    train = [("AdvBench", p) for p in adv_all if p in mb_train]
    train += fq_train
    # На проверку: прежние 104 целиком плюс отобранные по категориям.
    evalset = [("AdvBench", p) for p in adv_all if p in mb_test]
    evalset += fq_eval

    # Безобидных берём ровно столько же, сколько вредных: разность средних
    # считается по двум выборкам, и разный размер даёт перекос в пользу
    # большей. Их 25 тысяч, так что выбор ничем не ограничен.
    harmless = load_dataset("mlabonne/harmless_alpaca", split="train")["text"]
    rng2 = random.Random(SEED + 1)
    pool = sorted(set(harmless))
    rng2.shuffle(pool)
    good_train = pool[:len(train)]
    good_eval = pool[len(train):len(train) + len(evalset)]

    # Пересечение обязано быть пустым - это единственное, ради чего всё писалось.
    inter = {p for _, p in train} & {p for _, p in evalset}
    assert not inter, f"вредные наборы пересекаются на {len(inter)} промптах"
    assert not set(good_train) & set(good_eval), "безобидные наборы пересекаются"

    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train", train), ("eval", evalset)):
        with open(os.path.join(OUT, f"harmful_{name}.jsonl"), "w",
                  encoding="utf-8") as f:
            for cat, p in rows:
                f.write(json.dumps({"category": cat, "text": p},
                                   ensure_ascii=False) + "\n")

    print(f"  на построение направления: {len(train)}")
    for c, n in Counter(c for c, _ in train).most_common():
        print(f"      {n:>4}  {c}")
    print(f"\n  на проверку: {len(evalset)}")
    for c, n in Counter(c for c, _ in evalset).most_common():
        print(f"      {n:>4}  {c}")
    for name, rows in (("train", good_train), ("eval", good_eval)):
        with open(os.path.join(OUT, f"harmless_{name}.jsonl"), "w",
                  encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps({"category": "harmless", "text": p},
                                   ensure_ascii=False) + "\n")

    print(f"\n  безобидные: {len(good_train)} на направление, "
          f"{len(good_eval)} отложено")
    print(f"  пересечений нет, проверено")
    print(f"  записано в {OUT}/: harmful_train, harmful_eval, "
          f"harmless_train, harmless_eval")


if __name__ == "__main__":
    main()
