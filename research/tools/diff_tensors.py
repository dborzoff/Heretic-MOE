# -*- coding: utf-8 -*-
r"""Что на самом деле изменилось в весах: сборка против исходной модели.

Сравниваем побайтно, а не по именам. "Не изменился" здесь значит "присутствует и
совпадает бит в бит", а не "унаследован" - иначе потерянный по дороге тензор
выглядел бы как нетронутый.

Заодно проверяем, что суммы сходятся: всего = не изменилось + изменилось. Если
не сходится, значит тензоры пропали или появились, и таблицу публиковать нельзя.

Запуск:
  python diff_tensors.py <папка сборки> <папка исходной> [--md]
"""
import argparse
import hashlib
import json
import os

# Порядок важен: тензор попадает в первую подошедшую группу. Общие эксперты
# идут раньше маршрутизируемых, потому что имя shared_expert тоже содержит
# "expert" - при обратном порядке они бы слились в одну строку.
GROUPS = [
    ("Vision tower", lambda n: n.startswith("model.visual.")
                               or ".visual." in n),
    ("MTP", lambda n: ".mtp" in n.lower() or "mtp." in n.lower()),
    ("Shared expert", lambda n: "shared_expert" in n),
    ("Routed experts (fused)", lambda n: ".mlp.experts." in n),
    # Модель гибридная: из 40 слоёв только 10 несут полное внимание
    # (self_attn.q/k/v/o_proj), остальные 30 - линейное (linear_attn с A_log,
    # conv1d, dt_bias и in_proj_*). Выход у них называется по-разному, но роль
    # одна, и правка ложится на оба - поэтому в одну строку.
    ("Attention output", lambda n: "self_attn.o_proj" in n
                                   or "linear_attn.out_proj" in n),
]


def shards(path):
    """Имя тензора -> (файл, смещение, длина) по индексу safetensors."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx, encoding="utf-8"))["weight_map"]
        return m
    one = "model.safetensors"
    if os.path.exists(os.path.join(path, one)):
        from safetensors import safe_open
        with safe_open(os.path.join(path, one), framework="pt") as f:
            return {k: one for k in f.keys()}
    raise SystemExit(f"  не нашёл safetensors в {path}")


def digest(path, mapping, name):
    """Хеш сырых байт тензора - грузить в память целиком незачем.

    Байты берём через torch, а не через numpy: веса лежат в bfloat16, а такого
    типа в numpy нет вовсе, и попытка привести падает на ScalarType BFloat16.
    Смотрим на тот же кусок памяти как на байты - тип при этом неважен, а нам
    и нужно сравнение побайтное."""
    import torch
    from safetensors import safe_open
    with safe_open(os.path.join(path, mapping[name]), framework="pt") as f:
        t = f.get_slice(name)
        h = hashlib.blake2b(digest_size=16)
        # По первому измерению кусками: у экспертов это [256, 2048, 512],
        # целиком такое поднимать в память дорого и незачем.
        shape = t.get_shape()
        if not shape:            # скаляр: резать нечего, берём целиком
            h.update(f.get_tensor(name).contiguous()
                     .view(torch.uint8).numpy().tobytes())
            return h.digest()
        n = shape[0]
        step = max(1, min(n, 32))
        for lo in range(0, n, step):
            chunk = t[lo:lo + step].contiguous()
            h.update(chunk.view(torch.uint8).numpy().tobytes())
    return h.digest()


def group_of(name):
    for label, test in GROUPS:
        if test(name):
            return label
    return "Everything else"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("base")
    ap.add_argument("--md", action="store_true", help="готовая таблица markdown")
    a = ap.parse_args()

    mb, mo = shards(a.build), shards(a.base)
    order = [g for g, _ in GROUPS] + ["Everything else"]
    total = {g: 0 for g in order}
    same = {g: 0 for g in order}
    missing = []

    for name in sorted(mo):
        g = group_of(name)
        total[g] += 1
        if name not in mb:
            missing.append(name)
            continue
        if digest(a.base, mo, name) == digest(a.build, mb, name):
            same[g] += 1

    extra = sorted(set(mb) - set(mo))
    rows = ["| Group | In base model | Unchanged | Modified |", "|---|---|---|---|"]
    # Порядок строк как в карточке: сперва то, ради чего всё затевалось.
    for g in ["Routed experts (fused)", "Shared expert", "Attention output",
              "MTP", "Vision tower", "Everything else"]:
        t, s = total[g], same[g]
        if not t:
            continue
        mod = t - s
        cell = f"**{mod}**" if g == "Routed experts (fused)" else str(mod)
        rows.append(f"| {g} | {t} | {s} | {cell} |")
    T, S = sum(total.values()), sum(same.values())
    rows.append(f"| **Total** | **{T}** | **{S}** | **{T - S}** |")

    print("\n".join(rows) if a.md else "\n".join("  " + r for r in rows))
    if missing:
        print(f"\n  ПОТЕРЯНО {len(missing)} тензоров, таблицу публиковать нельзя:")
        for n in missing[:10]:
            print(f"    {n}")
    if extra:
        print(f"\n  ЛИШНИХ в сборке: {len(extra)} — {extra[:5]}")
    if not missing and not extra:
        print(f"\n  сходится: {T} тензоров, из них правлено {T - S}")


if __name__ == "__main__":
    main()
