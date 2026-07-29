# -*- coding: utf-8 -*-
r"""Насколько int8_convrot отличается от исходных весов.

Запустить наш формат нечем - загрузчик живёт в ComfyUI, а он такие модели не
гоняет. Но качество можно измерить прямо: развернуть квантованное обратно и
сравнить с оригиналом. Поворот Адамара ортогонален и нормирован, поэтому
повторное применение восстанавливает исходное пространство точно.

Считаем относительную ошибку по Фробениусу для каждого тензора и сводим по
семействам. Заодно тут же считаем, что дал бы простой absmax на тех же весах -
это и есть проверка, окупается ли наш перебор подрезания.

Запуск:
  python check_int8.py <bf16 папка или файл> <int8 файл>
"""
import glob
import os
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convrot import _build_hadamard, _rotate_weight   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def opener(path):
    """Папка с шардами или один файл - читаем одинаково."""
    files = ([path] if path.endswith(".safetensors")
             else sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    m = {}
    for p in files:
        f = safe_open(p, framework="pt")
        for k in f.keys():
            m[k] = f
    return m


def group(k):
    if "experts" in k:
        return "эксперты"
    if "shared_expert" in k:
        return "общий эксперт"
    if "o_proj" in k or "out_proj" in k:
        return "внимание"
    if "visual" in k:
        return "вижн"
    return "прочее"


def main():
    ref = opener(sys.argv[1])
    q8 = opener(sys.argv[2])
    gs = 256

    quantized = [k[: -len(".weight_scale")] for k in q8 if k.endswith(".weight_scale")]
    print(f"  квантованных слоёв: {len(quantized)}")

    stat = defaultdict(lambda: [0, 0.0, 0.0])   # число, наша ошибка, absmax
    for i, base in enumerate(sorted(quantized), 1):
        wk = base + ".weight" if base + ".weight" in ref else base
        if wk not in ref:
            continue
        w = ref[wk].get_tensor(wk).to(DEV, torch.float32)
        q = q8[wk].get_tensor(wk).to(DEV, torch.float32)
        s = q8[base + ".weight_scale"].get_tensor(base + ".weight_scale").to(DEV, torch.float32)

        shape = w.shape
        flat = w.reshape(-1, shape[-1])
        h = _build_hadamard(gs, device=DEV, dtype=torch.float32)

        # восстановление: развернуть масштаб и повернуть обратно
        rec = _rotate_weight((q.reshape(-1, shape[-1]) * s.reshape(-1, 1)), h, gs)
        err = (rec - flat).norm() / flat.norm().clamp(min=1e-12)

        # что дал бы простой absmax на тех же весах
        rot = _rotate_weight(flat, h, gs)
        am = rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        q0 = torch.clamp(torch.round(rot / am), -127, 127)
        rec0 = _rotate_weight(q0 * am, h, gs)
        err0 = (rec0 - flat).norm() / flat.norm().clamp(min=1e-12)

        g = stat[group(base)]
        g[0] += 1
        g[1] += err.item()
        g[2] += err0.item()
        if i % 60 == 0:
            print(f"    {i}/{len(quantized)}", flush=True)
        del w, q, s, rec, rot, q0, rec0

    print(f"\n  {'группа':16s} {'слоёв':>7s} {'наша ошибка':>13s} {'absmax':>10s} {'выигрыш':>9s}")
    tn = te = t0 = 0
    for name, (n, e, e0) in sorted(stat.items()):
        tn += n; te += e; t0 += e0
        gain = (1 - (e / n) / (e0 / n)) * 100 if e0 else 0
        print(f"  {name:16s} {n:>7d} {e/n*100:>12.4f}% {e0/n*100:>9.4f}% {gain:>8.1f}%")
    if tn:
        gain = (1 - te / t0) * 100
        print(f"  {'итого':16s} {tn:>7d} {te/tn*100:>12.4f}% {t0/tn*100:>9.4f}% {gain:>8.1f}%")


if __name__ == "__main__":
    main()
