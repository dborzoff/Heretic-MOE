# -*- coding: utf-8 -*-
r"""Разбор исходной Qwen3.6-35B-A3B по весам, а не по коду обвязки.

Что хочется понять перед доработкой heretic:
  1. где у модели масса - туда и должна целиться правка;
  2. как устроен выход каждого типа слоя - через него правка и работает;
  3. насколько слои неоднородны по величине - одна кривая на все или нет.

Считаем нормы Фробениуса по выходным проекциям: именно они пишут в остаточный
поток, и именно из них heretic вычитает направление отказа. Читаем срезами,
целиком в память ничего не поднимаем.
"""
import glob, os, re, sys
from collections import defaultdict

import torch
from safetensors import safe_open

SRC = sys.argv[1] if len(sys.argv) > 1 else "/workspace/Qwen3.6-35B-A3B"

idx = {}
for p in sorted(glob.glob(os.path.join(SRC, "*.safetensors"))):
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            idx[k] = p

def group(n):
    if ".visual." in n:                      return "вижн"
    if n.startswith("mtp."):                 return "MTP"
    if ".mlp.experts." in n:                 return "эксперты (сплав)"
    if "shared_expert" in n:                 return "общий эксперт"
    if ".mlp.gate" in n:                     return "маршрутизатор"
    if ".self_attn." in n:                   return "полное внимание"
    if ".linear_attn." in n:                 return "линейное внимание"
    return "прочее"

mass = defaultdict(int); cnt = defaultdict(int)
for k in idx:
    with safe_open(idx[k], framework="pt") as f:
        sh = f.get_slice(k).get_shape()
    n = 1
    for d in sh: n *= d
    mass[group(k)] += n; cnt[group(k)] += 1

tot = sum(mass.values())
print(f"  где живёт модель: {tot/1e9:.2f} млрд параметров, {len(idx)} тензоров\n")
print(f"  {'группа':22s} {'тензоров':>9s} {'параметров':>14s} {'доля':>7s}")
for g in sorted(mass, key=lambda g: -mass[g]):
    print(f"  {g:22s} {cnt[g]:9d} {mass[g]/1e9:12.2f}Б {100*mass[g]/tot:6.1f}%")

# Нормы выходных проекций по слоям: через них правка попадает в остаточный поток.
print("\n  норма Фробениуса выходных проекций по слоям")
print("  (чем больше, тем сильнее слой пишет в остаточный поток)\n")
pat = {
    "self_attn.o_proj":      "полное",
    "linear_attn.out_proj":  "линейное",
    "shared_expert.down_proj": "общий эксп.",
}
rows = defaultdict(dict)
for k in idx:
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)\.weight$", k)
    if not m: continue
    for suf, label in pat.items():
        if m.group(2).endswith(suf):
            with safe_open(idx[k], framework="pt") as f:
                w = f.get_tensor(k).to(torch.float32)
            rows[int(m.group(1))][label] = float(w.norm())
# Эксперты - трёхмерные, норму берём по всему сплаву и на одного эксперта.
for k in idx:
    m = re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.down_proj$", k)
    if not m: continue
    with safe_open(idx[k], framework="pt") as f:
        w = f.get_tensor(k).to(torch.float32)
    rows[int(m.group(1))]["эксперты(1шт)"] = float(w.norm()) / (w.shape[0] ** 0.5)

hdr = ["полное", "линейное", "общий эксп.", "эксперты(1шт)"]
print("  слой  " + "".join(f"{h:>15s}" for h in hdr))
for L in sorted(rows):
    line = f"  {L:4d}  "
    for h in hdr:
        v = rows[L].get(h)
        line += f"{v:15.1f}" if v is not None else f"{'—':>15s}"
    print(line)
