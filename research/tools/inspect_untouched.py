# -*- coding: utf-8 -*-
r"""Что в модели есть, но абляция туда не заглядывает.

Heretic правит выходные проекции: o_proj/out_proj у внимания и down_proj у MLP.
Всё остальное он не трогает. Смотрим, нет ли там рычагов подешевле - особенно
маршрутизатора: он решает, КАКИЕ эксперты сработают, и весит ничтожно мало по
сравнению с самими экспертами.
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

def shape(k):
    with safe_open(idx[k], framework="pt") as f:
        return f.get_slice(k).get_shape()

print("  ── формы того, что heretic НЕ правит (слой 3, где полное внимание)\n")
for k in sorted(idx):
    m = re.match(r"model\.language_model\.layers\.3\.(.+)$", k)
    if m:
        print(f"    {m.group(1):44s} {shape(k)}")

print("\n  ── слой 4 (линейное внимание)\n")
for k in sorted(idx):
    m = re.match(r"model\.language_model\.layers\.4\.(.+)$", k)
    if m:
        print(f"    {m.group(1):44s} {shape(k)}")

print("\n  ── что в слое MTP (heretic его вообще не трогает)\n")
for k in sorted(idx):
    if k.startswith("mtp."):
        print(f"    {k[4:]:44s} {shape(k)}")

print("\n  ── маршрутизатор: насколько он неоднороден по слоям")
print("     (норма строк = насколько охотно слой зовёт каждого эксперта)\n")
for L in (0, 10, 20, 30, 39):
    k = f"model.language_model.layers.{L}.mlp.gate.weight"
    if k not in idx: continue
    with safe_open(idx[k], framework="pt") as f:
        w = f.get_tensor(k).to(torch.float32)
    rn = w.norm(dim=1)
    print(f"    слой {L:2d}: {w.shape}, норма строк "
          f"мин {rn.min():.3f} мед {rn.median():.3f} макс {rn.max():.3f}, "
          f"верх/низ {rn.max()/rn.min():.2f}")

print("\n  ── вентиль общего эксперта: сколько чисел на слой\n")
for L in (0, 20, 39):
    k = f"model.language_model.layers.{L}.mlp.shared_expert_gate.weight"
    if k in idx:
        with safe_open(idx[k], framework="pt") as f:
            w = f.get_tensor(k).to(torch.float32)
        print(f"    слой {L:2d}: форма {list(w.shape)}, норма {w.norm():.4f}")
