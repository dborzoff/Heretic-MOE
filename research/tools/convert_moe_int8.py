# -*- coding: utf-8 -*-
r"""int8_convrot для MoE со сросшимися экспертами.

Обычный конвертер пропускает трёхмерные тензоры - у него стоит
`if t.ndim != 2: return False`. А у Qwen3.6 в них лежит 92% весов: 256
экспертов упакованы в один тензор [эксперты, выход, вход] на каждый слой.

Расширение выходит простым. Такой тензор разворачивается в
[эксперты x выход, вход], и весь двумерный код применяется без изменений:
поворот Адамара идёт по входной оси, масштаб считается построчно - значит у
каждого эксперта свои масштабы, как и должно быть. Обратная сборка формы -
одна операция.

Поворот берётся из convrot.py, дословной копии функций ComfyUI: переписывать
по памяти нельзя, разойдётся хоть в знаке - и файл станет несовместимым.

Запуск:
  python convert_moe_int8.py <папка HF> <папка результата> [--groupsize 256]
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convrot import _build_hadamard, _rotate_weight   # noqa: E402

SKIP = ("norm", "embed", "lm_head", "bias", "gate.weight", "router")
MAX_SHARD = 4 * 1024 ** 3


def eligible(name, t, groupsize):
    """Можно ли квантовать. Трёхмерные пропускаем не по размерности, а по сути."""
    low = name.lower()
    if any(tok in low for tok in SKIP):
        return False, "имя в списке пропуска"
    if t.ndim == 2:
        out_f, in_f = t.shape
    elif t.ndim == 3:
        _, out_f, in_f = t.shape
    else:
        return False, f"ndim={t.ndim}"
    if in_f % groupsize:
        return False, f"вход {in_f} не кратен {groupsize}"
    if min(out_f, in_f) < groupsize:
        return False, "слой мал"
    return True, ""


def build_ratios(lo=0.80):
    """Сетка подрезания: плотно у единицы, дальше разреженнее. Как в основном
    конвертере - у текст-энкодеров хвосты тяжёлые, и без длинного хвоста часть
    строк упирается в край перебора."""
    out = [round(1.0 - i * 0.001, 4) for i in range(61)]
    out += [round(0.938 - i * 0.002, 4) for i in range(20)]
    out += [round(0.895 - i * 0.005, 4) for i in range(20)]
    return tuple(r for r in out if r >= lo - 1e-9)


RATIOS = build_ratios()


def quantize(w, groupsize, device, clip_margin=0.05):
    """Поворот Адамара и подбор масштаба по минимуму ошибки.

    Простой absmax - это официальный рецепт. Наш добавляет перебор подрезания:
    редкие большие веса округляются грубее, зато основная масса ложится на сетку
    точнее. Отсюда "lean" в названии сборок.

    Подрезаем не везде, где минимум MSE, а только где выигрыш заметнее clip_margin.
    Эталонные сборки оставляют без подрезания больше строк, чем даёт чистый
    минимум квадрата, и на выходе это оказывается лучше - похоже, у выбросов есть
    роль, которую квадратичная ошибка не видит.
    """
    w = w.to(device=device, dtype=torch.float32)
    h = _build_hadamard(groupsize, device=w.device, dtype=w.dtype)
    rot = _rotate_weight(w, h, groupsize)

    absmax = rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    best_scale = absmax / 127.0
    best_err = torch.full_like(absmax, float("inf"))

    for r in RATIOS:
        scale = absmax * r / 127.0
        q = torch.clamp(torch.round(rot / scale), -127, 127)
        err = ((q * scale - rot) ** 2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
        del q, err

    if clip_margin > 0.0:
        s0 = absmax / 127.0
        q0 = torch.clamp(torch.round(rot / s0), -127, 127)
        e0 = ((q0 * s0 - rot) ** 2).sum(dim=1, keepdim=True)
        keep = best_err >= e0 * (1.0 - clip_margin)
        best_scale = torch.where(keep, s0, best_scale)
        del q0, e0

    q = torch.clamp(torch.round(rot / best_scale), -127, 127).to(torch.int8)
    return q, best_scale.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", help="путь к файлу .safetensors")
    ap.add_argument("--groupsize", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip-margin", type=float, default=0.05,
                    help="порог выигрыша, ниже которого не подрезаем")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)) or ".", exist_ok=True)

    shards = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
    conf = json.dumps({"format": "int8_tensorwise", "convrot": True,
                       "convrot_groupsize": a.groupsize}).encode("utf-8")

    # Копим всё в памяти и пишем одним файлом, как наши энкодеры для ComfyUI.
    # Результат около 34 ГиБ, оперативной памяти на машине втрое больше.
    out = {}
    n_q = n_keep = 0
    reasons = {}

    for shard in shards:
        f = safe_open(shard, framework="pt")
        for k in f.keys():
            t = f.get_tensor(k)
            ok, why = eligible(k, t, a.groupsize)
            if not ok:
                out[k] = t
                n_keep += 1
                reasons[why.split()[0]] = reasons.get(why.split()[0], 0) + 1
            else:
                shape = t.shape
                flat = t.reshape(-1, shape[-1])          # [E*out, in] или [out, in]
                q, scale = quantize(flat, a.groupsize, a.device, a.clip_margin)
                q = q.reshape(shape).cpu()
                scale = scale.reshape(*shape[:-1], 1).cpu()
                base = k[:-len(".weight")] if k.endswith(".weight") else k
                out[k] = q
                out[f"{base}.weight_scale"] = scale
                out[f"{base}.comfy_quant"] = torch.tensor(list(conf), dtype=torch.uint8)
                n_q += 1
            if (n_q + n_keep) % 200 == 0:
                print(f"  обработано {n_q + n_keep}, квантовано {n_q}", flush=True)

    print(f"  пишу один файл, {len(out)} тензоров...")
    save_file(out, a.dst, metadata={"format": "pt"})
    size = os.path.getsize(a.dst)

    print(f"\nитог: квантовано {n_q}, оставлено {n_keep}")
    print(f"  причины пропуска: {reasons}")
    print(f"  {a.dst}: {size / 1024**3:.1f} ГиБ")


if __name__ == "__main__":
    main()
