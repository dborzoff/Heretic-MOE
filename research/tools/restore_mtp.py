# -*- coding: utf-8 -*-
"""Вернуть в сваренную модель тензоры MTP, взяв их из оригинала.

Зачем. Heretic при экспорте не переносит слои multi-token prediction: в
оригинале 1045 тензоров, в варке 1026. Для transformers это терпимо, а вот
llama.cpp падает - его конвертер выводит число блоков из конфига (40 слоёв
плюс один MTP = 41) и требует blk.40, которого нет:

    missing tensor 'blk.40.attn_norm.weight'

Heretic эти тензоры не меняет вовсе, поэтому копия из оригинала точна побайтно.

Дописываем отдельным шардом, а не переписываем всю модель: MTP весят около
полутора гигабайт против шестидесяти шести у остального.

Запуск:
  python restore_mtp.py <папка оригинала> <папка варки>
"""
import glob
import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def tensors_of(path):
    """Имя тензора -> открытый файл, из которого его читать."""
    out = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        f = safe_open(shard, framework="pt")
        for k in f.keys():
            out[k] = f
    return out


def main():
    orig_dir, built_dir = sys.argv[1], sys.argv[2]
    orig = tensors_of(orig_dir)
    built = tensors_of(built_dir)
    print(f"  оригинал {len(orig)} тензоров, варка {len(built)}")

    missing = sorted(set(orig) - set(built))
    if not missing:
        print("  ничего не потеряно, делать нечего")
        return 0
    print(f"  недостаёт {len(missing)}:")
    for k in missing[:4]:
        print(f"    {k}")
    if len(missing) > 4:
        print(f"    ... ещё {len(missing) - 4}")

    shards = sorted(glob.glob(os.path.join(built_dir, "*.safetensors")))
    total = len(shards) + 1
    print(f"  шардов было {len(shards)}, станет {total}")

    # Переименовываем существующие под новый счёт: имена шардов несут в себе
    # общее число, и рассинхрон с индексом читается как поломка.
    renamed = {}
    for i, old in enumerate(shards, 1):
        new = os.path.join(built_dir, f"model-{i:05d}-of-{total:05d}.safetensors")
        if old != new:
            os.rename(old, new)
        renamed[os.path.basename(old)] = os.path.basename(new)

    # Новый шард только с потерянными тензорами.
    extra = os.path.join(built_dir, f"model-{total:05d}-of-{total:05d}.safetensors")
    data = {k: orig[k].get_tensor(k) for k in missing}
    size = sum(t.numel() * t.element_size() for t in data.values())
    save_file(data, extra, metadata={"format": "pt"})
    print(f"  дописан {os.path.basename(extra)}: {size / 1024**3:.2f} ГиБ")

    # Индекс: переписываем пути и добавляем новые тензоры.
    idx_path = os.path.join(built_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path, encoding="utf-8"))
    wm = {k: renamed.get(v, v) for k, v in idx["weight_map"].items()}
    for k in missing:
        wm[k] = os.path.basename(extra)
    idx["weight_map"] = wm
    idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + size
    json.dump(idx, open(idx_path, "w", encoding="utf-8"), indent=2)
    print(f"  индекс обновлён: {len(wm)} тензоров")

    # Проверка: всё ли из индекса действительно лежит на диске.
    have = tensors_of(built_dir)
    lost = sorted(set(wm) - set(have))
    print(f"  проверка: на диске {len(have)}, потеряно {len(lost)}")
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
