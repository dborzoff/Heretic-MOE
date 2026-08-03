# -*- coding: utf-8 -*-
r"""Собрать шардированную модель в один файл safetensors.

Наши энкодеры для ComfyUI живут одним файлом, и держать рядом такую же
однофайловую сборку удобнее, чем два десятка шардов с индексом.

Память: держим всё сразу, 67 ГиБ против 251 на машине - помещается.

Запуск:
  python merge_to_single.py <папка HF> <файл.safetensors>
"""
import glob
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    src, dst = sys.argv[1], sys.argv[2]
    out = {}
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    print(f"  шардов: {len(shards)}")
    for i, shard in enumerate(shards, 1):
        f = safe_open(shard, framework="pt")
        for k in f.keys():
            out[k] = f.get_tensor(k)
        print(f"    прочитан {i}/{len(shards)}, тензоров {len(out)}", flush=True)
    print(f"  пишу один файл, {len(out)} тензоров...")
    save_file(out, dst, metadata={"format": "pt"})
    print(f"  {dst}: {os.path.getsize(dst) / 1024**3:.1f} ГиБ")


if __name__ == "__main__":
    main()
