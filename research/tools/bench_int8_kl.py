# -*- coding: utf-8 -*-
r"""Во что обходится наш int8 на выходе модели, а не на весах.

Ошибка по Фробениусу показывает, насколько разъехались веса, но не говорит,
разъехались ли ответы: искажение могло лечь на неважные направления. Поэтому
меряем то же, что heretic меряет для расцензуривания - расхождение Кульбака
между исходной моделью и квантованной на безобидных промптах.

Тогда число становится сравнимым: у нашей правки KL 0.0126, и видно, добавляет
int8 к этому заметно или теряется в шуме.

Запускать наш формат не нужно - восстанавливаем веса обратно (поворот Адамара
сам себе обратный) и подменяем их в обычной модели.

Запуск:
  python bench_int8_kl.py <bf16 папка> <int8 файл> [--prompts 64]
"""
import argparse
import glob
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convrot import _build_hadamard, _rotate_weight   # noqa: E402


def open_all(path):
    files = ([path] if path.endswith(".safetensors")
             else sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    m = {}
    for p in files:
        f = safe_open(p, framework="pt")
        for k in f.keys():
            m[k] = f
    return m


@torch.no_grad()
def run(model, tok, prompts, device):
    """Возвращает эмбеддинги промпта и распределение следующего токена.

    Эмбеддинги - то же, что меряет наш bench_te для энкодеров: последний
    скрытый слой на всю последовательность. Метрика там ||y-x||/||x||, и здесь
    считаем её же, чтобы числа были в одной шкале с прежними замерами.
    """
    embs, logs = [], []
    for p in prompts:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True,
                                      return_tensors="pt")
        # В новых transformers apply_chat_template отдаёт BatchEncoding, а не
        # тензор, и он уходил в модель как есть - отсюда падение
        # "embedding(): indices must be Tensor, not BatchEncoding".
        # Берём сами идентификаторы, оставаясь совместимыми со старым поведением.
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(device)
        o = model(ids, output_hidden_states=True)
        embs.append(o.hidden_states[-1][0].float().cpu())
        logs.append(torch.log_softmax(o.logits[0, -1].float(), dim=-1).cpu())
    return embs, torch.stack(logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("int8")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    ds = load_dataset("mlabonne/harmless_alpaca", split=f"test[:{a.prompts}]")
    prompts = list(ds["text"])
    print(f"  промптов: {len(prompts)}")

    tok = AutoTokenizer.from_pretrained(a.bf16)
    model = AutoModelForImageTextToText.from_pretrained(
        a.bf16, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device
    print("  модель загружена, считаю эталон")
    base_emb, base_log = run(model, tok, prompts, dev)

    # подменяем веса восстановленными из int8
    q8 = open_all(a.int8)
    names = {k[: -len(".weight_scale")] for k in q8 if k.endswith(".weight_scale")}
    print(f"  подменяю {len(names)} слоёв")
    params = dict(model.named_parameters())
    h = _build_hadamard(a.groupsize, device=dev, dtype=torch.float32)
    done = 0
    for base_name in sorted(names):
        pname = base_name + ".weight"
        if pname not in params:
            pname = base_name
            if pname not in params:
                continue
        p = params[pname]
        q = q8[pname if pname in q8 else base_name].get_tensor(
            pname if pname in q8 else base_name).to(dev, torch.float32)
        s = q8[base_name + ".weight_scale"].get_tensor(
            base_name + ".weight_scale").to(dev, torch.float32)
        shape = q.shape
        rec = _rotate_weight((q.reshape(-1, shape[-1]) * s.reshape(-1, 1)), h,
                             a.groupsize).reshape(shape)
        p.data.copy_(rec.to(p.dtype))
        done += 1
        del q, s, rec
        if done % 60 == 0:
            print(f"    {done}/{len(names)}", flush=True)
    print(f"  подменено {done}, считаю квантованную")
    q_emb, q_log = run(model, tok, prompts, dev)

    # Метрика нашего bench_te: относительное отклонение эмбеддингов.
    rels = torch.tensor([((y - x).norm() / x.norm().clamp(min=1e-12)).item()
                         for x, y in zip(base_emb, q_emb)])
    kl = torch.nn.functional.kl_div(q_log, base_log, log_target=True,
                                    reduction="batchmean").item()
    top1 = (base_log.argmax(-1) == q_log.argmax(-1)).float().mean().item()

    print(f"\n  отклонение эмбеддингов: среднее {rels.mean()*100:.4f}%, "
          f"медиана {rels.median()*100:.4f}%, разброс {rels.std()*100:.4f}%")
    print(f"  KL по следующему токену: {kl:.6f}")
    print(f"  первый токен совпал: {top1*100:.1f}%")
    print("  для сравнения, правка heretic: KL 0.0126 у max, 0.0021 у balanced")


if __name__ == "__main__":
    main()
