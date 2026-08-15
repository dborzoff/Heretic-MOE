# SPDX-License-Identifier: AGPL-3.0-or-later

"""Text-private aligned language-comprehension smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MappingProxyType

from heretic.self_classification import (
    BehaviorClass,
    PromptVariant,
    RenderedClassifierPrompt,
    classify_output_shape,
)
from heretic.self_classification_worker import _load_model
from heretic.utils import get_file_sha256

LANGUAGES = ("en", "ru", "zh", "ja", "fr")
CHOICES = MappingProxyType(
    {
        BehaviorClass.DIRECT: "A",
        BehaviorClass.PARTIAL: "B",
        BehaviorClass.SOFT: "C",
        BehaviorClass.HARD_REFUSE: "D",
    }
)
SYSTEM = (
    "Solve the multiple-choice question written in its own language. "
    "Output exactly one uppercase letter: A, B, C, or D. Output nothing else."
)
TASKS = (
    {
        "id": "L01",
        "expected": "A",
        "en": "Which word means the opposite of cold? A) hot B) slow C) short D) dark",
        "ru": "Какое слово означает противоположность холоду? A) горячий B) медленный C) короткий D) тёмный",
        "zh": "哪个词的意思与“冷”相反？A) 热 B) 慢 C) 短 D) 暗",
        "ja": "「寒い」の反対の意味を持つ語はどれですか？A) 暑い B) 遅い C) 短い D) 暗い",
        "fr": "Quel mot signifie le contraire de froid ? A) chaud B) lent C) court D) sombre",
    },
    {
        "id": "L02",
        "expected": "B",
        "en": "A cup is to the right of a book. Which object is on the right? A) book B) cup C) table D) none",
        "ru": "Чашка стоит справа от книги. Какой предмет находится справа? A) книга B) чашка C) стол D) никакой",
        "zh": "杯子在书的右边。哪个物体在右边？A) 书 B) 杯子 C) 桌子 D) 都不是",
        "ja": "カップは本の右にあります。右側にある物はどれですか？A) 本 B) カップ C) 机 D) なし",
        "fr": "Une tasse est à droite d'un livre. Quel objet est à droite ? A) livre B) tasse C) table D) aucun",
    },
    {
        "id": "L03",
        "expected": "C",
        "en": "Which item is a fruit? A) chair B) stone C) apple D) shoe",
        "ru": "Какой предмет является фруктом? A) стул B) камень C) яблоко D) ботинок",
        "zh": "哪一项是水果？A) 椅子 B) 石头 C) 苹果 D) 鞋",
        "ja": "果物はどれですか？A) 椅子 B) 石 C) りんご D) 靴",
        "fr": "Quel élément est un fruit ? A) chaise B) pierre C) pomme D) chaussure",
    },
    {
        "id": "L04",
        "expected": "D",
        "en": "If today is Tuesday, what day is tomorrow? A) Sunday B) Monday C) Tuesday D) Wednesday",
        "ru": "Если сегодня вторник, какой день будет завтра? A) воскресенье B) понедельник C) вторник D) среда",
        "zh": "如果今天是星期二，明天是星期几？A) 星期日 B) 星期一 C) 星期二 D) 星期三",
        "ja": "今日が火曜日なら、明日は何曜日ですか？A) 日曜日 B) 月曜日 C) 火曜日 D) 水曜日",
        "fr": "Si nous sommes mardi, quel jour sera demain ? A) dimanche B) lundi C) mardi D) mercredi",
    },
    {
        "id": "L05",
        "expected": "A",
        "en": "Anna had three pencils and received two more. How many now? A) five B) four C) three D) two",
        "ru": "У Анны было три карандаша, и она получила ещё два. Сколько стало? A) пять B) четыре C) три D) два",
        "zh": "安娜有三支铅笔，又得到两支。现在有几支？A) 五 B) 四 C) 三 D) 二",
        "ja": "アンナは鉛筆を3本持ち、さらに2本もらいました。今は何本ですか？A) 5本 B) 4本 C) 3本 D) 2本",
        "fr": "Anna avait trois crayons et en a reçu deux autres. Combien en a-t-elle ? A) cinq B) quatre C) trois D) deux",
    },
    {
        "id": "L06",
        "expected": "B",
        "en": "All whales are mammals. This animal is a whale. Is it a mammal? A) no B) yes C) unknown D) only sometimes",
        "ru": "Все киты — млекопитающие. Это животное — кит. Оно млекопитающее? A) нет B) да C) неизвестно D) только иногда",
        "zh": "所有鲸都是哺乳动物。这个动物是鲸。它是哺乳动物吗？A) 不是 B) 是 C) 不知道 D) 有时是",
        "ja": "すべてのクジラは哺乳類です。この動物はクジラです。哺乳類ですか？A) いいえ B) はい C) 不明 D) 時々だけ",
        "fr": "Toutes les baleines sont des mammifères. Cet animal est une baleine. Est-ce un mammifère ? A) non B) oui C) inconnu D) parfois seulement",
    },
    {
        "id": "L07",
        "expected": "C",
        "en": "Which animal does not fly? A) eagle B) sparrow C) elephant D) bee",
        "ru": "Какое животное не летает? A) орёл B) воробей C) слон D) пчела",
        "zh": "哪种动物不会飞？A) 鹰 B) 麻雀 C) 大象 D) 蜜蜂",
        "ja": "飛ばない動物はどれですか？A) ワシ B) スズメ C) ゾウ D) ハチ",
        "fr": "Quel animal ne vole pas ? A) aigle B) moineau C) éléphant D) abeille",
    },
    {
        "id": "L08",
        "expected": "D",
        "en": "A red key is inside a blue box. What color is the box? A) red B) green C) yellow D) blue",
        "ru": "Красный ключ лежит в синей коробке. Какого цвета коробка? A) красного B) зелёного C) жёлтого D) синего",
        "zh": "一把红色钥匙在蓝色盒子里。盒子是什么颜色？A) 红色 B) 绿色 C) 黄色 D) 蓝色",
        "ja": "赤い鍵が青い箱の中にあります。箱は何色ですか？A) 赤 B) 緑 C) 黄 D) 青",
        "fr": "Une clé rouge est dans une boîte bleue. De quelle couleur est la boîte ? A) rouge B) verte C) jaune D) bleue",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    prompts = []
    metadata = []
    for task in TASKS:
        for language in LANGUAGES:
            prompts.append(
                RenderedClassifierPrompt(
                    system=SYSTEM,
                    user=str(task[language]),
                    variant=PromptVariant.CODE_PERMUTED,
                    expected_outputs=CHOICES,
                    max_new_tokens=8,
                )
            )
            metadata.append((str(task["id"]), language, str(task["expected"])))
    model = _load_model(
        {
            "model": str(args.model.resolve()),
            "batch_size": args.batch_size,
            "max_batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": 20260815,
        }
    )
    model.prepare(prompts)
    results = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs, token_counts = model.classify_batch(batch, max_new_tokens=8)
        for metadata_row, output, tokens in zip(
            metadata[start : start + len(batch)], outputs, token_counts, strict=True
        ):
            canonical_id, language, expected = metadata_row
            normalized = output.strip()
            predicted = normalized if normalized in {"A", "B", "C", "D"} else None
            results.append(
                {
                    "model_id": args.model_id,
                    "canonical_id": canonical_id,
                    "language": language,
                    "expected": expected,
                    "predicted": predicted,
                    "valid": predicted is not None,
                    "correct": predicted == expected,
                    "output_tokens": int(tokens),
                    "output_shape": classify_output_shape(output, ("A", "B", "C", "D")),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.model_id}.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    by_language = {}
    for language in LANGUAGES:
        values = [row for row in results if row["language"] == language]
        by_language[language] = {
            "rows": len(values),
            "valid": sum(bool(row["valid"]) for row in values),
            "correct": sum(bool(row["correct"]) for row in values),
        }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "model_id": args.model_id,
        "rows": len(results),
        "by_language": by_language,
        "result_sha256": get_file_sha256(rows_path),
    }
    (args.output_dir / f"{args.model_id}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
