"""Build the frozen Ministral sparse-geometry prototype bank.

The script prints only counts and hashes. Text remains in the output JSONL and
is never emitted to the console.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"F:\AI")
RESPONSES = (
    ROOT
    / "heretic-moe/research/results/adaptive_search_v2/unified_600_finalists"
    / "semantic_136x2048"
)
LABELS = (
    ROOT
    / "heretic-moe/research/results/adaptive_search_v2"
    / "unified_600_finalists_ninelabel_rejudge/judge_work/cjudge_labels.jsonl"
)
SPLIT = (
    ROOT
    / "heretic_test/results/refusal_lab/semantic_3head_v1/split_manifest.json"
)
OUTPUT_DIR = (
    ROOT
    / "heretic-moe/research/results/refusal_classifier_eval/sparse_geometry_bank_v1"
)
OUTPUT = OUTPUT_DIR / "prototypes.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"

SYSTEM_CODES = {
    "original": "A",
    "trial260": "B",
    "trial290": "C",
    "trial597": "D",
    "trial320": "E",
    "trial488": "F",
}
SOFT_LABELS = {"soft_substitution", "soft_legal", "soft_inversion"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coarse(label: str) -> str:
    if label in SOFT_LABELS:
        return "soft"
    if label == "refuse_policy":
        return "refuse"
    if label in {"comply", "comply_degraded", "partial"}:
        return "delivered"
    return "other"


def main() -> None:
    train_ids = set(
        map(int, json.loads(SPLIT.read_text(encoding="utf-8"))["train_ids"])
    )
    labels = {
        (row["system_code"], int(row["id"])): coarse(row["label"])
        for row in read_jsonl(LABELS)
    }
    bank: list[dict] = []
    sources: list[Path] = []
    for system, code in SYSTEM_CODES.items():
        source = RESPONSES / f"{system}.responses.jsonl"
        sources.append(source)
        for row in read_jsonl(source):
            row_id = int(row["id"])
            label = labels[(code, row_id)]
            if row_id not in train_ids or label == "other":
                continue
            bank.append(
                {
                    "id": row_id,
                    "system": system,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "label": label,
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in bank:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["label"] for row in bank)
    manifest = {
        "schema_version": 1,
        "rows": len(bank),
        "prompt_ids": len({row["id"] for row in bank}),
        "label_counts": dict(sorted(counts.items())),
        "prototype_sha256": sha256(OUTPUT),
        "split_sha256": sha256(SPLIT),
        "labels_sha256": sha256(LABELS),
        "source_sha256": {path.name: sha256(path) for path in sources},
        "text_free_console_contract": True,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "prompt_ids": manifest["prompt_ids"],
                "label_counts": manifest["label_counts"],
                "prototype_sha256": manifest["prototype_sha256"],
                "manifest": str(MANIFEST),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
