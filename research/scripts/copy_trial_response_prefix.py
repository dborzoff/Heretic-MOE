#!/usr/bin/env python3
"""Copy only a numeric trial prefix from a Heretic response archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

TRIAL_COLUMNS = ("trial_number", "trial_index", "trial")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--max-trial", type=int, required=True)
    args = parser.parse_args()

    if args.target.exists():
        raise FileExistsError(args.target)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.target)

    report: dict[str, object] = {
        "status": "PASS",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "target": str(args.target.resolve()),
        "max_trial_exclusive": args.max_trial,
        "tables": {},
    }
    connection = sqlite3.connect(args.target)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table in tables:
            quoted_table = quote_identifier(table)
            columns = [
                row[1]
                for row in connection.execute(f"PRAGMA table_info({quoted_table})")
            ]
            trial_column = next(
                (name for name in TRIAL_COLUMNS if name in columns), None
            )
            before = connection.execute(
                f"SELECT COUNT(*) FROM {quoted_table}"
            ).fetchone()[0]
            if trial_column is not None:
                quoted_column = quote_identifier(trial_column)
                connection.execute(
                    f"DELETE FROM {quoted_table} WHERE {quoted_column} >= ?",
                    (args.max_trial,),
                )
            after = connection.execute(
                f"SELECT COUNT(*) FROM {quoted_table}"
            ).fetchone()[0]
            report["tables"][table] = {
                "columns": columns,
                "trial_column": trial_column,
                "rows_before": before,
                "rows_after": after,
            }
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()

    report["target_sha256"] = sha256(args.target)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
