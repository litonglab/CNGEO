#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


STRATEGIES = [
    "baseline",
    "fluency",
    "statistics",
    "cite_sources",
    "structured_summary",
    "quotation",
    "authoritative",
    "technical_terms",
    "keyword_stuffing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact rewrite JSONL files by keeping the latest row per source and restoring source order."
    )
    parser.add_argument("--search-results", default="data/search_results_cleaned.jsonl")
    parser.add_argument("--rewrite-dir", default="data/rewrites")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--no-backup", action="store_true", help="Do not copy original files before rewriting.")
    return parser.parse_args()


def split_strategies(raw: str) -> list[str]:
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return strategies


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_source_order(path: Path) -> list[tuple[str, int]]:
    order: list[tuple[str, int]] = []
    for row in load_jsonl(path):
        order.append((str(row["query_id"]), int(row["rank"])))
    if len(order) != len(set(order)):
        raise ValueError(f"{path}: duplicate source keys in search results")
    return order


def compact_strategy_file(
    path: Path,
    strategy: str,
    source_order: list[tuple[str, int]],
) -> dict[str, Any]:
    source_keys = set(source_order)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    original_rows = 0

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            original_rows += 1
            row = json.loads(line)
            row_strategy = str(row.get("strategy"))
            if row_strategy != strategy:
                raise ValueError(f"{path}:{line_no}: strategy mismatch: {row_strategy} != {strategy}")
            key = (str(row.get("query_id")), int(row.get("rank")))
            if key not in source_keys:
                raise ValueError(f"{path}:{line_no}: source key not found in search results: {key}")
            latest[key] = row

    missing = [key for key in source_order if key not in latest]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} source keys, first={missing[:3]}")

    ordered_rows = [latest[key] for key in source_order]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)

    status_counts = Counter(str(row.get("rewrite_status")) for row in ordered_rows)
    return {
        "original_rows": original_rows,
        "cleaned_rows": len(ordered_rows),
        "removed_rows": original_rows - len(ordered_rows),
        "status_counts": dict(status_counts),
    }


def main() -> int:
    args = parse_args()
    strategies = split_strategies(args.strategies)
    rewrite_dir = Path(args.rewrite_dir)
    source_order = load_source_order(Path(args.search_results))

    backup_dir: Path | None = None
    if not args.no_backup:
        backup_dir = rewrite_dir.parent / f"{rewrite_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {}
    for strategy in strategies:
        path = rewrite_dir / f"{strategy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        if backup_dir is not None:
            shutil.copy2(path, backup_dir / path.name)
        summary[strategy] = compact_strategy_file(path, strategy, source_order)

    if backup_dir is not None:
        print(f"backup_dir: {backup_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
