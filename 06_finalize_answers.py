#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
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
        description="Keep five unique successful runs per condition and sort answer JSONL files."
    )
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--answer-root", default="data")
    parser.add_argument("--platforms", default="DP,TYQW,DB,WXY")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backup-root", default="outputs/backups")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_query_order(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    query_ids = [str(row.get("query_id", "")).strip() for row in rows]
    if not query_ids or any(not query_id for query_id in query_ids):
        raise ValueError(f"{path}: missing query_id")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{path}: duplicate query_id")
    return query_ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def expected_keys(query_ids: list[str], strategy: str, repeats: int) -> set[tuple[str, int, str, int]]:
    if strategy == "baseline":
        return {
            (query_id, 0, strategy, run_id)
            for query_id in query_ids
            for run_id in range(1, repeats + 1)
        }
    return {
        (query_id, rank, strategy, run_id)
        for query_id in query_ids
        for rank in range(1, 6)
        for run_id in range(1, repeats + 1)
    }


def finalize_file(
    path: Path,
    platform: str,
    strategy: str,
    query_ids: list[str],
    query_positions: dict[str, int],
    repeats: int,
    backup_dir: Path | None,
) -> tuple[int, int, int]:
    rows = load_jsonl(path)
    successful: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    non_success_count = 0
    duplicate_success_count = 0

    for line_number, row in enumerate(rows, start=1):
        if row.get("status") != "success":
            non_success_count += 1
            continue
        query_id = str(row.get("query_id", ""))
        row_strategy = str(row.get("strategy", ""))
        try:
            target_rank = int(row.get("target_rank"))
            run_id = int(row.get("run_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid target_rank or run_id") from exc
        if run_id < 1 or run_id > repeats:
            raise ValueError(f"{path}:{line_number}: run_id {run_id} outside 1..{repeats}")
        if row_strategy != strategy:
            raise ValueError(f"{path}:{line_number}: strategy {row_strategy!r} != {strategy!r}")
        key = (query_id, target_rank, row_strategy, run_id)
        if key in successful:
            duplicate_success_count += 1
            if successful[key].get("answer") != row.get("answer"):
                raise ValueError(f"{path}:{line_number}: conflicting successful duplicate for {key}")
            continue
        successful[key] = row

    expected = expected_keys(query_ids, strategy, repeats)
    actual = set(successful)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{path}: key mismatch, missing={len(missing)}, extra={len(extra)}, "
            f"missing_sample={missing[:3]}, extra_sample={extra[:3]}"
        )

    sorted_rows = sorted(
        successful.values(),
        key=lambda row: (
            query_positions[str(row["query_id"])],
            int(row["target_rank"]),
            int(row["run_id"]),
        ),
    )

    if backup_dir is not None:
        backup_path = backup_dir / platform / strategy / "answers.jsonl"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            for row in sorted_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(path)

    return len(sorted_rows), non_success_count, duplicate_success_count


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    query_ids = load_query_order(Path(args.queries))
    query_positions = {query_id: position for position, query_id in enumerate(query_ids)}
    platforms = split_csv(args.platforms)
    strategies = split_csv(args.strategies)
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = None if args.dry_run else Path(args.backup_root) / f"answers_before_finalize_{timestamp}"

    total_kept = 0
    total_removed = 0
    total_duplicates = 0
    for platform in platforms:
        for strategy in strategies:
            path = Path(args.answer_root) / platform / strategy / "answers.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            kept, removed, duplicates = finalize_file(
                path,
                platform,
                strategy,
                query_ids,
                query_positions,
                args.repeats,
                backup_dir,
            )
            total_kept += kept
            total_removed += removed
            total_duplicates += duplicates
            print(
                f"{platform}/{strategy}: kept={kept}, "
                f"removed_non_success={removed}, removed_duplicate_success={duplicates}"
            )

    print(
        f"summary: kept={total_kept}, removed_non_success={total_removed}, "
        f"removed_duplicate_success={total_duplicates}"
    )
    if backup_dir is not None:
        print(f"backup: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
