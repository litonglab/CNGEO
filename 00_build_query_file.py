#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_FILES = {
    "healthcare.csv": "health",
    "law_and_government.csv": "law_gov",
    "education_and_academia.csv": "education",
    "technology_and_digital.csv": "tech_digital",
    "life_culture_and_society.csv": "life_society",
    "finance_and_investment.csv": "finance",
}
FIELDNAMES = ["query_id", "domain", "query_zh"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the six released domain CSV files into one experiment input file."
    )
    parser.add_argument("--input-dir", default="dataset")
    parser.add_argument("--output", default="data/queries.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows: list[dict[str, str]] = []

    for filename, expected_domain in DATASET_FILES.items():
        path = input_dir / filename
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDNAMES:
                raise ValueError(f"{path}: expected columns {FIELDNAMES}, found {reader.fieldnames}")
            domain_rows = list(reader)
        if len(domain_rows) != 100:
            raise ValueError(f"{path}: expected 100 questions, found {len(domain_rows)}")
        if any(row["domain"] != expected_domain for row in domain_rows):
            raise ValueError(f"{path}: unexpected domain value")
        rows.extend(domain_rows)

    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id values found")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} questions: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
