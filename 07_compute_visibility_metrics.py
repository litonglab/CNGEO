#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STRATEGIES = [
    "fluency",
    "statistics",
    "cite_sources",
    "structured_summary",
    "quotation",
    "authoritative",
    "technical_terms",
    "keyword_stuffing",
]
SOURCE_RANKS = [1, 2, 3, 4, 5]
CITATION_RE = re.compile(r"【来源\s*([1-5])】")
SENTENCE_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fixed-source GEO visibility metrics from platform answer JSONL files."
    )
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--answer-root", default="data")
    parser.add_argument(
        "--platforms",
        default="DP,TYQW,DB,WXY",
        help="Comma-separated platform directories under --answer-root.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated GEO strategies. baseline is loaded automatically as the comparison group.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--run-output",
        default="data/metrics_runs.jsonl",
        help="Per-generation metric JSONL output.",
    )
    parser.add_argument(
        "--output",
        default="data/metrics.jsonl",
        help="Condition-level metric JSONL output after averaging repeated runs.",
    )
    parser.add_argument("--report-dir", default="outputs/reports", help="Directory for aggregate CSV reports.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Stratified question-cluster bootstrap replicates. Use 0 to disable.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument(
        "--bootstrap-domain-size",
        type=int,
        default=100,
        help="Questions sampled with replacement per domain in each replicate.",
    )
    parser.add_argument(
        "--visibility-alpha",
        type=float,
        default=0.7,
        help="Weight for Position-Aware Weighted Share in the composite Visibility metric.",
    )
    parser.add_argument(
        "--sensitivity-alphas",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated alpha values for the strategy-level sensitivity report.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--length-mode",
        choices=("content_chars", "raw_chars"),
        default="content_chars",
        help="content_chars excludes citation markers and whitespace from Word Share lengths.",
    )
    parser.add_argument("--allow-non-success", action="store_true", help="Skip non-success rows instead of failing.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print counts without writing outputs.")
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def split_float_csv(raw: str) -> list[float]:
    values = [float(item) for item in split_csv(raw)]
    invalid = [value for value in values if value < 0 or value > 1]
    if invalid:
        raise ValueError(f"alpha values must be in [0, 1]: {invalid}")
    return values


def load_queries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No query rows in {path}")
    required = {"query_id", "domain", "query_zh"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    return {row["query_id"]: row for row in rows}


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def answer_path(answer_root: Path, platform: str, strategy: str) -> Path:
    return answer_root / platform / strategy / "answers.jsonl"


def content_length(text: str, length_mode: str) -> int:
    if length_mode == "raw_chars":
        return len(text.strip())
    text = CITATION_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return len(text)


def split_sentences(answer: str) -> list[str]:
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
    return [sentence for sentence in sentences if sentence]


def citation_numbers(text: str) -> list[int]:
    return [int(match) for match in CITATION_RE.findall(text)]


def parse_answer_metrics(answer: str, length_mode: str) -> dict[int, dict[str, float]]:
    sentences = split_sentences(answer)
    sentence_lengths = [content_length(sentence, length_mode) for sentence in sentences]
    total_length = sum(sentence_lengths)
    source_metrics: dict[int, dict[str, float]] = {
        rank: {
            "citation_count": 0.0,
            "citation_share": 0.0,
            "word_share": 0.0,
            "position_word_share": 0.0,
        }
        for rank in SOURCE_RANKS
    }

    for citation in citation_numbers(answer):
        source_metrics[citation]["citation_count"] += 1.0

    total_citations = sum(source_metrics[rank]["citation_count"] for rank in SOURCE_RANKS)
    if total_citations > 0:
        for rank in SOURCE_RANKS:
            source_metrics[rank]["citation_share"] = source_metrics[rank]["citation_count"] / total_citations

    if total_length <= 0:
        return source_metrics

    sentence_count = max(1, len(sentences))
    sentence_weights = [math.exp(-index / sentence_count) for index in range(len(sentences))]
    weighted_total_length = sum(
        sentence_length * weight for sentence_length, weight in zip(sentence_lengths, sentence_weights)
    )
    for index, (sentence, sentence_length) in enumerate(zip(sentences, sentence_lengths)):
        if sentence_length <= 0:
            continue
        cited_sources = sorted(set(citation_numbers(sentence)))
        if not cited_sources:
            continue
        share = sentence_length / len(cited_sources)
        weight = sentence_weights[index]
        for source in cited_sources:
            source_metrics[source]["word_share"] += share / total_length
            if weighted_total_length > 0:
                source_metrics[source]["position_word_share"] += (share * weight) / weighted_total_length

    return source_metrics


def load_answer_index(
    path: Path,
    expected_strategy: str,
    allow_non_success: bool,
    repeats: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = load_jsonl(path)
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for line_no, row in rows:
        strategy = str(row.get("strategy", ""))
        if strategy != expected_strategy:
            raise ValueError(f"{path}:{line_no}: strategy mismatch {strategy!r} != {expected_strategy!r}")
        if row.get("status") != "success":
            if allow_non_success:
                continue
            raise ValueError(f"{path}:{line_no}: status is {row.get('status')!r}, expected success")
        try:
            run_id = int(row["run_id"])
            key = (str(row["query_id"]), int(row["target_rank"]), run_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: missing or invalid query_id/target_rank/run_id") from exc
        if run_id < 1 or run_id > repeats:
            raise ValueError(f"{path}:{line_no}: run_id {run_id} outside 1..{repeats}")
        if key in index:
            raise ValueError(f"{path}:{line_no}: duplicate answer key {key}")
        answer = str(row.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"{path}:{line_no}: empty answer for {key}")
        index[key] = {
            **row,
            "_line_no": line_no,
            "_path": str(path),
        }
    return index


def get_baseline_row(
    baseline_index: dict[tuple[str, int, int], dict[str, Any]],
    query_id: str,
    target_rank: int,
    run_id: int,
) -> dict[str, Any]:
    if (query_id, 0, run_id) in baseline_index:
        return baseline_index[(query_id, 0, run_id)]
    if (query_id, target_rank, run_id) in baseline_index:
        return baseline_index[(query_id, target_rank, run_id)]
    raise KeyError(
        f"Missing baseline answer for query_id={query_id!r}, target_rank={target_rank}, run_id={run_id}"
    )


def metric_value(metrics_by_source: dict[int, dict[str, float]], source_rank: int, metric_name: str) -> float:
    return float(metrics_by_source[source_rank][metric_name])


def make_metric_row(
    query: dict[str, str],
    platform: str,
    strategy: str,
    target_rank: int,
    run_id: int,
    baseline_row: dict[str, Any],
    after_row: dict[str, Any],
    length_mode: str,
    visibility_alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    baseline_metrics = parse_answer_metrics(str(baseline_row["answer"]), length_mode)
    after_metrics = parse_answer_metrics(str(after_row["answer"]), length_mode)

    baseline_citation_count = metric_value(baseline_metrics, target_rank, "citation_count")
    after_citation_count = metric_value(after_metrics, target_rank, "citation_count")
    baseline_citation_share = metric_value(baseline_metrics, target_rank, "citation_share")
    after_citation_share = metric_value(after_metrics, target_rank, "citation_share")
    baseline_word_share = metric_value(baseline_metrics, target_rank, "word_share")
    after_word_share = metric_value(after_metrics, target_rank, "word_share")
    baseline_position_word_share = metric_value(baseline_metrics, target_rank, "position_word_share")
    after_position_word_share = metric_value(after_metrics, target_rank, "position_word_share")

    baseline_visibility = (
        visibility_alpha * baseline_position_word_share + (1 - visibility_alpha) * baseline_citation_share
    )
    after_visibility = visibility_alpha * after_position_word_share + (1 - visibility_alpha) * after_citation_share
    absolute_gain = after_visibility - baseline_visibility
    relative_gain = absolute_gain / max(baseline_visibility, epsilon)
    baseline_visibility_is_zero = baseline_visibility <= epsilon
    relative_gain_nonzero_baseline = None if baseline_visibility_is_zero else relative_gain
    citation_share_relative_gain_nonzero_baseline = (
        None
        if baseline_citation_share <= epsilon
        else (after_citation_share - baseline_citation_share) / baseline_citation_share
    )
    position_word_share_relative_gain_nonzero_baseline = (
        None
        if baseline_position_word_share <= epsilon
        else (after_position_word_share - baseline_position_word_share) / baseline_position_word_share
    )

    return {
        "query_id": query["query_id"],
        "domain": query["domain"],
        "platform": platform,
        "target_rank": target_rank,
        "strategy": strategy,
        "run_id": run_id,
        "baseline_answer_path": baseline_row["_path"],
        "baseline_answer_line": baseline_row["_line_no"],
        "after_answer_path": after_row["_path"],
        "after_answer_line": after_row["_line_no"],
        "baseline_citation_count": int(baseline_citation_count),
        "after_citation_count": int(after_citation_count),
        "baseline_citation_share": baseline_citation_share,
        "after_citation_share": after_citation_share,
        "baseline_word_share": baseline_word_share,
        "after_word_share": after_word_share,
        "baseline_position_word_share": baseline_position_word_share,
        "after_position_word_share": after_position_word_share,
        "citation_count_gain": after_citation_count - baseline_citation_count,
        "citation_share_gain": after_citation_share - baseline_citation_share,
        "word_share_gain": after_word_share - baseline_word_share,
        "position_word_share_gain": after_position_word_share - baseline_position_word_share,
        "visibility_metric": "composite",
        "visibility_alpha": visibility_alpha,
        "baseline_visibility": baseline_visibility,
        "after_visibility": after_visibility,
        "absolute_gain": absolute_gain,
        "relative_gain": relative_gain,
        "baseline_visibility_is_zero": baseline_visibility_is_zero,
        "relative_gain_nonzero_baseline": relative_gain_nonzero_baseline,
        "citation_share_relative_gain_nonzero_baseline": citation_share_relative_gain_nonzero_baseline,
        "position_word_share_relative_gain_nonzero_baseline": position_word_share_relative_gain_nonzero_baseline,
        "is_positive": absolute_gain > 0,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


RUN_AVERAGE_FIELDS = [
    "baseline_citation_count",
    "after_citation_count",
    "baseline_citation_share",
    "after_citation_share",
    "baseline_word_share",
    "after_word_share",
    "baseline_position_word_share",
    "after_position_word_share",
]


def add_derived_metrics(row: dict[str, Any], visibility_alpha: float, epsilon: float) -> dict[str, Any]:
    updated = dict(row)
    baseline_citation_count = float(row["baseline_citation_count"])
    after_citation_count = float(row["after_citation_count"])
    baseline_citation_share = float(row["baseline_citation_share"])
    after_citation_share = float(row["after_citation_share"])
    baseline_word_share = float(row["baseline_word_share"])
    after_word_share = float(row["after_word_share"])
    baseline_position_word_share = float(row["baseline_position_word_share"])
    after_position_word_share = float(row["after_position_word_share"])
    baseline_visibility = (
        visibility_alpha * baseline_position_word_share + (1 - visibility_alpha) * baseline_citation_share
    )
    after_visibility = visibility_alpha * after_position_word_share + (1 - visibility_alpha) * after_citation_share
    absolute_gain = after_visibility - baseline_visibility
    relative_gain = absolute_gain / max(baseline_visibility, epsilon)
    baseline_visibility_is_zero = baseline_visibility <= epsilon
    updated.update(
        {
            "citation_count_gain": after_citation_count - baseline_citation_count,
            "citation_share_gain": after_citation_share - baseline_citation_share,
            "word_share_gain": after_word_share - baseline_word_share,
            "position_word_share_gain": after_position_word_share - baseline_position_word_share,
            "visibility_metric": "composite",
            "visibility_alpha": visibility_alpha,
            "baseline_visibility": baseline_visibility,
            "after_visibility": after_visibility,
            "absolute_gain": absolute_gain,
            "relative_gain": relative_gain,
            "baseline_visibility_is_zero": baseline_visibility_is_zero,
            "relative_gain_nonzero_baseline": None if baseline_visibility_is_zero else relative_gain,
            "citation_share_relative_gain_nonzero_baseline": (
                None
                if baseline_citation_share <= epsilon
                else (after_citation_share - baseline_citation_share) / baseline_citation_share
            ),
            "position_word_share_relative_gain_nonzero_baseline": (
                None
                if baseline_position_word_share <= epsilon
                else (after_position_word_share - baseline_position_word_share)
                / baseline_position_word_share
            ),
            "is_positive": absolute_gain > 0,
        }
    )
    return updated


def average_run_rows(
    run_rows: list[dict[str, Any]],
    repeats: int,
    allow_incomplete: bool,
    visibility_alpha: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    group_fields = ["query_id", "domain", "platform", "target_rank", "strategy"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    expected_run_ids = set(range(1, repeats + 1))
    averaged_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        run_ids = {int(row["run_id"]) for row in group}
        if not allow_incomplete and run_ids != expected_run_ids:
            raise ValueError(
                f"Incomplete repeated condition {key}: expected run_ids={sorted(expected_run_ids)}, "
                f"actual={sorted(run_ids)}"
            )
        item = {field: value for field, value in zip(group_fields, key)}
        item["run_count"] = len(group)
        item["run_ids"] = sorted(run_ids)
        for field in RUN_AVERAGE_FIELDS:
            item[field] = mean([float(row[field]) for row in group])
        averaged_rows.append(add_derived_metrics(item, visibility_alpha, epsilon))
    return averaged_rows


def aggregate_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        item = {field: value for field, value in zip(group_fields, key)}
        gains = [float(row["absolute_gain"]) for row in group]
        relative_gains = [float(row["relative_gain"]) for row in group]
        relative_gains_nonzero = [
            float(row["relative_gain_nonzero_baseline"])
            for row in group
            if row["relative_gain_nonzero_baseline"] is not None
        ]
        citation_share_relative_gains_nonzero = [
            float(row["citation_share_relative_gain_nonzero_baseline"])
            for row in group
            if row["citation_share_relative_gain_nonzero_baseline"] is not None
        ]
        position_word_share_relative_gains_nonzero = [
            float(row["position_word_share_relative_gain_nonzero_baseline"])
            for row in group
            if row["position_word_share_relative_gain_nonzero_baseline"] is not None
        ]
        positive_relative_gains_nonzero = [
            float(row["relative_gain_nonzero_baseline"])
            for row in group
            if row["is_positive"] and row["relative_gain_nonzero_baseline"] is not None
        ]
        positive_citation_share_relative_gains_nonzero = [
            float(row["citation_share_relative_gain_nonzero_baseline"])
            for row in group
            if row["is_positive"] and row["citation_share_relative_gain_nonzero_baseline"] is not None
        ]
        positive_position_word_share_relative_gains_nonzero = [
            float(row["position_word_share_relative_gain_nonzero_baseline"])
            for row in group
            if row["is_positive"] and row["position_word_share_relative_gain_nonzero_baseline"] is not None
        ]
        item.update(
            {
                "n": len(group),
                "visibility_alpha": group[0]["visibility_alpha"],
                "positive_rate": mean([1.0 if row["is_positive"] else 0.0 for row in group]),
                "baseline_zero_rate": mean([1.0 if row["baseline_visibility_is_zero"] else 0.0 for row in group]),
                "mean_absolute_gain": mean(gains),
                "median_absolute_gain": median(gains),
                "mean_relative_gain": mean(relative_gains),
                "median_relative_gain": median(relative_gains),
                "mean_relative_gain_nonzero_baseline": mean(relative_gains_nonzero),
                "median_relative_gain_nonzero_baseline": median(relative_gains_nonzero),
                "mean_citation_share_relative_gain_nonzero_baseline": mean(
                    citation_share_relative_gains_nonzero
                ),
                "mean_position_word_share_relative_gain_nonzero_baseline": mean(
                    position_word_share_relative_gains_nonzero
                ),
                "mean_positive_relative_gain_nonzero_baseline": mean(positive_relative_gains_nonzero),
                "mean_positive_citation_share_relative_gain_nonzero_baseline": mean(
                    positive_citation_share_relative_gains_nonzero
                ),
                "mean_positive_position_word_share_relative_gain_nonzero_baseline": mean(
                    positive_position_word_share_relative_gains_nonzero
                ),
                "mean_baseline_visibility": mean([float(row["baseline_visibility"]) for row in group]),
                "mean_after_visibility": mean([float(row["after_visibility"]) for row in group]),
                "mean_citation_count_gain": mean([float(row["citation_count_gain"]) for row in group]),
                "mean_citation_share_gain": mean([float(row["citation_share_gain"]) for row in group]),
                "mean_word_share_gain": mean([float(row["word_share_gain"]) for row in group]),
                "mean_position_word_share_gain": mean([float(row["position_word_share_gain"]) for row in group]),
                "mean_baseline_citation_share": mean([float(row["baseline_citation_share"]) for row in group]),
                "mean_after_citation_share": mean([float(row["after_citation_share"]) for row in group]),
                "mean_baseline_word_share": mean([float(row["baseline_word_share"]) for row in group]),
                "mean_after_word_share": mean([float(row["after_word_share"]) for row in group]),
                "mean_baseline_position_word_share": mean(
                    [float(row["baseline_position_word_share"]) for row in group]
                ),
                "mean_after_position_word_share": mean([float(row["after_position_word_share"]) for row in group]),
            }
        )
        item.update(
            {
                "pooled_visibility_relative_gain": (
                    item["mean_absolute_gain"] / item["mean_baseline_visibility"]
                    if item["mean_baseline_visibility"] > 0
                    else None
                ),
                "pooled_citation_share_relative_gain": (
                    item["mean_citation_share_gain"] / item["mean_baseline_citation_share"]
                    if item["mean_baseline_citation_share"] > 0
                    else None
                ),
                "pooled_position_word_share_relative_gain": (
                    item["mean_position_word_share_gain"] / item["mean_baseline_position_word_share"]
                    if item["mean_baseline_position_word_share"] > 0
                    else None
                ),
            }
        )
        out_rows.append(item)
    return out_rows


GROUP_FIELD_LABELS = {
    "platform": "platform",
    "domain": "domain",
    "target_rank": "target_rank",
    "strategy": "strategy",
    "visibility_alpha": "visibility_alpha",
}


def make_display_rows(aggregate_rows_: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in aggregate_rows_:
        item = {GROUP_FIELD_LABELS[field]: row[field] for field in group_fields}
        item["n"] = row["n"]
        if "visibility_alpha" not in group_fields:
            item.update(
                {
                    "baseline_citation_share_mean": row["mean_baseline_citation_share"],
                    "after_citation_share_mean": row["mean_after_citation_share"],
                    "citation_share_absolute_gain_mean": row["mean_citation_share_gain"],
                    "citation_share_pooled_relative_gain": row["pooled_citation_share_relative_gain"],
                    "baseline_position_word_share_mean": row["mean_baseline_position_word_share"],
                    "after_position_word_share_mean": row["mean_after_position_word_share"],
                    "position_word_share_absolute_gain_mean": row["mean_position_word_share_gain"],
                    "position_word_share_pooled_relative_gain": row[
                        "pooled_position_word_share_relative_gain"
                    ],
                }
            )
        item.update(
            {
                "baseline_visibility_mean": row["mean_baseline_visibility"],
                "after_visibility_mean": row["mean_after_visibility"],
                "visibility_absolute_gain_mean": row["mean_absolute_gain"],
                "visibility_pooled_relative_gain": row["pooled_visibility_relative_gain"],
            }
        )
        rows.append(item)
    rows.sort(key=lambda item: float(item["visibility_absolute_gain_mean"]), reverse=True)
    return rows


BOOTSTRAP_METRICS = {
    "citation_share": ("baseline_citation_share", "citation_share_gain"),
    "position_word_share": ("baseline_position_word_share", "position_word_share_gain"),
    "visibility": ("baseline_visibility", "absolute_gain"),
}


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def build_cluster_contributions(
    rows: list[dict[str, Any]],
    group_fields: list[str],
) -> dict[str, dict[tuple[Any, ...], dict[str, list[float]]]]:
    clusters: dict[str, dict[tuple[Any, ...], dict[str, list[float]]]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        group_key = tuple(row[field] for field in group_fields)
        group = clusters.setdefault(query_id, {}).setdefault(
            group_key,
            {metric: [0.0, 0.0] for metric in BOOTSTRAP_METRICS},
        )
        for metric, (baseline_field, gain_field) in BOOTSTRAP_METRICS.items():
            group[metric][0] += float(row[baseline_field])
            group[metric][1] += float(row[gain_field])
    return clusters


def bootstrap_confidence_intervals(
    rows: list[dict[str, Any]],
    report_specs: list[list[str]],
    samples: int,
    seed: int,
    domain_size: int,
) -> dict[tuple[str, ...], dict[tuple[Any, ...], dict[str, float | int | None]]]:
    if samples <= 0:
        return {}

    query_domains: dict[str, str] = {}
    for row in rows:
        query_id = str(row["query_id"])
        domain = str(row["domain"])
        previous = query_domains.setdefault(query_id, domain)
        if previous != domain:
            raise ValueError(f"query_id {query_id!r} appears in multiple domains")

    queries_by_domain: dict[str, list[str]] = {}
    for query_id, domain in query_domains.items():
        queries_by_domain.setdefault(domain, []).append(query_id)
    for query_ids in queries_by_domain.values():
        query_ids.sort()

    spec_keys = [tuple(spec) for spec in report_specs]
    contributions = {
        spec_key: build_cluster_contributions(rows, list(spec_key))
        for spec_key in spec_keys
    }
    draws: dict[
        tuple[str, ...],
        dict[tuple[Any, ...], dict[str, list[float]]],
    ] = {
        spec_key: {}
        for spec_key in spec_keys
    }
    rng = random.Random(seed)

    for _ in range(samples):
        sampled_query_ids: list[str] = []
        for domain in sorted(queries_by_domain):
            query_ids = queries_by_domain[domain]
            sample_size = domain_size if domain_size > 0 else len(query_ids)
            sampled_query_ids.extend(rng.choice(query_ids) for _ in range(sample_size))

        for spec_key in spec_keys:
            totals: dict[tuple[Any, ...], dict[str, list[float]]] = {}
            spec_contributions = contributions[spec_key]
            for query_id in sampled_query_ids:
                for group_key, metrics in spec_contributions.get(query_id, {}).items():
                    group = totals.setdefault(
                        group_key,
                        {metric: [0.0, 0.0] for metric in BOOTSTRAP_METRICS},
                    )
                    for metric in BOOTSTRAP_METRICS:
                        group[metric][0] += metrics[metric][0]
                        group[metric][1] += metrics[metric][1]

            spec_draws = draws[spec_key]
            for group_key, metrics in totals.items():
                group_draws = spec_draws.setdefault(
                    group_key,
                    {metric: [] for metric in BOOTSTRAP_METRICS},
                )
                for metric in BOOTSTRAP_METRICS:
                    baseline_sum, gain_sum = metrics[metric]
                    if baseline_sum > 0:
                        group_draws[metric].append(gain_sum / baseline_sum)

    intervals: dict[
        tuple[str, ...],
        dict[tuple[Any, ...], dict[str, float | int | None]],
    ] = {}
    for spec_key, spec_draws in draws.items():
        intervals[spec_key] = {}
        for group_key, metric_draws in spec_draws.items():
            summary: dict[str, float | int | None] = {"bootstrap_samples": samples}
            for metric, values in metric_draws.items():
                summary[f"{metric}_relative_ci_low"] = percentile(values, 0.025)
                summary[f"{metric}_relative_ci_high"] = percentile(values, 0.975)
                summary[f"{metric}_relative_bootstrap_median"] = percentile(values, 0.5)
            intervals[spec_key][group_key] = summary
    return intervals


def attach_bootstrap_intervals(
    rows: list[dict[str, Any]],
    group_fields: list[str],
    intervals: dict[tuple[Any, ...], dict[str, float | int | None]],
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        key = tuple(row[field] for field in group_fields)
        item.update(intervals.get(key, {}))
        attached.append(item)
    return attached


def recompute_visibility(row: dict[str, Any], visibility_alpha: float, epsilon: float) -> dict[str, Any]:
    updated = dict(row)
    baseline_visibility = (
        visibility_alpha * float(row["baseline_position_word_share"])
        + (1 - visibility_alpha) * float(row["baseline_citation_share"])
    )
    after_visibility = (
        visibility_alpha * float(row["after_position_word_share"])
        + (1 - visibility_alpha) * float(row["after_citation_share"])
    )
    absolute_gain = after_visibility - baseline_visibility
    relative_gain = absolute_gain / max(baseline_visibility, epsilon)
    baseline_visibility_is_zero = baseline_visibility <= epsilon
    updated.update(
        {
            "visibility_alpha": visibility_alpha,
            "baseline_visibility": baseline_visibility,
            "after_visibility": after_visibility,
            "absolute_gain": absolute_gain,
            "relative_gain": relative_gain,
            "baseline_visibility_is_zero": baseline_visibility_is_zero,
            "relative_gain_nonzero_baseline": None if baseline_visibility_is_zero else relative_gain,
            "is_positive": absolute_gain > 0,
        }
    )
    return updated


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    queries = load_queries(Path(args.queries))
    platforms = split_csv(args.platforms)
    strategies = split_csv(args.strategies)
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")

    answer_root = Path(args.answer_root)
    metric_rows: list[dict[str, Any]] = []
    loaded_counts: Counter[str] = Counter()

    for platform in platforms:
        baseline_file = answer_path(answer_root, platform, "baseline")
        baseline_index = load_answer_index(
            baseline_file,
            "baseline",
            args.allow_non_success,
            args.repeats,
        )
        loaded_counts[f"{platform}/baseline"] = len(baseline_index)

        expected_baselines = {
            (query_id, 0, run_id)
            for query_id in queries
            for run_id in range(1, args.repeats + 1)
        }
        missing_baselines = sorted(expected_baselines - set(baseline_index))
        if missing_baselines and not args.allow_non_success:
            raise ValueError(
                f"{baseline_file}: missing {len(missing_baselines)} baseline runs, "
                f"sample={missing_baselines[:5]}"
            )

        for strategy in strategies:
            strategy_file = answer_path(answer_root, platform, strategy)
            after_index = load_answer_index(
                strategy_file,
                strategy,
                args.allow_non_success,
                args.repeats,
            )
            loaded_counts[f"{platform}/{strategy}"] = len(after_index)
            expected = {
                (query_id, target_rank, run_id)
                for query_id in queries
                for target_rank in SOURCE_RANKS
                for run_id in range(1, args.repeats + 1)
            }
            missing = sorted(expected - set(after_index))
            if missing and not args.allow_non_success:
                sample = ", ".join(f"{query_id}:{rank}:run{run_id}" for query_id, rank, run_id in missing[:5])
                raise ValueError(f"{strategy_file}: missing {len(missing)} answer rows, sample: {sample}")

            for query_id in sorted(queries):
                for target_rank in SOURCE_RANKS:
                    for run_id in range(1, args.repeats + 1):
                        after_row = after_index.get((query_id, target_rank, run_id))
                        if after_row is None:
                            continue
                        try:
                            baseline_row = get_baseline_row(
                                baseline_index,
                                query_id,
                                target_rank,
                                run_id,
                            )
                        except KeyError:
                            if args.allow_non_success:
                                continue
                            raise
                        metric_rows.append(
                            make_metric_row(
                                queries[query_id],
                                platform,
                                strategy,
                                target_rank,
                                run_id,
                                baseline_row,
                                after_row,
                                args.length_mode,
                                args.visibility_alpha,
                                args.epsilon,
                            )
                        )

    print("loaded:", ", ".join(f"{key}={value}" for key, value in sorted(loaded_counts.items())), flush=True)
    return metric_rows


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.bootstrap_domain_size < 0:
        raise ValueError("--bootstrap-domain-size must be non-negative")
    if args.visibility_alpha < 0 or args.visibility_alpha > 1:
        raise ValueError("--visibility-alpha must be in [0, 1]")
    sensitivity_alphas = split_float_csv(args.sensitivity_alphas)
    run_metric_rows = build_run_metric_rows(args)
    metric_rows = average_run_rows(
        run_metric_rows,
        args.repeats,
        args.allow_non_success,
        args.visibility_alpha,
        args.epsilon,
    )
    print(f"computed run metric rows: {len(run_metric_rows)}", flush=True)
    print(f"averaged condition metric rows: {len(metric_rows)}", flush=True)

    if args.dry_run:
        return 0

    run_output_path = Path(args.run_output)
    write_jsonl(run_output_path, run_metric_rows)
    output_path = Path(args.output)
    write_jsonl(output_path, metric_rows)
    report_dir = Path(args.report_dir)
    report_specs = [
        ["strategy"],
        ["platform", "strategy"],
        ["domain", "strategy"],
        ["target_rank", "strategy"],
        ["platform", "domain", "strategy"],
    ]
    bootstrap_intervals = bootstrap_confidence_intervals(
        metric_rows,
        report_specs,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.bootstrap_domain_size,
    )

    def report_rows(group_fields: list[str]) -> list[dict[str, Any]]:
        display_rows = make_display_rows(aggregate_rows(metric_rows, group_fields), group_fields)
        return attach_bootstrap_intervals(
            display_rows,
            group_fields,
            bootstrap_intervals.get(tuple(group_fields), {}),
        )

    write_csv(report_dir / "strategy_summary.csv", report_rows(["strategy"]))
    write_csv(report_dir / "platform_strategy_summary.csv", report_rows(["platform", "strategy"]))
    write_csv(report_dir / "domain_strategy_summary.csv", report_rows(["domain", "strategy"]))
    write_csv(report_dir / "rank_strategy_summary.csv", report_rows(["target_rank", "strategy"]))
    write_csv(
        report_dir / "platform_domain_strategy_summary.csv",
        report_rows(["platform", "domain", "strategy"]),
    )
    sensitivity_rows = [
        recompute_visibility(row, visibility_alpha, args.epsilon)
        for visibility_alpha in sensitivity_alphas
        for row in metric_rows
    ]
    write_csv(
        report_dir / "alpha_sensitivity.csv",
        make_display_rows(
            aggregate_rows(sensitivity_rows, ["visibility_alpha", "strategy"]),
            ["visibility_alpha", "strategy"],
        ),
    )
    print(f"wrote run metrics: {run_output_path}", flush=True)
    print(f"wrote averaged metrics: {output_path}", flush=True)
    print(f"wrote reports: {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
