#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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

META_PREFIXES = (
    "以下是",
    "下面是",
    "改写结果",
    "已根据",
    "我将",
    "我已",
    "说明：",
    "分析：",
)


class RewriteError(RuntimeError):
    pass


class ProviderBlockedError(RewriteError):
    pass


FATAL_API_ERROR_PATTERNS = (
    "HTTP 401",
    "HTTP 403",
    "invalid api",
    "invalid-api",
    "invalidapikey",
    "unauthorized",
    "forbidden",
    "permission denied",
    "access denied",
    "insufficient balance",
    "insufficient quota",
    "quota exhausted",
    "quota exceeded",
    "billing",
    "arrear",
    "overdue",
    "欠费",
    "余额不足",
    "额度不足",
    "额度已用完",
    "配额不足",
    "配额已用完",
    "鉴权失败",
    "认证失败",
    "无效的api",
    "无效的 api",
    "未开通",
)


def is_fatal_api_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return any(pattern in message or pattern in lowered for pattern in FATAL_API_ERROR_PATTERNS)


def is_provider_blocked_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "data_inspection_failed" in lowered or "inappropriate content" in lowered


def make_ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite Top-K source bodies with GEO strategies via an OpenAI-compatible API."
    )
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--search-results", default="data/search_results_cleaned.jsonl")
    parser.add_argument("--output-dir", default="data/rewrites")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--log-output", default="outputs/logs/rewrite_debug.jsonl")
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated strategy IDs. Use baseline for no-op copy.",
    )
    parser.add_argument("--query-id", action="append", default=[], help="Only process the given query_id. Can be repeated.")
    parser.add_argument("--domain", action="append", default=[], help="Only process the given domain. Can be repeated.")
    parser.add_argument("--rank", action="append", type=int, default=[], help="Only process the given source rank. Can be repeated.")
    parser.add_argument("--limit-queries", type=int, default=0, help="Process first N query_ids after filtering.")
    parser.add_argument("--limit-sources", type=int, default=0, help="Process first N source rows after filtering.")
    parser.add_argument("--body-max-chars", type=int, default=12000, help="Maximum body chars sent to the rewrite API.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and prompts, print planned tasks, but do not write files or call APIs.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip rows already successfully written.")
    parser.add_argument("--strict-validation", action="store_true", help="Treat validation warnings as failed rewrites.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed API call or validation failure.")
    parser.add_argument(
        "--max-consecutive-api-failures",
        type=int,
        default=5,
        help="Stop after N consecutive API/task exceptions. Use 0 to disable.",
    )

    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Random sleep upper bound between rewrite API calls.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent rewrite workers. Keep 1 for sequential runs.")

    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default=None)
    return parser.parse_args()


def split_strategies(raw: str) -> list[str]:
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return strategies


def load_queries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No query rows in {path}")
    required = {"query_id", "domain", "query_zh"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"queries missing columns: {', '.join(sorted(missing))}")
    return {row["query_id"]: row for row in rows}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def load_done_keys(output_dir: Path, strategies: list[str]) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    done_statuses = {"success", "provider_blocked"}
    for strategy in strategies:
        path = output_dir / f"{strategy}.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row.get("rewrite_status") not in done_statuses:
                continue
            query_id = str(row.get("query_id", ""))
            try:
                rank = int(row.get("rank"))
            except (TypeError, ValueError):
                continue
            done.add((query_id, rank, strategy))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompts(prompt_dir: Path, strategies: list[str]) -> tuple[str, dict[str, str]]:
    common_path = prompt_dir / "common_rewrite_rules.txt"
    common_rules = common_path.read_text(encoding="utf-8").strip()
    strategy_prompts: dict[str, str] = {}
    for strategy in strategies:
        if strategy == "baseline":
            continue
        path = prompt_dir / "rewrite_prompts" / f"{strategy}.txt"
        strategy_prompts[strategy] = path.read_text(encoding="utf-8").strip()
    return common_rules, strategy_prompts


def filter_sources(args: argparse.Namespace, queries: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.search_results))
    query_filter = set(args.query_id)
    domain_filter = set(args.domain)
    rank_filter = set(args.rank)

    if args.limit_queries > 0:
        ordered_query_ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            query_id = str(row.get("query_id", ""))
            if query_id and query_id not in seen:
                ordered_query_ids.append(query_id)
                seen.add(query_id)
        query_filter = set(ordered_query_ids[: args.limit_queries]) if not query_filter else query_filter

    filtered: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id not in queries:
            continue
        query_row = queries[query_id]
        if query_filter and query_id not in query_filter:
            continue
        if domain_filter and query_row["domain"] not in domain_filter:
            continue
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            continue
        if rank_filter and rank not in rank_filter:
            continue
        body = str(row.get("body", "")).strip()
        if not body:
            continue
        merged = dict(row)
        merged["rank"] = rank
        merged["domain"] = query_row["domain"]
        merged["query_zh"] = query_row["query_zh"]
        filtered.append(merged)

    if args.limit_sources > 0:
        filtered = filtered[: args.limit_sources]
    return filtered


def truncate_body(body: str, max_chars: int) -> str:
    body = body.strip()
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip()


def build_prompt(
    source: dict[str, Any],
    strategy: str,
    common_rules: str,
    strategy_prompt: str,
    body_max_chars: int,
) -> str:
    body = truncate_body(str(source["body"]), body_max_chars)
    min_chars = int(len(body) * 0.85)
    max_chars = int(len(body) * 1.25)
    return f"""你将根据给定的 GEO 改写策略，对一篇中文网页正文进行改写。

【来源元数据】
query_id：{source["query_id"]}
领域：{source["domain"]}
用户问题：{source["query_zh"]}
百度来源排名：{source["rank"]}
标题：{source.get("title", "")}
策略：{strategy}

【公共改写约束】
{common_rules}

【当前改写策略】
{strategy_prompt}

【原始网页正文】
{body}

请严格按照公共约束和当前策略改写正文。
覆盖要求：除明显网页噪声、重复广告、无关导航和乱码外，原文中的关键事实、例子、步骤、清单、比较项、风险提示和结论都应保留或合并表达；不要只保留开头和结尾。
本次原文长度约为 {len(body)} 个中文字符，改写后正文应控制在 {min_chars} 到 {max_chars} 个中文字符之间；除非原文包含大量明显网页噪声，不要低于 {min_chars} 个中文字符。
只输出改写后的正文，不输出任何解释、分析、代码块或 JSON。"""


def normalize_api_output(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff").strip()
    return text


def strip_list_markers(text: str) -> str:
    return re.sub(r"(?m)^\s*(?:[-*]\s+|\d+[.)、]\s*)", "", text)


def extract_numbers(text: str) -> set[str]:
    cleaned = strip_list_markers(text)
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.])", cleaned))


def quoted_segments(text: str) -> list[str]:
    segments = re.findall(r"[“\"]([^”\"]{10,80})[”\"]", text)
    return [segment.strip() for segment in segments if segment.strip()]


def normalize_quote_match_text(text: str) -> str:
    return re.sub(r"[\s，。、“”\"'‘’：:；;！!？?（）()《》<>【】\[\]「」『』·・—\-、]", "", text)


def validate_output(
    output: str,
    source_body: str,
    strategy: str,
    strict: bool,
) -> tuple[bool, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    stripped = output.strip()
    if not stripped:
        blockers.append("empty output")
        return False, blockers, warnings
    if "```" in stripped:
        blockers.append("contains Markdown code fence")
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        blockers.append("looks like JSON wrapper")
    if any(stripped.startswith(prefix) for prefix in META_PREFIXES):
        blockers.append("looks like meta explanation")
    if len(stripped) < 50:
        blockers.append("too short")

    source_len = max(1, len(source_body.strip()))
    ratio = len(stripped) / source_len
    if ratio < 0.75:
        warnings.append(f"length ratio too low: {ratio:.2f}")
    elif ratio > 1.35:
        warnings.append(f"length ratio too high: {ratio:.2f}")

    if strategy == "statistics":
        source_numbers = extract_numbers(source_body)
        output_numbers = extract_numbers(stripped)
        new_numbers = sorted(output_numbers - source_numbers)
        if new_numbers:
            warnings.append("statistics may contain new numbers: " + ", ".join(new_numbers[:20]))

    if strategy == "quotation":
        normalized_source = normalize_quote_match_text(source_body)
        unmatched: list[str] = []
        for segment in quoted_segments(stripped):
            normalized_segment = normalize_quote_match_text(segment)
            if normalized_segment and normalized_segment not in normalized_source:
                unmatched.append(segment)
        if unmatched:
            warnings.append("quotation may contain non-original quote: " + " | ".join(unmatched[:5]))

    if strict and warnings:
        blockers.extend(warnings)
        warnings = []
    return not blockers, blockers, warnings


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=make_ssl_context()) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and ("data_inspection_failed" in error_body or "inappropriate content" in error_body.lower()):
            raise ProviderBlockedError(f"HTTP {exc.code}: {error_body[:1000]}") from exc
        raise RewriteError(f"HTTP {exc.code}: {error_body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RewriteError(f"URL error: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RewriteError(f"Invalid JSON response: {body[:1000]}") from exc


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def extract_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}

    result: dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for target_key, source_keys in aliases.items():
        for source_key in source_keys:
            value = usage.get(source_key)
            if isinstance(value, int):
                result[target_key] = value
                break
            if isinstance(value, str) and value.isdigit():
                result[target_key] = int(value)
                break

    if "total_tokens" not in result and {"prompt_tokens", "completion_tokens"} <= set(result):
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def call_rewrite_api(prompt: str, args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RewriteError(f"Missing API key. Set {args.api_key_env} before running non-baseline strategies.")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "你是严谨的中文网页正文改写助手，只输出改写后的正文。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.thinking is not None:
        payload["thinking"] = {"type": args.thinking}
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = post_json(chat_completions_url(args.base_url), payload, headers, args.timeout)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RewriteError(f"Unexpected rewrite API response: {json.dumps(response, ensure_ascii=False)[:1000]}") from exc
    return normalize_api_output(str(content)), extract_usage(response)


def call_rewrite_api_with_retries(prompt: str, args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    last_error: Exception | None = None
    attempts = max(1, args.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return call_rewrite_api(prompt, args)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(args.retry_sleep * attempt)
    assert last_error is not None
    raise RewriteError(str(last_error))


def make_output_row(source: dict[str, Any], strategy: str, body: str, status: str) -> dict[str, Any]:
    return {
        "query_id": source["query_id"],
        "rank": source["rank"],
        "strategy": strategy,
        "title": source.get("title", ""),
        "body": body,
        "rewrite_status": status,
    }


def maybe_sleep(max_sleep: float) -> None:
    if max_sleep > 0:
        time.sleep(random.uniform(0, max_sleep))


def process_rewrite_task(
    source: dict[str, Any],
    strategy: str,
    common_rules: str,
    strategy_prompts: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = Path(args.output_dir) / f"{strategy}.jsonl"
    log_base = {
        "query_id": source["query_id"],
        "rank": source["rank"],
        "strategy": strategy,
        "title": source.get("title", ""),
    }

    try:
        api_usage: dict[str, int] = {}
        if strategy == "baseline":
            rewritten = str(source["body"]).strip()
            blockers: list[str] = []
            warnings: list[str] = []
            status = "success"
        else:
            prompt = build_prompt(
                source,
                strategy,
                common_rules,
                strategy_prompts[strategy],
                args.body_max_chars,
            )
            rewritten, api_usage = call_rewrite_api_with_retries(prompt, args)
            valid, blockers, warnings = validate_output(rewritten, str(source["body"]), strategy, args.strict_validation)
            status = "success" if valid else "validation_failed"

        log_row = {
            **log_base,
            "status": status,
            "body_chars": len(str(source["body"])),
            "rewrite_chars": len(rewritten),
            "validation_blockers": blockers,
            "validation_warnings": warnings,
            "model": args.model,
            "prompt_tokens": api_usage.get("prompt_tokens"),
            "completion_tokens": api_usage.get("completion_tokens"),
            "total_tokens": api_usage.get("total_tokens"),
        }
        maybe_sleep(args.sleep)
        return {
            "status": status,
            "source": source,
            "strategy": strategy,
            "output_path": output_path,
            "output_row": make_output_row(source, strategy, rewritten if status == "success" else rewritten[:4000], status),
            "log_row": log_row,
            "api_usage": api_usage,
            "fatal": False,
        }
    except Exception as exc:
        status = "provider_blocked" if is_provider_blocked_error(exc) else "failed"
        maybe_sleep(args.sleep)
        return {
            "status": status,
            "source": source,
            "strategy": strategy,
            "output_path": output_path,
            "output_row": make_output_row(source, strategy, "", status),
            "log_row": {**log_base, "status": status, "error": f"{type(exc).__name__}: {exc}"},
            "api_usage": {},
            "fatal": is_fatal_api_error(exc),
            "error": exc,
        }


def handle_task_result(
    result: dict[str, Any],
    args: argparse.Namespace,
    counters: Counter[str],
    usage_totals: Counter[str],
    consecutive_api_failures: int,
) -> tuple[int, bool]:
    status = str(result["status"])
    source = result["source"]
    strategy = str(result["strategy"])

    append_jsonl(result["output_path"], result["output_row"])
    append_jsonl(Path(args.log_output), result["log_row"])

    api_usage = result.get("api_usage", {})
    for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage_totals[token_key] += api_usage.get(token_key, 0)

    counters[status] += 1
    if status in {"failed", "provider_blocked"}:
        consecutive_api_failures += 1
        error = result.get("error")
        print(f"{status}: {strategy} {source['query_id']} rank={source['rank']}: {error}", file=sys.stderr, flush=True)
        if status == "provider_blocked":
            return 0, False
        if result.get("fatal"):
            print(
                "aborted: fatal API/account error detected; check the provider console "
                f"and the {args.api_key_env} environment variable.",
                file=sys.stderr,
                flush=True,
            )
            return consecutive_api_failures, True
        if args.max_consecutive_api_failures > 0 and consecutive_api_failures >= args.max_consecutive_api_failures:
            print(
                f"aborted: {consecutive_api_failures} consecutive API/task failures; "
                "fix the error and rerun the same command to resume.",
                file=sys.stderr,
                flush=True,
            )
            return consecutive_api_failures, True
        if args.fail_fast:
            return consecutive_api_failures, True
    else:
        consecutive_api_failures = 0
        print(f"{status}: {strategy} {source['query_id']} rank={source['rank']}", flush=True)
        if status != "success" and args.fail_fast:
            return consecutive_api_failures, True

    return consecutive_api_failures, False


def run_tasks(
    pending_tasks: list[tuple[dict[str, Any], str]],
    common_rules: str,
    strategy_prompts: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Counter[str], Counter[str]]:
    counters: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    consecutive_api_failures = 0
    workers = max(1, args.workers)

    if workers == 1:
        for source, strategy in pending_tasks:
            result = process_rewrite_task(source, strategy, common_rules, strategy_prompts, args)
            consecutive_api_failures, abort = handle_task_result(
                result, args, counters, usage_totals, consecutive_api_failures
            )
            if abort:
                break
        return counters, usage_totals

    task_iter = iter(pending_tasks)
    in_flight: dict[Future[dict[str, Any]], None] = {}
    abort_requested = False

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            source, strategy = next(task_iter)
        except StopIteration:
            return False
        future = executor.submit(process_rewrite_task, source, strategy, common_rules, strategy_prompts, args)
        in_flight[future] = None
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(workers, len(pending_tasks))):
            submit_next(executor)

        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                in_flight.pop(future, None)
                result = future.result()
                consecutive_api_failures, abort = handle_task_result(
                    result, args, counters, usage_totals, consecutive_api_failures
                )
                if abort:
                    abort_requested = True
                if not abort_requested:
                    submit_next(executor)

    return counters, usage_totals


def main() -> int:
    args = parse_args()
    args.workers = max(1, args.workers)
    strategies = split_strategies(args.strategies)
    queries = load_queries(Path(args.queries))
    sources = filter_sources(args, queries)
    common_rules, strategy_prompts = load_prompts(Path(args.prompt_dir), strategies)
    done = set() if args.no_resume or args.dry_run else load_done_keys(Path(args.output_dir), strategies)

    pending_tasks = [
        (source, strategy)
        for source in sources
        for strategy in strategies
        if (source["query_id"], source["rank"], strategy) not in done
    ]
    total_tasks = len(pending_tasks)
    print(
        f"loaded {len(sources)} source rows, {len(strategies)} strategies, {total_tasks} pending tasks, "
        f"workers={args.workers}",
        flush=True,
    )

    if args.dry_run:
        for source in sources[:3]:
            print(f"- {source['query_id']} rank={source['rank']} title={source.get('title', '')[:60]}", flush=True)
        rewrite_strategies = [strategy for strategy in strategies if strategy != "baseline"]
        if sources and rewrite_strategies:
            preview_strategy = rewrite_strategies[0]
            prompt = build_prompt(
                sources[0],
                preview_strategy,
                common_rules,
                strategy_prompts[preview_strategy],
                args.body_max_chars,
            )
            print("\n--- prompt preview ---", flush=True)
            print(prompt[:2000], flush=True)
            print("--- end preview ---", flush=True)
        return 0

    counters, usage_totals = run_tasks(pending_tasks, common_rules, strategy_prompts, args)
    skipped = len(sources) * len(strategies) - total_tasks
    if skipped:
        counters["skipped"] = skipped

    print("summary:", ", ".join(f"{key}={value}" for key, value in sorted(counters.items())), flush=True)
    if usage_totals:
        print(
            "usage:",
            ", ".join(f"{key}={usage_totals[key]}" for key in ("prompt_tokens", "completion_tokens", "total_tokens")),
            flush=True,
        )
    return 0 if counters["failed"] == 0 and counters["validation_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
