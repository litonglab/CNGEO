#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from pathlib import Path
from typing import Any


META_PREFIXES = (
    "以下是清洗",
    "清洗后的",
    "清洗结果",
    "处理结果",
    "说明：",
    "分析：",
)

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


class CleaningError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean crawled webpage bodies with an OpenAI-compatible chat completion API."
    )
    parser.add_argument("--input", default="data/search_results.jsonl")
    parser.add_argument("--output", default="data/search_results_cleaned.jsonl")
    parser.add_argument("--prompt-file", default="prompts/source_cleaning.txt")
    parser.add_argument("--log-output", default="outputs/logs/source_cleaning_debug.jsonl")
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--body-max-chars", type=int, default=12000)
    parser.add_argument("--min-output-chars", type=int, default=100)
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the cleaned output instead of resuming from existing rows.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--max-consecutive-api-failures",
        type=int,
        default=5,
        help="Stop after N consecutive API exceptions. Use 0 to disable.",
    )

    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            try:
                key = (str(row["query_id"]), int(row["rank"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid query_id or rank") from exc
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate source key {key}")
            body = str(row.get("body", "")).strip()
            if not body:
                raise ValueError(f"{path}:{line_number}: empty body for {key}")
            seen.add(key)
            rows.append(
                {
                    "query_id": key[0],
                    "rank": key[1],
                    "title": str(row.get("title", "")).strip(),
                    "body": body,
                }
            )
    return rows


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["query_id"]), int(row["rank"]))
            if key in done:
                raise ValueError(f"{path}:{line_number}: duplicate cleaned key {key}")
            if not str(row.get("body", "")).strip():
                raise ValueError(f"{path}:{line_number}: empty cleaned body for {key}")
            done.add(key)
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    query_filter = set(args.query_id)
    rank_filter = set(args.rank)
    filtered = [
        row
        for row in rows
        if (not query_filter or row["query_id"] in query_filter)
        and (not rank_filter or row["rank"] in rank_filter)
    ]
    if args.limit > 0:
        filtered = filtered[: args.limit]
    return filtered


def build_prompt(row: dict[str, Any], cleaning_rules: str, body_max_chars: int) -> str:
    body = str(row["body"])
    if body_max_chars > 0:
        body = body[:body_max_chars]
    return f"""请按照以下规则清洗抓取到的中文网页文本。

【清洗规则】
{cleaning_rules}

【页面元数据】
query_id：{row["query_id"]}
来源位置：{row["rank"]}
网页标题：{row["title"]}

【抓取到的原始网页文本】
{body}

只输出清洗后的网页正文，不要输出解释、分析、标题标签、代码块或JSON。"""


def normalize_output(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip().lstrip("\ufeff").strip()


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?%?(?![A-Za-z0-9_.])", text))


def validate_cleaned_body(
    source_body: str,
    cleaned_body: str,
    min_output_chars: int,
    strict: bool,
) -> tuple[bool, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    cleaned = cleaned_body.strip()
    if not cleaned:
        blockers.append("empty output")
        return False, blockers, warnings
    if len(cleaned) < min_output_chars:
        blockers.append(f"output shorter than {min_output_chars} characters")
    if "```" in cleaned:
        blockers.append("contains Markdown code fence")
    if (cleaned.startswith("{") and cleaned.endswith("}")) or (
        cleaned.startswith("[") and cleaned.endswith("]")
    ):
        blockers.append("looks like a JSON wrapper")
    if any(cleaned.startswith(prefix) for prefix in META_PREFIXES):
        blockers.append("looks like a cleaning explanation")

    new_numbers = sorted(extract_numbers(cleaned) - extract_numbers(source_body))
    if new_numbers:
        blockers.append("contains numbers not found in source: " + ", ".join(new_numbers[:20]))

    source_length = max(1, len(source_body.strip()))
    length_ratio = len(cleaned) / source_length
    if length_ratio > 1.20:
        warnings.append(f"output longer than source: {length_ratio:.2f}")
    elif length_ratio < 0.20:
        warnings.append(f"large reduction in body length: {length_ratio:.2f}")

    if strict and warnings:
        blockers.extend(warnings)
        warnings = []
    return not blockers, blockers, warnings


def make_ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=make_ssl_context(),
        ) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise CleaningError(f"HTTP {exc.code}: {error_body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise CleaningError(f"URL error: {exc}") from exc
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise CleaningError(f"Invalid JSON response: {response_text[:1000]}") from exc


def extract_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    output: dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for output_key, source_keys in aliases.items():
        for source_key in source_keys:
            value = usage.get(source_key)
            if isinstance(value, int):
                output[output_key] = value
                break
    if "total_tokens" not in output and {"prompt_tokens", "completion_tokens"} <= set(output):
        output["total_tokens"] = output["prompt_tokens"] + output["completion_tokens"]
    return output


def is_fatal_api_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return any(pattern in message or pattern in lowered for pattern in FATAL_API_ERROR_PATTERNS)


def call_cleaning_api(
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, int]]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise CleaningError(f"Missing API key. Set {args.api_key_env} before running.")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是中文网页正文清洗助手。只删除网页噪声，保留原文事实和表达，"
                    "不要总结、改写或补充内容。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    response = post_json(
        chat_completions_url(args.base_url),
        payload,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        args.timeout,
    )
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CleaningError(
            f"Unexpected cleaning API response: {json.dumps(response, ensure_ascii=False)[:1000]}"
        ) from exc
    if choice.get("finish_reason") == "length":
        raise CleaningError(
            "Cleaning output reached the token limit. Increase --max-tokens and retry."
        )
    return normalize_output(str(content)), extract_usage(response)


def call_with_retries(
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, int]]:
    last_error: Exception | None = None
    attempts = max(1, args.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return call_cleaning_api(prompt, args)
        except Exception as exc:
            last_error = exc
            if is_fatal_api_error(exc):
                break
            if attempt < attempts:
                time.sleep(args.retry_sleep * attempt)
    assert last_error is not None
    raise CleaningError(str(last_error))


def process_source(
    row: dict[str, Any],
    cleaning_rules: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = build_prompt(row, cleaning_rules, args.body_max_chars)
    started = time.time()
    try:
        cleaned_body, usage = call_with_retries(prompt, args)
        validation_source = f"{row['title']}\n{row['body']}".strip()
        valid, blockers, warnings = validate_cleaned_body(
            validation_source,
            cleaned_body,
            args.min_output_chars,
            args.strict_validation,
        )
        status = "success" if valid else "validation_failed"
        output_row = (
            {
                "query_id": row["query_id"],
                "rank": row["rank"],
                "title": row["title"],
                "body": cleaned_body,
            }
            if valid
            else None
        )
        return {
            "status": status,
            "output_row": output_row,
            "log_row": {
                "query_id": row["query_id"],
                "rank": row["rank"],
                "status": status,
                "model": args.model,
                "source_body_chars": len(str(row["body"])),
                "cleaned_body_chars": len(cleaned_body),
                "validation_blockers": blockers,
                "validation_warnings": warnings,
                "latency_ms": int((time.time() - started) * 1000),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "usage": usage,
            "fatal": False,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "output_row": None,
            "log_row": {
                "query_id": row["query_id"],
                "rank": row["rank"],
                "status": "failed",
                "model": args.model,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": int((time.time() - started) * 1000),
            },
            "usage": {},
            "fatal": is_fatal_api_error(exc),
            "error": exc,
        }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log_output)
    cleaning_rules = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    sources = filter_sources(load_jsonl(input_path), args)

    if args.overwrite and not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    done = set() if args.overwrite else load_done_keys(output_path)
    pending = [row for row in sources if (row["query_id"], row["rank"]) not in done]

    print(
        f"loaded {len(sources)} source rows, {len(done)} existing cleaned rows, "
        f"{len(pending)} pending rows, model={args.model}",
        flush=True,
    )
    if args.dry_run:
        for row in pending[:3]:
            print(f"- {row['query_id']} rank={row['rank']} title={row['title'][:60]}", flush=True)
        if pending:
            print("\n--- cleaning prompt preview ---", flush=True)
            print(build_prompt(pending[0], cleaning_rules, args.body_max_chars)[:3000], flush=True)
            print("--- end preview ---", flush=True)
        return 0

    counters: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    consecutive_failures = 0
    for index, row in enumerate(pending, start=1):
        result = process_source(row, cleaning_rules, args)
        status = str(result["status"])
        counters[status] += 1
        append_jsonl(log_path, result["log_row"])
        if result["output_row"] is not None:
            append_jsonl(output_path, result["output_row"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[key] += result["usage"].get(key, 0)

        print(
            f"[{index}/{len(pending)}] {status}: {row['query_id']} rank={row['rank']}",
            flush=True,
        )
        if status == "failed":
            consecutive_failures += 1
            print(f"error: {result.get('error')}", file=sys.stderr, flush=True)
            if result["fatal"]:
                print(
                    f"aborted: check the provider console and {args.api_key_env}.",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if (
                args.max_consecutive_api_failures > 0
                and consecutive_failures >= args.max_consecutive_api_failures
            ):
                print("aborted: too many consecutive API failures.", file=sys.stderr, flush=True)
                break
        else:
            consecutive_failures = 0
        if args.fail_fast and status != "success":
            break
        if args.sleep > 0:
            time.sleep(random.uniform(0, args.sleep))

    print("summary:", ", ".join(f"{key}={value}" for key, value in sorted(counters.items())))
    print(
        "usage:",
        ", ".join(
            f"{key}={usage_totals[key]}"
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ),
    )
    return 0 if counters["failed"] == 0 and counters["validation_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
