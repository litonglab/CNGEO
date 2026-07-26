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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
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

PLATFORM_CONFIGS: dict[str, dict[str, str]] = {
    "DP": {
        "provider": "bailian",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "BAILIAN_API_KEY",
        "model": "deepseek-v4-flash",
    },
    "TYQW": {
        "provider": "bailian",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "BAILIAN_API_KEY",
        "model": "qwen3.6-flash-2026-04-16",
        "thinking": "disabled",
    },
    "DB": {
        "provider": "ark",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "ARK_API_KEY",
        "model": "doubao-seed-2-1-turbo-260628",
        "thinking": "disabled",
    },
    "WXY": {
        "provider": "qianfan",
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_key_env": "QIANFAN_API_KEY",
        "model": "ernie-4.5-turbo-20260402",
    },
}

DEFAULT_SOURCE_ORDER = [1, 2, 3, 4, 5]
DONE_STATUSES = {"success"}

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


class AnswerError(RuntimeError):
    pass


class ProviderBlockedError(AnswerError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-source GEO answer outputs from baseline and rewrite JSONL files."
    )
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--rewrite-dir", default="data/rewrites")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--log-output", default="outputs/logs/answer_debug.jsonl")
    parser.add_argument(
        "--platforms",
        default="DP,TYQW,DB,WXY",
        help="Comma-separated platform labels used as data/{platform}/... directories.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated strategies. baseline means no source is replaced.",
    )
    parser.add_argument("--query-id", action="append", default=[], help="Only process the given query_id. Can be repeated.")
    parser.add_argument("--domain", action="append", default=[], help="Only process the given domain. Can be repeated.")
    parser.add_argument(
        "--target-rank",
        action="append",
        type=int,
        default=[],
        help="Target rank(s) to replace for non-baseline strategies. Defaults to 1..5.",
    )
    parser.add_argument("--limit-queries", type=int, default=0, help="Process first N query_ids after filtering.")
    parser.add_argument("--limit-tasks", type=int, default=0, help="Process first N pending answer tasks after filtering.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent answer generations for each experimental condition.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("once", "per-rank"),
        default="once",
        help="once writes one baseline row per query with --baseline-target-rank; per-rank calls baseline for each target rank.",
    )
    parser.add_argument(
        "--baseline-target-rank",
        type=int,
        default=0,
        help="target_rank value written for baseline rows when --baseline-mode=once.",
    )
    parser.add_argument(
        "--source-max-chars",
        type=int,
        default=0,
        help="Maximum chars kept per source body in the answer prompt. 0 means no truncation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks and a prompt preview without API calls.")
    parser.add_argument(
        "--preview-prompts",
        type=int,
        default=1,
        help="Number of prompt previews printed during --dry-run.",
    )
    parser.add_argument(
        "--preview-prompt-chars",
        type=int,
        default=3000,
        help="Characters printed for each --dry-run prompt preview. 0 means print the full prompt.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing successful answer rows.")
    parser.add_argument("--strict-validation", action="store_true", help="Treat answer validation warnings as failed rows.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed API call or validation failure.")
    parser.add_argument(
        "--max-consecutive-api-failures",
        type=int,
        default=5,
        help="Stop after N consecutive API/task exceptions. Use 0 to disable.",
    )

    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="BAILIAN_API_KEY")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--platform-model",
        action="append",
        default=[],
        help="Optional per-platform model mapping, e.g. DP=deepseek-v4-flash or TYQW=qwen-plus. Can be repeated.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. When set, run_id N uses seed + N - 1.",
    )
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Random sleep upper bound between API calls.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent output-file workers. Each worker processes one platform/strategy answers file sequentially.",
    )
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def split_strategies(raw: str) -> list[str]:
    strategies = split_csv(raw)
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return strategies


def parse_platform_models(items: list[str], default_model: str) -> dict[str, str]:
    mapping = {platform: config["model"] for platform, config in PLATFORM_CONFIGS.items()}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--platform-model must be PLATFORM=MODEL, got: {item}")
        platform, model = item.split("=", 1)
        platform = platform.strip()
        model = model.strip()
        if not platform or not model:
            raise ValueError(f"--platform-model must be PLATFORM=MODEL, got: {item}")
        mapping[platform] = model
    mapping["__default__"] = default_model
    return mapping


def load_queries(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No query rows in {path}")
    required = {"query_id", "domain", "query_zh"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"queries missing columns: {', '.join(sorted(missing))}")
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_queries(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    query_filter = set(args.query_id)
    domain_filter = set(args.domain)
    filtered: list[dict[str, str]] = []
    for row in rows:
        if query_filter and row["query_id"] not in query_filter:
            continue
        if domain_filter and row["domain"] not in domain_filter:
            continue
        filtered.append(row)
    if args.limit_queries > 0:
        filtered = filtered[: args.limit_queries]
    return filtered


def load_rewrite_index(rewrite_dir: Path, strategy: str) -> dict[tuple[str, int], dict[str, Any]]:
    path = rewrite_dir / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for line_no, row in enumerate(rows, start=1):
        row_strategy = str(row.get("strategy", ""))
        if row_strategy != strategy:
            raise ValueError(f"{path}:{line_no}: strategy mismatch: {row_strategy} != {strategy}")
        if row.get("rewrite_status") != "success":
            raise ValueError(
                f"{path}:{line_no}: rewrite_status is {row.get('rewrite_status')!r}; "
                "answer generation expects finalized rewrite files."
            )
        try:
            key = (str(row["query_id"]), int(row["rank"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: missing or invalid query_id/rank") from exc
        if key in index:
            raise ValueError(f"{path}:{line_no}: duplicate key {key}")
        body = str(row.get("body", "")).strip()
        if not body:
            raise ValueError(f"{path}:{line_no}: empty body for {key}")
        row = dict(row)
        row["body"] = body
        index[key] = row
    return index


def load_rewrite_indexes(rewrite_dir: Path, strategies: list[str]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    required = sorted(set(strategies) | {"baseline"})
    return {strategy: load_rewrite_index(rewrite_dir, strategy) for strategy in required}


def output_path(output_root: Path, platform: str, strategy: str) -> Path:
    return output_root / platform / strategy / "answers.jsonl"


def load_done_keys(
    output_root: Path,
    platforms: list[str],
    strategies: list[str],
) -> set[tuple[str, str, int, str, int]]:
    done: set[tuple[str, str, int, str, int]] = set()
    for platform in platforms:
        for strategy in strategies:
            path = output_path(output_root, platform, strategy)
            if not path.exists():
                continue
            for row in load_jsonl(path):
                if row.get("status") not in DONE_STATUSES:
                    continue
                query_id = str(row.get("query_id", ""))
                try:
                    target_rank = int(row.get("target_rank"))
                    run_id = int(row.get("run_id"))
                except (TypeError, ValueError):
                    continue
                done.add((platform, query_id, target_rank, strategy, run_id))
    return done


def truncate_body(body: str, max_chars: int) -> str:
    body = body.strip()
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip()


def build_source_rows(
    query_id: str,
    target_rank: int,
    strategy: str,
    indexes: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in DEFAULT_SOURCE_ORDER:
        key = (query_id, rank)
        source_strategy = strategy if strategy != "baseline" and rank == target_rank else "baseline"
        try:
            rows.append(indexes[source_strategy][key])
        except KeyError as exc:
            raise ValueError(f"Missing rewrite row for strategy={source_strategy} query_id={query_id} rank={rank}") from exc
    return rows


def format_source_block(rank: int, row: dict[str, Any], max_chars: int) -> str:
    title = str(row.get("title", "")).strip()
    body = truncate_body(str(row["body"]), max_chars)
    if title:
        return f"来源 {rank}：\n标题：{title}\n正文：\n{body}"
    return f"来源 {rank}：\n{body}"


def build_prompt(query: dict[str, str], source_rows: list[dict[str, Any]], source_max_chars: int) -> str:
    source_blocks = "\n\n".join(
        format_source_block(rank, row, source_max_chars) for rank, row in zip(DEFAULT_SOURCE_ORDER, source_rows)
    )
    return f"""你是一个有帮助的中文问答助手。用户会提供一个问题以及 5 篇相关文章的内容。请你仅基于这 5 篇内容回答问题，不得使用外部知识。

回答要求：
- 用自然、流畅的中文段落直接回答问题。
- 每一句包含事实、观点、建议、判断或结论的陈述，句末都要标注来源，格式为【来源 X】。
- 如果一句话同时由多个来源支持，请连续标注，例如【来源 1】【来源 3】。
- 请优先使用与问题最相关、信息最充分、最能支持回答的来源。
- 如果某些来源与问题关系较弱、信息重复或无法支持回答，可以不引用。
- 不要为了平均覆盖 5 篇来源而强行引用不相关来源。
- 不要编造来源之外的信息。
- 不要输出参考文献列表，不要输出 URL，只在正文中使用【来源 X】标注。

Query：
{query["query_zh"]}

{source_blocks}"""


def normalize_api_output(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff").strip()
    return text


def citation_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in re.findall(r"【来源\s*([0-9]+)】", text):
        try:
            numbers.append(int(match))
        except ValueError:
            continue
    return numbers


def validate_answer(answer: str, strict: bool) -> tuple[bool, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    stripped = answer.strip()
    if not stripped:
        blockers.append("empty answer")
        return False, blockers, warnings
    if "```" in stripped:
        blockers.append("contains Markdown code fence")
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        blockers.append("looks like JSON wrapper")

    citations = citation_numbers(stripped)
    if not citations:
        blockers.append("missing source citations")
    invalid = sorted({num for num in citations if num not in DEFAULT_SOURCE_ORDER})
    if invalid:
        blockers.append("invalid source citation numbers: " + ", ".join(str(num) for num in invalid))

    if len(stripped) < 50:
        warnings.append("answer too short")

    if strict and warnings:
        blockers.extend(warnings)
        warnings = []
    return not blockers, blockers, warnings


def make_ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


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
        raise AnswerError(f"HTTP {exc.code}: {error_body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise AnswerError(f"URL error: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise AnswerError(f"Invalid JSON response: {body[:1000]}") from exc


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


def is_fatal_api_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    return any(pattern in message or pattern in lowered for pattern in FATAL_API_ERROR_PATTERNS)


def is_provider_blocked_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "data_inspection_failed" in lowered or "inappropriate content" in lowered


def resolve_platform_api(platform: str, args: argparse.Namespace) -> tuple[str, str]:
    config = PLATFORM_CONFIGS.get(platform)
    if config is None:
        api_key = os.environ.get(args.api_key_env, "").strip()
        return args.base_url, api_key

    api_key = os.environ.get(config["api_key_env"], "").strip()
    return config["base_url"], api_key


def resolve_thinking_mode(platform: str, args: argparse.Namespace) -> str | None:
    if args.thinking is not None:
        return args.thinking
    config = PLATFORM_CONFIGS.get(platform, {})
    return config.get("thinking")


def apply_thinking_mode(payload: dict[str, Any], platform: str, args: argparse.Namespace) -> None:
    thinking_mode = resolve_thinking_mode(platform, args)
    if thinking_mode is None:
        return

    provider = PLATFORM_CONFIGS.get(platform, {}).get("provider")
    if provider == "bailian":
        payload["enable_thinking"] = thinking_mode == "enabled"
    else:
        payload["thinking"] = {"type": thinking_mode}


def call_answer_api(
    prompt: str,
    platform: str,
    model: str,
    args: argparse.Namespace,
    seed: int | None,
) -> tuple[str, dict[str, int], str | None]:
    base_url, api_key = resolve_platform_api(platform, args)
    if not api_key:
        raise AnswerError(f"Missing API key for platform {platform}.")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是严格基于给定来源回答问题的中文问答助手，必须在正文中使用【来源 X】格式标注依据。",
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
    if seed is not None:
        payload["seed"] = seed
    apply_thinking_mode(payload, platform, args)
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = post_json(chat_completions_url(base_url), payload, headers, args.timeout)
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AnswerError(f"Unexpected answer API response: {json.dumps(response, ensure_ascii=False)[:1000]}") from exc
    finish_reason = choice.get("finish_reason")
    return normalize_api_output(str(content)), extract_usage(response), str(finish_reason) if finish_reason is not None else None


def call_answer_api_with_retries(
    prompt: str,
    platform: str,
    model: str,
    args: argparse.Namespace,
    seed: int | None,
) -> tuple[str, dict[str, int], str | None]:
    last_error: Exception | None = None
    attempts = max(1, args.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return call_answer_api(prompt, platform, model, args, seed)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(args.retry_sleep * attempt)
    assert last_error is not None
    raise AnswerError(str(last_error))


def maybe_sleep(max_sleep: float) -> None:
    if max_sleep > 0:
        time.sleep(random.uniform(0, max_sleep))


def make_answer_row(
    query_id: str,
    target_rank: int,
    strategy: str,
    run_id: int,
    answer: str,
    status: str,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "target_rank": target_rank,
        "strategy": strategy,
        "run_id": run_id,
        "source_order": DEFAULT_SOURCE_ORDER,
        "answer": answer,
        "status": status,
    }


def process_answer_task(task: dict[str, Any], args: argparse.Namespace, platform_models: dict[str, str]) -> dict[str, Any]:
    platform = str(task["platform"])
    strategy = str(task["strategy"])
    query = task["query"]
    query_id = str(query["query_id"])
    target_rank = int(task["target_rank"])
    run_id = int(task["run_id"])
    seed = None if args.seed is None else args.seed + run_id - 1
    output = output_path(Path(args.output_root), platform, strategy)
    model = platform_models.get(platform, platform_models["__default__"])
    prompt = str(task["prompt"])
    log_base = {
        "platform": platform,
        "provider": PLATFORM_CONFIGS.get(platform, {}).get("provider", "custom"),
        "model": model,
        "query_id": query_id,
        "domain": query.get("domain"),
        "target_rank": target_rank,
        "strategy": strategy,
        "run_id": run_id,
        "seed": seed,
        "prompt_chars": len(prompt),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": resolve_thinking_mode(platform, args),
    }
    started = time.time()

    try:
        answer, api_usage, finish_reason = call_answer_api_with_retries(prompt, platform, model, args, seed)
        valid, blockers, warnings = validate_answer(answer, args.strict_validation)
        status = "success" if valid else "validation_failed"
        latency_ms = int((time.time() - started) * 1000)
        log_row = {
            **log_base,
            "status": status,
            "answer_chars": len(answer),
            "validation_blockers": blockers,
            "validation_warnings": warnings,
            "latency_ms": latency_ms,
            "finish_reason": finish_reason,
            "prompt_tokens": api_usage.get("prompt_tokens"),
            "completion_tokens": api_usage.get("completion_tokens"),
            "total_tokens": api_usage.get("total_tokens"),
        }
        maybe_sleep(args.sleep)
        return {
            "status": status,
            "platform": platform,
            "strategy": strategy,
            "query_id": query_id,
            "target_rank": target_rank,
            "run_id": run_id,
            "output_path": output,
            "output_row": make_answer_row(
                query_id,
                target_rank,
                strategy,
                run_id,
                answer if status == "success" else answer[:4000],
                status,
            ),
            "log_row": log_row,
            "api_usage": api_usage,
            "fatal": False,
        }
    except Exception as exc:
        status = "provider_blocked" if is_provider_blocked_error(exc) else "failed"
        maybe_sleep(args.sleep)
        return {
            "status": status,
            "platform": platform,
            "strategy": strategy,
            "query_id": query_id,
            "target_rank": target_rank,
            "run_id": run_id,
            "output_path": output,
            "output_row": make_answer_row(query_id, target_rank, strategy, run_id, "", status),
            "log_row": {**log_base, "status": status, "error": f"{type(exc).__name__}: {exc}"},
            "api_usage": {},
            "fatal": is_fatal_api_error(exc),
            "error": exc,
        }


def build_tasks(
    args: argparse.Namespace,
    queries: list[dict[str, str]],
    platforms: list[str],
    strategies: list[str],
    indexes: dict[str, dict[tuple[str, int], dict[str, Any]]],
    done: set[tuple[str, str, int, str, int]],
) -> list[dict[str, Any]]:
    target_ranks = args.target_rank if args.target_rank else DEFAULT_SOURCE_ORDER
    invalid_ranks = sorted({rank for rank in target_ranks if rank not in DEFAULT_SOURCE_ORDER})
    if invalid_ranks:
        raise ValueError(f"target ranks must be 1..5: {invalid_ranks}")

    tasks: list[dict[str, Any]] = []
    for platform in platforms:
        for query in queries:
            query_id = query["query_id"]
            for strategy in strategies:
                if strategy == "baseline":
                    baseline_ranks = [args.baseline_target_rank] if args.baseline_mode == "once" else target_ranks
                    for target_rank in baseline_ranks:
                        source_rows = build_source_rows(query_id, int(target_rank), strategy, indexes)
                        prompt = build_prompt(query, source_rows, args.source_max_chars)
                        for run_id in range(1, args.repeats + 1):
                            done_key = (platform, query_id, int(target_rank), strategy, run_id)
                            if done_key in done:
                                continue
                            tasks.append(
                                {
                                    "platform": platform,
                                    "query": query,
                                    "target_rank": int(target_rank),
                                    "strategy": strategy,
                                    "run_id": run_id,
                                    "prompt": prompt,
                                }
                            )
                    continue

                for target_rank in target_ranks:
                    source_rows = build_source_rows(query_id, int(target_rank), strategy, indexes)
                    prompt = build_prompt(query, source_rows, args.source_max_chars)
                    for run_id in range(1, args.repeats + 1):
                        done_key = (platform, query_id, int(target_rank), strategy, run_id)
                        if done_key in done:
                            continue
                        tasks.append(
                            {
                                "platform": platform,
                                "query": query,
                                "target_rank": int(target_rank),
                                "strategy": strategy,
                                "run_id": run_id,
                                "prompt": prompt,
                            }
                        )
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    return tasks


def handle_task_result(
    result: dict[str, Any],
    args: argparse.Namespace,
    counters: Counter[str],
    usage_totals: Counter[str],
    consecutive_api_failures: int,
    write_lock: Lock,
) -> tuple[int, bool]:
    status = str(result["status"])
    prefix = (
        f"{result['platform']}/{result['strategy']} {result['query_id']} "
        f"target_rank={result['target_rank']} run_id={result['run_id']}"
    )

    with write_lock:
        append_jsonl(result["output_path"], result["output_row"])
        append_jsonl(Path(args.log_output), result["log_row"])

        api_usage = result.get("api_usage", {})
        for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_totals[token_key] += api_usage.get(token_key, 0)

        counters[status] += 1
        if status in {"failed", "provider_blocked"}:
            print(f"{status}: {prefix}: {result.get('error')}", file=sys.stderr, flush=True)
        else:
            print(f"{status}: {prefix}", flush=True)

    if status in {"failed", "provider_blocked"}:
        consecutive_api_failures += 1
        if status == "provider_blocked":
            return 0, False
        if result.get("fatal"):
            provider = PLATFORM_CONFIGS.get(str(result["platform"]), {}).get("provider", "configured provider")
            print(
                f"aborted: fatal API/account error detected for {result['platform']} ({provider}); "
                f"check the {provider} console or API key owner.",
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
        if status != "success" and args.fail_fast:
            return consecutive_api_failures, True
    return consecutive_api_failures, False


def group_tasks_by_output(tasks: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in tasks:
        key = (str(task["platform"]), str(task["strategy"]))
        grouped.setdefault(key, []).append(task)
    return grouped


def process_task_group(
    group_key: tuple[str, str],
    group_tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    platform_models: dict[str, str],
    counters: Counter[str],
    usage_totals: Counter[str],
    write_lock: Lock,
    abort_event: Event,
) -> None:
    consecutive_api_failures = 0
    platform, strategy = group_key
    for task in group_tasks:
        if abort_event.is_set():
            break
        result = process_answer_task(task, args, platform_models)
        consecutive_api_failures, abort = handle_task_result(
            result, args, counters, usage_totals, consecutive_api_failures, write_lock
        )
        if abort:
            with write_lock:
                print(f"aborting group {platform}/{strategy}", file=sys.stderr, flush=True)
            abort_event.set()
            break


def run_tasks(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    platform_models: dict[str, str],
) -> tuple[Counter[str], Counter[str]]:
    counters: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    write_lock = Lock()
    abort_event = Event()

    grouped = group_tasks_by_output(tasks)
    workers = min(max(1, args.workers), max(1, len(grouped)))

    if workers == 1:
        for group_key, group in grouped.items():
            process_task_group(group_key, group, args, platform_models, counters, usage_totals, write_lock, abort_event)
            if abort_event.is_set():
                break
        return counters, usage_totals

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_task_group,
                group_key,
                group,
                args,
                platform_models,
                counters,
                usage_totals,
                write_lock,
                abort_event,
            )
            for group_key, group in grouped.items()
        ]
        for future in futures:
            future.result()
    return counters, usage_totals


def main() -> int:
    args = parse_args()
    args.workers = max(1, args.workers)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    platforms = split_csv(args.platforms)
    if not platforms:
        raise ValueError("No platforms specified")
    strategies = split_strategies(args.strategies)
    platform_models = parse_platform_models(args.platform_model, args.model)
    queries = filter_queries(args, load_queries(Path(args.queries)))
    indexes = load_rewrite_indexes(Path(args.rewrite_dir), strategies)
    done = set() if args.no_resume or args.dry_run else load_done_keys(Path(args.output_root), platforms, strategies)
    tasks = build_tasks(args, queries, platforms, strategies, indexes, done)

    print(
        f"loaded {len(queries)} queries, {len(platforms)} platforms, {len(strategies)} strategies, "
        f"{len(tasks)} pending answer tasks, output_groups={len(group_tasks_by_output(tasks))}, workers={args.workers}",
        flush=True,
    )

    if args.dry_run:
        for task in tasks[:5]:
            task_platform = str(task["platform"])
            task_model = platform_models.get(task_platform, platform_models["__default__"])
            print(
                f"- {task['platform']}/{task['strategy']} {task['query']['query_id']} "
                f"target_rank={task['target_rank']} run_id={task['run_id']} "
                f"model={task_model} prompt_chars={len(task['prompt'])}",
                flush=True,
            )
        preview_count = max(0, args.preview_prompts)
        preview_chars = max(0, args.preview_prompt_chars)
        for index, task in enumerate(tasks[:preview_count], start=1):
            prompt = task["prompt"] if preview_chars == 0 else task["prompt"][:preview_chars]
            print(
                f"\n--- prompt preview {index}: {task['platform']}/{task['strategy']} "
                f"{task['query']['query_id']} target_rank={task['target_rank']} "
                f"run_id={task['run_id']} ---",
                flush=True,
            )
            print(prompt, flush=True)
            if preview_chars > 0 and len(task["prompt"]) > preview_chars:
                print(f"\n[truncated: {preview_chars}/{len(task['prompt'])} chars]", flush=True)
            print("--- end preview ---", flush=True)
        return 0

    counters, usage_totals = run_tasks(tasks, args, platform_models)
    skipped = 0 if args.no_resume else len(done)
    if skipped:
        counters["skipped_existing"] = skipped

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
