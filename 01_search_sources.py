#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter

from lxml import html


@dataclass
class SearchResult:
    rank: int
    title: str
    baidu_url: str
    target_url: str
    context_text: str = ""


EXCLUDED_RESULT_MARKERS = ("精选笔记", "精品笔记", "百度百科", "百度知了好学", "知了好学")


def excluded_result_reason(title: str, *urls: str, context_text: str = "") -> str:
    """Return why a Baidu candidate must not count toward Top-K, or an empty string."""
    searchable_text = f"{title} {context_text}"
    for marker in EXCLUDED_RESULT_MARKERS:
        if marker in searchable_text:
            return marker

    for url in urls:
        if not url:
            continue
        hostname = (urllib.parse.urlparse(url).hostname or "").lower()
        if hostname == "baike.baidu.com" or hostname.endswith(".baike.baidu.com"):
            return "百度百科"
        if hostname == "aistudy.baidu.com" or hostname.endswith(".aistudy.baidu.com"):
            return "百度知了好学"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Baidu Top-K results and crawl each result page body."
    )
    parser.add_argument("--queries", default="data/queries.csv")
    parser.add_argument("--config", default="configs/baidu_search.example.json")
    parser.add_argument("--output", default="data/search_results.jsonl")
    parser.add_argument("--debug-output", default="outputs/logs/search_debug.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20, help="Search candidates to inspect before selecting Top-K usable pages.")
    parser.add_argument("--max-search-pages", type=int, default=3, help="Fetch more Baidu result pages when the first page cannot fill Top-K.")
    parser.add_argument("--page-sleep-min", type=float, default=8.0, help="Minimum random sleep seconds between Baidu result pages for one query.")
    parser.add_argument("--page-sleep-max", type=float, default=18.0, help="Maximum random sleep seconds between Baidu result pages for one query.")
    parser.add_argument("--result-sleep-min", type=float, default=1.0, help="Minimum random sleep seconds between fetching candidate result pages.")
    parser.add_argument("--result-sleep-max", type=float, default=3.0, help="Maximum random sleep seconds between fetching candidate result pages.")
    parser.add_argument("--min-body-chars", type=int, default=200, help="Skip fetched pages whose cleaned body is shorter than this.")
    parser.add_argument("--keep-empty", action="store_true", help="Keep results with empty or very short body.")
    parser.add_argument("--write-partial", action="store_true", help="Write fewer than Top-K selected rows when not enough usable pages are found.")
    parser.add_argument("--max-consecutive-blocked", type=int, default=1, help="Stop after this many consecutive Baidu security/captcha pages.")
    parser.add_argument("--max-consecutive-empty", type=int, default=5, help="Stop after this many consecutive queries with zero search candidates.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N queries; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0, help="Deprecated: base sleep seconds between queries. Use --sleep-min/--sleep-max.")
    parser.add_argument("--sleep-min", type=float, default=8.0, help="Minimum random sleep seconds between queries.")
    parser.add_argument("--sleep-max", type=float, default=18.0, help="Maximum random sleep seconds between queries.")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--body-max-chars", type=int, default=12000)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip query_ids already present in output.")
    return parser.parse_args()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not config.get("endpoint"):
        raise ValueError("config missing endpoint")
    if not config.get("query_param"):
        raise ValueError("config missing query_param")
    return config


def load_queries(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"query_id", "domain", "query_zh"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"queries missing columns: {', '.join(sorted(missing))}")
    if limit > 0:
        return rows[:limit]
    return rows


def load_done_query_ids(path: Path, top_k: int) -> set[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("query_id"):
                counts[str(row["query_id"])] += 1
    return {query_id for query_id, count in counts.items() if count >= top_k}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_headers(config: dict[str, Any]) -> dict[str, str]:
    headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
    cookie_env = str(config.get("cookie_env") or "BAIDU_COOKIE")
    cookie = os.environ.get(cookie_env, "").strip()
    if cookie and "Cookie" not in headers:
        headers["Cookie"] = cookie
    headers.setdefault("User-Agent", "Mozilla/5.0")
    # Keep encodings to formats the stdlib can decode reliably.
    headers["Accept-Encoding"] = "gzip"
    return headers


def request_url(url: str, headers: dict[str, str], timeout: float) -> tuple[str, bytes]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            data = gzip.decompress(data)
        return resp.geturl(), data


def decode_html(data: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def build_baidu_url(config: dict[str, Any], query: str, page: int = 0) -> str:
    params = dict(config.get("extra_params") or {})
    params[str(config["query_param"])] = query
    if page > 0:
        params["pn"] = str(page * 10)
    return str(config["endpoint"]) + "?" + urllib.parse.urlencode(params)


def parse_baidu_results(search_html: str, top_k: int) -> list[SearchResult]:
    tree = html.fromstring(search_html)
    containers = tree.xpath(
        '//div[contains(concat(" ", normalize-space(@class), " "), " result ")]'
        '|//div[contains(concat(" ", normalize-space(@class), " "), " c-container ")]'
    )

    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    def target_url_from_container(container: Any) -> str:
        for attr in ("mu", "data-log", "data-mu"):
            value = container.get(attr)
            if value and value.startswith("http"):
                return value

        tools = container.get("data-tools")
        if tools:
            try:
                parsed = json.loads(tools)
            except json.JSONDecodeError:
                parsed = {}
            value = parsed.get("url")
            if isinstance(value, str) and value.startswith("http"):
                return value
        return ""

    def add_candidate(anchor: Any, container: Any | None = None) -> None:
        href = anchor.get("href") or ""
        title = collapse_ws(anchor.text_content())
        if not href.startswith("http") or not title:
            return
        if href in seen_urls:
            return
        seen_urls.add(href)
        target_url = target_url_from_container(container) if container is not None else ""
        context_text = collapse_ws(container.text_content()) if container is not None else title
        results.append(
            SearchResult(
                rank=len(results) + 1,
                title=title,
                baidu_url=href,
                target_url=target_url,
                context_text=context_text,
            )
        )

    for container in containers:
        anchors = container.xpath('.//h3//a[@href][1] | .//a[contains(@class, "result-title")][@href][1]')
        if anchors:
            add_candidate(anchors[0], container)
        if len(results) >= top_k:
            return results[:top_k]

    for anchor in tree.xpath("//h3//a[@href]"):
        add_candidate(anchor)
        if len(results) >= top_k:
            break

    return results[:top_k]


def is_baidu_security_page(final_url: str, search_html: str) -> bool:
    if "wappass.baidu.com" in final_url or "captcha" in final_url:
        return True
    lowered = search_html[:5000].lower()
    return "百度安全验证" in search_html or "网络不给力，请稍后重试" in search_html or "captcha" in lowered


def extract_body(page_html: str, max_chars: int) -> str:
    tree = html.fromstring(page_html)
    for bad in tree.xpath("//script|//style|//noscript|//svg|//canvas|//header|//footer|//nav"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    body_nodes = tree.xpath("//body")
    text = body_nodes[0].text_content() if body_nodes else tree.text_content()
    text = collapse_ws(text)
    if max_chars > 0:
        text = text[:max_chars]
    return text


def crawl_one_query(
    query_row: dict[str, str],
    config: dict[str, Any],
    headers: dict[str, str],
    args: argparse.Namespace,
) -> str:
    query_id = query_row["query_id"]
    query = query_row["query_zh"]

    selected_count = 0
    selected_rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    total_candidates = 0

    for page in range(max(1, args.max_search_pages)):
        search_url = build_baidu_url(config, query, page)
        try:
            final_search_url, search_bytes = request_url(search_url, headers, args.timeout)
            search_html = decode_html(search_bytes)
            if is_baidu_security_page(final_search_url, search_html):
                append_jsonl(args.debug_path, {
                    "query_id": query_id,
                    "query": query,
                    "page": page + 1,
                    "stage": "search",
                    "status": "blocked",
                    "final_url": final_search_url,
                    "error": "Baidu security verification",
                })
                return "blocked"
            candidate_k = max(args.candidate_k, args.top_k)
            results = parse_baidu_results(search_html, candidate_k)
        except Exception as exc:
            append_jsonl(args.debug_path, {
                "query_id": query_id,
                "query": query,
                "page": page + 1,
                "stage": "search",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            if page == 0:
                return "search_failed"
            continue

        total_candidates += len(results)

        for result in results:
            if selected_count >= args.top_k:
                break
            candidate_key = result.target_url or result.baidu_url or result.title
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            prefetch_exclusion = excluded_result_reason(
                result.title,
                result.baidu_url,
                result.target_url,
                context_text=result.context_text,
            )
            if prefetch_exclusion:
                append_jsonl(args.debug_path, {
                    "query_id": query_id,
                    "page": page + 1,
                    "search_rank": page * 10 + result.rank,
                    "selected_rank": None,
                    "selected": False,
                    "title": result.title,
                    "baidu_url": result.baidu_url,
                    "target_url": result.target_url,
                    "fetch_status": "excluded",
                    "excluded_reason": prefetch_exclusion,
                    "body_chars": 0,
                    "error": "",
                })
                continue

            final_url = ""
            body = ""
            fetch_status = "success"
            fetch_error = ""
            fetch_url = result.target_url or result.baidu_url
            try:
                final_url, page_bytes = request_url(fetch_url, headers, args.timeout)
                body = extract_body(decode_html(page_bytes), args.body_max_chars)
            except Exception as exc:
                fetch_status = "failed"
                fetch_error = f"{type(exc).__name__}: {exc}"

            postfetch_exclusion = excluded_result_reason(
                result.title,
                result.baidu_url,
                result.target_url,
                final_url,
                context_text=result.context_text,
            )
            selected = not postfetch_exclusion and (
                args.keep_empty or (fetch_status == "success" and len(body) >= args.min_body_chars)
            )
            if selected:
                selected_count += 1
                selected_rows.append({
                    "query_id": query_id,
                    "rank": selected_count,
                    "title": result.title,
                    "body": body,
                })
            append_jsonl(args.debug_path, {
                "query_id": query_id,
                "page": page + 1,
                "search_rank": page * 10 + result.rank,
                "selected_rank": selected_count if selected else None,
                "selected": selected,
                "title": result.title,
                "baidu_url": result.baidu_url,
                "target_url": result.target_url,
                "fetch_url": fetch_url,
                "final_url": final_url,
                "fetch_status": fetch_status,
                "body_chars": len(body),
                "excluded_reason": postfetch_exclusion,
                "error": fetch_error,
            })
            if selected_count < args.top_k:
                low = max(0.0, min(args.result_sleep_min, args.result_sleep_max))
                high = max(low, max(args.result_sleep_min, args.result_sleep_max))
                if high > 0:
                    time.sleep(random.uniform(low, high))

        if selected_count >= args.top_k:
            break
        if page < max(1, args.max_search_pages) - 1:
            low = max(0.0, min(args.page_sleep_min, args.page_sleep_max))
            high = max(low, max(args.page_sleep_min, args.page_sleep_max))
            time.sleep(random.uniform(low, high))

    if selected_rows and (selected_count >= args.top_k or args.write_partial):
        for row in selected_rows:
            append_jsonl(Path(args.output), row)

    if selected_count < args.top_k:
        append_jsonl(args.debug_path, {
            "query_id": query_id,
            "query": query,
            "stage": "select",
            "status": "not_enough_results",
            "selected": selected_count,
            "requested": args.top_k,
            "candidates": total_candidates,
            "pages": max(1, args.max_search_pages),
        })
        return "empty" if total_candidates == 0 else "not_enough"

    return "ok"


def main() -> int:
    args = parse_args()
    args.debug_path = Path(args.debug_output)
    config = load_config(Path(args.config))
    headers = make_headers(config)
    queries = load_queries(Path(args.queries), args.limit)
    done = set() if args.no_resume else load_done_query_ids(Path(args.output), args.top_k)

    total = len(queries)
    consecutive_blocked = 0
    consecutive_empty = 0
    for index, row in enumerate(queries, start=1):
        query_id = row["query_id"]
        if query_id in done:
            print(f"[{index}/{total}] skip {query_id} (already in output)", flush=True)
            continue
        print(f"[{index}/{total}] search {query_id}: {row['query_zh']}", flush=True)
        status = crawl_one_query(row, config, headers, args)
        if status == "blocked":
            consecutive_blocked += 1
        else:
            consecutive_blocked = 0

        if status == "empty":
            consecutive_empty += 1
        elif status != "blocked":
            consecutive_empty = 0

        if args.max_consecutive_blocked > 0 and consecutive_blocked >= args.max_consecutive_blocked:
            print(f"stop: Baidu security verification after {query_id}", flush=True)
            break
        if args.max_consecutive_empty > 0 and consecutive_empty >= args.max_consecutive_empty:
            print(f"stop: {consecutive_empty} consecutive empty search result pages after {query_id}", flush=True)
            break
        if args.sleep and args.sleep > 0:
            sleep_seconds = args.sleep + random.uniform(0, args.sleep * 0.4)
        else:
            low = max(0.0, min(args.sleep_min, args.sleep_max))
            high = max(low, max(args.sleep_min, args.sleep_max))
            sleep_seconds = random.uniform(low, high)
        print(f"sleep {sleep_seconds:.1f}s", flush=True)
        time.sleep(sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
