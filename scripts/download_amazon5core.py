"""Download Amazon Reviews 2023 Leave-Last-Out 5-Core data files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import requests
from requests import RequestException
from tqdm import tqdm


DEFAULT_CATEGORIES = (
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
)

DEFAULT_SPLITS = ("train", "valid", "test")
CONFIRMED_BAD_STATUS_CODES = {403, 404, 410}

SPLIT_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "benchmark/5core/last_out_w_his/{category}.{split}.csv.gz"
)

REVIEW_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/{category}.jsonl.gz"
)

META_URL_TEMPLATE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_{category}.jsonl.gz"
)


@dataclass(frozen=True)
class DownloadTask:
    category: str
    label: str
    url: str
    output_path: Path
    required: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "下载 Amazon Reviews 2023 Leave-Last-Out 5-Core 数据。"
            "默认只包含 train/valid/test，不下载 review/meta 大文件。"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="原始数据保存目录，默认: data/raw",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
        help="需要下载的商品类别，支持一个或多个类别，也支持 all；默认三个类别全部。",
    )
    parser.add_argument(
        "--include-review-meta",
        action="store_true",
        help="同时下载 review 和 meta 原始辅助文件；默认不下载。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS,
        default=list(DEFAULT_SPLITS),
        help="需要下载的 Leave-Last-Out 划分文件，默认: train valid test。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如果目标文件已存在，仍重新下载并覆盖。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印下载计划，不访问网络、不写入文件。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="单次网络请求超时时间，单位秒，默认: 30。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="流式下载分块大小，默认: 1048576 (1MB)。",
    )
    parser.add_argument(
        "--check-url",
        action="store_true",
        help="只做 URL 轻量可访问性检查，不下载完整文件。",
    )
    return parser.parse_args()


def normalize_categories(categories: Iterable[str]) -> list[str]:
    requested = list(categories)
    if any(category.lower() == "all" for category in requested):
        return list(DEFAULT_CATEGORIES)

    unknown = sorted(set(requested) - set(DEFAULT_CATEGORIES))
    if unknown:
        valid_values = ", ".join((*DEFAULT_CATEGORIES, "all"))
        raise ValueError(f"未知类别: {', '.join(unknown)}。可选值: {valid_values}")

    return requested


def build_tasks(args: argparse.Namespace) -> list[DownloadTask]:
    tasks: list[DownloadTask] = []
    categories = normalize_categories(args.categories)

    for category in categories:
        category_dir = args.data_root / category

        for split in args.splits:
            tasks.append(
                DownloadTask(
                    category=category,
                    label=split,
                    url=SPLIT_URL_TEMPLATE.format(category=category, split=split),
                    output_path=category_dir / f"{category}.{split}.csv.gz",
                    required=True,
                )
            )

        if args.include_review_meta:
            tasks.extend(
                [
                    DownloadTask(
                        category=category,
                        label="review",
                        url=REVIEW_URL_TEMPLATE.format(category=category),
                        output_path=category_dir / f"{category}.jsonl.gz",
                        required=False,
                    ),
                    DownloadTask(
                        category=category,
                        label="meta",
                        url=META_URL_TEMPLATE.format(category=category),
                        output_path=category_dir / f"meta_{category}.jsonl.gz",
                        required=False,
                    ),
                ]
            )

    return tasks


def format_size(byte_count: int) -> str:
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{byte_count} B"


def print_plan(tasks: list[DownloadTask], *, force: bool) -> None:
    print("Amazon Reviews 2023 5-Core 下载计划")
    print(f"任务数: {len(tasks)}")
    print(f"已存在文件策略: {'覆盖下载' if force else '跳过'}")

    for task in tasks:
        kind = "required" if task.required else "optional"
        exists = "exists" if task.output_path.exists() else "missing"
        print(f"\n[{task.category}] {task.label} ({kind}, {exists})")
        print(f"  URL: {task.url}")
        print(f"  OUT: {task.output_path}")


def check_url(task: DownloadTask, timeout: float) -> str:
    headers = {"User-Agent": "recommendation-coursework-downloader/1.0"}
    head_bad_status: int | None = None
    print(f"\n[{task.category}] {task.label}")
    print(f"  URL: {task.url}")

    try:
        response = requests.head(
            task.url,
            allow_redirects=True,
            timeout=timeout,
            headers=headers,
        )
        if response.ok:
            length = response.headers.get("Content-Length")
            size_text = format_size(int(length)) if length and length.isdigit() else "unknown"
            print(f"  HEAD OK: HTTP {response.status_code}, size={size_text}")
            return "ok"

        if response.status_code in CONFIRMED_BAD_STATUS_CODES:
            head_bad_status = response.status_code
            print(
                "  HEAD 返回明确失败状态: "
                f"HTTP {response.status_code}; 尝试 Range GET 复核。"
            )
        else:
            print(
                "  HEAD 未确认可用: "
                f"HTTP {response.status_code}; 尝试 Range GET 轻量检查。"
            )
        response.close()
    except RequestException as exc:
        print(f"  HEAD 请求失败: {exc}; 尝试 Range GET 轻量检查。")

    try:
        response = requests.get(
            task.url,
            stream=True,
            timeout=timeout,
            headers={**headers, "Range": "bytes=0-0"},
        )
        if response.status_code in {200, 206}:
            response.close()
            print(f"  Range GET OK: HTTP {response.status_code}")
            return "ok"

        if response.status_code in CONFIRMED_BAD_STATUS_CODES:
            status_code = response.status_code
            response.close()
            print(f"  Range GET 明确不可访问: HTTP {status_code}")
            return "bad"

        status_code = response.status_code
        response.close()
        if head_bad_status is not None:
            print(
                "  URL 轻量检查未确认可访问性，但这不等于真实下载失败，"
                "可尝试真实 GET 下载。"
            )
            return "unknown"

        print(f"  Range GET 未确认可用: HTTP {status_code}")
        print(
            "  URL 轻量检查未确认可访问性，但这不等于真实下载失败，"
            "可尝试真实 GET 下载。"
        )
        return "unknown"
    except RequestException as exc:
        print(f"  Range GET 请求失败: {exc}")
        if head_bad_status is not None:
            print(f"  HEAD 曾返回明确失败状态: HTTP {head_bad_status}")
            return "bad"

        print(
            "  URL 轻量检查未确认可访问性，但这不等于真实下载失败，"
            "可尝试真实 GET 下载。"
        )
        return "unknown"


def download_task(task: DownloadTask, *, force: bool, timeout: float, chunk_size: int) -> str:
    target = task.output_path
    part_path = target.with_name(f"{target.name}.part")

    if target.exists() and not force:
        print(f"[SKIP] {target} 已存在")
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    if part_path.exists():
        part_path.unlink()

    headers = {"User-Agent": "recommendation-coursework-downloader/1.0"}
    print(f"[GET] {task.url}")
    print(f"      -> {target}")

    try:
        with requests.get(
            task.url,
            stream=True,
            timeout=timeout,
            headers=headers,
        ) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", "0") or 0)

            with part_path.open("wb") as file_obj:
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=target.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        file_obj.write(chunk)
                        progress.update(len(chunk))

        part_path.replace(target)
        print(f"[DONE] {target} ({format_size(target.stat().st_size)})")
        return "downloaded"
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise


def main() -> int:
    args = parse_args()

    if args.chunk_size <= 0:
        print("错误: --chunk-size 必须大于 0", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("错误: --timeout 必须大于 0", file=sys.stderr)
        return 2

    try:
        tasks = build_tasks(args)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_plan(tasks, force=args.force)
        print("\nDRY-RUN: 未访问网络，未下载或写入任何文件。")
        return 0

    if args.check_url:
        print("Amazon Reviews 2023 5-Core URL 轻量检查")
        print("说明: HEAD 失败不一定代表 URL 不可用；会回退到 Range GET。")
        counts = {"ok": 0, "unknown": 0, "bad": 0}
        for task in tasks:
            result = check_url(task, args.timeout)
            counts[result] += 1
        print(
            "\nURL 检查完成: "
            f"OK={counts['ok']}, 未确认={counts['unknown']}, "
            f"明确不可访问={counts['bad']}, 总计={len(tasks)}"
        )
        if counts["unknown"] > 0:
            print(
                "警告: URL 轻量检查未确认可访问性，但这不等于真实下载失败，"
                "可尝试真实 GET 下载。"
            )
        return 1 if counts["bad"] > 0 else 0

    print_plan(tasks, force=args.force)
    print("\n开始下载。默认不会包含 review/meta，除非显式传入 --include-review-meta。")

    counts = {"downloaded": 0, "skipped": 0}
    for task in tasks:
        result = download_task(
            task,
            force=args.force,
            timeout=args.timeout,
            chunk_size=args.chunk_size,
        )
        counts[result] += 1

    print(
        "\n下载完成: "
        f"downloaded={counts['downloaded']}, skipped={counts['skipped']}, total={len(tasks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
