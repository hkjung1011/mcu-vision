from __future__ import annotations

import argparse
import mimetypes
import os
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .common import (
    append_jsonl,
    clean_html,
    load_yaml,
    read_jsonl,
    safe_stem,
    sha256_file,
    utc_now,
    write_json,
)


SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class CommonsClient:
    def __init__(self, api_url: str, user_agent: str, timeout: float, delay: float) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = {"format": "json", "formatversion": 2, **params}
        response = self.session.get(self.api_url, params=merged, timeout=self.timeout)
        response.raise_for_status()
        time.sleep(self.delay)
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"Wikimedia API error: {payload['error']}")
        return payload

    def category_titles(self, category: str, max_depth: int, max_files: int) -> list[str]:
        files: OrderedDict[str, None] = OrderedDict()
        queue: deque[tuple[str, int]] = deque([(category, 0)])
        visited: set[str] = set()

        while queue and len(files) < max_files:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            continuation: dict[str, Any] = {}
            while True:
                payload = self.get_json(
                    {
                        "action": "query",
                        "list": "categorymembers",
                        "cmtitle": current,
                        "cmnamespace": "6|14",
                        "cmlimit": "max",
                        **continuation,
                    }
                )
                for member in payload.get("query", {}).get("categorymembers", []):
                    namespace = member.get("ns")
                    title = member.get("title", "")
                    if namespace == 6:
                        files.setdefault(title, None)
                        if len(files) >= max_files:
                            break
                    elif namespace == 14 and depth < max_depth:
                        queue.append((title, depth + 1))
                if "continue" not in payload or len(files) >= max_files:
                    break
                continuation = payload["continue"]
        return list(files)

    def search_titles(self, query: str, maximum: int) -> list[str]:
        titles: list[str] = []
        continuation: dict[str, Any] = {}
        while len(titles) < maximum:
            payload = self.get_json(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 6,
                    "srlimit": "max",
                    **continuation,
                }
            )
            for item in payload.get("query", {}).get("search", []):
                title = item.get("title")
                if title:
                    titles.append(title)
                    if len(titles) >= maximum:
                        break
            if "continue" not in payload or len(titles) >= maximum:
                break
            continuation = payload["continue"]
        return titles

    def image_info(self, titles: Iterable[str], thumb_width: int) -> Iterable[dict[str, Any]]:
        title_list = list(titles)
        for start in range(0, len(title_list), 50):
            batch = title_list[start : start + 50]
            payload = self.get_json(
                {
                    "action": "query",
                    "prop": "imageinfo",
                    "titles": "|".join(batch),
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": thumb_width,
                }
            )
            yield from payload.get("query", {}).get("pages", [])


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        return clean_html(str(value.get("value", "")))
    return clean_html(str(value))


def license_allowed(license_name: str, allowed_prefixes: list[str]) -> bool:
    normalized = license_name.casefold().replace("_", " ").strip()
    return any(normalized.startswith(prefix.casefold().replace("_", " ")) for prefix in allowed_prefixes)


def keyword_allowed(title: str, class_config: dict[str, Any]) -> bool:
    normalized = title.casefold()
    include_any = [str(value).casefold() for value in class_config.get("include_any", [])]
    exclude_any = [str(value).casefold() for value in class_config.get("exclude_any", [])]
    if include_any and not any(keyword in normalized for keyword in include_any):
        return False
    return not any(keyword in normalized for keyword in exclude_any)


def page_url(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe=":()_-.")


def download_candidate(
    client: CommonsClient,
    image_url: str,
    destination_dir: Path,
    page_id: int,
    title: str,
    max_bytes: int,
) -> tuple[Path, int, int, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination_dir / f".{page_id}.part"
    downloaded = 0
    try:
        with client.session.get(image_url, stream=True, timeout=client.timeout) as response:
            response.raise_for_status()
            length = int(response.headers.get("Content-Length", 0) or 0)
            if length and length > max_bytes:
                raise ValueError(f"content_length_exceeds_limit:{length}")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"download_exceeds_limit:{downloaded}")
                    handle.write(chunk)
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            width, height = image.size
            image_format = image.format or ""
        suffix = FORMAT_SUFFIX.get(image_format.upper())
        if not suffix:
            raise ValueError(f"unsupported_decoded_format:{image_format}")
        title_stem = Path(title.replace("File:", "", 1)).stem
        final_path = destination_dir / f"commons_{page_id}_{safe_stem(title_stem)}{suffix}"
        temporary.replace(final_path)
        return final_path, width, height, image_format.upper()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def collect_class(
    config_path: Path,
    class_name: str,
    limit: int,
    output_root: Path,
    manifest_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    classes = config.get("classes", {})
    if class_name not in classes:
        raise KeyError(f"Unknown class {class_name!r}; configured: {', '.join(sorted(classes))}")
    class_config = classes[class_name]
    download_config = config.get("download", {})
    client = CommonsClient(
        api_url=str(config["api_url"]),
        user_agent=str(config["user_agent"]),
        timeout=float(download_config.get("timeout_seconds", 30)),
        delay=float(download_config.get("request_delay_seconds", 0.2)),
    )

    manifest_path = manifest_root / f"{class_name}.sources.jsonl"
    rejection_path = manifest_root / f"{class_name}.rejections.jsonl"
    summary_path = manifest_root / f"{class_name}.collection-summary.json"
    existing = list(read_jsonl(manifest_path))
    accepted_page_ids = {int(record["source_page_id"]) for record in existing if record.get("source_page_id")}
    accepted_hashes = {str(record["sha256"]) for record in existing if record.get("sha256")}
    accepted_count = sum(
        1
        for record in existing
        if record.get("status") == "ACCEPTED" and (output_root / str(record.get("relative_path", ""))).exists()
    )

    titles: OrderedDict[str, str] = OrderedDict()
    category_depth = int(class_config.get("category_depth", 2))
    discovery_config = config.get("discovery", {})
    multiplier = int(discovery_config.get("candidate_multiplier", 3))
    minimum_candidates = int(discovery_config.get("minimum_candidates_per_route", 100))
    category_maximum = min(
        max(limit * multiplier, minimum_candidates),
        int(discovery_config.get("max_category_files_per_class", 2000)),
    )
    for category in class_config.get("categories", []):
        for title in client.category_titles(
            str(category), max_depth=category_depth, max_files=category_maximum
        ):
            titles.setdefault(title, f"category:{category}")
    search_maximum = min(
        max(limit * multiplier, minimum_candidates),
        int(discovery_config.get("max_search_results_per_query", 750)),
    )
    for query in class_config.get("search_queries", []):
        for title in client.search_titles(str(query), maximum=search_maximum):
            titles.setdefault(title, f"search:{query}")

    stats: dict[str, Any] = {
        "class_name": class_name,
        "requested_total": limit,
        "existing_accepted": accepted_count,
        "discovered_titles": len(titles),
        "new_accepted": 0,
        "rejected": {},
        "dry_run": dry_run,
        "started_at": utc_now(),
    }

    if dry_run:
        stats["candidate_titles_after_keyword_filter"] = sum(
            1 for title in titles if keyword_allowed(title, class_config)
        )
        stats["finished_at"] = utc_now()
        write_json(summary_path, stats)
        return stats

    remaining_titles = [title for title in titles if keyword_allowed(title, class_config)]
    progress = tqdm(total=max(0, limit - accepted_count), desc=class_name, unit="image")
    try:
        for page in client.image_info(remaining_titles, int(download_config.get("max_dimension_px", 2048))):
            if accepted_count >= limit:
                break
            page_id = int(page.get("pageid", 0) or 0)
            title = str(page.get("title", ""))
            if not page_id or page_id in accepted_page_ids:
                continue
            image_info = (page.get("imageinfo") or [{}])[0]
            mime = str(image_info.get("mime", ""))
            metadata = image_info.get("extmetadata", {}) or {}
            license_name = metadata_value(metadata, "LicenseShortName")
            reason = ""
            if mime not in SUPPORTED_MIME_TYPES:
                reason = f"unsupported_mime:{mime}"
            elif not license_allowed(license_name, list(config.get("allowed_license_prefixes", []))):
                reason = f"license_not_allowed:{license_name or 'missing'}"
            elif int(image_info.get("width", 0) or 0) < int(download_config.get("min_width_px", 320)):
                reason = "source_width_too_small"
            elif int(image_info.get("height", 0) or 0) < int(download_config.get("min_height_px", 240)):
                reason = "source_height_too_small"
            image_url = str(image_info.get("thumburl") or image_info.get("url") or "")
            if not image_url:
                reason = reason or "missing_image_url"

            if reason:
                stats["rejected"][reason] = int(stats["rejected"].get(reason, 0)) + 1
                append_jsonl(
                    rejection_path,
                    {
                        "class_name": class_name,
                        "source_page_id": page_id,
                        "source_page_title": title,
                        "source_page_url": page_url(title),
                        "license": license_name,
                        "status": "REJECTED_METADATA",
                        "reason": reason,
                        "checked_at": utc_now(),
                    },
                )
                continue

            try:
                local_path, width, height, decoded_format = download_candidate(
                    client,
                    image_url,
                    output_root / class_name,
                    page_id,
                    title,
                    max_bytes=int(download_config.get("max_file_bytes", 30 * 1024 * 1024)),
                )
                if width < int(download_config.get("min_width_px", 320)) or height < int(
                    download_config.get("min_height_px", 240)
                ):
                    local_path.unlink(missing_ok=True)
                    raise ValueError(f"downloaded_dimensions_too_small:{width}x{height}")
                digest = sha256_file(local_path)
                if digest in accepted_hashes:
                    local_path.unlink(missing_ok=True)
                    raise ValueError("exact_duplicate_sha256")
            except (OSError, requests.RequestException, UnidentifiedImageError, ValueError) as exc:
                reason = f"download_or_decode:{type(exc).__name__}:{exc}"
                stats["rejected"][reason] = int(stats["rejected"].get(reason, 0)) + 1
                append_jsonl(
                    rejection_path,
                    {
                        "class_name": class_name,
                        "source_page_id": page_id,
                        "source_page_title": title,
                        "source_page_url": page_url(title),
                        "status": "REJECTED_DOWNLOAD",
                        "reason": reason,
                        "checked_at": utc_now(),
                    },
                )
                continue

            relative_path = local_path.relative_to(output_root).as_posix()
            record = {
                "class_name": class_name,
                "source_platform": "Wikimedia Commons",
                "source_page_id": page_id,
                "source_page_title": title,
                "source_page_url": page_url(title),
                "image_url": image_url,
                "discovery_route": titles.get(title, ""),
                "license": license_name,
                "license_url": metadata_value(metadata, "LicenseUrl"),
                "usage_terms": metadata_value(metadata, "UsageTerms"),
                "artist": metadata_value(metadata, "Artist"),
                "credit": metadata_value(metadata, "Credit"),
                "description": metadata_value(metadata, "ImageDescription"),
                "attribution_required": metadata_value(metadata, "AttributionRequired"),
                "retrieved_at": utc_now(),
                "relative_path": relative_path,
                "sha256": digest,
                "width": width,
                "height": height,
                "decoded_format": decoded_format,
                "source_mime": mime,
                "bytes": os.path.getsize(local_path),
                "status": "ACCEPTED",
                "qa_status": "PENDING_HUMAN_REVIEW",
            }
            append_jsonl(manifest_path, record)
            accepted_page_ids.add(page_id)
            accepted_hashes.add(digest)
            accepted_count += 1
            stats["new_accepted"] += 1
            progress.update(1)
    finally:
        progress.close()

    stats["accepted_total"] = accepted_count
    stats["target_met"] = accepted_count >= limit
    stats["finished_at"] = utc_now()
    write_json(summary_path, stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect license-traceable images from Wikimedia Commons.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--limit", type=int, default=1800, help="Desired total accepted candidates for this class.")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/wikimedia"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = collect_class(
        config_path=args.config.resolve(),
        class_name=args.class_name,
        limit=args.limit,
        output_root=args.output_root.resolve(),
        manifest_root=args.manifest_root.resolve(),
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    main()
