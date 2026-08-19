from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
import uuid
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote, urljoin, urlsplit

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .common import clean_html, load_yaml, safe_stem, sha256_file, utc_now


COLLECTOR_SCHEMA_VERSION = 2
SHA_ALGORITHM = "sha256"
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
API_HOST_ALLOWLIST = frozenset({"commons.wikimedia.org"})
DOWNLOAD_HOST_ALLOWLIST = frozenset({"commons.wikimedia.org", "upload.wikimedia.org"})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5
MAX_API_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_PIXELS = 25_000_000
DEFAULT_MAX_DECODED_DIMENSION = 8192
DEFAULT_MAX_FRAMES = 1
CLASS_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SHA1_HEX_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MEDIAWIKI_BASE36_SHA1_PATTERN = re.compile(r"[0-9a-z]{31}\Z")
MEDIAWIKI_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


def _collector_version() -> str:
    try:
        return version("mcu-data-tools")
    except PackageNotFoundError:
        return "uninstalled"


def validate_class_name(class_name: str) -> str:
    if not CLASS_NAME_PATTERN.fullmatch(class_name):
        raise ValueError(
            "class_name must contain only ASCII letters, digits, '_' or '-', "
            "must start with a letter or digit, and be at most 64 characters"
        )
    return class_name


def validate_wikimedia_url(url: str, allowed_hosts: frozenset[str], label: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{label}:missing_url")
    parsed = urlsplit(url.strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label}:invalid_port") from exc
    host = (parsed.hostname or "").rstrip(".").casefold()
    if parsed.scheme.casefold() != "https":
        raise ValueError(f"{label}:https_required")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label}:credentials_forbidden")
    if port not in (None, 443):
        raise ValueError(f"{label}:port_not_allowed:{port}")
    if host not in allowed_hosts:
        raise ValueError(f"{label}:host_not_allowed:{host or 'missing'}")
    if parsed.fragment:
        raise ValueError(f"{label}:fragment_forbidden")
    if not parsed.path.startswith("/"):
        raise ValueError(f"{label}:invalid_path")
    return url.strip()


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _assert_no_reparse_components(path: Path) -> Path:
    absolute = _absolute_without_resolving(path)
    parts = absolute.parts
    if not parts:
        raise ValueError("empty_path")
    current = Path(parts[0])
    if _is_reparse_or_symlink(current):
        raise ValueError(f"reparse_or_symlink_component:{current}")
    for part in parts[1:]:
        current = current / part
        if _is_reparse_or_symlink(current):
            raise ValueError(f"reparse_or_symlink_component:{current}")
    return absolute


def _secure_mkdir(path: Path) -> Path:
    absolute = _assert_no_reparse_components(path)
    absolute.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(absolute)
    if not absolute.is_dir():
        raise ValueError(f"not_a_directory:{absolute}")
    return absolute


def _safe_child(root: Path, *parts: str) -> Path:
    absolute_root = _absolute_without_resolving(root)
    candidate = _absolute_without_resolving(absolute_root.joinpath(*parts))
    try:
        candidate.relative_to(absolute_root)
    except ValueError as exc:
        raise ValueError(f"path_escape:{candidate}") from exc
    _assert_no_reparse_components(absolute_root)
    _assert_no_reparse_components(candidate)
    return candidate


def _safe_manifest_path(root: Path, class_name: str, suffix: str) -> Path:
    validate_class_name(class_name)
    return _safe_child(root, f"{class_name}.{suffix}")


def _safe_record_path(output_root: Path, class_name: str, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise ValueError(f"unsafe_relative_path:{relative_path!r}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe_relative_path:{relative_path!r}")
    if not pure.parts or pure.parts[0] != class_name:
        raise ValueError(f"relative_path_wrong_class:{relative_path!r}")
    return _safe_child(output_root, *pure.parts)


def _atomic_append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = _absolute_without_resolving(path)
    _secure_mkdir(path.parent)
    _assert_no_reparse_components(path)
    if path.exists() and not path.is_file():
        raise ValueError(f"manifest_not_regular_file:{path}")
    temporary = _safe_child(path.parent, f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as destination:
            if path.exists():
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                    if source.tell() > 0:
                        source.seek(-1, os.SEEK_END)
                        if source.read(1) != b"\n":
                            destination.write(b"\n")
            encoded = (
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        _assert_no_reparse_components(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    path = _absolute_without_resolving(path)
    _secure_mkdir(path.parent)
    _assert_no_reparse_components(path)
    if path.exists() and not path.is_file():
        raise ValueError(f"json_path_not_regular_file:{path}")
    temporary = _safe_child(path.parent, f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _assert_no_reparse_components(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    path = _absolute_without_resolving(path)
    _assert_no_reparse_components(path)
    if not path.exists():
        return []
    if not path.is_file():
        raise ValueError(f"manifest_not_regular_file:{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_record_not_object:{path}:{line_number}")
            records.append(value)
    return records


class CommonsClient:
    def __init__(self, api_url: str, user_agent: str, timeout: float, delay: float) -> None:
        self.api_url = validate_wikimedia_url(api_url, API_HOST_ALLOWLIST, "api_url")
        if not user_agent.strip():
            raise ValueError("user_agent_required")
        if timeout <= 0:
            raise ValueError("timeout_must_be_positive")
        if delay < 0:
            raise ValueError("delay_must_be_nonnegative")
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

    def _get_with_allowed_redirects(
        self,
        url: str,
        *,
        allowed_hosts: frozenset[str],
        label: str,
        params: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> requests.Response:
        current = validate_wikimedia_url(url, allowed_hosts, label)
        current_params = params
        for redirect_count in range(MAX_REDIRECTS + 1):
            response = self.session.get(
                current,
                params=current_params,
                stream=stream,
                timeout=self.timeout,
                allow_redirects=False,
            )
            current_params = None
            effective_request_url = str(getattr(response, "url", "") or current)
            try:
                validate_wikimedia_url(effective_request_url, allowed_hosts, f"{label}_effective")
                if response.status_code in REDIRECT_STATUS_CODES:
                    if redirect_count >= MAX_REDIRECTS:
                        raise ValueError(f"{label}:too_many_redirects")
                    location = response.headers.get("Location", "")
                    if not location:
                        raise ValueError(f"{label}:redirect_missing_location")
                    current = validate_wikimedia_url(
                        urljoin(effective_request_url, location),
                        allowed_hosts,
                        f"{label}_redirect",
                    )
                    response.close()
                    continue
                if 300 <= response.status_code < 400:
                    raise ValueError(f"{label}:unsupported_redirect_status:{response.status_code}")
                return response
            except Exception:
                response.close()
                raise
        raise AssertionError("redirect loop terminated unexpectedly")

    def get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        merged = {"format": "json", "formatversion": 2, **params}
        response = self._get_with_allowed_redirects(
            self.api_url,
            allowed_hosts=API_HOST_ALLOWLIST,
            label="api",
            params=merged,
            stream=True,
        )
        with response:
            response.raise_for_status()
            content_type = _normalized_content_type(response.headers.get("Content-Type", ""))
            if content_type not in {"application/json", "application/problem+json"}:
                raise ValueError(f"api:unexpected_content_type:{content_type or 'missing'}")
            declared_length = _content_length(response.headers.get("Content-Length"), "api")
            if declared_length is not None and declared_length > MAX_API_RESPONSE_BYTES:
                raise ValueError(f"api:content_length_exceeds_limit:{declared_length}")
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                raw.extend(chunk)
                if len(raw) > MAX_API_RESPONSE_BYTES:
                    raise ValueError(f"api:response_exceeds_limit:{len(raw)}")
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                raise ValueError("api:invalid_json") from exc
        time.sleep(self.delay)
        if not isinstance(payload, dict):
            raise ValueError("api:response_not_object")
        if "error" in payload:
            raise RuntimeError(f"Wikimedia API error: {payload['error']}")
        return payload

    def open_download(self, image_url: str) -> requests.Response:
        return self._get_with_allowed_redirects(
            image_url,
            allowed_hosts=DOWNLOAD_HOST_ALLOWLIST,
            label="download",
            stream=True,
        )

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
                    "prop": "info|imageinfo",
                    "titles": "|".join(batch),
                    "iiprop": "url|size|mime|timestamp|sha1|extmetadata",
                    "iilimit": 1,
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
    restriction_tokens = set(re.split(r"[^a-z0-9]+", normalized))
    if restriction_tokens.intersection({"nc", "nd", "noncommercial", "noderivatives"}):
        return False
    if "non-commercial" in normalized or "no derivatives" in normalized:
        return False
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


def page_revision_url(title: str, revision_id: int) -> str:
    if revision_id <= 0:
        raise ValueError("page_revision_id_must_be_positive")
    return f"{page_url(title)}?oldid={revision_id}"


def _normalized_content_type(value: str) -> str:
    return value.split(";", 1)[0].strip().casefold()


def _content_length(raw_value: Any, label: str) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        length = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}:invalid_content_length:{raw_value}") from exc
    if length < 0:
        raise ValueError(f"{label}:invalid_content_length:{length}")
    return length


def _magic_mime(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(16)
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _valid_metadata_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and port in {None, 80, 443}
    )


def _source_metadata_snapshot(page: dict[str, Any], image_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title"),
        "lastrevid": page.get("lastrevid"),
        "imageinfo": image_info,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_mediawiki_sha1(raw_value: Any) -> str:
    value = str(raw_value or "").strip().casefold()
    if SHA1_HEX_PATTERN.fullmatch(value):
        return value
    if MEDIAWIKI_BASE36_SHA1_PATTERN.fullmatch(value):
        numeric = int(value, 36)
        if numeric < 2**160:
            return f"{numeric:040x}"
    raise ValueError("invalid_source_image_sha1")


def _validate_mediawiki_timestamp(raw_value: Any) -> str:
    value = str(raw_value or "").strip()
    if not MEDIAWIKI_TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError("invalid_source_image_timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("invalid_source_image_timestamp") from exc
    return value


def _stable_source_revision(page: dict[str, Any], image_info: dict[str, Any]) -> tuple[int, str, str]:
    try:
        page_revision = int(page.get("lastrevid", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_source_page_revision") from exc
    if page_revision <= 0:
        raise ValueError("invalid_source_page_revision")
    image_timestamp = _validate_mediawiki_timestamp(image_info.get("timestamp"))
    image_sha1 = _normalize_mediawiki_sha1(image_info.get("sha1"))
    return page_revision, image_timestamp, image_sha1


def _verify_source_unchanged_after_download(
    client: CommonsClient,
    *,
    title: str,
    thumb_width: int,
    expected_page_id: int,
    expected_page_revision: int,
    expected_image_timestamp: str,
    expected_image_sha1: str,
    expected_snapshot_sha256: str,
) -> None:
    pages = list(client.image_info([title], thumb_width))
    if len(pages) != 1:
        raise ValueError(f"source_mutated_during_download:requery_page_count:{len(pages)}")
    page = pages[0]
    try:
        page_id = int(page.get("pageid", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_mutated_during_download:invalid_page_id") from exc
    current_title = str(page.get("title", ""))
    image_infos = page.get("imageinfo") or []
    if len(image_infos) != 1 or not isinstance(image_infos[0], dict):
        raise ValueError("source_mutated_during_download:invalid_imageinfo")
    image_info = image_infos[0]
    try:
        page_revision, image_timestamp, image_sha1 = _stable_source_revision(page, image_info)
    except ValueError as exc:
        raise ValueError(f"source_mutated_during_download:{exc}") from exc
    snapshot_sha256 = _canonical_sha256(_source_metadata_snapshot(page, image_info))
    comparisons = {
        "page_id": (expected_page_id, page_id),
        "title": (title, current_title),
        "page_revision": (expected_page_revision, page_revision),
        "image_timestamp": (expected_image_timestamp, image_timestamp),
        "image_sha1": (expected_image_sha1, image_sha1),
        "metadata_snapshot_sha256": (expected_snapshot_sha256, snapshot_sha256),
    }
    mismatches = [field for field, (expected, actual) in comparisons.items() if expected != actual]
    if mismatches:
        raise ValueError(f"source_mutated_during_download:{','.join(mismatches)}")


def _license_fields(
    metadata: dict[str, Any], allowed_prefixes: list[str]
) -> tuple[dict[str, str], str]:
    values = {
        "license": metadata_value(metadata, "LicenseShortName"),
        "license_url": metadata_value(metadata, "LicenseUrl"),
        "usage_terms": metadata_value(metadata, "UsageTerms"),
        "artist": metadata_value(metadata, "Artist"),
        "credit": metadata_value(metadata, "Credit"),
        "description": metadata_value(metadata, "ImageDescription"),
        "attribution_required": metadata_value(metadata, "AttributionRequired"),
    }
    missing = [key for key in ("license", "license_url", "usage_terms") if not values[key]]
    if missing:
        return values, f"missing_license_metadata:{','.join(missing)}"
    if not (values["artist"] or values["credit"]):
        return values, "missing_license_metadata:artist_or_credit"
    if not _valid_metadata_url(values["license_url"]):
        return values, "invalid_license_url"
    if not license_allowed(values["license"], allowed_prefixes):
        return values, f"license_not_allowed:{values['license']}"
    return values, ""


@dataclass(frozen=True)
class DownloadedCandidate:
    staging_path: Path
    final_path: Path
    width: int
    height: int
    frame_count: int
    decoded_format: str
    content_mime: str
    byte_count: int
    sha256: str
    image_url_original: str
    image_url_effective: str


def download_candidate(
    client: CommonsClient,
    image_url: str,
    destination_dir: Path,
    page_id: int,
    title: str,
    max_bytes: int,
    *,
    expected_mime: str = "",
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_dimension_px: int = DEFAULT_MAX_DECODED_DIMENSION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> DownloadedCandidate:
    if page_id <= 0:
        raise ValueError("invalid_page_id")
    if max_bytes <= 0 or max_pixels <= 0 or max_dimension_px <= 0 or max_frames <= 0:
        raise ValueError("download_limits_must_be_positive")
    original_url = validate_wikimedia_url(image_url, DOWNLOAD_HOST_ALLOWLIST, "image_url")
    destination_dir = _secure_mkdir(destination_dir)
    temporary = _safe_child(destination_dir, f".{page_id}.{uuid.uuid4().hex}.part")
    downloaded = 0
    try:
        response = client.open_download(original_url)
        with response:
            response.raise_for_status()
            effective_url = validate_wikimedia_url(
                str(getattr(response, "url", "") or original_url),
                DOWNLOAD_HOST_ALLOWLIST,
                "image_url_effective",
            )
            content_mime = _normalized_content_type(response.headers.get("Content-Type", ""))
            if content_mime not in SUPPORTED_MIME_TYPES:
                raise ValueError(f"unsupported_content_type:{content_mime or 'missing'}")
            if expected_mime and content_mime != _normalized_content_type(expected_mime):
                raise ValueError(f"source_content_mime_mismatch:{expected_mime}:{content_mime}")
            length = _content_length(response.headers.get("Content-Length"), "download")
            if length is not None and length > max_bytes:
                raise ValueError(f"content_length_exceeds_limit:{length}")
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(f"download_exceeds_limit:{downloaded}")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if downloaded == 0:
            raise ValueError("empty_download")
        magic_mime = _magic_mime(temporary)
        if not magic_mime:
            raise ValueError("unsupported_or_missing_image_magic")
        if magic_mime != content_mime:
            raise ValueError(f"magic_mime_mismatch:{magic_mime}:{content_mime}")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(temporary) as image:
                width, height = image.size
                image_format = (image.format or "").upper()
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                if width <= 0 or height <= 0:
                    raise ValueError(f"invalid_decoded_dimensions:{width}x{height}")
                if width > max_dimension_px or height > max_dimension_px:
                    raise ValueError(f"decoded_dimension_exceeds_limit:{width}x{height}")
                if width * height > max_pixels:
                    raise ValueError(f"decoded_pixels_exceed_limit:{width * height}")
                if frame_count > max_frames:
                    raise ValueError(f"frame_count_exceeds_limit:{frame_count}")
                image.seek(0)
                image.load()
            with Image.open(temporary) as image:
                image.verify()
        suffix = FORMAT_SUFFIX.get(image_format)
        decoded_mime = FORMAT_MIME.get(image_format)
        if not suffix or not decoded_mime:
            raise ValueError(f"unsupported_decoded_format:{image_format}")
        if decoded_mime != magic_mime:
            raise ValueError(f"decoded_magic_mime_mismatch:{decoded_mime}:{magic_mime}")
        title_stem = Path(title.replace("File:", "", 1)).stem
        final_path = _safe_child(destination_dir, f"commons_{page_id}_{safe_stem(title_stem)}{suffix}")
        digest = sha256_file(temporary)
        return DownloadedCandidate(
            staging_path=temporary,
            final_path=final_path,
            width=width,
            height=height,
            frame_count=frame_count,
            decoded_format=image_format,
            content_mime=content_mime,
            byte_count=downloaded,
            sha256=digest,
            image_url_original=original_url,
            image_url_effective=effective_url,
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _publish_candidate(candidate: DownloadedCandidate, manifest_path: Path, record: dict[str, Any]) -> None:
    _assert_no_reparse_components(candidate.staging_path)
    _assert_no_reparse_components(candidate.final_path)
    if candidate.final_path.exists():
        raise ValueError(f"resume_collision:unmanifested_destination:{candidate.final_path.name}")
    linked = False
    try:
        os.link(candidate.staging_path, candidate.final_path)
        linked = True
        if sha256_file(candidate.final_path) != candidate.sha256:
            raise ValueError("published_file_hash_changed")
        _atomic_append_jsonl(manifest_path, record)
    except Exception:
        if linked and candidate.final_path.exists():
            try:
                if sha256_file(candidate.final_path) == candidate.sha256:
                    candidate.final_path.unlink()
            except OSError:
                pass
        raise
    finally:
        candidate.staging_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ResumeState:
    accepted_page_ids: frozenset[int]
    accepted_hashes: frozenset[str]
    accepted_count: int
    legacy_records: int


def _validate_resume_manifest(manifest_path: Path, output_root: Path, class_name: str) -> ResumeState:
    page_ids: set[int] = set()
    hashes: set[str] = set()
    relative_paths: set[str] = set()
    manifested_paths: set[Path] = set()
    legacy_records = 0
    for index, record in enumerate(_read_jsonl_strict(manifest_path), start=1):
        if record.get("status") != "ACCEPTED":
            raise ValueError(f"resume_manifest_unexpected_status:{index}:{record.get('status')}")
        if record.get("class_name") != class_name:
            raise ValueError(f"resume_manifest_class_mismatch:{index}")
        if record.get("qa_status") != "PENDING_HUMAN_REVIEW":
            raise ValueError(f"resume_manifest_qa_status_mismatch:{index}")
        try:
            page_id = int(record.get("source_page_id", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resume_manifest_invalid_page_id:{index}") from exc
        digest = str(record.get("sha256", "")).casefold()
        relative_path = str(record.get("relative_path", ""))
        if page_id <= 0:
            raise ValueError(f"resume_manifest_invalid_page_id:{index}")
        if not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"resume_manifest_invalid_sha256:{index}")
        algorithm = str(record.get("sha_algorithm", SHA_ALGORITHM)).casefold()
        if algorithm != SHA_ALGORITHM:
            raise ValueError(f"resume_manifest_wrong_hash_algorithm:{index}:{algorithm}")
        if page_id in page_ids:
            raise ValueError(f"resume_collision:duplicate_page_id:{page_id}")
        if digest in hashes:
            raise ValueError(f"resume_collision:duplicate_sha256:{digest}")
        if relative_path in relative_paths:
            raise ValueError(f"resume_collision:duplicate_relative_path:{relative_path}")
        local_path = _safe_record_path(output_root, class_name, relative_path)
        if not local_path.exists() or not local_path.is_file():
            raise ValueError(f"resume_manifest_file_missing:{relative_path}")
        actual_digest = sha256_file(local_path)
        if actual_digest != digest:
            raise ValueError(f"resume_manifest_hash_mismatch:{relative_path}")
        if record.get("bytes") is not None and int(record["bytes"]) != local_path.stat().st_size:
            raise ValueError(f"resume_manifest_size_mismatch:{relative_path}")
        schema_version = int(record.get("collector_schema_version", 1) or 1)
        if schema_version > COLLECTOR_SCHEMA_VERSION:
            raise ValueError(f"resume_manifest_newer_schema_not_supported:{index}:{schema_version}")
        if schema_version == COLLECTOR_SCHEMA_VERSION:
            config_digest = str(record.get("collector_config_sha256", "")).casefold()
            metadata_digest = str(record.get("source_metadata_snapshot_sha256", "")).casefold()
            if not SHA256_PATTERN.fullmatch(config_digest):
                raise ValueError(f"resume_manifest_missing_config_hash:{index}")
            if str(record.get("collector_config_sha_algorithm", "")).casefold() != SHA_ALGORITHM:
                raise ValueError(f"resume_manifest_wrong_config_hash_algorithm:{index}")
            if not SHA256_PATTERN.fullmatch(metadata_digest):
                raise ValueError(f"resume_manifest_missing_metadata_hash:{index}")
            if (
                str(record.get("source_metadata_snapshot_sha_algorithm", "")).casefold()
                != SHA_ALGORITHM
            ):
                raise ValueError(f"resume_manifest_wrong_metadata_hash_algorithm:{index}")
            try:
                page_revision = int(record.get("source_page_latest_revision_id", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"resume_manifest_invalid_page_revision:{index}") from exc
            if page_revision <= 0:
                raise ValueError(f"resume_manifest_invalid_page_revision:{index}")
            try:
                _validate_mediawiki_timestamp(record.get("source_image_timestamp"))
                normalized_source_sha1 = _normalize_mediawiki_sha1(record.get("source_image_sha1"))
            except ValueError as exc:
                raise ValueError(f"resume_manifest_invalid_image_revision:{index}:{exc}") from exc
            if normalized_source_sha1 != record.get("source_image_sha1"):
                raise ValueError(f"resume_manifest_source_sha1_not_40_hex:{index}")
            if str(record.get("source_image_sha1_algorithm", "")).casefold() != "sha1":
                raise ValueError(f"resume_manifest_wrong_source_sha1_algorithm:{index}")
            source_page_title = str(record.get("source_page_title", ""))
            expected_page_url = page_url(source_page_title)
            expected_revision_url = page_revision_url(source_page_title, page_revision)
            if record.get("source_page_url") != expected_page_url:
                raise ValueError(f"resume_manifest_source_page_url_mismatch:{index}")
            if record.get("source_page_revision_url") != expected_revision_url:
                raise ValueError(f"resume_manifest_source_revision_url_mismatch:{index}")
            validate_wikimedia_url(
                expected_revision_url, API_HOST_ALLOWLIST, "resume_source_page_revision_url"
            )
            for url_field in ("image_url_original", "image_url_effective"):
                validate_wikimedia_url(
                    str(record.get(url_field, "")), DOWNLOAD_HOST_ALLOWLIST, f"resume_{url_field}"
                )
            missing_license_fields = [
                key for key in ("license", "license_url", "usage_terms") if not record.get(key)
            ]
            if missing_license_fields or not (record.get("artist") or record.get("credit")):
                raise ValueError(f"resume_manifest_missing_license_metadata:{index}")
            if not _valid_metadata_url(str(record["license_url"])):
                raise ValueError(f"resume_manifest_invalid_license_url:{index}")
            if record.get("training_eligibility") != "PROHIBITED_PENDING_HUMAN_REVIEW":
                raise ValueError(f"resume_manifest_training_eligibility_mismatch:{index}")
        else:
            legacy_records += 1
        page_ids.add(page_id)
        hashes.add(digest)
        relative_paths.add(relative_path)
        manifested_paths.add(local_path)
    class_root = _safe_child(output_root, class_name)
    if class_root.exists():
        if not class_root.is_dir():
            raise ValueError(f"resume_class_root_not_directory:{class_root}")
        for entry in class_root.iterdir():
            entry = _assert_no_reparse_components(entry)
            if not entry.is_file():
                raise ValueError(f"resume_collision:unexpected_directory_entry:{entry.name}")
            if entry not in manifested_paths:
                raise ValueError(f"resume_collision:unmanifested_file:{entry.name}")
    return ResumeState(frozenset(page_ids), frozenset(hashes), len(page_ids), legacy_records)


def _run_fields(run_id: str, run_started_at: str, config_sha256: str) -> dict[str, Any]:
    return {
        "collector_schema_version": COLLECTOR_SCHEMA_VERSION,
        "collector_version": _collector_version(),
        "collector_run_id": run_id,
        "collector_run_started_at": run_started_at,
        "collector_config_sha256": config_sha256,
        "collector_config_sha_algorithm": SHA_ALGORITHM,
    }


class _CollectionLock:
    def __init__(self, path: Path, class_name: str) -> None:
        self.path = path
        self.class_name = class_name
        self.acquired = False

    def __enter__(self) -> _CollectionLock:
        _secure_mkdir(self.path.parent)
        _assert_no_reparse_components(self.path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"collection_lock_exists:{self.path}") from exc
        try:
            payload = json.dumps(
                {
                    "class_name": self.class_name,
                    "pid": os.getpid(),
                    "started_at": utc_now(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
        except Exception:
            os.close(descriptor)
            self.path.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, *_args: Any) -> None:
        if not self.acquired:
            return
        _assert_no_reparse_components(self.path)
        self.path.unlink(missing_ok=False)
        self.acquired = False


def _canonical_collection_lock_paths(
    output_root: Path, manifest_root: Path, class_name: str
) -> tuple[Path, ...]:
    validate_class_name(class_name)
    candidates = (
        _safe_manifest_path(output_root, class_name, "collection.lock"),
        _safe_manifest_path(manifest_root, class_name, "collection.lock"),
    )
    unique: dict[str, Path] = {}
    for candidate in candidates:
        canonical_key = os.path.normcase(str(_absolute_without_resolving(candidate)))
        unique.setdefault(canonical_key, candidate)
    return tuple(unique[key] for key in sorted(unique))


class _CollectionLocks:
    def __init__(self, paths: Iterable[Path], class_name: str) -> None:
        unique: dict[str, Path] = {}
        for path in paths:
            absolute = _absolute_without_resolving(path)
            unique.setdefault(os.path.normcase(str(absolute)), absolute)
        self.paths = tuple(unique[key] for key in sorted(unique))
        self.class_name = class_name
        self.acquired: list[_CollectionLock] = []

    def __enter__(self) -> _CollectionLocks:
        try:
            for path in self.paths:
                lock = _CollectionLock(path, self.class_name)
                lock.__enter__()
                self.acquired.append(lock)
        except Exception as acquisition_error:
            try:
                self._release_all()
            except Exception as release_error:
                raise release_error from acquisition_error
            raise
        return self

    def _release_all(self) -> None:
        first_error: Exception | None = None
        for lock in reversed(self.acquired):
            try:
                lock.__exit__()
            except Exception as exc:
                first_error = first_error or exc
        self.acquired.clear()
        if first_error is not None:
            raise first_error

    def __exit__(self, *_args: Any) -> None:
        self._release_all()


def collect_class(
    config_path: Path,
    class_name: str,
    limit: int,
    output_root: Path,
    manifest_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    validate_class_name(class_name)
    output_root = _secure_mkdir(output_root)
    manifest_root = _secure_mkdir(manifest_root)
    lock_paths = _canonical_collection_lock_paths(output_root, manifest_root, class_name)
    with _CollectionLocks(lock_paths, class_name):
        return _collect_class_unlocked(
            config_path,
            class_name,
            limit,
            output_root,
            manifest_root,
            dry_run,
        )


def _collect_class_unlocked(
    config_path: Path,
    class_name: str,
    limit: int,
    output_root: Path,
    manifest_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    validate_class_name(class_name)
    if limit < 0:
        raise ValueError("limit_must_be_nonnegative")
    config_path = _assert_no_reparse_components(config_path)
    if not config_path.is_file():
        raise ValueError(f"config_not_regular_file:{config_path}")
    config_sha256 = sha256_file(config_path)
    config = load_yaml(config_path)
    classes = config.get("classes", {})
    if not isinstance(classes, dict) or class_name not in classes:
        configured = ", ".join(sorted(classes)) if isinstance(classes, dict) else ""
        raise KeyError(f"Unknown class {class_name!r}; configured: {configured}")
    class_config = classes[class_name]
    if not isinstance(class_config, dict):
        raise ValueError(f"class_config_must_be_mapping:{class_name}")
    allowed_license_prefixes = [
        str(value).strip() for value in config.get("allowed_license_prefixes", []) if str(value).strip()
    ]
    if not allowed_license_prefixes:
        raise ValueError("allowed_license_prefixes_must_not_be_empty")
    output_root = _secure_mkdir(output_root)
    manifest_root = _secure_mkdir(manifest_root)
    class_output_root = _secure_mkdir(_safe_child(output_root, class_name))
    manifest_path = _safe_manifest_path(manifest_root, class_name, "sources.jsonl")
    rejection_path = _safe_manifest_path(manifest_root, class_name, "rejections.jsonl")
    summary_path = _safe_manifest_path(manifest_root, class_name, "collection-summary.json")
    resume = _validate_resume_manifest(manifest_path, output_root, class_name)
    accepted_page_ids = set(resume.accepted_page_ids)
    accepted_hashes = set(resume.accepted_hashes)
    accepted_count = resume.accepted_count
    download_config = config.get("download", {})
    if not isinstance(download_config, dict):
        raise ValueError("download_config_must_be_mapping")
    client = CommonsClient(
        api_url=str(config["api_url"]),
        user_agent=str(config["user_agent"]),
        timeout=float(download_config.get("timeout_seconds", 30)),
        delay=float(download_config.get("request_delay_seconds", 0.2)),
    )
    run_id = uuid.uuid4().hex
    run_started_at = utc_now()
    run_fields = _run_fields(run_id, run_started_at, config_sha256)

    titles: OrderedDict[str, str] = OrderedDict()
    category_depth = int(class_config.get("category_depth", 2))
    discovery_config = config.get("discovery", {})
    if not isinstance(discovery_config, dict):
        raise ValueError("discovery_config_must_be_mapping")
    multiplier = int(discovery_config.get("candidate_multiplier", 3))
    minimum_candidates = int(discovery_config.get("minimum_candidates_per_route", 100))
    category_maximum = min(
        max(limit * multiplier, minimum_candidates),
        int(discovery_config.get("max_category_files_per_class", 2000)),
    )
    for category in class_config.get("categories", []):
        for title in client.category_titles(str(category), max_depth=category_depth, max_files=category_maximum):
            titles.setdefault(title, f"category:{category}")
    search_maximum = min(
        max(limit * multiplier, minimum_candidates),
        int(discovery_config.get("max_search_results_per_query", 750)),
    )
    for query in class_config.get("search_queries", []):
        for title in client.search_titles(str(query), maximum=search_maximum):
            titles.setdefault(title, f"search:{query}")

    stats: dict[str, Any] = {
        **run_fields,
        "class_name": class_name,
        "requested_total": limit,
        "existing_accepted": accepted_count,
        "existing_legacy_schema_records": resume.legacy_records,
        "discovered_titles": len(titles),
        "new_accepted": 0,
        "rejected": {},
        "dry_run": dry_run,
        "started_at": run_started_at,
    }
    if dry_run:
        stats["candidate_titles_after_keyword_filter"] = sum(
            1 for title in titles if keyword_allowed(title, class_config)
        )
        stats["finished_at"] = utc_now()
        _atomic_write_json(summary_path, stats)
        return stats

    min_width = int(download_config.get("min_width_px", 320))
    min_height = int(download_config.get("min_height_px", 240))
    max_bytes = int(download_config.get("max_file_bytes", 30 * 1024 * 1024))
    max_pixels = int(download_config.get("max_decoded_pixels", DEFAULT_MAX_PIXELS))
    max_dimension = int(download_config.get("max_decoded_dimension_px", DEFAULT_MAX_DECODED_DIMENSION))
    max_frames = int(download_config.get("max_frames", DEFAULT_MAX_FRAMES))
    if min(min_width, min_height, max_bytes, max_pixels, max_dimension, max_frames) <= 0:
        raise ValueError("download_limits_must_be_positive")
    thumb_width = int(download_config.get("max_dimension_px", 2048))
    if thumb_width <= 0:
        raise ValueError("max_dimension_px_must_be_positive")
    remaining_titles = [title for title in titles if keyword_allowed(title, class_config)]
    progress = tqdm(total=max(0, limit - accepted_count), desc=class_name, unit="image")
    try:
        for page in client.image_info(remaining_titles, thumb_width):
            if accepted_count >= limit:
                break
            try:
                page_id = int(page.get("pageid", 0) or 0)
            except (TypeError, ValueError):
                page_id = 0
            title = str(page.get("title", ""))
            if not page_id or page_id in accepted_page_ids:
                continue
            image_info = (page.get("imageinfo") or [{}])[0]
            if not isinstance(image_info, dict):
                image_info = {}
            mime = _normalized_content_type(str(image_info.get("mime", "")))
            metadata = image_info.get("extmetadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            snapshot_hash = _canonical_sha256(_source_metadata_snapshot(page, image_info))
            license_values, reason = _license_fields(metadata, allowed_license_prefixes)
            page_revision = 0
            image_timestamp = ""
            image_sha1 = ""
            if not reason:
                try:
                    page_revision, image_timestamp, image_sha1 = _stable_source_revision(page, image_info)
                except ValueError as exc:
                    reason = str(exc)
            if not reason and mime not in SUPPORTED_MIME_TYPES:
                reason = f"unsupported_mime:{mime or 'missing'}"
            if not reason and int(image_info.get("width", 0) or 0) < min_width:
                reason = "source_width_too_small"
            if not reason and int(image_info.get("height", 0) or 0) < min_height:
                reason = "source_height_too_small"
            image_url = str(image_info.get("thumburl") or image_info.get("url") or "")
            if not reason:
                try:
                    validate_wikimedia_url(image_url, DOWNLOAD_HOST_ALLOWLIST, "image_url")
                except ValueError as exc:
                    reason = str(exc)
            if reason:
                stats["rejected"][reason] = int(stats["rejected"].get(reason, 0)) + 1
                _atomic_append_jsonl(
                    rejection_path,
                    {
                        **run_fields,
                        "class_name": class_name,
                        "source_page_id": page_id,
                        "source_page_title": title,
                        "source_page_url": page_url(title),
                        "license": license_values["license"],
                        "source_metadata_snapshot_sha256": snapshot_hash,
                        "source_metadata_snapshot_sha_algorithm": SHA_ALGORITHM,
                        "status": "REJECTED_METADATA",
                        "reason": reason,
                        "checked_at": utc_now(),
                    },
                )
                continue
            candidate: DownloadedCandidate | None = None
            try:
                candidate = download_candidate(
                    client,
                    image_url,
                    class_output_root,
                    page_id,
                    title,
                    max_bytes=max_bytes,
                    expected_mime=mime,
                    max_pixels=max_pixels,
                    max_dimension_px=max_dimension,
                    max_frames=max_frames,
                )
                if candidate.width < min_width or candidate.height < min_height:
                    raise ValueError(f"downloaded_dimensions_too_small:{candidate.width}x{candidate.height}")
                _verify_source_unchanged_after_download(
                    client,
                    title=title,
                    thumb_width=thumb_width,
                    expected_page_id=page_id,
                    expected_page_revision=page_revision,
                    expected_image_timestamp=image_timestamp,
                    expected_image_sha1=image_sha1,
                    expected_snapshot_sha256=snapshot_hash,
                )
                if candidate.sha256 in accepted_hashes:
                    raise ValueError("exact_duplicate_sha256")
                relative_path = candidate.final_path.relative_to(output_root).as_posix()
                record = {
                    **run_fields,
                    "class_name": class_name,
                    "source_platform": "Wikimedia Commons",
                    "source_page_id": page_id,
                    "source_page_title": title,
                    "source_page_url": page_url(title),
                    "source_page_latest_revision_id": page_revision,
                    "source_page_revision_url": page_revision_url(title, page_revision),
                    "source_image_timestamp": image_timestamp,
                    "source_image_sha1": image_sha1,
                    "source_image_sha1_algorithm": "sha1",
                    "source_metadata_snapshot_sha256": snapshot_hash,
                    "source_metadata_snapshot_sha_algorithm": SHA_ALGORITHM,
                    "image_url_original": candidate.image_url_original,
                    "image_url_effective": candidate.image_url_effective,
                    "discovery_route": titles.get(title, ""),
                    **license_values,
                    "retrieved_at": utc_now(),
                    "relative_path": relative_path,
                    "sha_algorithm": SHA_ALGORITHM,
                    "sha256": candidate.sha256,
                    "width": candidate.width,
                    "height": candidate.height,
                    "frame_count": candidate.frame_count,
                    "decoded_format": candidate.decoded_format,
                    "source_mime": mime,
                    "content_mime": candidate.content_mime,
                    "bytes": candidate.byte_count,
                    "status": "ACCEPTED",
                    "qa_status": "PENDING_HUMAN_REVIEW",
                    "training_eligibility": "PROHIBITED_PENDING_HUMAN_REVIEW",
                }
                _publish_candidate(candidate, manifest_path, record)
            except (
                OSError,
                requests.RequestException,
                UnidentifiedImageError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                ValueError,
            ) as exc:
                if candidate is not None:
                    candidate.staging_path.unlink(missing_ok=True)
                reason = f"download_or_decode:{type(exc).__name__}:{exc}"
                rejection_status = (
                    "REJECTED_SOURCE_MUTATION"
                    if "source_mutated_during_download" in str(exc)
                    else "REJECTED_DOWNLOAD"
                )
                stats["rejected"][reason] = int(stats["rejected"].get(reason, 0)) + 1
                _atomic_append_jsonl(
                    rejection_path,
                    {
                        **run_fields,
                        "class_name": class_name,
                        "source_page_id": page_id,
                        "source_page_title": title,
                        "source_page_url": page_url(title),
                        "source_metadata_snapshot_sha256": snapshot_hash,
                        "source_metadata_snapshot_sha_algorithm": SHA_ALGORITHM,
                        "status": rejection_status,
                        "reason": reason,
                        "checked_at": utc_now(),
                    },
                )
                continue
            finally:
                if candidate is not None:
                    candidate.staging_path.unlink(missing_ok=True)
            accepted_page_ids.add(page_id)
            accepted_hashes.add(candidate.sha256)
            accepted_count += 1
            stats["new_accepted"] += 1
            progress.update(1)
    finally:
        progress.close()
    stats["accepted_total"] = accepted_count
    stats["target_met"] = accepted_count >= limit
    stats["finished_at"] = utc_now()
    _atomic_write_json(summary_path, stats)
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
        config_path=args.config,
        class_name=args.class_name,
        limit=args.limit,
        output_root=args.output_root,
        manifest_root=args.manifest_root,
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    main()
