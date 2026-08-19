from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, features

from mcu_data import wikimedia


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"",
        *,
        url: str = "https://upload.wikimedia.org/example.png",
        status_code: int = 200,
        content_type: str = "image/png",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        if headers:
            self.headers.update(headers)
        self.closed = False

    @property
    def content(self) -> bytes:
        return self._body

    def json(self) -> Any:
        return json.loads(self._body)

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self._body[index : index + chunk_size] for index in range(0, len(self._body), chunk_size)]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


class FakeDownloadClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def open_download(self, _image_url: str) -> FakeResponse:
        return self.response


def image_bytes(image_format: str = "PNG", size: tuple[int, int] = (32, 24)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(30, 80, 120)).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "value",
    [
        "http://commons.wikimedia.org/w/api.php",
        "https://evil.example/w/api.php",
        "https://commons.wikimedia.org.evil.example/w/api.php",
        "https://user@commons.wikimedia.org/w/api.php",
        "https://commons.wikimedia.org:444/w/api.php",
        "https://commons.wikimedia.org/w/api.php#fragment",
    ],
)
def test_wikimedia_url_allowlist_is_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        wikimedia.validate_wikimedia_url(value, wikimedia.API_HOST_ALLOWLIST, "api")


def test_wikimedia_url_allowlist_accepts_exact_https_hosts() -> None:
    assert wikimedia.validate_wikimedia_url(
        "https://commons.wikimedia.org/w/api.php",
        wikimedia.API_HOST_ALLOWLIST,
        "api",
    )
    assert wikimedia.validate_wikimedia_url(
        "https://upload.wikimedia.org/wikipedia/commons/a/a0/test.png",
        wikimedia.DOWNLOAD_HOST_ALLOWLIST,
        "download",
    )


def test_client_rejects_disallowed_redirect_before_following() -> None:
    first = FakeResponse(
        url="https://commons.wikimedia.org/w/api.php",
        status_code=302,
        content_type="text/html",
        headers={"Location": "https://evil.example/payload"},
    )
    client = wikimedia.CommonsClient(
        "https://commons.wikimedia.org/w/api.php", "test-agent", timeout=1, delay=0
    )
    session = FakeSession([first])
    client.session = session  # type: ignore[assignment]

    with pytest.raises(ValueError, match="host_not_allowed"):
        client.get_json({"action": "query"})

    assert len(session.calls) == 1
    assert session.calls[0]["allow_redirects"] is False
    assert first.closed


def test_client_rejects_disallowed_effective_url() -> None:
    response = FakeResponse(
        b"{}",
        url="https://evil.example/final",
        content_type="application/json",
    )
    client = wikimedia.CommonsClient(
        "https://commons.wikimedia.org/w/api.php", "test-agent", timeout=1, delay=0
    )
    client.session = FakeSession([response])  # type: ignore[assignment]

    with pytest.raises(ValueError, match="host_not_allowed"):
        client.get_json({"action": "query"})
    assert response.closed


def test_api_content_length_is_rejected_before_body_stream() -> None:
    response = FakeResponse(
        b"{}",
        url="https://commons.wikimedia.org/w/api.php",
        content_type="application/json",
        headers={"Content-Length": str(wikimedia.MAX_API_RESPONSE_BYTES + 1)},
    )
    client = wikimedia.CommonsClient(
        "https://commons.wikimedia.org/w/api.php", "test-agent", timeout=1, delay=0
    )
    session = FakeSession([response])
    client.session = session  # type: ignore[assignment]

    with pytest.raises(ValueError, match="content_length_exceeds_limit"):
        client.get_json({"action": "query"})

    assert session.calls[0]["stream"] is True
    assert response.closed


@pytest.mark.parametrize("class_name", ["../escape", "a/b", "a\\b", ".hidden", "", "한글"])
def test_class_name_rejects_path_traversal_and_unsafe_names(class_name: str) -> None:
    with pytest.raises(ValueError):
        wikimedia.validate_class_name(class_name)


def test_secure_path_rejects_symlink_component(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available on this host")

    with pytest.raises(ValueError, match="reparse_or_symlink"):
        wikimedia._secure_mkdir(link / "child")


def test_download_rejects_header_magic_mismatch_and_removes_staging(tmp_path: Path) -> None:
    body = image_bytes("PNG")
    response = FakeResponse(body, content_type="image/jpeg")

    with pytest.raises(ValueError, match="magic_mime_mismatch"):
        wikimedia.download_candidate(
            FakeDownloadClient(response),  # type: ignore[arg-type]
            response.url,
            tmp_path,
            11,
            "File:test.png",
            max_bytes=100_000,
        )

    assert not list(tmp_path.glob("*.part"))


def test_download_enforces_decoded_pixel_cap(tmp_path: Path) -> None:
    body = image_bytes("PNG", (20, 20))
    response = FakeResponse(body)

    with pytest.raises(ValueError, match="decoded_pixels_exceed_limit"):
        wikimedia.download_candidate(
            FakeDownloadClient(response),  # type: ignore[arg-type]
            response.url,
            tmp_path,
            12,
            "File:test.png",
            max_bytes=100_000,
            max_pixels=399,
        )


def test_download_enforces_frame_cap(tmp_path: Path) -> None:
    if not features.check("webp"):
        pytest.skip("WebP support is unavailable")
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (16, 16), color=color) for color in ("red", "blue")]
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:], duration=10)
    response = FakeResponse(
        buffer.getvalue(),
        url="https://upload.wikimedia.org/example.webp",
        content_type="image/webp",
    )

    with pytest.raises(ValueError, match="frame_count_exceeds_limit"):
        wikimedia.download_candidate(
            FakeDownloadClient(response),  # type: ignore[arg-type]
            response.url,
            tmp_path,
            13,
            "File:test.webp",
            max_bytes=100_000,
            max_frames=1,
        )


def test_download_removes_staging_on_decompression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = image_bytes("PNG")
    response = FakeResponse(body)

    def raise_bomb(*_args: Any, **_kwargs: Any) -> Any:
        raise Image.DecompressionBombError("simulated decompression bomb")

    monkeypatch.setattr(wikimedia.Image, "open", raise_bomb)
    with pytest.raises(Image.DecompressionBombError):
        wikimedia.download_candidate(
            FakeDownloadClient(response),  # type: ignore[arg-type]
            response.url,
            tmp_path,
            14,
            "File:test.png",
            max_bytes=100_000,
        )

    assert not list(tmp_path.glob("*.part"))


def test_license_metadata_and_source_revision_are_required() -> None:
    values, reason = wikimedia._license_fields(
        {"LicenseShortName": {"value": "CC BY 4.0"}}, ["CC BY"]
    )
    assert values["license"] == "CC BY 4.0"
    assert reason.startswith("missing_license_metadata")

    with pytest.raises(ValueError, match="source_page_revision"):
        wikimedia._stable_source_revision({"pageid": 1}, {})

    valid_page = {"lastrevid": 10}
    with pytest.raises(ValueError, match="image_timestamp"):
        wikimedia._stable_source_revision(valid_page, {"sha1": "a" * 40})
    with pytest.raises(ValueError, match="image_sha1"):
        wikimedia._stable_source_revision(
            valid_page,
            {"timestamp": "2026-08-19T00:00:00Z", "sha1": "not-a-valid-sha1"},
        )


def test_mediawiki_base36_sha1_is_stored_as_40_hex() -> None:
    normalized = wikimedia._normalize_mediawiki_sha1("1" * 31)
    assert len(normalized) == 40
    assert all(character in "0123456789abcdef" for character in normalized)


def test_license_allowlist_rejects_noncommercial_and_no_derivatives_variants() -> None:
    allowed = ["CC BY", "CC-BY"]
    assert wikimedia.license_allowed("CC BY 4.0", allowed)
    assert wikimedia.license_allowed("CC BY-SA 4.0", allowed)
    assert not wikimedia.license_allowed("CC BY-NC 4.0", allowed)
    assert not wikimedia.license_allowed("CC BY-ND 4.0", allowed)
    assert not wikimedia.license_allowed("CC BY-NC-ND 4.0", allowed)


def test_publish_rolls_back_file_when_manifest_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".candidate.part"
    staging.write_bytes(b"candidate")
    digest = hashlib.sha256(b"candidate").hexdigest()
    candidate = wikimedia.DownloadedCandidate(
        staging_path=staging,
        final_path=tmp_path / "final.png",
        width=1,
        height=1,
        frame_count=1,
        decoded_format="PNG",
        content_mime="image/png",
        byte_count=9,
        sha256=digest,
        image_url_original="https://upload.wikimedia.org/original.png",
        image_url_effective="https://upload.wikimedia.org/effective.png",
    )

    def fail_append(_path: Path, _record: dict[str, Any]) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(wikimedia, "_atomic_append_jsonl", fail_append)
    with pytest.raises(OSError, match="simulated manifest failure"):
        wikimedia._publish_candidate(candidate, tmp_path / "sources.jsonl", {"a": 1})

    assert not candidate.final_path.exists()
    assert not candidate.staging_path.exists()


def test_collection_lock_rejects_concurrent_writer_and_is_removed(tmp_path: Path) -> None:
    lock_path = tmp_path / "safe_class.collection.lock"
    with wikimedia._CollectionLock(lock_path, "safe_class"):
        assert lock_path.is_file()
        with pytest.raises(ValueError, match="collection_lock_exists"):
            with wikimedia._CollectionLock(lock_path, "safe_class"):
                pass
    assert not lock_path.exists()


def test_dual_lock_blocks_same_output_with_different_manifest_roots(tmp_path: Path) -> None:
    output_root = tmp_path / "shared-output"
    first_manifest_root = tmp_path / "manifest-a"
    second_manifest_root = tmp_path / "manifest-b"
    for root in (output_root, first_manifest_root, second_manifest_root):
        root.mkdir()
    first_paths = wikimedia._canonical_collection_lock_paths(
        output_root, first_manifest_root, "safe_class"
    )
    second_paths = wikimedia._canonical_collection_lock_paths(
        output_root, second_manifest_root, "safe_class"
    )

    with wikimedia._CollectionLocks(first_paths, "safe_class"):
        with pytest.raises(ValueError, match="collection_lock_exists"):
            with wikimedia._CollectionLocks(second_paths, "safe_class"):
                pass
        assert not (second_manifest_root / "safe_class.collection.lock").exists()

    assert not any(tmp_path.rglob("*.collection.lock"))


def test_dual_lock_deduplicates_same_output_and_manifest_root(tmp_path: Path) -> None:
    paths = wikimedia._canonical_collection_lock_paths(tmp_path, tmp_path, "safe_class")
    assert paths == (tmp_path / "safe_class.collection.lock",)


def test_resume_validates_relative_path_and_file_hash(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    class_root = output_root / "safe_class"
    class_root.mkdir(parents=True)
    image_path = class_root / "image.png"
    image_path.write_bytes(b"content")
    digest = hashlib.sha256(b"content").hexdigest()
    manifest = tmp_path / "safe_class.sources.jsonl"
    base_record = {
        "class_name": "safe_class",
        "source_page_id": 1,
        "relative_path": "safe_class/image.png",
        "sha256": digest,
        "bytes": len(b"content"),
        "status": "ACCEPTED",
        "qa_status": "PENDING_HUMAN_REVIEW",
    }
    wikimedia._atomic_append_jsonl(manifest, base_record)
    state = wikimedia._validate_resume_manifest(manifest, output_root, "safe_class")
    assert state.accepted_count == 1
    assert state.legacy_records == 1

    image_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash_mismatch"):
        wikimedia._validate_resume_manifest(manifest, output_root, "safe_class")


def test_resume_rejects_manifest_path_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest = tmp_path / "safe_class.sources.jsonl"
    record = {
        "class_name": "safe_class",
        "source_page_id": 1,
        "relative_path": "safe_class/../../escape.png",
        "sha256": "0" * 64,
        "status": "ACCEPTED",
        "qa_status": "PENDING_HUMAN_REVIEW",
    }
    wikimedia._atomic_append_jsonl(manifest, record)
    with pytest.raises(ValueError, match="unsafe_relative_path"):
        wikimedia._validate_resume_manifest(manifest, output_root, "safe_class")


def test_resume_rejects_unmanifested_file(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    class_root = output_root / "safe_class"
    class_root.mkdir(parents=True)
    (class_root / "orphan.png").write_bytes(b"orphan")

    with pytest.raises(ValueError, match="unmanifested_file"):
        wikimedia._validate_resume_manifest(
            tmp_path / "safe_class.sources.jsonl", output_root, "safe_class"
        )


def test_collection_records_fixed_provenance_and_pending_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = image_bytes("PNG")
    image_url = "https://upload.wikimedia.org/source.png"
    effective_url = "https://upload.wikimedia.org/effective.png"
    metadata = {
        "LicenseShortName": {"value": "CC BY 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
        "UsageTerms": {"value": "Creative Commons Attribution 4.0"},
        "Artist": {"value": "Example author"},
        "Credit": {"value": "Example credit"},
        "AttributionRequired": {"value": "true"},
    }

    class FakeCommonsClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def category_titles(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return ["File:Example board.png"]

        def search_titles(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

        def image_info(self, _titles: Any, _thumb_width: int) -> list[dict[str, Any]]:
            return [
                {
                    "pageid": 123,
                    "title": "File:Example board.png",
                    "lastrevid": 456,
                    "imageinfo": [
                        {
                            "thumburl": image_url,
                            "url": image_url,
                            "width": 32,
                            "height": 24,
                            "mime": "image/png",
                            "timestamp": "2026-08-19T00:00:00Z",
                            "sha1": "a" * 40,
                            "extmetadata": metadata,
                        }
                    ],
                }
            ]

        def open_download(self, _url: str) -> FakeResponse:
            return FakeResponse(body, url=effective_url)

    monkeypatch.setattr(wikimedia, "CommonsClient", FakeCommonsClient)
    config = tmp_path / "sources.yaml"
    config.write_text(
        "\n".join(
            [
                "api_url: https://commons.wikimedia.org/w/api.php",
                "user_agent: test-agent",
                "allowed_license_prefixes: [CC BY]",
                "download:",
                "  min_width_px: 1",
                "  min_height_px: 1",
                "  max_file_bytes: 100000",
                "  max_decoded_pixels: 10000",
                "  max_decoded_dimension_px: 100",
                "  max_frames: 1",
                "discovery:",
                "  minimum_candidates_per_route: 1",
                "classes:",
                "  safe_class:",
                "    categories: [Category:Example]",
                "    include_any: [example]",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    manifest_root = tmp_path / "manifests"

    summary = wikimedia.collect_class(
        config,
        "safe_class",
        1,
        output_root,
        manifest_root,
        dry_run=False,
    )
    record = json.loads((manifest_root / "safe_class.sources.jsonl").read_text(encoding="utf-8"))

    assert summary["new_accepted"] == 1
    assert record["collector_schema_version"] == wikimedia.COLLECTOR_SCHEMA_VERSION
    assert record["collector_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert record["source_page_latest_revision_id"] == 456
    assert record["source_page_revision_url"].endswith("?oldid=456")
    assert record["source_image_timestamp"] == "2026-08-19T00:00:00Z"
    assert record["source_image_sha1"] == "a" * 40
    assert record["source_image_sha1_algorithm"] == "sha1"
    assert record["source_metadata_snapshot_sha256"]
    assert record["sha_algorithm"] == "sha256"
    assert record["image_url_original"] == image_url
    assert record["image_url_effective"] == effective_url
    assert record["status"] == "ACCEPTED"
    assert record["qa_status"] == "PENDING_HUMAN_REVIEW"
    assert record["training_eligibility"] == "PROHIBITED_PENDING_HUMAN_REVIEW"

    resumed = wikimedia.collect_class(
        config,
        "safe_class",
        1,
        output_root,
        manifest_root,
        dry_run=False,
    )
    assert resumed["existing_accepted"] == 1
    assert resumed["new_accepted"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["page_id", "title", "page_revision", "image_timestamp", "image_sha1", "metadata"],
)
def test_collection_rejects_source_mutation_after_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    body = image_bytes("PNG")
    image_url = "https://upload.wikimedia.org/source.png"

    def build_page(*, mutate: bool) -> dict[str, Any]:
        metadata = {
            "LicenseShortName": {"value": "CC BY 4.0"},
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
            "UsageTerms": {"value": "Creative Commons Attribution 4.0"},
            "Artist": {"value": "Example author"},
            "Credit": {"value": "Example credit"},
            "AttributionRequired": {"value": "true"},
        }
        page: dict[str, Any] = {
            "pageid": 123,
            "title": "File:Example board.png",
            "lastrevid": 456,
            "imageinfo": [
                {
                    "thumburl": image_url,
                    "url": image_url,
                    "width": 32,
                    "height": 24,
                    "mime": "image/png",
                    "timestamp": "2026-08-19T00:00:00Z",
                    "sha1": "a" * 40,
                    "extmetadata": metadata,
                }
            ],
        }
        if mutate:
            if mutation == "page_id":
                page["pageid"] = 124
            elif mutation == "title":
                page["title"] = "File:Changed board.png"
            elif mutation == "page_revision":
                page["lastrevid"] = 457
            elif mutation == "image_timestamp":
                page["imageinfo"][0]["timestamp"] = "2026-08-19T00:00:01Z"
            elif mutation == "image_sha1":
                page["imageinfo"][0]["sha1"] = "b" * 40
            elif mutation == "metadata":
                page["imageinfo"][0]["extmetadata"]["Credit"] = {"value": "Changed credit"}
        return page

    class MutatingCommonsClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.image_info_calls = 0

        def category_titles(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return ["File:Example board.png"]

        def search_titles(self, *_args: Any, **_kwargs: Any) -> list[str]:
            return []

        def image_info(self, _titles: Any, _thumb_width: int) -> list[dict[str, Any]]:
            self.image_info_calls += 1
            return [build_page(mutate=self.image_info_calls > 1)]

        def open_download(self, _url: str) -> FakeResponse:
            return FakeResponse(body, url=image_url)

    monkeypatch.setattr(wikimedia, "CommonsClient", MutatingCommonsClient)
    config = tmp_path / "sources.yaml"
    config.write_text(
        "\n".join(
            [
                "api_url: https://commons.wikimedia.org/w/api.php",
                "user_agent: test-agent",
                "allowed_license_prefixes: [CC BY]",
                "download:",
                "  min_width_px: 1",
                "  min_height_px: 1",
                "  max_file_bytes: 100000",
                "  max_decoded_pixels: 10000",
                "  max_decoded_dimension_px: 100",
                "  max_dimension_px: 100",
                "  max_frames: 1",
                "discovery:",
                "  minimum_candidates_per_route: 1",
                "classes:",
                "  safe_class:",
                "    categories: [Category:Example]",
                "    include_any: [example]",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"
    manifest_root = tmp_path / "manifests"

    summary = wikimedia.collect_class(
        config,
        "safe_class",
        1,
        output_root,
        manifest_root,
        dry_run=False,
    )

    assert summary["new_accepted"] == 0
    assert summary["accepted_total"] == 0
    assert any("source_mutated_during_download" in reason for reason in summary["rejected"])
    assert not (manifest_root / "safe_class.sources.jsonl").exists()
    assert not list((output_root / "safe_class").iterdir())
    rejection = (manifest_root / "safe_class.rejections.jsonl").read_text(encoding="utf-8")
    assert "source_mutated_during_download" in rejection
    assert '"status":"REJECTED_SOURCE_MUTATION"' in rejection
    assert not any(tmp_path.rglob("*.collection.lock"))
