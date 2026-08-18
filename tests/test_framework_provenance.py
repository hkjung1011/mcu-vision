from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mcu_data.framework_provenance import verify_yolox_source


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_yolox_source_uses_actual_import_checkout_head_and_clean_state(tmp_path: Path) -> None:
    repo = tmp_path / "YOLOX"
    module = repo / "yolox" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("__version__ = 'fixture'\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "source")
    head = _git(repo, "rev-parse", "HEAD")

    evidence = verify_yolox_source(module, head)
    assert evidence["commit"] == head
    assert evidence["clean"] is True

    with pytest.raises(ValueError, match="differs from run manifest"):
        verify_yolox_source(module, "f" * 40)

    module.write_text("__version__ = 'dirty'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        verify_yolox_source(module, head)
