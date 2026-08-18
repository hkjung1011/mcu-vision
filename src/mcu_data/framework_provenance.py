from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def inspect_git_package_source(module_file: str | Path, *, framework: str) -> dict[str, Any]:
    """Resolve an imported source package to its actual Git checkout and require it clean."""
    module_path = Path(module_file).resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=15,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=15,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify imported {framework} Git source: {module_path}") from exc
    if status:
        raise ValueError(f"Imported {framework} Git checkout is dirty: {status}")
    return {"framework": framework, "commit": head, "clean": True, "module_file": str(module_path)}


def verify_yolox_source(module_file: str | Path, expected_commit: str | None = None) -> dict[str, Any]:
    evidence = inspect_git_package_source(module_file, framework="YOLOX")
    if expected_commit and evidence["commit"].lower() != expected_commit.lower():
        raise ValueError(
            "Imported YOLOX source commit differs from run manifest: "
            f"actual={evidence['commit']}, expected={expected_commit}"
        )
    return evidence
