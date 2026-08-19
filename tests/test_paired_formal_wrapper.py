from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    project = tmp_path / "fixture"
    scripts = project / "scripts"
    configs = project / "configs" / "experiments"
    scripts.mkdir(parents=True)
    configs.mkdir(parents=True)
    wrapper = scripts / "compare_formal_paired_2seed_runs.ps1"
    shutil.copy2(PROJECT_ROOT / "scripts" / wrapper.name, wrapper)
    for name in (
        "rpi_bootstrap_paired_2seed_release_v1.yaml",
        "mixed_commit_rpi_paired_2seed_v1_attestation.json",
    ):
        shutil.copy2(PROJECT_ROOT / "configs" / "experiments" / name, configs / name)
    runs = []
    for name, model, seed in (
        ("y11-42", "yolo11m", 42),
        ("y11-43", "yolo11m", 43),
        ("yx-42", "YOLOX-S", 42),
        ("yx-43", "YOLOX-S", 43),
    ):
        run = project / "runs" / name
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(
            json.dumps({"model": model, "protocol": {"seed": seed}}),
            encoding="utf-8",
        )
        runs.append(run)
    output = project / "output"
    output.mkdir()
    return wrapper, runs, output


def _command(wrapper: Path, runs: list[Path], output: Path, fake: Path) -> list[str]:
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(wrapper),
        "-Yolo11Seed42",
        str(runs[0]),
        "-Yolo11Seed43",
        str(runs[1]),
        "-YoloXSeed42",
        str(runs[2]),
        "-YoloXSeed43",
        str(runs[3]),
        "-OutputDirectory",
        str(output),
        "-CompareExecutableOverride",
        str(fake),
    ]


def _fake(path: Path, *, run_count: int | str = 4) -> None:
    policy_sha = "1865539e9b3569dd4942d9d17495a3644e059df70259b555bd7985e7bdf76f27"
    base_sha = "02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5"
    compatibility = {
        "release_ready": True,
        "comparable": True,
        "release_blockers": [],
        "critical_mismatches": [],
        "run_count": run_count,
        "formal_release_policy": {
            "policy_id": "rpi_bootstrap_paired_2seed_release_v1",
            "policy_sha256": policy_sha,
            "base_protocol_sha256": base_sha,
            "evidence_tier": "paired_2seed_descriptive",
        },
    }
    validation = {
        "status": "PASS",
        "run_count": run_count,
        "policy_id": "rpi_bootstrap_paired_2seed_release_v1",
        "policy_sha256": policy_sha,
        "base_protocol_sha256": base_sha,
        "evidence_tier": "paired_2seed_descriptive",
        "paired_n": 2,
        "degrees_of_freedom": 1,
        "interpretation": "descriptive_only",
    }
    compatibility_path = path.with_suffix(".compatibility.json")
    validation_path = path.with_suffix(".validation.json")
    compatibility_path.write_text(json.dumps(compatibility), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    path.write_text(
        "@echo off\r\n"
        "set FORMAL_FOUND=\r\n"
        "set POLICY_FOUND=\r\n"
        ":parse\r\n"
        "if \"%~1\"==\"\" goto execute\r\n"
        "if /I \"%~1\"==\"--formal\" set FORMAL_FOUND=1\r\n"
        "if /I \"%~1\"==\"--formal-release-policy\" set POLICY_FOUND=1\r\n"
        "shift\r\n"
        "goto parse\r\n"
        ":execute\r\n"
        "if not defined FORMAL_FOUND exit /b 23\r\n"
        "if not defined POLICY_FOUND exit /b 24\r\n"
        f'copy /y "%~dp0{compatibility_path.name}" "%PAIRED_WRAPPER_COMPATIBILITY%" >nul\r\n'
        f'copy /y "%~dp0{validation_path.name}" "%PAIRED_WRAPPER_VALIDATION%" >nul\r\n'
        "exit /b 0\r\n",
        encoding="utf-8",
    )


def _environment(output: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MCU_TEST_COMPARE_EXECUTABLE_OVERRIDE"] = "1"
    environment["PAIRED_WRAPPER_COMPATIBILITY"] = str(
        output / "protocol_compatibility.json"
    )
    environment["PAIRED_WRAPPER_VALIDATION"] = str(output / "formal_validation.json")
    return environment


def test_paired_wrapper_passes_policy_flag_and_exact_four_run_gate(tmp_path: Path) -> None:
    wrapper, runs, output = _fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake.cmd"
    _fake(fake)
    completed = subprocess.run(
        _command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_environment(output),
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_paired_wrapper_rejects_non_four_run_claim(tmp_path: Path) -> None:
    wrapper, runs, output = _fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake.cmd"
    _fake(fake, run_count=5)
    completed = subprocess.run(
        _command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_environment(output),
    )
    assert completed.returncode != 0
    assert "BLOCKED" in completed.stderr or "BLOCKED" in completed.stdout


def test_paired_wrapper_rejects_swapped_model_seed_slots(tmp_path: Path) -> None:
    wrapper, runs, output = _fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake.cmd"
    _fake(fake)
    swapped = [runs[1], runs[0], runs[2], runs[3]]
    completed = subprocess.run(
        _command(wrapper, swapped, output, fake),
        capture_output=True,
        text=True,
        env=_environment(output),
    )
    assert completed.returncode != 0
    assert "exact model/seed slot" in completed.stderr or "exact model/seed slot" in completed.stdout


def test_paired_wrapper_rejects_string_run_count(tmp_path: Path) -> None:
    wrapper, runs, output = _fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake.cmd"
    _fake(fake, run_count="4")
    completed = subprocess.run(
        _command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_environment(output),
    )
    assert completed.returncode != 0
    assert "BLOCKED" in completed.stderr or "BLOCKED" in completed.stdout
