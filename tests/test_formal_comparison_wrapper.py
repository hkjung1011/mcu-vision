from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def _wrapper_fixture(tmp_path: Path) -> tuple[Path, list[Path], Path]:
    project = tmp_path / "fixture"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source_script = Path(__file__).resolve().parents[1] / "scripts" / "compare_formal_mixed_runs.ps1"
    wrapper = scripts / source_script.name
    shutil.copy2(source_script, wrapper)
    attestation = project / "configs" / "experiments" / "mixed_commit_rpi_v2_attestation.json"
    attestation.parent.mkdir(parents=True)
    attestation.write_text("{}", encoding="utf-8")
    runs = []
    for index in range(6):
        run = project / "runs" / f"run{index}"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text("{}", encoding="utf-8")
        runs.append(run)
    output = project / "output"
    output.mkdir()
    return wrapper, runs, output


def _wrapper_command(wrapper: Path, runs: list[Path], output: Path, fake: Path) -> list[str]:
    return [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
        "-Yolo11Seed42", str(runs[0]), "-Yolo11Seed43", str(runs[1]),
        "-YoloXSeed42", str(runs[2]), "-YoloXSeed43", str(runs[3]),
        "-Yolo11Seed44", str(runs[4]), "-YoloXSeed44", str(runs[5]),
        "-OutputDirectory", str(output), "-CompareExecutableOverride", str(fake),
    ]


def _fake_comparator(path: Path, compatibility: str, *, write_validation: bool) -> None:
    compatibility_source = path.with_suffix(".compatibility.json")
    compatibility_source.write_text(compatibility, encoding="utf-8")
    commands = [
        "@echo off\r\n",
        "set FORMAL_FOUND=\r\n",
        ":parse_args\r\n",
        "if \"%~1\"==\"\" goto execute\r\n",
        "if /I \"%~1\"==\"--formal\" set FORMAL_FOUND=1\r\n",
        "shift\r\n",
        "goto parse_args\r\n",
        ":execute\r\n",
        "if not defined FORMAL_FOUND exit /b 23\r\n",
        f'copy /y "%~dp0{compatibility_source.name}" "%FORMAL_WRAPPER_RESULT%" >nul\r\n',
    ]
    if write_validation:
        validation_source = path.with_suffix(".validation.json")
        validation_source.write_text('{"status":"PASS","run_count":6}', encoding="utf-8")
        commands.append(
            f'copy /y "%~dp0{validation_source.name}" '
            '"%FORMAL_WRAPPER_VALIDATION%" >nul\r\n'
        )
    commands.append("exit /b 0\r\n")
    path.write_text("".join(commands), encoding="utf-8")


def _wrapper_environment(output: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["FORMAL_WRAPPER_RESULT"] = str(output / "protocol_compatibility.json")
    environment["FORMAL_WRAPPER_VALIDATION"] = str(output / "formal_validation.json")
    environment["MCU_TEST_COMPARE_EXECUTABLE_OVERRIDE"] = "1"
    return environment


def test_formal_wrapper_returns_nonzero_for_blocked_comparator_output(tmp_path: Path) -> None:
    wrapper, runs, output = _wrapper_fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake-comparator.cmd"
    _fake_comparator(
        fake,
        '{"release_ready":false,"comparable":true,'
        '"release_blockers":[{"field":"forged"}],"critical_mismatches":[]}',
        write_validation=False,
    )
    completed = subprocess.run(
        _wrapper_command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_wrapper_environment(output),
    )
    assert completed.returncode != 0
    assert "BLOCKED" in completed.stderr or "BLOCKED" in completed.stdout


def test_formal_wrapper_requires_final_validation_file(tmp_path: Path) -> None:
    wrapper, runs, output = _wrapper_fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake-comparator.cmd"
    _fake_comparator(
        fake,
        '{"release_ready":true,"comparable":true,'
        '"release_blockers":[],"critical_mismatches":[]}',
        write_validation=False,
    )

    completed = subprocess.run(
        _wrapper_command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_wrapper_environment(output),
    )

    assert completed.returncode != 0
    assert "formal_validation.json" in completed.stderr or "formal_validation.json" in completed.stdout


def test_formal_wrapper_passes_formal_flag_and_finishes_pass(tmp_path: Path) -> None:
    wrapper, runs, output = _wrapper_fixture(tmp_path)
    fake = tmp_path / "fixture" / "fake-comparator.cmd"
    _fake_comparator(
        fake,
        '{"release_ready":true,"comparable":true,'
        '"release_blockers":[],"critical_mismatches":[]}',
        write_validation=True,
    )

    completed = subprocess.run(
        _wrapper_command(wrapper, runs, output, fake),
        capture_output=True,
        text=True,
        env=_wrapper_environment(output),
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    validation = json.loads((output / "formal_validation.json").read_text(encoding="utf-8-sig"))
    assert validation == {"status": "PASS", "run_count": 6}
