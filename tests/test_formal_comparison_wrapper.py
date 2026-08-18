from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def test_formal_wrapper_returns_nonzero_for_blocked_comparator_output(tmp_path: Path) -> None:
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
    fake = project / "fake-comparator.cmd"
    fake.write_text(
        "@echo off\r\n"
        "powershell -NoProfile -Command \"Set-Content -LiteralPath $env:FORMAL_WRAPPER_RESULT "
        "-Value '{\\\"release_ready\\\":false,\\\"comparable\\\":true,"
        "\\\"release_blockers\\\":[{\\\"field\\\":\\\"forged\\\"}],"
        "\\\"critical_mismatches\\\":[]}'\"\r\n"
        "exit /b 0\r\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["FORMAL_WRAPPER_RESULT"] = str(output / "protocol_compatibility.json")
    environment["MCU_TEST_COMPARE_EXECUTABLE_OVERRIDE"] = "1"
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
        "-Yolo11Seed42", str(runs[0]), "-Yolo11Seed43", str(runs[1]),
        "-YoloXSeed42", str(runs[2]), "-YoloXSeed43", str(runs[3]),
        "-Yolo11Seed44", str(runs[4]), "-YoloXSeed44", str(runs[5]),
        "-OutputDirectory", str(output), "-CompareExecutableOverride", str(fake),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert completed.returncode != 0
    assert "BLOCKED" in completed.stderr or "BLOCKED" in completed.stdout
