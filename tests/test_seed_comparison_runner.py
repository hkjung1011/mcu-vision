from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_compare_seeds.ps1"

HARNESS = r'''param(
    [Parameter(Mandatory = $true)][string]$SourceScript,
    [Parameter(Mandatory = $true)][string]$Action,
    [string]$ComparisonDirectory
)
$ErrorActionPreference = "Stop"
$Tokens = $null
$ParseErrors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $SourceScript,
    [ref]$Tokens,
    [ref]$ParseErrors
)
if ($ParseErrors.Count -ne 0) {
    throw "Runner parse failed: $($ParseErrors[0].Message)"
}
foreach ($Name in @("Format-NativeCommand", "New-ComparisonArguments", "Assert-ComparisonResult")) {
    $Definition = $Ast.Find(
        {
            param($Node)
            $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                $Node.Name -eq $Name
        },
        $true
    )
    if ($null -eq $Definition) {
        throw "Runner function is missing: $Name"
    }
    Invoke-Expression $Definition.Extent.Text
}

if ($Action -eq "arguments") {
    $Runs = @("C:\fixture\yolo11m_seed42", "C:\fixture\yolox_s_seed42")
    $FullArguments = @(New-ComparisonArguments -RunDirectories $Runs -OutputDirectory "C:\fixture\full" -Formal)
    $SmokeArguments = @(New-ComparisonArguments -RunDirectories $Runs -OutputDirectory "C:\fixture\smoke")
    [PSCustomObject]@{
        full_arguments = $FullArguments
        smoke_arguments = $SmokeArguments
        full_command = Format-NativeCommand "C:\fixture\mcu-compare-runs.exe" $FullArguments
        smoke_command = Format-NativeCommand "C:\fixture\mcu-compare-runs.exe" $SmokeArguments
    } | ConvertTo-Json -Depth 5 -Compress
    return
}
if ($Action -eq "validate-full") {
    $Result = Assert-ComparisonResult -ComparisonDirectory $ComparisonDirectory
}
elseif ($Action -eq "validate-smoke") {
    $Result = Assert-ComparisonResult -ComparisonDirectory $ComparisonDirectory -Smoke
}
else {
    throw "Unknown action: $Action"
}
[PSCustomObject]@{
    comparable = $Result.Compatibility.comparable
    release_ready = $Result.Compatibility.release_ready
    formal_status = $Result.FormalValidation.status
    formal_run_count = $Result.FormalValidation.run_count
} | ConvertTo-Json -Depth 5 -Compress
'''


def _invoke(
    tmp_path: Path,
    action: str,
    comparison_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    harness = tmp_path / "runner-function-harness.ps1"
    harness.write_text(HARNESS, encoding="utf-8")
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(harness),
        "-SourceScript",
        str(RUNNER),
        "-Action",
        action,
    ]
    if comparison_dir is not None:
        command.extend(("-ComparisonDirectory", str(comparison_dir)))
    return subprocess.run(command, capture_output=True, text=True)


def _write_comparison(
    root: Path,
    compatibility: dict[str, object],
    formal_validation: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    (root / "protocol_compatibility.json").write_text(
        json.dumps(compatibility), encoding="utf-8"
    )
    if formal_validation is not None:
        (root / "formal_validation.json").write_text(
            json.dumps(formal_validation), encoding="utf-8"
        )
    return root


def _pass_compatibility() -> dict[str, object]:
    return {
        "release_ready": True,
        "comparable": True,
        "release_blockers": [],
        "critical_mismatches": [],
        "run_count": 6,
    }


def test_full_arguments_and_dry_run_command_include_formal_while_smoke_omits_it(
    tmp_path: Path,
) -> None:
    completed = _invoke(tmp_path, "arguments")
    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)

    assert result["full_arguments"].count("--formal") == 1
    assert "--formal" in result["full_command"].split()
    assert "--formal" not in result["smoke_arguments"]
    assert "--formal" not in result["smoke_command"].split()
    source = RUNNER.read_text(encoding="utf-8")
    assert "-Formal:(-not $Smoke)" in source
    assert "[COMPARE] $(Format-NativeCommand $CompareExecutable $ComparisonArguments)" in source


def test_full_comparison_requires_and_accepts_exact_formal_pass(tmp_path: Path) -> None:
    comparison = _write_comparison(
        tmp_path / "full-pass",
        _pass_compatibility(),
        {"status": "PASS", "run_count": 6},
    )
    completed = _invoke(tmp_path, "validate-full", comparison)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result == {
        "comparable": True,
        "release_ready": True,
        "formal_status": "PASS",
        "formal_run_count": 6,
    }


def test_smoke_comparison_omits_formal_and_allows_expected_release_blockers(
    tmp_path: Path,
) -> None:
    compatibility = _pass_compatibility()
    compatibility.update(
        release_ready=False,
        release_blockers=[{"field": "smoke_runs"}],
    )
    comparison = _write_comparison(tmp_path / "smoke-pass", compatibility)
    completed = _invoke(tmp_path, "validate-smoke", comparison)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    result = json.loads(completed.stdout)
    assert result["comparable"] is True
    assert result["release_ready"] is False
    assert result["formal_status"] is None


def test_full_comparison_blocks_missing_formal_validation(tmp_path: Path) -> None:
    comparison = _write_comparison(tmp_path / "missing-formal", _pass_compatibility())
    completed = _invoke(tmp_path, "validate-full", comparison)
    assert completed.returncode != 0
    assert "did not produce formal_validation.json" in completed.stderr


@pytest.mark.parametrize(
    "formal_validation",
    (
        {"status": "FAIL", "run_count": 6},
        {"status": "PASS", "run_count": 5},
        {"status": "PASS", "run_count": "6"},
    ),
)
def test_full_comparison_blocks_failed_or_non_six_run_formal_validation(
    tmp_path: Path,
    formal_validation: dict[str, object],
) -> None:
    comparison = _write_comparison(
        tmp_path / f"formal-{formal_validation['status']}-{formal_validation['run_count']}",
        _pass_compatibility(),
        formal_validation,
    )
    completed = _invoke(tmp_path, "validate-full", comparison)
    assert completed.returncode != 0
    assert "not an exact six-run PASS" in completed.stderr


@pytest.mark.parametrize(
    "changes",
    (
        {"comparable": False, "critical_mismatches": [{"field": "seed"}]},
        {"comparable": True, "critical_mismatches": [{"field": "dataset"}]},
        {"release_ready": False, "release_blockers": [{"field": "epochs"}]},
        {"release_ready": True, "release_blockers": [{"field": "forged"}]},
        {"release_ready": "true"},
        {"comparable": "true"},
        {"release_blockers": None},
        {"critical_mismatches": None},
        {"run_count": "6"},
        {"run_count": None},
    ),
)
def test_full_comparison_blocks_nonempty_compatibility_gates(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    compatibility = _pass_compatibility()
    compatibility.update(changes)
    comparison = _write_comparison(
        tmp_path / "blocked-compatibility",
        compatibility,
        {"status": "PASS", "run_count": 6},
    )
    completed = _invoke(tmp_path, "validate-full", comparison)
    assert completed.returncode != 0
    assert "unblocked PASS" in completed.stderr


def test_smoke_comparison_blocks_formal_validation_artifact(tmp_path: Path) -> None:
    compatibility = _pass_compatibility()
    compatibility.update(release_ready=False, release_blockers=[{"field": "smoke_runs"}])
    comparison = _write_comparison(
        tmp_path / "smoke-with-formal",
        compatibility,
        {"status": "PASS", "run_count": 6},
    )
    completed = _invoke(tmp_path, "validate-smoke", comparison)
    assert completed.returncode != 0
    assert "Smoke comparison must not produce formal_validation.json" in completed.stderr
