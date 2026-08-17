from __future__ import annotations

import argparse
import csv
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .common import sha256_file, write_json
from .runlog import configure_utf8_output


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repository_path(path: Path) -> str:
    """Return a portable path without exposing the local Windows user directory."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Protocol must be a YAML mapping: {path}")
    return document


def _flatten_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _rationale_rows(document: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in document.get("rationale", []):
        rows.append(
            {
                "id": str(item.get("id", "")),
                "item": str(item.get("item", "")),
                "label": str(item.get("label", item.get("item", ""))),
                "selected_value": _flatten_value(item.get("selected_value", "")),
                "status": str(item.get("status", "")),
                "reason": str(item.get("reason", "")).strip(),
                "adjustment_rule": str(item.get("adjustment_rule", "")).strip(),
                "optimality": str(item.get("optimality", "")).strip(),
                "verification_status": str(item.get("verification_status", "")).strip(),
                "references": ", ".join(str(value) for value in item.get("references", [])),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "id",
        "item",
        "label",
        "selected_value",
        "status",
        "reason",
        "adjustment_rule",
        "optimality",
        "verification_status",
        "references",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _protocol_table(document: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for section in ("common", "yolo11m", "yolox_s", "comparison_rules"):
        for key, value in document.get(section, {}).items():
            rows.append((section, key, _flatten_value(value)))
    return rows


def _write_methodology_markdown(
    path: Path,
    document: dict[str, Any],
    rows: list[dict[str, str]],
    source_path: Path,
) -> None:
    lines = [
        "# 실험 방법 및 수치 선정 근거",
        "",
        f"- protocol: `{document.get('protocol_id', '-')}`",
        f"- 상태: `{document.get('status', '-')}`",
        f"- 비교 유형: `{document.get('experiment_type', '-')}`",
        f"- 현재 task: `{document.get('task', '-')}`",
        f"- 원본 config SHA256: `{sha256_file(source_path)}`",
        "",
        "> 현재 protocol은 Raspberry Pi SBC 1-class bootstrap 전용입니다. Provisional MCU/SMD "
        "multi-class 학습 protocol이 아닙니다.",
        "> 이 문서는 설정 파일에서 자동 생성되었습니다. Validation 결과는 독립적인 실제 컨베이어 "
        "test 결과가 아니며, 이 benchmark는 순수 architecture ablation이 아닙니다.",
        "> 판단 근거는 YAML/CSV/JSON의 수치이며, PNG는 matplotlib로 렌더링한 비생성형 파생물입니다. "
        "ImageGen 등 생성형 AI는 사용하지 않습니다.",
        "",
        "## 고정 protocol",
        "",
        "| 구역 | 항목 | 값 |",
        "|---|---|---|",
    ]
    for section, key, value in _protocol_table(document):
        lines.append(f"| {section} | {key} | `{value.replace('|', '/ ')}` |")
    lines.extend(
        [
            "",
            "## 선정 근거와 조정 조건",
            "",
            "| ID | 항목 | 선택값 | 근거 상태 | 선정 이유 | 최적값 여부 | 재조정 조건 | 현재 검증 상태 | 근거 ID |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        values = [
            row["id"],
            row["item"],
            row["selected_value"],
            row["status"],
            row["reason"],
            row["optimality"],
            row["adjustment_rule"],
            row["verification_status"],
            row["references"],
        ]
        escaped = [value.replace("|", "/ ").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(["", "## 학습 알고리즘과 해석 범위", ""])
    lines.extend(
        [
            "- **YOLOX-S**: anchor-free one-stage detector, decoupled classification/regression "
            "head, SimOTA dynamic label assignment, BCE classification/objectness와 IoU regression을 "
            "사용합니다.",
            "- **YOLO11m**: 고정된 Ultralytics 구현의 anchor-free detector이며 box, class, DFL "
            "loss로 학습합니다. YOLO11 자체의 별도 peer-reviewed 논문은 없으므로 공식 문서와 "
            "고정 버전 source를 근거로 사용합니다.",
            "- 두 framework의 native loss 정의와 optimizer dynamics가 다르므로 raw loss 절대값은 "
            "서로 비교하지 않습니다. AP/AR는 동일 COCOeval, 운영점 P/R/F1은 동일 greedy matcher로 "
            "계산하고 실제 장치 latency를 함께 비교합니다.",
            "- `batch=8`은 같은 micro-batch/VRAM 조건입니다. YOLO11의 gradient accumulation 때문에 "
            "effective optimizer batch까지 같다는 뜻은 아닙니다.",
            "",
            "## 참고문헌·공식 구현",
            "",
        ]
    )
    for reference in document.get("references", []):
        lines.append(
            f"- **{reference.get('id', '-')}** — [{reference.get('title', '-')}]"
            f"({reference.get('url', '')}) ({reference.get('type', '-')})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_parameter_summary_markdown(
    path: Path,
    document: dict[str, Any],
    rows: list[dict[str, str]],
    source_path: Path,
) -> None:
    lines = [
        "# 핵심 수치 선정 이유와 검증 상태",
        "",
        f"- protocol: `{document.get('protocol_id', '-')}`",
        f"- 현재 task: `{document.get('task', '-')}`",
        f"- config SHA256: `{sha256_file(source_path)}`",
        "",
        "> 현재 protocol은 Raspberry Pi SBC 1-class bootstrap 전용이며 MCU/SMD multi-class 결과가 아닙니다.",
        "> 아래 값은 재현 가능한 1차 baseline이며 최적값이 아닙니다. 현재 결과는 validation 범위이고, "
        "독립적인 실제 카메라 test는 아직 없습니다.",
        "",
        "| ID·항목 | 값 | 왜 선택했는가 | 근거 유형 | 최적값 여부 | 다음 조정 조건 | 현재 검증 상태 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [
            f"{row['id']} {row['label']}",
            row["selected_value"],
            row["reason"],
            row["status"],
            row["optimality"],
            row["adjustment_rule"],
            row["verification_status"],
        ]
        escaped = [value.replace("|", "/ ").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## 해석 원칙",
            "",
            "- `PAPER_*`/`UPSTREAM_*`은 출처가 있다는 뜻이지 현재 MCU/SMD에서 최적이라는 뜻이 아닙니다.",
            "- `HARDWARE_DERIVED`는 현재 RTX 5060 Laptop 8 GB에서 실행 가능한 값입니다.",
            "- `ENGINEERING_BASELINE`/`TO_TUNE`은 gold validation과 실제 카메라 조건으로 다시 정해야 합니다.",
            "- 상세 알고리즘·참고문헌은 [전체 방법론](experiment_methodology.md), 실제 값은 "
            "[`configs/experiments/baseline_v1.yaml`](../../configs/experiments/baseline_v1.yaml)을 봅니다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_rationale(
    rows: list[dict[str, str]],
    path: Path,
    *,
    source_label: str,
    source_sha256: str,
    task: str,
) -> None:
    if not rows:
        return
    lines = [
        "MCU VISION — PROTOCOL RATIONALE",
        "PAPER/DOC = cited upstream evidence; ENGINEERING/TO_TUNE = project decision",
        f"TASK: {task} | SCOPE: Raspberry Pi SBC 1-class bootstrap",
        f"SOURCE: {source_label} | SHA256: {source_sha256}",
        f"RENDERER: matplotlib {matplotlib.__version__} | GENERATIVE_AI: false",
        "",
    ]
    for row in rows:
        title = f"{row['id']} | {row['item']} | {row['selected_value']} | {row['status']}"
        lines.append(title)
        lines.extend("  " + line for line in textwrap.wrap(row["reason"], width=108))
        lines.append("  refs: " + (row["references"] or "-"))
        lines.append("")
    height = max(6.0, len(lines) * 0.16)
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font = (
        font_manager.FontProperties(fname=str(font_path))
        if font_path.exists()
        else font_manager.FontProperties(family="sans-serif")
    )
    fig = plt.figure(figsize=(15, height), facecolor="#111318")
    fig.text(
        0.025,
        0.985,
        "\n".join(lines),
        fontproperties=font,
        fontsize=9.5,
        color="#e6edf3",
        va="top",
    )
    plt.axis("off")
    fig.savefig(
        path,
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        metadata={
            "Title": "MCU Vision protocol rationale",
            "Source": source_label,
            "SourceSHA256": source_sha256,
            "Renderer": f"matplotlib {matplotlib.__version__}",
            "GenerativeAI": "false",
        },
    )
    plt.close(fig)


def print_protocol_rationale(document: dict[str, Any]) -> None:
    rows = _rationale_rows(document)
    print("\nPROTOCOL RATIONALE (exact reasons are also saved as CSV/Markdown)")
    print("=" * 118)
    print(f"{'ID':<5} {'ITEM':<34} {'SELECTED VALUE':<34} {'EVIDENCE STATUS':<35}")
    print("-" * 118)
    for row in rows:
        selected = textwrap.shorten(row["selected_value"], width=32, placeholder="...")
        print(f"{row['id']:<5} {row['item']:<34.34} {selected:<34} {row['status']:<35.35}")
        for line in textwrap.wrap(row["reason"], width=106):
            print("      " + line)
        print(f"      adjustment: {row['adjustment_rule']}")
        print(f"      refs: {row['references'] or '-'}")
    print("NOTICE: cited defaults are starting points, not proof that they are optimal for MCU/SMD data.")


def write_protocol_artifacts(
    source_path: Path,
    output_dir: Path,
    *,
    print_terminal: bool = True,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document = load_protocol(source_path)
    rows = _rationale_rows(document)
    source_sha256 = sha256_file(source_path)
    source_label = _repository_path(source_path)
    snapshot = output_dir / "protocol_snapshot.yaml"
    if source_path != snapshot.resolve():
        shutil.copy2(source_path, snapshot)
    _write_csv(output_dir / "protocol_rationale.csv", rows)
    write_json(output_dir / "protocol_references.json", document.get("references", []))
    _write_methodology_markdown(
        output_dir / "experiment_methodology.md", document, rows, source_path
    )
    _write_parameter_summary_markdown(
        output_dir / "parameter_rationale.md", document, rows, source_path
    )
    _plot_rationale(
        rows,
        output_dir / "protocol_rationale.png",
        source_label=source_label,
        source_sha256=source_sha256,
        task=str(document.get("task", "-")),
    )
    artifact_paths = [
        snapshot,
        output_dir / "protocol_rationale.csv",
        output_dir / "protocol_references.json",
        output_dir / "experiment_methodology.md",
        output_dir / "parameter_rationale.md",
        output_dir / "protocol_rationale.png",
    ]
    result = {
        "schema_version": 1,
        "protocol_id": document.get("protocol_id"),
        "source": source_label,
        "source_sha256": source_sha256,
        "rationale_items": len(rows),
        "references": len(document.get("references", [])),
        "output_dir": _repository_path(output_dir),
        "policy": "Judgments use YAML/CSV/JSON numeric evidence; PNG files are visualization derivatives.",
        "generative_ai_used_for_images": False,
        "renderer": f"matplotlib {matplotlib.__version__}",
        "artifacts": [
            {
                "path": _repository_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "derived_visual" if path.suffix.lower() == ".png" else "evidence",
            }
            for path in artifact_paths
        ],
    }
    write_json(output_dir / "protocol_artifacts.json", result)
    if print_terminal:
        print_protocol_rationale(document)
        print(f"\nProtocol evidence artifacts: {output_dir.resolve()}")
    return result


def main(argv: list[str] | None = None) -> None:
    configure_utf8_output()
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Print the experiment rationale and render evidence artifacts"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "experiments" / "baseline_v1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "methodology")
    args = parser.parse_args(argv)
    write_protocol_artifacts(args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
