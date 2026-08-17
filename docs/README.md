# 문서 안내

이 저장소는 Windows에서 YOLOX-S/YOLO11 계열 모델을 학습하고, 검증된 가중치와 실험 증빙을
private GitHub로 전달한 뒤 Ubuntu 카메라 환경에서 시험하기 위한 작업 공간입니다.

## 먼저 읽을 문서

| 문서 | 목적 |
|---|---|
| [현재 상태](project_status.md) | 완료된 항목, 아직 검증되지 않은 항목, 다음 실행 순서 |
| [전체 로드맵](roadmap.md) | 단계별 Gate, 병행 Track, 구현 backlog와 정식 완료 정의 |
| [Windows 재현 절차](reproducibility_windows.md) | fresh clone부터 3-seed 실행·Git 승격까지 |
| [데이터 수집 계획](data_plan.md) | class별 1,000장 목표, 공개 데이터 한계, 자체 촬영·split 원칙 |
| [핵심 수치 선정 근거](../reports/methodology/parameter_rationale.md) | 12개 baseline 값의 이유·최적값 여부·재조정 조건·검증 상태 |
| [실험 방법론](../reports/methodology/experiment_methodology.md) | YOLOX-S/YOLO11m 알고리즘, 수치 선정 이유, 논문·공식 구현 근거 |
| [결과·증빙 정책](evidence_and_results_policy.md) | 로그 기반 판단, 비교 가능성 gate, 비생성형 그래프와 SHA-256 규칙 |
| [라벨링 규정](annotation_protocol.md) | CVAT, pseudo-label, SAHI, 사람 검수와 gold set 보호 |
| [Ubuntu 인계](ubuntu_handoff.md) | Git LFS clone, ONNX/TensorRT 원칙, 카메라 현장 시험 항목 |

## 단일 진실 공급원

- 학습·평가 수치: [`configs/experiments/baseline_v1.yaml`](../configs/experiments/baseline_v1.yaml)
- 오토라벨 수치: [`configs/annotation/autolabel_v1.yaml`](../configs/annotation/autolabel_v1.yaml)
- provisional class: [`configs/classes.provisional.yaml`](../configs/classes.provisional.yaml)
- 공개 데이터 출처: [`configs/datasets.curated.yaml`](../configs/datasets.curated.yaml)
- 검증된 Windows 환경: [`configs/windows_environment.verified.yaml`](../configs/windows_environment.verified.yaml)

README나 보고서에 적힌 값과 config가 다르면 config와 해당 실행의 `run_manifest.json`을 우선합니다.
실제 실행에서는 CLI override가 적용될 수 있으므로 최종 판단에는 run별 resolved 설정도 같이 봅니다.

현재 training wrapper와 YOLOX experiment는 Raspberry Pi 1-class bootstrap 기준입니다. 5개 provisional
class가 존재한다고 해서 multi-class 학습 경로까지 완료된 것은 아니며, 이 차이는
[전체 로드맵](roadmap.md)의 `REQ-MC-01`로 추적합니다.

## 저장소에 포함하지 않는 항목

- 원본·가공 이미지와 미승인 라벨
- 전체 `runs/` 작업 폴더와 smoke checkpoint
- TensorRT `.engine`
- API key, token, `.env`, 개인 계정 정보

완료되고 `release_ready=true` gate를 통과한 run만 `scripts/promote_run.py`와
`scripts/promote_comparison.py`로 Git 추적 폴더에 승격합니다.
