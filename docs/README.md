# 문서 안내

이 저장소는 Windows에서 YOLOX-S/YOLO11 계열 모델을 학습하고, 검증된 가중치와 실험 증빙을
공개 GitHub로 전달한 뒤 Ubuntu 카메라 환경에서 시험하기 위한 작업 공간입니다.

## 먼저 읽을 문서

| 문서 | 목적 |
|---|---|
| [현재 상태](project_status.md) | 완료된 항목, 아직 검증되지 않은 항목, 다음 실행 순서 |
| [Paired two-seed formal release](paired_2seed_formal_release.md) | 완료 seed 42/43의 descriptive-only 검증 절차와 주장 한계 |
| [2026-08-18 학습 진행본](../reports/progress/rpi_bootstrap_2026-08-18/README.md) | 6-run 계획 당시의 역사적 진행 snapshot |
| [전체 로드맵](roadmap.md) | 단계별 Gate, 병행 Track, 구현 backlog와 정식 완료 정의 |
| [Windows 재현 절차](reproducibility_windows.md) | 환경·dataset 재현과 역사적 3-seed 절차의 보존 범위 |
| [데이터 수집 계획](data_plan.md) | class별 1,000장 목표, 공개 데이터 한계, 자체 촬영·split 원칙 |
| [RPi 누수 방지 split](condition_split.md) | condition/pHash component 분할, 수량, hash, 잔여 한계 |
| [YOLO↔COCO 동등성](dataset_equivalence.md) | 두 framework의 image/class/bbox canonical 일치 gate |
| [ONNX 전체 split 검증](onnx_split_evaluation.md) | phash_v2 val/test의 ONNX 공통 metric·native 동등성 gate |
| [핵심 수치 선정 근거](../reports/methodology/parameter_rationale.md) | 14개 baseline 값의 이유·최적값 여부·재조정 조건·검증 상태 |
| [실험 방법론](../reports/methodology/experiment_methodology.md) | YOLOX-S/YOLO11m 알고리즘, 수치 선정 이유, 논문·공식 구현 근거 |
| [결과·증빙 정책](evidence_and_results_policy.md) | 로그 기반 판단, 비교 가능성 gate, 비생성형 그래프와 SHA-256 규칙 |
| [라벨링 규정](annotation_protocol.md) | CVAT, pseudo-label, SAHI, 사람 검수와 gold set 보호 |
| [STM32/SMD P0–P2 반입](smallchip_ingest_p0.md) | 고정 ontology, rights/hash ingest, trust registry, CVAT exact round-trip/승격 gate |
| [Small-chip 공개 데이터 검토](smallchip_source_review.md) | 공식 출처별 license·수량·class mapping·채택/제외 판정 |
| [Ubuntu 인계](ubuntu_handoff.md) | Git LFS clone, ONNX/TensorRT 원칙, 카메라 현장 시험 항목 |

## 단일 진실 공급원

- 역사적 RPi 학습 protocol: [`configs/experiments/baseline_v1.yaml`](../configs/experiments/baseline_v1.yaml)
- Paired two-seed formal policy: [`configs/experiments/rpi_bootstrap_paired_2seed_release_v1.yaml`](../configs/experiments/rpi_bootstrap_paired_2seed_release_v1.yaml)
- RPi test evidence supplement: [`configs/experiments/rpi_test_evidence_supplement_v1.yaml`](../configs/experiments/rpi_test_evidence_supplement_v1.yaml)
- STM32/SMD 단계형 학습: [`configs/experiments/smallchip_staged_training_v1.yaml`](../configs/experiments/smallchip_staged_training_v1.yaml)
- 오토라벨 수치: [`configs/annotation/autolabel_v1.yaml`](../configs/annotation/autolabel_v1.yaml)
- canonical 6-class ID: [`configs/classes.smd_v1.yaml`](../configs/classes.smd_v1.yaml)
- 한국어 표시 sidecar: [`configs/classes.smd_v1.display.ko.yaml`](../configs/classes.smd_v1.display.ko.yaml)
- 공개 데이터 출처: [`configs/datasets.curated.yaml`](../configs/datasets.curated.yaml)
- 오토라벨 승인 registry: [`configs/data_trust_registry.yaml`](../configs/data_trust_registry.yaml)
- 검증된 Windows 환경: [`configs/windows_environment.verified.yaml`](../configs/windows_environment.verified.yaml)

README나 보고서에 적힌 값과 config가 다르면 config와 해당 실행의 `run_manifest.json`을 우선합니다.
실제 실행에서는 CLI override가 적용될 수 있으므로 최종 판단에는 run별 resolved 설정도 같이 봅니다.

현재 검증된 모델 데이터는 Raspberry Pi 1-class bootstrap입니다. Matched seed 42/43 formal comparison과
두 모델의 ONNX val/test gate는 PASS했지만 `n=2`, `df=1`의 기술통계이며 test는 internal pHash split입니다.
Canonical 6-class와 한국어 presentation sidecar의 exact binding은 integration test가 검증합니다. 다만
class contract가 존재한다고 해서 multi-class 승인 데이터나 실제 STM32/SMD 성능이 확보된 것은 아닙니다.

## 저장소에 포함하지 않는 항목

- 원본·가공 이미지와 미승인 라벨
- 전체 `runs/` 작업 폴더와 smoke checkpoint
- TensorRT `.engine`
- API key, token, `.env`, 개인 계정 정보

Policy-driven `release_ready=true`와 publication gate를 모두 통과한 artifact만 Git 추적 폴더에
승격합니다. 역사적 RPi 100-epoch evidence는 변경하지 않으며, 향후 STM32/SMD 학습은 1e smoke,
최대 10e pilot, 최대 50e candidate 및 early stopping을 적용합니다.
