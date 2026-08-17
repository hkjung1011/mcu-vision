# 현재 프로젝트 상태

기준일: **2026-08-17**

## 목표

Windows RTX GPU에서 Raspberry Pi/STM32/소형 전자부품 detector를 fine-tune하고, 학습 과정·가중치·평가
근거를 private GitHub에 보존한 뒤 Ubuntu 목표 장치에서 실제 카메라 입력으로 검증합니다. Exact part
number나 표면 marking 인식이 필요하면 detector 뒤에 고해상도 crop classifier/OCR 단계를 별도로 둡니다.

## 상태 요약

| 항목 | 값 | 상태 | 근거/비고 |
|---|---|---|---|
| Windows GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB | CONFIRMED | `configs/windows_environment.verified.yaml` |
| PyTorch | 2.12.1+cu130, CUDA 사용 가능 | CONFIRMED | YOLO11 smoke 및 YOLOX CUDA forward 통과 |
| 별도 CUDA Toolkit | 현재 학습에는 불필요 | CONFIRMED | PyTorch wheel CUDA runtime 사용; `nvcc` custom build 시 재검토 |
| YOLO11 | Ultralytics 8.4.120 | CONFIRMED | 별도 YOLO11 peer-reviewed 논문이 없어 고정 버전 공식 source 사용 |
| YOLOX | 0.3.0, commit `6ddff482...` | CONFIRMED | pretrained YOLOX-S checkpoint SHA-256 기록 |
| Raspberry Pi bootstrap data | train 1,500 / validation 375 | CONFIRMED | specimen 기준 source split; 독립 test는 아님 |
| 소형 SMD 실제 데이터 | 아직 canonical dataset에 없음 | NOT VERIFIED | 공개 raw 데이터 다운로드·감사 및 자체 촬영 필요 |
| full 3-seed 비교 | 아직 없음 | NOT VERIFIED | seed 42/43/44, 동일 protocol로 실행 필요 |
| 독립 컨베이어 test | 없음 | NOT VERIFIED | Ubuntu 카메라로 새 session을 촬영해 test 고정 필요 |
| 배포용 trained weight | 없음 | NOT VERIFIED | 현재 Git LFS에는 YOLOX-S pretrained weight만 존재 |
| Ubuntu 카메라 연동 | 미실행 | NOT VERIFIED | 목표 GPU/CPU, 카메라 해상도, FPS 확정 후 진행 |

## 현재 결과의 해석

현재 생성된 smoke run은 로깅·학습·평가 배선 검증용입니다. 일부 smoke 비교는 `fraction`, `batch` 등의
조건이 달라 protocol gate가 `NOT COMPARABLE`이므로 어느 모델이 더 우수하다는 근거로 사용하지 않습니다.
GitHub의 `reports/`에도 이런 smoke 수치를 정식 성능 결과로 승격하지 않습니다.

## 완료 판정 조건

| REQ ID | 검증 조건 | 현재 판정 |
|---|---|---|
| REQ-TR-01 | 두 모델 모두 동일 dataset hash·640·batch 8·100 epochs·seed 42/43/44 완료 | NOT VERIFIED |
| REQ-EV-01 | common COCO evaluator에서 AP50-95/AP50/AP75/AP_small/AR100 산출 | 구현 PASS, full run NOT VERIFIED |
| REQ-EV-02 | batch 1 p50/p95 latency, FPS, VRAM을 동일 GPU에서 측정 | 구현 PASS, full run NOT VERIFIED |
| REQ-EV-03 | `terminal.log`, CSV/JSON, checkpoint SHA-256, config hash 보존 | LOCAL PASS / Git 승격 재검증 필요 |
| REQ-DATA-01 | class별 승인된 고유 실사 1,000장 목표와 provenance 확보 | NOT VERIFIED |
| REQ-AUTO-01 | YOLO11 proposal을 `pending` 분리 출력 | LOCAL PASS / val·test source 강제 차단 없음 |
| REQ-AUTO-02 | CVAT round-trip 및 reviewer/hash 기반 강제 승인 gate | NOT VERIFIED |
| REQ-UB-01 | Ubuntu 실제 카메라 test set에서 정확도와 end-to-end latency 측정 | NOT VERIFIED |

## 다음 실행 순서

1. 소형 SMD raw 데이터와 사용자 촬영본을 수집하고 license/중복/specimen ID를 감사합니다.
2. 대표 이미지 200장에 보이는 모든 목표 instance를 수동 라벨링합니다.
3. 1차 YOLO11 teacher로 tiled proposal을 생성하고, CVAT import/export를 먼저 검증한 뒤 전량 검수합니다.
4. 고정된 dataset hash로 YOLOX-S와 YOLO11m을 seed 42/43/44에서 순차 학습합니다.
5. protocol gate가 PASS인 비교만 `reports/comparisons/`와 `weights/trained/`로 승격합니다.
6. Ubuntu에서 Git LFS weight를 내려받고 실제 카메라 test를 수행합니다.

## 주장 경계

- Validation AP를 실제 컨베이어 최종 정확도라고 부르지 않습니다.
- YOLO11m과 YOLOX-S는 모델 규모와 native optimizer dynamics가 달라 순수 architecture 우열로 해석하지 않습니다.
- 서로 다른 loss 정의의 절대값을 모델 간 직접 비교하지 않습니다.
- 외형만으로 exact STM32 SKU나 전기적 기능을 확정하지 않습니다.
