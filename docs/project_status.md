# 현재 프로젝트 상태

기준일: **2026-08-18**

## 목표

Windows RTX GPU에서 Raspberry Pi/STM32/소형 전자부품 detector를 fine-tune하고, 학습 과정·가중치·평가
근거를 공개 GitHub에 보존한 뒤 Ubuntu 목표 장치에서 실제 카메라 입력으로 검증합니다. Exact part
number나 표면 marking 인식이 필요하면 detector 뒤에 고해상도 crop classifier/OCR 단계를 별도로 둡니다.

## 상태 요약

| 항목 | 값 | 상태 | 근거/비고 |
|---|---|---|---|
| Windows GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB | CONFIRMED | `configs/windows_environment.verified.yaml` |
| PyTorch | 2.12.1+cu130, CUDA 사용 가능 | CONFIRMED | YOLO11 smoke 및 YOLOX CUDA forward 통과 |
| 별도 CUDA Toolkit | 현재 학습에는 불필요 | CONFIRMED | PyTorch wheel CUDA runtime 사용; `nvcc` custom build 시 재검토 |
| YOLO11 | Ultralytics 8.4.120 | CONFIRMED | 별도 YOLO11 peer-reviewed 논문이 없어 고정 버전 공식 source 사용 |
| YOLOX | 0.3.0, commit `6ddff482...` | CONFIRMED | pretrained YOLOX-S checkpoint SHA-256 기록 |
| Raspberry Pi bootstrap data | train/validation/test 1,500/195/180 | PASS (BOOTSTRAP) | 모델별 500/65/60; physical specimen 독립성은 미확인 |
| RPi condition overlap | condition group 및 pHash component cross-split 0 | PASS | v2 split assignment SHA-256 고정 |
| Raspberry Pi near-duplicate | pHash 후보 3,511쌍, cross-split 0 | PASS (CONSERVATIVE GROUPING) | 후보 연결요소 전체를 같은 split에 배정 |
| 현재 학습 범위 | Raspberry Pi 1-class bootstrap | PARTIAL | 실제 MCU/SMD 승인 데이터는 아직 없음 |
| Provisional multi-class 경로 | dataset CLI, dynamic class 수, canonical equivalence gate 구현 | PARTIAL | 실제 multi-class dataset smoke 필요 |
| SMD/STM32 ingest gate | dataset allowlist, exact alias, STM32 specimen evidence, ZIP resource/atomic gate | CODE PASS | 실제 archive 반입 0장 |
| 소형 SMD 실제 데이터 | trusted registry 승인 canonical dataset 0장 | NOT VERIFIED | 공개 raw 데이터 다운로드·감사 및 자체 촬영 필요 |
| 오토라벨 신뢰 경계 | project registry/locked split/exact image binding, finite calibration | CODE PASS / DATA BLOCKED | empty registry에 의해 실제 proposal fail closed |
| CVAT 승격 | exact reference/full class map/all-reviewed 및 rejected filtering | CODE PASS | 실제 CVAT export 검증 필요 |
| full 3-seed 학습 | 완료 3/6; YOLOX-S seed43 70/100 중단; seed44 2개 미시작 | PARTIAL | seed 42/43/44 동일 protocol 완료 필요 |
| 독립 컨베이어 test | 없음 | NOT VERIFIED | Ubuntu 카메라로 새 session을 촬영해 test 고정 필요 |
| 공개 progress weight | 완료 best 3개 + 중단 best/resume 2개 | INTERIM_PROGRESS | metadata 비식별·SHA·load 검증, `release_ready=false` |
| 배포용 trained weight | 없음 | NOT VERIFIED | 6-run formal comparison·ONNX val/test 뒤 승격 |
| 정식 release gate | 6-run·100-epoch·protocol 고정·canonical dataset/checkpoint hash 강제 | IMPLEMENTED / AWAITING RUNS | YOLO↔COCO 1,500/195 동등성 PASS |
| Ubuntu 카메라 연동 | 미실행 | NOT VERIFIED | 목표 GPU/CPU, 카메라 해상도, FPS 확정 후 진행 |

## 현재 결과의 해석

현재 [공개 진행본](../reports/progress/rpi_bootstrap_2026-08-18/README.md)은 smoke가 아니라 full-data
학습 3개와 중단 run 1개의 실제 기록입니다. 그러나 계획한 6-run matrix가 완료되지 않아
`release_ready=false`이며 mean ± SD 기반의 정식 모델 비교는 아직 할 수 없습니다. 완료 run의 common
validation 수치와 중단 run의 native epoch 수치도 직접 같은 성능 열로 비교하지 않습니다.

## 완료 판정 조건

| REQ ID | 검증 조건 | 현재 판정 |
|---|---|---|
| REQ-TR-01 | 두 모델 모두 동일 dataset hash·640·batch 8·100 epochs·seed 42/43/44 완료 | 3/6 PASS, 1 INTERRUPTED, 2 NOT STARTED |
| REQ-EV-01 | common COCO evaluator에서 AP50-95/AP50/AP75/AP_small/AR100 산출 | 완료 run 3개 PASS / matrix 미완료 |
| REQ-EV-02 | batch 1 p50/p95 latency, FPS, VRAM을 동일 GPU에서 측정 | 완료 run 3개 PASS / thermal repeat 미완료 |
| REQ-EV-03 | `terminal.log`, CSV/JSON, checkpoint SHA-256, config hash 보존 | PUBLIC PROGRESS PASS / formal 승격 미완료 |
| REQ-DATA-01 | class별 승인된 고유 실사 1,000장 목표와 provenance 확보 | NOT VERIFIED |
| REQ-DATA-02 | physical item/session 기준 split과 YOLO↔COCO label 동등성 hash | RPi 9-hash 및 file SHA split gate PASS / physical item NOT VERIFIED |
| REQ-MC-01 | 임의 canonical dataset/class 수를 두 framework가 동일하게 학습·평가 | 코드 PASS / multi-class data NOT VERIFIED |
| REQ-AUTO-01 | YOLO11 proposal을 `pending` 분리 출력 | 코드 PASS / trusted locked split에 승인 데이터 0장이라 실행 차단 |
| REQ-AUTO-02 | CVAT round-trip 및 reviewer/hash 기반 강제 승인 gate | 코드 PASS / rejected filtering 포함, 실제 CVAT export NOT VERIFIED |
| REQ-UB-01 | Ubuntu 실제 카메라 test set에서 정확도와 end-to-end latency 측정 | NOT VERIFIED |

## 다음 실행 순서

1. YOLOX-S seed43을 100 epoch까지 재개 또는 clean rerun하고 seed44 두 모델을 완료합니다.
2. 6-run 공통 평가와 release gate가 PASS한 최적 weight·ONNX·로그 기반 보고서를 정식 승격합니다.
3. 보드 검출·package 검출·exact marking OCR의 범위와 provisional class 포함/제외 규칙을 동결합니다.
4. 소형 SMD raw 데이터와 사용자 촬영본을 수집하고 license/중복/specimen/session ID를 감사합니다.
5. 대표 이미지 200장에 보이는 모든 목표 instance를 수동 라벨링하고 locked gold validation을 만듭니다.
6. CVAT import/export를 검증한 뒤 1차 teacher의 tiled proposal을 전량 사람이 수정·승인합니다.
7. `640 full-frame`과 `640 tile`, 필요 시 `960`을 pilot으로 비교하고 multi-class protocol을 동결합니다.
8. Ubuntu에서 Git LFS weight를 내려받고 실제 카메라 test를 수행합니다.

각 단계의 종료 조건과 구현 backlog는 [전체 로드맵](roadmap.md)에서 추적합니다.

## 주장 경계

- Validation AP를 실제 컨베이어 최종 정확도라고 부르지 않습니다.
- YOLO11m과 YOLOX-S는 모델 규모와 native optimizer dynamics가 달라 순수 architecture 우열로 해석하지 않습니다.
- 서로 다른 loss 정의의 절대값을 모델 간 직접 비교하지 않습니다.
- 외형만으로 exact STM32 SKU나 전기적 기능을 확정하지 않습니다.
