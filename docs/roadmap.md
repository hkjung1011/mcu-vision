# 전체 진행 로드맵과 완료 Gate

## 선정 결정

정식 MCU/SMD 비교 학습보다 `class 동결 → 데이터 확보 → 수동 gold → multi-class 경로 일반화 →
오토라벨 전량 검수 → 3-seed 비교`를 먼저 완료합니다. 현재 Raspberry Pi 1-class run은 학습·보고
pipeline 검증용이며, 최종 제품군 성능으로 해석하지 않습니다.

## 단계별 상태

| 단계 | 현재 상태 | 판정 | 완료 Gate |
|---:|---|---|---|
| 1. Windows/GPU 환경 | RTX 5060 Laptop, PyTorch CUDA, YOLO11/YOLOX smoke 확인 | PASS | 환경 lock·GPU smoke 재현 |
| 2. Class ontology | SMD/STM32 6개 ID와 Dainius alias P0 동결 | PARTIAL | 실제 specimen으로 포함/제외·OCR 정책 승인 |
| 3. Class별 데이터 | RPi 1,875장만 반입; Pico/STM32/SMD 승인본 0장 | NOT VERIFIED | class별 provenance·중복 감사된 실사 목표 충족 |
| 4. Split 독립성 | condition/pHash component 기준 1,500/195/180, cross-split 후보 0 | PASS (BOOTSTRAP) | 새 physical item/session 독립 test |
| 5. 수동 seed/gold | 미시작 | NOT VERIFIED | 대표 200장 전체 instance 라벨, locked val/test 생성 |
| 6. CVAT 검수 경로 | full class-map/image binding verifier와 filtered-COCO 승격 gate 구현 | PARTIAL | 실제 CVAT export에서 왕복 손실 0 확인 |
| 7. Autolabel bootstrap | trusted registry/locked split/teacher/calibration exact binding 구현 | PARTIAL | 승인 registry entry·실제 domain teacher·전량 사람 승인 |
| 8. Multi-class 학습 경로 | dataset CLI·dynamic class 수·canonical equivalence 구현 | PARTIAL | 승인 multi-class data로 양쪽 smoke |
| 9. 정식 비교 | full 3/6 완료, 1/6 중단, 2/6 미시작 | PARTIAL | 2 models × 3 seeds × 100 epochs, protocol PASS |
| 10. Git release | 2026-08-18 public progress snapshot·LFS weight 게시 | PARTIAL | 6-run trained checkpoint·report·SHA-256 정식 승격 |
| 11. Ubuntu 시험 | 인계 문서만 존재 | NOT VERIFIED | ONNX 동등성, 카메라 E2E 정확도·p50/p95·FPS 측정 |

## 데이터 현황과 해석

| 대상 | 승인 수량 | 현재 근거 | 잔여 위험·조치 |
|---|---:|---|---|
| `raspberry_pi_sbc` | 1,875장 | micro-PCB 3개 RPi 모델의 반복 관측 | v2 cross-split pHash 후보 0; physical item 독립 test는 여전히 필요 |
| `raspberry_pi_pico` | 0장 | 공개 후보만 조사 | 자체 촬영과 provenance manifest 필요 |
| `stm32_dev_board` | 0장 | 공개 독립 실사 1,000장 미확인 | Nucleo/Discovery 등 포함 범위부터 동결 |
| `stm32_bare_ic` | 0장 | 정확한 대규모 공개 원출처 미확인 | package 검출과 top-mark OCR을 분리하고 자체 촬영 |
| Dainius SMD 4-class | 0장 | Roboflow raw v2 후보와 PDM assertion만 기록 | 인증 후 원본 반입·archive/hash 감사, 실제 컨베이어 촬영 병행 |

`near-duplicate 3,511쌍`은 삭제 대상이 아니라 pHash 기반 **검토 후보**입니다. v2는 후보 연결요소와
condition group을 원자적으로 배정해 cross-split pair를 0으로 만들었습니다. 다만 원본에 physical
specimen ID가 없으므로 이 분할은 새 보드·새 카메라 일반화를 입증하지 않습니다.

## 두 개의 병행 Track

### Track A — 기존 RPi로 실험 pipeline 검증

1. v2 manifest·class map·image list·YOLO/COCO label 동등성 hash를 run마다 fail-fast 재검증합니다.
2. seed42 두 모델과 YOLO11m seed43은 완료했습니다. YOLOX-S seed43을 재개하고 seed44 두 run을 실행합니다.
3. 6-run release gate가 PASS한 비교만 Git에 승격합니다.
4. 이 결과는 RPi bootstrap 시스템 비교로만 표기하고 SMD 성능으로 확장 해석하지 않습니다.

### Track B — 실제 목표 MCU/SMD dataset 구축

1. `보드 검출`, `package 종류 검출`, `exact SKU/marking 인식`을 분리합니다.
2. class별 촬영 폴더와 `physical_item_id`, `capture_session`, `video_run_id`, `lot_id`를 기록합니다.
3. 대표 200장에는 일부가 아니라 보이는 모든 target instance를 수동 라벨링합니다.
4. 자동 제안이 없는 locked gold validation을 만들고, pseudo-label은 train 후보에만 사용합니다.
5. class·annotation·split hash를 동결한 뒤 정식 multi-class 학습을 시작합니다.

## 구현 Backlog

### P0 — 정식 multi-class 학습 전 필수

- RPi 전용 path와 `num_classes=1` 하드코딩 제거 — **구현 완료, multi-class data smoke 대기**
- canonical COCO/YOLO dataset과 class map을 config/CLI로 주입 — **구현 완료**
- dataset manifest·class map·image list hash와 YOLO↔COCO box/class 동등성 validator — **PASS**
- condition/pHash component leakage 감사 — **v2 PASS**, physical specimen ID는 미제공
- 두 framework의 class-aware NMS(`class_agnostic_nms=false`)와 top-K 조건을 accuracy/latency에 동일 적용
- CVAT full class-map/image-binding verifier와 filtered canonical COCO 승격 gate — **코드 구현, 실제 export 검증 대기**
- project trusted registry와 locked split 기준 validation/test SHA 재선언 차단 — **코드 PASS, 승인 entry 0건**
- STM32 specimen/part-number evidence와 dataset별 source class allowlist — **코드 PASS, 실제 data 대기**
- train/val/test exact image SHA leakage와 RPi 9-hash formal evidence — **PASS**, physical specimen ID는 미제공
- 동일 파일 stem의 `.jpg/.png` label 충돌 방지

### P1 — 오토라벨 효율·소형 객체 성능

- domain YOLO teacher calibration provenance 확인
- tile batching, tile 경계 중복, 밀집 객체 NMS 감사
- `640 full-frame` / `640 tile` / 필요 시 `960` pilot
- 전처리 후 bbox pixel-width/height/area bin별 recall 또는 AP 보고
- 사람 수정시간, proposal recall, 빈 예측·저신뢰 queue 수치화
- 필요할 때만 Grounding DINO/SAM2 또는 SAHI adapter를 별도 실험

### P2 — Ubuntu 전달

- native checkpoint → ONNX export와 동일 test subset 수치 비교
- Ubuntu dependency lock과 camera adapter
- 목표 장치 batch 1 capture+preprocess+inference+NMS p50/p95 latency·FPS·VRAM/RAM
- 필요 시 목표 Ubuntu 장치에서 TensorRT engine 생성·검증

## 정식 결과의 완료 정의

| REQ ID | 검증 근거 | 완료 기준 | 현재 |
|---|---|---|---|
| REQ-DATA-01 | provenance manifest·audit·split hash | class 규정과 승인 데이터가 동결됨 | NOT VERIFIED |
| REQ-DATA-02 | canonical record hash·format round-trip report | YOLO와 COCO의 image/class/box가 동일함 | RPi v2 PASS |
| REQ-LABEL-01 | CVAT export·reviewer·label SHA-256 | gold와 reviewed train label의 출처 추적 가능 | NOT VERIFIED |
| REQ-MC-01 | 두 run의 resolved config·dataset hash | 동일 canonical multi-class dataset 사용 | 코드 PASS / data 대기 |
| REQ-TR-01 | 6개 complete run manifest | seed 42/43/44, 2개 모델, 100 epochs | 3 PASS / 1 INTERRUPTED / 2 NOT STARTED |
| REQ-EV-01 | common evaluator JSON/CSV | AP50-95/AP50/AP75/AP_small/AR100와 P/R/F1 생성 | 완료 run 3개 PASS / matrix 미완료 |
| REQ-EV-02 | latency sample·GPU log | 같은 장치에서 batch 1 p50/p95·FPS·VRAM | 완료 run 3개 PASS / repeat 미완료 |
| REQ-GIT-01 | artifact manifest·LFS·remote commit | 수치·weight SHA-256와 비생성형 그래프 승격 | progress snapshot 준비 / formal release 없음 |
| REQ-UB-01 | 새 카메라 test report | 정확도와 E2E 성능을 독립 session에서 확인 | NOT VERIFIED |

## 앞으로 필요한 사용자 입력

다음 세 항목은 실제 데이터 촬영·Ubuntu 단계 전에 확정해야 합니다.

1. 정확히 구분할 대상: 보드 종류, package 종류, exact part number/OCR 중 어디까지 필요한지
2. 촬영 가능한 실제 칩·보드 수와 lot, 카메라 해상도·렌즈·조명·컨베이어 속도
3. Ubuntu 목표 장치의 GPU/Jetson/CPU, 목표 FPS와 허용 miss/false-positive 기준

나머지 코드 일반화·문서·검증 자동화는 이 세 항목을 기다리지 않고 진행할 수 있습니다.
