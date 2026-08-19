# 전체 진행 로드맵과 완료 Gate

## 선정 결정

RPi bootstrap은 `paired_2seed_descriptive` formal 및 ONNX deployment 검증으로 종료합니다. 과거 완료
100-epoch run과 immutable protocol은 보존하며 추가 RPi 100-epoch 학습을 실행하지 않습니다. 이후 자원은
실제 목표인 STM32/SMD dataset, 사람 검수, 단계형 학습 및 Ubuntu 독립 촬영 시험에 배분합니다.

## 단계별 상태

| 단계 | 현재 상태 | 판정 | 다음 Gate |
|---:|---|---|---|
| 1. Windows/GPU 환경 | 두 framework CUDA 및 test suite | **PASS** | dependency lock 유지 |
| 2. RPi data/evidence | immutable base + versioned test sidecar, pHash cross-split 0 | **PASS (internal)** | 새 physical item/session test |
| 3. RPi formal comparison | 2 models × seeds 42/43, `n=2`, `df=1` | **PASS (descriptive-only)** | 추가 학습 없음 |
| 4. RPi ONNX | 두 모델 native↔ONNX val/test gate | **PASS** | Ubuntu runtime·camera 시험 |
| 5. Class ontology | 6 canonical IDs와 presentation-only 한국어 sidecar | **CONTRACT PASS** | specimen 기반 포함/제외 승인 |
| 6. Small-chip data | ingest·trust·hash·specimen gate 구현, 승인 data 0장 | **DATA WAITING** | 실제 archive 반입 감사 |
| 7. Gold/CVAT | exact image/class/reviewer gate 구현 | **CODE PASS** | 실제 200장과 CVAT round-trip |
| 8. Autolabel | teacher/calibration/locked-split binding 구현 | **CODE PASS** | 실제 teacher proposal 전량 검수 |
| 9. STM32/SMD staged training | versioned 1e/10e/50e contract | **PLANNED** | 단계별 promotion gate |
| 10. Ubuntu independent test | 미실행 | **NOT VERIFIED** | 새 camera session E2E 측정 |

## Track A — RPi bootstrap 종료 및 보존

1. Historical `baseline_v1`과 hash-bound evidence bytes를 변경하지 않습니다.
2. 완료된 matched seed 42/43 네 run만 paired descriptive policy에 포함합니다.
3. Seed 44는 완료 여부와 관계없이 두 모델 모두 비교 입력에서 제외합니다.
4. 허용 주장은 per-run metric, mean, sample SD, paired delta로 제한합니다.
5. 내부 pHash test와 ONNX PASS를 independent test 또는 production readiness로 확대하지 않습니다.

## Track B — STM32/SMD dataset 구축

1. `board detection`, `package detection`, `marking OCR/exact SKU`를 서로 다른 task로 정의합니다.
2. 각 이미지에 source/license, `physical_item_id`, `capture_session`, `video_run_id`, `lot_id`를 기록합니다.
3. 대표 200장에는 보이는 모든 target instance를 수동 라벨링합니다.
4. Pseudo-label은 train 후보에만 사용하고 locked gold val/test에는 사용하지 않습니다.
5. CVAT export는 exact image/class map 및 reviewer 승인 후 canonical dataset으로 승격합니다.
6. Split은 같은 실물·연속 촬영·near-duplicate가 경계를 넘지 않도록 group 단위로 고정합니다.

## Track C — 단계형 학습

실행 계약은 [`smallchip_staged_training_v1.yaml`](../configs/experiments/smallchip_staged_training_v1.yaml)입니다.

| 단계 | 학습 한도 | 필수 확인 | 다음 단계 승격 조건 |
|---|---:|---|---|
| Smoke | 1 epoch | contract·class 수·finite loss·checkpoint/log | NaN/OOM 없음, evidence 완결 |
| Pilot | 최대 10 epochs, `patience=3` | validation trend·오류 사례 | 사전 정의 metric과 error analysis 통과 |
| Candidate | 최대 50 epochs, `patience=10` | per-class metric·운영점·ONNX val/test | overfit·leakage·runtime gate 통과 |

50 epochs 이후 연장은 자동 실행하지 않습니다. Validation curve 미수렴, overfitting 점검, compute/time
budget, leakage gate 및 사용자 승인을 문서화한 새 versioned contract가 있어야 합니다.

## 구현 Backlog

### P0 — 실제 multi-class 학습 전

- 실제 STM32/SMD source license와 specimen evidence 확보
- Trusted registry entry와 canonical manifest 생성
- 200장 full-instance gold annotation 및 실제 CVAT round-trip
- Class별 수량·bbox pixel-size·촬영 조건 imbalance 보고
- 두 framework에서 동일 canonical dataset 1-epoch smoke

### P1 — 정확도·효율

- `640 full-frame`과 tiled inference 비교
- 원본 및 resize 후 bbox width/height/area bin별 recall/AP
- Class별 confidence calibration과 error taxonomy
- 사람 수정시간, proposal recall, 빈 예측·저신뢰 queue 수치화
- 필요성이 입증된 경우에만 SAHI 또는 별도 proposal adapter 평가

### P2 — Ubuntu 전달

- Git LFS checkpoint/ONNX SHA-256 검증
- 목표 장치 dependency lock과 camera adapter
- 새 camera session에서 capture+preprocess+inference+NMS p50/p95 latency·FPS·RAM/VRAM 측정
- 필요 시 목표 장치에서만 TensorRT engine 생성·검증

## 완료 정의

| REQ ID | 완료 기준 | 현재 |
|---|---|---|
| REQ-RPI-01 | paired2 formal policy와 exact 4-run evidence | **PASS (descriptive-only)** |
| REQ-ONNX-01 | 두 모델 val/test native equivalence와 artifact gate | **PASS** |
| REQ-DATA-01 | provenance·license·specimen/session 기반 승인 dataset | **NOT VERIFIED** |
| REQ-LABEL-01 | gold 및 reviewed train label의 추적 가능한 승격 | **CODE PASS / DATA WAITING** |
| REQ-MC-01 | 두 framework의 동일 multi-class contract smoke | **CODE PASS / DATA WAITING** |
| REQ-UB-01 | 새 camera session의 accuracy·E2E 성능 | **NOT VERIFIED** |

실제 촬영 전에 구분 대상(board/package/OCR), 보유 specimen·lot, 카메라/조명/속도, Ubuntu 목표 장치와
허용 miss/false-positive 기준을 확정해야 합니다.
