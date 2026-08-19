# 현재 프로젝트 상태

기준일: **2026-08-19**

## 선정 결정

Raspberry Pi bootstrap은 추가 100-epoch 학습 없이, 이미 완료된 YOLO11m·YOLOX-S의 matched seed
42/43 네 run을 `paired_2seed_descriptive` evidence tier로 동결합니다. STM32/SMD는 승인 데이터를 먼저
확보한 뒤 `1 epoch smoke → 최대 10 epoch pilot → 최대 50 epoch candidate` 순서로 진행합니다.

## 검증 상태

| 항목 | 검증 근거 | 판정 |
|---|---|---|
| Windows GPU stack | RTX 5060 Laptop, PyTorch 2.12.1+cu130, framework CUDA smoke | **PASS** |
| RPi immutable protocol | canonical SHA-256 `02facd21…`, 기존 evidence hash binding | **PASS / FROZEN** |
| RPi dataset split | train/val/test `1500/195/180`, condition·pHash component cross-split 0 | **PASS (internal)** |
| RPi paired formal | 2 models × seeds 42/43 × 100 epochs, exact 4-run matrix | **PASS (descriptive-only)** |
| 통계 범위 | `n=2`, `df=1`; per-run·mean·sample SD·paired delta | **LIMITED** |
| ONNX deployment gate | 두 모델 native/ONNX val·test, artifact hash, split/policy binding | **PASS** |
| Independent test | physical item ID·새 camera session 독립성 미입증 | **NOT VERIFIED** |
| Canonical multi-class | 6-class ID·alias 동결, 한국어 display sidecar exact binding | **CONTRACT PASS** |
| Small-chip ingest | allowlist·rights/hash·specimen·ZIP resource·atomic output gate | **CODE PASS / DATA WAITING** |
| Small-chip source review | Dainius auth gate, IoTKITs archive SHA·COCO·leakage audit, 대체 원천 검토 | **REVIEWED / SOURCE BLOCKED** |
| CVAT·autolabel | exact image/teacher/calibration binding, reviewed-only promotion | **CODE PASS / DATA WAITING** |
| STM32/SMD 실제 학습 data | trusted registry 승인 canonical dataset 0장 | **NOT VERIFIED** |
| Ubuntu 실제 카메라 | ONNX artifact 검증 완료, 현장 E2E 측정 미실행 | **NOT VERIFIED** |

## RPi 결과와 해석 한계

| 모델 | seed | AP50-95 mean | sample SD |
|---|---|---:|---:|
| YOLO11m | 42, 43 | `1.0000000000` | `0.0000000000` |
| YOLOX-S | 42, 43 | `0.9846059844` | `0.0074003961` |

YOLO11m−YOLOX-S paired delta는 `0.0153940156 ± 0.0074003961`입니다. 이는 같은 두 seed에서 관찰한
기술통계일 뿐, 통계적 유의성이나 모집단 성능 우월성을 의미하지 않습니다. 목표 축소가 사후 결정된 점을
policy에 명시했고 seed 44는 결과 선택 편향을 피하기 위해 양 모델에서 일괄 제외했습니다.

ONNX val/test AP50-95는 YOLO11m `1.0000000000 / 1.0000000000`, YOLOX-S
`0.9908195841 / 0.9852172241`이며 두 deployment gate가 PASS했습니다. 여기서 test는 internal pHash
split입니다. 독립 촬영 test, 현장 일반화 또는 production readiness의 근거로 사용하지 않습니다.
CPU p50 val/test는 YOLO11m `215.698/221.226 ms`, YOLOX-S `78.778/78.323 ms`였으나 해당
hardware·runtime에 종속됩니다. Internal test는 threshold selection에 사용하지 않습니다.

## 요구사항 검증

| REQ ID | 검증 조건 | 현재 판정 |
|---|---|---|
| REQ-RPI-01 | exact seed 42/43 pair, 각 100 epochs, immutable base protocol | **PASS** |
| REQ-RPI-02 | 공통 COCO evaluator와 paired descriptive report | **PASS** |
| REQ-RPI-03 | policy/base SHA, mixed-commit attestation, checkpoint hash | **PASS** |
| REQ-ONNX-01 | native↔ONNX val/test numeric equivalence 및 공개 artifact scan | **PASS** |
| REQ-DATA-01 | provenance·license·specimen/session이 확인된 small-chip 실사 | **NOT VERIFIED** |
| REQ-LABEL-01 | 모든 target instance의 사람 검수와 CVAT round-trip | **CODE PASS / DATA WAITING** |
| REQ-MC-01 | 동일 canonical multi-class dataset으로 두 framework 실행 | **CODE PASS / DATA WAITING** |
| REQ-UB-01 | Ubuntu 새 camera session에서 정확도·E2E latency·FPS | **NOT VERIFIED** |

## 다음 실행 순서

1. 검출 범위를 board/package detection과 marking OCR·exact SKU classification으로 분리해 승인합니다.
2. STM32·SMD 실사를 provenance, license, `physical_item_id`, `capture_session`과 함께 반입합니다.
3. 대표 200장에 보이는 모든 target instance를 수동 라벨링하고 locked gold val을 만듭니다.
4. CVAT round-trip과 reviewed-only promotion을 실제 export에서 검증합니다.
5. `smallchip_staged_training_v1`에 따라 1e smoke, 최대 10e pilot, 최대 50e candidate를 순차 실행합니다.
6. validation 결과로 full-frame·tiling·입력 해상도를 비교하고, 필요 시 근거가 포함된 새 contract만 승인합니다.
7. Ubuntu에서 새 camera session을 고정해 accuracy, p50/p95 E2E latency, FPS와 자원 사용량을 측정합니다.

## 주장 경계

- Internal pHash split을 independent test라고 부르지 않습니다.
- ONNX deployment PASS를 Ubuntu 카메라 현장 검증 또는 production-ready로 표현하지 않습니다.
- `n=2`, `df=1` 결과로 통계적 유의성이나 architecture의 모집단 우월성을 주장하지 않습니다.
- RPi bootstrap 결과를 STM32/SMD 성능으로 확장하지 않습니다.
- 외형만으로 exact STM32 part number, 전기적 기능 또는 정격을 확정하지 않습니다.
