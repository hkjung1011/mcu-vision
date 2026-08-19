# 핵심 수치 선정 이유와 검증 상태

- protocol: `micropcb_rpi_phash_component_bootstrap_v2`
- 현재 task: `one_class_raspberry_pi_sbc_detection`
- config SHA256: `02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5`

> 현재 protocol은 Raspberry Pi SBC 1-class bootstrap 전용이며 MCU/SMD multi-class 결과가 아닙니다.
> 아래 값은 재현 가능한 1차 baseline이며 최적값이 아닙니다. 현재 결과는 validation 범위이고, 독립적인 실제 카메라 test는 아직 없습니다.

| ID·항목 | 값 | 왜 선택했는가 | 근거 유형 | 최적값 여부 | 다음 조정 조건 | 현재 검증 상태 |
|---|---|---|---|---|---|---|
| R01 비교 범위 | framework-native recipe system benchmark | 데이터 split, 입력 크기, 공통 evaluator와 동일 GPU를 고정하되 optimizer와 loss는 각 framework의 native recipe를 유지한다. 따라서 실사용 후보 비교이며 architecture 자체의 통제 실험으로 해석하지 않는다. | ENGINEERING_BASELINE | 해당 없음 — architecture ablation이 아님 | architecture 우열을 주장하려면 유사 규모 모델과 동일 optimizer ablation을 별도로 수행한다. | Protocol/config SHA 고정, 3-seed full comparison 미실행 |
| R02 Transfer learning | COCO pretrained, no user-requested backbone freeze | 현재 train 1,500장의 소규모 custom dataset에서 수렴 안정성과 시간을 확보하되, COCO와 전자부품 domain 차이에 대응하도록 framework가 trainable로 정의한 backbone/head weight를 학습한다. Ultralytics의 always-frozen DFL projection처럼 고정된 parameter는 manifest에 기록한다. | ENGINEERING_BASELINE | 아니오 — 소규모 데이터용 1차 baseline | overfitting이 확인되면 frozen-backbone ablation을 추가한다. | Weight transfer와 1-epoch smoke 확인, 100 epochs 미검증 |
| R03 입력 크기 | 640 x 640 fixed | 두 모델의 입력 조건을 같게 하고 RTX 5060 Laptop 8 GB에서 batch 8을 유지하기 위한 1차 기준이다. YOLOX/YOLO11 공식 표에도 640 결과가 있으나 소형 칩 최적 해상도라는 뜻은 아니다. | ENGINEERING_BASELINE | 아니오 — 소형 칩 최적값 미확인 | resize 후 객체 크기 분포와 AP_small을 본 뒤 960/1280 또는 tiling을 비교한다. | 640 CUDA smoke PASS, 실제 SMD의 pixel-size/AP_small 미검증 |
| R04 Micro-batch | micro-batch 8 | 두 모델이 현재 8 GB GPU에서 공통으로 실행 가능한 micro-batch다. YOLO11은 nbs=64로 gradient accumulation을 사용하고 YOLOX는 batch 8마다 step하므로 effective batch는 같지 않다. | HARDWARE_DERIVED | 아니오 — 현재 8 GB GPU의 공통 실행값 | resolved accumulation과 optimizer group을 매 run 기록하며 순수 optimizer 비교로 해석하지 않는다. | 두 framework 1-epoch smoke PASS, 장시간 열·VRAM 미검증 |
| R05 Epochs | 100 | pretrained custom-data 1차 비교를 위한 동일 epoch/data-exposure budget이다. 두 모델의 FLOPs와 optimizer step 수가 달라 동일 compute budget은 아니다. YOLOX COCO scratch 논문의 300 epochs를 재현하는 값이 아니며 최적 epoch라는 보장도 없다. | TO_TUNE | 아니오 — TO_TUNE | validation AP plateau, overfitting gap과 loss curve를 보고 동일 규칙으로 연장한다. | 100-epoch full run 미실행 |
| R06 반복 seed | 42, 43, 44 | 단일 운 좋은 run을 피하고 mean ± sample SD를 제시하기 위한 최소 반복이다. | ENGINEERING_BASELINE | 아니오 — 최소 반복 수 | 모델 차이가 run 간 표준편차와 비슷하면 5회 이상으로 늘린다. | seed 42/43/44 full comparison 미실행 |
| R07 Mixed precision | AMP enabled | 메모리와 시간을 줄여 동일 GPU에서 학습하며 NaN/overflow와 실제 precision mode를 기록한다. | UPSTREAM_SUPPORTED | 해당 없음 — 실행 방식 | 수치 불안정 또는 NaN 발생 시 동일 조건의 FP32 검증 run을 수행한다. | 두 framework CUDA smoke PASS, FP32 control 미실행 |
| R08 YOLOX optimizer | SGD Nesterov, nominal_lr=0.01*batch/64, momentum=0.9, wd=0.0005 | YOLOX 원 논문의 SGD/momentum/weight decay와 linear LR scaling, 고정 source의 Nesterov 구현을 따른다. batch 8의 nominal/base LR은 0.00125이며 실제 warmup은 0에서 시작한다. | PAPER_AND_PINNED_SOURCE_DERIVED_WITH_PROJECT_EPOCH_CHANGE | 아니오 — 전자부품 domain 최적값 미확인 | 본 baseline 후 optimizer/LR 변경은 별도 protocol ID로 분리한다. | Resolved config/manifest 확인, 100 epochs 미검증 |
| R09 YOLOX augmentation | MixUp off, Mosaic scale 0.5..1.5; project no-aug final 10/100 epochs | YOLOX 논문은 S/Tiny/Nano에서 MixUp 제거와 Mosaic scale 0.5..1.5를 보고했다. 논문의 300 epochs 중 final no-aug 15 epochs와 달리 현재 10 epochs는 100-epoch transfer baseline의 프로젝트 변경값이다. | PAPER_DERIVED_AUGMENTATION_WITH_PROJECT_NO_AUG_ADAPTATION | 아니오 — 논문 근거와 프로젝트 변경의 결합 baseline | 고정 카메라 SMD 데이터가 준비되면 weak/strong augmentation ablation을 수행한다. | 설정/source 일치 확인, full-run 증강 효과 미검증 |
| R10 YOLO11 recipe | pinned Ultralytics v8.4.120 defaults plus project optimizer=SGD override | upstream optimizer 기본값은 auto다. 이 프로젝트는 버전별 자동 선택을 피하도록 SGD로 override하고, 나머지 관련 기본값은 v8.4.120 source에 고정해 실제 적용값을 기록한다. | UPSTREAM_DEFAULT_WITH_PROJECT_OVERRIDE | 아니오 — pinned default와 project override | 모든 변경은 resolved config와 새 protocol ID로 기록한다. | Resolved config와 1-epoch smoke PASS, 100 epochs 미검증 |
| R11 주 평가 기준 | COCO AP50-95 | IoU 0.50..0.95를 0.05 간격으로 평균해 단일 느슨한 IoU 기준보다 위치 정확도를 엄격히 본다. | STANDARD_METRIC | 해당 없음 — 표준 평가 metric | AP50, AP75, AP_small, AR100, P/R/F1, TP/FP/FN을 함께 보고한다. | 평가 경로/maxDets 100 확인, 정식 모델 결과 없음 |
| R12 Thresholds | floor=0.001, report point=0.25, match IoU=0.50, NMS IoU=0.65, class-aware NMS | 낮은 prediction floor로 PR curve를 보존하고 두 framework에 같은 numeric threshold와 class-aware NMS를 적용한다. 서로 다른 class가 겹쳐도 상호 억제하지 않으며, COCO AP/AR와 운영점 greedy matcher 모두 이미지당 score 상위 100개로 제한한다. | ENGINEERING_BASELINE | 아니오 — 공통 보고점 | 배포 confidence와 pseudo-label threshold는 gold validation에서 목표 P/R로 산출한 뒤 test 전에 동결한다. | Top-100 matcher 구현 확인, 배포 threshold 미보정 |
| R13 데이터 분할 | pHash-connected condition components; 1500/195/180 train/val/test | 동일 rotation/x/y의 5개 반복 capture와 pHash 거리 4 이하 후보를 연결요소 단위로 묶어 split 사이에 같은 조건 또는 현재 검출된 근접중복 후보가 넘어가지 않게 한다. 모델별 image 수는 train/val/test 500/65/60으로 유지한다. | DATA_AUDIT_DERIVED | 현재 공개 원본에서 가능한 보수적 bootstrap 분할이며 독립 specimen test는 아님 | 새 물리 보드와 새 촬영 session을 확보하면 physical_item/session 단위로 다시 분할한다. | condition overlap 0, cross-split pHash 후보 0/3511; physical item ID NOT VERIFIED |
| R14 DataLoader workers | 0 (single-process loading) | 현재 Windows laptop의 총 RAM은 15.12 GiB이고 학습 시작 전 가용 RAM이 약 3 GiB다. PyTorch 문서상 multi-process worker는 parent process의 Python object memory를 추가로 소비할 수 있으므로 장시간 6-run의 paging·worker crash 위험을 줄이기 위해 main process에서 로드한다. 이 값은 optimizer나 정확도 정의가 아니라 데이터 공급 방식이며 두 모델에 동일하게 적용한다. | HARDWARE_DERIVED | 아니오 — 현재 16 GB Windows 장치의 안정성 우선값 | 전용 RAM이 충분한 장비에서는 workers 1/2 처리량 pilot 후 새 campaign ID로 조정한다. | workers 0 smoke PASS, 100-epoch 장시간 안정성 미검증 |

## 해석 원칙

- `PAPER_*`/`UPSTREAM_*`은 출처가 있다는 뜻이지 현재 MCU/SMD에서 최적이라는 뜻이 아닙니다.
- `HARDWARE_DERIVED`는 현재 RTX 5060 Laptop 8 GB에서 실행 가능한 값입니다.
- `ENGINEERING_BASELINE`/`TO_TUNE`은 gold validation과 실제 카메라 조건으로 다시 정해야 합니다.
- 상세 알고리즘·참고문헌은 [전체 방법론](experiment_methodology.md), 이 결과에 실제로 사용한 고정 값은 [immutable protocol snapshot](protocol_snapshot.yaml)을 봅니다.
