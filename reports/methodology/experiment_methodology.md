# 실험 방법 및 수치 선정 근거

- protocol: `micropcb_detector_baseline_v1`
- 상태: `provisional_validation_protocol`
- 비교 유형: `framework_native_recipe_system_benchmark`
- 현재 task: `one_class_raspberry_pi_sbc_detection`
- 원본 config SHA256: `500ecb6186e9da371669eefe785b6201bed8d0de8eee6fb208c453b8f9468141`

> 현재 protocol은 Raspberry Pi SBC 1-class bootstrap 전용입니다. Provisional MCU/SMD multi-class 학습 protocol이 아닙니다.
> 이 문서는 설정 파일에서 자동 생성되었습니다. Validation 결과는 독립적인 실제 컨베이어 test 결과가 아니며, 이 benchmark는 순수 architecture ablation이 아닙니다.
> 판단 근거는 YAML/CSV/JSON의 수치이며, PNG는 matplotlib로 렌더링한 비생성형 파생물입니다. ImageGen 등 생성형 AI는 사용하지 않습니다.

## 고정 protocol

| 구역 | 항목 | 값 |
|---|---|---|
| common | train_images | `1500` |
| common | validation_images | `375` |
| common | independent_test_available | `false` |
| common | image_size | `640` |
| common | batch_size | `8` |
| common | epochs | `100` |
| common | seeds | `[42, 43, 44]` |
| common | device | `0` |
| common | amp | `true` |
| common | fixed_training_scale | `true` |
| common | pretrained | `true` |
| common | full_fine_tune | `true` |
| common | validation_each_epoch | `true` |
| common | prediction_floor | `0.001` |
| common | nms_iou | `0.65` |
| common | class_agnostic_nms | `false` |
| common | max_detections_for_coco_ap | `100` |
| common | operating_confidence | `0.25` |
| common | operating_match_iou | `0.5` |
| yolo11m | optimizer | `SGD` |
| yolo11m | nominal_batch_size | `64` |
| yolo11m | lr0 | `0.01` |
| yolo11m | final_lr_ratio | `0.01` |
| yolo11m | momentum | `0.937` |
| yolo11m | weight_decay | `0.0005` |
| yolo11m | warmup_epochs | `3.0` |
| yolo11m | close_mosaic_epochs | `10` |
| yolo11m | freeze_layers | `0` |
| yolo11m | cos_lr | `false` |
| yolo11m | augmentations | `{"copy_paste": 0.0, "cutmix": 0.0, "degrees": 0.0, "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "mixup": 0.0, "mosaic": 1.0, "perspective": 0.0, "scale": 0.5, "shear": 0.0, "translate": 0.1}` |
| yolox_s | optimizer | `SGD_Nesterov` |
| yolox_s | base_lr_per_image | `0.00015625` |
| yolox_s | base_lr_at_batch_8 | `0.00125` |
| yolox_s | min_lr_ratio | `0.05` |
| yolox_s | minimum_lr_at_batch_8 | `6.25e-05` |
| yolox_s | momentum | `0.9` |
| yolox_s | weight_decay | `0.0005` |
| yolox_s | warmup_epochs | `5` |
| yolox_s | no_augmentation_epochs | `10` |
| yolox_s | multiscale_range | `0` |
| yolox_s | augmentations | `{"degrees": 10.0, "enable_mixup": false, "flip_probability": 0.5, "hsv_probability": 1.0, "mosaic_scale": [0.5, 1.5], "shear": 2.0, "translate": 0.1}` |
| comparison_rules | required_models | `["yolo11m", "YOLOX-S"]` |
| comparison_rules | release_requires_complete_seed_matrix | `true` |
| comparison_rules | required_dataset_evidence | `["canonical_dataset_manifest_sha256", "class_map_sha256", "train_image_list_sha256", "val_image_list_sha256", "canonical_train_records_sha256", "canonical_val_records_sha256"]` |
| comparison_rules | coco_ap_ar_source | `common_pycocotools_cocoeval_2_0_11` |
| comparison_rules | operating_metrics_source | `common_score_sorted_class_aware_greedy_one_to_one_matcher` |
| comparison_rules | primary_metric | `AP50_95` |
| comparison_rules | report_metrics | `["AP50_95", "AP50", "AP75", "AP_small", "AR100", "precision", "recall", "F1", "TP", "FP", "FN"]` |
| comparison_rules | latency | `batch_1_fp16_preloaded_frame_preprocess_inference_nms` |
| comparison_rules | loss_note | `YOLO11_and_YOLOX_native_losses_are_not_cross_framework_comparable` |
| comparison_rules | final_report | `three_seed_mean_and_standard_deviation` |
| comparison_rules | validation_note | `do_not_call_validation_accuracy_a_final_test_result` |

## 선정 근거와 조정 조건

| ID | 항목 | 선택값 | 근거 상태 | 선정 이유 | 최적값 여부 | 재조정 조건 | 현재 검증 상태 | 근거 ID |
|---|---|---|---|---|---|---|---|---|
| R01 | experiment_scope | framework-native recipe system benchmark | ENGINEERING_BASELINE | 데이터 split, 입력 크기, 공통 evaluator와 동일 GPU를 고정하되 optimizer와 loss는 각 framework의 native recipe를 유지한다. 따라서 실사용 후보 비교이며 architecture 자체의 통제 실험으로 해석하지 않는다. | 해당 없음 — architecture ablation이 아님 | architecture 우열을 주장하려면 유사 규모 모델과 동일 optimizer ablation을 별도로 수행한다. | Protocol/config SHA 고정, 3-seed full comparison 미실행 | YOLOX_PAPER, YOLO11_MODEL, ULTRALYTICS_SOURCE |
| R02 | transfer_learning | COCO pretrained, no user-requested backbone freeze | ENGINEERING_BASELINE | 현재 train 1,500장의 소규모 custom dataset에서 수렴 안정성과 시간을 확보하되, COCO와 전자부품 domain 차이에 대응하도록 framework가 trainable로 정의한 backbone/head weight를 학습한다. Ultralytics의 always-frozen DFL projection처럼 고정된 parameter는 manifest에 기록한다. | 아니오 — 소규모 데이터용 1차 baseline | overfitting이 확인되면 frozen-backbone ablation을 추가한다. | Weight transfer와 1-epoch smoke 확인, 100 epochs 미검증 | YOLOX_REPO, ULTRALYTICS_TRAIN |
| R03 | image_size | 640 x 640 fixed | ENGINEERING_BASELINE | 두 모델의 입력 조건을 같게 하고 RTX 5060 Laptop 8 GB에서 batch 8을 유지하기 위한 1차 기준이다. YOLOX/YOLO11 공식 표에도 640 결과가 있으나 소형 칩 최적 해상도라는 뜻은 아니다. | 아니오 — 소형 칩 최적값 미확인 | resize 후 객체 크기 분포와 AP_small을 본 뒤 960/1280 또는 tiling을 비교한다. | 640 CUDA smoke PASS, 실제 SMD의 pixel-size/AP_small 미검증 | YOLOX_PAPER, YOLO11_MODEL, SAHI_PAPER |
| R04 | batch_size | micro-batch 8 | HARDWARE_DERIVED | 두 모델이 현재 8 GB GPU에서 공통으로 실행 가능한 micro-batch다. YOLO11은 nbs=64로 gradient accumulation을 사용하고 YOLOX는 batch 8마다 step하므로 effective batch는 같지 않다. | 아니오 — 현재 8 GB GPU의 공통 실행값 | resolved accumulation과 optimizer group을 매 run 기록하며 순수 optimizer 비교로 해석하지 않는다. | 두 framework 1-epoch smoke PASS, 장시간 열·VRAM 미검증 | ULTRALYTICS_SOURCE, YOLOX_PAPER |
| R05 | epochs | 100 | TO_TUNE | pretrained custom-data 1차 비교를 위한 동일 epoch/data-exposure budget이다. 두 모델의 FLOPs와 optimizer step 수가 달라 동일 compute budget은 아니다. YOLOX COCO scratch 논문의 300 epochs를 재현하는 값이 아니며 최적 epoch라는 보장도 없다. | 아니오 — TO_TUNE | validation AP plateau, overfitting gap과 loss curve를 보고 동일 규칙으로 연장한다. | 100-epoch full run 미실행 | ULTRALYTICS_TRAIN, YOLOX_PAPER |
| R06 | seeds | 42, 43, 44 | ENGINEERING_BASELINE | 단일 운 좋은 run을 피하고 mean ± sample SD를 제시하기 위한 최소 반복이다. | 아니오 — 최소 반복 수 | 모델 차이가 run 간 표준편차와 비슷하면 5회 이상으로 늘린다. | seed 42/43/44 full comparison 미실행 | PYTORCH_REPRO |
| R07 | mixed_precision | AMP enabled | UPSTREAM_SUPPORTED | 메모리와 시간을 줄여 동일 GPU에서 학습하며 NaN/overflow와 실제 precision mode를 기록한다. | 해당 없음 — 실행 방식 | 수치 불안정 또는 NaN 발생 시 동일 조건의 FP32 검증 run을 수행한다. | 두 framework CUDA smoke PASS, FP32 control 미실행 | MIXED_PRECISION_PAPER, PYTORCH_AMP |
| R08 | yolox_optimizer | SGD Nesterov, nominal_lr=0.01*batch/64, momentum=0.9, wd=0.0005 | PAPER_AND_PINNED_SOURCE_DERIVED_WITH_PROJECT_EPOCH_CHANGE | YOLOX 원 논문의 SGD/momentum/weight decay와 linear LR scaling, 고정 source의 Nesterov 구현을 따른다. batch 8의 nominal/base LR은 0.00125이며 실제 warmup은 0에서 시작한다. | 아니오 — 전자부품 domain 최적값 미확인 | 본 baseline 후 optimizer/LR 변경은 별도 protocol ID로 분리한다. | Resolved config/manifest 확인, 100 epochs 미검증 | YOLOX_PAPER, YOLOX_SOURCE |
| R09 | yolox_small_model_augmentation | MixUp off, Mosaic scale 0.5..1.5; project no-aug final 10/100 epochs | PAPER_DERIVED_AUGMENTATION_WITH_PROJECT_NO_AUG_ADAPTATION | YOLOX 논문은 S/Tiny/Nano에서 MixUp 제거와 Mosaic scale 0.5..1.5를 보고했다. 논문의 300 epochs 중 final no-aug 15 epochs와 달리 현재 10 epochs는 100-epoch transfer baseline의 프로젝트 변경값이다. | 아니오 — 논문 근거와 프로젝트 변경의 결합 baseline | 고정 카메라 SMD 데이터가 준비되면 weak/strong augmentation ablation을 수행한다. | 설정/source 일치 확인, full-run 증강 효과 미검증 | YOLOX_PAPER |
| R10 | yolo11_optimizer_and_augmentation | pinned Ultralytics v8.4.120 defaults plus project optimizer=SGD override | UPSTREAM_DEFAULT_WITH_PROJECT_OVERRIDE | upstream optimizer 기본값은 auto다. 이 프로젝트는 버전별 자동 선택을 피하도록 SGD로 override하고, 나머지 관련 기본값은 v8.4.120 source에 고정해 실제 적용값을 기록한다. | 아니오 — pinned default와 project override | 모든 변경은 resolved config와 새 protocol ID로 기록한다. | Resolved config와 1-epoch smoke PASS, 100 epochs 미검증 | ULTRALYTICS_DEFAULTS, ULTRALYTICS_AUGMENT |
| R11 | primary_accuracy_metric | COCO AP50-95 | STANDARD_METRIC | IoU 0.50..0.95를 0.05 간격으로 평균해 단일 느슨한 IoU 기준보다 위치 정확도를 엄격히 본다. | 해당 없음 — 표준 평가 metric | AP50, AP75, AP_small, AR100, P/R/F1, TP/FP/FN을 함께 보고한다. | 평가 경로/maxDets 100 확인, 정식 모델 결과 없음 | COCO_PAPER, COCO_API |
| R12 | prediction_and_operating_thresholds | floor=0.001, report point=0.25, match IoU=0.50, NMS IoU=0.65, class-aware NMS | ENGINEERING_BASELINE | 낮은 prediction floor로 PR curve를 보존하고 두 framework에 같은 numeric threshold와 class-aware NMS를 적용한다. 서로 다른 class가 겹쳐도 상호 억제하지 않으며, COCO AP/AR와 운영점 greedy matcher 모두 이미지당 score 상위 100개로 제한한다. | 아니오 — 공통 보고점 | 배포 confidence와 pseudo-label threshold는 gold validation에서 목표 P/R로 산출한 뒤 test 전에 동결한다. | Top-100 matcher 구현 확인, 배포 threshold 미보정 | COCO_API, YOLOX_SOURCE, ULTRALYTICS_PREDICT |

## 학습 알고리즘과 해석 범위

- **YOLOX-S**: anchor-free one-stage detector, decoupled classification/regression head, SimOTA dynamic label assignment, BCE classification/objectness와 IoU regression을 사용합니다.
- **YOLO11m**: 고정된 Ultralytics 구현의 anchor-free detector이며 box, class, DFL loss로 학습합니다. YOLO11 자체의 별도 peer-reviewed 논문은 없으므로 공식 문서와 고정 버전 source를 근거로 사용합니다.
- 두 framework의 native loss 정의와 optimizer dynamics가 다르므로 raw loss 절대값은 서로 비교하지 않습니다. AP/AR는 동일 COCOeval, 운영점 P/R/F1은 동일 greedy matcher로 계산하고 실제 장치 latency를 함께 비교합니다.
- `batch=8`은 같은 micro-batch/VRAM 조건입니다. YOLO11의 gradient accumulation 때문에 effective optimizer batch까지 같다는 뜻은 아닙니다.

## 참고문헌·공식 구현

- **YOLOX_PAPER** — [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430) (paper)
- **YOLOX_REPO** — [Megvii YOLOX custom data and pretrained checkpoints](https://github.com/Megvii-BaseDetection/YOLOX) (official_repository)
- **YOLOX_SOURCE** — [YOLOX base experiment at commit 6ddff482](https://github.com/Megvii-BaseDetection/YOLOX/blob/6ddff4824372906469a7fae2dc3206c7aa4bbaee/yolox/exp/yolox_base.py) (pinned_source)
- **YOLO11_MODEL** — [Ultralytics YOLO11 model documentation (no separate research paper)](https://docs.ultralytics.com/models/yolo11/) (official_documentation)
- **ULTRALYTICS_TRAIN** — [Ultralytics train mode settings](https://docs.ultralytics.com/modes/train/) (official_documentation)
- **ULTRALYTICS_PREDICT** — [Ultralytics predict mode settings](https://docs.ultralytics.com/modes/predict/) (official_documentation)
- **ULTRALYTICS_DEFAULTS** — [Ultralytics v8.4.120 default training configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.120/ultralytics/cfg/default.yaml) (pinned_source)
- **ULTRALYTICS_AUGMENT** — [Ultralytics data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) (official_documentation)
- **ULTRALYTICS_SOURCE** — [Ultralytics v8.4.120 trainer implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.120/ultralytics/engine/trainer.py) (pinned_source)
- **COCO_PAPER** — [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) (paper)
- **COCO_API** — [pycocotools 2.0.11 package used by the common evaluator](https://pypi.org/project/pycocotools/2.0.11/) (official_source)
- **MIXED_PRECISION_PAPER** — [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (paper)
- **PYTORCH_AMP** — [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html) (official_documentation)
- **PYTORCH_REPRO** — [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) (official_documentation)
- **SAHI_PAPER** — [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934) (paper)
