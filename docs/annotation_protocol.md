# MCU/SMD 라벨링 및 오토라벨 검수 규정

## 선정 방식

목표 workflow는 **CVAT Community + 수동 seed 라벨**, 이후 **학습된 domain YOLO teacher + tiled
pseudo-label + 전량 사람 검수**입니다. 현재 구현된 CLI는 **Ultralytics YOLO11 `.pt` 기반 proposal
생성기와 자체 tile/class-aware NMS**까지입니다. CVAT 서버 자동 import/export는 없지만, 받은 COCO/YOLO
export의 round-trip verifier와 reviewer/hash 기반 승격 gate는 구현되었습니다. 실제 CVAT export를 넣은
통합 검증, Grounding DINO, SAM2, 실제 SAHI package와 native YOLOX `.pth` backend는 아직 미완료입니다.

| 단계 | 입력 | 출력 상태 | 학습 사용 |
|---|---|---|---|
| 수동 seed | 사람이 모든 목표 instance를 box/class 지정 | `manual_seed` | train 가능 |
| 범용 초기 제안 | Grounding DINO text prompt | `pending` | NOT IMPLEMENTED |
| domain teacher 제안 | 현재는 Ultralytics YOLO11 checkpoint | `pending` | 검수 전 금지 |
| 사람 승인·수정 | 누락 추가, 오검출/중복 삭제, class 수정 | `reviewed_train` | train 가능 |
| validation | 모델 제안 없이 수동 생성·독립 QA | `gold_validation_locked` | train에 사용 금지 |
| test | 새 실물/session을 수동 생성·잠금 | `test_locked` | train·threshold 선정 금지 |

이미지에서 보이는 목표 부품 중 일부만 box로 표시하면 나머지는 background로 학습됩니다. 따라서
“200개 라벨”이 200 box라는 뜻이면 부족할 수 있습니다. 이 protocol의 시작값은 **대표 이미지
200장에 보이는 모든 목표 instance**이며, 첫 모델의 FP/FN과 새 촬영 조건을 보고 늘립니다.

## 오픈소스 선택 근거

| 도구 | 역할 | 라이선스 | 구현 상태·판단 |
|---|---|---|---|
| [CVAT Community](https://github.com/cvat-ai/cvat) | bbox 검수·QA·YOLO/COCO 입출력 | MIT | 권장 UI, 현재 SDK/ZIP round-trip NOT IMPLEMENTED |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | text-prompt bbox 초기 제안 | Apache-2.0 | REFERENCE ONLY; 세부 전자부품 domain gap 때문에 전량 검수 필요 |
| [SAM2](https://github.com/facebookresearch/sam2) | point/box prompt mask 보정 | Apache-2.0 | REFERENCE ONLY; semantic class detector가 아님 |
| [SAHI](https://github.com/obss/sahi) | small-object sliced inference | MIT | REFERENCE ONLY; 현재 코드는 SAHI가 아닌 자체 tile+NMS |
| Ultralytics YOLO11 teacher | 현장 class pseudo-label | AGPL-3.0 또는 Enterprise | IMPLEMENTED; seed 이후 현장 외관에 적응 |
| native YOLOX teacher | 현장 class pseudo-label | Apache-2.0 | NOT IMPLEMENTED |

Grounding DINO 공식 원본 GPU build는 custom CUDA operator 때문에 `CUDA_HOME/nvcc`가 필요할 수
있습니다. 현재 학습용 PyTorch에는 별도 CUDA Toolkit이 필요 없으며, 초기 라벨링도 우선 수동
seed + 현재 YOLO 환경으로 진행합니다. Grounding DINO를 실제 도입할 때는 별도 환경에서
Hugging Face implementation 또는 WSL2/Linux build를 검증합니다.

## 수치 선정 및 승인 규칙

| 항목 | 시작값 | 상태 | 선정 이유 | 확정 방법 |
|---|---:|---|---|---|
| 수동 seed | 200 images | PROVISIONAL | 조건별 pilot와 teacher 생성용 | learning curve와 FP/FN 포화 확인 |
| proposal floor | 0.10 | TO_TUNE | 검수 후보를 넓게 보존 | gold set에서 누락/검수량 비교 |
| high-confidence 목표 P | point estimate ≥0.98, Wilson 95% lower bound ≥0.95 | ENGINEERING_BASELINE | 소수의 우연한 TP를 100% precision으로 오인하지 않기 위한 보수적 기준 | `gold_validation_locked`의 class별 threshold sweep |
| tile | 640 px | TO_TUNE | 원본 작은 객체의 pixel 보존과 현 모델 입력 일치 | full-frame/640/960 tile 비교 |
| overlap | 0.20 | TO_TUNE | sliced inference의 일반 시작값 | tile 경계 FN과 중복 box 비교 |
| merge NMS IoU | 0.50 | TO_TUNE | 겹친 tile의 동일 class 중복 제거 시작값 | gold set FP/FN 및 밀집 객체 합침 오류 확인 |

`high-confidence`는 자동 승인 기준이 아닙니다. `mcu-evaluate-predictions`가 잠긴 수동 validation에서
목표 precision과 Wilson 95% lower bound를 모두 만족하는 class별 후보 threshold를
`autolabel_thresholds.csv`에 계산하며, 그
값을 사용해도 사람 검수를 생략하지 않습니다. 객체가 하나도 예측되지 않은 이미지는 true negative로
자동 확정하지 않고 누락 우선 검수 대상으로 둡니다.

## 로컬 pseudo-label 실행

```powershell
.\.venv-yolo11\Scripts\mcu-autolabel-yolo.exe `
  --source data\staging\unlabeled_smd `
  --source-manifest data\manifests\unlabeled_smd.session-001.json `
  --model runs\benchmarks\<seed-run>\native\weights\best.pt `
  --teacher-manifest runs\benchmarks\<seed-run>\teacher_manifest.json `
  --ontology configs\classes.smd_v1.yaml `
  --calibration runs\benchmarks\<seed-run>\final_metrics.json `
  --tile-size 640 --tile-overlap 0.20 `
  --run-id smd_pending_v1
```

산출물은 `runs/autolabel/smd_pending_v1/`에 저장됩니다.

- `terminal.log`: 진행 과정과 최종 수치
- `predictions.csv`: 모든 proposal의 score, class, pixel 좌표, 검수 상태
- `review_queue.csv`: 빈 예측과 저신뢰 이미지를 앞에 둔 검수 순서
- `labels_pending/`: YOLO 형식의 미승인 라벨
- `previews/`: box, class, score가 그려진 검수 이미지
- `run_manifest.json`: teacher hash, threshold, tile/NMS 값, 환경

`labels_pending`은 canonical dataset 경로가 아니며 이를 train에 자동 합치지 않습니다. 현재 출력은
CVAT import ZIP이 아닙니다. `mcu-verify-cvat-roundtrip`과 `mcu-promote-reviewed`를 사용해 실제 CVAT
COCO export의 왕복 검증, reviewer·승인 시각·export/ontology hash, 전 이미지 disposition이 모두
PASS한 승격 manifest를 만든 뒤에만 `reviewed_train`으로 다룹니다. 파일을 수동 복사한 것만으로는
승인 데이터가 되지 않습니다.

## 수동 라벨 가이드

- `raspberry_pi_*`/개발보드는 보드의 보이는 외곽을 box로 잡고 케이블·그림자·지그는 제외합니다.
- bare IC/소형 부품은 보이는 package와 부착된 lead까지 포함하고 cast shadow는 제외합니다.
- 가림 객체는 하나의 실물로 식별 가능한 경우 각각 visible extent box를 만들고, 판단 불가는
  `ambiguous` 검수 목록으로 격리합니다.
- 화면 경계에서 잘린 객체는 canonical COCO에 truncation 상태를 기록합니다. YOLO export에서 해당
  속성이 손실될 수 있으므로 manifest와 함께 유지합니다.
- marking을 읽을 수 없어도 generic detector label은 가능하지만 exact SKU/classifier/OCR 정답으로는
  사용하지 않습니다.
- 원본 이미지/CVAT 화면을 기준으로 판정합니다. JPEG `previews/`는 작은 marking 정답 판정에 쓰지 않습니다.
- class ID/order와 ontology hash를 고정하고, 최소 10~20%를 2차 검수해 불일치는 합의 기록을 남깁니다.

## 논문 근거와 한계

- [Grounding DINO](https://arxiv.org/abs/2303.05499)는 image-text pair에서 open-set box를 찾지만,
  package 외형이 비슷한 STM32 part number까지 구분한다는 근거는 없습니다.
- [SAHI](https://arxiv.org/abs/2202.06934)는 sliced inference로 small-object detection 개선을
  보고했지만, 현재 자체 tile+NMS는 SAHI 구현 재현이 아니며 640 tile이 최적이라는 뜻도 아닙니다.
- [Soft Teacher](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html)는
  신뢰도 높은 pseudo box를 이용한 semi-supervised detection의 근거를 제공하지만, 이 프로젝트의
  단순 proposal-review loop가 논문을 그대로 재현하는 것은 아닙니다.

마킹 문자열이나 exact part number가 목표이면 detector box 이후 고해상도 crop과 OCR/classifier를
분리해야 합니다. 개요 사진의 package 외형만으로 전기적 기능이나 exact SKU를 확정하지 않습니다.
