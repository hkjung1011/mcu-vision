# MCU Vision Dataset & Training Workspace

Windows에서 라이선스가 확인되는 공개 이미지와 사용자 촬영본을 수집·검수하고, YOLOX/YOLO11 학습 결과를 재현 가능하게 보존하기 위한 프로젝트입니다.

> **현재 상태:** Windows GPU·로깅·공통 평가·오토라벨 proposal pipeline은 검증했지만, 동일 조건의
> 3-seed full 비교와 독립 컨베이어 test는 아직 수행하지 않았습니다. 현재 smoke 결과로 모델 우열을
> 판단하지 않습니다.

## 문서 지도

- [현재 프로젝트 상태](docs/project_status.md)
- [Windows fresh-clone 재현 절차](docs/reproducibility_windows.md)
- [데이터 수집 및 class별 1,000장 계획](docs/data_plan.md)
- [실험 방법론·논문 및 수치 선정 근거](reports/methodology/experiment_methodology.md)
- [로그 기반 결과·증빙 정책](docs/evidence_and_results_policy.md)
- [라벨링·오토라벨 사람 검수 규정](docs/annotation_protocol.md)
- [Ubuntu 카메라 시험 인계서](docs/ubuntu_handoff.md)
- [전체 문서 안내](docs/README.md)
- [Third-party license 및 재배포 경계](THIRD_PARTY_NOTICES.md)

## 현재 범위

- 이미지 출처와 라이선스를 JSONL manifest로 기록
- 손상 파일, 저해상도 파일, exact/near duplicate 검사
- 서로 겹치는 범주의 임시 분리를 위한 provisional taxonomy
- 학습 checkpoint와 ONNX를 Git LFS로 관리

현재 class 정의는 `configs/classes.provisional.yaml`에 있습니다. 특히 다음은 서로 다른 class로 유지합니다.

- `raspberry_pi_sbc`: Raspberry Pi SBC 전체 보드
- `raspberry_pi_pico`: Pico 계열 MCU 개발보드
- `stm32_dev_board`: STM32 개발보드
- `stm32_bare_ic`: 검증 가능한 bare STM32 IC
- `small_component_generic`: 첨부 사진 기반 detector-only 임시 class

사진 속 소형 부품의 최종 class는 근접 사진 또는 part number가 확인되기 전까지 만들지 않습니다.

## Windows 초기 설정

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_collection.ps1
```

환경 활성화:

```powershell
.\.venv-collect\Scripts\Activate.ps1
```

## Windows YOLO11 GPU 환경

수집 환경과 학습 환경은 별도로 유지합니다. 공식 PyTorch CUDA 13.0 wheel을 설치하며, 현재 RTX 5060
Laptop GPU에서 검증했습니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_yolo11.ps1
.\.venv-yolo11\Scripts\Activate.ps1
```

`setup_yolo11.ps1` 마지막 출력에서 `cuda_available`이 `true`이고 GPU 이름이 표시되어야 합니다.

### CUDA Toolkit 설치 여부

현재 PC에는 `nvcc`가 포함된 별도 CUDA Toolkit이 없지만 추가 설치할 필요가 없습니다. 설치된 PyTorch `2.12.1+cu130` wheel이 CUDA runtime 13.0과 cuDNN을 포함하며, RTX 5060 Laptop GPU에서 YOLO11 학습과 YOLOX forward를 통과했습니다. NVIDIA driver는 설치되어 있습니다.

아래 명령으로 이 값을 터미널에서 언제든 다시 확인하고 JSON으로도 저장할 수 있습니다.

```powershell
.\.venv-collect\Scripts\mcu-show-status.exe `
  --output runs\status\windows_status.json
```

별도 CUDA Toolkit은 custom CUDA extension/plugin을 `nvcc`로 컴파일하거나 Ubuntu 목표 장치의
TensorRT toolchain이 요구할 때 검토합니다.

준비된 Raspberry Pi subset으로 CUDA 1-epoch 최소 smoke test를 실행할 수 있습니다.

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\smoke_yolo11.py
```

YOLOX-S는 오래된 의존성과의 충돌을 피하도록 또 다른 환경에 설치합니다. 공식 YOLOX source를 고정 commit으로 받아 CUDA forward pass까지 확인합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_yolox.ps1
```

Raspberry Pi bootstrap 학습은 로그 wrapper로 실행합니다. `-o/--occupy`는 GPU 메모리를 선점해 VRAM 비교를 왜곡하므로 사용하지 않습니다.

```powershell
.\.venv-yolox\Scripts\python.exe .\scripts\train_yolox_logged.py `
  --run-id yolox_s_seed42 `
  --epochs 100 --batch 8 --imgsz 640 --seed 42
```

YOLO11m은 같은 640 입력과 같은 split으로 실행합니다.

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\train_yolo11_logged.py `
  --run-id yolo11m_seed42 --model yolo11m.pt `
  --epochs 100 --batch 8 --imgsz 640 --seed 42
```

두 모델을 순차 실행하고 마지막에 같은 표와 그래프로 합치는 명령은 다음 하나입니다.

```powershell
.\scripts\run_compare.ps1 -Epochs 100 -Batch 8 -ImageSize 640 -Seed 42
```

배선과 로깅만 빠르게 확인할 때는 아래 smoke 명령을 사용합니다. 1 epoch 결과는 모델 우열 판단에 사용하지 않습니다.

```powershell
.\scripts\run_compare.ps1 -Smoke
```

공통 baseline 수치는 `configs/experiments/baseline_v1.yaml`에 고정했습니다. YOLO11m은 COCO pretrained
checkpoint에 `freeze=0`을 요청해 사용자가 지정한 backbone freeze 없이 trainable weight를 fine-tune합니다.
Ultralytics가 항상 고정하는 DFL projection 등 실제 trainable/frozen 수는 manifest에 기록합니다. YOLOX-S는
shape가 맞는 pretrained tensor를 이식하고 class 수가 달라진 `cls_preds` output conv 6개만 재초기화합니다.
두 framework optimizer recipe가 서로 다르므로 실제 optimizer group, LR, momentum, weight decay도 학습
시작 터미널과 manifest에 기록합니다.

이 YAML은 설명용 사본이 아니라 두 training wrapper의 실제 기본값입니다. 각 수치는
`PAPER_DERIVED`, `UPSTREAM_DEFAULT`, `HARDWARE_DERIVED`, `ENGINEERING_BASELINE`, `TO_TUNE`으로
구분하고, 선정 이유·조정 조건·논문/공식 source URL도 같은 파일에서 관리합니다. 터미널과 PNG,
CSV, Markdown 근거표를 학습 없이 먼저 만들 수 있습니다.

```powershell
.\.venv-collect\Scripts\mcu-show-protocol.exe `
  --output-dir reports\methodology
```

YOLO11에는 별도의 공식 peer-reviewed YOLO11 논문이 없으므로 공식 live YOLO11 문서와 Ultralytics
v8.4.120 고정 source를 구분해 근거로 사용합니다. YOLOX-S는 원 논문의 anchor-free/decoupled head/SimOTA와
small-model augmentation 근거를 사용하되, 이 프로젝트의 100 epochs·transfer learning·fixed 640은
현 데이터/하드웨어용 변경값으로 명시합니다.

### 터미널·수치·그래프 기록

각 단일 학습 실행은 `runs/benchmarks/<run-id>/` 아래에 다음을 자동 생성합니다.

- `terminal.log`: 화면에 표시된 전체 학습 로그의 ANSI 제거본
- `run_manifest.json`: 명령, Python/PyTorch/CUDA/cuDNN/GPU, seed, data hash, 실제 optimizer group
- `epoch_metrics.csv`: 공통 epoch 표. YOLO11 callback 원값은 `epoch_metrics_extra.jsonl`, YOLOX 원값은 `epoch_metrics.jsonl`
- `gpu_samples.csv`, `gpu_summary.json`: `nvidia-smi` 기반 memory/utilization/temperature/power
- `pretrained_weights_summary.csv`, `best_weights_summary.csv`: tensor 이름·shape·dtype·numel·mean/std/min/max/L2
- `predictions.coco.json`, `final_metrics.json`, `per_class_metrics.csv`: 두 모델 공통 evaluator 입력과 결과
- `confidence_curve.csv`, `autolabel_thresholds.csv`: threshold별 TP/FP/FN/P/R/F1과 목표 precision 기반 class별 후보값
- `latency_samples.csv`, `latency.json`: batch 1 GPU/E2E p50·p95, FPS, peak CUDA memory
- `plots/epochs/*.png`: 매 epoch 자동 캡처 그래프
- `protocol_rationale.csv/png`, `experiment_methodology.md`: 수치 선정 이유·조정 조건·출처

둘 이상의 run을 비교하면 `runs/comparisons/<comparison-id>/` 아래에 다음을 생성합니다.

- `comparison_terminal.txt`: CLI에 출력한 것과 동일한 터미널 표 원문
- `comparison.csv/json`, `aggregate_comparison.csv/json`: run별 수치와 seed 평균·sample standard deviation
- `comparison_dashboard.png`, `training_curves.png`, `terminal_summary.png`: 로그/CSV 기반 비교용 PNG
- `evidence_manifest.json`: 숫자 원본과 파생 이미지의 SHA-256, renderer, 생성형 AI 미사용 기록
- `sources/<run-id>/`, `sources_manifest.json`: 사용자 경로를 제거한 비교 입력 로그·수치의 self-contained bundle

여러 seed run을 한 번에 `mcu-compare-runs`에 넘기면 `aggregate_comparison.csv/json`에 모델별 평균과 sample standard deviation도 생성합니다. 두 모델의 seed 집합이 다르면 protocol FAIL로 표시합니다.

판단 근거는 `terminal.log`, CSV, JSON뿐이며 PNG를 수치 원본으로 사용하지 않습니다. 모든 PNG는
ImageGen 같은 생성형 AI가 아니라 Python `matplotlib`가 로그/CSV 숫자를 그린 비생성형 파생물입니다.
`terminal_summary.png`는 실제 화면 캡처가 아니며 `comparison_terminal.txt` 원문을 그대로 코드로
렌더링한 이미지입니다. 원본과 이미지 SHA-256은 `evidence_manifest.json`에서 추적합니다.
YOLO11의 `box/cls/dfl loss`와 YOLOX의 `iou/conf/cls/l1 loss`는 정의가 달라
절대값을 서로 비교하지 않습니다. 최종 비교에서 AP50-95/AP50/AP75/AP_small/AR100은 공통
`pycocotools==2.0.11` COCOeval로 계산합니다. 고정 `confidence=0.25`, `match IoU=0.50`의
P/R/F1/TP/FP/FN은 별도의 공통 score-sorted class-aware greedy 1:1 matcher로 계산합니다. 두 경로 모두
이미지당 score 상위 100개 prediction을 사용합니다. COCO `small`은 annotation area `< 32² px²`입니다.

Epoch train loss는 framework callback의 메모리상 tensor에서 직접 저장합니다. Epoch native
validation P/R/AP는 framework가 제공하는 값이고, 최종 비교 수치는 두 모델의 COCO prediction을
별도 공통 평가 경로로 다시 계산한 JSON float입니다. 보고서에는 반올림해 표시하지만
CSV/JSON에는 계산 결과를 더 높은 정밀도로 보존합니다.

`confidence=0.25`는 고정 보고점일 뿐 최종 배포값이 아닙니다. 수동 gold validation의 confidence
sweep(`0.00..1.00`, step `0.01`; F1 동률이면 더 낮은 첫 threshold)에서 best-F1 또는 목표 Recall에
맞춰 선택하고 test 전에 동결합니다. 오토라벨 후보값은
기본 point precision 0.98과 Wilson 95% lower bound 0.95를 모두 만족하는 class별 threshold를
별도 계산하지만, 소수의 우연한 TP를 과신하지 않기 위한 조건일 뿐 그 box도 사람 검수를
생략하지 않습니다.

YOLO11은 `batch=8`, `nbs=64`에서 gradient accumulation을 사용할 수 있고 YOLOX는 batch 8마다
optimizer step을 수행합니다. 따라서 이 실험은 동일 데이터·입력·공통 평가·동일 GPU에서 각
framework native recipe를 비교하는 실사용 benchmark이며 순수 architecture ablation은 아닙니다.

가중치 tensor별 shape·평균·표준편차·최솟값·최댓값·L2 norm도 터미널에서 직접 볼 수 있습니다.

```powershell
.\.venv-collect\Scripts\mcu-show-weights.exe `
  --csv runs\benchmarks\<run-id>\best_weights_summary.csv `
  --sort l2 --top 30
```

현재 375장은 validation이며 독립적인 실제 컨베이어 test가 아닙니다. 최종 모델 판단은 실제 카메라로 새로 촬영한 고정 test set에서 해야 하고, 안정적인 비교는 seed `42`, `43`, `44` 3회 평균과 표준편차로 진행합니다.

3개 seed를 GPU 한 장에서 순차 실행하고 한 보고서로 합치는 명령입니다.

```powershell
.\scripts\run_compare_seeds.ps1 -Seeds 42,43,44 -Epochs 100 -Batch 8 -ImageSize 640
```

모델 차이가 sample standard deviation과 비슷하면 3회로 우열을 확정하지 않고 5회 이상으로 늘립니다.

## 오토라벨링과 사람 검수

오픈소스로 가능합니다. 목표 workflow는 `CVAT Community + 수동 seed/gold → domain YOLO teacher
pseudo-label → 사람 전량 승인·수정`입니다. 현재 CLI 구현은 **Ultralytics YOLO11 `.pt` proposal과
자체 tile/class-aware NMS**까지이며, CVAT 자동 연동·Grounding DINO·SAM2·실제 SAHI package·native
YOLOX `.pth` 오토라벨 backend는 아직 구현되지 않았습니다.

세부 규정과 출처는 `docs/annotation_protocol.md`, 실행값은
`configs/annotation/autolabel_v1.yaml`에 기록했습니다. 시작값 200은 **box 200개가 아니라 대표
이미지 200장에 보이는 모든 목표 instance를 라벨링**한다는 뜻입니다. 일부 목표만 라벨하면 나머지가
background로 학습되므로 안 됩니다.

수동 seed 모델과 gold validation이 준비된 뒤 다음처럼 pending proposal을 만듭니다.

```powershell
.\.venv-yolo11\Scripts\mcu-autolabel-yolo.exe `
  --source data\staging\unlabeled_smd `
  --model runs\benchmarks\<seed-run>\native\weights\best.pt `
  --calibration runs\benchmarks\<seed-run>\final_metrics.json `
  --tile-size 640 --tile-overlap 0.20 `
  --run-id smd_pending_v1
```

결과의 `labels_pending/`은 canonical train label과 분리됩니다. `review_queue.csv`는 빈 예측·저신뢰
이미지를 우선 배치하고, `previews/`에는 box/class/score가 그려집니다. 자동 편입 명령은 제공하지
않지만 수동 복사를 막는 reviewer/hash 기반 강제 gate도 아직 없으므로 승인 전 train 폴더로 옮기면
안 됩니다. 현재 PyTorch 학습과 YOLO11 오토라벨에는 별도 CUDA Toolkit이 필요 없습니다.
Grounding DINO 공식 GPU source build를 나중에 선택할 경우에는 custom CUDA operator 때문에 별도
Toolkit/WSL2 환경을 검토합니다.

## 라이선스 추적 수집

Wikimedia Commons collector는 `CC0`, Public Domain, `CC BY`, `CC BY-SA` 계열만 후보로 받고 각 파일의 원문 페이지와 저작자·라이선스를 manifest에 남깁니다. 다운로드 완료는 학습 승인과 같지 않습니다. 모든 후보는 사람 검수가 필요합니다.

먼저 실제 후보 수를 확인합니다.

```powershell
mcu-collect-commons `
  --config configs\sources.wikimedia.yaml `
  --class-name raspberry_pi_sbc `
  --limit 1800 `
  --dry-run
```

실제 수집:

```powershell
mcu-collect-commons `
  --config configs\sources.wikimedia.yaml `
  --class-name raspberry_pi_sbc `
  --limit 1800
```

같은 방식으로 `raspberry_pi_pico`, `stm32_dev_board`, `stm32_bare_ic`를 실행합니다. `1,800`은 검수 탈락과 중복을 고려해 고유 승인 이미지 1,000장 이상을 남기기 위한 raw 후보 목표이며, 공개 소스에서 해당 수량을 찾을 수 있다는 보장은 없습니다.

## 무결성 및 중복 감사

```powershell
mcu-audit `
  --data-root data\raw `
  --classes-config configs\classes.provisional.yaml `
  --report-root data\reports\audit
```

단일 class로 이미 분리된 폴더에는 `--default-class raspberry_pi_sbc`처럼 명시할 수 있습니다.

이 명령은 파일을 삭제하지 않습니다. 결과는 다음에 기록됩니다.

- `data/reports/audit/image-audit.csv`
- `data/reports/audit/duplicates.json`
- `data/reports/audit/audit-summary.json`

`pHash` 근접 후보는 자동 삭제하지 않고 사람이 확인합니다. 비슷하게 생긴 서로 다른 전자부품을 오삭제할 수 있기 때문입니다.

## 검토된 공개 데이터셋

Raspberry Pi SBC용 `micro-PCB Images` 데이터셋은 다음 명령으로 받습니다.

```powershell
mcu-download-curated `
  --config configs\datasets.curated.yaml `
  --dataset micro_pcb_images
```

이 데이터셋은 Raspberry Pi 3개 모델 합계 1,875장을 포함하지만 실제 보드 개체 수가 적습니다. 따라서 동일 보드의 각도·위치 변형을 서로 다른 train/test에 섞지 않습니다.

다운로드 후 Raspberry Pi 보드만 골라 YOLO 1-class 형식으로 준비합니다. 이미지는 NTFS hardlink로 연결하므로 1,875장을 다시 복사하지 않습니다.

```powershell
mcu-prepare-micropcb `
  --source-root data\raw\curated\micro_pcb_images `
  --output-root data\processed\micropcb_rpi `
  --manifest data\manifests\micropcb_raspberry_pi_sbc.csv `
  --coco-output-root data\processed\micropcb_rpi_coco
```

원 데이터의 train 실물과 test 실물은 서로 다른 specimen으로 취급하며, source test는 `val`로만 사용합니다. 최종 `test`는 실제 컨베이어에서 새로 촬영해야 합니다.

사진과 유사한 conveyor SMD 데이터는 `smd_components_raw` 항목에 출처를 기록했습니다. Roboflow 인증이 필요할 수 있으므로 raw-images v2를 내려받아야 하며, 증강된 v3-v6은 독립 실사 수량으로 계산하지 않습니다.

## 데이터 원칙

- 증강 이미지는 실제 고유 이미지 1,000장에 포함하지 않습니다.
- 동일 실물의 연속 프레임은 독립 표본으로 세지 않습니다.
- 공개 제품 사진은 보조 데이터로 사용하고, 최종 test set은 실제 컨베이어 카메라 촬영본으로 구성합니다.
- train/val/test는 실물·촬영 session·동영상 단위로 분리합니다.
- Google/Bing 검색 결과나 쇼핑몰 이미지는 재사용 권한이 불명확하므로 자동 수집하지 않습니다.

## Git 저장 정책

- 코드, config, manifest, 평가 결과: 일반 Git
- `*.pt`, `*.pth`, `*.onnx`: Git LFS
- 원본 이미지: 로컬/NAS/object storage, Git에서 제외
- TensorRT `.engine`: Ubuntu 목표 장비에서 생성, Git에서 제외

완료된 full run만 Git/LFS 추적 폴더로 복사합니다. smoke run은 기본적으로 차단되고, raw image와 대용량 prediction JSON은 복사하지 않습니다.

```powershell
.\.venv-collect\Scripts\python.exe .\scripts\promote_run.py `
  --run-dir runs\benchmarks\<완료된-run-id>

.\.venv-collect\Scripts\python.exe .\scripts\promote_comparison.py `
  --comparison-dir runs\comparisons\<완료된-3-seed-comparison> `
  --release-name detector_baseline_v1

git add weights\trained reports\runs reports\comparisons
git commit -m "Add trained detector artifact"
git push
```

`promote_comparison.py`는 protocol 불일치 비교를 기본 차단하며, 표·CSV·JSON·그래프와 각 파일의
SHA-256을 Git 추적 폴더로 복사합니다. 실제 push는 사용자가 private remote와 포함 파일을 검토한
뒤 수행합니다.

private GitHub remote 연결과 push는 저장소 권한을 확인한 뒤 진행합니다.

## 라이선스 주의

- YOLOX source는 Apache-2.0입니다.
- Ultralytics 패키지는 AGPL-3.0 또는 별도 Enterprise license 조건이 적용됩니다.
- private GitHub 저장소라고 해서 dataset·모델·소프트웨어의 재배포 조건이 사라지지는 않습니다.
