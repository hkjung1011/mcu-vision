# Windows 재현 절차

이 문서는 public GitHub를 새 PC에 clone한 뒤 현재 Windows 학습 환경과 비교 pipeline을 재현하는
순서를 한 곳에 모은 것입니다. 기준 환경은 Python 3.11과 NVIDIA RTX 5060 Laptop GPU입니다.

## 1. 사전 조건

- Windows PowerShell
- Python 3.11 (`py -3.11` 사용 가능)
- Git과 Git LFS
- 현재 GPU를 지원하는 NVIDIA driver
- public GitHub 접근이 가능한 network

별도 CUDA Toolkit은 현재 PyTorch 학습에 필요하지 않습니다. PyTorch CUDA 13.0 wheel이 runtime을
포함하며, `nvcc`가 필요한 custom extension이나 목표 TensorRT toolchain을 사용할 때만 추가 검토합니다.

## 2. clone과 LFS 확인

```powershell
git lfs install
git clone https://github.com/hkjung1011/mcu-vision.git
Set-Location .\mcu-vision
git lfs pull
git lfs ls-files
Get-FileHash .\weights\pretrained\yolox_s.pth -Algorithm SHA256
Get-FileHash .\weights\pretrained\yolo11m.pt -Algorithm SHA256
```

YOLOX-S pretrained hash는 [`weights/README.md`](../weights/README.md)의 값과 같아야 합니다.

## 3. 세 환경 생성

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_collection.ps1
.\scripts\setup_yolo11.ps1
.\scripts\setup_yolox.ps1
```

각 setup은 추적된 lock file을 사용합니다. YOLO11은 Ultralytics `8.4.120`, 공통 evaluator는
`pycocotools==2.0.11`, YOLOX source는 commit `6ddff482...`로 고정합니다.

## 4. 코드·GPU 상태 확인

```powershell
.\.venv-collect\Scripts\python.exe -m pytest
.\.venv-collect\Scripts\mcu-show-status.exe --output runs\status\windows_status.json
.\.venv-yolox\Scripts\python.exe .\scripts\smoke_yolox.py
```

이 단계의 YOLOX smoke는 GPU forward wiring 확인일 뿐 모델 성능 결과가 아닙니다. YOLO11 학습
smoke는 dataset 준비가 끝난 다음 단계에서 실행합니다.

## 5. Raspberry Pi bootstrap dataset

```powershell
.\.venv-collect\Scripts\mcu-download-curated.exe `
  --config configs\datasets.curated.yaml `
  --dataset micro_pcb_images

.\.venv-collect\Scripts\mcu-prepare-micropcb.exe `
  --source-root data\raw\curated\micro_pcb_images `
  --output-root data\processed\micropcb_rpi `
  --manifest data\manifests\micropcb_raspberry_pi_sbc.csv `
  --coco-output-root data\processed\micropcb_rpi_coco

.\.venv-collect\Scripts\mcu-split-condition-groups.exe `
  --input-manifest data\manifests\micropcb_raspberry_pi_sbc.csv `
  --output-manifest data\manifests\micropcb_raspberry_pi_sbc.phash_v2.csv `
  --source-root data\raw\curated\micro_pcb_images `
  --yolo-output-root data\processed\micropcb_rpi_phash_v2 `
  --coco-output-root data\processed\micropcb_rpi_phash_v2_coco `
  --duplicates-report data\reports\micropcb_rpi_audit\duplicates.json

.\.venv-collect\Scripts\mcu-verify-dataset-equivalence.exe `
  --yolo-data data\processed\micropcb_rpi_phash_v2\dataset.yaml `
  --coco-train data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_train2017.json `
  --coco-val data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_val2017.json `
  --output-dir data\evidence\micropcb_rpi_phash_v2
```

준비 후 `train/validation/test=1,500/195/180`, cross-split condition/pHash 후보 0, canonical
동등성 PASS를 확인합니다. 이 test도 physical-item 독립 컨베이어 test는 아닙니다.

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\smoke_yolo11.py
```

이 1-epoch smoke도 wiring 확인 전용이며 성능 비교나 weight 승격에 사용하지 않습니다.

## 6. 3-seed full 비교

먼저 `-DryRun`으로 protocol·dataset evidence·COCO/YOLO 경로, 6개 실제 wrapper 명령과 기존 run의
재사용 가능 여부를 확인합니다. 기본 `CampaignId`는 protocol ID, full/smoke, epochs/batch/imgsz/workers/seeds,
protocol SHA-256, dataset-evidence SHA-256, wrapper/config/pretrained hash에서 결정되므로 같은 입력으로
다시 실행해도 같은 경로를 사용합니다.

```powershell
.\scripts\run_compare_seeds.ps1 `
  -ProtocolConfig configs\experiments\baseline_v1.yaml `
  -YoloData data\processed\micropcb_rpi_phash_v2\dataset.yaml `
  -CocoRoot data\processed\micropcb_rpi_phash_v2_coco `
  -DatasetEvidence data\evidence\micropcb_rpi_phash_v2\dataset_evidence.json `
  -Seeds 42,43,44 `
  -Epochs 100 `
  -Batch 8 `
  -ImageSize 640 `
  -Workers 0 `
  -DryRun

.\scripts\run_compare_seeds.ps1 `
  -ProtocolConfig configs\experiments\baseline_v1.yaml `
  -YoloData data\processed\micropcb_rpi_phash_v2\dataset.yaml `
  -CocoRoot data\processed\micropcb_rpi_phash_v2_coco `
  -DatasetEvidence data\evidence\micropcb_rpi_phash_v2\dataset_evidence.json `
  -Seeds 42,43,44 `
  -Epochs 100 `
  -Batch 8 `
  -ImageSize 640 `
  -Workers 0
```

`Workers=0`은 이 16 GB RAM laptop에서 Windows worker subprocess의 추가 메모리를 피하기 위한
장시간 안전 기본값입니다. 출력은
`runs\benchmarks\<CampaignId>\<model>_seed<seed>`와 `runs\comparisons\<CampaignId>`에 고정됩니다.
재실행할 때는 `status=complete`, epoch row 수, 공통 평가·latency·GPU artifact,
dataset-evidence hash와 실제 checkpoint SHA-256까지 모두 확인된 run만 건너뜁니다. 같은 경로에
실패·중단·조건 불일치 run이 있으면 자동 덮어쓰기나 부분 재개를 하지 않고 해당 정확한 경로와 이유를
출력한 뒤 학습 전에 중단합니다. `terminal.log`를 확인하고 그 run 하나만 별도로 보관하거나 제거한 뒤
같은 명령을 다시 실행합니다.

`protocol_compatibility.json`의 `comparable=true`일 때만 모델 비교표를 해석하고,
`release_ready=true`일 때만 trained weight를 승격합니다. 비교 결과는 AP/AR의 공통 COCOeval,
운영점 공통 matcher, batch-1 latency, VRAM과 seed 평균±sample SD를 포함해야 합니다.
서로 다른 Git commit의 run을 섞는 경우 `mcu-compare-runs --provenance-attestation
configs/experiments/mixed_commit_rpi_v2_attestation.json`을 지정해야 하며, exact commit/blob와
model별 framework/pretrained/protocol hash가 맞지 않으면 `release_ready=false`입니다.

## 7. Git 승격 전 확인

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\promote_run.py `
  --run-dir runs\benchmarks\<CampaignId>\<model>_seed<best-seed> `
  --comparison-dir runs\comparisons\<CampaignId> `
  --release-name rpi_phash_v2_<model>

.\.venv-collect\Scripts\python.exe .\scripts\promote_comparison.py `
  --comparison-dir runs\comparisons\<CampaignId> `
  --release-name rpi_phash_v2_3seed_comparison

git status --short
git lfs ls-files
git diff --check
```

YOLO11 promotion은 Ultralytics/PyTorch 검증 때문에 `.venv-yolo11`에서 실행합니다. YOLOX promotion은
`.venv-collect`에서도 실행할 수 있습니다. Promotion copy는 로컬 사용자 경로와 raw process list를
제거하고 original/published SHA-256을 모두 기록합니다. Smoke 또는 protocol mismatch 결과는
기본적으로 차단됩니다.

## 알려진 제한

- 2026-08-18 progress에는 완료 run 3개와 중단 run 1개가 있지만, 정식 3-seed comparison과
  `weights/trained` release는 아직 없습니다.
- SMD canonical dataset과 CVAT round-trip은 아직 없습니다.
- 현재 autolabel CLI는 Ultralytics YOLO11 `.pt`만 지원합니다.
- ONNX export·전체 val/test 동등성·deployment 승격 도구는 구현됐지만 정식 full-run artifact로는 아직 실행되지 않았습니다.
- Ubuntu CPU dependency lock은 준비됐지만 실제 목표 Ubuntu 장치 실행은 NOT VERIFIED입니다.
- Canonical manifest/class/image-list와 YOLO↔COCO box/class 동등성은 RPi v2에서 PASS했습니다.
- full run 실측은 YOLO11m 약 168분/run, YOLOX-S seed42 약 336분/run이지만 laptop 열·전력 상태에
  따라 달라지므로 재개 전 여유 공간과 전원/열 조건을 직접 확인합니다.

```powershell
Get-PSDrive -Name C
Get-ChildItem . -Directory -Force | ForEach-Object {
  [PSCustomObject]@{Name=$_.Name; GiB=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum / 1GB}
}
```
