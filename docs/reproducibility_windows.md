# Windows 재현 절차

이 문서는 private GitHub를 새 PC에 clone한 뒤 현재 Windows 학습 환경과 비교 pipeline을 재현하는
순서를 한 곳에 모은 것입니다. 기준 환경은 Python 3.11과 NVIDIA RTX 5060 Laptop GPU입니다.

## 1. 사전 조건

- Windows PowerShell
- Python 3.11 (`py -3.11` 사용 가능)
- Git과 Git LFS
- 현재 GPU를 지원하는 NVIDIA driver
- private GitHub 저장소 접근 권한

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
.\.venv-yolo11\Scripts\python.exe .\scripts\smoke_yolo11.py
.\.venv-yolox\Scripts\python.exe .\scripts\smoke_yolox.py
```

Smoke는 wiring 확인일 뿐 모델 성능 결과가 아닙니다.

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
```

준비 후 `train=1,500`, `validation=375`, class map, annotation SHA-256을 확인합니다. Validation은
독립 컨베이어 test가 아닙니다.

## 6. 3-seed full 비교

```powershell
.\scripts\run_compare_seeds.ps1 `
  -Seeds 42,43,44 `
  -Epochs 100 `
  -Batch 8 `
  -ImageSize 640
```

`protocol_compatibility.json`의 `comparable=true`일 때만 모델 비교표를 해석하고,
`release_ready=true`일 때만 trained weight를 승격합니다. 비교 결과는 AP/AR의 공통 COCOeval,
운영점 공통 matcher, batch-1 latency, VRAM과 seed 평균±sample SD를 포함해야 합니다.

## 7. Git 승격 전 확인

```powershell
.\.venv-collect\Scripts\python.exe .\scripts\promote_run.py `
  --run-dir runs\benchmarks\<completed-run-id> `
  --comparison-dir runs\comparisons\<completed-comparison-id>

.\.venv-collect\Scripts\python.exe .\scripts\promote_comparison.py `
  --comparison-dir runs\comparisons\<completed-comparison-id> `
  --release-name detector_baseline_v1

git status --short
git lfs ls-files
git diff --check
```

Promotion copy는 로컬 사용자 경로와 raw process list를 제거하고 original/published SHA-256을 모두
기록합니다. Smoke 또는 protocol mismatch 결과는 기본적으로 차단됩니다.

## 알려진 제한

- 정식 3-seed 결과와 trained weight는 아직 없습니다.
- SMD canonical dataset과 CVAT round-trip은 아직 없습니다.
- 현재 autolabel CLI는 Ultralytics YOLO11 `.pt`만 지원합니다.
- ONNX export·동등성 검증과 Ubuntu 설치 lock은 아직 구현되지 않았습니다.
- Canonical manifest/class/image-list와 YOLO↔COCO box/class 동등성 hash 생성은 아직 구현되지 않아
  정식 release gate가 의도적으로 BLOCKED입니다.
- full run 시간과 전체 디스크 사용량은 아직 측정되지 않았으므로 실행 전 여유 공간과 전원/열 조건을
  직접 확인합니다.

```powershell
Get-PSDrive -Name C
Get-ChildItem . -Directory -Force | ForEach-Object {
  [PSCustomObject]@{Name=$_.Name; GiB=(Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum).Sum / 1GB}
}
```
