# MCU Vision Dataset & Training Workspace

Windows에서 라이선스가 확인되는 공개 이미지와 사용자 촬영본을 수집·검수하고, YOLOX/YOLO11 학습 결과를 재현 가능하게 보존하기 위한 프로젝트입니다.

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

수집 환경과 학습 환경은 별도로 유지합니다. RTX 50-series용 공식 CUDA 13.0 PyTorch wheel을 설치합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_yolo11.ps1
.\.venv-yolo11\Scripts\Activate.ps1
```

`setup_yolo11.ps1` 마지막 출력에서 `cuda_available`이 `true`이고 GPU 이름이 표시되어야 합니다.

준비된 Raspberry Pi subset으로 CUDA 1-epoch 최소 smoke test를 실행할 수 있습니다.

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\smoke_yolo11.py
```

YOLOX-S는 오래된 의존성과의 충돌을 피하도록 또 다른 환경에 설치합니다. 공식 YOLOX source를 고정 commit으로 받아 CUDA forward pass까지 확인합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_yolox.ps1
```

Raspberry Pi bootstrap 학습 명령은 다음과 같습니다.

```powershell
.\.venv-yolox\Scripts\python.exe -m yolox.tools.train `
  -f configs\yolox_s_micropcb.py `
  -d 1 -b 8 --fp16 -o `
  -c weights\pretrained\yolox_s.pth
```

YOLO11m은 같은 640 입력과 같은 split으로 실행합니다.

```powershell
yolo detect train `
  model=yolo11m.pt `
  data=data\processed\micropcb_rpi\dataset.yaml `
  imgsz=640 epochs=100 batch=8 device=0 seed=42 `
  project=runs\yolo11 name=yolo11m_micropcb
```

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

## 라이선스 주의

- YOLOX source는 Apache-2.0입니다.
- Ultralytics 패키지는 AGPL-3.0 또는 별도 Enterprise license 조건이 적용됩니다.
- private GitHub 저장소라고 해서 dataset·모델·소프트웨어의 재배포 조건이 사라지지는 않습니다.
