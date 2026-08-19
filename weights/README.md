# 모델 가중치 및 배포 산출물

이 디렉터리에는 출처와 SHA-256을 확인한 선별 산출물만 보존합니다.

- native best checkpoint (`.pt` 또는 `.pth`)
- 필요한 경우 resume checkpoint (`last.pt` 또는 `latest_ckpt.pth`)
- 휴대 가능한 ONNX export
- 실행 manifest와 평가 보고서

TensorRT `.engine`은 목표 Ubuntu 장치의 GPU·TensorRT 버전에 종속되므로 저장소에 포함하지 않고 해당
장치에서 생성합니다.

## 현재 목록

| 경로 | 상태 | 용도·검증 범위 |
|---|---|---|
| `pretrained/yolo11m.pt` | CONFIRMED | YOLO11m fine-tuning용 공식 Ultralytics COCO 초기 가중치 |
| `pretrained/yolox_s.pth` | CONFIRMED | YOLOX-S fine-tuning용 upstream COCO 초기 가중치 |
| `progress/rpi_bootstrap_2026-08-18/` | INTERIM_PROGRESS | 완료 best 3개와 중단 best/resume 2개, `release_ready=false` |
| `trained/` | NOT AVAILABLE | protocol을 충족하여 승격된 정식 학습 가중치 없음 |

사전학습 체크포인트는 MCU/SMD 검출기가 아닙니다. 중간 체크포인트의 범위와 제한은
[`progress/rpi_bootstrap_2026-08-18/README.md`](progress/rpi_bootstrap_2026-08-18/README.md) 및
[한국어 모델 카드](progress/rpi_bootstrap_2026-08-18/MODEL_CARD.ko.md)를 확인하십시오.
`trained/<release>/`에는 학습 완료, checkpoint SHA-256 일치, 공통 비교 protocol 통과가 모두 확인된
산출물만 승격합니다.

## Git LFS 및 무결성 확인

Binary checkpoint와 ONNX 파일은 Git LFS로 관리합니다. clone 후 다음과 같이 확인하십시오.

```powershell
git lfs pull
git lfs ls-files
Get-FileHash .\weights\pretrained\yolox_s.pth -Algorithm SHA256
Get-FileHash .\weights\pretrained\yolo11m.pt -Algorithm SHA256
```

PyTorch `.pt/.pth`는 pickle 기반이므로 출처가 불명확하거나 SHA-256이 일치하지 않는 파일을 로드하지
마십시오.

## 사전학습 가중치 출처

### `pretrained/yolo11m.pt`

- source: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
- upstream: Ultralytics Assets, YOLO11 detection COCO checkpoint
- license embedded in checkpoint: `AGPL-3.0`; Ultralytics licensing terms apply
- bytes: `40,684,120`
- SHA-256: `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`

### `pretrained/yolox_s.pth`

- source: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
- upstream: Megvii-BaseDetection/YOLOX
- license: `Apache-2.0`
- bytes: `72,089,125`
- SHA-256: `F55DED7181E1B0C13285C56E7790B8F0E8F8DB590FE4EDB37F0B7F345C913A30`
