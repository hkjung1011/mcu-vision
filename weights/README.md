# 모델 가중치 및 배포 산출물

이 디렉터리에는 출처, SHA-256, 실행 protocol과 평가 binding을 확인한 선별 산출물만 보존합니다.
Binary checkpoint와 ONNX는 Git LFS로 관리합니다. TensorRT `.engine`은 대상 Ubuntu 장치의 GPU와
TensorRT version에 종속되므로 저장소에 넣지 않고 그 장치에서 생성합니다.

## 현재 목록

| 경로 | 상태 | 검증 범위 |
|---|---|---|
| `pretrained/yolo11m.pt` | CONFIRMED | Ultralytics COCO 초기 가중치; MCU/SMD detector 아님 |
| `pretrained/yolox_s.pth` | CONFIRMED | YOLOX upstream COCO 초기 가중치; MCU/SMD detector 아님 |
| `progress/rpi_bootstrap_2026-08-18/` | HISTORICAL | 당시 6-run 계획의 중간 snapshot; `release_ready=false` 보존 |
| `trained/rpi_phash_v2_paired2_yolo11m/` | **FORMAL PASS** | paired2 policy, native/ONNX val·internal-test 및 공개 artifact gate |
| `trained/rpi_phash_v2_paired2_yolox_s/` | **FORMAL PASS** | paired2 policy, native/ONNX val·internal-test 및 공개 artifact gate |

Formal PASS는 `paired_2seed_descriptive` tier의 artifact 승격 요건을 통과했다는 뜻입니다. 표본은 matched
seed 두 쌍(`n=2`, `df=1`)이며 test도 locked internal pHash split입니다. 독립 촬영 성능, 통계적
유의성, 모집단 우월성 또는 production readiness를 의미하지 않습니다.

모델별 사용 범위와 입력 계약은 [YOLO11m 모델 카드](../docs/model_cards/rpi_phash_v2_paired2_yolo11m.ko.md)와
[YOLOX-S 모델 카드](../docs/model_cards/rpi_phash_v2_paired2_yolox_s.ko.md)를 참조하십시오.

## Formal artifact 식별자

| 모델 | 선택 run | Native SHA-256 | ONNX SHA-256 | ONNX val/test AP50-95 |
|---|---|---|---|---:|
| YOLO11m | `yolo11m_seed43` | `54cafd9348bde613945218fb0696e703de76ddef6d09210ab30d092bd1e3f2d4` | `ad488f6758af0cb7cfe1937ae128411ab73feeec5f3bdc8377f07ffaebc7ebfc` | `1.0000000000 / 1.0000000000` |
| YOLOX-S | `yolox_s_seed43` | `2c0dda7e81c9c664bd870d4ba9e98ea905a4f0186580439a9050f2e1d92f1f4f` | `589b5c9aec5459473567c1f513649fb309e7440b52122e02682493c02f72c6a1` | `0.9908195841 / 0.9852172241` |

각 release manifest는 checkpoint load, SHA-256, native↔ONNX numeric equivalence, val/test formal split,
policy binding, 공개 경로 privacy scan과 Git LFS gate를 검증합니다. Ubuntu에서 사용하기 전에도 clone한
파일의 SHA-256을 다시 확인하십시오.

```powershell
git lfs install
git lfs pull
git lfs ls-files
Get-FileHash .\weights\trained\rpi_phash_v2_paired2_yolo11m\* -Algorithm SHA256
Get-FileHash .\weights\trained\rpi_phash_v2_paired2_yolox_s\* -Algorithm SHA256
```

PyTorch `.pt/.pth`는 pickle 기반입니다. 출처가 불명확하거나 release manifest의 SHA-256과 일치하지
않는 checkpoint를 로드하지 마십시오.

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
