# Raspberry Pi SBC 1-class 중간 체크포인트

이 디렉터리는 2026-08-18 공개 snapshot의 **재현·검토용 중간 체크포인트**입니다.
`release_ready=false`이며 독립 Ubuntu 카메라 평가를 통과한 운영 배포용 모델이 아닙니다. 전체 수치와
검증 한계는 [중간 보고서](../../../reports/progress/rpi_bootstrap_2026-08-18/README.md), 의도된 용도와
금지된 용도는 [한국어 모델 카드](MODEL_CARD.ko.md)를 확인하십시오.

## SHA-256 확인

Windows:

```powershell
Get-FileHash .\weights\progress\rpi_bootstrap_2026-08-18\*.* -Algorithm SHA256
.\.venv-collect\Scripts\python.exe .\scripts\verify_progress_snapshot.py `
  --report-dir reports\progress\rpi_bootstrap_2026-08-18
```

Linux:

```bash
sha256sum weights/progress/rpi_bootstrap_2026-08-18/*.pt \
  weights/progress/rpi_bootstrap_2026-08-18/*.pth
```

## YOLO11m 추론 예시

```python
from ultralytics import YOLO

model = YOLO(
    "weights/progress/rpi_bootstrap_2026-08-18/yolo11m_seed42_best.pt",
    task="detect",
)
results = model.predict(source="your_image.jpg", imgsz=640, conf=0.25)
```

공개 전 로컬 경로 metadata를 제거했으며, 원본과 공개본의 tensor bitwise equality, zero-input forward
`max_abs_difference=0`, Ultralytics load를 검증했습니다. 이 검증은 checkpoint 변환 무결성에 대한
것이며 독립 카메라 성능을 보증하지 않습니다.

## YOLOX-S 추론 모델 로드 예시

```python
import torch
from yolox.exp import get_exp

exp = get_exp("configs/yolox_s_micropcb.py", None)
model = exp.get_model()
checkpoint = torch.load(
    "weights/progress/rpi_bootstrap_2026-08-18/yolox_s_seed42_best.pth",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

카메라 전처리에는 학습과 동일한 `640×640 letterbox`, framework별 color order와 scale, 좌표 역변환,
class-aware NMS가 필요합니다. confidence `0.25`는 공개 보고서의 operating point이지 현장 배포용으로
검증된 임계값이 아닙니다. 현재 가중치는 formal ONNX deployment package가 아니므로 Ubuntu 카메라에
연동한 뒤 별도의 동등성·정확도·지연 시간 검증이 필요합니다.

`yolox_s_seed43_resume_epoch_70.pth`에는 epoch 70의 model/optimizer state가 포함되어 있으나, 공개본을
이용한 직접 resume 절차는 end-to-end로 재검증하지 않았습니다. 따라서 `RESUME_CANDIDATE`로만
취급합니다.

## 보안과 라이선스

- PyTorch `.pt/.pth`는 pickle 기반입니다. SHA-256과 출처를 확인한 신뢰 파일만 로드하십시오.
- YOLO11/Ultralytics checkpoint: `AGPL-3.0` 또는 별도 Ultralytics Enterprise 계약 조건
- YOLOX checkpoint: `Apache-2.0`
- 학습 데이터: Adam Byerly, micro-PCB Images, `CC BY 4.0`
