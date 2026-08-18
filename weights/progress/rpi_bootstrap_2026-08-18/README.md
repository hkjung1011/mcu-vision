# RPi 1-class progress checkpoints

이 디렉터리는 2026-08-18 시점의 **중간 진행본**입니다. `release_ready=false`이며 독립 Ubuntu 카메라
test를 통과한 production model이 아닙니다. 전체 수치·한계·원본 hash는
[`reports/progress/rpi_bootstrap_2026-08-18`](../../../reports/progress/rpi_bootstrap_2026-08-18/README.md)을
확인하십시오.

## SHA-256 확인

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

## YOLO11m inference

```python
from ultralytics import YOLO

model = YOLO(
    "weights/progress/rpi_bootstrap_2026-08-18/yolo11m_seed42_best.pt",
    task="detect",
)
results = model.predict(source="your_image.jpg", imgsz=640, conf=0.25)
```

이 checkpoint는 공개 전 로컬 path metadata를 제거했습니다. 원본과 공개본의 tensor bitwise equality,
zero-input forward `max_abs_difference=0`, Ultralytics load를 검증했습니다.

## YOLOX-S inference model load

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

카메라 전처리에는 학습과 동일한 `640×640 letterbox`, RGB/BGR 순서, scale, NMS 설정이 필요합니다.
현재 progress weight는 formal ONNX deployment package가 아니므로 Ubuntu camera acceptance에는
사용 후 별도 검증이 필요합니다.

`yolox_s_seed43_resume_epoch_70.pth`에는 epoch 70의 model/optimizer state가 들어 있지만, 공개본에서
직접 resume하는 절차는 아직 end-to-end 재검증하지 않았습니다. 따라서 `RESUME_CANDIDATE`로만
취급합니다.

## 보안과 라이선스

- PyTorch `.pt/.pth`는 pickle 기반입니다. SHA-256과 출처를 확인한 신뢰 파일만 로드하십시오.
- YOLO11/Ultralytics checkpoint: `AGPL-3.0` 또는 별도 Ultralytics Enterprise 계약 조건
- YOLOX checkpoint: `Apache-2.0`
- 학습 데이터: Adam Byerly, micro-PCB Images, `CC BY 4.0`
