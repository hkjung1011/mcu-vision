# Model artifacts

Store only curated artifacts here:

- best native checkpoint (`.pt` or `.pth`)
- optional resume checkpoint (`last.pt` or `latest_ckpt.pth`)
- portable ONNX export
- run manifest and evaluation report

TensorRT `.engine` files are intentionally excluded because they should be built on the target Ubuntu platform and GPU.

## Current inventory

| artifact | status | purpose |
|---|---|---|
| `pretrained/yolo11m.pt` | CONFIRMED | Official Ultralytics COCO initialization for YOLO11m fine-tuning |
| `pretrained/yolox_s.pth` | CONFIRMED | Upstream COCO initialization for YOLOX-S fine-tuning |
| `trained/` | NOT AVAILABLE | No protocol-matched full training release has been promoted yet |

The pretrained checkpoint is not a trained MCU/SMD detector. A checkpoint is added to `trained/<release>/`
only after the run is complete, its SHA-256 matches the run manifest, and the comparison protocol passes.

Binary checkpoints and ONNX files are stored through Git LFS. After cloning, verify them with:

```powershell
git lfs pull
git lfs ls-files
Get-FileHash .\weights\pretrained\yolox_s.pth -Algorithm SHA256
Get-FileHash .\weights\pretrained\yolo11m.pt -Algorithm SHA256
```

## Pretrained source

`pretrained/yolo11m.pt`

- source: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
- upstream: Ultralytics Assets, YOLO11 detection COCO checkpoint
- license embedded in checkpoint: AGPL-3.0 / Ultralytics licensing terms apply
- bytes: `40,684,120`
- SHA-256: `D5FFC1A674953A08E11A8D21E022781B1B23A19B730AFC309290BD9FB5305B95`

`pretrained/yolox_s.pth`

- source: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
- upstream: Megvii-BaseDetection/YOLOX
- license: Apache-2.0
- bytes: `72,089,125`
- SHA-256: `F55DED7181E1B0C13285C56E7790B8F0E8F8DB590FE4EDB37F0B7F345C913A30`
