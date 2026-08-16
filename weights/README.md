# Model artifacts

Store only curated artifacts here:

- best native checkpoint (`.pt` or `.pth`)
- optional resume checkpoint (`last.pt` or `latest_ckpt.pth`)
- portable ONNX export
- run manifest and evaluation report

TensorRT `.engine` files are intentionally excluded because they should be built on the target Ubuntu platform and GPU.

## Pretrained source

`pretrained/yolox_s.pth`

- source: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
- upstream: Megvii-BaseDetection/YOLOX
- license: Apache-2.0
- bytes: `72,089,125`
- SHA-256: `F55DED7181E1B0C13285C56E7790B8F0E8F8DB590FE4EDB37F0B7F345C913A30`
