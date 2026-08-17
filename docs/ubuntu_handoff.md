# Ubuntu 카메라 시험 인계서

## 인계 결정

Windows에서는 학습과 portable checkpoint를 만들고, 검증된 ONNX export를 향후 추가합니다. TensorRT `.engine`은 Ubuntu 목표
장치에서 해당 CUDA/TensorRT/GPU 조합으로 생성합니다. `.engine`은 환경 의존성이 크므로 GitHub에서
Windows와 Ubuntu 사이의 공통 교환 형식으로 사용하지 않습니다.

## GitHub에서 내려받을 항목

- 코드와 고정 config
- dataset/class/split manifest와 SHA-256
- native best checkpoint (`.pt`/`.pth`)
- ONNX export와 native 동등성 보고서(현재 NOT IMPLEMENTED)
- run manifest, common evaluation, latency/VRAM 보고서
- preprocessing, NMS/confidence 설정

원본 데이터와 TensorRT engine은 기본적으로 포함하지 않습니다.

```bash
git lfs install
git clone https://github.com/hkjung1011/mcu-vision.git
cd mcu-vision
git lfs pull
git lfs ls-files
```

Ubuntu용 설치 script와 목표 장치별 lock file은 아직 검증되지 않았습니다. 목표 GPU/Jetson/CPU가 확정된
뒤 해당 장치에서 Python, PyTorch, CUDA/cuDNN, ONNX Runtime 또는 TensorRT 버전을 고정해야 합니다.

## 모델 동등성 확인

Ubuntu camera에 연결하기 전에 Windows에서 사용한 고정 이미지 subset으로 다음을 확인합니다.

| 검사 | 합격 기준 |
|---|---|
| checkpoint/ONNX SHA-256 | release manifest와 동일 |
| RGB/BGR, letterbox, normalization | 학습 wrapper와 동일 |
| output decode, NMS IoU, max detections | run protocol과 동일 |
| native ↔ ONNX 예측 | 허용 오차 내 box/score/class 일치 |
| ONNX ↔ TensorRT 예측 | FP16 허용 오차와 AP 저하 기준을 사전 정의 후 충족 |

TensorRT 변환 성공만으로 배포 검증을 끝내지 않고 동일 validation subset의 AP와 개별 예측 차이를 다시
계산합니다.

## 실제 카메라 성능 측정

데스크톱 GPU의 model-only FPS 대신 아래 end-to-end 구간을 batch 1로 측정합니다.

`camera capture → color conversion → resize/letterbox → inference → decode/NMS → result handoff`

- warmup 이후 raw latency sample 최소 500개
- p50/p95 latency와 sustained FPS
- GPU/CPU/RAM/VRAM, 온도, clock, power
- camera 해상도·FPS·exposure·gain·focus·조명
- 객체 pixel width/height, marking의 pixel-per-character
- miss, wrong class, duplicate, false positive를 분리한 오류 log

## 현장 test set

Windows 학습/validation에서 사용하지 않은 실물, lot, 촬영 session으로 구성합니다. 카메라 threshold와
preprocessing을 test 결과를 보며 반복 조정하면 test leakage가 되므로 validation에서 먼저 고정합니다.

| REQ ID | Ubuntu 검증 | 상태 |
|---|---|---|
| UB-01 | 목표 장치에서 checkpoint/ONNX load | NOT VERIFIED |
| UB-02 | native/ONNX/TensorRT 정확도 동등성 | NOT VERIFIED |
| UB-03 | 카메라 end-to-end p50/p95/FPS | NOT VERIFIED |
| UB-04 | 새 실물·session test의 AP/P/R/F1 | NOT VERIFIED |
| UB-05 | 조명·반사·blur·겹침별 오류 분석 | NOT VERIFIED |

목표 Ubuntu 하드웨어, 카메라 모델·해상도, 요구 FPS가 확정되면 이 문서의 허용 기준을 수치로 갱신합니다.
