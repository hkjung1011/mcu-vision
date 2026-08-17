# Ubuntu 카메라 시험 인계서

## 인계 결정

Windows에서는 학습 checkpoint를 만들고 `scripts/export_deployment.py`로 batch 1, FP32, 640 px
ONNX를 생성합니다. 이 도구는 COCO가 참조하는 실제 validation 이미지 1장을 같은 전처리로 입력하여
native output과 ONNX Runtime output의 모든 원소를 비교하고, PASS일 때만 일반 `.onnx` 이름으로
승격합니다. AI 생성 이미지는 이 검증에 사용하지 않습니다. TensorRT `.engine`은 Ubuntu 목표 장치의
CUDA/TensorRT/GPU 조합으로 생성하며 Windows와 Ubuntu 사이의 공통 교환 형식으로 사용하지 않습니다.

## GitHub에서 내려받을 항목

- 코드와 고정 config
- dataset/class/split manifest와 SHA-256
- native best checkpoint (`.pt`/`.pth`)
- ONNX export와 같은 stem의 `*.deployment.json` 동등성 보고서
- run manifest, common evaluation, latency/VRAM 보고서
- preprocessing, NMS/confidence 설정

원본 데이터와 TensorRT engine은 기본적으로 포함하지 않습니다.

이 저장소의 Python 요구사항은 **3.11 (`>=3.11,<3.12`)**입니다. Ubuntu 장치에서 먼저
`python3.11 --version`이 성공하는지 확인하고, 없다면 해당 Ubuntu 배포판의 관리 절차로 Python 3.11과
venv 모듈을 설치한 뒤 진행합니다. 다른 minor version의 `python3`로 lock 설치를 강행하지 않습니다.

```bash
git lfs install
git clone https://github.com/hkjung1011/mcu-vision.git
cd mcu-vision
git lfs pull
git lfs ls-files
```

## Windows export와 승격 gate

아래 경로의 `<RUN>`과 `<RELEASE>`만 실제 release 경로로 바꿉니다. condition split과 일치하는
COCO annotation/image root를 두 모델에 똑같이 사용합니다.

```powershell
.\.venv-yolo11\Scripts\python.exe .\scripts\export_deployment.py `
  --framework yolo11 `
  --checkpoint .\runs\benchmarks\<RUN>\native\weights\best.pt `
  --run-manifest .\runs\benchmarks\<RUN>\run_manifest.json `
  --comparison-dir .\runs\comparisons\<COMPARISON> `
  --output-dir .\weights\trained\<RELEASE>\yolo11m `
  --output-name yolo11m_rpi_b1_640 `
  --coco-annotations .\data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_val2017.json `
  --coco-images .\data\processed\micropcb_rpi_phash_v2_coco\val2017

.\.venv-yolox\Scripts\python.exe .\scripts\export_deployment.py `
  --framework yolox `
  --checkpoint .\runs\benchmarks\<RUN>\best_ckpt.pth `
  --run-manifest .\runs\benchmarks\<RUN>\run_manifest.json `
  --comparison-dir .\runs\comparisons\<COMPARISON> `
  --output-dir .\weights\trained\<RELEASE>\yolox_s `
  --output-name yolox_s_rpi_b1_640 `
  --coco-annotations .\data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_val2017.json `
  --coco-images .\data\processed\micropcb_rpi_phash_v2_coco\val2017
```

기본 수치 gate는 `abs(error) <= 1e-3 + 1e-4 * abs(native)`를 모든 raw output 원소가 만족하는지
확인합니다. 이 기준은 FP32 CPU graph의 export 수치 동등성 smoke gate이며 AP 동등성이나 현장 정확도를
대체하지 않습니다. 결과 JSON에는 checkpoint/ONNX/검증 이미지/COCO annotation의 SHA-256, byte size,
class map, 전처리, output 의미, ONNX opset, raw error, post-NMS 개수가 기록됩니다. FAIL export는
`*.verification-failed.onnx`로 격리되고 카메라 runner가 받아들이지 않습니다.
`.pt`/`.pth`는 Python pickle을 포함할 수 있으므로 이 명령에는 해당 GitHub release에서 직접 받은
자체 checkpoint처럼 출처와 SHA-256을 확인한 파일만 입력합니다.

export 뒤에는 [ONNX 전체 split 검증](onnx_split_evaluation.md)의 formal val/test를 모델별로 모두
실행하고 `scripts/promote_deployment.py`를 통과시킵니다. Git에 올릴 최종 배포 증빙은
`reports/deployments/<RELEASE>/deployment_release_manifest.json`의 `status=PASS`와 모든 `gates=PASS`를
확인한 release뿐입니다. 이 manifest는 promoted native checkpoint, ONNX, 원 comparison,
val/test metric의 SHA-256을 하나로 묶으며 기존 경로를 덮어쓰지 않습니다.

## Ubuntu CPU 카메라 첫 실행

목표 GPU용 CUDA/TensorRT 조합이 확정되기 전에는 CPU provider로 artifact와 카메라 경로부터 검증합니다.
아래 명령은 **fresh clone** 기준입니다. `<RELEASE>`는
`reports/deployments/<RELEASE>/deployment_release_manifest.json`이 존재하고 모든 gate가 PASS인 모델별
release 이름으로 바꿉니다.

```bash
sudo apt-get update
sudo apt-get install -y git-lfs libgl1 libglib2.0-0
python3.11 --version
git lfs install
git clone https://github.com/hkjung1011/mcu-vision.git
cd mcu-vision
git lfs pull
git lfs ls-files

python3.11 -m venv .venv-deploy
source .venv-deploy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/deploy-cpu.lock.txt

python scripts/infer_onnx_camera.py \
  --metadata weights/trained/<RELEASE>/yolo11m/yolo11m_rpi_b1_640.deployment.json \
  --release-manifest reports/deployments/<RELEASE>/deployment_release_manifest.json \
  --camera 0 --provider CPUExecutionProvider --warmup 30
```

정식 모드에서는 `--release-manifest`가 필수입니다. runner는 카메라를 열기 전에 manifest의 val/test
formal PASS, 모든 gate, published report, deployment metadata, ONNX의 SHA-256을 확인합니다. 명시적
`--diagnostic`과 `DIAGNOSTIC_ONLY` metadata 조합만 이 승격 manifest 요구에서 제외됩니다.

창 없이 **warmup 제외 500 frame**의 batch-1 E2E p50/p95를 유효한 단일 JSON 파일로 남길 때는
`--output-json`을 사용합니다. 진행 상황은 stderr로 출력되므로 stdout을 `tee ...json`으로 저장하지
않습니다.

```bash
python scripts/infer_onnx_camera.py \
  --metadata weights/trained/<RELEASE>/yolox_s/yolox_s_rpi_b1_640.deployment.json \
  --release-manifest reports/deployments/<RELEASE>/deployment_release_manifest.json \
  --camera 0 --provider CPUExecutionProvider \
  --warmup 30 --max-frames 530 --min-measured-frames 500 --no-display \
  --output-json runs/ubuntu_camera/<RELEASE>/camera_yolox_s_cpu.json
```

`--confidence`와 `--nms-iou`를 생략하면 release JSON의 값을 그대로 사용합니다. ONNX 파일의 SHA-256이
JSON과 다르면 카메라를 열기 전에 즉시 중단됩니다. GPU provider는 목표 Ubuntu 장치의 CUDA/cuDNN
호환성을 확인한 뒤 별도 lock으로 고정합니다.

fresh clone에는 `data/processed/`와 `runs/deployment_eval/` 원본이 포함되지 않습니다. 이는 원본 이미지,
중간 prediction과 license-sensitive data를 Git에 넣지 않는 정책입니다. 카메라 실행에는 이 데이터가
필요하지 않고, 승격된 정식 val/test 수치 증빙은 `reports/deployments/<RELEASE>/`에 있습니다. Ubuntu에서
full-split 추론을 다시 계산하려면 exact COCO annotation과 processed image를 별도 전달하거나 허가된
원본에서 재생성한 뒤 protocol/split manifest의 SHA-256과 일치하는지 먼저 검증해야 합니다.

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

camera runner의 최상위 `status=PASS`는 **runtime 측정 frame 수와 artifact 무결성 gate의 PASS**입니다.
출력의 `scope=runtime measurement only`와 `acceptance_status=NOT_EVALUATED`가 나타내듯 정확도·안전
acceptance PASS가 아닙니다. 정확도 acceptance는 새 실물/lot/session에 사람이 검수한 ground truth를
만들고 AP/P/R/F1 및 condition별 miss/false-positive 허용기준을 별도로 충족해야 합니다.

## 현장 test set

Windows 학습/validation에서 사용하지 않은 실물, lot, 촬영 session으로 구성합니다. 카메라 threshold와
preprocessing을 test 결과를 보며 반복 조정하면 test leakage가 되므로 validation에서 먼저 고정합니다.

| REQ ID | Ubuntu 검증 | 상태 |
|---|---|---|
| UB-01 | 목표 장치에서 checkpoint/ONNX load | ONNX export tooling IMPLEMENTED; 장치 실행 NOT VERIFIED |
| UB-02 | native/ONNX/TensorRT 정확도 동등성 | NOT VERIFIED |
| UB-03 | 카메라 end-to-end p50/p95/FPS | NOT VERIFIED |
| UB-04 | 새 실물·session test의 AP/P/R/F1 | NOT VERIFIED |
| UB-05 | 조명·반사·blur·겹침별 오류 분석 | NOT VERIFIED |

목표 Ubuntu 하드웨어, 카메라 모델·해상도, 요구 FPS가 확정되면 이 문서의 허용 기준을 수치로 갱신합니다.
