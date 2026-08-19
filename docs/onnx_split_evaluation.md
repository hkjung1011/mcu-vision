# ONNX 전체 split 검증

## 목적과 합격 판정

`scripts/evaluate_onnx_split.py`는 `scripts/export_deployment.py`가 PASS로 승격한 batch-1 FP32
ONNX를 COCO split의 **모든 이미지**에 실행합니다. 단일 export 이미지의 raw-output 검증을 대신하는
도구가 아니라, 그 다음 단계인 AP/AR/P/R/F1 동등성 gate입니다.

- `*.deployment.json`과 ONNX SHA-256을 먼저 확인합니다.
- COCO annotation, 개별 실사 이미지, image manifest, prediction, metric 파일의 SHA-256을 남깁니다.
- `predictions.coco.json`을 만든 뒤 기존 공통 `evaluate_predictions`를 그대로 호출합니다.
- `--native-final-metrics`가 있으면 선택 metric 각각에
  `abs(onnx-native) <= atol + rtol*abs(native)`를 적용하며 하나라도 벗어나면 exit code 2입니다.
- 정식 gate에서는 `--require-native-equivalence`와 `--require-split-evidence`를 사용합니다. split CSV의
  SHA-256, 정확한 image set, 각 processed image SHA-256이 summary/manifest와 다르면 추론 전에 중단합니다.
- 기본 `--mode formal`은 protocol/split evidence를 필수로 하고, validation에서는 native metric 비교도
  필수입니다. 근거 없이 기능만 점검할 때에만 `--mode diagnostic`을 명시하며 결과 status는
  `DIAGNOSTIC_ONLY`라서 release PASS로 오인할 수 없습니다.
- formal metadata의 `release_validation.status=PASS`, `formal_release=true`를 확인하며, val에 입력한
  native `final_metrics.json`의 실제 SHA-256이 원 comparison에서 동결한
  `release_validation.native_final_metrics_sha256`과 정확히 같아야 추론을 시작합니다.
- JSON 경로는 repository 기준 `portable_path`로 기록합니다. 다른 PC의 사용자명 경로에 의존하지
  않도록 ONNX는 metadata 옆의 `file_name`을 우선 찾습니다.

기본 metric gate는 AP50-95/AP50/AP75, AR1/10/100, precision/recall/F1에 절대 오차 0.005입니다.
이 0.5%p는 배포 승격용 시작 기준이며 실제 validation 실행 결과를 본 뒤 더 엄격하게 고정할 수
있습니다. 비교 기준 native metric은 **같은 COCO annotation, prediction floor, NMS, max detections,
operating confidence**에서 생성한 `final_metrics.json`이어야 합니다.

## Windows CPU 실행

```powershell
py -3.11 -m venv .venv-deploy-eval
.\.venv-deploy-eval\Scripts\python.exe -m pip install --upgrade pip
.\.venv-deploy-eval\Scripts\python.exe -m pip install -r requirements\deploy-eval-cpu.lock.txt

.\.venv-deploy-eval\Scripts\python.exe scripts\evaluate_onnx_split.py `
  --metadata weights\trained\<RELEASE>\yolo11m\yolo11m_rpi_b1_640.deployment.json `
  --coco-annotations data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_val2017.json `
  --coco-images data\processed\micropcb_rpi_phash_v2_coco\val2017 `
  --split val `
  --mode formal `
  --protocol configs\experiments\baseline_v1.yaml `
  --provider CPUExecutionProvider `
  --native-final-metrics runs\benchmarks\<YOLO11_RUN>\final_metrics.json `
  --require-native-equivalence `
  --split-manifest data\manifests\micropcb_raspberry_pi_sbc.phash_v2.csv `
  --split-summary data\manifests\micropcb_raspberry_pi_sbc.phash_v2.summary.json `
  --require-split-evidence `
  --output-dir runs\deployment_eval\<YOLO11_RUN>\val
```

YOLOX-S도 metadata와 native metric 경로만 바꾸고 동일 명령을 사용합니다. COCO AP를 위해
`--prediction-floor 0.001`이 기본이고, 공통 운영점 계산에는 metadata의 confidence(기본 0.25)를
사용합니다. override가 필요하면 `--operating-confidence`, `--nms-iou`, `--max-detections`를 명시해
결과 JSON에 고정합니다.

## phash_v2 test 실행

validation에서 threshold와 tolerance를 고정한 다음 test를 한 번 실행합니다. 정식 native↔ONNX
수치 동등성 gate는 validation에서 같은 run의 `final_metrics.json`으로 이미 통과해야 하며, test는
고정된 ONNX의 held-out bootstrap 수치를 한 번 산출하는 용도로 사용합니다.

test 산출물은 threshold 보정 자료가 아닙니다. `final_metrics.json`과
`onnx_split_evaluation.json`은 `threshold_selection.status=NOT_FOR_THRESHOLD_SELECTION`,
`usage=diagnostic_metrics_only`를 명시하고, test에서는 `best_f1_confidence`와
`autolabel_thresholds.csv`를 생성하지 않습니다. test 결과를 보고 confidence·pseudo-label·autolabel
threshold를 다시 고르면 독립 평가가 아니므로 deployment promotion이 이를 fail-closed로 거부합니다.

```powershell
.\.venv-deploy-eval\Scripts\python.exe scripts\evaluate_onnx_split.py `
  --metadata weights\trained\<RELEASE>\yolo11m\yolo11m_rpi_b1_640.deployment.json `
  --coco-annotations data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_test2017.json `
  --coco-images data\processed\micropcb_rpi_phash_v2_coco\test2017 `
  --split test `
  --mode formal `
  --protocol configs\experiments\baseline_v1.yaml `
  --provider CPUExecutionProvider `
  --split-manifest data\manifests\micropcb_raspberry_pi_sbc.phash_v2.csv `
  --split-summary data\manifests\micropcb_raspberry_pi_sbc.phash_v2.summary.json `
  --require-split-evidence `
  --output-dir runs\deployment_eval\<YOLO11_RUN>\test
```

이 val/test 두 명령을 같은 deployment metadata/ONNX 조합으로 실행해야 합니다. YOLOX-S는 두 명령의
metadata, release, output run 경로를 모두 YOLOX-S 값으로 바꿔 한 쌍을 별도로 생성합니다.

산출물은 다음과 같습니다.

| 파일 | 근거 |
|---|---|
| `image_manifest.json` | image ID/file name/bytes/SHA-256와 실사 표식 |
| `predictions.coco.json` | ONNX 전체 split 검출 결과 |
| `final_metrics.json` 및 CSV/PNG | 공통 COCO/운영점 evaluator 결과 |
| `onnx_split_evaluation.json` | 모든 입력·산출물 hash, runtime, latency, metric gate |

validation의 `autolabel_thresholds.csv`는 사람 검수가 필요한 후보값일 뿐입니다. test bundle에는 이
파일이 없어야 하며 confidence curve가 남더라도 진단용 수치 시각화로만 해석합니다.

## 최종 deployment release 승격

각 모델의 **동일한 ONNX**에 대해 formal val과 formal test를 모두 완료한 뒤에만 다음 명령을
실행합니다. `<RELEASE>`는 먼저 `scripts/promote_run.py`로 만든 native release 이름이며,
`<COMPARISON>`은 그 native release를 검증한 원본 comparison 디렉터리입니다.

```powershell
.\.venv-collect\Scripts\python.exe scripts\promote_deployment.py `
  --native-artifact reports\runs\<RELEASE>\artifact_manifest.json `
  --deployment-metadata weights\trained\<RELEASE>\yolo11m\yolo11m_rpi_b1_640.deployment.json `
  --onnx weights\trained\<RELEASE>\yolo11m\yolo11m_rpi_b1_640.onnx `
  --val-evaluation runs\deployment_eval\<YOLO11_RUN>\val\onnx_split_evaluation.json `
  --test-evaluation runs\deployment_eval\<YOLO11_RUN>\test\onnx_split_evaluation.json `
  --comparison-dir runs\comparisons\<COMPARISON>
```

YOLOX-S는 세 모델 경로를 `yolox_s` release 경로로 바꿔 별도로 실행합니다. 도구는 다음 항목이
전부 맞을 때만 `reports/deployments/<RELEASE>/deployment_release_manifest.json`을 새로 만듭니다.

- native checkpoint와 export checkpoint의 SHA-256 일치
- export의 `release_validation=PASS` 및 native↔ONNX raw-output 수치 검증 PASS
- 원 comparison의 run manifest와 `final_metrics.json` SHA-256 일치
- val/test 모두 `status=PASS`, `mode=formal`, protocol/split binding PASS
- val의 native metric equivalence PASS 및 비교에 동결된 native metric SHA-256 일치
- test의 `NOT_FOR_THRESHOLD_SELECTION` 선언 및 pseudo-label/autolabel 후보값 부재
- ONNX, metadata, COCO annotation, image manifest, prediction, metric 파일의 실제 SHA-256/byte 검증
- `.pt`/`.pth`와 `.onnx`의 Git LFS rule 존재

기존 release 보고서가 있으면 덮어쓰지 않고 즉시 중단합니다. 승격 보고서에는 경로를 정제한
JSON/CSV와 evaluator가 실제 수치로 만든 PNG만 복사하며, 원본 이미지와
`predictions.coco.json`은 복사하지 않습니다. checkpoint와 ONNX는
`weights/trained/<RELEASE>/`의 Git LFS 대상 파일을 그대로 참조합니다.

native run manifest schema 4는 weight release를 임의 파일 허용 디렉터리로 취급하지 않습니다. 루트의
native checkpoint exact inventory와 `formal_deployment_bundle_v1` 확장 계약을 미리 선언하며, 각
model subdirectory는 같은 stem의 `*.deployment.json`·`*.onnx` 정확히 두 파일만 가질 수 있습니다.
metadata는 native artifact/checkpoint/comparison와 ONNX byte size·SHA-256을 모두 결합해야 합니다.
임의 extra, 중첩 경로, stem 불일치 또는 hash 불일치는 native release 자체를 무효화합니다. 기존
schema 3 release는 checkpoint 단독일 때만 읽기 호환되며 deployment 확장은 허용하지 않습니다.

phash_v2 test는 condition/pHash 연결요소가 train/val과 겹치지 않는 bootstrap test이지만, 새
physical item/촬영 session/Ubuntu conveyor camera의 독립 acceptance test를 대체하지 않습니다.

## Ubuntu에서 full-split 재계산할 때

fresh clone에는 `data/processed/`, `runs/deployment_eval/`, 원본 prediction이 포함되지 않으므로 아래
명령은 clone 직후 그대로 실행되지 않습니다. 일반 Ubuntu 인계는 이미 승격된
`reports/deployments/<RELEASE>/` 증빙을 검증하고 카메라 runner를 실행하면 됩니다. full-split을 Ubuntu에서
다시 계산해야 할 때만 exact processed COCO image/annotation을 별도 전달하거나 허가된 원본에서
재생성하고, protocol/split manifest SHA-256 일치를 확인합니다. formal val 재계산에는 comparison이 동결한
원본 native `final_metrics.json`도 같은 SHA-256으로 전달해야 합니다.

데이터 경로만 점검하는 명시적 diagnostic 예시는 다음과 같습니다. 결과는 `DIAGNOSTIC_ONLY`이며 정식
release 증빙을 대체하지 않습니다.

```bash
python3.11 -m venv .venv-deploy-eval
source .venv-deploy-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/deploy-eval-cpu.lock.txt
python scripts/evaluate_onnx_split.py \
  --metadata weights/trained/<RELEASE>/yolo11m/yolo11m_rpi_b1_640.deployment.json \
  --coco-annotations data/processed/micropcb_rpi_phash_v2_coco/annotations/instances_val2017.json \
  --coco-images data/processed/micropcb_rpi_phash_v2_coco/val2017 \
  --split val --mode diagnostic --provider CPUExecutionProvider \
  --split-manifest data/manifests/micropcb_raspberry_pi_sbc.phash_v2.csv \
  --split-summary data/manifests/micropcb_raspberry_pi_sbc.phash_v2.summary.json \
  --require-split-evidence \
  --output-dir runs/deployment_eval/<YOLO11_RUN>/val
```
