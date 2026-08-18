# STM32/SMD 데이터 반입·오토라벨 P0

## 판정

이 구현은 **데이터 계약과 fail-closed 검증 경로**만 제공합니다. 저장소에는 Dainius SMD 원본이나
STM32 자체 촬영본이 없으며, 승인 canonical STM32/SMD 데이터 수량은 계속 **0장**입니다. 테스트의
작은 단색 이미지는 코드 검증 중 임시 폴더에만 생성되는 synthetic fixture이고 학습 데이터가 아닙니다.

## 고정 ontology

`configs/classes.smd_v1.yaml`의 ID는 재정렬하거나 재사용하지 않습니다.

| ID | class | 승인 전제 |
|---:|---|---|
| 0 | `smd_capacitor` | source label `Condensator`/`Capacitor`를 명시적 alias로 변환 |
| 1 | `smd_resistor` | source label `Resistor` |
| 2 | `smd_diode` | source label `Diode` |
| 3 | `smd_transistor` | source label `Transistor` |
| 4 | `stm32_dev_board` | STM32 탑재 보드라는 실물/BOM 근거 |
| 5 | `stm32_bare_ic` | 읽을 수 있는 top marking 또는 신뢰 가능한 specimen metadata |

Package 외형만으로 STM32 또는 exact part number를 확정하지 않습니다. Exact SKU가 목표이면 detector
crop 뒤 OCR/classifier를 별도 구축합니다.

## 원천 권리 기록

Dainius source는 다음 문구를 그대로 기록합니다.

> PDM-1.0 asserted by the Roboflow Universe project maintained by Dainius; this records the source assertion and Public Domain Mark is not a license grant.

PDM은 source가 표시한 권리 상태이며 이 프로젝트가 새 license를 부여한다는 뜻이 아닙니다. 원본 ZIP과
이미지는 `.gitignore` 대상이고 Git에는 URL, author, version, rights statement, archive/image SHA-256,
수량과 감사 결과만 올립니다. Provider의 random split은 formal validation/test로 인정하지 않고
`bootstrap_train_only`로 반입합니다.

## 1. 인증 필요 상태 기록

```powershell
Set-Location <repository-root>
.\.venv-collect\Scripts\mcu-download-curated.exe --dataset smd_components_raw
```

이 명령은 다운로드를 우회하지 않고 `MANUAL_AUTH_REQUIRED` source record만 생성합니다. Roboflow에서
직접 인증한 뒤 **raw-images version 2의 COCO ZIP**을 `data/staging/incoming/`에 둡니다. Augmented
version은 사용하지 않습니다.

## 2. Candidate-only canonical COCO 반입과 YOLO 파생

```powershell
.\.venv-collect\Scripts\mcu-ingest-detection.exe `
  --archive data\staging\incoming\smdcomponents-v2-coco.zip `
  --dataset smd_components_raw `
  --ontology configs\classes.smd_v1.yaml `
  --output-root data\quarantine\dainius_smdcomponents_v2
```

반입기는 다음 항목을 fail-fast 검증합니다.

- ZIP path traversal, 중복 member, decode/dimension 오류
- source category의 ontology/alias 매핑
- 이미지당 여러 bbox와 빈 negative 이미지
- bbox 범위, crowd annotation, exact file duplicate
- archive/image/ontology SHA-256와 rights/source ID

출력은 `CANDIDATE_ONLY_NOT_APPROVED`입니다. `source_manifest.json`의 count는 archive를 실제 읽은
결과이며 웹페이지의 2,997 images 또는 논문의 3,005 images를 그대로 복사한 값이 아닙니다.

## 3. CVAT 왕복 검증

Canonical COCO를 CVAT에 import하고 사람이 검수한 뒤 COCO와 YOLO를 각각 export합니다. 이 프로젝트는
CVAT 계정/서버를 자동 생성하지 않으며 받은 export를 검증합니다.

```powershell
.\.venv-collect\Scripts\mcu-verify-cvat-roundtrip.exe `
  --reference-coco data\...\instances_train2017.json `
  --roundtrip data\staging\cvat\reviewed-coco.zip `
  --format coco --ontology configs\classes.smd_v1.yaml `
  --output data\reports\cvat\coco-roundtrip.json

.\.venv-collect\Scripts\mcu-verify-cvat-roundtrip.exe `
  --reference-coco data\...\instances_train2017.json `
  --roundtrip data\staging\cvat\reviewed-yolo.zip `
  --format yolo --ontology configs\classes.smd_v1.yaml `
  --output data\reports\cvat\yolo-roundtrip.json
```

COCO gate는 image/class/bbox와 `occluded`/`truncated`를 비교합니다. YOLO gate는 format 한계 때문에
image/class/bbox만 비교하며 속성 보존의 근거로 사용할 수 없습니다.

## 4. Hash-bound 오토라벨

`mcu-autolabel-yolo`는 이제 다음 입력을 모두 요구합니다.

- `mcu.autolabel-source.v1`, role=`unlabeled_train`, 실제 image-list SHA-256
- frozen ontology
- checkpoint SHA-256, class map과 training annotation state를 담은 teacher manifest
- calibration 사용 시 role=`gold_validation_locked`와 teacher/ontology/image-list SHA-256

```powershell
.\.venv-yolo11\Scripts\mcu-autolabel-yolo.exe `
  --source data\staging\unlabeled_smd `
  --source-manifest data\manifests\unlabeled_smd.session-001.json `
  --model runs\teachers\smd_v1\best.pt `
  --teacher-manifest runs\teachers\smd_v1\teacher_manifest.json `
  --ontology configs\classes.smd_v1.yaml `
  --calibration runs\teachers\smd_v1\gold_calibration.json `
  --tile-size 640 --tile-overlap 0.20 --run-id smd_pending_v1
```

Validation, test, gold 또는 hash가 다른 source는 proposal 생성 전에 거부됩니다. Calibration이 없으면
모든 proposal은 `review_required`이고, 수동 `--high-confidence` override도 금지됩니다.

## 5. 사람 승인 승격

Review manifest schema는 `mcu.cvat-review.v1`이며 reviewer ID/name, timezone이 있는 승인 시각, CVAT
task ID/job IDs, pending run/export/round-trip/ontology/image-list SHA-256와 모든 이미지의 disposition을
포함해야 합니다.

```powershell
.\.venv-collect\Scripts\mcu-promote-reviewed.exe `
  --pending-run-manifest runs\autolabel\smd_pending_v1\run_manifest.json `
  --review-manifest data\staging\cvat\review_manifest.json `
  --cvat-export data\staging\cvat\reviewed-coco.zip `
  --roundtrip-report data\reports\cvat\coco-roundtrip.json `
  --ontology configs\classes.smd_v1.yaml `
  --output data\manifests\smd_reviewed_train.promotion.json
```

COCO round-trip PASS, 모든 이미지 검수, reviewer/task/job/hash 중 하나라도 없으면 승격하지 않습니다.
승격 manifest만으로 validation/test 사용은 허용되지 않습니다.

## 완료 Gate

| Gate | PASS 조건 |
|---|---|
| RIGHTS | source/version/author/PDM assertion/archive SHA 존재 |
| INTEGRITY | decode·dimension·bbox 오류 0, exact duplicate 0 |
| ONTOLOGY | source alias와 frozen ID/hash 일치 |
| ROUNDTRIP | COCO 및 YOLO image/class/bbox 차이 0; COCO 속성 차이 0 |
| AUTOLABEL | `unlabeled_train` + source/teacher/calibration hash binding |
| APPROVAL | 전 이미지 disposition + reviewer/CVAT/export/hash 증거 |
| COUNT | image 수와 instance 수를 별도로 보고; augmentation/인접 frame을 독립 1,000장으로 주장하지 않음 |
| EVALUATION | 자체 촬영 physical-item/session 독립 gold validation/test 사용 |
