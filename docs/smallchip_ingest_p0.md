# STM32/SMD 데이터 반입·오토라벨 P0–P2

## 판정

이 구현은 **데이터 계약과 fail-closed 검증 경로**만 제공합니다. 저장소에는 Dainius SMD 원본이나
STM32 자체 촬영본이 없으며, 승인 canonical STM32/SMD 데이터 수량은 계속 **0장**입니다. Wikimedia
Commons `stm32_dev_board` 후보 11건은 model-assisted draft box 11개와 보수적 leakage group 6개를
포함한 검수 전용 bundle로 준비되었지만, 사람 승인은 **0건**이고 학습 사용은 금지됩니다. 테스트의 작은
단색 이미지는 코드 검증 중 임시 폴더에만 생성되는 synthetic fixture이고 학습 데이터가 아닙니다.

## 고정 ontology

`configs/classes.smd_v1.yaml`의 ID는 재정렬하거나 재사용하지 않습니다.

| ID | class | 승인 전제 |
|---:|---|---|
| 0 | `smd_capacitor` | Dainius source label `Condensator`만 명시적 alias로 변환 |
| 1 | `smd_resistor` | source label `Resistor` |
| 2 | `smd_diode` | source label `Diode` |
| 3 | `smd_transistor` | source label `Transistor` |
| 4 | `stm32_dev_board` | STM32 탑재 보드라는 실물/BOM 근거 |
| 5 | `stm32_bare_ic` | 읽을 수 있는 top marking 또는 신뢰 가능한 specimen metadata |

Package 외형만으로 STM32 또는 exact part number를 확정하지 않습니다. STM32 class 반입에는 이미지별
`specimen_id`, 승인된 evidence type, `STM32...`로 시작하는 `verified_part_number`가 필요합니다. Exact
SKU가 목표이면 detector crop 뒤 OCR/classifier를 별도 구축합니다.

## 원천 권리 기록

Dainius source는 다음 문구를 그대로 기록합니다.

> PDM-1.0 asserted by the Roboflow Universe project maintained by Dainius; this records the source assertion and Public Domain Mark is not a license grant.

PDM은 source가 표시한 권리 상태이며 이 프로젝트가 새 license를 부여한다는 뜻이 아닙니다. 원본 ZIP과
이미지는 `.gitignore` 대상이고 Git에는 URL, author, version, rights statement, archive/image SHA-256,
수량과 감사 결과만 올립니다. Provider의 random split은 formal validation/test로 인정하지 않고
`bootstrap_train_only`로 반입합니다.

## Wikimedia Commons STM32 수동 seed 검수

Revision-bound Commons 수집물 가운데 현재 ontology와 일치하는 항목은 `stm32_dev_board` 후보
**11건**입니다. 준비된 reference COCO에는 후보별 model-assisted draft box **11개**가 있으며,
유사 촬영·동일 실물 가능성을 보수적으로 묶은 leakage group은 **6개**입니다. 이는 데이터 승인이나
ground truth 확정이 아니라 사람 검수를 위한 초기 제안입니다. 사람 승인 수는 **0건**이고,
`training_use_allowed=false`를 유지합니다. 현재 로컬 bundle은
`data/staging/manual_seed/wikimedia_commons_stm32_dev_board_v2_review2/`에 있습니다. 원본 이미지와
검수 bundle은 Git 추적 제외(Gitignored)이며 공개 Git에는 provenance·hash·상태 증빙만 기록합니다.
추적되는 상태 증빙은
[`wikimedia_stm32_dev_board.manual-seed-review-preparation.json`](../data/manifests/wikimedia_stm32_dev_board.manual-seed-review-preparation.json)입니다.

새 review task를 준비하는 명령은 다음과 같습니다. 출력 경로가 이미 존재하면 덮어쓰지 않고 실패하므로
재실행에는 새 `run-id`와 새 directory를 사용합니다.

```powershell
.\.venv-collect\Scripts\mcu-prepare-manual-seed.exe `
  --output-dir data\staging\manual_seed\<new-review-task> `
  --run-id <new-review-run-id>
```

다음 단계는 CVAT 원본 해상도에서 11개 후보의 `stm32_dev_board` identity를 사람이 확인하고, 각 bbox의
경계·누락·오검출을 전수 수정하는 것입니다. 이 검수와 COCO round-trip 검증이 끝나기 전에는 reference
COCO나 draft annotation을 canonical train split으로 복사하지 않습니다.

## 1. 인증 필요 상태 기록

2026-08-19 로그아웃 상태에서 공식 v2 페이지와 COCO export 경로를 확인했다. v2는 `raw-images`,
2,997 images(2,397 train / 300 valid / 300 test), `Auto-Orient` 적용, augmentation 없음으로 표시된다.
그러나 `https://universe.roboflow.com/dainius/smdcomponents/dataset/2/download/coco`는 즉시
`Login or create a free account` dialog를 열고, 계속하면 Terms of Service와 Privacy Policy에
동의한다고 고지한다. 따라서 인증 없는 공식 COCO ZIP 취득은 확인되지 않았고 우회하지 않는다.

공식 근거 URL은 다음과 같다.

- version page: `https://universe.roboflow.com/dainius/smdcomponents/dataset/2`
- Roboflow download guide: `https://docs.roboflow.com/universe/download-a-universe-dataset`
- Roboflow CLI authentication/download reference: `https://github.com/roboflow/roboflow-python/blob/main/CLI-COMMANDS.md`
- 원 논문: `https://doi.org/10.3390/app12115608`

원 논문은 initial dataset을 3,005 images(2,405/300/300)로 기술하지만 Roboflow v2 표시는
2,997 images(2,397/300/300)다. 또한 논문의 Data Availability Statement와 Universe의 Public Domain
표시는 서로 다른 provenance assertion이므로, PDM 표시는 license grant로 확대 해석하지 않는다.

```powershell
Set-Location <repository-root>
.\.venv-collect\Scripts\mcu-download-curated.exe --dataset smd_components_raw
```

이 명령은 다운로드를 우회하지 않고 `MANUAL_AUTH_REQUIRED` source record만 생성합니다. 사용자가
Roboflow에 sign in(또는 free account 생성)하고 고지된 약관을 확인한 뒤 **raw-images version 2의
COCO JSON ZIP**을 `data/staging/incoming/smdcomponents-v2-coco.zip`에 둡니다. Augmented version은
사용하지 않습니다. 이것이 현재 필요한 단일 외부 조치이며, API key는 UI ZIP export에는 필요하지
않고 CLI 자동화에만 별도 인증 수단으로 고려합니다.

## 2. Candidate-only canonical COCO 반입과 YOLO 파생

```powershell
.\.venv-collect\Scripts\mcu-ingest-detection.exe `
  --archive data\staging\incoming\smdcomponents-v2-coco.zip `
  --dataset smd_components_raw `
  --ontology configs\classes.smd_v1.yaml `
  --output-root data\quarantine\dainius_smdcomponents_v2
```

반입기는 다음 항목을 fail-fast 검증합니다.

- ZIP path traversal, symlink/encryption, 중복 member, member/총 해제 크기·압축비·pixel 상한
- source category의 dataset별 allowlist와 ontology alias 매핑(Dainius는 정확히 4개 source label)
- 이미지당 여러 bbox와 빈 negative 이미지
- bbox 범위, crowd annotation, exact file duplicate
- archive/image/ontology SHA-256와 rights/source ID
- 출력이 프로젝트의 Git-ignore 대상 `data/raw|staging|processed|quarantine|cache/<dataset>` 하위인지

출력은 `CANDIDATE_ONLY_NOT_APPROVED`입니다. `source_manifest.json`의 count는 archive를 실제 읽은
결과이며 웹페이지의 2,997 images 또는 논문의 3,005 images를 그대로 복사한 값이 아닙니다.
전체 archive 검증은 sibling 임시 폴더에서 끝낸 뒤에만 atomic rename으로 게시합니다. 오류가 나면
부분 출력 폴더를 승인 데이터처럼 남기지 않습니다.

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

Reference COCO의 모든 image에는 stable `mcu_image_id` 또는 `mcu_asset_id`, encoded image SHA-256,
width/height가 있어야 합니다. COCO와 YOLO export 모두 6개 frozen class map의 ID/name을 정확히 전부
포함해야 하며 subset/prefix는 거부합니다. COCO gate는 image/class/bbox와
`occluded`/`truncated`를 비교합니다. YOLO gate는 format 한계 때문에 image/class/bbox만 비교하며
속성 보존의 근거로 사용할 수 없습니다.

## 4. Hash-bound 오토라벨

`mcu-autolabel-yolo`는 caller가 만든 manifest 자체를 신뢰 근거로 사용하지 않습니다. 저장소에 추적된
`configs/data_trust_registry.yaml`의 `APPROVED` dataset entry와 그 entry가 가리키는 locked split
evidence가 먼저 있어야 합니다. 현재 registry는 의도적으로 비어 있으므로 실제 승인 dataset **0장**
상태에서는 fail closed입니다. 승인 전에는 entry를 임의로 추가하지 않습니다.

실행 시 다음 입력을 모두 요구합니다.

- `mcu.autolabel-source.v1`, role=`unlabeled_train`, stable image ID/path/dimensions/SHA-256
- project trusted registry ID/hash, exact dataset-entry hash, locked split evidence hash
- frozen ontology
- checkpoint SHA-256, 전체 frozen class map과 training annotation state를 담은 teacher manifest
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

Source의 모든 image는 approved `unlabeled_train` set과 정확히 같아야 합니다. Validation/test SHA를
새 path나 ID로 재선언하거나 일부만 선택해도 proposal 생성 전에 거부됩니다. Calibration threshold는
finite `[0,1]` 값만 허용합니다. Calibration이 없으면 모든 proposal은 `review_required`이고, 수동
`--high-confidence` override도 금지됩니다. 출력 `pending_reference.coco.json`은 이후 승격의 exact
reference이며 stable image binding과 full class map hash를 보존합니다.

## 5. 사람 승인 승격

Review manifest schema는 `mcu.cvat-review.v1`이며 reviewer ID/name, timezone이 있는 승인 시각, CVAT
task ID/job IDs, pending run/export/round-trip/ontology/image-list SHA-256와 모든 이미지의 disposition을
포함해야 합니다.

```powershell
.\.venv-collect\Scripts\mcu-promote-reviewed.exe `
  --pending-run-manifest runs\autolabel\smd_pending_v1\run_manifest.json `
  --review-manifest data\staging\cvat\review_manifest.json `
  --cvat-export data\staging\cvat\reviewed-coco.zip `
  --roundtrip-reference runs\autolabel\smd_pending_v1\pending_reference.coco.json `
  --roundtrip-report data\reports\cvat\coco-roundtrip.json `
  --ontology configs\classes.smd_v1.yaml `
  --source-manifest data\manifests\unlabeled_smd.session-001.json `
  --source-root data\staging\unlabeled_smd `
  --trusted-registry configs\data_trust_registry.yaml `
  --output data\manifests\smd_reviewed_train.promotion.json `
  --filtered-coco data\processed\smd_reviewed_train\instances_train.json
```

Pending의 image ID/path/dimensions/SHA, full class map, ontology, reference COCO, CVAT export와 round-trip
report가 정확히 같은 run에 묶이지 않으면 승격하지 않습니다. 모든 이미지의 review가 끝나야 하며
reviewer/task/job/hash 중 하나라도 없으면 실패합니다. `rejected` image와 annotation은 filtered canonical
COCO에서 실제 제거되고, manifest에는 included/excluded stable ID와 downstream에서 반드시 확인할 filtered
COCO SHA-256이 기록됩니다. 또한 승격기는 `--source-manifest`의 image binding을 `--source-root`의 실제
파일 및 `--trusted-registry`의 현재 승인 evidence와 다시 대조합니다. Pending 생성 당시의 trust 기록만
일치하고 현재 원천·registry가 달라진 경우에도 fail closed입니다. `--trusted-registry`는 프로젝트의
canonical `configs/data_trust_registry.yaml`과 정확히 같은 경로여야 하며 임의 승인 registry로 대체할 수
없습니다. 승격 manifest만으로 validation/test 사용은 허용되지 않습니다.

## Locked 평가 evidence

현재 RPi bootstrap protocol은 train/val 6개 기본 hash에 locked test image list/records와 COCO annotation
attributes를 더한 9개 formal hash를 사용합니다. YOLO와 COCO 각각에서 train/val/test 사이 encoded image
SHA-256 중복을 학습 전 거부합니다. 이것은 file-level leakage gate이지, RPi 원본에 없는 physical item
독립성을 새로 입증하는 근거는 아닙니다. Test가 없는 다른 protocol은
`locked_test_evidence_enabled: false`로 기존 6-hash 계약을 유지합니다.

## 완료 Gate

| Gate | PASS 조건 |
|---|---|
| RIGHTS | source/version/author/PDM assertion/archive SHA 존재 |
| INTEGRITY | decode·dimension·bbox 오류 0, exact duplicate 0 |
| ONTOLOGY | dataset allowlist·source alias와 전체 frozen ID/hash 일치 |
| ROUNDTRIP | stable image binding + full class map + COCO/YOLO image/class/bbox 차이 0; COCO 속성 차이 0 |
| AUTOLABEL | trusted registry/locked split + exact image set + teacher/calibration hash binding |
| APPROVAL | 전 이미지 disposition + exact reference/reviewer/CVAT/export/hash + rejected 물리 제거 |
| COUNT | image 수와 instance 수를 별도로 보고; augmentation/인접 frame을 독립 1,000장으로 주장하지 않음 |
| EVALUATION | 자체 촬영 physical-item/session 독립 gold validation/test 사용 |
