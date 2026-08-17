# YOLO txt ↔ COCO JSON 데이터 동등성 Gate

정식 학습 전에 두 framework가 **같은 이미지·class·bbox**를 읽는지 먼저 고정한다. 원본
annotation 파일의 SHA-256만 비교하면 표현 형식과 JSON 순서가 달라 항상 다른 값이 되므로,
두 형식을 하나의 canonical detection record로 변환한 뒤 비교한다.

## 실행

```powershell
.\.venv-collect\Scripts\python.exe -m mcu_data.dataset_evidence `
  --yolo-data data\processed\micropcb_rpi_phash_v2\dataset.yaml `
  --coco-train data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_train2017.json `
  --coco-val data\processed\micropcb_rpi_phash_v2_coco\annotations\instances_val2017.json `
  --output-dir data\evidence\micropcb_rpi_phash_v2
```

이동한 dataset.yaml에 과거 절대 경로가 남아 있으면 `--yolo-root <현재 데이터 루트>`를
추가한다. COCO 이미지 폴더가 표준 `train2017`, `val2017` 배치가 아니면
`--coco-train-images`, `--coco-val-images`로 지정한다.

종료 코드는 `0=PASS`, `1=동등성 FAIL`, `2=입력/구조 ERROR`다. FAIL/ERROR를 학습
승인으로 바꾸지 않는다.

## 생성 증거

| 파일 | 의미 |
|---|---|
| `dataset_evidence.json` | run manifest에 주입할 6개 release-gate SHA-256 |
| `canonical_dataset_manifest.json` | canonical 규칙·split별 수량·하위 hash |
| `class_map.json` | YOLO zero-based class index와 NFC-normalized 이름 |
| `train_image_list.json`, `val_image_list.json` | basename, encoded-file SHA-256, 실제 decode 크기 |
| `canonical_*_records.jsonl` | 정렬된 image/class/bbox record와 개별 `record_sha256` |
| `dataset_equivalence_report.json` | YOLO-only/COCO-only 차이 수와 제한된 예시 |

run manifest의 `dataset`에는 다음 값을 그대로 넣는다.

```python
from mcu_data.dataset_evidence import load_dataset_evidence

manifest["dataset"].update(
    load_dataset_evidence(Path("data/evidence/micropcb_rpi_phash_v2/dataset_evidence.json"))
)
```

`load_dataset_evidence()`는 JSON에 적힌 문자열만 신뢰하지 않고 연결된 6개 artifact를
다시 hash한다. 학습 wrapper는 더 강한 `verify_dataset_against_evidence()`를 호출해 현재
YOLO/COCO 파일을 다시 canonicalize한다. 따라서 evidence 생성 뒤 이미지·label·class map이
바뀐 경우에도 학습 시작 전에 실패한다.

## Canonical 규칙

- 절대 경로와 source record 순서는 hash에 포함하지 않는다.
- image key는 Unicode NFC basename이며 같은 split의 basename 중복은 오류다.
- 이미지 파일 bytes의 SHA-256과 실제 decode 크기를 image identity에 포함한다.
- COCO category id는 class name으로 YOLO zero-based index에 대응시킨다.
- YOLO normalized `cxcywh`를 실제 pixel `xywh`로 변환하고 기본 0.001 pixel 단위,
  `ROUND_HALF_EVEN`으로 고정한다. 이는 YOLO 8-decimal 저장의 왕복 오차를 제거한다.
- annotation id, JSON key/list 순서, COCO `area`, `segmentation`, `supercategory`는
  detection 동등성에 영향을 주지 않는다.
- YOLO txt로 보존할 수 없는 COCO `iscrowd=1`과 segmentation-only label은 오류다.

Dataset을 추가·삭제·재분할·재라벨링했다면 기존 evidence를 재사용하지 말고 빈 새
output directory에 다시 생성한다.
