# micro-PCB leakage-safe split v2

## 선정 결정

정식 Raspberry Pi bootstrap 학습에는 `micropcb_rpi_phash_v2`를 사용합니다. 단순
`condition_group_id` 분리만 적용한 v1은 pHash 거리 4 이하 후보 3,511쌍 중 357쌍이 split 경계를
넘으므로 정식 평가용으로 사용하지 않습니다.

v2는 다음 순서로 배정합니다.

1. 같은 `condition_group_id`의 capture 5장을 하나의 단위로 묶습니다.
2. 기존 `duplicates.json`의 모든 pHash 거리 0/2/4 edge로 condition group들을 연결합니다.
3. 연결요소 전체를 하나의 `leakage_component_id`로 배정합니다.
4. 여러 Raspberry Pi 모델을 연결하는 component는 train에만 배정합니다.
5. 나머지는 seed 42의 deterministic SHA-256 순서와 subset dynamic programming으로 배정하여
   모델별 80/10/10 group 목표를 정확히 맞춥니다.

## 재생성 명령

출력 경로는 비어 있는 새 경로여야 합니다. 명령은 다른 내용의 기존 파일을 덮어쓰지 않으며 원본을
삭제하지 않습니다.

```powershell
.venv-collect\Scripts\python.exe -m mcu_data.condition_split `
  --input-manifest data\manifests\micropcb_raspberry_pi_sbc.csv `
  --output-manifest data\manifests\micropcb_raspberry_pi_sbc.phash_v2.csv `
  --source-root data\raw\curated\micro_pcb_images `
  --yolo-output-root data\processed\micropcb_rpi_phash_v2 `
  --coco-output-root data\processed\micropcb_rpi_phash_v2_coco `
  --duplicates-report data\reports\micropcb_rpi_audit\duplicates.json `
  --seed 42 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1
```

## 생성 결과

| 항목 | train | val | test |
|---|---:|---:|---:|
| 전체 image | 1,500 | 195 | 180 |
| condition group | 300 | 39 | 36 |
| Raspberry Pi 모델별 image | 500 | 65 | 60 |
| Raspberry Pi 모델별 condition group | 100 | 13 | 12 |
| pHash 후보 pair | 2,829 | 366 | 316 |

- train↔val/test condition group overlap: `0`
- cross-split pHash 후보 pair: `0 / 3,511`
- cross-split pHash component: `0 / 219`
- 최대 component: condition group 18개, image 90장
- 두 모델 이상을 잇는 component 2개: train에 강제 배정
- assignment SHA-256: `5a818e0fd1f264a1a364cfd0cfe2280e72ec52bfbaa35ff425224bbf4a0a4e33`

기계 판독 가능한 전체 근거는
`data/manifests/micropcb_raspberry_pi_sbc.phash_v2.summary.json`에 저장됩니다. YOLO dataset YAML은
`data/processed/micropcb_rpi_phash_v2/dataset.yaml`, YOLOX용 COCO annotation은
`data/processed/micropcb_rpi_phash_v2_coco/annotations/instances_{split}2017.json`입니다.

## 잔여 위험

v2가 검증하는 것은 제공된 `condition_group_id`와 현재 감사에서 검출된 pHash 후보의 split 독립성입니다.
원본 dataset에는 실제 보드 개체를 나타내는 `physical_item_id`가 없으므로 물리 개체 독립성은 여전히
`NOT VERIFIED`입니다. 따라서 이 결과는 Raspberry Pi bootstrap 비교에는 사용할 수 있지만, 새 카메라·새
보드에 대한 최종 일반화 성능을 입증하지는 않습니다.
