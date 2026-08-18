# 데이터 수집 및 분할 계획

## 선정 결정

`class별 1,000장`은 augmentation이나 연속 영상 frame 수가 아니라, provenance가 확인되고 중복을 제거한
승인 실사를 목표로 합니다. 공개 자료만으로 모든 세부 MCU class를 1,000장씩 확보할 수 없으므로
Raspberry Pi bootstrap 공개 데이터와 자체 컨베이어 촬영을 병행합니다.

## 현재 class와 확보 상태

| class | 목표 | 현재 확인된 자료 | 판정 |
|---|---:|---|---|
| `raspberry_pi_sbc` | 1,000+ | micro-PCB의 3개 RPi 모델 합계 1,875장 | 수량 PASS, 실물 다양성 부족 |
| `raspberry_pi_pico` | 1,000+ | 공개 독립 실사 1,000장 미확인 | TO_VERIFY / 자체 촬영 필요 |
| `stm32_dev_board` | 1,000+ | 공개 독립 실사 1,000장 미확인 | TO_VERIFY / 자체 촬영 필요 |
| `stm32_bare_ic` | 1,000+ | 공개 독립 실사 1,000장 미확인 | TO_VERIFY / 자체 촬영 필요 |
| `small_component_generic` | 1,000+ | conveyor SMD 공개 raw 후보 존재, 아직 로컬 미승인 | TO_VERIFY |

micro-PCB의 장수는 많아도 동일 실물 보드를 위치·회전·원근만 바꿔 반복 촬영한 구조입니다. 따라서
1,875장을 1,875개의 독립 실물로 해석하지 않습니다. 현재 source partition은 같은 조건의 capture
serial 1–4를 train, 5를 validation으로 둔 것이며 physical specimen 독립성은 확인되지 않았습니다.
Formal AP 전에 condition/physical-item 기준으로 재분할하고, 최종 test는 실제 컨베이어의 새로운
실물·촬영 session으로 구성합니다.

## 소형 칩 class 정의

첨부된 컨베이어 사진 수준에서는 작은 black package를 검출할 수 있어도 exact part number를 확정하기
어렵습니다. 다음 2단계로 분리합니다.

1. detector: `chip/package`의 위치와 거친 종류를 bbox로 검출
2. recognition: 충분한 pixel-per-character를 가진 crop에 classifier 또는 OCR 적용

외형이 같은 SOT/LQFP 계열을 전기적 기능별 class로 바로 나누지 않습니다. 마킹을 읽을 수 없는 이미지에는
exact SKU label을 만들지 않고 `unknown/unreadable` 정책을 적용합니다.

## 자체 촬영 단위

| 기록 항목 | 예시 | 용도 |
|---|---|---|
| `physical_item_id` | 실제 칩/보드 개체 ID | 동일 실물의 train/test 누수 방지 |
| `capture_session` | 날짜·카메라·렌즈·조명 setting | session domain shift 측정 |
| `video_run_id` | 컨베이어 1회 운전 ID | 인접 frame 누수 방지 |
| `lot_id` | 제조 lot/구매 batch | 표면 marking·외관 shortcut 방지 |
| `source_url`, `creator`, `license` | 공개 자료 provenance | GitHub 재배포 판단 |
| `sha256`, `phash` | exact/near duplicate | 중복 감사 |

조명, 반사, 거리, 회전, 초점, motion blur, 밀도, 겹침, 배경, 부분가림과 객체가 없는 hard negative를
의도적으로 포함합니다. 촬영한 연속 frame은 먼저 `video_run_id`로 묶은 뒤 split합니다.

## split 및 라벨 승인

- 권장 시작 split: train/validation/test = 70/15/15. 실제 수량보다 specimen/session 독립성을 우선합니다.
- test는 모델·threshold 선정에 사용하지 않고 Ubuntu 현장 최종 평가 전까지 잠급니다.
- 대표 이미지 200장은 pilot이며, 각 이미지에 보이는 **모든 목표 instance**를 라벨링합니다.
- auto-label 결과는 `pending`; 사람이 box/class를 승인·수정한 train label은 `reviewed_train`으로
  구분합니다. 모델 제안 없이 만든 잠긴 validation/test와 섞지 않습니다.
- validation/test는 자동 라벨을 사용하지 않습니다.
- augmentation은 split 이후 train에만 적용하며 고유 실사 수에 포함하지 않습니다.

## 공개 데이터와 GitHub

원본 이미지는 private 저장소라도 저작권 조건이 사라지지 않으므로 Git에 넣지 않습니다. 저장소에는
다운로더, provenance manifest, split ID, class map, audit report만 넣고 원본은 로컬/NAS/object storage에
보관합니다. Public Domain rights statement 또는 CC BY 자료를 사용할 때도 creator와 rights/license URL을 유지합니다.

구체적인 출처와 라이선스는 [`configs/datasets.curated.yaml`](../configs/datasets.curated.yaml)과
[`configs/sources.wikimedia.yaml`](../configs/sources.wikimedia.yaml)을 기준으로 합니다.
