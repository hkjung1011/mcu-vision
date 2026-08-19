# STM32/SMD 공개 데이터 원천 검토

기준일: **2026-08-19**

## 선정 결정

현재 canonical 6-class에 직접 반입할 수 있는 우선 원천은 Dainius `smdComponents` v2의 네 SMD
class뿐입니다. Roboflow 인증 후 받은 raw-images COCO ZIP도 곧바로 승인하지 않고
`CANDIDATE_ONLY_NOT_APPROVED` quarantine으로 반입합니다. STM32 개발 보드와 단품 IC 패키지는 공개
자료만으로 수량·정체성·촬영 독립성을 충족하지 못하므로 자체 촬영을 주 원천으로 둡니다.

## 원천별 판정

| 원천 | 공식 근거 | 표시 수량·형식 | 권리·접근 | canonical mapping | 판정 |
|---|---|---|---|---|---|
| Dainius `smdComponents` v2 | [Roboflow version 2](https://universe.roboflow.com/dainius/smdcomponents/dataset/2) | 2,997 images, 2,397/300/300, 4-class bbox, raw-images | Universe가 Public Domain으로 표시하나 PDM은 license grant가 아님; COCO export는 sign-in 필요 | `Condensator/Resistor/Diode/Transistor`를 exact alias로 0–3에 mapping | **우선 후보 / 인증 대기** |
| IoTKITs v1 | [Mendeley Data](https://data.mendeley.com/datasets/x5thzmkxhy/1), [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12149570/) | 설명 3,200 images; 실제 COCO ZIP 3,108 images, 33 categories, `STM32` 100 images/101 boxes | CC BY 4.0 표시; Google Image Search 유래 설명과 논문 provenance 서술 간 차이 | `STM32`를 4에 조건부 mapping 가능 | **quarantine / provenance·leakage BLOCKED** |
| DeteksiKomponen | [Roboflow project](https://universe.roboflow.com/yolopcv/deteksikomponen) | 158 source images, `esp32`·`stm32`; v5 379 generated images | uploader CC BY 4.0 표시, export login 필요; 촬영 provenance 없음 | board bbox는 4에 조건부 mapping | **보류** |
| Microcontroller Detection | [Roboflow project](https://universe.roboflow.com/nguyen-phutn/microcontroller-detection-zgjau) | 510 source images, 4 classes 중 `stm32 blue pill` | uploader CC BY 4.0 표시; advertised v3 archive URL은 2026-08-19 기준 404 | 의미상 4에 mapping 가능 | **archive 접근 복구 전 보류** |
| DoAnTN | [Roboflow project](https://universe.roboflow.com/minh-73cyf/doantn-ttmtb) | 101 source images, 20 classes 중 `IC STM32F`; v1 243 generated | uploader CC BY 4.0 표시, 촬영 provenance 없음 | 공개 sample은 board-mounted IC이므로, 개발 보드에 실장되지 않은 개별 package를 뜻하는 5에 직접 mapping 불가 | **직접 mapping 제외** |
| Wikimedia Commons | [Commons API 설정](../configs/sources.wikimedia.yaml) | 현재 dry-run: board 13, IC 13 keyword 후보; bbox 없음 | 파일별 CC0/CC BY/CC BY-SA와 attribution을 기록 | 사람 QA와 bbox 작성 후 4–5에만 사용 가능 | **보조 후보 / 수량 부족** |
| ElectroCom61 v2 | [Mendeley Data DOI](https://doi.org/10.17632/6scy6h8sjz.2) | 2,121 images, 12,937 annotations, 61 classes, YOLO | CC BY 4.0, 공개 다운로드 | generic component·`IC-chip`은 SMD 또는 STM32 정체성을 입증하지 않음 | **직접 mapping 제외** |
| PCB-Vision | [RODARE record](https://rodare.hzdr.de/record/2704) | 53 RGB + 53 hyperspectral cubes, semantic masks | CC BY 4.0, Open Access | generic `IC`·`Capacitors`; STM32/SMD identity 없음 | **직접 mapping 제외** |
| PCB DSLR | [Zenodo 3886553](https://doi.org/10.5281/zenodo.3886553) | 748 PCB images, 9,313 generic IC boxes | non-commercial research use로 명시 | IC box는 STM32 여부를 구분하지 않음 | **공개 배포 학습에서 제외** |
| WACV PCB component dataset | [공식 project page](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection) | 47 high-resolution images, 31 types, 약 62k instances | 공식 페이지에서 재사용 license를 확인하지 못함 | generic resistor/capacitor/IC를 canonical STM32/SMD로 자동 변환할 수 없음 | **rights 확인 전 제외** |
| DeepICLogo | [공식 dataset page](https://physicaldb.ece.ufl.edu/index.php/deeplogoic/) | 980 IC crops, 119 logo classes, 1,010 logo instances | 명시적 data license 없음, verified-credential access 원칙 | package가 아닌 logo 검출이며 ST logo도 STM32를 보장하지 않음 | **서면 허가 전 제외** |

ElectroCom61은 전자부품 detector의 auxiliary research에는 유용하지만, 일반 capacitor·resistor와 SMD
package를 같은 class로 합치면 현재 ontology 의미가 바뀝니다. `IC-chip`도 STM32 top marking 또는
specimen metadata가 없으므로 `stm32_bare_ic`에 넣지 않습니다. PCB DSLR은 generic IC localization
pretraining 후보일 수 있으나, non-commercial 조건과 공개 model artifact의 관계를 별도로 확인하기
전에는 이 공개 저장소의 학습 원천으로 사용하지 않습니다.

## IoTKITs archive audit

공식 public file에서 `32.v1i.coco.zip`을 `data/staging/incoming/`의 Git-ignore 경로에 내려받아
read-only 구조 감사를 수행했습니다. 원본 archive SHA-256은
`5a22c88daafa8cc21b4fda46259cb9dbb13e2abff9ac6745b81d933c5d2b3bf5`입니다.

| 항목 | 실제 관찰값 | 판정 |
|---|---:|---|
| archive bytes | 141,166,460 | 기록 완료 |
| COCO images | train 2,488 + valid 620 = 3,108 | 설명의 3,200과 불일치 |
| categories | 33 | category 0은 비정상 장문 이름, annotation 0 |
| `STM32` | train 80 images/81 boxes, valid 20/20 | 100 images/101 boxes |
| exact image SHA | 92 unique / 100 | 8 duplicate pairs |
| pHash≤4 graph | 32 components, largest 19 images | 독립 실사 100장 아님 |
| train↔valid pHash≤4 | 81 candidate pairs, 4 cross-split components | provider split 평가 금지 |
| test split | 없음 | 독립 평가 불가 |

README는 resize 640×640 stretch, augmentation 없음으로 기록합니다. 그러나 다른 `rf` suffix를 가진
동일 bytes가 같은 split에 반복되고, 연속 촬영·유사 장면이 split 경계를 넘습니다. pHash는 자동
판정이 아니라 검수 후보이지만, 현재 수치만으로도 provider split을 독립 validation으로 해석할 수
없습니다. 추적 가능한 source URL·creator가 없는 파일은 공개 모델 학습에서 제외하고, 남은 이미지도
physical item/session group을 새로 구성해야 합니다. 기계 판독 결과는
[`iotkits_v1.acquisition-probe.json`](../data/manifests/iotkits_v1.acquisition-probe.json)에 보존합니다.

## Commons diagnostic probe

`mcu-collect-commons --dry-run --limit 50`으로 파일을 받지 않고 title discovery를 수행했습니다.

| class route | discovered titles | keyword-filtered candidates | 승인 수량 |
|---|---:|---:|---:|
| `stm32_dev_board` | 47 | 13 | 0 |
| `stm32_bare_ic` | 40 | 13 | 0 |

제외어 보강 전 초기 설정에서 격리한 `stm32_bare_ic` 후보 7개를 품질 점검했습니다. 구성은 STM32 package
close-up 1개, 개발 보드 2개, exposed die/SEM microscopy 4개였습니다. 이는 검색 title과 category가
class label이 될 수 없음을 보여 주는 diagnostic sample이며 dataset 수량으로 계산하지 않습니다. 원본과
임시 manifest는 Git에 넣지 않았습니다. 설정에는 board/die 관련 명시적 제외어를 추가했지만 최종 승인은
여전히 사람이 image content와 marking을 확인해야 합니다.

## 반입 Gate

| 단계 | 필수 근거 | 현재 |
|---|---|---|
| RIGHTS | source/version/author/권리 표시/archive SHA-256 | Dainius archive 대기 |
| IDENTITY | source label alias 또는 STM32 marking/specimen metadata | SMD alias만 계약 완료 |
| INTEGRITY | decode·dimension·bbox·중복·class 분포 감사 | archive 대기 |
| REVIEW | 전체 target instance bbox와 CVAT round-trip | 실제 export 대기 |
| SPLIT | physical item/session/near-duplicate group 분리 | 자체 촬영 전 |
| TRAINING | 1e smoke → 최대 10e pilot → 최대 50e candidate | 승인 dataset 전 실행 금지 |

Roboflow 로그인을 우회하거나 제3자 mirror를 사용하지 않습니다. 사용자가 공식 v2 COCO ZIP을
`data/staging/incoming/smdcomponents-v2-coco.zip`에 둔 뒤에만 documented ingest 명령을 실행합니다.
