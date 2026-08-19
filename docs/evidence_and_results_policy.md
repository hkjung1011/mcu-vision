# 결과 및 증빙 정책

## 판단 원칙

모델 성능 결론은 실제 실행에서 생성된 `terminal.log`, CSV, JSON만으로 내립니다. PNG는 사람이 비교를
빨리 확인하기 위한 파생 시각화이며 숫자의 원본도, 별도의 성능 증거도 아닙니다.

| 우선순위 | 파일 | 역할 |
|---:|---|---|
| 1 | `run_manifest.json`, resolved args/config | 실행 조건·환경·hash 확인 |
| 2 | `final_metrics.json`, `comparison.csv/json` | 공통 COCO AP/AR와 운영점 matcher의 최종 수치 |
| 3 | `epoch_metrics.csv/jsonl`, `latency_samples.csv` | epoch·raw sample 분석 |
| 4 | `terminal.log`, `comparison_terminal.txt` | 사용자가 본 CLI 출력과 오류 추적 |
| 5 | `*.png` | 위 수치의 비생성형 그래프·표 렌더링 |

표시 반올림 때문에 terminal 표와 JSON의 마지막 자릿수가 다르면 JSON float를 사용합니다. 숫자를 PNG에서
OCR로 다시 읽어 판단하지 않습니다.

## 이미지 생성 정책

- ImageGen, diffusion model 등 생성형 AI로 결과 이미지나 성능 그래프를 만들지 않습니다.
- 그래프는 Python `matplotlib`가 CSV/JSON 숫자를 결정론적으로 렌더링합니다.
- `terminal_summary.png`는 실제 Windows Terminal screenshot이 아니라 `comparison_terminal.txt`를 그대로
  렌더링한 이미지입니다.
- 실제 화면 screenshot이 필요하면 별도 증빙으로 추가할 수 있지만, 수치 판정에는 사용하지 않습니다.
- 생성된 PNG에는 source 파일, source SHA-256, renderer와 `GENERATIVE_AI=false`를 표시합니다.
- `evidence_manifest.json`과 `protocol_artifacts.json` 모두에
  `generative_ai_used_for_images=false`를 JSON boolean으로 기록하고, formal validator가 exact-match와
  각 artifact의 byte size/SHA-256을 확인합니다.

## 비교 가능성 gate

다음 핵심 조건 중 하나라도 다르면 `PROTOCOL: FAIL - NOT COMPARABLE`로 처리합니다.

- canonical dataset manifest·split·class-map·image-list hash
- YOLO label과 COCO annotation의 canonical box/class 동등성 hash
- model별 seed 집합
- train fraction, epochs, input size, micro-batch
- pretrained/full fine-tune/AMP 조건
- prediction floor, NMS IoU, operating confidence, match IoU
- common evaluator와 target GPU

FAIL 결과는 로깅·디버깅에는 보존하지만 모델 우열 표나 formal promotion에는 포함하지 않습니다.
Formal promotion에는 비교 가능성/release-ready gate 우회 옵션이 없습니다.

`comparable=true`는 **입력된 run끼리 조건이 같다**는 뜻일 뿐 정식 release 완료를 뜻하지 않습니다.
Legacy 3-seed policy에서 `release_ready=true`는 추가로 YOLO11m·YOLOX-S 각각 seed 42/43/44의 6개
complete non-smoke run, 100 epoch row, baseline 기대값, canonical dataset/YOLO↔COCO 동등성 hash,
final metric·latency·GPU log, checkpoint 존재와 SHA-256 일치까지 통과해야 합니다. Formal 판정은 선택한
policy ID와 SHA-256에 결합되며 다른 evidence tier의 조건을 묵시적으로 혼합하지 않습니다.

완료된 matched seed 42/43만 사용하는 별도 `paired_2seed_descriptive` tier는 기존 6-run gate를
대체하거나 완화하지 않습니다. 독립된 policy/attestation으로 exact 4-run matrix와 post-hoc 범위
축소를 명시하며, 결과는 기술통계로만 해석합니다. 실행법과 금지 주장은
[paired 2-seed formal release policy](paired_2seed_formal_release.md)를 따릅니다.

현재 paired release는 exact seed 42/43 네 run으로 formal gate를 통과했지만 `n=2`, `df=1`입니다.
Per-run metric, mean, sample SD와 paired seed delta만 보고하며 statistical significance와 population
superiority를 주장하지 않습니다. 목표 축소가 사후 결정되었고 seed 44는 양 모델에서 일괄 제외됐다는
decision record도 결과와 함께 보존합니다.

## 내부 test와 독립 test의 구분

RPi `test`는 동일 원천 데이터에서 condition 및 pHash component를 group으로 묶어 분리한 locked internal
split입니다. Exact SHA cross-split gate는 PASS하지만 physical item ID와 별도 camera session 독립성은
확인되지 않았습니다. 따라서 이 수치를 independent test, field generalization 또는 production-ready
근거로 표현하지 않습니다. Independent test는 새 실물과 새 촬영 session을 동결한 Ubuntu 현장 평가로
별도 수행합니다.

## 공통 평가 수치

| 수치 | 용도 | 선정 이유 |
|---|---|---|
| AP50-95 | 주 정확도 | COCO 표준 IoU 0.50~0.95 평균으로 위치 품질까지 반영 |
| AP50 / AP75 | 보조 정확도 | 느슨한 검출 성공과 엄격한 위치 정확도를 분리 |
| AP_small | COCO small 객체 | 원본 COCO annotation area `<32² px²`인 객체의 표준 AP 확인 |
| AR100 | 누락 경향 | 이미지당 최대 100개 예측에서 recall 확인 |
| P/R/F1, TP/FP/FN | 운영점 | 공통 score-sorted class-aware greedy 1:1 matcher, confidence 0.25, match IoU 0.50 |
| p50/p95 latency, FPS, VRAM | 배포성 | batch 1 실제 추론의 중앙값·tail·메모리 비교 |

AP/AR는 `pycocotools==2.0.11` COCOeval, 운영점 수치는 별도 공통 matcher에서 계산하며 두 경로 모두
이미지당 상위 100개 prediction을 사용합니다. `confidence=0.25`는 공통 보고 시작점이지 최종 배포값이
아닙니다. 수동 gold validation에서 목표 Recall
또는 best-F1 threshold를 선택하고 independent test 전에 동결합니다. 오토라벨 threshold도 같은 방식으로
class별 calibration하되 사람 검수를 생략하지 않습니다.

threshold는 validation에서만 후보를 산출하고 independent test 전에 동결합니다. test 결과에는
`NOT_FOR_THRESHOLD_SELECTION`을 명시하며 `best_f1_confidence`와
`autolabel_thresholds.csv`를 제공하지 않습니다. test 수치를 보고 운영·pseudo-label·autolabel
threshold를 선택하거나 수정하면 독립 test 주장을 철회하고 새 test set을 마련해야 합니다.

`AP_small`은 학습 전처리로 640에 resize된 뒤의 pixel 크기가 아니라 원본 COCO annotation area로
구분됩니다. 따라서 실제 소형 칩 판단에는 letterbox/resize 후 bbox width·height·area 분포와 pixel-size
bin별 recall 또는 AP를 함께 보고합니다.

## Git 승격 규칙

1. run 상태가 `complete`인지 확인합니다.
2. smoke run이 아닌지 확인합니다.
3. comparison protocol gate가 PASS이고 해당 run의 동일 manifest가 비교 source bundle에 있는지 확인합니다.
4. checkpoint SHA-256이 manifest와 같은지 확인합니다.
5. 검증된 formal allowlist 전체를 scan해 unlisted/stale file, raw image, report 내부 weight가 없는지 확인합니다.
6. `scripts/promote_run.py --comparison-dir ...`/`scripts/promote_comparison.py`로 `weights/trained/`, `reports/`에 복사합니다.
7. 비밀정보·절대 사용자 경로·대용량 파일을 재검사한 뒤 Git LFS로 push합니다.

새 native run release는 schema 4 `weight_payload_contract`를 사용합니다. native checkpoint 외에는
계약에 맞는 동일 stem ONNX/metadata pair만 한 단계 하위 bundle에 둘 수 있고, metadata가 native
artifact·checkpoint·comparison·ONNX hash를 정확히 결합해야 합니다. 선언되지 않은 파일은 확장자가
무엇이든 승격을 차단합니다. schema 3 공개본은 checkpoint 단독 inventory에 한해 읽기 호환됩니다.

승격은 재귀 복사가 아니라 `formal_validation.json`에서 재계산한 allowlist만 사용합니다. 따라서 source에
추가된 stale/raw/weight file이 있으면 승격 전에 중단합니다. 승격본은 로컬 project/user 경로와 raw
`nvidia-smi` process 목록을 제거합니다. `artifact_manifest.json`과
`sources_manifest.json`에는 로컬 원본 SHA-256과 공개용 사본 SHA-256을 모두 남기며 수치 값은 바꾸지
않습니다. 비교 보고서는 `sources/<run-id>/`에 필요한 log/CSV/JSON을 함께 묶어 다른 clone에서도
절대경로 없이 감사할 수 있게 합니다.

현재 저장소의 RPi formal 결과는 `paired_2seed_descriptive` tier입니다. 두 모델의 ONNX val/internal-test
deployment gate도 PASS했으나, 이 판정은 artifact·numeric equivalence·split/policy binding에 한정되며
Ubuntu 카메라 현장 검증을 대체하지 않습니다.
