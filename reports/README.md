# 실험 보고서와 검증 증빙

이 디렉터리에는 수치 원본과 그 원본에서 결정론적으로 생성한 보고서만 보존합니다. 성능 판정은 JSON,
CSV, terminal log, resolved config와 artifact SHA-256을 기준으로 하며 PNG는 검토 편의를 위한
비생성형 파생 시각화입니다.

## 현재 정식 결과

| 공개 경로 | 상태 | 내용 |
|---|---|---|
| `comparisons/rpi_phash_v2_paired2_comparison/` | **FORMAL PASS** | exact 4-run paired descriptive comparison, policy/base/attestation binding |
| `runs/rpi_phash_v2_paired2_yolo11m/` | **PASS** | 선택 YOLO11m run의 비식별 실행·평가 증빙 |
| `runs/rpi_phash_v2_paired2_yolox_s/` | **PASS** | 선택 YOLOX-S run의 비식별 실행·평가 증빙 |
| `deployments/rpi_phash_v2_paired2_yolo11m/` | **PASS** | native→ONNX val/test 및 artifact publication gate |
| `deployments/rpi_phash_v2_paired2_yolox_s/` | **PASS** | native→ONNX val/test 및 artifact publication gate |
| `progress/rpi_bootstrap_2026-08-18/` | HISTORICAL | 6-run 계획 당시의 중간 snapshot; 정식 paired 결과와 구분 |

Formal policy ID는 `rpi_bootstrap_paired_2seed_release_v1`, policy SHA-256은
`1865539e9b3569dd4942d9d17495a3644e059df70259b555bd7985e7bdf76f27`, base protocol canonical
SHA-256은 `02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5`입니다.

## 기술통계

| 항목 | AP50-95 mean ± sample SD |
|---|---:|
| YOLO11m, seeds 42/43 | `1.0000000000 ± 0.0000000000` |
| YOLOX-S, seeds 42/43 | `0.9846059844 ± 0.0074003961` |
| Paired delta (YOLO11m − YOLOX-S) | `0.0153940156 ± 0.0074003961` |

이는 `n=2`, `df=1`의 기술통계입니다. Per-run metric, model mean, sample SD와 paired seed delta만
허용하며 statistical significance, population superiority, production-ready 및 independent-test 주장은
금지합니다. 목표 축소는 사후 결정되었고 seed 44는 두 모델 모두 formal 입력에서 제외했습니다.

## ONNX 평가

| 모델 | ONNX val AP50-95 | ONNX internal test AP50-95 | CPU p50 val/test | Gate |
|---|---:|---:|---:|---|
| YOLO11m | `1.0000000000` | `1.0000000000` | `215.698 / 221.226 ms` | **PASS** |
| YOLOX-S | `0.9908195841` | `0.9852172241` | `78.778 / 78.323 ms` | **PASS** |

ONNX deployment PASS는 native/ONNX numeric equivalence, formal split, hash, policy와 공개 artifact
검사를 통과했다는 뜻입니다. Test split은 독립 수집 표본이 아니라 내부 pHash grouping으로 만든 locked
split이므로 새 실물·새 카메라 일반화를 입증하지 않습니다. 실제 Ubuntu capture pipeline의 accuracy,
latency와 FPS는 별도 현장 test에서 측정해야 합니다.
위 CPU p50은 schema 4 bundle을 재검증한 특정 hardware·runtime의 측정값이므로 다른 장치로 일반화하지
않습니다. Internal test는 `NOT_FOR_THRESHOLD_SELECTION`이며 test split에서 threshold를 선택하거나
autolabel CSV를 생성하지 않습니다.

## 디렉터리 규칙

| 경로 | 내용·공개 조건 |
|---|---|
| `methodology/` | 수치 선정 근거, 알고리즘, 참고문헌 |
| `progress/<snapshot>/` | 미완료 campaign의 역사적 snapshot; formal 결과 아님 |
| `runs/<release>/` | 완료 실행의 manifest, 비식별 log, CSV/JSON과 그래프 |
| `comparisons/<release>/` | 동일 policy·protocol·dataset hash를 통과한 비교 |
| `deployments/<release>/` | native checkpoint·ONNX·val/test를 hash로 결합한 배포 증빙 |

현재 methodology의 학습 task는 Raspberry Pi 1-class bootstrap입니다. SMD/STM32 6-class ontology의
학습 결과로 해석하지 마십시오. 새 STM32/SMD 결과는 별도 versioned data/training contract와 staged
1e→10e→50e gate를 통과한 뒤에만 추가합니다.

PNG는 ImageGen이나 diffusion model로 만든 자료가 아닙니다. YAML/CSV/JSON/비식별 log를
`matplotlib`로 렌더링하며, source file·SHA-256·renderer와 `GENERATIVE_AI=false`를 artifact manifest로
검증합니다.
