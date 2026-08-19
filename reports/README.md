# 실험 보고서와 검증 증빙

이 디렉터리에는 수치 원본과 그 원본에서 결정론적으로 생성한 보고서만 보존합니다.

| 경로 | 내용·공개 조건 |
|---|---|
| `methodology/` | baseline 설정 파일에서 생성한 수치 선정 근거와 참고문헌 |
| `progress/<snapshot>/` | 미완료 campaign의 비식별 로그·수치·그래프·checkpoint hash, 정식 공개본 아님 |
| `runs/<release>/` | 완료된 단일 학습 실행의 manifest, 로그, CSV/JSON, 그래프 |
| `comparisons/<release>/` | 동일 protocol과 데이터 hash를 통과한 반복 실행 비교 |
| `deployments/<release>/` | native checkpoint·ONNX·formal validation/test를 hash로 결합한 배포 증빙 |

현재 [`progress/rpi_bootstrap_2026-08-18/`](progress/rpi_bootstrap_2026-08-18/README.md)에는 완료된
전체 학습 실행 3개와 중단 실행 1개의 실제 기록이 있습니다. 공개 당시 계획한 6-run matrix가
완료되지 않았으므로 `runs/<release>`의 정식 결과로 승격하지 않았습니다. Smoke test 또는 protocol이
일치하지 않는 실행도 정식 성능 근거로 사용하지 않습니다.

현재 methodology의 task는 `one_class_raspberry_pi_sbc_detection`입니다. SMD/STM32 6-class ontology의
학습 protocol이나 해당 클래스의 성능 결과로 해석해서는 안 됩니다.

- [`methodology/parameter_rationale.md`](methodology/parameter_rationale.md): 14개 핵심 수치의 선정 근거,
  최적값 여부, 재조정 조건, 현재 검증 상태
- [`methodology/experiment_methodology.md`](methodology/experiment_methodology.md): 학습 알고리즘,
  전체 protocol, 논문 및 공식 출처

PNG는 생성형 AI 이미지가 아니라 YAML/CSV/JSON/비식별 로그를 `matplotlib`로 렌더링한 비생성형
파생물입니다. 판정은 수치 원본을 기준으로 하며 renderer와 SHA-256은 각 artifact manifest에서
확인할 수 있습니다.
