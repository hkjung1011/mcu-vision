# Reports

GitHub에는 검증 가능한 숫자 원본과 그 파생 보고서만 보존합니다.

| 경로 | 내용 |
|---|---|
| `methodology/` | baseline config에서 자동 생성한 수치 선정 근거와 참고문헌 |
| `runs/<release>/` | 완료된 단일 학습 run의 manifest, log, CSV/JSON, 그래프 |
| `comparisons/<release>/` | 동일 protocol을 통과한 다중 seed 모델 비교 |

현재는 `methodology/`만 존재합니다. Smoke test나 protocol mismatch 결과는 정식 성능 결과로 올리지
않습니다.

현재 methodology의 task는 `one_class_raspberry_pi_sbc_detection`입니다. MCU/SMD 5-class protocol이나
trained 결과로 해석하지 않습니다.

- [`methodology/parameter_rationale.md`](methodology/parameter_rationale.md): 12개 핵심 수치의 선정 이유,
  최적값 여부, 재조정 조건, 현재 검증 상태
- [`methodology/experiment_methodology.md`](methodology/experiment_methodology.md): 학습 알고리즘,
  전체 protocol, 논문·공식 source

PNG는 생성형 AI가 만든 이미지가 아니라 YAML/CSV/JSON을 `matplotlib`로 렌더링한 파생물입니다.
판단은 숫자 원본으로 하며, renderer와 SHA-256은 각 artifact manifest에서 확인합니다.
