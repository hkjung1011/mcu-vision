# Reports

GitHub에는 검증 가능한 숫자 원본과 그 파생 보고서만 보존합니다.

| 경로 | 내용 |
|---|---|
| `methodology/` | baseline config에서 자동 생성한 수치 선정 근거와 참고문헌 |
| `runs/<release>/` | 완료된 단일 학습 run의 manifest, log, CSV/JSON, 그래프 |
| `comparisons/<release>/` | 동일 protocol을 통과한 다중 seed 모델 비교 |

현재는 `methodology/`만 존재합니다. Smoke test나 protocol mismatch 결과는 정식 성능 결과로 올리지
않습니다.

PNG는 생성형 AI가 만든 이미지가 아니라 YAML/CSV/JSON을 `matplotlib`로 렌더링한 파생물입니다.
판단은 숫자 원본으로 하며, renderer와 SHA-256은 각 artifact manifest에서 확인합니다.
