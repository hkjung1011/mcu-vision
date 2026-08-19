# 모델 카드: `<MODEL_OR_RELEASE_ID>`

> 이 문서는 모델의 사용 범위와 검증 경계를 기록하는 공개 템플릿입니다. 확인되지 않은 항목은
> `NOT VERIFIED`로 유지하고, 측정되지 않은 수치를 추정하여 채우지 않습니다.

## 1. 모델 식별 정보

| 항목 | 값 |
|---|---|
| 모델 또는 release ID | `<MODEL_OR_RELEASE_ID>` |
| framework / version | `<FRAMEWORK_VERSION>` |
| checkpoint 파일 | `<CHECKPOINT_PATH>` |
| checkpoint SHA-256 | `<SHA256>` |
| 학습 protocol SHA-256 | `<PROTOCOL_SHA256>` |
| 데이터 증빙 SHA-256 | `<DATASET_EVIDENCE_SHA256>` |
| source commit | `<GIT_COMMIT>` |
| publication stage | `<INTERIM_PROGRESS_OR_FORMAL_RELEASE>` |

## 2. 의도된 용도와 금지된 용도

**의도된 용도**

- `<DETECTION_TASK_AND_TARGET_DOMAIN>`
- `<RESEARCH_OR_DEPLOYMENT_BOUNDARY>`

**금지된 해석·용도**

- 검증 세트 성능을 독립 현장 평가 결과로 해석하지 않습니다.
- 학습하지 않은 클래스, 촬영 환경 또는 part number로 성능을 확장하여 주장하지 않습니다.
- 외형만으로 전기적 정체성, 정확한 SKU 또는 안전 관련 판정을 확정하지 않습니다.
- `release_ready=false`인 체크포인트를 운영 배포용 모델로 표시하지 않습니다.

## 3. 클래스와 데이터

| canonical key | 표시명 | 데이터 근거 | 검증 상태 |
|---|---|---|---|
| `<CANONICAL_CLASS_KEY>` | `<DISPLAY_NAME>` | `<SOURCE_AND_SPLIT>` | `<STATUS>` |

- 클래스 정의 원본: `<CANONICAL_ONTOLOGY_PATH_AND_HASH>`
- 표시 계층: `<DISPLAY_SIDECAR_PATH>`
- train/validation/test: `<COUNTS_AND_SPLIT_UNIT>`
- 출처·변경 이력(provenance): `<SOURCE_LICENSE_SPECIMEN_SESSION_EVIDENCE>`
- 알려진 데이터 한계: `<LEAKAGE_DOMAIN_SHIFT_AND_COVERAGE_LIMITS>`

## 4. 입력·출력 및 전처리

| 항목 | 값 |
|---|---|
| 입력 크기 | `<WIDTH_X_HEIGHT>` |
| color / scale | `<COLOR_ORDER_AND_NORMALIZATION>` |
| resize | `<LETTERBOX_OR_OTHER>` |
| confidence / NMS IoU | `<VALUES_AND_SELECTION_SOURCE>` |
| 출력 | `<BOX_FORMAT_CLASS_SCORE>` |

confidence와 NMS 임계값이 검증 세트에서 선정되었다면 테스트 세트를 열기 전에 동결하고, 운영 환경에서
조정한 경우 그 변경을 별도 version으로 기록합니다.

## 5. 평가 결과

| 구분 | 데이터 | AP50-95 | AP50 | AP75 | AP_small | AR100 | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| validation | `<SPLIT>` | `<VALUE>` | `<VALUE>` | `<VALUE>` | `<VALUE>` | `<VALUE>` | `<EVALUATOR>` |
| test | `<SPLIT>` | `<VALUE_OR_NOT_VERIFIED>` |  |  |  |  | `<LOCKED_TEST_POLICY>` |
| 독립 현장 평가 | `<CAMERA_SESSION>` | `<VALUE_OR_NOT_VERIFIED>` |  |  |  |  | `<DOMAIN>` |

| 장치·정밀도 | batch | E2E p50 / p95 | FPS | VRAM/RAM | 측정 상태 |
|---|---:|---:|---:|---:|---|
| `<DEVICE_AND_PRECISION>` | `<BATCH>` | `<MS>` | `<FPS>` | `<MEMORY>` | `<STATUS>` |

모델 간 비교에는 동일한 데이터 hash, evaluator, operating point, 장치 조건을 사용합니다. 반복 수가
통계적 추론에 부족하면 mean ± sample SD를 기술통계로만 보고하고 우월성 또는 유의성을 주장하지
않습니다.

## 6. 라이선스와 보안

- 저장소 코드: `<LICENSE>`
- framework / checkpoint: `<UPSTREAM_LICENSE>`
- 학습 데이터: `<DATA_LICENSE_AND_ATTRIBUTION>`
- PyTorch `.pt/.pth`는 pickle 기반이므로 SHA-256과 출처가 확인된 파일만 로드합니다.

## 7. 알려진 실패 양상과 잔여 위험

- `<SMALL_OBJECT_DENSE_OVERLAP_MOTION_BLUR_FAILURES>`
- `<UNSEEN_CAMERA_LIGHTING_BACKGROUND_FAILURES>`
- `<CLASS_AMBIGUITY_OR_TOP_MARK_LIMITATIONS>`
- `<REMAINING_EXTERNAL_VALIDATION>`

## 8. 재현 및 검증

```text
학습 명령: <COMMAND_OR_MANIFEST_PATH>
평가 명령: <COMMAND_OR_REPORT_PATH>
검증 명령: <HASH_AND_LOAD_VERIFICATION>
```

| 요구사항 | 근거 | 판정 |
|---|---|---|
| checkpoint 무결성 | `<SHA256_EVIDENCE>` | `<PASS_OR_NOT_VERIFIED>` |
| validation 재현 | `<METRIC_ARTIFACT>` | `<PASS_OR_NOT_VERIFIED>` |
| ONNX 동등성 | `<ONNX_REPORT>` | `<PASS_OR_NOT_VERIFIED>` |
| 독립 카메라 평가 | `<FIELD_REPORT>` | `<PASS_OR_NOT_VERIFIED>` |
