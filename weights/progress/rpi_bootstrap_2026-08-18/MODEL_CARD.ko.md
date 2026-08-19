# 모델 카드: Raspberry Pi SBC 1-class 중간 공개본 (2026-08-18)

> **INTERIM_PROGRESS · `release_ready=false` · 독립 현장 평가 미수행**
>
> 이 모델 묶음은 Raspberry Pi SBC 1-class 객체 검출 파이프라인의 재현·검토용 중간 산출물입니다.
> STM32/SMD 객체 검출, top-mark OCR, 정확한 part number 인식 또는 운영 배포용 모델이 아닙니다.

## 1. 식별 정보와 무결성

| 항목 | 값 |
|---|---|
| snapshot ID | `rpi_bootstrap_2026-08-18` |
| task | `one_class_raspberry_pi_sbc_detection` |
| publication stage | `INTERIM_PROGRESS` |
| source commit | `63bc679d88d56907433b6586e2a7efacb46a25d6` |
| protocol SHA-256 | `02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5` |
| dataset evidence SHA-256 | `bea2fbaddc87f29181596230404c49753926eccd580b0df7aff848a15d8c5617` |
| formal release | `false` |
| independent camera tested | `false` |

| 체크포인트 | 상태·역할 | 공개 SHA-256 |
|---|---|---|
| `yolo11m_seed42_best.pt` | COMPLETE, 추론 검토용 best | `f48e6e662190f0fd650ca55767cc4e32b18803a9ce3ebec240e51e2f54df5a28` |
| `yolo11m_seed43_best.pt` | COMPLETE, 추론 검토용 best | `901b8609ab93b4ed727912bf32f490acabbb397e20bef3ffd30e01b2f8de34ab` |
| `yolox_s_seed42_best.pth` | COMPLETE, 추론 검토용 best | `345ae7b7d54f49a447d3c6ab5c57d202c5593451a41678440c2ea72e94fb0538` |
| `yolox_s_seed43_best.pth` | INTERRUPTED, epoch 34 best | `ca85c3eb17cbf0619f2345f0547ae67fc7c34f1a81d15db365fd6dff49f31c3b` |
| `yolox_s_seed43_resume_epoch_70.pth` | INTERRUPTED, resume candidate | `9783f59224fba8c6bc880ef7cfd553235638686c83e970ad7a6c20d9cdb85dfd` |

위 값은 [`progress_manifest.json`](../../../reports/progress/rpi_bootstrap_2026-08-18/progress_manifest.json)의
공개 증빙과 일치해야 합니다.

## 2. 의도된 용도와 금지된 용도

**의도된 용도**

- 동일 protocol에서의 YOLO11m/YOLOX-S 학습·로그 수집·공통 COCO 평가 경로 재현
- 연구 개발 단계의 Raspberry Pi SBC 검출 추론과 Ubuntu 인계 절차 검토
- 정식 release 전 checkpoint load, SHA-256, 전처리 및 평가 코드 검증

**금지된 해석·용도**

- STM32, SMD, Raspberry Pi Pico 또는 세부 보드 모델을 구분하는 모델로 사용
- 검증 세트 수치를 실제 컨베이어·카메라 환경의 최종 정확도로 해석
- 안전·품질 보증·재고 판정 등 오검출이 위해나 손실로 이어지는 자동 의사결정에 사용
- `release_ready=false`인 체크포인트를 운영 배포용 모델 또는 모델 우열의 최종 근거로 게시

## 3. 클래스와 데이터 경계

| canonical key | 표시명 | 데이터 | 판정 |
|---|---|---|---|
| `raspberry_pi_sbc` | Raspberry Pi SBC | micro-PCB Images의 Pi 1 B+, Pi 3 B+, Pi A+ | 1-class pipeline 범위에서만 CONFIRMED |

- 출처: Adam Byerly, micro-PCB Images, `CC BY 4.0`
- 이미지 수: train 1,500 / validation 195 / test 180
- 분할 검증: condition 및 pHash component cross-split 후보 0
- 잔여 한계: 원출처에 physical specimen ID가 없어 동일 개체 독립성을 입증하지 못했습니다.
- 공개 snapshot의 수치는 validation 195장에 한정되며, test 180장과 독립 카메라 데이터의 정식 평가는
  아직 공개되지 않았습니다.

## 4. 입력·출력 및 operating point

| 항목 | 값 | 주의사항 |
|---|---|---|
| 입력 크기 | `640×640` | 원본 종횡비를 유지하는 letterbox 전처리 |
| color / scale | framework별 pinned adapter 적용 | YOLO11과 YOLOX의 입력 순서를 임의로 교환하지 않음 |
| 출력 | bounding box, `raspberry_pi_sbc`, confidence | exact board model 분류 결과가 아님 |
| 보고 confidence | `0.25` | 검증 세트의 공통 보고 operating point, 배포 임계값 아님 |
| NMS IoU | `0.65` | 공통 protocol 값 |
| matching IoU | `0.50` | precision/recall/F1 산출용, NMS와 용도가 다름 |
| max detections | `100` per image | 공통 evaluator 기준 |

카메라 연동 시 학습 코드와 동일한 letterbox, color order, scale, 좌표 역변환, class-aware NMS를 적용해야
합니다. 임계값은 locked validation에서 선정한 뒤 독립 test를 보기 전에 동결해야 합니다.

## 5. 공개 snapshot의 평가 결과

| Run | 평가 데이터·방법 | AP50-95 | AP50 | AP75 | AP_small | AR100 |
|---|---|---:|---:|---:|---:|---:|
| YOLO11m seed42 | validation, common COCO | 1.000000 | 1.000000 | 1.000000 | 산출 불가¹ | 1.000000 |
| YOLO11m seed43 | validation, common COCO | 1.000000 | 1.000000 | 1.000000 | 산출 불가¹ | 1.000000 |
| YOLOX-S seed42 | validation, common COCO | 0.979373 | 1.000000 | 1.000000 | 산출 불가¹ | 0.984615 |
| YOLOX-S seed43 | validation, native epoch 34 best | 0.857410 | 1.000000 | 미보고 | 산출 불가¹ | 미보고 |

¹ validation annotation 195개가 모두 COCO `large` 영역에 해당하여 `AP_small`의 ground truth가 0개입니다.
이는 소형 객체 성능이 높다는 뜻이 아니라 해당 지표를 평가할 수 없다는 뜻입니다.

중단 실행의 native metric은 완료 실행의 common COCO 최종 metric과 직접 비교하지 않습니다. 완료된 반복
matrix가 아니므로 mean ± sample SD에 기반한 정식 우열 결론도 제시하지 않습니다.

| Run | 장치·정밀도 | batch | E2E p50 / p95 | FPS | 측정 범위 |
|---|---|---:|---:|---:|---|
| YOLO11m seed42 | RTX 5060 Laptop, FP16 | 1 | 15.68 / 16.96 ms | 64.13 | validation inference 경로 |
| YOLO11m seed43 | RTX 5060 Laptop, FP16 | 1 | 11.52 / 12.46 ms | 84.92 | validation inference 경로 |
| YOLOX-S seed42 | RTX 5060 Laptop, FP16 | 1 | 17.76 / 20.05 ms | 56.43 | validation inference 경로 |

노트북의 열·전력·background load를 통제한 반복 측정이 아니므로 위 지연 시간은 장치의 보장 성능이
아닙니다. 카메라 capture와 후처리를 포함한 Ubuntu end-to-end 성능도 아직 측정하지 않았습니다.

## 6. 라이선스와 checkpoint 보안

- 저장소 코드: `AGPL-3.0`
- YOLO11/Ultralytics checkpoint와 사용 코드: 기본 `AGPL-3.0` 또는 별도 Ultralytics 계약 조건
- YOLOX checkpoint와 코드: `Apache-2.0`
- 학습 데이터: Adam Byerly, micro-PCB Images, `CC BY 4.0`
- PyTorch `.pt/.pth`는 pickle 기반입니다. 이 문서 또는 manifest의 SHA-256과 출처를 확인한 파일만
  신뢰 환경에서 로드하십시오.

## 7. 알려진 실패 양상과 잔여 위험

- STM32/SMD처럼 학습 범위 밖인 대상에는 검출 성능을 주장할 수 없습니다.
- 작은 부품, 고밀도 겹침, motion blur, 반사, 부분 가림에 대한 검증 자료가 없습니다.
- 배경·조명·렌즈·카메라 거리가 바뀌면 micro-PCB 데이터의 높은 validation AP가 재현되지 않을 수 있습니다.
- physical specimen 독립성, test split, ONNX 동등성, Ubuntu 독립 현장 평가가 남아 있습니다.
- `yolox_s_seed43_resume_epoch_70.pth`의 공개본 직접 resume 절차는 end-to-end로 재검증하지 않았습니다.

## 8. 재현 및 검증

```powershell
git lfs pull
.\.venv-collect\Scripts\python.exe .\scripts\verify_progress_snapshot.py `
  --report-dir reports\progress\rpi_bootstrap_2026-08-18
```

| 요구사항 | 공개 근거 | 판정 |
|---|---|---|
| checkpoint SHA-256·load | `progress_manifest.json`과 공개 파일 | 중간 공개 범위 PASS |
| validation metric 재현 | run별 CSV/JSON·비식별 로그 | 완료 실행 3개 PASS |
| 정식 반복 비교 | 6-run baseline matrix | NOT VERIFIED |
| ONNX val/test 동등성 | formal deployment report | NOT VERIFIED |
| Ubuntu 독립 카메라 평가 | 새 session report | NOT VERIFIED |
