# 모델 카드: RPi YOLO11m paired-2seed 공개본

> 이 카드는 `rpi_phash_v2_paired2_yolo11m` 공개본의 용도, 수치 근거와 검증 한계를 기록합니다.
> 이 모델은 **Raspberry Pi SBC 1-class 연구용 bootstrap**이며 MCU·STM32·SMD 검출 모델이 아닙니다.

## 1. 식별 정보

| 항목 | 값 |
|---|---|
| 공개본 ID | `rpi_phash_v2_paired2_yolo11m` |
| 선택 실행 | `yolo11m_seed43` |
| 프레임워크 | Ultralytics `8.4.120` |
| 학습 방식 | COCO 사전학습 YOLO11m 전체 detector fine-tuning |
| 학습 source commit | `63bc679d88d56907433b6586e2a7efacb46a25d6` |
| Native checkpoint | [`best.pt`](../../weights/trained/rpi_phash_v2_paired2_yolo11m/best.pt) |
| Native SHA-256 | `54cafd9348bde613945218fb0696e703de76ddef6d09210ab30d092bd1e3f2d4` |
| ONNX | [`yolo11m_rpi_b1_640.onnx`](../../weights/trained/rpi_phash_v2_paired2_yolo11m/yolo11m/yolo11m_rpi_b1_640.onnx) |
| ONNX SHA-256 | `ad488f6758af0cb7cfe1937ae128411ab73feeec5f3bdc8377f07ffaebc7ebfc` |
| 공개 단계 | `FORMAL PASS — paired_2seed_descriptive` |

정식 비교 정책은 [`rpi_bootstrap_paired_2seed_release_v1.yaml`](../../configs/experiments/rpi_bootstrap_paired_2seed_release_v1.yaml),
base protocol은 [`baseline_v1.yaml`](../../configs/experiments/baseline_v1.yaml)입니다. 각각의 공개 증빙 SHA-256은
`1865539e9b3569dd4942d9d17495a3644e059df70259b555bd7985e7bdf76f27`과
`02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5`입니다.

## 2. 모델 설명과 의도된 용도

YOLO11m detection checkpoint를 640×640 입력, batch 8, AMP 조건에서 100 epochs fine-tuning한
one-stage object detector입니다. 20,114,688개 parameter를 사용하며, 출력은 bbox와
`raspberry_pi_sbc` 클래스 score입니다.

허용 용도는 다음으로 제한합니다.

- 동일 데이터 계약에서 학습·평가·ONNX export 파이프라인을 재현하는 연구
- Raspberry Pi 보드가 비교적 크게 보이는 내부 이미지에서의 검출 실험
- 동일한 paired seed 42/43 조건에서 YOLOX-S와 기술통계 수준으로 비교

금지된 해석과 용도는 다음과 같습니다.

- STM32, IC package, SMD 부품 또는 학습하지 않은 Raspberry Pi 세부 모델의 검출 성능으로 해석
- 안전·품질·재고 판정, exact SKU 식별 또는 무인 운영 배포
- 내부 test 수치를 독립 현장 성능이나 모집단 성능으로 해석
- `n=2`, `df=1` 결과로 통계적 유의성 또는 architecture 우월성을 주장

## 3. 클래스와 데이터

| 항목 | 값 | 검증 상태 |
|---|---|---|
| canonical class | `raspberry_pi_sbc` | PASS |
| 이미지 수 | train 1,500 / validation 195 / internal test 180 | PASS |
| 분할 단위 | condition group + pHash connected component | PASS |
| cross-split pHash 후보 | 0 | PASS |
| physical specimen 독립성 | 원출처에 specimen ID 없음 | **NOT VERIFIED** |
| 데이터 출처 | micro-PCB Images, CC BY 4.0 | 출처 기록 PASS |

Canonical dataset evidence SHA-256은
`bea2fbaddc87f29181596230404c49753926eccd580b0df7aff848a15d8c5617`입니다.
분할은 condition/pHash 누수를 막지만 같은 실제 개체의 독립성을 입증하지 못합니다. 따라서 test는
**locked internal pHash split**으로만 부릅니다.

## 4. 입력·출력 계약

| 항목 | 값 |
|---|---|
| 입력 | batch 1, FP32, `1×3×640×640`; `OpenCV BGR -> RGB` |
| resize | aspect ratio 유지, centered letterbox |
| padding | 114 |
| normalization | `divide by 255.0` |
| 출력 tensor | `[1, 5, 8400]`; decoded xywh + class probability |
| confidence | 0.25 |
| NMS IoU | 0.65, class-aware |
| max detections | 100 |

ONNX graph에는 NMS가 포함되지 않습니다. 공개 metadata와 동일한 전처리·후처리를 적용해야 합니다.

## 5. 평가 결과

| 평가 | AP50-95 | AP50 | AP75 | AR100 | 범위 |
|---|---:|---:|---:|---:|---|
| 선택 native 실행, common COCO validation | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 195 images |
| ONNX formal validation | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 195 images |
| ONNX locked internal test | 1.0000000000 | 1.0000000000 | 1.0000000000 | 1.0000000000 | 180 images |
| 독립 카메라 평가 | **NOT VERIFIED** |  |  |  | 새 촬영 session 없음 |

Paired seed 42/43의 common COCO AP50-95는 `1.0000000000 ± 0.0000000000`입니다.
이는 기술통계일 뿐입니다. 데이터에 small/medium COCO area instance가 없어 `AP_small`과
`AP_medium`은 산출할 수 없습니다.

| Runtime | split | inference p50 / p95 | 상태 |
|---|---|---:|---|
| ONNX Runtime 1.28.0, CPUExecutionProvider, batch 1 | validation | 215.698 / 228.287 ms | 측정 PASS |
| ONNX Runtime 1.28.0, CPUExecutionProvider, batch 1 | internal test | 221.226 / 226.393 ms | 측정 PASS |
| Ubuntu 카메라 | 독립 현장 | **NOT VERIFIED** | 미측정 |

CPU 수치는 해당 Windows hardware/runtime의 측정값이며 다른 장치의 latency나 FPS를 예측하지 않습니다.

## 6. 공개·무결성 검증

- Native metadata의 사용자 경로를 제거한 뒤 state dict bitwise equality와 zero-input forward
  `max_abs_difference=0.0`을 확인했습니다.
- Native↔ONNX raw output 비교는 `max_absolute_error=0.000274658203125`, 허용 비율 100%로 PASS했습니다.
- Formal validation과 internal test는 동일 ONNX SHA, protocol, split evidence에 결속되어 있습니다.
- Test split은 `NOT_FOR_THRESHOLD_SELECTION`이며 test에서 pseudo-label threshold를 선택하지 않습니다.
- Native, ONNX와 manifest는 Git LFS·SHA-256·공개 경로 privacy gate를 통과했습니다.

근거: [native artifact manifest](../../reports/runs/rpi_phash_v2_paired2_yolo11m/artifact_manifest.json),
[deployment release manifest](../../reports/deployments/rpi_phash_v2_paired2_yolo11m/deployment_release_manifest.json),
[formal comparison](../../reports/comparisons/rpi_phash_v2_paired2_comparison/experiment_report.md).

## 7. 라이선스와 잔여 위험

- 프로젝트 코드: AGPL-3.0
- Ultralytics checkpoint와 프레임워크: AGPL-3.0 또는 별도 유효한 Enterprise 조건을 준수해야 합니다.
- 학습 데이터: CC BY 4.0; 원저작자 표시와 변경 고지를 유지해야 합니다.
- `.pt`는 pickle 기반이므로 manifest의 SHA-256과 일치하는 신뢰된 파일만 로드합니다.

잔여 위험은 촬영 배경·조명·거리·motion blur의 domain shift, 작은 보드·가림·중첩, Raspberry Pi가
아닌 유사 PCB의 false positive, 실제 specimen 중복 가능성입니다. 공개본은 production-ready 또는
독립 현장 검증 완료로 표시할 수 없습니다.
