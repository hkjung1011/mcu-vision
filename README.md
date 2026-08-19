# MCU Vision — YOLOX-S / YOLO11m 재현형 검출 실험

Windows에서 MCU·소형 전자부품 detector를 학습하고, 실행 로그·평가 수치·가중치의 무결성을 확인한 뒤
Ubuntu 카메라 환경으로 전달하기 위한 연구용 저장소입니다. 프로젝트 코드는 `AGPL-3.0`이며,
third-party model·dataset에는 해당 원 라이선스가 함께 적용됩니다.

> **현재 판정 (2026-08-19):** Raspberry Pi 1-class bootstrap은 이미 완료된 seed 42/43의
> YOLO11m·YOLOX-S 4개 100-epoch run을 `paired_2seed_descriptive` 정책으로 정식 검증했습니다.
> Formal comparison과 두 모델의 native→ONNX val/test 검증은 **PASS**입니다. 다만 `n=2`, `df=1`의
> 기술통계이며 내부 pHash split을 사용했으므로 통계적 유의성, 모집단 우월성, 독립 test 또는
> production-ready를 주장하지 않습니다. 추가 RPi 100-epoch 학습은 진행하지 않습니다.

## 진행상황 대시보드

| 영역 | 확인 결과 | 판정 |
|---|---|---|
| Windows 학습 환경 | RTX 5060 Laptop 8,151 MiB, PyTorch 2.12.1+cu130 | **PASS** |
| RPi data contract | train/val/test 1,500/195/180, condition 및 pHash component cross-split 0 | **PASS (internal bootstrap)** |
| RPi paired formal | YOLO11m·YOLOX-S × seed 42/43, 4개 run, 각 100 epochs | **PASS (descriptive-only)** |
| RPi ONNX deployment gate | 두 모델 val/test 공통 평가, native↔ONNX numeric equivalence, artifact hash | **PASS** |
| 독립성 | physical item·새 camera session 독립성 증빙 없음 | **NOT VERIFIED** |
| MCU/SMD ontology | canonical 6-class ID 동결, 한국어는 presentation-only sidecar | **CONTRACT PASS / DATA EMPTY** |
| 소형칩 반입·CVAT·오토라벨 경로 | provenance·hash·specimen·round-trip·승격 gate 구현 | **CODE PASS / DATA WAITING** |
| STM32 공개 원천 수동 seed | Wikimedia Commons `stm32_dev_board` 후보 11건, model-assisted draft box 11개, 보수적 leakage group 6개 | **PENDING HUMAN REVIEW / APPROVED 0** |
| STM32/SMD 승인 실사 | trusted registry에 승인된 실제 학습 dataset 없음 | **NOT VERIFIED** |
| Ubuntu 실제 카메라 시험 | ONNX artifact 준비, 현장 capture·latency·정확도 미측정 | **NOT VERIFIED** |

상세 판정은 [현재 프로젝트 상태](docs/project_status.md), 단계별 계획은
[전체 로드맵](docs/roadmap.md)을 참조하십시오.

## RPi formal 결과

정식 정책은 [`rpi_bootstrap_paired_2seed_release_v1.yaml`](configs/experiments/rpi_bootstrap_paired_2seed_release_v1.yaml)에
고정되어 있습니다. 목표 축소는 학습 후 결정되었으며 seed 44는 두 모델을 한 쌍으로 모두 제외했습니다.

| 비교 항목 | COCO AP50-95 mean ± sample SD |
|---|---:|
| YOLO11m, seeds 42/43 | `1.0000000000 ± 0.0000000000` |
| YOLOX-S, seeds 42/43 | `0.9846059844 ± 0.0074003961` |
| Paired delta (YOLO11m − YOLOX-S) | `0.0153940156 ± 0.0074003961` |

허용되는 해석은 per-run metric, mean, sample SD, paired seed delta까지입니다. 표본 수가 두 쌍뿐이므로
통계적 유의성 또는 일반 모집단에서의 architecture 우월성을 결론내리지 않습니다. 또한 이 결과는
Raspberry Pi 보드 검출 bootstrap이며 STM32/SMD 성능을 나타내지 않습니다.

ONNX 공통 평가도 두 모델 모두 PASS했습니다.

| 모델 | 선택 run | ONNX val AP50-95 | ONNX internal test AP50-95 | CPU p50 val/test |
|---|---|---:|---:|---:|
| YOLO11m | seed 43 | `1.0000000000` | `1.0000000000` | `215.698 / 221.226 ms` |
| YOLOX-S | seed 43 | `0.9908195841` | `0.9852172241` | `78.778 / 78.323 ms` |

여기서 test는 새 실물·새 촬영 session으로 구성한 independent test가 아니라 기존 데이터의
**locked internal pHash split**입니다. ONNX PASS는 export artifact, evaluator, split binding 및
native↔ONNX 수치 동등성의 통과를 의미하며 Ubuntu 카메라 실시간 성능 완료를 의미하지 않습니다.
CPU latency는 해당 재실행 hardware·runtime에 종속된 측정값이며 다른 Ubuntu 장치의 예측치가 아닙니다.
Internal test는 threshold 또는 model 선택에 사용하지 않습니다.

## 이후 학습 정책

역사적 RPi 100-epoch protocol과 완료 증빙은 변경하지 않습니다. 새 RPi 100-epoch run은 추가하지 않고,
STM32/SMD는 [`smallchip_staged_training_v1.yaml`](configs/experiments/smallchip_staged_training_v1.yaml)에
따라 단계적으로 진행합니다.

| 단계 | epoch 상한 | early stopping | 승격 목적 |
|---|---:|---|---|
| Smoke | `1` | 해당 없음 | data/class/framework 계약과 유한 loss 확인 |
| Pilot | `10` | `patience=3` | 불안정하거나 가치가 낮은 recipe 조기 제외 |
| Candidate | `50` | `patience=10` | 사전 정의 metric·운영점 기반 후보 평가 |

50 epochs를 넘기는 연장은 자동으로 수행하지 않습니다. validation curve 미수렴, overfitting 점검,
compute/time budget, leakage gate 및 사용자 승인 근거를 기록한 새 versioned contract가 있어야 합니다.

## 데이터·클래스 계약

- 추론에 사용하는 canonical class ID와 영문 key:
  [`classes.smd_v1.yaml`](configs/classes.smd_v1.yaml)
- 문서·UI 표시용 한국어 명칭:
  [`classes.smd_v1.display.ko.yaml`](configs/classes.smd_v1.display.ko.yaml)
- RPi immutable base protocol:
  [`baseline_v1.yaml`](configs/experiments/baseline_v1.yaml)
- 기존 protocol을 바꾸지 않고 test sidecar만 결합하는 계약:
  [`rpi_test_evidence_supplement_v1.yaml`](configs/experiments/rpi_test_evidence_supplement_v1.yaml)
- STM32/SMD 공개 원천의 채택·제외 근거:
  [`smallchip_source_review.md`](docs/smallchip_source_review.md)

한국어 sidecar는 presentation-only입니다. canonical ID, class 순서, source alias 또는 학습 label을
재정의하지 않으며 integration test가 두 파일의 ontology ID와 exact key order를 검증합니다.

## STM32 Commons 수동 seed 검수 상태

Wikimedia Commons에서 revision과 권리 근거를 고정한 `stm32_dev_board` 후보 11건을 하나의
**검수 전용 bundle**로 준비했습니다. 이 bundle에는 후보별 model-assisted draft box 11개와
보수적으로 정의한 leakage group 6개가 포함됩니다. 현재 사람 승인 수는 **0건**이며, draft box는
ground truth가 아닙니다. 로컬 bundle 경로는
`data/staging/manual_seed/wikimedia_commons_stm32_dev_board_v2_review2/`입니다. 원본 이미지와 검수
작업물은 Git 추적 제외(Gitignored) 데이터이므로 GitHub에 게시하거나 학습 입력으로 사용하지 않습니다.
GitHub에는 원본 대신 [검수 준비 기록](data/manifests/wikimedia_stm32_dev_board.manual-seed-review-preparation.json)만
공개하며, 여기에도 사람 승인 0건과 학습 금지 상태를 명시했습니다.

동일한 계약으로 새 검수 작업을 만들 때는 기존 output directory를 덮어쓰지 않고 새 `run-id`와 경로를
지정합니다.

```powershell
.\.venv-collect\Scripts\mcu-prepare-manual-seed.exe `
  --output-dir data\staging\manual_seed\<new-review-task> `
  --run-id <new-review-run-id>
```

다음 단계는 CVAT에서 11개 이미지의 **`stm32_dev_board` identity와 bbox를 사람이 전수 확인·수정**하는
것입니다. 검수 완료, round-trip 검증 및 hash-bound 승인 절차 전까지 `training_use_allowed=false`를
유지합니다.

## 처음 실행

별도 CUDA Toolkit은 현재 PyTorch 학습에 필수적이지 않습니다. 검증 환경은 NVIDIA driver와 PyTorch CUDA
wheel을 사용합니다. `nvcc` custom extension 또는 특정 TensorRT toolchain이 필요할 때만 별도 설치를
검토합니다.

```powershell
git lfs install
git lfs pull
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_collection.ps1
.\scripts\setup_yolo11.ps1
.\scripts\setup_yolox.ps1
.\.venv-collect\Scripts\python.exe -m pytest
```

RPi paired 결과 재검증은 [paired two-seed formal release 절차](docs/paired_2seed_formal_release.md)를,
Windows 환경 재현은 [Windows 재현 절차](docs/reproducibility_windows.md)를 따릅니다. 과거의 3-seed
100-epoch 명령을 새 학습 계획으로 사용하지 마십시오.

## 증빙 원칙

판정 원본은 `terminal.log`, CSV, JSON, resolved config와 checkpoint SHA-256입니다. 보고서 PNG는 이
수치로 `matplotlib`가 생성한 **비생성형 파생물**이며 AI가 지어낸 성능 이미지가 아닙니다.
`terminal_summary.png`도 실제 Terminal 사진이 아니라 text log를 결정론적으로 렌더링한 자료입니다.

| 궁금한 내용 | 문서 |
|---|---|
| 상태·잔여 위험 | [프로젝트 상태](docs/project_status.md) · [전체 로드맵](docs/roadmap.md) |
| 수치·알고리즘 선정 근거 | [Parameter rationale](reports/methodology/parameter_rationale.md) · [실험 방법론](reports/methodology/experiment_methodology.md) |
| 데이터·라벨·오토라벨 | [데이터 계획](docs/data_plan.md) · [공개 원천 검토](docs/smallchip_source_review.md) · [라벨링 규정](docs/annotation_protocol.md) · [Small-chip ingest](docs/smallchip_ingest_p0.md) |
| 비교·증빙·배포 | [증빙 정책](docs/evidence_and_results_policy.md) · [Reports](reports/README.md) · [Weights](weights/README.md) |
| Ubuntu 현장 시험 | [Ubuntu 인계](docs/ubuntu_handoff.md) |

원본 이미지·미승인 라벨·전체 `runs/`와 TensorRT `.engine`은 Git에 포함하지 않습니다. `.pt`, `.pth`,
`.onnx`는 Git LFS로 관리하며, 공개 전 metadata와 절대 경로를 검사합니다.
