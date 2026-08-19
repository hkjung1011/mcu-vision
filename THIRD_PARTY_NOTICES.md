# Third-party notices

확인일: **2026-08-18**

이 공개 저장소의 project-level license는 [`AGPL-3.0`](LICENSE)입니다. 이는 Ultralytics YOLO 코드와
fine-tuned model을 공개하는 현재 사용 방식에 맞춘 것입니다. 별도의 유효한 Ultralytics Enterprise
계약이 있다면 해당 계약 조건이 우선할 수 있습니다. Project-level license는 third-party software,
pretrained/fine-tuned weights, dataset의 원 라이선스 조건을 없애거나 변경하지 않습니다.

| 구성요소 | 사용/예정 범위 | 라이선스·근거 | 주의 |
|---|---|---|---|
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 학습 source와 pretrained/fine-tuned YOLOX-S | Apache-2.0; [license 사본](LICENSES/Apache-2.0.txt) | 고정 commit과 upstream weight hash 보존 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO11 학습·현재 autolabel backend | AGPL-3.0 또는 Enterprise | private/상용/embedded 이용은 적용 조건 별도 검토 |
| [PyTorch](https://github.com/pytorch/pytorch) | CUDA 학습 runtime | upstream license 참조 | binary wheel의 bundled component notice 포함 |
| [ONNX](https://github.com/onnx/onnx) | portable model graph와 구조 검증 | Apache-2.0 | fixed batch-1 FP32 graph로 export |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | export 수치 검증과 Ubuntu camera inference | MIT | 목표 장치 provider/CUDA 호환성 별도 검증 |
| [OpenCV](https://github.com/opencv/opencv-python) | COCO image 전처리와 camera capture/UI | Apache-2.0 | RGB/BGR·letterbox 규약을 release metadata에 고정 |
| [pycocotools 2.0.11](https://pypi.org/project/pycocotools/2.0.11/) | 공통 COCO AP/AR | package metadata/license 참조 | exact version 고정 |
| [CVAT Community](https://github.com/cvat-ai/cvat) | 권장 검수 UI와 export verifier | MIT | 서버 자동화 없음; 실제 export round-trip 대기 |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | 초기 bbox 제안 후보, 미구현 | Apache-2.0 | 현재 환경/weight 미포함 |
| [SAM 2](https://github.com/facebookresearch/sam2) | 선택적 mask 보조 후보, 미구현 | upstream license 참조 | semantic class detector가 아님 |
| [SAHI](https://github.com/obss/sahi) | sliced inference 참고문헌, package 미사용 | MIT | 현재 코드는 자체 tile+NMS이며 SAHI 재현이 아님 |

## 데이터와 이미지

- 원본 공개 이미지와 사용자 촬영 이미지는 GitHub에 포함하지 않습니다.
- 공개 데이터의 creator, source URL, license, license URL은 manifest에 보존합니다.
- 데이터별 조건은 [`configs/datasets.curated.yaml`](configs/datasets.curated.yaml)과
  [`configs/sources.wikimedia.yaml`](configs/sources.wikimedia.yaml)을 우선합니다.
- 논문이 CC BY라고 해서 논문이 소개한 모든 원본 이미지의 재배포 권한까지 자동으로 생기는 것은
  아닙니다.

## 모델 artifact

`weights/pretrained/yolox_s.pth`의 source, byte size와 SHA-256은
[`weights/README.md`](weights/README.md)에 기록되어 있습니다. 2026-08-18 학습 진행 checkpoint는
[`reports/progress/rpi_bootstrap_2026-08-18`](reports/progress/rpi_bootstrap_2026-08-18/README.md)에
framework/dataset license, config hash, source/public checkpoint SHA-256과 함께 기록했습니다.

## 학습 데이터 attribution

- Creator: Adam Byerly (`AdamByerly` / Kaggle handle `frettapper`)
- Dataset: [micro-PCB Images](https://www.kaggle.com/datasets/frettapper/micropcb-images)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 사용 범위: Raspberry Pi 1 B+, Raspberry Pi 3 B+, Raspberry Pi A+ 합계 1,875장
- 적용한 변경: bbox 변환, pHash-connected split, resize/augmentation, YOLO11m/YOLOX-S 학습
- 원본/processed image는 이 GitHub snapshot에 포함하지 않음

## Candidate source — Git에 데이터 미포함

- Dataset: [smdComponents raw-images v2](https://universe.roboflow.com/dainius/smdcomponents/dataset/2)
- Authors/project: Dainius Varna and Vytautas Abromavicius
- Rights record: `PDM-1.0 asserted by the Roboflow Universe project maintained by Dainius; this records the source assertion and Public Domain Mark is not a license grant.`
- 상태: candidate only; source image와 annotation archive는 Git에 포함하지 않음
- 범위: provider split은 독립 자체촬영 validation/test가 생기기 전까지 bootstrap-train-only
