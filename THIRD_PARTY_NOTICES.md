# Third-party notices

확인일: **2026-08-17**

이 저장소 자체의 project-level license는 아직 지정되지 않았습니다. 저장소가 private이라는 사실은
third-party software, pretrained weights, dataset의 라이선스 조건을 변경하지 않습니다. 소유자가 별도
라이선스를 선택하기 전에는 허가된 저장소 사용자 외부로 프로젝트 코드를 재배포하지 않습니다.

| 구성요소 | 사용/예정 범위 | 라이선스·근거 | 주의 |
|---|---|---|---|
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 학습 source와 pretrained YOLOX-S | Apache-2.0 | 고정 commit과 upstream weight hash 보존 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO11 학습·현재 autolabel backend | AGPL-3.0 또는 Enterprise | private/상용/embedded 이용은 적용 조건 별도 검토 |
| [PyTorch](https://github.com/pytorch/pytorch) | CUDA 학습 runtime | upstream license 참조 | binary wheel의 bundled component notice 포함 |
| [pycocotools 2.0.11](https://pypi.org/project/pycocotools/2.0.11/) | 공통 COCO AP/AR | package metadata/license 참조 | exact version 고정 |
| [CVAT Community](https://github.com/cvat-ai/cvat) | 권장 검수 UI, 아직 미통합 | MIT | SDK/import round-trip NOT VERIFIED |
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
[`weights/README.md`](weights/README.md)에 기록되어 있습니다. 향후 trained checkpoint/ONNX를 추가할
때는 학습에 사용한 framework와 dataset license, config hash, checkpoint SHA-256을 함께 기록합니다.
