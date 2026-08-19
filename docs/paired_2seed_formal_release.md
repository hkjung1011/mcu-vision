# Paired 2-seed formal release policy

이 경로는 추가 학습 없이 완료된 seed 42/43의 matched pair만 기술통계로 비교하기 위한 별도
release tier입니다. 학습 protocol인 `configs/experiments/baseline_v1.yaml`과 기존 run 증빙은
변경하지 않습니다.

## 고정 계약

- release policy: `configs/experiments/rpi_bootstrap_paired_2seed_release_v1.yaml`
- base protocol SHA-256: `02facd21ef061fc6530c064d4397ab82e36af3e0601cb502d46f7a6ec34f46f5`
- exact matrix: YOLO11m/YOLOX-S × seed 42/43 × 100 epochs = 4 runs
- evidence tier: `paired_2seed_descriptive`
- 기술통계: model별 n=2, paired n=2, df=1
- 허용: per-run metric, mean, sample SD, 같은 seed의 paired delta
- 금지: statistical significance, population superiority, production-ready, independent-test 주장

이 범위 축소는 학습 시작 후 결정되었습니다. 따라서 seed44는 YOLO11m 완료 run을 포함해 두
모델 모두에서 일괄 제외하며, 중단된 YOLOX-S retry3/retry4도 입력으로 사용할 수 없습니다.

## 실행

```powershell
.\scripts\compare_formal_paired_2seed_runs.ps1 `
  -Yolo11Seed42 <yolo11m-seed42-run> `
  -Yolo11Seed43 <yolo11m-seed43-run> `
  -YoloXSeed42 <yolox-s-seed42-run> `
  -YoloXSeed43 <yolox-s-seed43-run> `
  -OutputDirectory <empty-output-directory>
```

wrapper와 Python CLI는 policy/attestation/base protocol SHA, exact 4-pair matrix, 100개의 ordered
epoch row, complete non-smoke 상태, 공통 dataset·metric·latency·GPU 증빙을 모두 재검증합니다.
99 epoch, missing/extra/duplicate pair, seed44 포함, unpaired fifth run, policy/attestation tamper는
fail-closed 처리됩니다.

산출물의 `protocol_compatibility.json`, `formal_execution_status.json`,
`formal_validation.json`, Markdown report와 모델 비교 chart에는 policy ID/SHA, base SHA,
evidence tier, n=2, df=1, descriptive-only 범위와 paired seed delta가 기록됩니다.
