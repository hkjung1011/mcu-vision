from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcu_data.progress_publishing import verify_progress_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify every hash in a progress snapshot.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = verify_progress_snapshot(args.report_dir, project_root=args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
