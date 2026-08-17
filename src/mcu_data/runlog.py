from __future__ import annotations

import contextlib
import csv
import io
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

from .common import sha256_file, write_json


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def configure_utf8_output() -> None:
    """Keep redirected Windows terminal/log output Unicode-safe and machine-readable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass


def utc_now_precise() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE.sub("", value)


class TeeStream(io.TextIOBase):
    """Write to the live terminal and an ANSI-free UTF-8 log at the same time."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", None) or "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def write(self, value: str) -> int:
        try:
            self.terminal.write(value)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", None) or "utf-8"
            safe_value = value.encode(encoding, errors="replace").decode(encoding)
            self.terminal.write(safe_value)
        self.log.write(strip_ansi(value))
        self.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


@contextlib.contextmanager
def tee_console(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = TeeStream(stdout, handle)
        sys.stderr = TeeStream(stderr, handle)
        try:
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


def run_streamed(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Run a child process while preserving every line in the surrounding tee."""

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise


class GpuSampler:
    """Sample NVIDIA-SMI to CSV without requiring a CUDA Toolkit installation."""

    FIELD_NAMES = [
        "sample_utc",
        "gpu_index",
        "gpu_name",
        "memory_used_mib",
        "memory_total_mib",
        "gpu_util_percent",
        "temperature_c",
        "power_w",
    ]

    def __init__(self, output_csv: Path, interval_seconds: float = 1.0) -> None:
        self.output_csv = output_csv
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict[str, Any]] = []

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self.interval_seconds * 3))
        self._write_csv()
        numeric = {
            key: [float(row[key]) for row in self._rows if row.get(key) not in (None, "")]
            for key in (
                "memory_used_mib",
                "gpu_util_percent",
                "temperature_c",
                "power_w",
            )
        }
        return {
            "samples": len(self._rows),
            "peak_memory_used_mib": max(numeric["memory_used_mib"], default=None),
            "mean_gpu_util_percent": _mean(numeric["gpu_util_percent"]),
            "peak_temperature_c": max(numeric["temperature_c"], default=None),
            "peak_power_w": max(numeric["power_w"], default=None),
        }

    def __enter__(self) -> "GpuSampler":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            row = self._query()
            if row is not None:
                self._rows.append(row)
            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _query() -> dict[str, Any] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
        values = [item.strip() for item in first_line.split(",")]
        if len(values) != 7:
            return None
        return dict(zip(GpuSampler.FIELD_NAMES, [utc_now_precise(), *values], strict=True))

    def _write_csv(self) -> None:
        with self.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self._rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def collect_system_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_utc": utc_now_precise(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "cpu": platform.processor(),
        "cuda_toolkit_nvcc": None,
        "nvidia_smi": None,
    }
    for command, key in [(["nvcc", "--version"], "cuda_toolkit_nvcc"), (["nvidia-smi"], "nvidia_smi")]:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            environment[key] = (result.stdout or result.stderr).strip() or None
        except (FileNotFoundError, subprocess.SubprocessError):
            environment[key] = None
    return environment


def collect_torch_environment() -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "gpu_total_memory_mib": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
                ),
            }
        )
    return result


def state_dict_statistics(state_dict: dict[str, Any], output_csv: Path) -> dict[str, Any]:
    """Persist per-tensor checkpoint statistics without dumping raw weights."""

    import torch

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_numel = 0
    total_float_numel = 0
    total_square_sum = 0.0
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue
            detached = tensor.detach().cpu()
            numel = detached.numel()
            total_numel += numel
            row: dict[str, Any] = {
                "name": name,
                "shape": "x".join(str(value) for value in detached.shape),
                "dtype": str(detached.dtype).removeprefix("torch."),
                "numel": numel,
                "mean": "",
                "std": "",
                "min": "",
                "max": "",
                "l2_norm": "",
            }
            if numel and (detached.is_floating_point() or detached.is_complex()):
                values = detached.float()
                square_sum = float(torch.sum(values * values).item())
                total_float_numel += numel
                total_square_sum += square_sum
                row.update(
                    {
                        "mean": float(values.mean().item()),
                        "std": float(values.std(unbiased=False).item()),
                        "min": float(values.min().item()),
                        "max": float(values.max().item()),
                        "l2_norm": square_sum**0.5,
                    }
                )
            rows.append(row)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "tensor_count": len(rows),
        "parameter_and_buffer_values": total_numel,
        "floating_values": total_float_numel,
        "global_l2_norm": total_square_sum**0.5,
        "csv": str(output_csv.resolve()),
    }


def checkpoint_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "mib": path.stat().st_size / 1024**2,
        "sha256": sha256_file(path),
    }


def save_environment(path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    value = collect_system_environment()
    if extra:
        value.update(extra)
    write_json(path, value)
    return value


def collect_git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_paths": status}
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def write_pip_freeze(path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout, encoding="utf-8", newline="\n")


def print_section(title: str, values: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    width = max((len(str(key)) for key in values), default=0)
    for key, value in values.items():
        print(f"{key:<{width}} : {value}")


def ensure_deterministic_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
