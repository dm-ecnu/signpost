from __future__ import annotations

"""Run a pipeline command and append a stage timing JSONL record."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO

from signpost.config.context import resolve_project_path


def run_timed_stage(
    *,
    dataset: str,
    stage: str,
    log_path: Path,
    command: list[str],
    method_scope: str = "",
    input_path: str = "",
    output_path: str = "",
    method: str = "",
    llm_calls: float = 0,
    input_tokens: float = 0,
    output_tokens: float = 0,
    disk_path: str = "",
    metrics_path: Path | None = None,
    auto_metrics: bool = False,
    capture_output: bool = False,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    status = "ok"
    return_code = 0
    error = ""
    try:
        if stdout_log is not None or stderr_log is not None:
            stdout_handle = open_text_log(stdout_log) if stdout_log is not None else None
            stderr_handle = open_text_log(stderr_log) if stderr_log is not None else None
            try:
                completed = subprocess.run(command, check=False, text=True, stdout=stdout_handle, stderr=stderr_handle)
            finally:
                if stdout_handle is not None:
                    stdout_handle.close()
                if stderr_handle is not None:
                    stderr_handle.close()
        elif capture_output:
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
        else:
            completed = subprocess.run(command, check=False)
        return_code = completed.returncode
        if return_code != 0:
            status = "failed"
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper.
        status = "failed"
        return_code = 1
        error = str(exc)
    finished = time.time()
    extra_metrics: dict[str, Any] = {}
    metrics_error = ""
    if metrics_path and metrics_path.exists():
        extra_metrics.update(read_metrics_json(metrics_path))
    if auto_metrics:
        try:
            from signpost.benchmark.artifact_metrics import collect_stage_metrics

            extra_metrics.update(collect_stage_metrics(stage=stage, input_path=input_path, output_path=output_path))
        except Exception as exc:  # pragma: no cover - metrics collection must not hide stage status.
            metrics_error = str(exc)
    if metrics_path and auto_metrics and extra_metrics:
        write_metrics_json(metrics_path, extra_metrics)
    row = {
        "dataset": dataset,
        "method": method,
        "stage": stage,
        "method_scope": method_scope,
        "input_path": input_path,
        "output_path": output_path,
        "command": command,
        "started_at": started,
        "finished_at": finished,
        "wall_time_seconds": finished - started,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "disk_bytes": _path_size(resolve_project_path(disk_path or output_path)) if (disk_path or output_path) else 0,
        "stdout_log": str(stdout_log) if stdout_log else "",
        "stderr_log": str(stderr_log) if stderr_log else "",
        "metrics_path": str(metrics_path) if metrics_path else "",
        "extra_metrics": extra_metrics,
        "status": status,
        "return_code": return_code,
    }
    if error:
        row["error"] = error
    if metrics_error:
        row["metrics_error"] = metrics_error
    append_jsonl(log_path, row)
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def open_text_log(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metrics JSON must be an object: {path}")
    return data


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Time an experiment stage and append one JSONL log row.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--method-scope", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--input-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--llm-calls", type=float, default=0)
    parser.add_argument("--input-tokens", type=float, default=0)
    parser.add_argument("--output-tokens", type=float, default=0)
    parser.add_argument("--disk-path", default="", help="Optional artifact path whose recursive size should be recorded.")
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional JSON object emitted by the wrapped stage. It is copied into extra_metrics after the command finishes.",
    )
    parser.add_argument(
        "--auto-metrics",
        action="store_true",
        help="After the command finishes, summarize known input/output artifacts into extra_metrics.",
    )
    parser.add_argument("--capture-output", action="store_true", help="Capture command stdout/stderr instead of streaming them.")
    parser.add_argument("--stdout-log", help="Optional path to stream wrapped command stdout while it runs.")
    parser.add_argument("--stderr-log", help="Optional path to stream wrapped command stderr while it runs.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run. Put -- before the command.")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing command after --")
    row = run_timed_stage(
        dataset=args.dataset,
        stage=args.stage,
        log_path=resolve_project_path(args.log),
        command=command,
        method_scope=args.method_scope,
        method=args.method,
        input_path=args.input_path,
        output_path=args.output_path,
        llm_calls=args.llm_calls,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        disk_path=args.disk_path,
        metrics_path=resolve_project_path(args.metrics_json) if args.metrics_json else None,
        auto_metrics=args.auto_metrics,
        capture_output=args.capture_output,
        stdout_log=resolve_project_path(args.stdout_log) if args.stdout_log else None,
        stderr_log=resolve_project_path(args.stderr_log) if args.stderr_log else None,
    )
    print(f"stage={row['stage']} status={row['status']} wall_time_seconds={row['wall_time_seconds']:.3f}")
    return int(row["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())
