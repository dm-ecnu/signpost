from __future__ import annotations

"""H200 F10 graph ES sync auto-recovery runner.

This runner does not call any LLM. It controls the embedding service tmux
session, runs F10 with checkpointing, and only enables multi-vector fallback for
graph objects that repeatedly crash the embedding service at the same parent id.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path


EMBED_START_COMMAND = (
    "CUDA_VISIBLE_DEVICES=2 VLLM_USE_DEEP_GEMM=0 "
    "python -m vllm.entrypoints.openai.api_server "
    "--model /data/srl/nemotron-8b "
    "--runner pooling "
    "--port 8001 "
    "--trust-remote-code"
)


def restart_embed(session: str, *, wait_seconds: int) -> None:
    run(["tmux", "has-session", "-t", session])
    run(["tmux", "send-keys", "-t", session, "C-c"], check=False)
    time.sleep(3)
    run(["tmux", "send-keys", "-t", session, "cd /data/srl", "C-m"])
    run(["tmux", "send-keys", "-t", session, "conda activate /data/srl/.conda_envs/vllm", "C-m"])
    run(["tmux", "send-keys", "-t", session, EMBED_START_COMMAND, "C-m"])
    wait_for_embed(wait_seconds)


def wait_for_embed(wait_seconds: int) -> None:
    deadline = time.time() + wait_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            request_json("http://localhost:8001/v1/models", None)
            request_json(
                "http://localhost:8001/v1/embeddings",
                {"model": "/data/srl/nemotron-8b", "input": ["embedding health check"]},
            )
            print("[embed] 8001 ready", flush=True)
            return
        except Exception as exc:  # pragma: no cover - H200 runtime guard.
            last_error = str(exc)
            time.sleep(5)
    raise RuntimeError(f"8001 did not become healthy: {last_error}")


def request_json(url: str, body: dict[str, Any] | None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def run_f10(args: argparse.Namespace, *, recreate: bool, resume: bool) -> int:
    cmd = [
        "python",
        "-m",
        "signpost.benchmark.time_stage",
        "--dataset",
        args.dataset,
        "--stage",
        "F10_graph_es_sync",
        "--method-scope",
        "method_offline_index",
        "--method",
        "signpost",
        "--input-path",
        f"datasets/processed/{args.dataset}/graph.unified.json",
        "--output-path",
        f"signpost-{args.namespace}-graph",
        "--metrics-json",
        f"outputs/{args.dataset}/logs/stage_metrics/F10_graph_es_sync.json",
        "--stdout-log",
        f"outputs/{args.dataset}/logs/F10_graph_es_sync.stdout.log",
        "--stderr-log",
        f"outputs/{args.dataset}/logs/F10_graph_es_sync.stderr.log",
        "--log",
        f"outputs/{args.dataset}/logs/stage_timing.jsonl",
        "--auto-metrics",
        "--",
        "python",
        "-m",
        "signpost.indexing.graph_es_sync",
        "--namespace",
        args.namespace,
        "--graph",
        f"datasets/processed/{args.dataset}/graph.unified.json",
        "--embedding-provider",
        "ecnu",
        "--batch-size",
        str(args.batch_size),
        "--update-chunk-parents",
        "--progress-log",
        str(args.progress_log),
        "--state-file",
        str(args.state_file),
        "--multi-vector-parts-file",
        str(args.parts_file),
    ]
    if recreate:
        cmd.append("--recreate")
    if resume:
        cmd.append("--resume")
    return subprocess.run(cmd, check=False).returncode


def latest_failed_parent(progress_log: Path) -> tuple[str, int] | None:
    if not progress_log.exists():
        return None
    failed = None
    for line in progress_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "failed":
            failed = row
    if not failed:
        return None
    parent = str(failed.get("graph_parent_id") or failed.get("id") or "")
    chars = int(failed.get("content_chars") or 0)
    return (parent, chars) if parent else None


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def update_recovery_decision(args: argparse.Namespace, parent_id: str, content_chars: int, repeats: int) -> None:
    parts = read_json_object(args.parts_file)
    current = max(1, int(parts.get(parent_id, 1)))
    next_parts = min(args.max_parts, max(2, current * 2))
    parts[parent_id] = next_parts
    write_json_object(args.parts_file, parts)
    row = {
        "time": time.time(),
        "dataset": args.dataset,
        "namespace": args.namespace,
        "graph_parent_id": parent_id,
        "content_chars_at_failure": content_chars,
        "consecutive_failures": repeats,
        "previous_parts": current,
        "new_parts": next_parts,
        "reason": "same graph object repeatedly failed embedding; enable object-local multi-vector fallback without truncation",
    }
    append_jsonl(args.decision_log, row)
    append_jsonl(args.multivector_log, row)
    print(f"[decision] parent={parent_id} failures={repeats} parts {current}->{next_parts}", flush=True)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run F10 with H200 embedding-service auto recovery.")
    parser.add_argument("--dataset", default="legal")
    parser.add_argument("--namespace", default="legal")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--repeat-threshold", type=int, default=3)
    parser.add_argument("--max-parts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embed-session", default="embed")
    parser.add_argument("--embed-wait-seconds", type=int, default=600)
    parser.add_argument("--project-dir", default=os.environ.get("RAG_PROJECT_BASE", "/data/srl/signpost_re"))
    args = parser.parse_args()

    os.chdir(args.project_dir)
    base = resolve_project_path(f"outputs/{args.dataset}/logs")
    args.progress_log = base / "F10_graph_es_sync.progress.jsonl"
    args.state_file = base / "F10_graph_es_sync.state.json"
    args.parts_file = base / "F10_graph_es_sync.multivector_parts.json"
    args.decision_log = base / "F10_graph_es_sync.recovery_decisions.jsonl"
    args.multivector_log = base / "F10_graph_es_sync.multivector_objects.jsonl"

    failure_counts: dict[str, int] = {}
    recreate = not args.progress_log.exists()
    for attempt in range(1, args.max_attempts + 1):
        print(f"[f10-auto] attempt={attempt}/{args.max_attempts} recreate={recreate}", flush=True)
        restart_embed(args.embed_session, wait_seconds=args.embed_wait_seconds)
        rc = run_f10(args, recreate=recreate, resume=not recreate)
        if rc == 0:
            print("[f10-auto] F10 completed", flush=True)
            return 0
        failed = latest_failed_parent(args.progress_log)
        if not failed:
            print("[f10-auto] F10 failed before recording a graph object; retry as service failure", flush=True)
            recreate = False
            continue
        parent_id, chars = failed
        failure_counts[parent_id] = failure_counts.get(parent_id, 0) + 1
        print(f"[f10-auto] failed_parent={parent_id} chars={chars} repeats={failure_counts[parent_id]}", flush=True)
        if failure_counts[parent_id] >= args.repeat_threshold:
            update_recovery_decision(args, parent_id, chars, failure_counts[parent_id])
            failure_counts[parent_id] = 0
        recreate = False
    print("[f10-auto] exhausted attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
