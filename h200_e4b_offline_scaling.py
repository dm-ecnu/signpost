#!/usr/bin/env python3
"""H200 helper for E4b isolated offline SIGNPOST index scaling.

Run this script on H200 from the project root after activating the project
environment. It creates timestamped processed subsets and unique ES namespaces.
Entity/relation extraction is explicitly out of scope: each subset semantic
graph is filtered from the already-produced semantic graph, and timed stages
only cover structure/sequence construction, graph merge, and ES synchronization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_graph_counts(graph_path: Path) -> Dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_counts: Dict[str, int] = {}
    edge_counts: Dict[str, int] = {}
    for node in nodes:
        key = str(node.get("node_type", "unknown"))
        node_counts[key] = node_counts.get(key, 0) + 1
    for edge in edges:
        key = str(edge.get("edge_type", "unknown"))
        edge_counts[key] = edge_counts.get(key, 0) + 1
    retrievable = (
        node_counts.get("chunk", 0)
        + node_counts.get("summary", 0)
        + node_counts.get("entity", 0)
        + edge_counts.get("semantic", 0)
    )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "chunk_nodes": node_counts.get("chunk", 0),
        "summary_nodes": node_counts.get("summary", 0),
        "entity_nodes": node_counts.get("entity", 0),
        "semantic_edges": edge_counts.get("semantic", 0),
        "structure_edges": edge_counts.get("structure", 0),
        "sequence_edges": edge_counts.get("sequence", 0),
        "source_edges": edge_counts.get("source", 0),
        "retrievable_objects": retrievable,
        "graph_json_bytes": graph_path.stat().st_size,
    }


def stage_seconds(stage_log: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not stage_log.exists():
        return out
    for row in read_jsonl(stage_log):
        stage = row.get("stage")
        if stage:
            out[str(stage)] = float(row.get("wall_time_seconds") or 0.0)
    return out


def run_logged(cmd: List[str], project_root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=str(project_root), check=True, stdout=log, stderr=subprocess.STDOUT)


def select_doc_ids(chunks: Sequence[Dict[str, Any]], fraction: float) -> List[str]:
    doc_ids = sorted({str(row.get("doc_id")) for row in chunks if row.get("doc_id")})
    if not doc_ids:
        raise RuntimeError("No doc_id values found in chunks.jsonl")
    keep = max(1, int(math.ceil(len(doc_ids) * fraction)))
    return doc_ids[:keep]


def edge_source_chunk_ids(edge: Dict[str, Any]) -> set:
    out = set(edge.get("source_chunk_ids") or [])
    source_mapping = edge.get("source_mapping") or {}
    for key in source_mapping:
        if ":" in key:
            out.add(key.rsplit(":", 1)[-1])
    for key in ("source_chunk_id", "target_chunk_id"):
        if edge.get(key):
            out.add(edge.get(key))
    return out


def node_source_chunk_ids(node: Dict[str, Any]) -> set:
    out = set(node.get("source_chunk_ids") or [])
    chunk_id = node.get("chunk_id")
    if chunk_id:
        out.add(chunk_id)
    source_mapping = node.get("source_mapping") or {}
    for key in source_mapping:
        if ":" in key:
            out.add(key.rsplit(":", 1)[-1])
    return out


def filter_semantic_graph(source_graph: Path, output_graph: Path, selected_chunk_ids: set, selected_doc_ids: set) -> Dict[str, Any]:
    graph = json.loads(source_graph.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    kept_edges = []
    required_node_ids = set()
    for edge in edges:
        source_chunks = edge_source_chunk_ids(edge)
        if source_chunks and source_chunks.isdisjoint(selected_chunk_ids):
            continue
        if edge.get("edge_type") == "source":
            target = edge.get("target")
            if target and str(target).startswith("chunk:"):
                cid = str(target).split("chunk:", 1)[1]
                if cid not in selected_chunk_ids:
                    continue
        kept_edges.append(edge)
        if edge.get("source"):
            required_node_ids.add(edge.get("source"))
        if edge.get("target"):
            required_node_ids.add(edge.get("target"))

    kept_nodes = []
    for node in nodes:
        node_id = node.get("node_id")
        node_type = node.get("node_type")
        keep = False
        if node_type == "chunk":
            keep = node.get("chunk_id") in selected_chunk_ids or node.get("doc_id") in selected_doc_ids
        elif node_id in required_node_ids:
            keep = True
        else:
            source_chunks = node_source_chunk_ids(node)
            keep = bool(source_chunks and not source_chunks.isdisjoint(selected_chunk_ids))
        if keep:
            kept_nodes.append(node)
            if node_id:
                required_node_ids.add(node_id)

    kept_node_ids = {n.get("node_id") for n in kept_nodes}
    kept_edges = [e for e in kept_edges if e.get("source") in kept_node_ids and e.get("target") in kept_node_ids]
    filtered = {
        "metadata": {
            **(graph.get("metadata") or {}),
            "filtered_for_e4b": True,
            "source_graph": str(source_graph),
            "selected_chunks": len(selected_chunk_ids),
            "selected_docs": len(selected_doc_ids),
            "note": "Filtered from existing semantic graph; entity/relation extraction is not rerun or timed.",
        },
        "nodes": kept_nodes,
        "edges": kept_edges,
    }
    write_json(output_graph, filtered)
    return {
        "semantic_graph_input_nodes": len(nodes),
        "semantic_graph_input_edges": len(edges),
        "semantic_graph_filtered_nodes": len(kept_nodes),
        "semantic_graph_filtered_edges": len(kept_edges),
    }


def prepare_subset(
    source_processed: Path,
    subset_processed: Path,
    selected_doc_ids: Sequence[str],
) -> Dict[str, Any]:
    selected = set(selected_doc_ids)
    chunks = [r for r in read_jsonl(source_processed / "chunks.jsonl") if r.get("doc_id") in selected]
    trees = [r for r in read_jsonl(source_processed / "document_trees.jsonl") if r.get("doc_id") in selected]
    selected_chunk_ids = {r.get("chunk_id") for r in chunks}
    subset_processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset_processed / "chunks.jsonl", chunks)
    write_jsonl(subset_processed / "document_trees.jsonl", trees)
    semantic_stats = filter_semantic_graph(
        source_processed / "graph.semantic.llm.json",
        subset_processed / "graph.semantic.llm.json",
        selected_chunk_ids,
        selected,
    )
    return {
        "doc_count": len(selected_doc_ids),
        "chunk_count": len(chunks),
        "document_tree_count": len(trees),
        **semantic_stats,
    }


def run_offline_stages(project_root: Path, subset_name: str, namespace: str, exp_run_dir: Path) -> Dict[str, Any]:
    processed = Path("datasets") / "processed" / subset_name
    stage_log = exp_run_dir / "stage_timing.jsonl"
    cmd_log = exp_run_dir / "commands.log"

    commands = [
        [
            "python",
            "-m",
            "signpost.benchmark.time_stage",
            "--dataset",
            subset_name,
            "--stage",
            "E4b_F7_structure_graph",
            "--method-scope",
            "method_offline_index",
            "--method",
            "signpost",
            "--log",
            str(stage_log),
            "--input-path",
            str(processed / "chunks.jsonl"),
            "--output-path",
            str(processed / "graph.structure.json"),
            "--disk-path",
            str(processed / "graph.structure.json"),
            "--auto-metrics",
            "--",
            "python",
            "-m",
            "signpost.indexing.structure_graph",
            "--namespace",
            namespace,
            "--chunks",
            str(processed / "chunks.jsonl"),
            "--document-trees",
            str(processed / "document_trees.jsonl"),
            "--output",
            str(processed / "graph.structure.json"),
            "--summarizer",
            os.environ.get("SUMMARIZER", "deterministic"),
        ],
        [
            "python",
            "-m",
            "signpost.benchmark.time_stage",
            "--dataset",
            subset_name,
            "--stage",
            "E4b_F8_sequence_graph",
            "--method-scope",
            "method_offline_index",
            "--method",
            "signpost",
            "--log",
            str(stage_log),
            "--input-path",
            str(processed / "chunks.jsonl"),
            "--output-path",
            str(processed / "graph.sequence.json"),
            "--disk-path",
            str(processed / "graph.sequence.json"),
            "--auto-metrics",
            "--",
            "python",
            "-m",
            "signpost.indexing.sequence_graph",
            "--namespace",
            namespace,
            "--chunks",
            str(processed / "chunks.jsonl"),
            "--output",
            str(processed / "graph.sequence.json"),
        ],
        [
            "python",
            "-m",
            "signpost.benchmark.time_stage",
            "--dataset",
            subset_name,
            "--stage",
            "E4b_F9_unified_graph",
            "--method-scope",
            "method_offline_index",
            "--method",
            "signpost",
            "--log",
            str(stage_log),
            "--output-path",
            str(processed / "graph.unified.json"),
            "--disk-path",
            str(processed / "graph.unified.json"),
            "--auto-metrics",
            "--",
            "python",
            "-m",
            "signpost.graph.merge",
            "--namespace",
            namespace,
            "--semantic",
            str(processed / "graph.semantic.llm.json"),
            "--structure",
            str(processed / "graph.structure.json"),
            "--sequence",
            str(processed / "graph.sequence.json"),
            "--output",
            str(processed / "graph.unified.json"),
        ],
        [
            "python",
            "-m",
            "signpost.benchmark.time_stage",
            "--dataset",
            subset_name,
            "--stage",
            "E4b_F10_graph_es_sync",
            "--method-scope",
            "method_offline_index",
            "--method",
            "signpost",
            "--log",
            str(stage_log),
            "--input-path",
            str(processed / "graph.unified.json"),
            "--output-path",
            str(exp_run_dir / "F10_graph_es_sync.done"),
            "--auto-metrics",
            "--",
            "python",
            "-m",
            "signpost.indexing.graph_es_sync",
            "--namespace",
            namespace,
            "--graph",
            str(processed / "graph.unified.json"),
            "--embedding-provider",
            os.environ.get("EMBEDDING_PROVIDER", "ecnu"),
            "--recreate",
            "--update-chunk-parents",
        ],
    ]

    for cmd in commands:
        run_logged(cmd, project_root, cmd_log)
    return stage_seconds(stage_log)


def package_return(project_root: Path, exp_root: Path, stamp: str) -> Path:
    tar_path = project_root.parent / f"signpost_e4b_offline_scaling_{stamp}_return.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in [
            exp_root.relative_to(project_root) / "manifest.json",
            exp_root.relative_to(project_root) / "summary",
            exp_root.relative_to(project_root) / "runs",
        ]:
            tar.add(project_root / rel, arcname=str(rel))
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", default="agriculture")
    parser.add_argument("--fractions", default="0.25,0.50,1.00")
    parser.add_argument("--stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    source_processed = project_root / "datasets" / "processed" / args.dataset
    if not source_processed.exists():
        raise RuntimeError(f"Missing source processed directory: {source_processed}")
    source_chunks = read_jsonl(source_processed / "chunks.jsonl")
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    exp_root = project_root / "outputs" / f"paper_e4b_offline_scaling_{args.stamp}"
    summary_dir = exp_root / "summary"
    run_root = exp_root / "runs"
    summary_rows: List[Dict[str, Any]] = []
    manifest = {
        "dataset": args.dataset,
        "fractions": fractions,
        "stamp": args.stamp,
        "project_root": str(project_root),
        "note": "Isolated offline SIGNPOST index scaling. Entity/relation extraction and answer generation are not run or timed.",
    }

    for fraction in fractions:
        pct = int(round(fraction * 100))
        subset_name = f"paper_e4b_{args.dataset}_{pct:03d}_{args.stamp}"
        namespace = subset_name
        subset_processed = project_root / "datasets" / "processed" / subset_name
        selected_doc_ids = select_doc_ids(source_chunks, fraction)
        prep = prepare_subset(source_processed, subset_processed, selected_doc_ids)
        run_dir = run_root / subset_name
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "subset_manifest.json", {"subset": subset_name, "namespace": namespace, **prep})
        times = run_offline_stages(project_root, subset_name, namespace, run_dir)
        graph_counts = load_graph_counts(subset_processed / "graph.unified.json")
        build_f7_f10 = sum(times.get(k, 0.0) for k in ["E4b_F7_structure_graph", "E4b_F8_sequence_graph", "E4b_F9_unified_graph", "E4b_F10_graph_es_sync"])
        row = {
            "dataset": args.dataset,
            "subset": subset_name,
            "namespace": namespace,
            "fraction": fraction,
            **prep,
            **graph_counts,
            "build_seconds_f7_f10": f"{build_f7_f10:.6f}",
            "seconds_f7_structure": f"{times.get('E4b_F7_structure_graph', 0.0):.6f}",
            "seconds_f8_sequence": f"{times.get('E4b_F8_sequence_graph', 0.0):.6f}",
            "seconds_f9_unified": f"{times.get('E4b_F9_unified_graph', 0.0):.6f}",
            "seconds_f10_es_sync": f"{times.get('E4b_F10_graph_es_sync', 0.0):.6f}",
        }
        summary_rows.append(row)

    fields = [
        "dataset",
        "subset",
        "namespace",
        "fraction",
        "doc_count",
        "chunk_count",
        "document_tree_count",
        "semantic_graph_input_nodes",
        "semantic_graph_input_edges",
        "semantic_graph_filtered_nodes",
        "semantic_graph_filtered_edges",
        "nodes",
        "edges",
        "chunk_nodes",
        "summary_nodes",
        "entity_nodes",
        "semantic_edges",
        "structure_edges",
        "sequence_edges",
        "source_edges",
        "retrievable_objects",
        "graph_json_bytes",
        "build_seconds_f7_f10",
        "seconds_f7_structure",
        "seconds_f8_sequence",
        "seconds_f9_unified",
        "seconds_f10_es_sync",
    ]
    write_tsv(summary_dir / "e4b_offline_scaling_summary.tsv", summary_rows, fields)
    write_json(summary_dir / "e4b_offline_scaling_summary.json", summary_rows)
    write_json(exp_root / "manifest.json", manifest)
    tar_path = package_return(project_root, exp_root, args.stamp)
    print(f"E4b done. Return package: {tar_path}")


if __name__ == "__main__":
    main()
