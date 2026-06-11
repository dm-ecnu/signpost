from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "outputs" / "suffer10_20260609_final_metrics_by_dataset"
OUT_ROOT = ROOT / "outputs" / "suffer10_20260609_final_metrics_combined"

CSV_NAMES = [
    "answer_quality_summary.csv",
    "evidence_navigation_summary.csv",
    "agent_process_summary.csv",
    "online_efficiency_summary.csv",
    "ablation_summary.csv",
    "per_query_final_metrics.csv",
    "index_efficiency_summary.csv",
    "layer_graph_summary.csv",
    "shared_or_auxiliary_stage_summary.csv",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dataset_dirs() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir())


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = []
    for csv_name in CSV_NAMES:
        rows: list[dict[str, Any]] = []
        for dataset_dir in dataset_dirs():
            for row in read_csv(dataset_dir / "metrics" / csv_name):
                row.setdefault("dataset", dataset_dir.name)
                rows.append(row)
        if rows:
            write_csv(OUT_ROOT / "metrics" / csv_name, rows)
            manifests.append({"file": f"metrics/{csv_name}", "rows": len(rows)})

    final_metrics = {}
    for dataset_dir in dataset_dirs():
        path = dataset_dir / "metrics" / "final_metrics.json"
        if path.exists():
            final_metrics[dataset_dir.name] = json.loads(path.read_text(encoding="utf-8"))
    (OUT_ROOT / "metrics" / "final_metrics_by_dataset.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifests.append({"file": "metrics/final_metrics_by_dataset.json", "datasets": len(final_metrics)})
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in manifests:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
