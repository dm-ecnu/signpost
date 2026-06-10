from __future__ import annotations

"""F0 smoke CLI.

The command verifies that the project can construct an experiment context and
read local configuration without touching external services.
"""

import argparse
import json

from signpost.config.context import ExperimentContext
from signpost.config.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test F0 config and experiment context")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--run-id", default="default")
    args = parser.parse_args()

    dataset_id = args.dataset_id or args.namespace
    context = ExperimentContext(namespace=args.namespace, dataset_id=dataset_id, run_id=args.run_id)
    settings = load_settings()
    payload = {
        "context": {
            "namespace": context.namespace,
            "dataset_id": context.dataset_id,
            "run_id": context.run_id,
            "output_dir": str(context.resolved_output_dir()),
        },
        "project_root": str(settings.project_root),
        "env_keys": sorted(settings.env.keys()),
        "service_conf_keys": sorted(settings.service_conf.keys()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

