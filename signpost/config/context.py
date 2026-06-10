from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# `PROJECT_ROOT` is always the `signpost_re` directory.  All CLI modules use this
# as the base for relative paths so commands can be run reproducibly from the
# project root without depending on the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentContext:
    """Minimal experiment identity used instead of user/tenant concepts.

    The old project used `tenant_id`, `user_id`, and `kb_id` because it was
    shaped like a product backend.  The research version only needs the
    experiment namespace, dataset identity, and run output location.
    """

    namespace: str
    dataset_id: str
    run_id: str = "default"
    output_dir: Path | None = None

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return PROJECT_ROOT / "outputs" / self.namespace / self.run_id


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a CLI/config path against `signpost_re` if it is relative."""

    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value
