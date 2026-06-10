from __future__ import annotations

"""F0 configuration loading.

This module intentionally keeps configuration small and explicit.  It reads the
project `.env` file and `conf/service_conf.yaml`, then exposes only the fields
needed by the research pipeline.  Product concepts from the old backend, such as
tenant tables and API-token settings, are not represented here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signpost.config.context import PROJECT_ROOT, resolve_project_path


@dataclass(frozen=True)
class Settings:
    """Merged runtime settings for F0-F3 smoke commands."""

    project_root: Path
    env: dict[str, str]
    service_conf: dict[str, Any]

    def env_value(self, key: str, default: str = "") -> str:
        return self.env.get(key, default)


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read a simple KEY=VALUE `.env` file without mutating `os.environ`."""

    env_path = resolve_project_path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_service_conf(path: str | Path = "conf/service_conf.yaml") -> dict[str, Any]:
    """Load the YAML service configuration.

    PyYAML is available in the target environment, but the fallback keeps F0
    smoke usable even in a minimal Python environment.
    """

    conf_path = resolve_project_path(path)
    if not conf_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return _tiny_yaml_mapping(conf_path.read_text(encoding="utf-8"))


def load_settings() -> Settings:
    """Load all F0 settings from the project root."""

    return Settings(project_root=PROJECT_ROOT, env=load_dotenv(), service_conf=load_service_conf())


def _tiny_yaml_mapping(text: str) -> dict[str, Any]:
    """Very small YAML fallback for top-level `key: value` pairs.

    It is not intended to replace a YAML parser; it only gives useful diagnostics
    when PyYAML is absent.
    """

    data: dict[str, Any] = {}
    current_parent: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if indent == 0:
            current_parent = key
            data[key] = value if value else {}
        elif current_parent and isinstance(data.get(current_parent), dict):
            data[current_parent][key] = value
    return data

