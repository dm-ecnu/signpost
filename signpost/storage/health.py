from __future__ import annotations

"""F2 storage health checks.

These checks intentionally stop at connectivity and service health.  They do
not create product tables, buckets, indexes, users, or permissions.
"""

import socket
import urllib.request
from dataclasses import dataclass
from typing import Any

from signpost.config.settings import load_settings


@dataclass(frozen=True)
class HealthResult:
    service: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"service": self.service, "ok": self.ok, "detail": self.detail}


def check_elasticsearch() -> HealthResult:
    settings = load_settings()
    url = _service_url(settings.service_conf, "elasticsearch", aliases=("es",), default="http://127.0.0.1:9200")
    return _http_get("elasticsearch", url)


def check_minio() -> HealthResult:
    settings = load_settings()
    url = _service_url(settings.service_conf, "minio", default="http://127.0.0.1:9000").rstrip("/") + "/minio/health/live"
    return _http_get("minio", url)


def check_redis() -> HealthResult:
    settings = load_settings()
    redis_conf = settings.service_conf.get("redis") or settings.service_conf.get("valkey") or {}
    host, port = _host_port(redis_conf, default_host="127.0.0.1", default_port=6379)
    try:
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(b"*1\r\n$4\r\nPING\r\n")
            response = sock.recv(64)
        ok = response.startswith(b"+PONG")
        return HealthResult("redis", ok, response.decode("utf-8", errors="replace").strip())
    except OSError as exc:
        return HealthResult("redis", False, str(exc))


def check_postgres() -> HealthResult:
    settings = load_settings()
    pg_conf = settings.service_conf.get("postgres") or {}
    host, port = _host_port(pg_conf, default_host="127.0.0.1", default_port=5432)
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
        return HealthResult("postgres", True, f"tcp://{host}:{port} reachable")
    except OSError as exc:
        return HealthResult("postgres", False, str(exc))


def _http_get(service: str, url: str) -> HealthResult:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return HealthResult(service, 200 <= response.status < 300, f"HTTP {response.status}")
    except Exception as exc:
        return HealthResult(service, False, str(exc))


def _service_url(conf: dict[str, Any], key: str, *, aliases: tuple[str, ...] = (), default: str) -> str:
    value = conf.get(key)
    for alias in aliases:
        if value is None:
            value = conf.get(alias)
    if isinstance(value, dict):
        if value.get("url"):
            return str(value["url"])
        host_value = value.get("hosts") or value.get("host") or "127.0.0.1"
        if isinstance(host_value, list):
            host_value = host_value[0] if host_value else "127.0.0.1"
        host_text = str(host_value)
        if host_text.startswith("http://") or host_text.startswith("https://"):
            return host_text
        host, port = _host_port(value, default_host=host_text, default_port=0)
        scheme = value.get("scheme", "http")
        if port:
            return f"{scheme}://{host}:{port}"
        return f"{scheme}://{host}"
    return default


def _host_port(conf: Any, *, default_host: str, default_port: int) -> tuple[str, int]:
    """Parse either split host/port fields or a `host:port` string."""

    if not isinstance(conf, dict):
        return default_host, default_port
    host = str(conf.get("host") or conf.get("hosts") or default_host)
    port_value = conf.get("port")
    if ":" in host and not host.startswith("http://") and not host.startswith("https://"):
        host_part, port_part = host.rsplit(":", 1)
        if port_value is None and port_part.isdigit():
            return host_part, int(port_part)
    return host, int(port_value or default_port)
