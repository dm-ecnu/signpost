from __future__ import annotations

"""Minimal Elasticsearch client for research indexing.

This wrapper uses the Elasticsearch HTTP API directly.  It avoids the old
backend's document-store abstraction because F5 only needs deterministic index
creation, bulk upsert, and search for experiment objects.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from signpost.config.settings import load_settings


@dataclass(frozen=True)
class ElasticsearchConfig:
    url: str


def load_elasticsearch_config() -> ElasticsearchConfig:
    settings = load_settings()
    es_conf = settings.service_conf.get("es") or settings.service_conf.get("elasticsearch") or {}
    host = "http://127.0.0.1:9200"
    if isinstance(es_conf, dict):
        host = str(es_conf.get("hosts") or es_conf.get("host") or es_conf.get("url") or host)
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
    return ElasticsearchConfig(url=host.rstrip("/"))


class ElasticsearchClient:
    """Tiny JSON HTTP client for the subset of ES APIs used by F5."""

    def __init__(self, config: ElasticsearchConfig | None = None, timeout: int = 60):
        self.config = config or load_elasticsearch_config()
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | str | None = None, *, content_type: str = "application/json") -> Any:
        url = self.config.url + "/" + path.lstrip("/")
        data: bytes | None = None
        headers = {"Content-Type": content_type}
        if isinstance(body, dict):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Elasticsearch {method} {path} failed: HTTP {exc.code}: {detail}") from exc

    def exists_index(self, index_name: str) -> bool:
        try:
            self.request("HEAD", index_name)
            return True
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return False
            raise

    def create_index(self, index_name: str, mapping: dict[str, Any], *, recreate: bool = False) -> None:
        if recreate and self.exists_index(index_name):
            self.request("DELETE", index_name)
        if not self.exists_index(index_name):
            self.request("PUT", index_name, mapping)

    def bulk(self, operations: list[dict[str, Any]]) -> dict[str, Any]:
        lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in operations]
        body = "\n".join(lines) + "\n"
        result = self.request("POST", "_bulk", body, content_type="application/x-ndjson")
        if result.get("errors"):
            failures = [item for item in result.get("items", []) if item.get("index", {}).get("error")]
            raise RuntimeError(f"Elasticsearch bulk had {len(failures)} failures; first={failures[:1]}")
        return result

    def refresh(self, index_name: str) -> None:
        self.request("POST", f"{index_name}/_refresh")

    def update_doc(self, index_name: str, doc_id: str, partial_doc: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"{index_name}/_update/{doc_id}", {"doc": partial_doc})
