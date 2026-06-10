from __future__ import annotations

"""F6 entity and relation extraction.

The LLM extractor implements the paper's per-chunk extraction and gleaning loop.
The deterministic extractor exists so tests and pipeline smoke runs can execute
without spending model calls; production graph construction should use `llm`.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from signpost.llm.client import OpenAICompatibleClient


@dataclass(frozen=True)
class EntityRecord:
    name: str
    entity_type: str
    description: str


@dataclass(frozen=True)
class RelationRecord:
    source: str
    target: str
    description: str
    keywords: list[str]
    weight: float


@dataclass(frozen=True)
class ExtractionResult:
    entities: list[EntityRecord]
    relations: list[RelationRecord]


class SemanticExtractor(Protocol):
    def extract(self, chunk: dict[str, Any]) -> ExtractionResult:
        ...


class LLMSemanticExtractor:
    def __init__(self, client: OpenAICompatibleClient | None = None, gleaning_rounds: int = 2, retries: int = 3, retry_sleep: float = 2.0, timeout: int = 120):
        self.client = client or OpenAICompatibleClient(timeout=timeout)
        self.gleaning_rounds = gleaning_rounds
        self.retries = retries
        self.retry_sleep = retry_sleep

    def extract(self, chunk: dict[str, Any]) -> ExtractionResult:
        result = self._extract_once(chunk, previous=None)
        for _ in range(self.gleaning_rounds):
            supplement = self._extract_once(chunk, previous=result)
            if not supplement.entities and not supplement.relations:
                break
            result = _merge_results(result, supplement)
        return result

    def _extract_once(self, chunk: dict[str, Any], previous: ExtractionResult | None) -> ExtractionResult:
        previous_json = "" if previous is None else json.dumps(_result_to_json(previous), ensure_ascii=False)
        prompt = (
            "Extract entities and semantic relations from the chunk. Return strict JSON only.\n"
            "JSON schema:\n"
            "{\n"
            '  "entities": [{"name": "...", "type": "PERSON|ORG|CONCEPT|LAW|EVENT|PLACE|OTHER", "description": "..."}],\n'
            '  "relations": [{"source": "...", "target": "...", "description": "...", "keywords": ["..."], "weight": 1.0}]\n'
            "}\n"
            "Rules:\n"
            "- Entity names must be short canonical names.\n"
            "- Relation endpoints must use entity names.\n"
            "- Descriptions must be grounded in the chunk.\n"
            "- If previous extraction is provided, return only missing entities or relations.\n\n"
            f"Previous extraction:\n{previous_json}\n\n"
            f"Chunk id: {chunk.get('chunk_id')}\n"
            f"Source: {chunk.get('file_name')} lines {chunk.get('start_line')}-{chunk.get('end_line')}\n"
            f"Text:\n{chunk.get('content', '')}"
        )
        response = self._chat_with_retry(
            [
                {"role": "system", "content": "You are a precise knowledge graph extraction engine."},
                {"role": "user", "content": prompt},
            ],
            chunk=chunk,
        )
        return parse_extraction_response(response)

    def _chat_with_retry(self, messages: list[dict[str, str]], *, chunk: dict[str, Any]) -> str:
        attempts = max(1, self.retries + 1)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client.chat(messages)
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(
            "LLM semantic extraction failed after "
            f"{attempts} attempts: chunk_id={chunk.get('chunk_id')} "
            f"chars={len(str(chunk.get('content', '')))} tokens={chunk.get('metadata', {}).get('token_count')}"
        ) from last_exc


class DeterministicSemanticExtractor:
    """Local extractor for smoke tests.

    It recognizes repeated capitalized phrases, Chinese chapter terms, and a few
    domain keywords, then creates co-occurrence relations within a chunk.
    """

    _ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b|[\u4e00-\u9fff]{2,12}")

    def extract(self, chunk: dict[str, Any]) -> ExtractionResult:
        text = re.sub(r"\[CONTENT\]", " ", chunk.get("content", ""))
        candidates: list[str] = []
        for match in self._ENTITY_RE.finditer(text):
            value = match.group(0).strip(" .,:;!?()[]{}")
            if len(value) < 2 or value in {"CONTENT"}:
                continue
            if value not in candidates:
                candidates.append(value)
            if len(candidates) >= 8:
                break
        entities = [EntityRecord(name=item, entity_type=_guess_type(item), description=f"Mentioned in {chunk.get('chunk_id')}") for item in candidates]
        relations: list[RelationRecord] = []
        for source, target in zip(candidates, candidates[1:], strict=False):
            relations.append(RelationRecord(source=source, target=target, description=f"{source} co-occurs with {target} in this chunk.", keywords=["co_occurs"], weight=1.0))
        return ExtractionResult(entities=entities, relations=relations)


def parse_extraction_response(text: str) -> ExtractionResult:
    payload = _load_json_object(text)
    entities = []
    for row in payload.get("entities", []):
        if not isinstance(row, dict) or not row.get("name"):
            continue
        entities.append(
            EntityRecord(
                name=str(row.get("name", "")).strip(),
                entity_type=str(row.get("type") or row.get("entity_type") or "OTHER").strip() or "OTHER",
                description=str(row.get("description", "")).strip(),
            )
        )
    relations = []
    for row in payload.get("relations", []):
        if not isinstance(row, dict) or not row.get("source") or not row.get("target"):
            continue
        keywords = row.get("keywords") or row.get("relation_type") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        relations.append(
            RelationRecord(
                source=str(row.get("source", "")).strip(),
                target=str(row.get("target", "")).strip(),
                description=str(row.get("description", "")).strip(),
                keywords=[str(item).strip() for item in keywords if str(item).strip()],
                weight=float(row.get("weight", 1.0) or 1.0),
            )
        )
    return ExtractionResult(entities=entities, relations=relations)


def create_semantic_extractor(name: str, *, gleaning_rounds: int = 2, retries: int = 3, retry_sleep: float = 2.0, timeout: int = 120) -> SemanticExtractor:
    if name == "llm":
        return LLMSemanticExtractor(gleaning_rounds=gleaning_rounds, retries=retries, retry_sleep=retry_sleep, timeout=timeout)
    if name == "deterministic":
        return DeterministicSemanticExtractor()
    raise ValueError(f"Unknown semantic extractor: {name}")


def _load_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"entities": [], "relations": []}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    return data if isinstance(data, dict) else {"entities": [], "relations": []}


def _merge_results(left: ExtractionResult, right: ExtractionResult) -> ExtractionResult:
    entities = {(item.name.lower(), item.entity_type.lower()): item for item in left.entities}
    for item in right.entities:
        entities.setdefault((item.name.lower(), item.entity_type.lower()), item)
    relations = {(item.source.lower(), item.target.lower(), item.description.lower()): item for item in left.relations}
    for item in right.relations:
        relations.setdefault((item.source.lower(), item.target.lower(), item.description.lower()), item)
    return ExtractionResult(entities=list(entities.values()), relations=list(relations.values()))


def _result_to_json(result: ExtractionResult) -> dict[str, Any]:
    return {
        "entities": [item.__dict__ for item in result.entities],
        "relations": [item.__dict__ for item in result.relations],
    }


def extraction_result_to_dict(result: ExtractionResult) -> dict[str, Any]:
    return _result_to_json(result)


def extraction_result_from_dict(payload: dict[str, Any]) -> ExtractionResult:
    entities = [
        EntityRecord(
            name=str(item.get("name", "")).strip(),
            entity_type=str(item.get("entity_type") or item.get("type") or "OTHER").strip() or "OTHER",
            description=str(item.get("description", "")).strip(),
        )
        for item in payload.get("entities", [])
        if isinstance(item, dict) and item.get("name")
    ]
    relations = [
        RelationRecord(
            source=str(item.get("source", "")).strip(),
            target=str(item.get("target", "")).strip(),
            description=str(item.get("description", "")).strip(),
            keywords=[str(keyword).strip() for keyword in (item.get("keywords") or []) if str(keyword).strip()],
            weight=float(item.get("weight", 1.0) or 1.0),
        )
        for item in payload.get("relations", [])
        if isinstance(item, dict) and item.get("source") and item.get("target")
    ]
    return ExtractionResult(entities=entities, relations=relations)


def _guess_type(name: str) -> str:
    if re.search(r"cancer|signpost|graph|ppr|retrieval", name, flags=re.IGNORECASE):
        return "CONCEPT"
    if name.isupper() and len(name) <= 8:
        return "ORG"
    if re.search(r"章|节|条|款|图|结构|检索|推荐", name):
        return "CONCEPT"
    return "OTHER"
