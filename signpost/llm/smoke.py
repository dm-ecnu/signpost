from __future__ import annotations

"""F1 smoke CLI.

Without flags this command checks whether model names and endpoint settings are
loaded.  Real API calls are opt-in so local structural tests do not require a key.
"""

import argparse
import json

from signpost.llm.client import OpenAICompatibleClient, load_llm_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test F1 LLM client")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    config = load_llm_config()
    payload: dict[str, object] = {
        "api_base": config.api_base,
        "embedding_api_base": config.embedding_api_base or config.api_base,
        "has_api_key": bool(config.api_key and "replace_with" not in config.api_key),
        "has_embedding_api_key": bool((config.embedding_api_key or config.api_key) and "replace_with" not in (config.embedding_api_key or config.api_key)),
        "chat_model": config.chat_model,
        "reasoning_model": config.reasoning_model,
        "embedding_model": config.embedding_model,
        "rerank_model": config.rerank_model,
    }
    client = OpenAICompatibleClient(config)
    if args.chat:
        payload["chat"] = client.chat([{"role": "user", "content": "Reply with OK."}])
    if args.embedding:
        payload["embedding_dimensions"] = len(client.embedding(["Signpost"])[0])
    if args.rerank:
        payload["rerank_scores"] = client.rerank("graph retrieval", ["graph retrieval", "unrelated text"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
