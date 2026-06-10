import json

from signpost.llm.client import LLMConfig, OpenAICompatibleClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({"data": [{"embedding": [1.0, 0.0]}]}).encode("utf-8")


def test_embedding_uses_embedding_endpoint_and_key(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        LLMConfig(
            api_base="https://chat.example/open/api/v1",
            embedding_api_base="https://embed.example/open/api/v1/embeddings",
            api_key="chat-key",
            embedding_api_key="embedding-key",
            chat_model="ecnu-plus",
            reasoning_model="ecnu-max",
            embedding_model="ecnu-embedding-small",
            rerank_model="ecnu-rerank",
        ),
        timeout=17,
    )

    assert client.embedding(["Signpost"]) == [[1.0, 0.0]]
    assert captured["url"] == "https://embed.example/open/api/v1/embeddings"
    assert captured["authorization"] == "Bearer embedding-key"
    assert captured["timeout"] == 17
