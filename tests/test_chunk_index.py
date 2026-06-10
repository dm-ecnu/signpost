from signpost.indexing.chunk_schema import chunk_index_mapping, chunk_index_name, chunk_to_es_doc
from signpost.indexing.embedding import HashEmbeddingProvider


def test_chunk_index_name_is_namespace_scoped() -> None:
    assert chunk_index_name("Legal Run") == "signpost-legal-run-chunks"


def test_hash_embedding_dimensions() -> None:
    provider = HashEmbeddingProvider(dimensions=16)
    vector = provider.embed(["graph retrieval graph"])[0]
    assert len(vector) == 16
    assert any(value != 0 for value in vector)


def test_chunk_mapping_and_document_shape() -> None:
    mapping = chunk_index_mapping(16)
    assert mapping["mappings"]["properties"]["content_vector"]["dims"] == 16
    doc = chunk_to_es_doc(
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "file_name": "d.txt",
            "content": "hello",
            "start_line": 1,
            "end_line": 2,
            "section_path": ["A"],
            "metadata": {"chunk_index": 0, "token_count": 1},
        },
        namespace="mini",
        dataset_id="mini",
        vector=[0.0] * 16,
    )
    assert doc["id"] == "c1"
    assert doc["type"] == "chunk"
    assert doc["namespace"] == "mini"

