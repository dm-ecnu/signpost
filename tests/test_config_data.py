from signpost.config.context import ExperimentContext
from signpost.config.settings import load_settings
from signpost.data.create_ultradomain_subset import create_ultradomain_subset
from signpost.data.validate import validate_dataset
from signpost.llm.client import load_llm_config
from signpost.parsing.io import write_jsonl


def test_experiment_context_defaults() -> None:
    context = ExperimentContext(namespace="mini", dataset_id="mini")
    assert context.resolved_output_dir().as_posix().endswith("outputs/mini/default")


def test_settings_load() -> None:
    settings = load_settings()
    assert settings.project_root.name == "signpost_re"
    assert "ECNU_CHAT_MODEL" in settings.env


def test_llm_config_env_overrides_dotenv(monkeypatch) -> None:
    monkeypatch.setenv("ECNU_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("ECNU_EMBEDDING_API_BASE", "https://embedding.example/v1")
    monkeypatch.setenv("ECNU_API_KEY", "env-key")
    monkeypatch.setenv("ECNU_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv("ECNU_EMBEDDING_MODEL", "env-embedding")

    config = load_llm_config()

    assert config.api_base == "https://env.example/v1"
    assert config.embedding_api_base == "https://embedding.example/v1"
    assert config.api_key == "env-key"
    assert config.embedding_api_key == "embedding-key"
    assert config.embedding_model == "env-embedding"


def test_validate_agriculture_dataset(tmp_path, monkeypatch) -> None:
    base = tmp_path / "datasets" / "processed" / "agriculture"
    base.mkdir(parents=True)
    write_jsonl(
        base / "raw_corpus.jsonl",
        [
            {
                "doc_id": "doc1",
                "file_name": "doc1.txt",
                "source_format": "text",
                "text": "Agriculture document.",
                "metadata": {"dataset": "agriculture"},
            }
        ],
    )
    write_jsonl(
        base / "questions.jsonl",
        [
            {
                "question_id": "q1",
                "question": "What is covered?",
                "answer": "Agriculture.",
                "doc_ids": ["doc1"],
                "metadata": {"dataset": "agriculture"},
            }
        ],
    )
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

    summary = validate_dataset("agriculture")
    assert summary["documents"] == 1
    assert summary["questions"] == 1
    assert summary["questions_with_doc_ids"] == 1


def test_create_ultradomain_subset_writes_raw_subset_before_f3(tmp_path) -> None:
    source = tmp_path / "raw"
    source.mkdir(parents=True)
    (source / "legal.jsonl").write_text(
        "\n".join(
            [
                '{"input":"Q1","context":"Doc one","context_id":"d1","answers":["A1"],"_id":"q1","label":"legal"}',
                '{"input":"Q2","context":"Doc two","context_id":"d2","answers":["A2"],"_id":"q2","label":"legal"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = create_ultradomain_subset(source_dataset="legal", target_dataset="legal_test", doc_ids=["d1"], raw_root=source)

    assert summary["questions"] == 1
    target_raw = (source / "legal_test.jsonl").read_text(encoding="utf-8")
    assert '"dataset":"legal_test"' in target_raw
    assert '"context_id":"d1"' in target_raw
