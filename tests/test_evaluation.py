import json

from signpost.evaluation.metrics import evaluate_rows, extract_answer_from_prediction
from signpost.evaluation.schema import build_prediction_text, normalize_prediction_record, validate_prediction_record
from signpost.evaluation.validate_predictions import validate_predictions
from signpost.parsing.io import write_jsonl


def test_normalize_agent_row_to_f16_schema() -> None:
    row = {
        "id": "q1",
        "question": "What?",
        "gold_answer": "Gold",
        "answer": "Generated",
        "total_tokens": 17,
        "latency_seconds": 1.25,
        "retrieved_chunks": [{"chunk_id": "c1"}],
        "metadata": {"dataset": "mini"},
    }
    normalized = normalize_prediction_record(row)

    assert normalized["question_id"] == "q1"
    assert normalized["answer"] == "Gold"
    assert "<answer>" in normalized["prediction"]
    assert normalized["metadata"] == {"dataset": "mini", "method": "signpost"}
    assert normalized["total_tokens"] == 17
    assert normalized["latency_seconds"] == 1.25
    assert normalized["retrieved_chunks"] == [{"chunk_id": "c1"}]


def test_validate_prediction_record_reports_missing_metadata() -> None:
    issues = validate_prediction_record({"question_id": "q1", "question": "Q", "answer": "A", "prediction": "P", "metadata": {}}, 1)
    assert {issue.field for issue in issues} == {"metadata.method", "metadata.dataset"}


def test_extract_answer_and_basic_metrics() -> None:
    prediction = build_prediction_text(answer="The quick fox", rationale="because")
    assert extract_answer_from_prediction(prediction) == "The quick fox"

    result = evaluate_rows([{"question_id": "q1", "answer": "quick fox", "prediction": prediction, "metadata": {"method": "m", "dataset": "d"}}])
    assert result["num_scored"] == 1
    assert result["metrics"]["f1"] > 0.0


def test_validate_predictions_can_normalize_and_write_output(tmp_path) -> None:
    source = tmp_path / "agent.jsonl"
    output = tmp_path / "predictions.jsonl"
    write_jsonl(source, [{"id": "q1", "question": "Q", "gold_answer": "A", "answer": "P", "dataset": "mini"}])

    result = validate_predictions(source, normalize=True, output_path=output, default_dataset="mini")

    assert result["valid"] is True
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["question_id"] == "q1"
    assert row["metadata"]["dataset"] == "mini"
