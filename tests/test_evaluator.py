import pytest

from src.evaluator import EvaluationReport, SimplificationEvaluator


def test_compute_flesch_returns_score_in_readability_range():
    evaluator = SimplificationEvaluator()

    score = evaluator.compute_flesch(
        "The cat sat on the mat. The dog ran."
    )

    assert 0.0 <= score <= 100.0


def test_compute_flesch_returns_zero_for_text_without_words():
    assert SimplificationEvaluator().compute_flesch("...?!") == 0.0


def test_compute_overlap_uses_non_stopword_jaccard_similarity():
    evaluator = SimplificationEvaluator()

    assert evaluator.compute_overlap("The API helps users.", "API helps people.") == 0.5
    assert evaluator.compute_overlap("the and", "a the") == 1.0


def test_evaluate_reports_improvement_and_compression():
    evaluator = SimplificationEvaluator()
    original = (
        "The utilization of computational methodologies facilitates "
        "the implementation of sophisticated operational procedures."
    )
    simplified = "Computers help people do complex tasks."

    report = evaluator.evaluate(original, simplified)

    assert isinstance(report, EvaluationReport)
    assert report.flesch_delta > 0
    assert report.compression_ratio < 1.0
    assert 0.0 <= report.lexical_overlap_score <= 1.0
    assert report.readability_verdict in {
        "Significantly Easier",
        "Moderately Easier",
        "Unchanged",
        "Harder",
    }


def test_benchmark_dataset_aggregates_mean_metrics():
    evaluator = SimplificationEvaluator()
    samples = [
        {
            "original_text": "The utilization of tools enables implementation.",
            "simplified_text": "Tools help people work.",
        },
        {
            "original_text": "The methodology provides additional functionality.",
            "simplified_text": "The method adds features.",
        },
    ]

    benchmark = evaluator.benchmark_dataset(samples)

    assert benchmark["sample_count"] == 2
    assert benchmark["mean_flesch_delta"] == pytest.approx(
        sum(evaluator.evaluate(**sample).flesch_delta for sample in samples) / 2
    )
    assert 0.0 <= benchmark["mean_lexical_overlap_score"] <= 1.0


def test_benchmark_dataset_empty_input_returns_zeroed_aggregate():
    benchmark = SimplificationEvaluator().benchmark_dataset([])

    assert benchmark["sample_count"] == 0
    assert benchmark["mean_flesch_delta"] == 0.0


def test_benchmark_rejects_samples_missing_required_text():
    with pytest.raises(ValueError, match="original_text and simplified_text"):
        SimplificationEvaluator().benchmark_dataset([{"original_text": "Text."}])
