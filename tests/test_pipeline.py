import importlib
import re
import sys
import types
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass(frozen=True)
class Candidate:
    text: str
    start_char: int
    end_char: int


class _Token:
    def __init__(self, text: str, start: int) -> None:
        self.text = text
        self.idx = start
        self.is_stop = text.casefold() in {"the", "a", "an", "via", "of"}
        self.is_alpha = text.isalpha()
        self.is_upper = text.isupper()


class _Chunk:
    def __init__(self, text: str, start: int, end: int, tokens: list[_Token]) -> None:
        self.text = text
        self.start_char = start
        self.end_char = end
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class _Doc:
    def __init__(self, text: str) -> None:
        self._tokens = [
            _Token(match.group(), match.start())
            for match in re.finditer(r"\w+|[^\w\s]", text)
        ]
        chunks = []
        for phrase in ("The GPU", "the LLM", "CUDA cores", "the"):
            start = text.find(phrase)
            if start >= 0:
                end = start + len(phrase)
                tokens = [
                    token
                    for token in self._tokens
                    if start <= token.idx < end
                ]
                chunks.append(_Chunk(phrase, start, end, tokens))
        self.noun_chunks = chunks

    def __iter__(self):
        return iter(self._tokens)


@pytest.fixture
def extractor_module(monkeypatch):
    spacy_stub = types.ModuleType("spacy")
    spacy_stub.load = lambda model: (lambda text: _Doc(text))
    language_stub = types.ModuleType("spacy.language")
    language_stub.Language = object
    monkeypatch.setitem(sys.modules, "spacy", spacy_stub)
    monkeypatch.setitem(sys.modules, "spacy.language", language_stub)
    sys.modules.pop("src.extractor", None)
    module = importlib.import_module("src.extractor")
    yield module
    sys.modules.pop("src.extractor", None)


@pytest.fixture
def pipeline_components():
    from src.pipeline import JargonBusterPipeline
    from src.scorer import JargonScorer
    from src.simplifier import JargonSimplifier

    class Extractor:
        def extract(self, text):
            matches = (
                ("GPU", "graphics processor"),
                ("CUDA cores", "parallel processing units"),
                ("API", "software interface"),
            )
            return [
                Candidate(term, match.start(), match.end())
                for term, _ in matches
                if (match := re.search(re.escape(term), text))
            ]

    class Simplifier:
        def simplify_candidate(self, term, context):
            replacements = {
                "GPU": "graphics processor",
                "CUDA cores": "parallel processing units",
                "API": "software interface",
            }
            return types.SimpleNamespace(
                plain_definition=f"A simple explanation of {term}.",
                suggested_replacement=replacements[term],
            )

    return JargonBusterPipeline(Extractor(), JargonScorer(), Simplifier())


def test_extractor_finds_multiword_chunks_and_acronyms(extractor_module):
    extractor = extractor_module.JargonExtractor()
    text = "The GPU accelerates the LLM via CUDA cores"
    terms = extractor.extract(text)

    assert "CUDA cores" in [term.text for term in terms]
    assert {"GPU", "LLM", "CUDA"}.issubset({term.text for term in terms})
    assert all(text[start:end] == term for term, start, end in (
        (item.text, item.start_char, item.end_char) for item in terms
    ))


def test_extractor_omits_stopwords_and_punctuation(extractor_module):
    terms = extractor_module.JargonExtractor().extract("the !")

    assert [term.text for term in terms] == []


def test_scorer_filters_everyday_words():
    from src.scorer import JargonScorer

    scorer = JargonScorer()
    common = ["apple", "run", "table"]

    assert all(scorer.score(word) < 0.65 for word in common)
    assert scorer.filter_jargon(common, threshold=0.65) == []


@pytest.mark.parametrize("term", ["GPU", "LLM", "CUDA"])
def test_scorer_boosts_acronyms_above_threshold(term):
    from src.scorer import JargonScorer

    assert JargonScorer().score(term) > 0.65


def test_scorer_gives_rare_compound_a_high_score():
    from src.scorer import JargonScorer

    scorer = JargonScorer()

    assert scorer.score("xenobiotechnology") > 0.65


def test_domain_glossary_boosts_matching_terms_and_filters_by_domain():
    from src.scorer import JargonScorer

    scorer = JargonScorer()
    baseline = scorer.score("hypertension")
    medical_score = scorer.score("hypertension", domain="medical")

    assert medical_score == min(round(baseline + 0.25, 4), 1.0)
    assert medical_score > baseline
    assert scorer.score("hypertension", domain="general") == baseline


def test_simplifier_prefers_selected_domain_glossary(monkeypatch):
    import src.simplifier as module

    monkeypatch.setattr(
        module,
        "_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("glossary entries should bypass model inference")
        ),
    )
    result = module.JargonSimplifier().simplify_candidate(
        "hypertension", "The patient has hypertension.", domain="medical"
    )

    assert result.plain_definition == (
        "Blood pressure that stays higher than is healthy."
    )
    assert result.suggested_replacement == "high blood pressure"


def test_simplifier_offline_fallback_returns_definition_and_replacement():
    from src.simplifier import JargonSimplifier

    simplifier = JargonSimplifier()
    simplifier._model_unavailable = True
    result = simplifier.simplify_candidate("API", "The API connects services.")

    assert result.plain_definition
    assert result.suggested_replacement
    assert len(result.plain_definition.split()) < 20


def test_simplifier_model_mode_returns_definition_and_replacement(monkeypatch):
    import src.simplifier as module

    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return lambda prompt, **options: [
            {
                "generated_text": (
                    "Definition: A system that finds relevant documents. "
                    "Replacement: document search"
                )
            }
        ]

    monkeypatch.setattr(module, "_torch", object())
    monkeypatch.setattr(module, "_pipeline", fake_pipeline)
    result = module.JargonSimplifier().simplify_candidate(
        "retriever", "A retriever finds relevant documents."
    )

    assert result.plain_definition == "A system that finds relevant documents"
    assert result.suggested_replacement == "document search"
    assert calls[0][1]["model_kwargs"]["local_files_only"] is True


def test_simplifier_caches_resolved_definitions(monkeypatch):
    import src.simplifier as module

    calls = []

    def fake_pipeline(*args, **kwargs):
        return lambda prompt, **options: (
            calls.append(prompt)
            or [
                {
                    "generated_text": (
                        "Definition: A quick explanation. "
                        "Replacement: simple phrase"
                    )
                }
            ]
        )

    monkeypatch.setattr(module, "_torch", object())
    monkeypatch.setattr(module, "_pipeline", fake_pipeline)
    simplifier = module.JargonSimplifier()
    first = simplifier.simplify_candidate(
        "retriever", "A retriever retrieves data."
    )
    second = simplifier.simplify_candidate(
        "retriever", "A retriever retrieves data."
    )

    assert first is second
    assert len(calls) == 1


def test_pipeline_replaces_overlapping_spans_without_corruption(pipeline_components):
    text = "The GPU uses CUDA cores with an API."
    result = pipeline_components.analyze(text)

    assert result.original_text == text
    assert result.jargon_count == 3
    assert result.simplified_text == (
        "The graphics processor uses parallel processing units "
        "with an software interface."
    )
    assert 0.0 <= result.reading_level_score <= 100.0
    assert result.evaluation is not None
    assert result.evaluation.original_flesch_score >= 0.0
    assert result.evaluation.simplified_flesch_score == result.reading_level_score
    assert all(item.plain_definition for item in result.jargon_terms)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_pipeline_handles_empty_and_whitespace(pipeline_components, text):
    result = pipeline_components.analyze(text)

    assert result.jargon_count == 0
    assert result.jargon_terms == []
    assert result.simplified_text == text
    assert result.reading_level_score == 0.0


def test_pipeline_handles_text_with_zero_jargon():
    from src.pipeline import JargonBusterPipeline
    from src.scorer import JargonScorer

    class EmptyExtractor:
        def extract(self, text):
            return []

    class UnusedSimplifier:
        def simplify_candidate(self, term, context):
            raise AssertionError("no jargon should need simplification")

    text = "A short and ordinary sentence."
    result = JargonBusterPipeline(
        EmptyExtractor(), JargonScorer(), UnusedSimplifier()
    ).analyze(text)

    assert result.jargon_count == 0
    assert result.simplified_text == text
    assert result.reading_level_score > 0.0


def test_pipeline_forwards_domain_to_scorer_and_simplifier():
    from src.pipeline import JargonBusterPipeline

    calls = {}

    class Extractor:
        def extract(self, text):
            return [Candidate("hypertension", 0, len("hypertension"))]

    class Scorer:
        def filter_jargon(self, candidates, threshold, domain):
            calls["scorer_domain"] = domain
            return [
                types.SimpleNamespace(
                    text="hypertension",
                    start_char=0,
                    end_char=len("hypertension"),
                    score=0.9,
                )
            ]

    class Simplifier:
        def simplify(self, term, context, domain):
            calls["simplifier_domain"] = domain
            return types.SimpleNamespace(
                plain_definition="High blood pressure.",
                suggested_replacement="high blood pressure",
            )

    result = JargonBusterPipeline(Extractor(), Scorer(), Simplifier()).analyze(
        "hypertension", domain="medical"
    )

    assert calls == {"scorer_domain": "medical", "simplifier_domain": "medical"}
    assert result.jargon_terms[0].domain == "medical"
    assert result.simplified_text == "high blood pressure"


def test_pipeline_handles_text_longer_than_2000_characters(pipeline_components):
    text = "Ordinary explanatory text. " * 100
    result = pipeline_components.analyze(text)

    assert len(text) > 2_000
    assert result.original_text == text
    assert result.simplified_text == text
    assert result.jargon_count == 0


def test_api_health_endpoint():
    from src.api import create_app

    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "jargon-buster-api",
    }


def test_api_analyze_endpoint_accepts_valid_payload(pipeline_components):
    from src.api import create_app

    client = TestClient(create_app(pipeline_components))
    response = client.post(
        "/api/v1/analyze",
        json={"text": "The GPU uses an API.", "threshold": 0.65},
    )

    assert response.status_code == 200
    assert response.json()["jargon_count"] == 2
    assert response.json()["jargon_terms"][0]["term"] == "GPU"
    assert response.json()["evaluation"]["original_flesch_score"] >= 0.0
    assert "flesch_delta" in response.json()["evaluation"]
    assert response.json()["jargon_terms"][0]["domain"] == "general"


def test_api_accepts_domain_and_returns_domain_matched_jargon():
    from src.api import create_app
    from src.pipeline import JargonBusterPipeline
    from src.scorer import JargonScorer
    from src.simplifier import JargonSimplifier

    class MedicalExtractor:
        def extract(self, text):
            start = text.index("hypertension")
            return [Candidate("hypertension", start, start + len("hypertension"))]

    client = TestClient(
        create_app(
            JargonBusterPipeline(
                MedicalExtractor(), JargonScorer(), JargonSimplifier()
            )
        )
    )
    response = client.post(
        "/api/v1/analyze",
        json={"text": "hypertension", "domain": "medical"},
    )

    assert response.status_code == 200
    item = response.json()["jargon_terms"][0]
    assert item["domain"] == "medical"
    assert item["replacement_suggestion"] == "high blood pressure"


def test_api_rejects_unknown_domain():
    from src.api import create_app

    response = TestClient(create_app()).post(
        "/api/v1/analyze", json={"text": "some text", "domain": "unknown"}
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": " \n "},
        {"text": "Some text", "threshold": -0.01},
        {"text": "Some text", "threshold": 1.01},
    ],
)
def test_api_analyze_endpoint_rejects_invalid_values(payload):
    from src.api import create_app

    response = TestClient(create_app()).post("/api/v1/analyze", json=payload)

    assert response.status_code == 400


def test_api_analyze_endpoint_rejects_text_over_10000_characters():
    from src.api import create_app

    response = TestClient(create_app()).post(
        "/api/v1/analyze", json={"text": "x" * 10_001}
    )

    assert response.status_code == 422
