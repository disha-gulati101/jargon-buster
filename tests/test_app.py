from dataclasses import asdict

from src.app import (
    _build_export_result,
    _rebuild_simplified_text,
    _replacement_options,
)
from src.pipeline import JargonBusterResult, JargonItem


def test_replacement_options_include_primary_and_plain_language_alternative():
    options = _replacement_options(
        {
            "term": "API",
            "plain_definition": (
                "A software interface that lets separate programs communicate."
            ),
            "replacement_suggestion": "software connection",
        },
        "general",
    )

    assert 2 <= len(options) <= 3
    assert "software connection" in options
    assert len({option.casefold() for option in options}) == len(options)


def test_user_selected_replacements_rebuild_the_text_without_span_corruption():
    original = "Use the API before RAG."
    terms = [
        {
            "term": "API",
            "start": 8,
            "end": 11,
            "score": 0.9,
            "replacement_suggestion": "software connection",
        },
        {
            "term": "RAG",
            "start": 19,
            "end": 22,
            "score": 0.8,
            "replacement_suggestion": "retrieval method",
        },
    ]
    selected = {
        "replacement_8_11_API": "software interface",
        "replacement_19_22_RAG": "document search",
    }

    assert _rebuild_simplified_text(original, terms, selected) == (
        "Use the software interface before document search."
    )


def test_export_reflects_selected_replacements_and_full_evaluation():
    result = asdict(
        JargonBusterResult(
            original_text="The API helps.",
            jargon_count=1,
            reading_level_score=50.0,
            jargon_terms=[
                JargonItem(
                    term="API",
                    score=0.9,
                    start=4,
                    end=7,
                    plain_definition="A software interface.",
                    replacement_suggestion="software connection",
                )
            ],
            simplified_text="The software connection helps.",
        )
    )
    selected = {"replacement_4_7_API": "software interface"}

    export = _build_export_result(result, selected)

    assert export["simplified_text"] == "The software interface helps."
    assert export["jargon_terms"][0]["selected_replacement"] == "software interface"
    assert export["evaluation"]["flesch_delta"] == (
        export["evaluation"]["simplified_flesch_score"]
        - export["evaluation"]["original_flesch_score"]
    )
    assert result["jargon_terms"][0]["replacement_suggestion"] == "software connection"


def test_results_ui_renders_selectors_and_downloads(monkeypatch):
    import src.app as app

    result = {
        "original_text": "The API helps.",
        "jargon_count": 1,
        "reading_level_score": 50.0,
        "jargon_terms": [
            {
                "term": "API",
                "score": 0.9,
                "start": 4,
                "end": 7,
                "plain_definition": "A software interface.",
                "replacement_suggestion": "software connection",
                "domain": "general",
            }
        ],
        "simplified_text": "The software connection helps.",
        "domain": "general",
    }
    state = {"replacement_4_7_API": "software interface"}
    downloads = {}
    text_areas = {}

    class Column:
        def metric(self, *args, **kwargs):
            pass

        def write(self, *args, **kwargs):
            pass

    class FakeStreamlit:
        session_state = state

        def columns(self, count):
            return [Column() for _ in range(count)]

        def subheader(self, *args, **kwargs):
            pass

        def markdown(self, *args, **kwargs):
            pass

        def expander(self, *args, **kwargs):
            return _ContextManager()

        def write(self, *args, **kwargs):
            pass

        def caption(self, *args, **kwargs):
            pass

        def selectbox(self, label, options, key):
            assert len(options) >= 2
            return state[key]

        def text_area(self, label, value, **kwargs):
            text_areas[label] = value

        def download_button(self, label, data, **kwargs):
            downloads[label] = data

    class _ContextManager:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(app, "st", FakeStreamlit())
    app._render_result(result)

    assert text_areas["Reconstructed text with jargon replaced"] == (
        "The software interface helps."
    )
    assert "The software interface helps." in downloads[
        "Download Simplified Text (.txt)"
    ]
    assert '"selected_replacement": "software interface"' in downloads[
        "Download Full Analysis Report (.json)"
    ]
