import html
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import streamlit as st
except ImportError:
    st = None


DEFAULT_TEXT = (
    "The API uses retrieval-augmented generation (RAG) to provide context-aware "
    "responses. This approach improves the accuracy of large language models "
    "(LLMs) by grounding their outputs in relevant documentation."
)
API_URL = "http://localhost:8000/api/v1/analyze"
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    raise TypeError("analysis result must be a dataclass, Pydantic model, or mapping")


def _highlighted_text(original_text: str, jargon_terms: list[dict[str, Any]]) -> str:
    selected = []
    for item in sorted(
        jargon_terms,
        key=lambda term: (
            -float(term.get("score", 0.0)),
            -(int(term["end"]) - int(term["start"])),
            int(term["start"]),
        ),
    ):
        start, end = int(item["start"]), int(item["end"])
        if start < 0 or end < start or end > len(original_text):
            continue
        if all(
            end <= other_start or start >= other_end
            for other_start, other_end, _ in selected
        ):
            selected.append((start, end, item))

    parts = []
    cursor = 0
    for start, end, item in sorted(selected):
        parts.append(html.escape(original_text[cursor:start]))
        term = html.escape(original_text[start:end])
        definition = html.escape(str(item.get("plain_definition", "")), quote=True)
        parts.append(f'<mark title="{definition}">{term}</mark>')
        cursor = end
    parts.append(html.escape(original_text[cursor:]))
    return (
        '<div style="white-space: pre-wrap; line-height: 1.7; padding: 1rem; '
        'border: 1px solid #ddd; border-radius: .5rem;">'
        + "".join(parts)
        + "</div>"
    )


def _mock_analyze(text: str) -> dict[str, Any]:
    words = _WORD_PATTERN.findall(text)
    return {
        "original_text": text,
        "jargon_count": 0,
        "reading_level_score": 0.0 if not words else 50.0,
        "jargon_terms": [],
        "simplified_text": text,
        "_fallback": True,
    }


def _analyze_direct(text: str, threshold: float, session_state: Any) -> dict[str, Any]:
    pipeline = session_state.get("jargon_pipeline")
    if pipeline is None:
        try:
            try:
                from .pipeline import JargonBusterPipeline
            except ImportError:
                from pipeline import JargonBusterPipeline

            pipeline = JargonBusterPipeline()
            session_state["jargon_pipeline"] = pipeline
        except (ImportError, OSError, RuntimeError):
            return _mock_analyze(text)

    try:
        return _as_dict(pipeline.analyze(text, threshold=threshold))
    except (ImportError, OSError, RuntimeError):
        return _mock_analyze(text)


def _analyze_api(text: str, threshold: float) -> dict[str, Any]:
    payload = json.dumps({"text": text, "threshold": threshold}).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Could not reach the API backend at http://localhost:8000."
        ) from error
    if not isinstance(result, dict):
        raise RuntimeError("API returned an invalid analysis response.")
    return result


def _render_result(result: dict[str, Any]) -> None:
    if st is None:
        return

    jargon_terms = result.get("jargon_terms", [])
    col_count, col_reading, col_status = st.columns(3)
    col_count.metric("Total Jargon Count", result.get("jargon_count", len(jargon_terms)))
    col_reading.metric("Reading Ease Score", result.get("reading_level_score", 0.0))
    status = "Fallback mode (ML unavailable)" if result.get("_fallback") else "Complete"
    col_status.metric("Simplification Status", status)

    st.subheader("Identified jargon")
    st.markdown(
        _highlighted_text(result.get("original_text", ""), jargon_terms),
        unsafe_allow_html=True,
    )
    with st.expander("Jargon terms and explanations", expanded=bool(jargon_terms)):
        if jargon_terms:
            for item in jargon_terms:
                st.markdown(
                    f"**{item.get('term', '')}** · "
                    f"Confidence: {float(item.get('score', 0.0)):.0%}"
                )
                st.write(item.get("plain_definition", ""))
                st.caption(
                    f"Suggested replacement: {item.get('replacement_suggestion', '')}"
                )
        else:
            st.write("No jargon met the selected sensitivity threshold.")

    st.subheader("Simplified text")
    st.text_area(
        "Reconstructed text with jargon replaced",
        value=result.get("simplified_text", ""),
        height=180,
        disabled=True,
    )


def main() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the project requirements to run the dashboard."
        )

    st.set_page_config(page_title="Jargon Buster", page_icon="📖", layout="wide")
    st.title("Jargon Buster")
    st.write("Make technical, legal, and documentation-heavy text easier to understand.")

    threshold = st.sidebar.slider(
        "Jargon Sensitivity Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.65,
        step=0.05,
    )
    mode = st.sidebar.radio(
        "Analysis mode",
        ("Direct Pipeline Mode", "API Backend Mode (http://localhost:8000)"),
    )
    text = st.text_area(
        "Paste technical paragraphs, legal text, or documentation",
        value=DEFAULT_TEXT,
        height=220,
        max_chars=10_000,
    )

    if st.button("Analyze Text", type="primary"):
        if not text.strip():
            st.error("Enter some text before running the analysis.")
            return
        with st.spinner("Analyzing text..."):
            try:
                if mode == "Direct Pipeline Mode":
                    result = _analyze_direct(text, threshold, st.session_state)
                else:
                    result = _analyze_api(text, threshold)
                st.session_state["jargon_analysis_result"] = result
            except RuntimeError as error:
                st.error(str(error))
                return

    result = st.session_state.get("jargon_analysis_result")
    if result is not None:
        _render_result(result)


if __name__ == "__main__":
    main()
