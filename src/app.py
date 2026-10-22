import html
import json
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from .domains import DOMAIN_LABELS
except ImportError:
    from domains import DOMAIN_LABELS

try:
    from .domains import glossary_entry
    from .evaluator import SimplificationEvaluator
except ImportError:
    from domains import glossary_entry
    from evaluator import SimplificationEvaluator


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


def _replacement_options(item: dict[str, Any], domain: str = "general") -> list[str]:
    term = str(item.get("term", ""))
    suggestions = [str(item.get("replacement_suggestion", "")).strip()]
    entry = glossary_entry(term, domain)
    if entry:
        suggestions.append(entry["replacement"])

    definition = str(item.get("plain_definition", "")).strip()
    definition_phrase = re.sub(r"^[Aa]n?\s+", "", definition).rstrip(".!?")
    definition_phrase = re.split(
        r"\b(?:that|which|who|because|when|before|after)\b",
        definition_phrase,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:")
    if definition_phrase:
        suggestions.append(definition_phrase)

    if len([suggestion for suggestion in suggestions if suggestion]) < 2:
        suggestions.append(f"a simpler explanation of {term}")
    if len([suggestion for suggestion in suggestions if suggestion]) < 2:
        suggestions.append("this concept")

    unique = []
    seen = set()
    for suggestion in suggestions:
        normalized = suggestion.casefold()
        if suggestion and normalized not in seen:
            unique.append(suggestion)
            seen.add(normalized)
    if len(unique) < 2:
        unique.append(f"plain-language {term}")
    return unique[:3]


def _selection_key(item: dict[str, Any]) -> str:
    return (
        f"replacement_{item.get('start', 0)}_{item.get('end', 0)}_"
        f"{item.get('term', '')}"
    )


def _rebuild_simplified_text(
    original_text: str,
    jargon_terms: list[dict[str, Any]],
    selected_replacements: dict[str, str],
) -> str:
    selected = []
    for item in sorted(
        jargon_terms,
        key=lambda candidate: (
            -float(candidate.get("score", 0.0)),
            -(int(candidate.get("end", 0)) - int(candidate.get("start", 0))),
            int(candidate.get("start", 0)),
        ),
    ):
        start, end = int(item.get("start", -1)), int(item.get("end", -1))
        if start < 0 or end < start or end > len(original_text):
            continue
        if all(
            end <= other_start or start >= other_end
            for other_start, other_end, _ in selected
        ):
            replacement = selected_replacements.get(
                _selection_key(item), str(item.get("replacement_suggestion", ""))
            )
            selected.append((start, end, replacement))

    for start, end, replacement in sorted(selected, reverse=True):
        original_text = original_text[:start] + replacement + original_text[end:]
    return original_text


def _build_export_result(
    result: dict[str, Any],
    selected_replacements: dict[str, str],
) -> dict[str, Any]:
    export_result = deepcopy(result)
    jargon_terms = export_result.get("jargon_terms", [])
    for item in jargon_terms:
        item["selected_replacement"] = selected_replacements.get(
            _selection_key(item), item.get("replacement_suggestion", "")
        )
    simplified_text = _rebuild_simplified_text(
        export_result.get("original_text", ""),
        jargon_terms,
        selected_replacements,
    )
    export_result["simplified_text"] = simplified_text
    evaluation = SimplificationEvaluator().evaluate(
        export_result.get("original_text", ""), simplified_text
    )
    export_result["reading_level_score"] = evaluation.simplified_flesch_score
    export_result["evaluation"] = asdict(evaluation)
    return export_result


def _mock_analyze(text: str, domain: str = "general") -> dict[str, Any]:
    words = _WORD_PATTERN.findall(text)
    return {
        "original_text": text,
        "jargon_count": 0,
        "reading_level_score": 0.0 if not words else 50.0,
        "jargon_terms": [],
        "simplified_text": text,
        "domain": domain,
        "_fallback": True,
    }


def _analyze_direct(
    text: str,
    threshold: float,
    session_state: Any,
    domain: str = "general",
) -> dict[str, Any]:
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
            return _mock_analyze(text, domain)

    try:
        return _as_dict(
            pipeline.analyze(text, threshold=threshold, domain=domain)
        )
    except (ImportError, OSError, RuntimeError):
        return _mock_analyze(text, domain)


def _analyze_api(
    text: str, threshold: float, domain: str = "general"
) -> dict[str, Any]:
    payload = json.dumps(
        {"text": text, "threshold": threshold, "domain": domain}
    ).encode("utf-8")
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
    selected_replacements = {}
    for item in jargon_terms:
        item_domain = str(item.get("domain", result.get("domain", "general")))
        options = _replacement_options(item, item_domain)
        key = _selection_key(item)
        current = st.session_state.get(key)
        if current not in options:
            current = options[0]
            st.session_state[key] = current
        selected_replacements[key] = current

    simplified_text = _rebuild_simplified_text(
        result.get("original_text", ""), jargon_terms, selected_replacements
    )
    export_result = _build_export_result(result, selected_replacements)
    evaluation = export_result.get("evaluation")
    flesch_delta = (
        float(evaluation.get("flesch_delta", 0.0))
        if isinstance(evaluation, dict)
        else 0.0
    )
    col_count, col_reading, col_status, col_shift = st.columns(4)
    col_count.metric("Total Jargon Count", result.get("jargon_count", len(jargon_terms)))
    col_reading.metric(
        "Reading Ease Score", export_result.get("reading_level_score", 0.0)
    )
    status = "Fallback mode (ML unavailable)" if result.get("_fallback") else "Complete"
    col_status.metric("Simplification Status", status)
    col_shift.metric(
        "Readability Shift",
        f"{flesch_delta:+.2f} points" if evaluation else "Unavailable",
        delta=f"{flesch_delta:+.2f}" if evaluation else None,
    )

    st.subheader("Identified jargon")
    st.markdown(
        _highlighted_text(result.get("original_text", ""), jargon_terms),
        unsafe_allow_html=True,
    )
    with st.expander("Jargon terms and explanations", expanded=bool(jargon_terms)):
        if jargon_terms:
            for item in jargon_terms:
                item_domain = str(item.get("domain", result.get("domain", "general")))
                domain_label = DOMAIN_LABELS.get(item_domain, item_domain.title())
                st.markdown(
                    f"**{item.get('term', '')}** · "
                    f"Confidence: {float(item.get('score', 0.0)):.0%} · "
                    f"<span style=\"background-color:#e8eefc; padding:2px 8px; "
                    f"border-radius:10px;\">{html.escape(domain_label)}</span>",
                    unsafe_allow_html=True,
                )
                st.write(item.get("plain_definition", ""))
                st.caption(
                    f"Suggested replacement: {item.get('replacement_suggestion', '')}"
                )
                options = _replacement_options(item, item_domain)
                selected = st.selectbox(
                    f"Choose replacement for {item.get('term', 'term')}",
                    options,
                    key=_selection_key(item),
                )
                selected_replacements[_selection_key(item)] = selected
        else:
            st.write("No jargon met the selected sensitivity threshold.")

    st.subheader("Simplified text")
    st.text_area(
        "Reconstructed text with jargon replaced",
        value=simplified_text,
        height=180,
        disabled=True,
    )
    summary = (
        f"Executive summary: {export_result.get('jargon_count', len(jargon_terms))} "
        f"jargon terms identified. Reading ease changed by "
        f"{float(evaluation.get('flesch_delta', 0.0)):+.2f} points "
        f"({evaluation.get('readability_verdict', 'Unchanged')})."
    )
    st.download_button(
        "Download Simplified Text (.txt)",
        data=f"{summary}\n\n{simplified_text}",
        file_name="jargon-buster-simplified.txt",
        mime="text/plain",
    )
    st.download_button(
        "Download Full Analysis Report (.json)",
        data=json.dumps(export_result, ensure_ascii=False, indent=2),
        file_name="jargon-buster-analysis.json",
        mime="application/json",
    )
    with st.expander("Quantitative NLP Evaluation & Benchmarking"):
        if evaluation:
            original_score = float(evaluation.get("original_flesch_score", 0.0))
            simplified_score = float(evaluation.get("simplified_flesch_score", 0.0))
            compression_ratio = float(evaluation.get("compression_ratio", 0.0))
            token_change = (compression_ratio - 1.0) * 100.0
            overlap_score = float(evaluation.get("lexical_overlap_score", 0.0))
            verdict = str(evaluation.get("readability_verdict", "Unchanged"))
            score_col, ratio_col, overlap_col = st.columns(3)
            score_col.write(
                f"**Flesch Reading Ease:** {original_score:.2f} → "
                f"{simplified_score:.2f}"
            )
            ratio_col.write(
                f"**Compression Ratio:** {compression_ratio:.1%} "
                f"({token_change:+.1f}% token length)"
            )
            overlap_col.write(f"**Lexical Retention:** {overlap_score:.1%}")
            st.markdown(f"**Readability Verdict:** `{html.escape(verdict)}`")
        else:
            st.write("Quantitative evaluation is unavailable for this result.")


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
    domain_label = st.sidebar.selectbox(
        "Domain Scope", ["General Tech", "Medical", "Legal", "Finance"]
    )
    domain = next(
        key for key, label in DOMAIN_LABELS.items() if label == domain_label
    )
    uploaded_file = st.file_uploader(
        "Load a text document",
        type=["txt"],
        help="Upload a .txt document to populate the text area.",
    )
    if "jargon_input_text" not in st.session_state:
        st.session_state["jargon_input_text"] = DEFAULT_TEXT
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        upload_key = (uploaded_file.name, len(upload_bytes), hash(upload_bytes))
        if st.session_state.get("jargon_uploaded_file_key") != upload_key:
            st.session_state["jargon_input_text"] = upload_bytes.decode(
                "utf-8-sig", errors="replace"
            )
            st.session_state["jargon_uploaded_file_key"] = upload_key

    text = st.text_area(
        "Paste technical paragraphs, legal text, or documentation",
        key="jargon_input_text",
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
                    result = _analyze_direct(
                        text, threshold, st.session_state, domain
                    )
                else:
                    result = _analyze_api(text, threshold, domain)
                st.session_state["jargon_analysis_result"] = result
            except RuntimeError as error:
                st.error(str(error))
                return

    result = st.session_state.get("jargon_analysis_result")
    if result is not None:
        _render_result(result)


if __name__ == "__main__":
    main()
