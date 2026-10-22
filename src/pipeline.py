import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

try:
    from .domains import normalize_domain
    from .evaluator import EvaluationReport, SimplificationEvaluator
except ImportError:
    from domains import normalize_domain
    from evaluator import EvaluationReport, SimplificationEvaluator

if TYPE_CHECKING:
    from .extractor import JargonExtractor
    from .scorer import JargonScorer
    from .simplifier import JargonSimplifier


@dataclass(frozen=True)
class JargonItem:
    term: str
    score: float
    start: int
    end: int
    plain_definition: str
    replacement_suggestion: str
    domain: str = "general"


@dataclass(frozen=True)
class JargonBusterResult:
    original_text: str
    jargon_count: int
    reading_level_score: float
    jargon_terms: list[JargonItem]
    simplified_text: str
    evaluation: Optional[EvaluationReport] = None
    domain: str = "general"


_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
_SYLLABLE_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)


class JargonBusterPipeline:
    def __init__(
        self,
        extractor: "JargonExtractor | None" = None,
        scorer: "JargonScorer | None" = None,
        simplifier: "JargonSimplifier | None" = None,
    ) -> None:
        if extractor is None or scorer is None or simplifier is None:
            try:
                from .extractor import JargonExtractor
                from .scorer import JargonScorer
                from .simplifier import JargonSimplifier
            except ImportError:
                from extractor import JargonExtractor
                from scorer import JargonScorer
                from simplifier import JargonSimplifier

            if extractor is None:
                extractor = JargonExtractor()
            if scorer is None:
                scorer = JargonScorer()
            if simplifier is None:
                simplifier = JargonSimplifier()

        self.extractor = extractor
        self.scorer = scorer
        self.simplifier = simplifier
        self.evaluator = SimplificationEvaluator()

    def analyze(
        self, text: str, threshold: float = 0.5, domain: str = "general"
    ) -> JargonBusterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        domain = normalize_domain(domain)

        candidates = self.extractor.extract(text)
        scored_candidates = self.scorer.filter_jargon(
            candidates, threshold=threshold, domain=domain
        )
        jargon_terms = []
        for candidate in scored_candidates:
            start = candidate.start_char
            end = candidate.end_char
            if start < 0 or end < start or end > len(text):
                raise ValueError("candidate span is outside the input text")

            context = _sentence_context(text, start, end)
            simplify = getattr(self.simplifier, "simplify", None)
            if callable(simplify):
                simplified = simplify(candidate.text, context, domain=domain)
            else:
                simplify_candidate = self.simplifier.simplify_candidate
                if domain == "general":
                    simplified = simplify_candidate(candidate.text, context)
                else:
                    simplified = simplify_candidate(
                        candidate.text, context, domain=domain
                    )
            jargon_terms.append(
                JargonItem(
                    term=candidate.text,
                    score=candidate.score,
                    start=start,
                    end=end,
                    plain_definition=simplified.plain_definition,
                    replacement_suggestion=simplified.suggested_replacement,
                    domain=domain,
                )
            )

        simplified_text = _replace_non_overlapping(text, jargon_terms)
        evaluation = self.evaluator.evaluate(text, simplified_text)
        return JargonBusterResult(
            original_text=text,
            jargon_count=len(jargon_terms),
            reading_level_score=_reading_ease(simplified_text),
            jargon_terms=jargon_terms,
            simplified_text=simplified_text,
            evaluation=evaluation,
            domain=domain,
        )


def _sentence_context(text: str, start: int, end: int) -> str:
    left = max(
        (text.rfind(mark, 0, start) for mark in ".!?"),
        default=-1,
    ) + 1
    boundaries = [
        position
        for mark in ".!?"
        if (position := text.find(mark, end)) != -1
    ]
    right = min(boundaries) + 1 if boundaries else len(text)
    return text[left:right].strip()


def _replace_non_overlapping(text: str, items: list[JargonItem]) -> str:
    selected: list[JargonItem] = []
    for item in sorted(
        items,
        key=lambda candidate: (
            -candidate.score,
            -(candidate.end - candidate.start),
            candidate.start,
        ),
    ):
        if all(
            item.end <= chosen.start or item.start >= chosen.end
            for chosen in selected
        ):
            selected.append(item)

    for item in sorted(selected, key=lambda candidate: candidate.start, reverse=True):
        text = (
            text[: item.start]
            + item.replacement_suggestion
            + text[item.end :]
        )
    return text


def _reading_ease(text: str) -> float:
    words = _WORD_PATTERN.findall(text)
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syllables = sum(_count_syllables(word) for word in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (
        syllables / len(words)
    )
    return round(min(max(score, 0.0), 100.0), 2)


def _count_syllables(word: str) -> int:
    normalized = word.casefold()
    groups = len(_SYLLABLE_PATTERN.findall(normalized))
    if normalized.endswith("e") and groups > 1 and not normalized.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)
