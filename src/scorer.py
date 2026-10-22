import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

try:
    from .domains import glossary_entry, normalize_domain
except ImportError:
    from domains import glossary_entry, normalize_domain

try:
    from wordfreq import zipf_frequency as _zipf_frequency
except ImportError:
    _COMMON_WORDS = {
        "about", "after", "and", "are", "because", "been", "before", "being",
        "between", "could", "each", "every", "from", "have", "into", "just",
        "more", "most", "other", "over", "said", "some", "such", "than",
        "that", "their", "them", "there", "these", "they", "this", "through",
        "time", "very", "was", "were", "what", "when", "which", "who", "will",
        "with", "would", "your",
    }

    def _zipf_frequency(word: str, language: str) -> float:
        normalized = word.casefold().strip()
        if normalized in _COMMON_WORDS:
            return 6.0
        if normalized.isalpha() and len(normalized) <= 4:
            return 4.5
        if " " in normalized or any(
            len(part) >= 7 for part in re.findall(r"[a-z]+", normalized)
        ):
            return 2.5
        return 3.5


_ACRONYM_PATTERN = re.compile(r"(?:[A-Z][A-Z0-9]{1,9})(?:[-/][A-Z0-9]{2,10})*")
_VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class ScoredCandidate:
    text: str
    start_char: int
    end_char: int
    score: float


class JargonScorer:
    def score(self, term: str, domain: str = "general") -> float:
        """Return a normalized jargon-likelihood score for a term."""
        domain = normalize_domain(domain)
        normalized = term.strip()
        if not normalized:
            return 0.0

        zipf = _zipf_frequency(normalized, "en")
        rarity = _clamp((5.5 - zipf) / 4.5)
        compact = re.sub(r"[^A-Za-z0-9]", "", normalized)
        length_signal = _clamp((len(compact) - 3) / 17)
        syllable_count = sum(
            _count_syllables(word) for word in _WORD_PATTERN.findall(normalized)
        )
        syllable_signal = _clamp((syllable_count - 1) / 5)
        is_compound = bool(re.search(r"[\s_/-]", normalized.strip()))

        result = (
            0.65 * rarity
            + 0.20 * length_signal
            + 0.10 * syllable_signal
            + (0.20 if is_compound else 0.0)
        )
        if _ACRONYM_PATTERN.fullmatch(normalized):
            result = max(result + 0.35, 0.85)
        if glossary_entry(normalized, domain) is not None:
            result = min(result + 0.25, 1.0)
        return round(_clamp(result), 4)

    def filter_jargon(
        self,
        candidates: Iterable[object],
        threshold: float = 0.65,
        domain: str = "general",
    ) -> list[ScoredCandidate]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        domain = normalize_domain(domain)

        jargon = []
        for candidate in candidates:
            text, start_char, end_char = _candidate_fields(candidate)
            calculated_score = self.score(text, domain=domain)
            if calculated_score >= threshold:
                jargon.append(
                    ScoredCandidate(
                        text=text,
                        start_char=start_char,
                        end_char=end_char,
                        score=calculated_score,
                    )
                )
        return jargon


def _candidate_fields(candidate: object) -> tuple[str, int, int]:
    if isinstance(candidate, str):
        text, start_char, end_char = candidate, 0, len(candidate)
    elif isinstance(candidate, Mapping):
        text = candidate.get("text")
        start_char = candidate.get("start_char", 0)
        end_char = candidate.get("end_char")
        if end_char is None and isinstance(text, str):
            end_char = start_char + len(text) if isinstance(start_char, int) else None
    else:
        text = getattr(candidate, "text", None)
        start_char = getattr(candidate, "start_char", 0)
        end_char = getattr(candidate, "end_char", None)
        if end_char is None and isinstance(text, str):
            end_char = start_char + len(text) if isinstance(start_char, int) else None

    if (
        not isinstance(text, str)
        or not isinstance(start_char, int)
        or not isinstance(end_char, int)
        or start_char < 0
        or end_char < start_char
    ):
        raise ValueError("candidate must have text and valid character spans")
    return text, start_char, end_char


def _count_syllables(word: str) -> int:
    normalized = word.casefold()
    groups = len(_VOWEL_GROUP_PATTERN.findall(normalized))
    if normalized.endswith("e") and groups > 1 and not normalized.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
