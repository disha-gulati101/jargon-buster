import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationReport:
    original_flesch_score: float
    simplified_flesch_score: float
    flesch_delta: float
    compression_ratio: float
    lexical_overlap_score: float
    readability_verdict: str


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_SENTENCE_PATTERN = re.compile(r"[.!?]+")
_VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself just me more most must
    mustn't my myself no nor not of off on once only or other ought our ours
    ourselves out over own same she she'd she'll she's should shouldn't so
    some such than that that's the their theirs them themselves then there
    there's these they they'd they'll they're they've this those through to
    too under until up very was wasn't we we'd we'll we're we've were weren't
    what what's when when's where where's which while who who's whom why
    why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
    """.split()
)


class SimplificationEvaluator:
    def compute_flesch(self, text: str) -> float:
        """Calculate Flesch Reading Ease, clamped to its conventional 0–100 range."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        words = _TOKEN_PATTERN.findall(text)
        if not words:
            return 0.0

        sentences = max(1, len(_SENTENCE_PATTERN.findall(text)))
        syllables = sum(_count_syllables(word) for word in words)
        score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (
            syllables / len(words)
        )
        return round(min(max(score, 0.0), 100.0), 2)

    def compute_overlap(self, orig_text: str, simp_text: str) -> float:
        """Calculate Jaccard similarity between non-stopword token sets."""
        if not isinstance(orig_text, str) or not isinstance(simp_text, str):
            raise TypeError("texts must be strings")
        original = _content_words(orig_text)
        simplified = _content_words(simp_text)
        union = original | simplified
        if not union:
            return 1.0
        return round(len(original & simplified) / len(union), 4)

    def evaluate(self, original_text: str, simplified_text: str) -> EvaluationReport:
        original_tokens = _TOKEN_PATTERN.findall(original_text)
        simplified_tokens = _TOKEN_PATTERN.findall(simplified_text)
        original_score = self.compute_flesch(original_text)
        simplified_score = self.compute_flesch(simplified_text)
        delta = round(simplified_score - original_score, 2)
        compression = (
            round(len(simplified_tokens) / len(original_tokens), 4)
            if original_tokens
            else 0.0
        )

        if delta >= 20:
            verdict = "Significantly Easier"
        elif delta >= 5:
            verdict = "Moderately Easier"
        elif delta <= -5:
            verdict = "Harder"
        else:
            verdict = "Unchanged"

        return EvaluationReport(
            original_flesch_score=original_score,
            simplified_flesch_score=simplified_score,
            flesch_delta=delta,
            compression_ratio=compression,
            lexical_overlap_score=self.compute_overlap(
                original_text, simplified_text
            ),
            readability_verdict=verdict,
        )

    def benchmark_dataset(self, test_samples: list[dict]) -> dict:
        """Aggregate mean readability, compression, and overlap metrics.

        Each sample must contain ``original_text`` and ``simplified_text`` keys.
        """
        reports = []
        for index, sample in enumerate(test_samples):
            if not isinstance(sample, dict):
                raise TypeError(f"sample {index} must be a dictionary")
            try:
                original_text = sample["original_text"]
                simplified_text = sample["simplified_text"]
            except KeyError as error:
                raise ValueError(
                    f"sample {index} must include original_text and simplified_text"
                ) from error
            reports.append(self.evaluate(original_text, simplified_text))

        if not reports:
            return {
                "sample_count": 0,
                "mean_original_flesch_score": 0.0,
                "mean_simplified_flesch_score": 0.0,
                "mean_flesch_delta": 0.0,
                "mean_compression_ratio": 0.0,
                "mean_lexical_overlap_score": 0.0,
            }

        return {
            "sample_count": len(reports),
            "mean_original_flesch_score": _mean(
                report.original_flesch_score for report in reports
            ),
            "mean_simplified_flesch_score": _mean(
                report.simplified_flesch_score for report in reports
            ),
            "mean_flesch_delta": _mean(report.flesch_delta for report in reports),
            "mean_compression_ratio": _mean(
                report.compression_ratio for report in reports
            ),
            "mean_lexical_overlap_score": _mean(
                report.lexical_overlap_score for report in reports
            ),
        }


def _content_words(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOPWORDS
    }


def _count_syllables(word: str) -> int:
    normalized = word.casefold()
    groups = len(_VOWEL_GROUP_PATTERN.findall(normalized))
    if normalized.endswith("e") and groups > 1 and not normalized.endswith(("le", "ye")):
        groups -= 1
    return max(groups, 1)


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4)
