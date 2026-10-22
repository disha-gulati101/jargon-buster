import re
from collections import OrderedDict
from dataclasses import dataclass

try:
    import torch as _torch
except (ImportError, OSError, RuntimeError):
    _torch = None

try:
    from transformers import pipeline as _pipeline
except (ImportError, OSError, RuntimeError):
    _pipeline = None

try:
    from nltk.corpus import wordnet as _wordnet
except (ImportError, OSError, RuntimeError):
    _wordnet = None


@dataclass(frozen=True)
class SimplificationResult:
    term: str
    plain_definition: str
    suggested_replacement: str


_KNOWN_TERMS = {
    "api": (
        "A software interface that lets separate programs communicate.",
        "software connection",
    ),
    "ebitda": (
        "A measure of company earnings before interest, taxes, and certain costs.",
        "operating earnings",
    ),
    "gpu": ("A processor designed to handle many calculations at once.", "graphics processor"),
    "llm": ("An AI system trained on large amounts of text to generate language.", "text-generating AI"),
    "machine learning": (
        "A way for computers to learn patterns from examples instead of fixed rules.",
        "computer learning",
    ),
    "nlp": ("Technology that helps computers work with human language.", "language technology"),
    "rag": (
        "A method where AI retrieves information before generating an answer.",
        "AI answers grounded in retrieved information",
    ),
}
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
_DEFINITION_PATTERN = re.compile(
    r"\b(?:means|refers to|is defined as)\s+(.+?)(?=[.!?;]|$)", re.IGNORECASE
)


class JargonSimplifier:
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        cache_size: int = 256,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.model_name = model_name
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], SimplificationResult] = OrderedDict()
        self._generator = None
        self._model_unavailable = _torch is None or _pipeline is None

    def simplify_candidate(self, term: str, context: str) -> SimplificationResult:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("term must be a non-empty string")
        if not isinstance(context, str):
            raise TypeError("context must be a string")

        normalized_term = term.strip()
        key = (normalized_term.casefold(), context.strip().casefold())
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        result = self._generate_definition(normalized_term, context)
        if result is None:
            result = self._fallback_definition(normalized_term, context)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _generate_definition(
        self, term: str, context: str
    ) -> SimplificationResult | None:
        if self._model_unavailable:
            return None

        if self._generator is None:
            try:
                self._generator = _pipeline(
                    "text2text-generation",
                    model=self.model_name,
                    model_kwargs={"local_files_only": True},
                    tokenizer_kwargs={"local_files_only": True},
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                self._model_unavailable = True
                return None

        prompt = (
            f"In the context '{context}', explain '{term}' in plain English "
            "using fewer than 20 words. Then give a short everyday replacement. "
            "Format exactly: Definition: ... Replacement: ..."
        )
        try:
            generated = self._generator(prompt, max_new_tokens=64, do_sample=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._model_unavailable = True
            return None

        if not generated or not isinstance(generated[0], dict):
            return None
        output = generated[0].get("generated_text")
        if not isinstance(output, str):
            return None

        definition_match = re.search(
            r"Definition:\s*(.*?)(?=\s*Replacement:|$)", output, re.IGNORECASE | re.DOTALL
        )
        replacement_match = re.search(
            r"Replacement:\s*(.*)", output, re.IGNORECASE | re.DOTALL
        )
        definition = _limit_words(
            definition_match.group(1) if definition_match else output
        )
        replacement = (
            _limit_words(replacement_match.group(1))
            if replacement_match
            else ""
        )
        if not definition or not replacement:
            return None
        return SimplificationResult(term, definition, replacement)

    def _fallback_definition(self, term: str, context: str) -> SimplificationResult:
        known = _KNOWN_TERMS.get(term.casefold())
        if known is not None:
            return SimplificationResult(term, known[0], known[1])

        contextual_definition = _definition_from_context(term, context)
        if contextual_definition:
            return SimplificationResult(
                term,
                _limit_words(contextual_definition),
                _fallback_replacement(term),
            )

        wordnet_result = _definition_from_wordnet(term)
        if wordnet_result is not None:
            return SimplificationResult(
                term,
                _limit_words(wordnet_result[0]),
                wordnet_result[1],
            )

        short_term = _limit_words(term, 12)
        return SimplificationResult(
            term,
            f"A specialized idea or method called {short_term}.",
            _fallback_replacement(term),
        )


def _definition_from_context(term: str, context: str) -> str | None:
    if not context:
        return None
    escaped_term = re.escape(term.strip())
    match = re.search(
        rf"\b{escaped_term}\b\s+{_DEFINITION_PATTERN.pattern}",
        context,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _definition_from_wordnet(term: str) -> tuple[str, str] | None:
    if _wordnet is None:
        return None
    lookup = term.replace(" ", "_")
    try:
        synsets = _wordnet.synsets(lookup)
    except (LookupError, OSError):
        return None
    for synset in synsets:
        definition = synset.definition()
        lemmas = [
            lemma.name().replace("_", " ")
            for lemma in synset.lemmas()
            if lemma.name().casefold() != lookup.casefold()
        ]
        replacement = lemmas[0] if lemmas else _fallback_replacement(term)
        if definition:
            return definition, replacement
    return None


def _fallback_replacement(term: str) -> str:
    return "simpler explanation"


def _limit_words(text: str, limit: int = 19) -> str:
    words = _WORD_PATTERN.findall(text.strip())
    return " ".join(words[:limit]).rstrip(".,;:!?")
