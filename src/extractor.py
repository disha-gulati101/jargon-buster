from dataclasses import dataclass

import spacy
from spacy.language import Language


@dataclass(frozen=True)
class CandidateTerm:
    text: str
    start_char: int
    end_char: int


class JargonExtractor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp: Language = spacy.load(model)

    def extract(self, text: str) -> list[CandidateTerm]:
        doc = self.nlp(text)
        candidates: dict[tuple[int, int], CandidateTerm] = {}

        for chunk in doc.noun_chunks:
            if not any(not token.is_stop for token in chunk):
                continue
            candidates[(chunk.start_char, chunk.end_char)] = CandidateTerm(
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )

        for token in doc:
            if not token.is_alpha or not token.is_upper or token.is_stop:
                continue
            if len(token.text) < 2:
                continue
            candidates[(token.idx, token.idx + len(token.text))] = CandidateTerm(
                text=token.text,
                start_char=token.idx,
                end_char=token.idx + len(token.text),
            )

        return sorted(
            candidates.values(),
            key=lambda candidate: (candidate.start_char, candidate.end_char),
        )
