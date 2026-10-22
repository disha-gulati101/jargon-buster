"""Plain-English glossaries for the supported Jargon Buster domains."""

DOMAIN_GLOSSARIES: dict[str, dict[str, dict[str, str]]] = {
    "general": {
        "api": {
            "definition": "A software interface that lets separate programs communicate.",
            "replacement": "software connection",
        },
        "rag": {
            "definition": "A method where AI retrieves information before generating an answer.",
            "replacement": "AI answers grounded in retrieved information",
        },
        "llm": {
            "definition": "An AI system trained on large amounts of text to generate language.",
            "replacement": "text-generating AI",
        },
        "gpu": {
            "definition": "A processor designed to handle many calculations at once.",
            "replacement": "graphics processor",
        },
        "cache": {
            "definition": "A temporary place to store data for faster access.",
            "replacement": "quick-access storage",
        },
        "latency": {
            "definition": "The delay before a system responds.",
            "replacement": "response delay",
        },
    },
    "medical": {
        "myocardial infarction": {
            "definition": "Damage to the heart muscle caused by blocked blood flow.",
            "replacement": "heart attack",
        },
        "hypertension": {
            "definition": "Blood pressure that stays higher than is healthy.",
            "replacement": "high blood pressure",
        },
        "idiopathic": {
            "definition": "Happening for a reason that is not known.",
            "replacement": "cause unknown",
        },
        "ambulation": {
            "definition": "The ability or act of walking.",
            "replacement": "walking",
        },
        "prognosis": {
            "definition": "A healthcare provider's estimate of how an illness may develop.",
            "replacement": "expected health outcome",
        },
    },
    "legal": {
        "indemnification": {
            "definition": "A promise to cover another party's losses or legal costs.",
            "replacement": "covering someone else's losses",
        },
        "force majeure": {
            "definition": "An extraordinary event that prevents someone from meeting a contract.",
            "replacement": "unavoidable extraordinary event",
        },
        "hereinafter": {
            "definition": "A term used to refer to someone or something by a name used later.",
            "replacement": "called from now on",
        },
        "tort": {
            "definition": "A wrongful act that causes harm and may lead to a legal claim.",
            "replacement": "civil wrong",
        },
        "liquidated damages": {
            "definition": "An amount agreed in advance as payment for breaking a contract.",
            "replacement": "pre-agreed contract penalty",
        },
    },
    "finance": {
        "ebitda": {
            "definition": "A measure of company earnings before interest, taxes, and certain costs.",
            "replacement": "operating earnings",
        },
        "amortization": {
            "definition": "Spreading a cost or loan payment over a period of time.",
            "replacement": "gradual cost repayment",
        },
        "liquidity": {
            "definition": "How easily an asset can be turned into cash.",
            "replacement": "ease of getting cash",
        },
        "capital expenditure": {
            "definition": "Money spent to buy or improve long-term business assets.",
            "replacement": "long-term business spending",
        },
        "cagr": {
            "definition": "The average yearly growth rate of an investment over time.",
            "replacement": "average annual growth",
        },
    },
}

DOMAIN_LABELS = {
    "general": "General Tech",
    "medical": "Medical",
    "legal": "Legal",
    "finance": "Finance",
}


def normalize_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    normalized = domain.strip().casefold()
    if normalized == "general tech":
        normalized = "general"
    if normalized not in DOMAIN_GLOSSARIES:
        supported = ", ".join(DOMAIN_GLOSSARIES)
        raise ValueError(f"unsupported domain {domain!r}; choose one of: {supported}")
    return normalized


def glossary_entry(term: str, domain: str = "general") -> dict[str, str] | None:
    glossary = DOMAIN_GLOSSARIES[normalize_domain(domain)]
    normalized_term = " ".join(term.casefold().split())
    return glossary.get(normalized_term)
