# Jargon Buster

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Not%20specified-lightgrey.svg)](#license)
[![Tests](https://img.shields.io/badge/Tests-44%20passing-brightgreen.svg)](#testing)

Jargon Buster identifies specialized terms in technical, medical, legal, and
financial writing, explains them in plain English, and lets readers replace
jargon interactively. Use it as a Streamlit dashboard or serve its analysis
pipeline through a FastAPI REST API.

## Architecture

```text
                         +-------------------+
                         | Input text / .txt |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | spaCy Extractor   |
                         | chunks + acronyms |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Wordfreq / Domain |
                         | Scorer            |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Simplifier        |
                         | glossary / model  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Evaluator         |
                         | Flesch / overlap  |
                         +---------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
        +-----------------------+     +-----------------------+
        | Streamlit Dashboard  |     | FastAPI REST API      |
        | direct pipeline      |     | /api/v1/analyze       |
        +-----------------------+     +-----------------------+
```

## Key features

- **Multi-domain jargon detection:** general technology, medical, legal, and
  finance glossaries provide domain-aware scoring, definitions, and
  plain-language replacements.
- **Quantitative evaluation:** compare original and simplified text using
  Flesch Reading Ease and its delta, token compression ratio, and non-stopword
  Jaccard lexical overlap.
- **Interactive UI:** choose alternative replacements for each detected term,
  upload `.txt` files, inspect highlighted terms, and download simplified text
  or a complete JSON analysis report.
- **Dual-mode serving:** run the Streamlit dashboard with the pipeline
  directly, or use the FastAPI backend for REST-based integrations.
- **Offline-tolerant simplification:** domain glossary and rule-based/WordNet
  fallbacks work when model libraries or locally cached model weights are
  unavailable. The default FLAN-T5 model is loaded with local-only weights.

## Setup and quickstart

Python 3.11 is recommended.

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies and the spaCy English model

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The transformer model `google/flan-t5-small` is optional. The simplifier only
uses locally available model weights; without them, it falls back to domain
glossaries, context rules, and WordNet when its corpus is installed.

### 3. Run the FastAPI service

From the repository root:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://localhost:8000`. Interactive API
documentation is at `http://localhost:8000/docs`.

### 4. Run the Streamlit dashboard

In a second terminal with the virtual environment activated:

```bash
python -m streamlit run src/app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.
Choose **Direct Pipeline Mode** to analyze in the dashboard process, or
**API Backend Mode** to use the service on port 8000.

## API reference

### `POST /api/v1/analyze`

Accepts text of up to 10,000 characters. Whitespace-only text, thresholds
outside `[0.0, 1.0]`, and unsupported domains are rejected with HTTP 400.
Requests exceeding the text-length limit fail request validation.

#### Request

```json
{
  "text": "The API uses retrieval-augmented generation to answer questions.",
  "threshold": 0.65,
  "domain": "general"
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `text` | string | required | Text to analyze; maximum 10,000 characters. |
| `threshold` | number | `0.65` | Minimum jargon confidence from `0.0` to `1.0`. |
| `domain` | string | `"general"` | Glossary scope: `general`, `medical`, `legal`, or `finance`. |

#### Response

```json
{
  "original_text": "The API uses retrieval-augmented generation to answer questions.",
  "jargon_count": 1,
  "reading_level_score": 65.4,
  "jargon_terms": [
    {
      "term": "API",
      "score": 1.0,
      "start": 4,
      "end": 7,
      "plain_definition": "A software interface that lets separate programs communicate.",
      "replacement_suggestion": "software connection",
      "domain": "general"
    }
  ],
  "simplified_text": "The software connection uses retrieval-augmented generation to answer questions.",
  "evaluation": {
    "original_flesch_score": 58.1,
    "simplified_flesch_score": 65.4,
    "flesch_delta": 7.3,
    "compression_ratio": 1.0,
    "lexical_overlap_score": 0.8,
    "readability_verdict": "Moderately Easier"
  },
  "domain": "general"
}
```

`jargon_terms` spans (`start`, `end`) are zero-based character offsets into
`original_text`, with `end` exclusive. Jargon confidence and lexical overlap
are between `0.0` and `1.0`; compression ratio is non-negative and can exceed
`1.0` when the simplified version uses more tokens. Flesch Reading Ease is
between `0` and `100`.

The response values above are illustrative; actual scores and extracted spans
depend on the input, selected domain, and installed NLP resources.

### `GET /api/v1/health`

Returns the service health status:

```json
{
  "status": "ok",
  "service": "jargon-buster-api"
}
```

## Testing

Install the project requirements (including `pytest`) and run from the
repository root:

```bash
python -m pytest -q
```

The tests cover extraction and scoring, glossary/domain behavior, online and
offline simplification paths, pipeline replacement and evaluation, API
validation and serialization, and dashboard interactions and exports.

## License

No license file or license declaration is currently included in this
repository. All rights are reserved by default unless the project owner
publishes a license.
