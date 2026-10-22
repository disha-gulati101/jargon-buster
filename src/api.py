from dataclasses import asdict, dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .pipeline import JargonBusterPipeline, JargonBusterResult
except ImportError:
    from pipeline import JargonBusterPipeline, JargonBusterResult


class AnalyzeRequest(BaseModel):
    text: str = Field(max_length=10_000)
    threshold: float = 0.65


@dataclass(frozen=True)
class _MockPipelineResult:
    original_text: str
    jargon_count: int
    reading_level_score: float
    jargon_terms: list[Any]
    simplified_text: str


class _MockPipeline:
    def analyze(self, text: str, threshold: float = 0.65) -> _MockPipelineResult:
        return _MockPipelineResult(
            original_text=text,
            jargon_count=0,
            reading_level_score=0.0,
            jargon_terms=[],
            simplified_text=text,
        )


def create_app(pipeline: Any | None = None) -> FastAPI:
    application = FastAPI(title="Jargon Buster API")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if pipeline is None:
        try:
            pipeline = JargonBusterPipeline()
        except (ImportError, OSError, RuntimeError):
            pipeline = _MockPipeline()
    application.state.pipeline = pipeline

    @application.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "jargon-buster-api"}

    @application.post("/api/v1/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        if not 0.0 <= request.threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="threshold must be between 0.0 and 1.0",
            )
        result = application.state.pipeline.analyze(
            request.text, threshold=request.threshold
        )
        if isinstance(result, JargonBusterResult):
            return asdict(result)
        if hasattr(result, "__dataclass_fields__"):
            return asdict(result)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result

    return application


app = create_app()
