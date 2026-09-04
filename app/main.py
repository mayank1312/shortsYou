import logging

from fastapi import FastAPI

from app.features.transcribe.router import router as transcribe_router
from app.shared.application_logger import configure_logging
from app.features.analyze.router import router as analyze_router
from app.features.emotion.router import router as emotion_router



configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShortsYou NLP",
    description="ML intelligence service for ShortsYou",
    version="1.0.0",
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "shorts-nlp",
    }


app.include_router(
    transcribe_router,
    prefix="/api/v1",
)
app.include_router(
    analyze_router,
    prefix="/api/v1",
)
app.include_router(
    emotion_router,
    prefix="/api/v1",
)