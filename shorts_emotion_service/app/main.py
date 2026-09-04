import logging

from fastapi import FastAPI

from app.shared.application_logger import configure_logging
from app.features.emotion.router import router as emotion_router


configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShortsYou Emotion",
    description="Emotion detection service for ShortsYou",
    version="1.0.0",
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    logger.info("Health check requested")

    return {
        "status": "ok",
        "service": "shorts-emotion",
    }


app.include_router(
    emotion_router,
    prefix="/api/v1",
)
