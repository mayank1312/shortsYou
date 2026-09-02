import logging

from fastapi import APIRouter, Depends, HTTPException

from app.features.transcribe.schema import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from app.features.transcribe.service import transcribe_video
from app.auth.auth import verify_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Transcription"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
)
async def transcribe(
    request: TranscriptionRequest,
):
    logger.info(
        "Transcription request received | video_id=%s | audio_url=%s",
        request.videoId,
        request.audioUrl,
    )

    try:
        result = await transcribe_video(
            video_id=request.videoId,
            audio_url=request.audioUrl,
        )

        logger.info(
            "Transcription request completed | video_id=%s | segments=%d | language=%s",
            request.videoId,
            len(result.get("segments", [])),
            result.get("language"),
        )

        return result

    except Exception as error:
        logger.exception(
            "Transcription request failed | video_id=%s",
            request.videoId,
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error