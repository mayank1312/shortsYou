import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.features.transcribe.schema import (
    TranscriptionRequest,
)
from app.features.transcribe.service import (
    run_transcription_job,
)
from app.auth.auth import verify_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Transcription"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/transcribe")
async def transcribe(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(
        "Transcription request received | "
        "job_id=%s | video_id=%s | audio_url=%s",
        request.job_id,
        request.videoId,
        request.audioUrl,
    )

    background_tasks.add_task(
        run_transcription_job,
        request.job_id,
        request.videoId,
        request.userId,
        request.audioUrl,
        request.language,
    )

    return {
        "job_id": request.job_id,
        "accepted": True,
    }
