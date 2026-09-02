import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.features.transcribe.schema import (
    TranscriptionRequest,
)
from app.features.transcribe.service import (
    run_transcription_job,
    transcribe_video,
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
        "job_id=%s | video_id=%s | audio_url=%s | callback=%s",
        request.job_id,
        request.videoId,
        request.audioUrl,
        bool(request.callbackUrl),
    )

    # ---------------------------------------------------------
    # ASYNC PATH (callbackUrl provided)
    # ---------------------------------------------------------
    #
    # Respond immediately, do the work in the background, and
    # POST the result (or an error) to callbackUrl once done.
    # ---------------------------------------------------------

    if request.callbackUrl:

        background_tasks.add_task(
            run_transcription_job,
            request.job_id,
            request.videoId,
            request.userId,
            request.audioUrl,
            request.language,
            request.callbackUrl,
            request.internalKey,
        )

        return {
            "job_id": request.job_id,
            "accepted": True,
        }

    # ---------------------------------------------------------
    # SYNCHRONOUS FALLBACK (no callbackUrl)
    # ---------------------------------------------------------
    #
    # Kept for manual testing (Postman/curl) without needing a
    # live callback receiver. Blocks and returns the full result
    # directly, same as before.
    # ---------------------------------------------------------

    try:
        result = await transcribe_video(
            video_id=request.videoId,
            audio_url=request.audioUrl,
        )

        logger.info(
            "Synchronous transcription completed | "
            "video_id=%s | segments=%d | language=%s",
            request.videoId,
            len(result.get("segments", [])),
            result.get("language"),
        )

        return result

    except Exception as error:
        logger.exception(
            "Synchronous transcription failed | video_id=%s",
            request.videoId,
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
