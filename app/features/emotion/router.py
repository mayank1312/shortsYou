import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.features.emotion.schema import EmotionRequest
from app.features.emotion.service import detect_emotions, run_emotion_job
from app.auth.auth import verify_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Emotion"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/emotion")
async def emotion_endpoint(
    request: EmotionRequest,
    background_tasks: BackgroundTasks,
):
    logger.info(
        "Emotion request received | job_id=%s | video_id=%s | "
        "segments=%d | callback=%s",
        request.job_id,
        request.video_id,
        len(request.segments),
        bool(request.callbackUrl),
    )

    segments = [
        segment.model_dump()
        for segment in request.segments
    ]

    # ---------------------------------------------------------
    # ASYNC PATH (callbackUrl provided)
    # ---------------------------------------------------------

    if request.callbackUrl:

        background_tasks.add_task(
            run_emotion_job,
            request.job_id,
            request.video_id,
            request.user_id,
            request.audio_url,
            segments,
            request.callbackUrl,
            request.internalKey,
        )

        return {
            "job_id": request.job_id,
            "accepted": True,
        }

    # ---------------------------------------------------------
    # SYNCHRONOUS FALLBACK (no callbackUrl) - manual testing only
    # ---------------------------------------------------------

    try:
        results = await detect_emotions(
            video_id=request.video_id,
            audio_url=request.audio_url,
            segments=segments,
        )

        return {
            "job_id": request.job_id,
            "video_id": request.video_id,
            "segments": results,
        }

    except Exception as error:
        logger.exception(
            "Synchronous emotion detection failed | video_id=%s",
            request.video_id,
        )

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
