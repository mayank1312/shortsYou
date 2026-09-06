import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.features.emotion.schema import EmotionRequest
from app.features.emotion.service import run_emotion_job
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
        "segments=%d",
        request.job_id,
        request.video_id,
        len(request.segments),
    )

    segments = [
        segment.model_dump()
        for segment in request.segments
    ]

    background_tasks.add_task(
        run_emotion_job,
        request.job_id,
        request.video_id,
        request.user_id,
        request.audio_url,
        segments,
    )

    return {
        "job_id": request.job_id,
        "accepted": True,
    }
