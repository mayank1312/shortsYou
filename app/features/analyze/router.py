import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.features.analyze.schema import AnalyzeRequest
from app.features.analyze.service import analyze_transcription
from app.auth.auth import verify_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[Depends(verify_api_key)],
)


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict:

    logger.info(
        "Analyze request received | job_id=%s | video_id=%s | segments=%d",
        request.job_id,
        request.video_id,
        len(request.segments),
    )

    background_tasks.add_task(
        analyze_transcription,
        request,
    )

    return {
        "job_id": request.job_id,
        "accepted": True,
    }