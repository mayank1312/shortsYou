import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.features.creator_dna.schema import CreatorDNARequest
from app.features.creator_dna.service import generate_creator_dna
from app.auth.auth import verify_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Creator DNA"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/generate-dna")
async def generate_dna_endpoint(
    request: CreatorDNARequest,
    background_tasks: BackgroundTasks,
):
    logger.info(
        "Creator DNA request received | job_id=%s | user_id=%s | "
        "videos=%d",
        request.job_id,
        request.user_id,
        len(request.all_transcript_texts),
    )

    background_tasks.add_task(
        generate_creator_dna,
        request,
    )

    return {
        "job_id": request.job_id,
        "accepted": True,
    }
