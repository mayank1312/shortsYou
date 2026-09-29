import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.features.creator_dna.schema import (
    CreatorDNARequest,
)
from app.features.creator_dna.service import (
    generate_creator_dna,
    generate_creator_dna_sync,
)
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
        "Creator DNA request received | user_id=%s",
        request.userId,
    )

    background_tasks.add_task(
        generate_creator_dna,
        request.userId,
    )

    return {
        "accepted": True,
        "userId": request.userId,
    }


@router.post("/generate-dna/sync")
async def generate_dna_sync_endpoint(
    request: CreatorDNARequest,
    background_tasks: BackgroundTasks,
):
    """
    Only userId is required here. We fetch the creator's own latest
    transcripts and their existing DNA profile ourselves, then blend
    the new result into it rather than requiring Abbas to gather and
    send anything.
    """

    logger.info(
        "Creator DNA sync requested | userId=%s",
        request.userId,
    )

    background_tasks.add_task(
        generate_creator_dna_sync,
        request.userId,
    )

    return {
        "accepted": True,
        "userId": request.userId,
    }
