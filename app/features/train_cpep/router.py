import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.auth.auth import verify_api_key
from app.features.train_cpep.schema import TrainCpepRequest
from app.features.train_cpep.service import train_cpep_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Train CPEP"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/train-cpep")
async def train_cpep(request: TrainCpepRequest, background_tasks: BackgroundTasks):
    logger.info(
        "Train CPEP request | job_id=%s | user_id=%s | clips=%s",
        request.job_id, request.user_id, len(request.training_data),
    )
    background_tasks.add_task(train_cpep_model, request)
    return {"job_id": request.job_id, "accepted": True}