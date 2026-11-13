import logging

from fastapi import APIRouter, Depends

from app.auth.auth import verify_api_key
from app.features.trend_forecast.schema import (
    ExtractGenresRequest,
    ExtractGenresResponse,
    PredictTrendsRequest,
    PredictTrendsResponse,
)
from app.features.trend_forecast.service import (
    extract_genres,
    predict_trends,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Trend Forecast"],
    dependencies=[Depends(verify_api_key)],
)


# Plain `def` (not async) on purpose: Mongo and Groq calls here are
# blocking, so FastAPI runs them in its threadpool instead of freezing
# the event loop. No callback and no background task - the result is
# returned directly AND saved to shortsyou.trend_forecasts.


@router.post(
    "/extract-genres",
    response_model=ExtractGenresResponse,
)
def extract_genres_endpoint(request: ExtractGenresRequest):

    logger.info("Extract genres requested | user_id=%s", request.userId)

    return extract_genres(request.userId)


@router.post(
    "/predict-trends",
    response_model=PredictTrendsResponse,
)
def predict_trends_endpoint(request: PredictTrendsRequest):

    logger.info(
        "Predict trends requested | user_id=%s", request.userId
    )

    return predict_trends(request.userId)
