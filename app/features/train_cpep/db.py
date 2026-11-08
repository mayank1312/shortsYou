import logging
from datetime import datetime, timezone

from app.db.database import get_database

logger = logging.getLogger(__name__)

FEATURE_ORDER = ["emotion_score", "semantic_score", "hook_score", "duration_seconds"]


class CpepModels:
    def __init__(self) -> None:
        self.collection = get_database()["cpep_models"]

    def get_latest_version_number(self, user_id: str) -> int:
        doc = self.collection.find_one(
            {"userId": user_id}, sort=[("versionNumber", -1)]
        )
        return int(doc["versionNumber"]) if doc else 0

    def save_model(
        self,
        job_id: str,
        user_id: str,
        version_number: int,
        model_json: str,
        pearson_r: float,
        rmse: float,
        r2: float,
        feature_importances: list,
        n_samples: int,
    ) -> None:
        self.collection.insert_one(
            {
                "jobId": job_id,
                "userId": user_id,
                "versionNumber": version_number,
                "modelVersion": f"v{version_number}",
                "modelJson": model_json,
                "featureOrder": FEATURE_ORDER,
                "pearsonR": pearson_r,
                "rmse": rmse,
                "r2": r2,
                "featureImportances": feature_importances,
                "nSamples": n_samples,
                "createdAt": datetime.now(timezone.utc),
            }
        )
        logger.info(
            "CPEP model saved | user_id=%s | version=v%s | n=%s",
            user_id, version_number, n_samples,
        )


cpep_models_db = CpepModels()