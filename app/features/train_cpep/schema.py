from typing import List
from pydantic import BaseModel, Field


class TrainingClip(BaseModel):
    clip_id: str
    emotion_score: float
    semantic_score: float
    hook_score: float
    duration_seconds: float
    views_48h: float


class TrainCpepRequest(BaseModel):
    job_id: str
    # a real creator id, or the sentinel "global-model" for the pooled model
    user_id: str
    # order assumed chronological (oldest first) for the temporal split
    training_data: List[TrainingClip] = Field(default_factory=list)


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class TrainCpepResponse(BaseModel):
    job_id: str | None = None
    user_id: str
    model_version: str = "v0"
    pearson_r: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    feature_importances: List[FeatureImportance] = Field(default_factory=list)
    n_samples: int = 0