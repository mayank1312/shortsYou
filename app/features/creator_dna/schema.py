from typing import List

from pydantic import BaseModel, Field


class CreatorDNARequest(BaseModel):

    userId: str


class TopicWeight(BaseModel):

    topic: int

    label: str

    weight: float


class EmotionalRangeStats(BaseModel):

    mean: float = 0.0

    std: float = 0.0

    dominant: str = "unknown"


class CreatorDNAResponse(BaseModel):

    job_id: str | None = None

    user_id: str

    topic_distribution: List[TopicWeight] = Field(
        default_factory=list
    )

    # 6 dims, 0-1 scale: topicDiversity, emotionalRange, speechPace,
    # contentDepth, hookStrength, evergreenRatio - matches the radar
    # chart axes from the project doc (VIZ-02).
    dna_vector: List[float] = Field(
        default_factory=list
    )

    top_vocabulary: List[str] = Field(
        default_factory=list
    )

    emotional_range: EmotionalRangeStats = Field(
        default_factory=EmotionalRangeStats
    )

    avg_speech_rate: float = 0.0

    # Requires the DistilBERT evergreen classifier (Sammnyu's
    # dataset) which does not exist in the codebase yet - always 0.0
    # for now, not a fabricated number.
    evergreens_ratio: float = 0.0

    topic_coherence_score: float = 0.0
