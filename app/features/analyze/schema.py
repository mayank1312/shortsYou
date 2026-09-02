from typing import List

from pydantic import BaseModel, Field

from app.features.transcribe.schema import (
    FillerWord,
    SilenceGap,
    TranscriptSegment,
)


class AnalyzeRequest(BaseModel):

    job_id: str

    video_id: str

    user_id: str

    segments: List[TranscriptSegment] = Field(
        default_factory=list
    )

    language: str = "unknown"

    fillerWords: List[FillerWord] = Field(
        default_factory=list
    )

    silenceGaps: List[SilenceGap] = Field(
        default_factory=list
    )

    # -----------------------------------------------------------
    # ASYNC CALLBACK FIELDS (optional)
    # -----------------------------------------------------------
    #
    # /analyze always responds immediately with {"accepted": true}
    # (it already ran as a background task before this change).
    # These fields let that background task actually notify Go
    # when it's done, instead of only writing to MongoDB.
    #
    # If callbackUrl is omitted, results are still saved to
    # MongoDB as before, but no callback is sent - useful for
    # manual testing without a live receiver.
    # -----------------------------------------------------------

    callbackUrl: str | None = None

    internalKey: str | None = None


class AnalyzeSegmentResult(BaseModel):

    index: int

    embedding: List[float] = Field(
        default_factory=list
    )

    semantic_score: float

    novelty_score: float

    clarity_score: float

    hook_score: float

    semantic_labels: List[str] = Field(
        default_factory=list
    )

    suggested_hook: str


class AnalyzeResponse(BaseModel):

    job_id: str

    video_id: str

    segments: List[AnalyzeSegmentResult] = Field(
        default_factory=list
    )