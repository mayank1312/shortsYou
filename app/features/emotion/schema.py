from typing import List

from pydantic import BaseModel, Field


class EmotionSegmentRequest(BaseModel):

    index: int

    start: float

    end: float


class EmotionRequest(BaseModel):

    job_id: str

    video_id: str

    user_id: str | None = None

    audio_url: str

    segments: List[EmotionSegmentRequest] = Field(
        default_factory=list
    )

    # -----------------------------------------------------------
    # ASYNC CALLBACK FIELDS (optional)
    # -----------------------------------------------------------
    #
    # Same contract as /transcribe and /analyze: if callbackUrl is
    # provided, /emotion responds immediately with
    # {"job_id": ..., "accepted": true} and POSTs the result (or an
    # error) to callbackUrl once done. If omitted, results are only
    # saved to MongoDB - useful for manual testing.
    # -----------------------------------------------------------

    callbackUrl: str | None = None

    internalKey: str | None = None


class EmotionSegmentResult(BaseModel):

    index: int

    emotion_type: str

    emotion_score: float

    speech_emphasis_score: float

    speech_rate_wpm: float


class EmotionResponse(BaseModel):

    job_id: str

    video_id: str

    segments: List[EmotionSegmentResult] = Field(
        default_factory=list
    )
