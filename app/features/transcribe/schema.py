from typing import List

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):

    word: str

    start: float

    end: float

    probability: float


class FillerWord(BaseModel):

    word: str

    start: float

    end: float

    probability: float | None = None


class SilenceGap(BaseModel):

    start: float

    end: float

    duration: float


class TranscriptSegment(BaseModel):

    index: int

    start: float

    end: float

    text: str

    words: List[WordTimestamp] = Field(
        default_factory=list
    )


class TranscriptionResponse(BaseModel):

    segments: List[TranscriptSegment]

    language: str

    fillerWords: List[FillerWord] = Field(
        default_factory=list
    )

    silenceGaps: List[SilenceGap] = Field(
        default_factory=list
    )


class TranscriptionRequest(BaseModel):

    videoId: str

    audioUrl: str

    userId: str | None = None

    language: str | None = None

    # -----------------------------------------------------------
    # ASYNC CALLBACK FIELDS (all optional)
    # -----------------------------------------------------------
    #
    # If job_id + callbackUrl are provided, /transcribe responds
    # immediately with {"job_id": ..., "accepted": true} and POSTs
    # the finished result (or an error) to callbackUrl once done.
    #
    # If callbackUrl is omitted, /transcribe falls back to the old
    # synchronous behavior - it blocks and returns the full
    # TranscriptionResponse directly. This keeps manual testing
    # (Postman, curl) working without needing a live callback
    # receiver.
    # -----------------------------------------------------------

    job_id: str | None = None

    callbackUrl: str | None = None

    internalKey: str | None = None


class TranscriptionAccepted(BaseModel):

    job_id: str | None

    accepted: bool = True