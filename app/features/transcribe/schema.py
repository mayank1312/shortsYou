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

    language: str | None = None