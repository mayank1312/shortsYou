from typing import List

from pydantic import BaseModel, Field


class ExtractGenresRequest(BaseModel):

    userId: str


class PredictTrendsRequest(BaseModel):

    userId: str


class GenreResult(BaseModel):

    name: str

    confidence: float = 0.0


class ExtractGenresResponse(BaseModel):

    userId: str

    genres: List[GenreResult] = Field(default_factory=list)

    clipsAnalyzed: int = 0

    videosAnalyzed: int = 0


class Trend(BaseModel):

    # Always ONE of the requested genres - trends are never combined.
    genre: str

    name: str

    whyTrending: str = ""

    momentum: str = "rising"  # emerging | rising | peaking

    keywords: List[str] = Field(default_factory=list)


class VideoIdea(BaseModel):

    title: str

    hook: str = ""

    # Ideas CAN blend several genres / trends (e.g. gaming + tech).
    genres: List[str] = Field(default_factory=list)

    trends: List[str] = Field(default_factory=list)


class PredictTrendsResponse(BaseModel):

    userId: str

    genres: List[str] = Field(default_factory=list)

    trends: List[Trend] = Field(default_factory=list)

    videoIdeas: List[VideoIdea] = Field(default_factory=list)

    usedWebSearch: bool = False

    generatedAt: str = ""
