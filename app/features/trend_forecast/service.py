import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from app.core.configuration import settings
from app.features.trend_forecast.db import trend_forecasts_db
from app.features.trend_forecast.schema import (
    ExtractGenresResponse,
    GenreResult,
    PredictTrendsResponse,
    Trend,
    VideoIdea,
)
from app.shared.api_rotator import groq_rotator
from app.shared.web_search import search_many


logger = logging.getLogger(__name__)

# how much of the creator's library we look at - free-tier LLM, so we
# cap token spend regardless of how many clips a creator has
GENRE_CLIPS_LIMIT = 30
GENRE_VIDEOS_LIMIT = 30
WORDS_PER_CLIP = 40

MAX_GENRES_FROM_PROFILE = 4
MAX_GENRES_PER_PREDICTION = 5
MAX_GENRE_NAME_LENGTH = 40

TRENDS_PER_GENRE_MAX = 3
VIDEO_IDEAS_MAX = 8

# video fields that may describe what a video is about - we don't know
# the Go server's exact videos schema, so we read whichever exist.
VIDEO_TEXT_FIELDS = (
    "title",
    "name",
    "description",
    "caption",
    "category",
    "genre",
    "tags",
    "hashtags",
)

VALID_MOMENTUM = {"emerging", "rising", "peaking"}


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------


def _truncate_words(text: str, limit: int) -> str:

    return " ".join(str(text).split()[:limit])


def _label_to_text(label: Any) -> str:

    if isinstance(label, dict):
        return str(
            label.get("label") or label.get("name") or ""
        ).strip()

    return str(label).strip()


def _normalize_genre(name: Any) -> str:

    return " ".join(str(name).lower().split())[:MAX_GENRE_NAME_LENGTH]


def _dedupe_genres(genres: list[Any]) -> list[str]:

    seen: set[str] = set()
    result: list[str] = []

    for genre in genres:

        normalized = _normalize_genre(genre)

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def _parse_json_object(content: str) -> dict[str, Any]:
    """
    Compound (web search) models can't be forced into JSON mode, so
    they sometimes wrap the object in prose or ```json fences. Pull
    out the outermost {...} and parse that.
    """

    cleaned = re.sub(r"```(?:json)?", "", content or "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object")

    return parsed


def _require_groq() -> None:

    if not groq_rotator.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured",
        )


# ----------------------------------------------------------------
# route 1: genres out of the user's clips + videos
# ----------------------------------------------------------------


def _build_library_digest(
    clips: list[dict[str, Any]],
    videos: list[dict[str, Any]],
) -> str:

    lines: list[str] = []

    for clip in clips:

        labels = [
            _label_to_text(label)
            for label in clip.get("semanticLabels") or []
        ]

        parts = []

        if any(labels):
            parts.append(
                "labels: " + ", ".join(label for label in labels if label)
            )

        hook = clip.get("originalHook") or clip.get("selectedHook")

        if hook:
            parts.append("hook: " + _truncate_words(hook, 20))

        text = clip.get("transcriptText")

        if text:
            parts.append(
                "text: " + _truncate_words(text, WORDS_PER_CLIP)
            )

        if parts:
            lines.append("CLIP | " + " | ".join(parts))

    for video in videos:

        parts = []

        for field in VIDEO_TEXT_FIELDS:

            value = video.get(field)

            if not value:
                continue

            if isinstance(value, list):
                value = ", ".join(_label_to_text(v) for v in value)

            parts.append(f"{field}: {_truncate_words(value, 30)}")

        if parts:
            lines.append("VIDEO | " + " | ".join(parts))

    return "\n".join(lines)


def extract_genres(user_id: str) -> ExtractGenresResponse:

    clips = trend_forecasts_db.get_user_clips(
        user_id, GENRE_CLIPS_LIMIT
    )
    videos = trend_forecasts_db.get_user_videos(
        user_id, GENRE_VIDEOS_LIMIT
    )

    digest = _build_library_digest(clips, videos)

    if not digest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No clips or videos found for this user",
        )

    _require_groq()

    prompt = f"""You are classifying what kind of content a short-form video creator makes.

Below is a sample of their clips and videos. Content may be in Hindi, Hinglish or English.

Return the creator's content GENRES - broad, audience-facing categories like
"tech", "gaming", "finance", "fitness", "comedy", "education", "news",
"lifestyle", "food", "travel", "motivation". Not sub-topics.

Rules:
- 1 to {MAX_GENRES_FROM_PROFILE} genres, most dominant first.
- Lowercase English names, 1-2 words each.
- confidence is 0-1: how much of the library fits that genre.
- Only genres the content actually supports - do not pad.
- Return ONLY JSON: {{"genres": [{{"name": "tech", "confidence": 0.8}}]}}

LIBRARY:
{digest}"""

    try:

        response = groq_rotator.create_chat_completion(
            model=settings.TREND_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        parsed = _parse_json_object(
            response.choices[0].message.content
        )

    except Exception as error:

        logger.exception(
            "Genre extraction failed | user_id=%s", user_id
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Genre extraction failed",
        ) from error

    genres: list[GenreResult] = []
    seen: set[str] = set()

    for item in parsed.get("genres") or []:

        if isinstance(item, str):
            item = {"name": item}

        if not isinstance(item, dict):
            continue

        name = _normalize_genre(item.get("name", ""))

        if not name or name in seen:
            continue

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        seen.add(name)

        genres.append(
            GenreResult(
                name=name,
                confidence=round(min(max(confidence, 0.0), 1.0), 3),
            )
        )

        if len(genres) == MAX_GENRES_FROM_PROFILE:
            break

    if not genres:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not determine any genre for this user",
        )

    trend_forecasts_db.save_genres(
        user_id,
        [genre.model_dump() for genre in genres],
        clips_analyzed=len(clips),
        videos_analyzed=len(videos),
    )

    return ExtractGenresResponse(
        userId=user_id,
        genres=genres,
        clipsAnalyzed=len(clips),
        videosAnalyzed=len(videos),
    )


# ----------------------------------------------------------------
# route 2: predict next trends for the given genres
# ----------------------------------------------------------------


def _creator_context(clips: list[dict[str, Any]]) -> str:

    label_counts: Counter[str] = Counter()
    hooks: list[str] = []

    for clip in clips:

        for label in clip.get("semanticLabels") or []:

            text = _label_to_text(label).lower()

            if text:
                label_counts[text] += 1

        hook = clip.get("originalHook") or clip.get("selectedHook")

        if hook and len(hooks) < 5:
            hooks.append(_truncate_words(hook, 15))

    parts = []

    if label_counts:
        parts.append(
            "Topics they usually cover: "
            + ", ".join(label for label, _ in label_counts.most_common(8))
        )

    if hooks:
        parts.append(
            "Example hooks they've used: " + " / ".join(hooks)
        )

    return "\n".join(parts)


def _previously_suggested(document: dict[str, Any] | None) -> list[str]:

    forecast = (document or {}).get("forecast") or {}

    names = [
        trend.get("name", "") for trend in forecast.get("trends") or []
    ] + [
        idea.get("title", "")
        for idea in forecast.get("videoIdeas") or []
    ]

    return [name for name in names if name][:20]


def _gather_web_context(genres: list[str]) -> str:
    """
    Live web results per genre via Tavily. Returns "" when search is
    unavailable (no key / failures) - the forecast then runs without
    web data instead of failing.
    """

    queries = {
        genre: f"trending {genre} topics for short video creators"
        for genre in genres
    }

    found = search_many(list(queries.values()))

    blocks: list[str] = []

    for genre, query in queries.items():

        results = found.get(query) or []

        if not results:
            continue

        lines = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in results
        )

        blocks.append(f"[{genre}]\n{lines}")

    return "\n\n".join(blocks)


def _call_forecast_llm(prompt: str) -> dict[str, Any]:

    response = groq_rotator.create_chat_completion(
        model=settings.TREND_MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        response_format={"type": "json_object"},
    )

    return _parse_json_object(response.choices[0].message.content)


def _clean_forecast(
    parsed: dict[str, Any],
    genres: list[str],
) -> tuple[list[Trend], list[VideoIdea]]:

    genre_lookup = {genre: genre for genre in genres}

    trends: list[Trend] = []
    per_genre: Counter[str] = Counter()

    for item in parsed.get("trends") or []:

        if not isinstance(item, dict):
            continue

        genre = genre_lookup.get(_normalize_genre(item.get("genre", "")))
        name = str(item.get("name", "")).strip()

        # trends must belong to exactly one requested genre
        if not genre or not name:
            continue

        if per_genre[genre] >= TRENDS_PER_GENRE_MAX:
            continue

        momentum = str(item.get("momentum", "rising")).lower()

        keywords = item.get("keywords") or []

        trends.append(
            Trend(
                genre=genre,
                name=name,
                whyTrending=str(item.get("whyTrending", "")).strip(),
                momentum=(
                    momentum if momentum in VALID_MOMENTUM else "rising"
                ),
                keywords=[
                    str(k).strip() for k in keywords if str(k).strip()
                ][:6]
                if isinstance(keywords, list)
                else [],
            )
        )

        per_genre[genre] += 1

    trend_names = {trend.name.lower(): trend.name for trend in trends}

    ideas: list[VideoIdea] = []

    for item in parsed.get("videoIdeas") or []:

        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()

        if not title:
            continue

        idea_genres = _dedupe_genres(item.get("genres") or [])
        idea_genres = [g for g in idea_genres if g in genre_lookup]

        idea_trends = [
            trend_names[str(t).strip().lower()]
            for t in item.get("trends") or []
            if str(t).strip().lower() in trend_names
        ]

        ideas.append(
            VideoIdea(
                title=title,
                hook=str(item.get("hook", "")).strip(),
                genres=idea_genres,
                trends=idea_trends,
            )
        )

        if len(ideas) == VIDEO_IDEAS_MAX:
            break

    return trends, ideas


def predict_trends(user_id: str) -> PredictTrendsResponse:

    # genres always come from the `genre` collection
    # (saved by /extract-genres)
    genre_document = trend_forecasts_db.get_genre_document(user_id)

    saved = (genre_document or {}).get("genres") or []
    genres = _dedupe_genres([g.get("name", "") for g in saved])

    if not genres:
        # first time for this user: work the genres out now and save
        # them, so the caller never has to hit /extract-genres first
        logger.info(
            "No saved genres for user_id=%s - extracting first",
            user_id,
        )

        extracted = extract_genres(user_id)

        genres = _dedupe_genres([g.name for g in extracted.genres])

    genres = genres[:MAX_GENRES_PER_PREDICTION]

    # previous forecast (if any) so a re-run suggests different things
    document = trend_forecasts_db.get_forecast_document(user_id)

    _require_groq()

    clips = trend_forecasts_db.get_user_clips(user_id, GENRE_CLIPS_LIMIT)
    context = _creator_context(clips)
    already_suggested = _previously_suggested(document)

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    web_context = _gather_web_context(genres)

    if web_context:
        web_instructions = (
            "Below are fresh web results from the last month. Base the "
            "trends on what they show is rising NOW, and look for what is "
            "about to trend next. Ignore anything stale.\n\n"
            "WEB RESULTS:\n" + web_context
        )
    else:
        web_instructions = (
            "No live web data is available. Use your best knowledge of "
            "what is emerging, and avoid topics already past their peak."
        )

    web_instructions += (
        "\nPrefer emerging and rising topics. Old news (for tech: "
        '"ChatGPT", "LangChain", basic "prompt engineering") is not a trend.'
    )

    prompt = f"""Today is {today}. You are a short-form video trend analyst.

A creator makes content in these genres: {", ".join(genres)}.
{context}

{web_instructions}

PART 1 - TRENDS
For EACH genre give {TRENDS_PER_GENRE_MAX - 1} to {TRENDS_PER_GENRE_MAX} separate trends. Every trend belongs to exactly ONE genre.
Never merge genres inside a trend.

PART 2 - VIDEO IDEAS
Give 5 to 6 video ideas the creator could make. Ideas MAY blend several of
the genres and trends (e.g. gaming + tech -> "Building a working CPU
inside Minecraft"). Each idea needs a strong scroll-stopping hook line.
{"Do NOT repeat any of these earlier suggestions: " + "; ".join(already_suggested) if already_suggested else ""}

Return ONLY a JSON object, no prose, in exactly this shape:
{{
  "trends": [
    {{"genre": "<one of: {", ".join(genres)}>", "name": "...", "whyTrending": "one sentence", "momentum": "emerging|rising|peaking", "keywords": ["..."]}}
  ],
  "videoIdeas": [
    {{"title": "...", "hook": "...", "genres": ["..."], "trends": ["<exact trend name from above>"]}}
  ]
}}"""

    try:

        parsed = _call_forecast_llm(prompt)
        used_search = bool(web_context)

    except Exception as error:

        logger.exception(
            "Trend forecast failed | user_id=%s", user_id
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Trend forecast failed",
        ) from error

    trends, ideas = _clean_forecast(parsed, genres)

    if not trends:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Model returned no usable trends",
        )

    response = PredictTrendsResponse(
        userId=user_id,
        genres=genres,
        trends=trends,
        videoIdeas=ideas,
        usedWebSearch=used_search,
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )

    trend_forecasts_db.save_forecast(
        user_id,
        response.model_dump(exclude={"userId"}),
    )

    return response