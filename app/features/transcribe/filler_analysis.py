import logging
from typing import Any

from app.features.transcribe.db import transcription
from app.features.transcribe.filler_detection import filler_detector


logger = logging.getLogger(__name__)


async def analyze_transcription_for_fillers(
    video_id: str,
) -> list[dict[str, Any]]:

    logger.info(
        "Starting filler analysis | video_id=%s",
        video_id,
    )

    document = transcription.get_transcription(
        video_id=video_id,
    )

    if not document:

        logger.warning(
            "No transcription found for filler analysis | video_id=%s",
            video_id,
        )

        return []

    transcription_data = document.get(
        "transcription",
        {},
    )

    language = transcription_data.get(
        "language",
        "unknown",
    )

    segments = transcription_data.get(
        "segments",
        [],
    )

    if not segments:

        logger.warning(
            "No transcription segments found for filler analysis | "
            "video_id=%s",
            video_id,
        )

        return []

    words: list[dict[str, Any]] = []

    for segment in segments:

        segment_words = segment.get(
            "words",
            [],
        )

        if not isinstance(segment_words, list):
            continue

        words.extend(segment_words)

    if not words:

        logger.warning(
            "No words found in transcription | video_id=%s",
            video_id,
        )

        return []

    logger.info(
        "Complete transcription prepared for filler analysis | "
        "video_id=%s | language=%s | segments=%d | words=%d",
        video_id,
        language,
        len(segments),
        len(words),
    )

    try:

        # ---------------------------------------------------------
        # GROQ FILLER DETECTION
        # ---------------------------------------------------------

        filler_words = await filler_detector.detect_fillers(
            words=words,
            language=language,
        )

        logger.info(
            "Filler analysis completed | "
            "video_id=%s | fillers=%d | data=%r",
            video_id,
            len(filler_words),
            filler_words,
        )

        # ---------------------------------------------------------
        # SAVE FILLERS TO MONGODB
        # ---------------------------------------------------------

        updated = transcription.update_filler_words(
            transcription_id=document["_id"],
            filler_words=filler_words,
        )

        logger.info(
            "Filler analysis saved to MongoDB | "
            "video_id=%s | updated=%s",
            video_id,
            updated,
        )

        # ---------------------------------------------------------
        # RETURN FILLERS TO CALLER
        # ---------------------------------------------------------

        return filler_words

    except Exception:

        logger.exception(
            "Filler analysis failed | video_id=%s",
            video_id,
        )

        return []