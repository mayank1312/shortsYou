
import logging
import os
import wave
from typing import Any

from app.features.transcribe.db import transcription
from app.features.transcribe.model import transcription_model
from app.features.transcribe.filler_analysis import (
    analyze_transcription_for_fillers,
)
from app.shared.downloader import download_audio


logger = logging.getLogger(__name__)


def get_audio_duration(
    audio_path: str,
) -> float:

    try:

        with wave.open(
            audio_path,
            "rb",
        ) as audio:

            frame_count = audio.getnframes()
            frame_rate = audio.getframerate()

            if frame_rate <= 0:
                return 0.0

            duration = frame_count / frame_rate

            return round(
                float(duration),
                3,
            )

    except Exception:

        logger.exception(
            "Failed to determine audio duration | path=%s",
            audio_path,
        )

        return 0.0


def detect_silence_gaps(
    words: list[dict[str, Any]],
    audio_duration: float | None = None,
    minimum_gap: float = 0.5,
) -> list[dict[str, float]]:

    silence_gaps: list[dict[str, float]] = []

    if not words:
        return silence_gaps

    # ---------------------------------------------------------
    # GAPS BETWEEN WORDS
    # ---------------------------------------------------------

    for previous_word, current_word in zip(
        words,
        words[1:],
    ):

        previous_end = previous_word.get(
            "end"
        )

        current_start = current_word.get(
            "start"
        )

        if previous_end is None or current_start is None:
            continue

        gap = float(current_start) - float(previous_end)

        if gap >= minimum_gap:

            silence_gaps.append(
                {
                    "start": round(
                        float(previous_end),
                        3,
                    ),
                    "end": round(
                        float(current_start),
                        3,
                    ),
                    "duration": round(
                        float(gap),
                        3,
                    ),
                }
            )

    # ---------------------------------------------------------
    # TRAILING SILENCE
    # ---------------------------------------------------------
    #
    # Whisper stops producing words when speech ends.
    #
    # Therefore, if the final recognized word ends at 39.8s
    # and the actual audio ends at 52.0s, we explicitly record:
    #
    # 39.8s -> 52.0s
    #
    # as a silence gap.
    # ---------------------------------------------------------

    if audio_duration is not None and audio_duration > 0:

        last_word_end = words[-1].get(
            "end"
        )

        if last_word_end is not None:

            trailing_gap = (
                float(audio_duration)
                - float(last_word_end)
            )

            if trailing_gap >= minimum_gap:

                silence_gaps.append(
                    {
                        "start": round(
                            float(last_word_end),
                            3,
                        ),
                        "end": round(
                            float(audio_duration),
                            3,
                        ),
                        "duration": round(
                            float(trailing_gap),
                            3,
                        ),
                    }
                )

    return silence_gaps


def calculate_confidence(
    words: list[dict[str, Any]],
) -> float:

    probabilities = [
        float(word["probability"])
        for word in words
        if word.get("probability") is not None
    ]

    if not probabilities:
        return 0.0

    return sum(probabilities) / len(probabilities)


async def process_whisper_result(
    result: dict[str, Any],
    audio_duration: float,
) -> dict[str, Any]:

    language = result.get(
        "language",
        "unknown",
    )

    raw_segments = result.get(
        "segments",
        [],
    )

    logger.info(
        "Whisper result received | "
        "language=%s | segments=%d | text=%r | duration=%.3f",
        language,
        len(raw_segments),
        result.get("text", ""),
        audio_duration,
    )

    processed_segments: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # PROCESS EACH WHISPER SEGMENT
    # ---------------------------------------------------------

    for index, whisper_segment in enumerate(raw_segments):

        logger.info(
            "Processing Whisper segment | "
            "index=%d | start=%s | end=%s | text=%r",
            index,
            whisper_segment.get("start"),
            whisper_segment.get("end"),
            whisper_segment.get("text"),
        )

        words: list[dict[str, Any]] = []

        for word in whisper_segment.get(
            "words",
            [],
        ):

            word_text = word.get(
                "word",
                "",
            ).strip()

            if not word_text:
                continue

            words.append(
                {
                    "word": word_text,
                    "start": float(
                        word.get(
                            "start",
                            0.0,
                        )
                    ),
                    "end": float(
                        word.get(
                            "end",
                            0.0,
                        )
                    ),
                    "probability": float(
                        word.get(
                            "probability",
                            0.0,
                        )
                    ),
                }
            )

        logger.info(
            "Segment words extracted | "
            "index=%d | words=%d",
            index,
            len(words),
        )

        processed_segments.append(
            {
                "index": index,
                "start": round(
                    float(
                        whisper_segment.get(
                            "start",
                            0.0,
                        )
                    ),
                    3,
                ),
                "end": round(
                    float(
                        whisper_segment.get(
                            "end",
                            0.0,
                        )
                    ),
                    3,
                ),
                "text": whisper_segment.get(
                    "text",
                    "",
                ).strip(),
                "words": words,
            }
        )

    # ---------------------------------------------------------
    # COLLECT ALL WORDS
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Do this AFTER processing all segments.
    #
    # We intentionally do NOT use the variable name `segment`
    # here because it can overwrite the Whisper segment currently
    # being processed.
    # ---------------------------------------------------------

    all_words: list[dict[str, Any]] = []

    for processed_segment in processed_segments:

        segment_words = processed_segment.get(
            "words",
            [],
        )

        if isinstance(
            segment_words,
            list,
        ):

            all_words.extend(
                segment_words
            )

    logger.info(
        "All Whisper words collected | "
        "language=%s | segments=%d | words=%d",
        language,
        len(processed_segments),
        len(all_words),
    )

    # ---------------------------------------------------------
    # GLOBAL SILENCE DETECTION
    # ---------------------------------------------------------

    silence_gaps = detect_silence_gaps(
        words=all_words,
        audio_duration=audio_duration,
    )

    logger.info(
        "Global silence detection completed | "
        "duration=%.3f | gaps=%d",
        audio_duration,
        len(silence_gaps),
    )

    for gap in silence_gaps:

        logger.info(
            "Silence gap detected | "
            "start=%.3f | end=%.3f | duration=%.3f",
            gap["start"],
            gap["end"],
            gap["duration"],
        )

    logger.info(
        "Whisper processing completed | "
        "language=%s | processed_segments=%d",
        language,
        len(processed_segments),
    )

    return {
        "segments": processed_segments,
        "language": language,
        "silenceGaps": silence_gaps,
    }


async def transcribe_video(
    video_id: str,
    audio_url: str,
) -> dict[str, Any]:

    media_path = None

    logger.info(
        "Transcription started | "
        "video_id=%s | audio_url=%s",
        video_id,
        audio_url,
    )

    try:

        # ---------------------------------------------------------
        # DOWNLOAD AUDIO
        # ---------------------------------------------------------

        logger.info(
            "Downloading video for audio extraction | "
            "video_id=%s",
            video_id,
        )

        media_path = await download_audio(
            video_url=audio_url,
        )

        logger.info(
            "Audio ready | "
            "video_id=%s | path=%s | size=%d bytes",
            video_id,
            media_path,
            os.path.getsize(media_path),
        )

        # ---------------------------------------------------------
        # ACTUAL AUDIO DURATION
        # ---------------------------------------------------------

        audio_duration = get_audio_duration(
            audio_path=media_path,
        )

        logger.info(
            "Actual audio duration detected | "
            "video_id=%s | duration=%.3f seconds",
            video_id,
            audio_duration,
        )

        # ---------------------------------------------------------
        # WHISPER
        # ---------------------------------------------------------

        logger.info(
            "Starting Whisper transcription | "
            "video_id=%s",
            video_id,
        )

        # Language is intentionally NOT passed.
        # Whisper automatically detects the language.

        whisper_result = transcription_model.transcribe(
            media_path,
        )
        logger.info(
            "WHISPER DEBUG | duration=%s | segments=%d",
            whisper_result.get("duration"),
            len(whisper_result.get("segments", [])),
        )

        for i, segment in enumerate(
            whisper_result.get("segments", [])
        ):
            logger.info(
                "WHISPER SEGMENT | index=%d | start=%s | end=%s | text=%r | words=%d",
                i,
                segment.get("start"),
                segment.get("end"),
                segment.get("text"),
                len(segment.get("words", [])),
            )

        if whisper_result.get("segments"):
            last_segment = whisper_result["segments"][-1]

            logger.info(
                "WHISPER LAST SEGMENT | start=%s | end=%s | text=%r",
                last_segment.get("start"),
                last_segment.get("end"),
                last_segment.get("text"),
            )

            if last_segment.get("words"):
                logger.info(
                    "WHISPER LAST WORD | %r",
                    last_segment["words"][-1],
                )
        logger.info(
            "========== WHISPER TIMESTAMP DEBUG END =========="
        )
        logger.info(
            "Whisper transcription completed | "
            "video_id=%s | detected_language=%s | segments=%d | text=%r",
            video_id,
            whisper_result.get("language"),
            len(
                whisper_result.get(
                    "segments",
                    [],
                )
            ),
            whisper_result.get(
                "text",
                "",
            ),
        )

        # ---------------------------------------------------------
        # LOCAL PROCESSING
        # ---------------------------------------------------------

        processed_result = await process_whisper_result(
            result=whisper_result,
            audio_duration=audio_duration,
        )

        # ---------------------------------------------------------
        # SAVE TRANSCRIPTION
        # ---------------------------------------------------------

        logger.info(
            "Saving transcription to MongoDB | "
            "video_id=%s",
            video_id,
        )

        transcription.save_transcription(
            video_id=video_id,
            data=processed_result,
        )

        logger.info(
            "Transcription saved successfully | "
            "video_id=%s",
            video_id,
        )

        # ---------------------------------------------------------
        # FILLER ANALYSIS
        # ---------------------------------------------------------

        logger.info(
            "Starting filler analysis | "
            "video_id=%s",
            video_id,
        )

        filler_words = await analyze_transcription_for_fillers(
            video_id=video_id,
        )

        if filler_words is None:
            filler_words = []

        logger.info(
            "Filler analysis completed | "
            "video_id=%s | fillers=%d",
            video_id,
            len(filler_words),
        )

        # ---------------------------------------------------------
        # ADD FILLERS TO RESPONSE
        # ---------------------------------------------------------

        processed_result["fillerWords"] = filler_words

        logger.info(
            "Final transcription response prepared | "
            "video_id=%s | segments=%d | fillers=%d | gaps=%d",
            video_id,
            len(
                processed_result.get(
                    "segments",
                    [],
                )
            ),
            len(filler_words),
            len(
                processed_result.get(
                    "silenceGaps",
                    [],
                )
            ),
        )

        return processed_result

    except Exception:

        logger.exception(
            "Transcription failed | "
            "video_id=%s",
            video_id,
        )

        raise

    finally:

        if media_path and os.path.exists(
            media_path
        ):

            logger.info(
                "Removing temporary media | "
                "path=%s",
                media_path,
            )

            os.remove(
                media_path
            )