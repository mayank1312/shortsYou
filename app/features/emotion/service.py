import logging
import os
from typing import Any

import librosa
import numpy as np

from app.core.configuration import settings
from app.features.emotion.db import emotion
from app.features.emotion.model import emotion_model
from app.features.transcribe.db import transcription
from app.shared.callback import send_callback
from app.shared.downloader import download_audio


logger = logging.getLogger(__name__)


def _collect_transcript_words(
    video_id: str,
) -> list[dict[str, Any]]:
    
    document = transcription.get_transcription(video_id)

    if not document:

        logger.warning(
            "No saved transcription found for video_id=%s | "
            "speech_rate_wpm will fall back to 0.0 for all segments",
            video_id,
        )

        return []

    data = document.get("transcription", {})

    words: list[dict[str, Any]] = []

    for segment in data.get("segments", []):
        words.extend(segment.get("words", []))

    return words


def _speech_rate_wpm(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> float:

    duration_minutes = (end - start) / 60.0

    if duration_minutes <= 0:
        return 0.0

    word_count = sum(
        1
        for word in words
        if word.get("start", 0.0) >= start
        and word.get("end", 0.0) <= end
    )

    return round(word_count / duration_minutes, 2)


def _speech_emphasis_score(
    segment_audio: np.ndarray,
) -> float:

    if segment_audio.size == 0:
        return 0.0

    rms = float(
        librosa.feature.rms(y=segment_audio).mean()
    )

    # Normalize into a roughly 0-1 range. RMS on 16-bit PCM audio
    # loaded as float32 typically sits well under 0.1 for normal
    # speech, so a x10 scale followed by a clip keeps this sane.
    return round(
        min(1.0, rms * 10.0),
        4,
    )


def _classify_emotion(
    segment_audio: np.ndarray,
) -> tuple[str, float]:

    if segment_audio.size == 0:
        return "neutral", 0.0

    try:

        predictions = emotion_model.classify(segment_audio)

        top = max(
            predictions,
            key=lambda item: item["score"],
        )

        return top["label"], round(float(top["score"]), 4)

    except Exception:

        logger.exception(
            "Emotion classification failed for a segment - "
            "falling back to neutral"
        )

        return "neutral", 0.0


async def detect_emotions(
    video_id: str,
    audio_url: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    logger.info(
        "Emotion detection started | video_id=%s | segments=%d",
        video_id,
        len(segments),
    )

    audio_path = await download_audio(audio_url)

    try:

        waveform, sample_rate = librosa.load(
            audio_path,
            sr=emotion_model.SAMPLE_RATE,
            mono=True,
        )

        words = _collect_transcript_words(video_id)

        results: list[dict[str, Any]] = []

        for segment in segments:

            index = segment["index"]
            start = float(segment["start"])
            end = float(segment["end"])

            start_sample = max(
                0,
                int(start * sample_rate),
            )

            end_sample = min(
                len(waveform),
                int(end * sample_rate),
            )

            segment_audio = waveform[start_sample:end_sample]

            emotion_type, emotion_score = _classify_emotion(
                segment_audio
            )

            results.append(
                {
                    "index": index,
                    "emotion_type": emotion_type,
                    "emotion_score": emotion_score,
                    "speech_emphasis_score": _speech_emphasis_score(
                        segment_audio
                    ),
                    "speech_rate_wpm": _speech_rate_wpm(
                        words,
                        start,
                        end,
                    ),
                }
            )

            logger.info(
                "Emotion segment processed | index=%d | "
                "emotion=%s (%.2f) | emphasis=%.2f | wpm=%.1f",
                index,
                emotion_type,
                emotion_score,
                results[-1]["speech_emphasis_score"],
                results[-1]["speech_rate_wpm"],
            )

        logger.info(
            "Emotion detection completed | video_id=%s | segments=%d",
            video_id,
            len(results),
        )

        return results

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)


async def run_emotion_job(
    job_id: str | None,
    video_id: str,
    user_id: str | None,
    audio_url: str,
    segments: list[dict[str, Any]],
) -> None:
    """
    Background-task wrapper for the async callback pattern, same
    shape as run_transcription_job. job_id is only used for logging
    here - Go correlates the callback to a video via videoId, not
    job_id.
    """

    logger.info(
        "Async emotion job started | job_id=%s | video_id=%s",
        job_id,
        video_id,
    )

    try:

        results = await detect_emotions(
            video_id=video_id,
            audio_url=audio_url,
            segments=segments,
        )

        emotion.save_emotion(
            job_id=job_id,
            video_id=video_id,
            user_id=user_id,
            segments=results,
        )

        payload = {
            "videoId": video_id,
            "userId": user_id,
            "segments": results,
            "error": "",
        }

        await send_callback(
            callback_url=settings.EMOTION_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload=payload,
        )

    except Exception as error:

        logger.exception(
            "Async emotion job failed | job_id=%s | video_id=%s",
            job_id,
            video_id,
        )

        await send_callback(
            callback_url=settings.EMOTION_CALLBACK_URL,
            internal_key=settings.INTERNAL_CALLBACK_KEY,
            payload={
                "videoId": video_id,
                "userId": user_id,
                "segments": [],
                "error": str(error) or "emotion detection failed",
            },
        )
