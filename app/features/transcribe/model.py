import logging
import os
from typing import Any

from deepgram import DeepgramClient

logger = logging.getLogger(__name__)


class TranscriptionModel:
    MODEL_NAME = "nova-3"

    def __init__(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY")

        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY is not configured"
            )

        logger.info(
            "Initializing Deepgram transcription | model=%s",
            self.MODEL_NAME,
        )

        self.client = DeepgramClient(
            api_key=api_key,
        )

        logger.info(
            "Deepgram transcription initialized successfully | model=%s",
            self.MODEL_NAME,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> dict[str, Any]:

        logger.info(
            "Starting Deepgram transcription | "
            "audio=%s | model=%s",
            audio_path,
            self.MODEL_NAME,
        )

        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()

        options: dict[str, Any] = {
            "model": self.MODEL_NAME,
            "smart_format": True,
            "punctuate": True,
            "utterances": True,
        }

        # Deepgram Nova-3 defaults to English.
        # Use "multi" when we want multilingual
        # language detection.
        if language:
            options["language"] = language
        else:
            options["language"] = "multi"

        response = self.client.listen.v1.media.transcribe_file(
            request=audio_data,
            **options,
            request_options={
                "timeout_in_seconds": 300,
                "max_retries": 3,
            },
        )

        result = response.model_dump()

        logger.info(
            "Deepgram transcription response received",
        )

        return self._normalize_response(result)

    def _normalize_response(
        self,
        response: dict[str, Any],
    ) -> dict[str, Any]:

        results = response.get(
            "results",
            {},
        )

        channels = results.get(
            "channels",
            [],
        )

        if not channels:
            logger.warning(
                "Deepgram returned no channels"
            )

            return {
                "text": "",
                "language": "unknown",
                "segments": [],
            }

        alternative = (
            channels[0]
            .get("alternatives", [{}])[0]
        )

        transcript = alternative.get(
            "transcript",
            "",
        )

        languages = alternative.get("languages", [])

        if languages:        
            language = ",".join(languages)
        else:
            language = alternative.get("detected_language", "unknown")

        words = alternative.get(
            "words",
            [],
        )

        utterances = results.get(
            "utterances",
            [],
        )

        segments: list[dict[str, Any]] = []

        for utterance in utterances:

            utterance_start = float(
                utterance.get(
                    "start",
                    0.0,
                )
            )

            utterance_end = float(
                utterance.get(
                    "end",
                    0.0,
                )
            )

            utterance_text = utterance.get(
                "transcript",
                "",
            )

            utterance_words: list[
                dict[str, Any]
            ] = []

            for word in words:

                word_start = float(
                    word.get(
                        "start",
                        0.0,
                    )
                )

                word_end = float(
                    word.get(
                        "end",
                        0.0,
                    )
                )

                if (
                    word_start >= utterance_start
                    and word_end <= utterance_end
                ):
                    utterance_words.append(
                        {
                            "word": word.get(
                                "word",
                                "",
                            ),
                            "start": word_start,
                            "end": word_end,
                            "probability": float(
                                word.get(
                                    "confidence",
                                    0.0,
                                )
                            ),
                        }
                    )

            segments.append(
                {
                    "start": utterance_start,
                    "end": utterance_end,
                    "text": utterance_text,
                    "words": utterance_words,
                }
            )

        logger.info(
            "Deepgram response normalized | "
            "segments=%d | words=%d | language=%s",
            len(segments),
            len(words),
            language,
        )

        return {
            "text": transcript,
            "language": language,
            "segments": segments,
        }


transcription_model = TranscriptionModel()

