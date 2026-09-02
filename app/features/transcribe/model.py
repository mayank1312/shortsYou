import logging
import os
import time
from typing import Any

from deepgram import AsyncDeepgramClient, DeepgramClient

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

        self.async_client = AsyncDeepgramClient(
            api_key=api_key,
        )

        logger.info(
            "Deepgram transcription initialized successfully | model=%s",
            self.MODEL_NAME,
        )

    async def transcribe_url(
        self,
        audio_url: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio directly from a remote URL.

        Deepgram fetches the file itself, server-to-server,
        instead of us downloading it locally and re-uploading
        the bytes over our own (often much slower) connection.
        This is almost always faster than transcribe(), which
        requires a full local download + upload round trip.
        """

        options: dict[str, Any] = {
            "model": self.MODEL_NAME,
            "smart_format": True,
            "punctuate": True,
            "utterances": True,
        }

        options["language"] = language or "hi"

        logger.info(
            "Calling Deepgram via URL | url=%s | model=%s | language=%s",
            audio_url,
            self.MODEL_NAME,
            options["language"],
        )

        request_started_at = time.monotonic()

        response = await self.async_client.listen.v1.media.transcribe_url(
            url=audio_url,
            **options,
            request_options={
                "timeout_in_seconds": 300,
                "max_retries": 3,
            },
        )

        request_duration = time.monotonic() - request_started_at

        result = response.model_dump()

        logger.info(
            "Deepgram URL transcription response received | "
            "duration=%.2f seconds",
            request_duration,
        )

        if request_duration > 20:

            logger.warning(
                "SLOW DEEPGRAM URL RESPONSE | duration=%.2f seconds | "
                "url=%s | if this is still slow, the bottleneck is "
                "Deepgram-side processing, not our upload bandwidth",
                request_duration,
                audio_url,
            )

        return self._normalize_response(
            result,
            requested_language=options["language"],
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

        # ---------------------------------------------------------
        # LANGUAGE
        # ---------------------------------------------------------
        #
        # "multi" makes nova-3 run internal language-switch
        # detection across the whole file, which costs extra
        # processing time. Default to a fixed language ("hi")
        # unless the caller explicitly asks for multi-language
        # detection - your content is majority Hindi with English
        # loanwords, which nova-3 handles fine in single-language
        # mode without the multi-pass overhead.
        # ---------------------------------------------------------

        if language:
            options["language"] = language
        else:
            options["language"] = "hi"

        logger.info(
            "Calling Deepgram | audio=%s | model=%s | language=%s",
            audio_path,
            self.MODEL_NAME,
            options["language"],
        )

        request_started_at = time.monotonic()

        response = self.client.listen.v1.media.transcribe_file(
            request=audio_data,
            **options,
            request_options={
                "timeout_in_seconds": 300,
                "max_retries": 3,
            },
        )

        request_duration = time.monotonic() - request_started_at

        result = response.model_dump()

        logger.info(
            "Deepgram transcription response received | "
            "duration=%.2f seconds",
            request_duration,
        )

        if request_duration > 20:

            logger.warning(
                "SLOW DEEPGRAM RESPONSE | duration=%.2f seconds | "
                "audio=%s | this is unusually slow for a "
                "prerecorded request - check network, retries, "
                "and file size",
                request_duration,
                audio_path,
            )

        return self._normalize_response(
            result,
            requested_language=options["language"],
        )

    def transcribe_gap(
        self,
        slice_path: str,
        time_offset: float,
        language: str = "hi",
    ) -> list[dict[str, Any]]:
        """
        Transcribe an isolated audio slice (e.g. a suspected
        missed window) and return its words with timestamps
        shifted back onto the original file's timeline using
        time_offset.

        Deliberately does NOT use language="multi" here - this
        is the recovery pass, and multi-language detection is
        the leading suspect for why the first pass missed the
        window in the first place.
        """

        logger.info(
            "Re-transcribing gap slice | "
            "path=%s | offset=%.3f | language=%s",
            slice_path,
            time_offset,
            language,
        )

        with open(slice_path, "rb") as audio_file:
            audio_data = audio_file.read()

        response = self.client.listen.v1.media.transcribe_file(
            request=audio_data,
            model=self.MODEL_NAME,
            smart_format=True,
            punctuate=True,
            language=language,
            request_options={
                "timeout_in_seconds": 120,
                "max_retries": 2,
            },
        )

        result = response.model_dump()

        results = result.get("results", {})
        channels = results.get("channels", [])

        if not channels:

            logger.warning(
                "Gap re-transcription returned no channels | "
                "path=%s",
                slice_path,
            )

            return []

        alternative = channels[0].get("alternatives", [{}])[0]
        raw_words = alternative.get("words", [])

        recovered_words: list[dict[str, Any]] = []

        for word in raw_words:

            recovered_words.append(
                {
                    "word": word.get("word", ""),
                    "start": float(
                        word.get("start", 0.0)
                    ) + time_offset,
                    "end": float(
                        word.get("end", 0.0)
                    ) + time_offset,
                    "probability": float(
                        word.get("confidence", 0.0)
                    ),
                }
            )

        logger.info(
            "Gap re-transcription completed | "
            "path=%s | recovered_words=%d",
            slice_path,
            len(recovered_words),
        )

        return recovered_words

    def _normalize_response(
        self,
        response: dict[str, Any],
        requested_language: str | None = None,
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
                "language": requested_language or "unknown",
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

        # ---------------------------------------------------------
        # LANGUAGE
        # ---------------------------------------------------------
        #
        # When we pass a fixed language (e.g. "hi") to Deepgram
        # instead of "multi", Deepgram does NOT echo back
        # `languages` / `detected_language` in the response - it
        # already knows what we told it, so there's nothing to
        # "detect" and report. Deepgram DID transcribe correctly
        # in that language; this is purely a normalization gap
        # on our side if we don't account for it.
        #
        # So: prefer whatever Deepgram actually reports (relevant
        # when language="multi" was used), and only fall back to
        # the language WE requested if Deepgram is silent on it.
        # Only fall back to "unknown" if neither is available.
        # ---------------------------------------------------------

        languages = alternative.get("languages", [])

        if languages:
            language = ",".join(languages)
        else:
            language = alternative.get(
                "detected_language"
            ) or requested_language or "unknown"

        words = alternative.get(
            "words",
            [],
        )

        utterances = results.get(
            "utterances",
            [],
        )

        # ---------------------------------------------------------
        # NORMALIZE THE FULL FLAT WORD LIST FIRST
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        # We normalize ALL words from `alternative.words` up front,
        # independent of utterance boundaries. Deepgram's
        # `utterances` array can have gaps between utterance
        # start/end that do NOT correspond to real silence -
        # they can just be utterance-segmentation boundaries.
        #
        # If we only keep words that fall strictly inside an
        # utterance's [start, end] window (the old containment
        # filter), any word sitting in a gap between two
        # utterances is silently dropped, even though Deepgram
        # transcribed it. That previously showed up downstream
        # as a false "silence gap".
        # ---------------------------------------------------------

        normalized_words: list[dict[str, Any]] = []

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

            normalized_words.append(
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

        segments: list[dict[str, Any]] = []

        if utterances:

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

                utterance_words = [
                    normalized_word
                    for normalized_word in normalized_words
                    if (
                        normalized_word["start"] >= utterance_start
                        and normalized_word["end"] <= utterance_end
                    )
                ]

                segments.append(
                    {
                        "start": utterance_start,
                        "end": utterance_end,
                        "text": utterance_text,
                        "words": utterance_words,
                    }
                )

            # -------------------------------------------------
            # RECOVER ANY WORDS THAT FELL BETWEEN UTTERANCES
            # -------------------------------------------------
            #
            # Any word not claimed by an utterance above still
            # needs to surface downstream (for silence-gap
            # detection, filler detection, etc). We append it
            # as its own single-word "segment" rather than
            # silently discarding it.
            # -------------------------------------------------

            claimed_ranges = [
                (
                    float(utterance.get("start", 0.0)),
                    float(utterance.get("end", 0.0)),
                )
                for utterance in utterances
            ]

            def _is_claimed(
                normalized_word: dict[str, Any],
            ) -> bool:

                for range_start, range_end in claimed_ranges:

                    if (
                        normalized_word["start"] >= range_start
                        and normalized_word["end"] <= range_end
                    ):
                        return True

                return False

            unclaimed_words = [
                normalized_word
                for normalized_word in normalized_words
                if not _is_claimed(normalized_word)
            ]

            if unclaimed_words:

                logger.warning(
                    "Words fell outside all Deepgram utterance "
                    "boundaries; recovering them into a fallback "
                    "segment | count=%d",
                    len(unclaimed_words),
                )

                segments.append(
                    {
                        "start": unclaimed_words[0]["start"],
                        "end": unclaimed_words[-1]["end"],
                        "text": " ".join(
                            unclaimed_word["word"]
                            for unclaimed_word in unclaimed_words
                        ),
                        "words": unclaimed_words,
                    }
                )

                segments.sort(
                    key=lambda segment: segment["start"]
                )

        else:

            # No utterances at all - fall back to a single
            # segment covering every normalized word so nothing
            # is lost.

            if normalized_words:

                segments.append(
                    {
                        "start": normalized_words[0]["start"],
                        "end": normalized_words[-1]["end"],
                        "text": transcript,
                        "words": normalized_words,
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