import json
import logging
from typing import Any

from groq import Groq

from app.core.configuration import settings


logger = logging.getLogger(__name__)


class FillerDetector:

    def __init__(self) -> None:

        self.client = None

        if settings.GROQ_API_KEY_1:
            self.client = Groq(
                api_key=settings.GROQ_API_KEY_1
            )

    async def detect_fillers(
        self,
        words: list[dict[str, Any]],
        language: str,
    ) -> list[dict[str, Any]]:

        if not words:
            logger.info(
                "No words available for filler detection."
            )

            return []

        if self.client is None:
            logger.warning(
                "Groq API key unavailable; "
                "skipping AI filler detection."
            )

            return []

        indexed_words = []

        for index, word in enumerate(words):

            text = word.get(
                "word",
                "",
            ).strip()

            if not text:
                continue

            indexed_words.append(
                {
                    "index": index,
                    "word": text,
                }
            )

        if not indexed_words:
            logger.info(
                "No valid words available for filler detection."
            )

            return []

        transcript = " ".join(
            word["word"]
            for word in indexed_words
        )

        logger.info(
            "Sending complete transcript to Groq for filler detection | "
            "language=%s | words=%d",
            language,
            len(indexed_words),
        )

        prompt = f"""
Analyze the complete spoken transcript below.

Identify every word that is functioning as a speech
filler or disfluency.

A filler is a word or short phrase used primarily to:

- hesitate
- stall while thinking
- fill unnecessary silence
- maintain speech flow
- restart a thought
- buy time while speaking
- structure speech without meaningful content

Rules:

- Analyze the complete transcript using context.
- Analyze each occurrence independently.
- A word can be a filler in one occurrence and meaningful
  in another.
- Support Hindi, English, Hinglish, and mixed-language speech.
- Preserve the original words.
- Do not translate anything.
- Do not invent words.
- Only use indexes from the indexed word list.
- If there are no fillers, return an empty fillers array.

Language:
{language}

Complete transcript:
{transcript}

Indexed words:
{json.dumps(indexed_words, ensure_ascii=False)}

Return only the JSON object.
"""

        # ---------------------------------------------------------
        # DEBUG: LOG EXACT PROMPT BEING SENT TO GROQ
        # ---------------------------------------------------------

        logger.warning(
            "========== GROQ FILLER PROMPT START =========="
        )

        logger.warning(
            "%s",
            prompt,
        )

        logger.warning(
            "========== GROQ FILLER PROMPT END =========="
        )

        try:

            logger.warning(
                "Calling Groq filler detection | model=%s",
                "qwen/qwen3.6-27b",
            )

            response = self.client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a multilingual speech filler "
                            "classification system. "
                            "Return ONLY a JSON object. "
                            "The JSON object MUST contain exactly one "
                            "key named fillers. "
                            "The value of fillers MUST be an array. "
                            "Each array item MUST contain only an integer "
                            "index field."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                reasoning_effort="none",
                max_completion_tokens=512,
                response_format={
                    "type": "json_object",
                },
            )

            # -----------------------------------------------------
            # DEBUG: LOG COMPLETE GROQ RESPONSE
            # -----------------------------------------------------

            logger.warning(
                "========== GROQ RAW RESPONSE START =========="
            )

            logger.warning(
                "%r",
                response,
            )

            logger.warning(
                "========== GROQ RAW RESPONSE END =========="
            )

            content = response.choices[0].message.content

            logger.warning(
                "Groq response content | content=%r",
                content,
            )

            if not content:

                logger.warning(
                    "Groq returned an empty filler detection response."
                )

                return []

            result = json.loads(content)

            logger.info(
                "Parsed Groq JSON successfully | result=%r",
                result,
            )

            fillers = result.get(
                "fillers",
                [],
            )

            if not isinstance(fillers, list):

                logger.warning(
                    "Groq returned an invalid fillers structure | type=%s",
                    type(fillers).__name__,
                )

                return []

            logger.info(
                "Groq filler detection completed | fillers=%d | data=%r",
                len(fillers),
                fillers,
            )

            mapped_results = self._map_results_to_timestamps(
                words=words,
                fillers=fillers,
            )

            logger.info(
                "Final mapped filler results | fillers=%d | data=%r",
                len(mapped_results),
                mapped_results,
            )

            return mapped_results

        except json.JSONDecodeError:

            logger.exception(
                "Groq returned invalid JSON for filler detection | "
                "raw_content=%r",
                content if "content" in locals() else None,
            )

            return []

        except Exception:

            logger.exception(
                "AI filler detection failed | model=%s",
                "qwen/qwen3.6-27b",
            )

            return []

    def _map_results_to_timestamps(
        self,
        words: list[dict[str, Any]],
        fillers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Map Groq's returned word indexes back to the
        original Whisper word timestamps.
        """

        results = []

        seen_indexes: set[int] = set()

        for filler in fillers:

            if not isinstance(filler, dict):
                logger.warning(
                    "Skipping non-dictionary filler result | filler=%r",
                    filler,
                )
                continue

            try:

                index = int(
                    filler.get("index")
                )

            except (TypeError, ValueError):

                logger.warning(
                    "Invalid filler index returned by Groq | filler=%s",
                    filler,
                )

                continue

            if index in seen_indexes:
                logger.warning(
                    "Duplicate filler index returned by Groq | index=%d",
                    index,
                )

                continue

            if index < 0 or index >= len(words):

                logger.warning(
                    "Groq returned out-of-range filler index | "
                    "index=%d | total_words=%d",
                    index,
                    len(words),
                )

                continue

            word = words[index]

            result = {
                "word": word.get(
                    "word",
                    "",
                ).strip(),
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
            }

            results.append(result)

            seen_indexes.add(index)

            logger.info(
                "Filler mapped | index=%d | word=%r | start=%s | end=%s",
                index,
                result["word"],
                result["start"],
                result["end"],
            )

        logger.info(
            "Filler timestamps mapped | fillers=%d",
            len(results),
        )

        return results


filler_detector = FillerDetector()