import logging
import statistics
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient

from app.core.configuration import settings
from app.db.database import get_database


logger = logging.getLogger(__name__)

MAX_WORDS_PER_TRANSCRIPT = 250

LATEST_TRANSCRIPTS_LIMIT = 5


def _truncate_to_word_limit(
    text: str,
    limit: int = MAX_WORDS_PER_TRANSCRIPT,
) -> str:

    words = text.split()

    return " ".join(words[:limit])


class CreatorDNA:

    COLLECTION_NAME = "creator_dna_profiles"

    def __init__(self) -> None:

        database = get_database()

        self.collection = database[self.COLLECTION_NAME]

        # Best-effort collection names, matching the conventions
        # used elsewhere in this codebase. If your actual MongoDB
        # names these differently, adjust the two names below -
        # everything that reads from them is wrapped defensively,
        # so a mismatch degrades to zeros rather than crashing.
        self.emotions_collection = database["emotions"]
        self.analysis_collection = database["analyses"]

        # -------------------------------------------------------
        # ABBAS'S OWN DATABASE (Go server's, same MongoDB cluster,
        # different database name) - this is where the AUTHORITATIVE
        # transcripts collection actually lives, with userId already
        # attached to every document. We read from it directly here
        # rather than relying on our own shorts-nlp "transcriptions"
        # collection, which does not carry userId.
        # -------------------------------------------------------

        go_client = MongoClient(settings.MONGODB_URL)

        go_database = go_client[settings.GO_SERVER_DATABASE_NAME]

        self.go_transcripts_collection = go_database["transcripts"]

    def get_latest_transcript_texts(
        self,
        user_id: str,
        limit: int = LATEST_TRANSCRIPTS_LIMIT,
        max_words: int = MAX_WORDS_PER_TRANSCRIPT,
    ) -> list[str]:
        """
        Pulls the creator's most recent `limit` transcripts from
        Abbas's own database, joins each transcript's segment texts
        into one string, and truncates each to `max_words` words -
        we're on a free-tier LLM plan, so this caps token spend
        regardless of how long the source videos actually are.
        """

        try:

            documents = list(
                self.go_transcripts_collection.find(
                    {"userId": user_id}
                )
                .sort("createdAt", -1)
                .limit(limit)
            )

        except Exception:

            logger.exception(
                "Could not query Go server's transcripts "
                "collection for user_id=%s",
                user_id,
            )

            return []

        texts = []

        for document in documents:

            segment_texts = [
                segment.get("text", "")
                for segment in document.get("segments", [])
            ]

            full_text = " ".join(segment_texts).strip()

            if full_text:
                texts.append(
                    _truncate_to_word_limit(full_text, max_words)
                )

        logger.info(
            "Fetched %d transcript(s) for user_id=%s (of up to %d "
            "requested)",
            len(texts),
            user_id,
            limit,
        )

        return texts

    def get_latest_dna_profile(
        self,
        user_id: str,
    ) -> dict[str, Any] | None:
        """
        The creator's most recently saved DNA profile from OUR OWN
        collection, if one exists - used as the "old" side of the
        sync/blend. Returns None on a first-time sync.
        """

        try:

            return self.collection.find_one(
                {"userId": user_id},
                sort=[("createdAt", -1)],
            )

        except Exception:

            logger.exception(
                "Could not query existing DNA profile for "
                "user_id=%s - treating as first-time sync",
                user_id,
            )

            return None

    def save_dna(
        self,
        job_id: str | None,
        user_id: str,
        profile: dict[str, Any],
    ) -> str:

        document = {
            "jobId": job_id,
            "userId": user_id,
            **profile,
            "createdAt": datetime.now(timezone.utc),
        }

        result = self.collection.insert_one(document)

        return str(result.inserted_id)

    def get_emotion_stats(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Best-effort aggregation across this creator's saved /emotion
        results. Returns zeros/defaults if nothing is found - most
        likely because the emotion service hasn't processed anything
        for this creator yet.
        """

        try:
            documents = list(
                self.emotions_collection.find({"userId": user_id})
            )
        except Exception:
            logger.warning(
                "Could not query emotions collection for user_id=%s - "
                "defaulting emotional_range/avg_speech_rate to 0",
                user_id,
            )
            documents = []

        emotion_scores: list[float] = []
        emotion_types: list[str] = []
        speech_rates: list[float] = []

        for document in documents:

            for segment in document.get("segments", []):

                score = segment.get("emotion_score")

                # a real softmax score is never exactly 0 - skip the
                # neutral/0.0 fallbacks saved by failed classifications
                if score:
                    emotion_scores.append(score)
                    emotion_types.append(
                        segment.get("emotion_type", "unknown")
                    )

                rate = segment.get("speech_rate_wpm")

                if rate:
                    speech_rates.append(rate)

        if not emotion_scores:

            return {
                "mean": 0.0,
                "std": 0.0,
                "dominant": "unknown",
                "avg_speech_rate": 0.0,
            }

        dominant = max(
            set(emotion_types),
            key=emotion_types.count,
        )

        return {
            "mean": round(statistics.mean(emotion_scores), 4),
            "std": (
                round(statistics.pstdev(emotion_scores), 4)
                if len(emotion_scores) > 1
                else 0.0
            ),
            "dominant": dominant,
            "avg_speech_rate": (
                round(statistics.mean(speech_rates), 2)
                if speech_rates
                else 0.0
            ),
        }

    def get_avg_hook_score(
        self,
        user_id: str,
    ) -> float:
        """
        Best-effort average hook_score across this creator's saved
        /analyze results. Returns 0.0 if nothing is found.
        """

        try:
            documents = list(
                self.analysis_collection.find({"userId": user_id})
            )
        except Exception:
            logger.warning(
                "Could not query analyses collection for user_id=%s - "
                "defaulting hook-based dna_vector dimension to 0",
                user_id,
            )
            documents = []

        hook_scores: list[float] = []

        for document in documents:

            for segment in document.get("segments", []):

                score = segment.get("hookScore") or segment.get(
                    "hook_score"
                )

                if score is not None:
                    hook_scores.append(score)

        if not hook_scores:
            return 0.0

        return round(sum(hook_scores) / len(hook_scores), 4)


creator_dna_db = CreatorDNA()
