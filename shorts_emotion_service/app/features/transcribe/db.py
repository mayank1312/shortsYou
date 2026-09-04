from datetime import datetime, timezone
from typing import Any

from app.db.database import get_database


class Transcription:
    COLLECTION_NAME = "transcriptions"

    def __init__(self) -> None:
        database = get_database()
        self.collection = database[self.COLLECTION_NAME]

    def save_transcription(
        self,
        video_id: str,
        data: dict[str, Any],
    ) -> str:

        document = {
            "videoId": video_id,
            "transcription": data,
            "createdAt": datetime.now(timezone.utc),
        }

        result = self.collection.insert_one(document)

        return str(result.inserted_id)

    def get_transcription(
        self,
        video_id: str,
    ) -> dict[str, Any] | None:

        return self.collection.find_one(
            {"videoId": video_id},
            sort=[("createdAt", -1)],
        )

    def update_filler_words(
        self,
        transcription_id: Any,
        filler_words: list[dict[str, Any]],
    ) -> bool:

        result = self.collection.update_one(
            {"_id": transcription_id},
            {
                "$set": {
                    "transcription.fillerWords": filler_words,
                }
            },
        )

        return result.modified_count > 0


transcription = Transcription()