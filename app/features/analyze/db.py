from datetime import datetime, timezone
from typing import Any

from app.db.database import get_database


class Analysis:
    COLLECTION_NAME = "analyses"

    def __init__(self) -> None:
        database = get_database()
        self.collection = database[self.COLLECTION_NAME]

    def save_analysis(
        self,
        job_id: str,
        video_id: str,
        user_id: str,
        language: str,
        segments: list[dict[str, Any]],
    ) -> str:

        document = {
            "jobId": job_id,
            "videoId": video_id,
            "userId": user_id,
            "language": language,
            "segments": segments,
            "createdAt": datetime.now(timezone.utc),
        }

        result = self.collection.insert_one(document)

        return str(result.inserted_id)

    def get_analysis(
        self,
        video_id: str,
    ) -> dict[str, Any] | None:

        return self.collection.find_one(
            {"videoId": video_id},
            sort=[("createdAt", -1)],
        )


analysis = Analysis()