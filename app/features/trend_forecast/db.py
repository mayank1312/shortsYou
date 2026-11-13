import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from app.core.configuration import settings
from app.db.database import database_connection


logger = logging.getLogger(__name__)


def to_object_id(value: str) -> ObjectId | None:

    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


class TrendForecasts:
    """
    Everything here lives in the GO SERVER's database (shortsyou), not
    shortsyou_nlp. clips / videos are read from it, and results go to
    two separate collections, one document per user in each:

      - `genre`           -> the user's genres (from /extract-genres)
      - `trend_forecasts` -> the user's trend forecast (from /predict-trends)

    The Go server stores userId as an ObjectId, but nothing else in this
    service does, so reads match both the ObjectId and the plain-string
    form of the id.
    """

    COLLECTION_NAME = "trend_forecasts"
    GENRE_COLLECTION_NAME = "genre"

    def __init__(self) -> None:

        database = database_connection.client[
            settings.GO_SERVER_DATABASE_NAME or "shortsyou"
        ]

        self.collection = database[self.COLLECTION_NAME]
        self.genre_collection = database[self.GENRE_COLLECTION_NAME]
        self.clips_collection = database["clips"]
        self.videos_collection = database["videos"]

    @staticmethod
    def _user_filter(user_id: str) -> dict[str, Any]:

        variants: list[Any] = [user_id]

        object_id = to_object_id(user_id)

        if object_id is not None:
            variants.append(object_id)

        return {"userId": {"$in": variants}}

    @staticmethod
    def _stored_user_id(user_id: str) -> Any:

        # match the Go server's convention when the id is a valid ObjectId
        return to_object_id(user_id) or user_id

    def get_user_clips(
        self,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:

        return list(
            self.clips_collection.find(
                self._user_filter(user_id),
                {
                    "videoId": 1,
                    "semanticLabels": 1,
                    "originalHook": 1,
                    "selectedHook": 1,
                    "transcriptText": 1,
                    "createdAt": 1,
                },
            )
            .sort("createdAt", -1)
            .limit(limit)
        )

    def get_user_videos(
        self,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:

        return list(
            self.videos_collection.find(self._user_filter(user_id))
            .sort("createdAt", -1)
            .limit(limit)
        )

    def get_genre_document(
        self,
        user_id: str,
    ) -> dict[str, Any] | None:

        return self.genre_collection.find_one(self._user_filter(user_id))

    def get_forecast_document(
        self,
        user_id: str,
    ) -> dict[str, Any] | None:

        return self.collection.find_one(self._user_filter(user_id))

    def save_genres(
        self,
        user_id: str,
        genres: list[dict[str, Any]],
        clips_analyzed: int,
        videos_analyzed: int,
    ) -> None:

        now = datetime.now(timezone.utc)

        self.genre_collection.update_one(
            self._user_filter(user_id),
            {
                "$set": {
                    "userId": self._stored_user_id(user_id),
                    "genres": genres,
                    "genreSource": {
                        "clipsAnalyzed": clips_analyzed,
                        "videosAnalyzed": videos_analyzed,
                    },
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )

        logger.info(
            "Genres saved to `%s` | user_id=%s | genres=%s",
            self.GENRE_COLLECTION_NAME,
            user_id,
            [genre["name"] for genre in genres],
        )

    def save_forecast(
        self,
        user_id: str,
        forecast: dict[str, Any],
    ) -> None:

        now = datetime.now(timezone.utc)

        self.collection.update_one(
            self._user_filter(user_id),
            {
                "$set": {
                    "userId": self._stored_user_id(user_id),
                    "forecast": forecast,
                    "forecastUpdatedAt": now,
                    "updatedAt": now,
                },
                # genres used to be stored here by mistake - they now
                # live in the `genre` collection, so clean them out
                "$unset": {
                    "genres": "",
                    "genreSource": "",
                    "genresUpdatedAt": "",
                },
                "$inc": {"forecastVersion": 1},
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )

        logger.info(
            "Trend forecast saved to `%s` | user_id=%s | genres=%s",
            self.COLLECTION_NAME,
            user_id,
            forecast.get("genres"),
        )


trend_forecasts_db = TrendForecasts()