import logging
from typing import Any

from groq import Groq

from app.core.configuration import settings


logger = logging.getLogger(__name__)


class GroqKeyRotator:

    def __init__(self) -> None:

        self._keys = [
            key
            for key in (
                settings.GROQ_API_KEY_1,
                settings.GROQ_API_KEY_2,
            )
            if key
        ]

        if not self._keys:

            logger.warning(
                "No Groq API keys configured (GROQ_API_KEY_1 / "
                "GROQ_API_KEY_2) | Groq-dependent features will "
                "be unavailable"
            )

        self._clients = [
            Groq(api_key=key)
            for key in self._keys
        ]

        self._current_index = 0

        logger.info(
            "Groq key rotator initialized | keys_configured=%d",
            len(self._clients),
        )

    @property
    def available(self) -> bool:

        return bool(self._clients)

    def _current_client(self) -> Groq:

        return self._clients[self._current_index]

    def _rotate(self) -> None:

        previous_index = self._current_index

        self._current_index = (
            self._current_index + 1
        ) % len(self._clients)

        logger.warning(
            "Rotating Groq API key | from_index=%d | to_index=%d",
            previous_index,
            self._current_index,
        )

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        

        status_code = getattr(error, "status_code", None)

        if status_code == 429:
            return True

        if status_code is not None and status_code >= 500:
            return True

        message = str(error).lower()

        return (
            "rate limit" in message
            or "rate_limit" in message
            or "429" in message
            or "quota" in message
        )

    def create_chat_completion(self, **kwargs: Any) -> Any:
       

        if not self.available:

            raise RuntimeError(
                "No Groq API keys are configured"
            )

        attempts = len(self._clients)
        last_error: Exception | None = None

        for attempt in range(attempts):

            client = self._current_client()

            try:

                return client.chat.completions.create(**kwargs)

            except Exception as error:

                last_error = error

                is_last_attempt = attempt == attempts - 1

                if (
                    self._is_retryable_error(error)
                    and not is_last_attempt
                ):

                    logger.warning(
                        "Groq request failed on key index=%d, "
                        "rotating and retrying | attempt=%d/%d | "
                        "error=%s",
                        self._current_index,
                        attempt + 1,
                        attempts,
                        error,
                    )

                    self._rotate()
                    continue

                logger.exception(
                    "Groq request failed and no more keys to "
                    "rotate to | key_index=%d",
                    self._current_index,
                )

                raise

        if last_error:
            raise last_error

        raise RuntimeError(
            "Groq request failed for an unknown reason"
        )


groq_rotator = GroqKeyRotator()