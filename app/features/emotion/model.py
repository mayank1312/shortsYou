import asyncio
import logging
import os

import httpx


logger = logging.getLogger(__name__)


class EmotionModel:

    MODEL_NAME = "superb/hubert-large-superb-er"


    SAMPLE_RATE = 16000

    API_URL = (
        f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
    )

    MAX_RETRIES = 3

    RETRY_WAIT_SECONDS = 5.0

    def __init__(self) -> None:

        self.hf_token = os.getenv("HF_TOKEN", "")

        if not self.hf_token:

            logger.warning(
                "HF_TOKEN is not set - HuggingFace Inference API "
                "calls will use the anonymous rate limit, which is "
                "low and shared across all unauthenticated users. "
                "Set HF_TOKEN in the environment (a free HuggingFace "
                "account read token) before relying on this in "
                "production."
            )

    async def classify(
        self,
        audio_bytes: bytes,
    ) -> list[dict]:
        """
        audio_bytes must be a complete WAV file (with header), not a
        raw numpy array - the Inference API expects an actual audio
        file body, not tensor data.

        Returns [] on failure rather than raising - callers are
        expected to fall back to a neutral/0.0 result rather than
        crash the whole segment loop over one bad API call.
        """

        headers = {}

        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        async with httpx.AsyncClient(timeout=30.0) as client:

            for attempt in range(1, self.MAX_RETRIES + 1):

                try:

                    response = await client.post(
                        self.API_URL,
                        headers=headers,
                        content=audio_bytes,
                    )

                except Exception:

                    logger.exception(
                        "HF Inference API request failed | attempt=%d/%d",
                        attempt,
                        self.MAX_RETRIES,
                    )

                    continue

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 503:

                    # HF is cold-loading the model on their side -
                    # this is normal on first use after idle, not
                    # an error. Their response body usually includes
                    # an estimated_time, but a fixed backoff is
                    # simpler and good enough here.
                    logger.warning(
                        "HF Inference API model is loading, "
                        "retrying | attempt=%d/%d",
                        attempt,
                        self.MAX_RETRIES,
                    )

                    await asyncio.sleep(self.RETRY_WAIT_SECONDS)

                    continue

                logger.error(
                    "HF Inference API error | status=%d | body=%s",
                    response.status_code,
                    response.text[:300],
                )

                return []

        logger.error(
            "HF Inference API failed after %d attempts",
            self.MAX_RETRIES,
        )

        return []


emotion_model = EmotionModel()