import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_SECONDS = 30.0
CALLBACK_MAX_ATTEMPTS = 3


async def send_callback(
    callback_url: str | None,
    internal_key: str | None,
    payload: dict[str, Any],
) -> None:
    """
    POST a completed (or failed) job result back to the Go backend's
    internal webhook endpoint.

    This deliberately never raises. A background task's job is to
    notify Go that work is done - if the notification itself fails,
    that's a delivery problem to log loudly (so it's visible in
    Render logs), not something that should crash the worker or be
    silently swallowed.

    If callback_url is missing, this is a no-op - callers that were
    invoked without a callback_url are expected to have already
    returned their result synchronously instead.
    """

    if not callback_url:

        logger.warning(
            "No callback_url provided | skipping callback | job_id=%s",
            payload.get("job_id") or payload.get("jobId"),
        )

        return

    headers = {"Content-Type": "application/json"}

    if internal_key:
        headers["X-Internal-API-Key"] = internal_key

    async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:

        for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):

            try:

                response = await client.post(
                    callback_url,
                    json=payload,
                    headers=headers,
                )

                response.raise_for_status()

                logger.info(
                    "Callback delivered | url=%s | attempt=%d | status=%d",
                    callback_url,
                    attempt,
                    response.status_code,
                )

                return

            except Exception:

                logger.exception(
                    "Callback attempt failed | url=%s | attempt=%d/%d",
                    callback_url,
                    attempt,
                    CALLBACK_MAX_ATTEMPTS,
                )

    logger.error(
        "Callback permanently failed after %d attempts | url=%s | job_id=%s",
        CALLBACK_MAX_ATTEMPTS,
        callback_url,
        payload.get("job_id") or payload.get("jobId"),
    )


def build_error_payload(
    job_id: str | None,
    error: Exception,
    retryable: bool = True,
) -> dict[str, Any]:

    return {
        "job_id": job_id,
        "error": str(error),
        "retryable": retryable,
    }
