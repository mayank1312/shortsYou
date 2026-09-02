import logging

from fastapi import Header, HTTPException, status

from app.core.configuration import settings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


async def verify_api_key(
    authorization: str | None = Header(default=None),
) -> None:
    

    if not settings.API_ACCESS_KEY:

        logger.warning(
            "API_ACCESS_KEY is not configured - this endpoint is "
            "currently UNPROTECTED. Set API_ACCESS_KEY in the "
            "environment before deploying."
        )

        return

    if not authorization:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    provided_key = authorization

    if authorization.lower().startswith("bearer "):
        provided_key = authorization[len("bearer "):]

    provided_key = provided_key.strip()

    if provided_key != settings.API_ACCESS_KEY:

        logger.warning(
            "Rejected request with invalid API key"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )