import logging

from groq import Groq
from sentence_transformers import SentenceTransformer

from app.core.configuration import settings


logger = logging.getLogger(__name__)


class AnalyzeModel:

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    GROQ_MODEL_NAME = "openai/gpt-oss-120b"

    def __init__(self) -> None:

        logger.info(
            "Loading Analyze embedding model | model=%s",
            self.EMBEDDING_MODEL_NAME,
        )

        self.embed_model = SentenceTransformer(
            self.EMBEDDING_MODEL_NAME
        )

        self.groq_client = None

        if settings.GROQ_API_KEY_1:

            logger.info(
                "Initializing Groq client for Analyze"
            )

            self.groq_client = Groq(
                api_key=settings.GROQ_API_KEY_1
            )

        else:

            logger.warning(
                "GROQ_API_KEY_1 is not configured | "
                "Analyze LLM scoring will be unavailable"
            )


analyze_model = AnalyzeModel()