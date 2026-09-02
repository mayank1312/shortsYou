import logging

from sentence_transformers import SentenceTransformer

from app.shared.api_rotator import groq_rotator


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

        if not groq_rotator.available:

            logger.warning(
                "No Groq API keys configured | "
                "Analyze LLM scoring will be unavailable"
            )


analyze_model = AnalyzeModel()