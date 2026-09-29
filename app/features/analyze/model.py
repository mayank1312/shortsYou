import logging

from sentence_transformers import SentenceTransformer

from app.shared.api_rotator import groq_rotator


logger = logging.getLogger(__name__)


class AnalyzeModel:

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    GROQ_MODEL_NAME = "openai/gpt-oss-120b"

    def __init__(self) -> None:

        self._embed_model: SentenceTransformer | None = None

        if not groq_rotator.available:

            logger.warning(
                "No Groq API keys configured | "
                "Analyze LLM scoring will be unavailable"
            )

    @property
    def embed_model(self) -> SentenceTransformer:

        if self._embed_model is None:

            logger.info(
                "Loading Analyze embedding model (first use) | "
                "model=%s",
                self.EMBEDDING_MODEL_NAME,
            )

            self._embed_model = SentenceTransformer(
                self.EMBEDDING_MODEL_NAME
            )

        return self._embed_model


analyze_model = AnalyzeModel()