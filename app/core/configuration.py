import os

from dotenv import load_dotenv
load_dotenv()
class Settings:

    APP_NAME: str = os.getenv("APP_NAME", "shorts-nlp")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    MONGODB_URL: str = os.getenv(
        "MONGODB_URL",
        "mongodb://localhost:27017",
    )

    MONGODB_DATABASE: str = os.getenv(
        "MONGODB_DATABASE",
        "shorts_ml",
    )

    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1", "")
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2", "")

    WHISPER_MODEL: str = os.getenv(
        "WHISPER_MODEL",
        "small",
    )


settings = Settings()

