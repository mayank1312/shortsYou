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
    )

    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1", "")
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2", "")

    API_ACCESS_KEY: str = os.getenv("API_ACCESS_KEY", "")


    INTERNAL_CALLBACK_KEY: str = os.getenv(
        "INTERNAL_CALLBACK_KEY",
    )

    TRANSCRIBE_CALLBACK_URL: str = os.getenv(
        "TRANSCRIBE_CALLBACK_URL",
    )

    ANALYZE_CALLBACK_URL: str = os.getenv(
        "ANALYZE_CALLBACK_URL",
    )

    EMOTION_CALLBACK_URL: str = os.getenv(
        "EMOTION_CALLBACK_URL",
    )

    WHISPER_MODEL: str = os.getenv(
        "WHISPER_MODEL",
        "small",
    )
    DNA_CALLBACK_URL: str = os.getenv(
        "DNA_CALLBACK_URL",
    )
    GO_SERVER_DATABASE_NAME: str = os.getenv(
        "GO_SERVER_DATABASE_NAME",
    )



settings = Settings()