from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Provider backends
    stt_backend: str = "qwen"
    separation_backend: str = "sam_audio"
    translation_backend: str = "gemini"
    tts_backend: str = "fish_audio"

    # External service URLs
    stt_url: str = "http://localhost:8001"
    sam_audio_url: str = "http://localhost:8002"

    # Infrastructure
    redis_url: str = "redis://localhost:6379"
    data_dir: str = "./data/jobs"

    # Job lifecycle
    job_ttl_seconds: int = 86400  # 24h

    # API keys
    fish_audio_api_key: str = ""
    gemini_api_key: str = ""


settings = Settings()
