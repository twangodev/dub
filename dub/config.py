from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

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

    # Iterative refinement
    max_generations: int = 2
    samples_per_generation: int = 4
    top_k_samples: int = 3
    plateau_threshold: float = 2.0
    min_fluency_score: float = 98.0
    eval_script_target_duration: float = 30.0

    # Duration fitting
    duration_tolerance: float = 0.10
    max_fit_attempts: int = 3
    samples_per_step: int = 2
    speed_min: float = 0.85
    speed_max: float = 1.3


settings = Settings()
