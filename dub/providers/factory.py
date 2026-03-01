from dub.config import Settings
from dub.providers.stt.qwen import QwenSTT
from dub.providers.separation.sam_audio import SAMAudioSeparator
from dub.providers.translation.gemini import GeminiTranslation
from dub.providers.tts.fish_audio import FishAudioTTS


def create_stt(settings: Settings) -> QwenSTT:
    if settings.stt_backend == "qwen":
        return QwenSTT(stt_url=settings.stt_url)
    raise ValueError(f"Unknown STT backend: {settings.stt_backend}")


def create_separator(settings: Settings) -> SAMAudioSeparator:
    if settings.separation_backend == "sam_audio":
        return SAMAudioSeparator(sam_audio_url=settings.sam_audio_url)
    raise ValueError(f"Unknown separation backend: {settings.separation_backend}")


def create_translator(settings: Settings) -> GeminiTranslation:
    if settings.translation_backend == "gemini":
        return GeminiTranslation(api_key=settings.gemini_api_key)
    raise ValueError(f"Unknown translation backend: {settings.translation_backend}")


def create_tts(settings: Settings) -> FishAudioTTS:
    if settings.tts_backend == "fish_audio":
        return FishAudioTTS(api_key=settings.fish_audio_api_key)
    raise ValueError(f"Unknown TTS backend: {settings.tts_backend}")
