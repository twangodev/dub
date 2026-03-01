from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageType(StrEnum):
    SEPARATE_AUDIO = "separate_audio"
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    TRANSLATE = "translate"
    VOICE_CLONE = "voice_clone"
    TTS = "tts"
    SPEED_ADJUST = "speed_adjust"
    ASSEMBLE = "assemble"
    MUX = "mux"
