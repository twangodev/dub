from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageType(StrEnum):
    EXTRACT_AUDIO = "extract_audio"
    SEPARATE_AUDIO = "separate_audio"
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    TRANSLATE = "translate"
    TTS = "tts"
    SPEED_ADJUST = "speed_adjust"
    ASSEMBLE = "assemble"
    MUX = "mux"
