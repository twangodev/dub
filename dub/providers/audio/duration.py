import io

from mutagen import File as MutagenFile


def get_audio_duration(audio_bytes: bytes) -> float:
    """Get duration of audio bytes (WAV, MP3, OGG, etc.) using mutagen."""
    f = MutagenFile(io.BytesIO(audio_bytes))
    if f is None or f.info is None:
        raise ValueError("Could not parse audio format")
    return f.info.length