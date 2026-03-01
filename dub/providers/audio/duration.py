import struct


def get_wav_duration(audio_bytes: bytes) -> float:
    """Get duration of WAV audio from bytes by parsing the RIFF/WAV header.

    Expects standard PCM WAV. Returns duration in seconds.
    """
    if len(audio_bytes) < 44:
        raise ValueError("Audio data too short to be a valid WAV file")
    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    # Walk chunks to find 'fmt ' and 'data'
    byte_rate = 0
    data_size = 0
    pos = 12  # skip RIFF header + 'WAVE'

    while pos + 8 <= len(audio_bytes):
        chunk_id = audio_bytes[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", audio_bytes, pos + 4)[0]

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("Invalid fmt chunk")
            byte_rate = struct.unpack_from("<I", audio_bytes, pos + 16)[0]
        elif chunk_id == b"data":
            data_size = chunk_size
            break

        pos += 8 + chunk_size

    if byte_rate == 0:
        raise ValueError("Could not determine byte rate from WAV header")

    return data_size / byte_rate
