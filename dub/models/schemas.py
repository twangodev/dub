from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class TranslatedSegment(BaseModel):
    start: float
    end: float
    original_text: str
    translated_text: str


class SeparatedAudio(BaseModel):
    speech_path: Path
    background_path: Path


class JobInfo(BaseModel):
    job_id: str
    status: str = "pending"
    stage: str | None = None
    detail: str | None = None
    target_lang: str = ""
    source_lang: str | None = None
    error: str | None = None
    output_path: str | None = None


class ProgressEvent(BaseModel):
    stage: str
    status: str
    detail: str | None = None
