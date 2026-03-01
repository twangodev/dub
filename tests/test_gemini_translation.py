import json
from unittest.mock import MagicMock, patch

import pytest

from dub.models.schemas import Segment, TranslatedSegment, Word
from dub.providers.translation.gemini import (
    CONTEXT_SEGMENTS,
    MAX_WINDOW_CHARS,
    PAUSE_THRESHOLD,
    GeminiTranslation,
    TranslationResponse,
    Window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_word(text: str, start: float, end: float) -> Word:
    return Word(start=start, end=end, text=text)


def make_segment(text: str, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(start=start, end=end, text=text)


def make_translated(original: str, translated: str, start: float = 0.0, end: float = 1.0) -> TranslatedSegment:
    return TranslatedSegment(start=start, end=end, original_text=original, translated_text=translated)


def make_response(blocks: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.text = json.dumps({"blocks": blocks})
    return mock


# ---------------------------------------------------------------------------
# Window dataclass
# ---------------------------------------------------------------------------

class TestWindow:
    def test_empty_char_count(self):
        assert Window().char_count == 0

    def test_char_count_sums_segments(self):
        w = Window(segments=[make_segment("hello"), make_segment("world")])
        assert w.char_count == len("hello") + len("world")

    def test_default_empty_segments(self):
        assert Window().segments == []


# ---------------------------------------------------------------------------
# _group_into_sentences
# ---------------------------------------------------------------------------

class TestGroupIntoSentences:
    def test_empty_returns_empty(self):
        assert GeminiTranslation._group_into_sentences([]) == []

    def test_single_word(self):
        words = [make_word("hello", 0.0, 0.5)]
        result = GeminiTranslation._group_into_sentences(words)
        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].start == 0.0
        assert result[0].end == 0.5

    def test_words_within_threshold_grouped(self):
        words = [make_word("hello", 0.0, 0.5), make_word("world", 0.6, 1.0)]
        result = GeminiTranslation._group_into_sentences(words, pause_threshold=0.7)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_gap_above_threshold_splits(self):
        words = [
            make_word("hello", 0.0, 0.5),
            make_word("world", 1.3, 1.8),  # gap = 0.8 > 0.7
        ]
        result = GeminiTranslation._group_into_sentences(words, pause_threshold=0.7)
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_exact_threshold_splits(self):
        words = [make_word("a", 0.0, 0.5), make_word("b", 1.2, 1.5)]  # gap = 0.7
        result = GeminiTranslation._group_into_sentences(words, pause_threshold=0.7)
        assert len(result) == 2

    def test_timestamps_correct(self):
        words = [make_word("a", 1.0, 1.5), make_word("b", 1.6, 2.0)]
        result = GeminiTranslation._group_into_sentences(words)
        assert result[0].start == 1.0
        assert result[0].end == 2.0

    def test_all_words_accounted_for(self):
        words = [make_word(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)]
        result = GeminiTranslation._group_into_sentences(words)
        recovered = " ".join(seg.text for seg in result)
        assert all(f"w{i}" in recovered for i in range(10))


# ---------------------------------------------------------------------------
# _build_windows
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def test_empty_returns_empty(self):
        assert GeminiTranslation._build_windows([], max_chars=100) == []

    def test_all_fit_in_one_window(self):
        segs = [make_segment("hi"), make_segment("there")]
        windows = GeminiTranslation._build_windows(segs, max_chars=1000)
        assert len(windows) == 1
        assert windows[0].segments == segs

    def test_splits_on_budget(self):
        segs = [make_segment("a" * 50), make_segment("b" * 50), make_segment("c" * 50)]
        windows = GeminiTranslation._build_windows(segs, max_chars=60)
        assert len(windows) == 3

    def test_oversized_segment_gets_own_window(self):
        big = make_segment("x" * 5000)
        small = make_segment("y" * 10)
        windows = GeminiTranslation._build_windows([big, small], max_chars=100)
        assert len(windows) == 2
        assert windows[0].segments == [big]

    def test_no_data_loss(self):
        segs = [make_segment(f"s{i}") for i in range(20)]
        windows = GeminiTranslation._build_windows(segs, max_chars=10)
        recovered = [s for w in windows for s in w.segments]
        assert recovered == segs


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def setup_method(self):
        self.provider = GeminiTranslation(api_key="test")

    def test_no_context_omits_context_section(self):
        window = Window(segments=[make_segment("Hello", 0.0, 1.0)])
        prompt = self.provider._build_prompt(window, [], "English", "Spanish")
        assert "Previous translations" not in prompt
        assert "Hello" in prompt

    def test_context_includes_original_and_translated(self):
        window = Window(segments=[make_segment("New")])
        ctx = [make_translated("Hello", "Hola")]
        prompt = self.provider._build_prompt(window, ctx, "English", "Spanish")
        assert "Previous translations" in prompt
        assert "Hello" in prompt
        assert "Hola" in prompt
        assert "→" in prompt

    def test_context_do_not_include_instruction(self):
        window = Window(segments=[make_segment("New")])
        ctx = [make_translated("A", "B")]
        prompt = self.provider._build_prompt(window, ctx, "English", "Spanish")
        assert "do NOT include in output" in prompt

    def test_segments_include_timestamps(self):
        window = Window(segments=[make_segment("Hello", 1.5, 3.25)])
        prompt = self.provider._build_prompt(window, [], "English", "Spanish")
        assert "1.50s" in prompt
        assert "3.25s" in prompt

    def test_segments_numbered(self):
        window = Window(segments=[make_segment("First"), make_segment("Second")])
        prompt = self.provider._build_prompt(window, [], "English", "Spanish")
        assert "1." in prompt
        assert "2." in prompt

    def test_languages_in_prompt(self):
        window = Window(segments=[make_segment("text")])
        prompt = self.provider._build_prompt(window, [], "Japanese", "French")
        assert "Japanese" in prompt
        assert "French" in prompt


# ---------------------------------------------------------------------------
# translate_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTranslateChunks:
    async def test_no_api_key_returns_stubs(self):
        provider = GeminiTranslation(api_key="")
        words = [make_word("Hello", 0.0, 0.5), make_word("world", 0.6, 1.0)]
        result = await provider.translate_chunks(words, "Spanish")
        assert len(result) == 1  # grouped into one sentence
        assert "[Spanish]" in result[0].translated_text

    async def test_successful_translation(self):
        provider = GeminiTranslation(api_key="fake-key")
        words = [
            make_word("Hello", 0.0, 0.4),
            make_word("world", 0.5, 0.9),
            # big gap → new sentence
            make_word("Goodbye", 2.0, 2.5),
        ]
        response_data = [
            {"start": 0.0, "end": 0.9, "translated_text": "Hola mundo"},
            {"start": 2.0, "end": 2.5, "translated_text": "Adiós"},
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_response(response_data)

        with patch("google.genai.Client", return_value=mock_client), \
             patch("asyncio.to_thread", side_effect=lambda fn, **kw: fn(**kw)):
            result = await provider.translate_chunks(words, "Spanish", "English")

        assert len(result) == 2
        assert result[0].translated_text == "Hola mundo"
        assert result[1].translated_text == "Adiós"
        assert result[0].original_text == "Hello world"

    async def test_api_error_falls_back_to_original(self):
        provider = GeminiTranslation(api_key="fake-key")
        words = [make_word("Hello", 0.0, 0.5)]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")

        with patch("google.genai.Client", return_value=mock_client), \
             patch("asyncio.to_thread", side_effect=lambda fn, **kw: fn(**kw)):
            result = await provider.translate_chunks(words, "Spanish", "English")

        assert len(result) == 1
        assert result[0].translated_text == "Hello"

    async def test_rolling_context_has_original_and_translated(self):
        provider = GeminiTranslation(api_key="fake-key")
        # Two windows: first word is one sentence, second word (big gap) is another
        words = [make_word("First", 0.0, 0.5), make_word("Second", 5.0, 5.5)]
        windows = [
            Window(segments=[make_segment("First", 0.0, 0.5)]),
            Window(segments=[make_segment("Second", 5.0, 5.5)]),
        ]
        prompts_seen: list[str] = []

        def fake_generate(**kwargs):
            prompts_seen.append(kwargs["contents"])
            n = len(windows[len(prompts_seen) - 1].segments)
            return make_response([
                {"start": 0.0, "end": 0.5, "translated_text": f"Trans{len(prompts_seen)}"}
                for _ in range(n)
            ])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = fake_generate

        with patch("dub.providers.translation.gemini.GeminiTranslation._build_windows",
                   return_value=windows), \
             patch("google.genai.Client", return_value=mock_client), \
             patch("asyncio.to_thread", side_effect=lambda fn, **kw: fn(**kw)):
            await provider.translate_chunks(words, "Spanish", "English")

        # Second prompt should have both original and translated in context
        assert "→" in prompts_seen[1]
        assert "First" in prompts_seen[1]
        assert "Trans1" in prompts_seen[1]

    async def test_output_count_matches_sentence_count(self):
        provider = GeminiTranslation(api_key="fake-key")
        # 5 closely spaced words → 1 sentence
        words = [make_word(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(5)]
        response_data = [{"start": 0.0, "end": 0.45, "translated_text": "translated"}]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_response(response_data)

        with patch("google.genai.Client", return_value=mock_client), \
             patch("asyncio.to_thread", side_effect=lambda fn, **kw: fn(**kw)):
            result = await provider.translate_chunks(words, "Spanish")

        assert len(result) == 1

    async def test_timestamps_come_from_input_not_model(self):
        """start/end on TranslatedSegment should match input sentence boundaries."""
        provider = GeminiTranslation(api_key="fake-key")
        words = [make_word("Hello", 1.23, 1.99)]
        # Model returns different timestamps — we should ignore them
        response_data = [{"start": 99.0, "end": 99.9, "translated_text": "Hola"}]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_response(response_data)

        with patch("google.genai.Client", return_value=mock_client), \
             patch("asyncio.to_thread", side_effect=lambda fn, **kw: fn(**kw)):
            result = await provider.translate_chunks(words, "Spanish")

        assert result[0].start == pytest.approx(1.23)
        assert result[0].end == pytest.approx(1.99)
