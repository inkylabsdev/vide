"""Unit tests for the pure subtitle logic (vide.subtitles).

These functions have no GPU/model dependency, so they are tested directly
against crafted cue/segment data rather than through the CLI.
"""

import vide.subtitles as S


def w(word, start, end, score=0.9, **extra):
    """A whisperx-style aligned word entry."""
    return {"word": word, "start": start, "end": end, "score": score, **extra}


# --------------------------------------------------------------------------- #
# format_timestamp
# --------------------------------------------------------------------------- #
def test_format_timestamp_none_and_negative():
    assert S.format_timestamp(None) == "00:00:00,000"
    assert S.format_timestamp(-1) == "00:00:00,000"


def test_format_timestamp_basic():
    assert S.format_timestamp(3661.5) == "01:01:01,500"


def test_format_timestamp_millisecond_rollover():
    # 59.9999s must roll over to 00:01:00,000, not the invalid 00:00:60,000.
    assert S.format_timestamp(59.9999) == "00:01:00,000"


# --------------------------------------------------------------------------- #
# Line wrapping
# --------------------------------------------------------------------------- #
def test_greedy_lines_wraps_and_keeps_oversize_word():
    lines = S._greedy_lines(["a", "b", "cccccc"], max_chars=3)
    assert lines == [["a", "b"], ["cccccc"]]


def test_balance_two_prefers_punctuation_and_penalizes_function_word():
    # Breaking after "there," (punctuation) beats other splits.
    words = ["hello", "there,", "my", "old", "friend"]
    left, right = S._balance_two(words, max_chars=42)
    assert left == ["hello", "there,"]


def test_balance_two_penalizes_break_after_function_word():
    # A candidate split right after "the" is penalized, so the break lands
    # elsewhere even though it is closer to the middle.
    words = ["please", "keep", "the", "small", "cat", "warm"]
    left, right = S._balance_two(words, max_chars=42)
    assert left[-1] != "the"


def test_balance_two_returns_none_when_no_split_fits():
    assert S._balance_two(["x" * 50, "y" * 50], max_chars=42) is None


def test_split_subtitle_empty_and_single_line():
    assert S.split_subtitle("") == ""
    assert S.split_subtitle("short text") == "short text"


def test_split_subtitle_balances_two_lines():
    text = "the quick brown fox jumps over the lazy dog while the sun sets"
    out = S.split_subtitle(text, max_chars=32)
    assert out.count("\n") == 1
    assert all(len(line) <= 32 for line in out.split("\n"))


def test_split_subtitle_keeps_greedy_when_balance_impossible():
    # Two oversize words: greedy gives 2 lines, _balance_two returns None.
    out = S.split_subtitle("x" * 50 + " " + "y" * 50, max_chars=42)
    assert out == "x" * 50 + "\n" + "y" * 50


def test_line_count_empty_is_zero():
    assert S._line_count("", 42) == 0
    assert S._line_count("one two", 42) == 1


# --------------------------------------------------------------------------- #
# Sentence / clause splitting
# --------------------------------------------------------------------------- #
def test_best_split_index_finds_punctuation():
    words = "one two, three four five six".split()
    assert S._best_split_index(words, 3) == 2  # right after "two,"


def test_best_split_index_falls_back_to_mid():
    words = "one two three four".split()
    assert S._best_split_index(words, 2) == 2


def test_split_to_fit_empty_and_single_word():
    assert S._split_to_fit("   ", 42, 2) == []
    # A single indivisible word is returned as-is even when it cannot fit the
    # (here degenerate, zero-line) budget — it can't be split further.
    assert S._split_to_fit("x" * 80, 10, 0) == ["x" * 80]


def test_split_sentence_heuristically_fits():
    assert S.split_sentence_heuristically("a short one", 42, 2) == ["a short one"]


def test_split_sentence_heuristically_splits_long_sentence():
    long_sentence = " ".join(["word"] * 40) + ", and " + " ".join(["more"] * 40)
    parts = S.split_sentence_heuristically(long_sentence, 42, 2)
    assert len(parts) > 1
    assert all(S._line_count(p, 42) <= 2 for p in parts)


# --------------------------------------------------------------------------- #
# Cue construction helpers
# --------------------------------------------------------------------------- #
def test_cue_speaker_majority_and_none():
    assert S._cue_speaker(None) is None
    assert S._cue_speaker([w("hi", 0, 1)]) is None  # no speaker labels
    words = [
        w("a", 0, 1, speaker="SPEAKER_00"),
        w("b", 1, 2, speaker="SPEAKER_01"),
        w("c", 2, 3, speaker="SPEAKER_00"),
    ]
    assert S._cue_speaker(words) == "SPEAKER_00"


def test_norm_token_strips_punctuation():
    assert S._norm_token("Hello,") == "hello"


def test_align_clause_words_exact_match():
    wd = [w("hello", 0, 1), w("world", 1, 2)]
    matched, ptr = S._align_clause_words(["hello", "world"], wd, 0)
    assert matched == wd and ptr == 2


def test_align_clause_words_resyncs_past_stray_entry():
    # word_data has a stray "uh" the text does not contain; matcher skips it.
    wd = [w("uh", 0, 1), w("hello", 1, 2), w("world", 2, 3)]
    matched, ptr = S._align_clause_words(["hello", "world"], wd, 0)
    assert [m["word"] for m in matched] == ["hello", "world"]
    assert ptr == 3


def test_align_clause_words_skips_token_with_no_aligned_word():
    # "there" is missing from word_data; it contributes no timing but the
    # pointer is kept so later words still line up.
    wd = [w("hello", 0, 1), w("friend", 1, 2)]
    matched, ptr = S._align_clause_words(["hello", "there", "friend"], wd, 0)
    assert [m["word"] for m in matched] == ["hello", "friend"]
    assert ptr == 2


def test_align_clause_words_positional_fallback():
    # No token matches by string; fall back to the positional slice.
    wd = [w("aaa", 0, 1), w("bbb", 1, 2)]
    matched, ptr = S._align_clause_words(["xxx", "yyy"], wd, 0)
    assert matched == wd and ptr == 2


def test_first_start_last_end():
    assert S._first_start([w("a", None, None), w("b", 2.0, 3.0)]) == 2.0
    assert S._last_end([w("a", 1.0, 2.0), w("b", None, None)]) == 2.0
    assert S._first_start([w("a", None, None)]) is None
    assert S._last_end([w("a", None, None)]) is None


def test_segment_sentences_regex_fallback():
    assert S._segment_sentences("One. Two.", None) == ["One.", "Two."]


# --------------------------------------------------------------------------- #
# split_at_sentence_end
# --------------------------------------------------------------------------- #
def test_split_at_sentence_end_with_word_timings():
    text = "Hello world. Nice day."
    wd = [w("Hello", 0.0, 0.5), w("world.", 0.5, 1.0), w("Nice", 1.5, 2.0), w("day.", 2.0, 2.5)]
    cues = S.split_at_sentence_end(None, text, wd)
    assert [c["text"] for c in cues] == ["Hello world.", "Nice day."]
    assert cues[0]["start"] == 0.0 and cues[0]["end"] == 1.0


def test_split_at_sentence_end_whitespace_sentence_skipped():
    assert S.split_at_sentence_end(None, "   ", []) == []


def test_split_at_sentence_end_without_timings_anchors_to_prev_end():
    # No word_data: each clause anchors to the previous cue's end (0.0 first).
    cues = S.split_at_sentence_end(None, "First. Second.", [])
    assert [c["start"] for c in cues] == [0.0, 0.0]
    assert all(c["word_data"] is None for c in cues)


def test_split_at_sentence_end_warns_on_count_mismatch(caplog):
    # 3 text tokens but 2 aligned words triggers the desync-avoidance warning.
    wd = [w("Hello", 0.0, 0.5), w("world.", 0.5, 1.0)]
    with caplog.at_level("WARNING"):
        S.split_at_sentence_end(None, "Hello there world.", wd)
    assert "word/text count mismatch" in caplog.text


# --------------------------------------------------------------------------- #
# Merging / splitting
# --------------------------------------------------------------------------- #
def test_reading_duration_zero_cps():
    assert S._reading_duration("abc", 0) == 0.0
    assert S._reading_duration("abcde", 5) == 1.0


def test_displayed_length_counts_speaker_prefix():
    assert S._displayed_length({"text": "hi"}) == 2
    assert S._displayed_length({"text": "hi", "speaker": "SPEAKER_00"}) == 2 + len("[SPEAKER_00] ")


def test_merge_short_cues_merges_adjacent_short_cue():
    cues = [
        {"text": "Hi", "start": 0.0, "end": 0.3, "word_data": [w("Hi", 0.0, 0.3)]},
        {"text": "there", "start": 0.4, "end": 0.8, "word_data": [w("there", 0.4, 0.8)]},
    ]
    merged = S.merge_short_cues(cues)
    assert len(merged) == 1
    assert merged[0]["text"] == "Hi there"
    assert merged[0]["end"] == 0.8


def test_merge_short_cues_does_not_merge_across_large_gap():
    cues = [
        {"text": "Hi", "start": 0.0, "end": 0.3, "word_data": None},
        {"text": "there", "start": 5.0, "end": 5.4, "word_data": None},
    ]
    assert len(S.merge_short_cues(cues)) == 2


def test_merge_short_cues_none_word_data_stays_none():
    cues = [
        {"text": "Hi", "start": 0.0, "end": 0.3, "word_data": None},
        {"text": "there", "start": 0.4, "end": 0.8, "word_data": [w("there", 0.4, 0.8)]},
    ]
    merged = S.merge_short_cues(cues)
    assert len(merged) == 1 and merged[0]["word_data"] is None


def test_make_chunk_cue_falls_back_to_parent_timing():
    parent = {"start": 4.0, "end": 5.0}
    cue = S._make_chunk_cue(["a", "b"], [w("a", None, None), w("b", None, None)], parent)
    assert cue["start"] == 4.0 and cue["end"] == 5.0


def test_split_long_cue_without_word_timings():
    text = " ".join(["word"] * 40)
    cue = {"text": text, "start": 0.0, "end": 10.0, "word_data": None}
    chunks = S.split_long_cue_without_word_timings(cue, max_line_length=42, max_lines=2)
    assert len(chunks) > 1
    assert abs(chunks[-1]["end"] - 10.0) < 1e-6  # duration distributed to the end


def test_split_long_cue_without_word_timings_empty_text():
    # Degenerate: empty text yields a zero-length chunk (proportion falls to 0).
    cue = {"text": "", "start": 2.0, "end": 5.0, "word_data": None}
    chunks = S.split_long_cue_without_word_timings(cue)
    assert chunks[0]["start"] == 2.0 and chunks[0]["end"] == 2.0


def test_split_at_pauses_splits_on_clause_boundary_pause():
    words = [w("Hello,", 0.0, 0.5), w("world.", 0.5, 1.0), w("Goodbye.", 3.0, 3.5)]
    cue = {"text": "Hello, world. Goodbye.", "start": 0.0, "end": 3.5, "word_data": words}
    out = S.split_at_pauses([cue], pause_threshold=1.0)
    assert len(out) == 2
    assert out[0]["text"] == "Hello, world." and out[1]["text"] == "Goodbye."


def test_split_at_pauses_no_word_data_returns_cue_unchanged():
    cue = {"text": "a b", "start": 0.0, "end": 1.0, "word_data": None}
    assert S.split_at_pauses([cue]) == [cue]


def test_split_at_pauses_no_qualifying_pause():
    words = [w("Hello", 0.0, 0.5), w("world.", 0.5, 1.0)]
    cue = {"text": "Hello world.", "start": 0.0, "end": 1.0, "word_data": words}
    assert S.split_at_pauses([cue]) == [cue]


def test_split_long_cues_with_word_timings_chunks_on_word_timing():
    words = [w(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(30)]
    text = " ".join(f"w{i}" for i in range(30))
    cue = {"text": text, "start": 0.0, "end": 15.0, "word_data": words}
    out = S.split_long_cues_with_word_timings([cue], max_line_length=20, max_lines=2)
    assert len(out) > 1
    assert all(o["word_data"] is not None for o in out)


def test_split_long_cues_with_word_timings_no_word_data_paths():
    fits = {"text": "short", "start": 0.0, "end": 1.0, "word_data": None}
    long = {"text": " ".join(["word"] * 40), "start": 0.0, "end": 10.0, "word_data": None}
    out = S.split_long_cues_with_word_timings([fits, long])
    assert out[0] is fits
    assert len(out) > 2  # the long cue got split into several


def test_split_long_cues_with_word_timings_fits_with_word_data():
    words = [w("hi", 0.0, 0.5), w("there", 0.5, 1.0)]
    cue = {"text": "hi there", "start": 0.0, "end": 1.0, "word_data": words}
    assert S.split_long_cues_with_word_timings([cue]) == [cue]


# --------------------------------------------------------------------------- #
# normalize_cues
# --------------------------------------------------------------------------- #
def test_normalize_cues_empty():
    assert S.normalize_cues([]) == []


def test_normalize_cues_enforces_reading_floor_and_min_gap():
    cues = [
        {"text": "Hello", "start": 0.0, "end": 0.1, "word_data": None},
        {"text": "world", "start": 5.0, "end": 5.1, "word_data": None},
    ]
    out = S.normalize_cues(cues, min_duration=1.0)
    assert out[0]["end"] - out[0]["start"] >= 1.0 - 1e-9  # reading floor
    assert out[1]["start"] >= out[0]["end"] + S.MIN_GAP - 1e-9  # min gap


def test_normalize_cues_caps_lead_out_past_last_word():
    words = [w("Hi", 0.0, 0.5)]
    cues = [{"text": "Hi", "start": 0.0, "end": 20.0, "word_data": words}]
    out = S.normalize_cues(cues, min_duration=0.1, max_cps=1000)
    # end must not linger far past the last spoken word (0.5 + MAX_LEAD_OUT).
    assert out[0]["end"] <= 0.5 + S.MAX_LEAD_OUT + 1e-9


def test_normalize_cues_caps_max_duration():
    cues = [{"text": "x" * 500, "start": 0.0, "end": 100.0, "word_data": None}]
    out = S.normalize_cues(cues, max_duration=7.0, max_cps=1)
    assert out[0]["end"] - out[0]["start"] <= 7.0 + 1e-9


def test_normalize_cues_zero_cps_uses_min_duration():
    cues = [{"text": "hi", "start": 0.0, "end": 0.1, "word_data": None}]
    out = S.normalize_cues(cues, min_duration=1.0, max_cps=0)
    assert out[0]["end"] - out[0]["start"] >= 1.0 - 1e-9


# --------------------------------------------------------------------------- #
# generate_srt / _render_srt
# --------------------------------------------------------------------------- #
def _segment(text, words):
    return {"text": text, "words": words, "start": words[0]["start"], "end": words[-1]["end"]}


def test_generate_srt_end_to_end():
    words = [
        w("Hello", 0.0, 0.4), w("there,", 0.4, 0.8), w("how", 0.9, 1.1),
        w("are", 1.1, 1.3), w("you", 1.3, 1.6), w("today?", 1.6, 2.1),
    ]
    srt = S.generate_srt([_segment("Hello there, how are you today?", words)], "en")
    assert srt.startswith("1\n")
    assert " --> " in srt
    assert "Hello" in srt
    # Blocks are separated by a blank line and the file ends with one.
    assert srt.endswith("\n\n")


def test_generate_srt_unsupported_language_uses_regex_fallback(caplog):
    words = [w("One.", 0.0, 0.5), w("Two.", 1.0, 1.5)]
    with caplog.at_level("WARNING"):
        srt = S.generate_srt([_segment("One. Two.", words)], "zz")
    assert "regex fallback" in caplog.text
    assert srt.count(" --> ") >= 1


def test_render_srt_includes_speaker_prefix():
    cues = [{"text": "hello", "start": 0.0, "end": 1.0, "word_data": None, "speaker": "SPEAKER_00"}]
    out = S._render_srt(cues, 42)
    assert "[SPEAKER_00] hello" in out
