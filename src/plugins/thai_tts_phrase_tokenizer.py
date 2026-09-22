"""Thai TTS phrase tokenizer.

Groups words into TTS-friendly phrases using pythainlp word segmentation and Thai
linguistic boundaries.

Strategy:
1. Tokenize text into words with pythainlp
2. Group words into phrases, splitting at:
   a) Particle + space boundaries (highest priority)
   b) Space boundaries
   c) Max character limit (ensures no phrase exceeds target length)
"""

from __future__ import annotations

import functools

from livekit.agents.tokenize import token_stream, tokenizer
from pythainlp.tokenize import word_tokenize

# Thai sentence-final particles (longer compounds first to avoid partial matches)
_THAI_PARTICLES = frozenset(
    [
        "นะครับ",
        "นะคะ",
        "นะค่ะ",
        "ครับ",
        "ค่ะ",
        "คะ",
        "ไหม",
        "มั้ย",
        "หรือ",
        "หรอ",
        "เหรอ",
        "นะ",
        "สิ",
        "ซิ",
        "เถอะ",
        "หน่อย",
        "เลย",
        "แหละ",
        "หรอก",
        "ล่ะ",
        "จ้ะ",
        "จ๊ะ",
        "จ้า",
        "วะ",
        "ว่ะ",
        "โว้ย",
        "เว้ย",
        "อ่ะ",
    ]
)

MIN_PHRASE_LEN = 30
MAX_PHRASE_LEN = 60


def _is_particle(word: str) -> bool:
    return word in _THAI_PARTICLES


def split_into_phrases(
    text: str, *, min_phrase_len: int = 10
) -> list[tuple[str, int, int]]:
    """Split Thai text into TTS-friendly phrases at particle/space boundaries.

    Splits after a sentence-final particle once the phrase reaches ``min_phrase_len``,
    at a space once it reaches ``MIN_PHRASE_LEN``, or forcibly at ``MAX_PHRASE_LEN``.
    Returns a list of (phrase_text, start_offset, end_offset).
    """
    if not text.strip():
        return []

    if len(text) <= MAX_PHRASE_LEN:
        return [(text, 0, len(text))]

    words = word_tokenize(text, keep_whitespace=True)

    phrases = []
    current_phrase = ""
    phrase_start = 0
    pos = 0

    for i, word in enumerate(words):
        current_phrase += word
        pos += len(word)

        # Determine if this is a split point
        is_boundary = False
        current_len = len(current_phrase.strip())

        # Check if current word is a particle followed by space/end
        if _is_particle(word):
            next_word = words[i + 1] if i + 1 < len(words) else None
            if next_word is None or next_word.isspace():
                is_boundary = current_len >= min_phrase_len

        # Check if current word is whitespace (space boundary)
        elif (
            word.isspace() and current_len >= MIN_PHRASE_LEN
        ) or current_len >= MAX_PHRASE_LEN:
            is_boundary = True

        if is_boundary:
            phrases.append((current_phrase, phrase_start, pos))
            current_phrase = ""
            phrase_start = pos

    # Remaining text
    if current_phrase.strip():
        if phrases and len(current_phrase.strip()) < min_phrase_len:
            # Merge short trailing fragment with previous
            prev_text, prev_start, _ = phrases.pop()
            phrases.append((prev_text + current_phrase, prev_start, pos))
        else:
            phrases.append((current_phrase, phrase_start, pos))

    if not phrases and text.strip():
        phrases.append((text, 0, len(text)))

    return phrases


class ThaiTTSPhraseTokenizer(tokenizer.SentenceTokenizer):
    def __init__(
        self,
        *,
        min_phrase_len: int = 10,
        stream_context_len: int = 10,
    ) -> None:
        self._min_phrase_len = min_phrase_len
        self._stream_context_len = stream_context_len

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [
            s[0] for s in split_into_phrases(text, min_phrase_len=self._min_phrase_len)
        ]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(
            tokenizer=functools.partial(
                split_into_phrases,
                min_phrase_len=self._min_phrase_len,
            ),
            min_token_len=self._min_phrase_len,
            min_ctx_len=self._stream_context_len,
        )
