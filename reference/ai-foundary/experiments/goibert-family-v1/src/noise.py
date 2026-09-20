"""Deterministic Vietnamese text corruption for noise-consistency training."""

from __future__ import annotations

import math
import random
import re
import unicodedata


TEENCODE_MAP = {
    "không": "ko",
    "được": "dc",
    "với": "vs",
    "gì": "j",
    "mình": "mk",
    "biết": "bit",
    "rồi": "r",
    "bạn": "b",
    "ơi": "oi",
    "vâng": "vg",
    "shop": "shp",
    "anh": "a",
    "chị": "c",
    "em": "e",
    "quá": "wa",
    "yêu": "iu",
    "buồn": "bun",
    "tiền": "xiền",
    "thích": "thik",
    "đi": "dj",
    "chờ": "cho",
    "giờ": "h",
}
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese combining marks and normalize stroked D characters."""
    translated = text.translate(str.maketrans({"đ": "d", "Đ": "D"}))
    decomposed = unicodedata.normalize("NFD", translated)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def teencode(
    text: str,
    rng: random.Random,
    p_word: float = 0.3,
) -> str:
    """Replace known words with Vietnamese chat forms at a per-match probability."""
    if not 0.0 <= p_word <= 1.0:
        raise ValueError("p_word must be between 0 and 1")

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = TEENCODE_MAP.get(word.lower())
        if replacement is None or rng.random() >= p_word:
            return word
        return _match_case(word, replacement)

    return WORD_RE.sub(replace, text)


def char_noise(
    text: str,
    rng: random.Random,
    p_char: float = 0.02,
) -> str:
    """Apply low-rate adjacent swaps, duplicates, and drops to non-space characters."""
    if not 0.0 <= p_char <= 1.0:
        raise ValueError("p_char must be between 0 and 1")
    output: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace() or rng.random() >= p_char:
            output.append(character)
            index += 1
            continue
        operation = rng.randrange(3)
        if operation == 0 and index + 1 < len(text) and not text[index + 1].isspace():
            output.extend((text[index + 1], character))
            index += 2
        elif operation == 1:
            output.extend((character, character))
            index += 1
        else:
            index += 1
    return "".join(output)


def corrupt(text: str, rng: random.Random) -> str:
    """Create one bounded, non-empty noisy view using only the supplied RNG."""
    source = text.strip() or "."
    corrupted = strip_diacritics(source) if rng.random() < 0.5 else source
    corrupted = teencode(corrupted, rng, p_word=0.3)
    corrupted = char_noise(corrupted, rng, p_char=0.02)
    max_length = max(len(source) + 8, math.ceil(len(source) * 1.1))
    corrupted = corrupted[:max_length].strip()
    return corrupted or source


def _self_test() -> None:
    samples = [
        "Bạn ơi, mình không biết giờ đi đâu rồi.",
        "Vâng, em chờ anh với.",
        "Shop có được đổi hàng không?",
        "Chị ơi, giá này quá cao.",
        "Mình thích sản phẩm này quá.",
        "Bạn biết tiền ship là gì không?",
        "Em buồn rồi, anh đi đâu vậy?",
        "Tôi yêu dịch vụ của shop.",
        "Không được thì mình chờ với bạn.",
        "Giờ chị gọi em nhé.",
    ]
    first_rng = random.Random(20260903)
    second_rng = random.Random(20260903)
    first = [corrupt(sample, first_rng) for sample in samples]
    second = [corrupt(sample, second_rng) for sample in samples]
    assert first == second, "fixed seed must produce fixed corruption"
    assert any(noisy != clean for noisy, clean in zip(first, samples)), (
        "at least one sample must change"
    )
    assert all(corrupt(sample, random.Random(index)) for index, sample in enumerate(samples))
    assert corrupt("", random.Random(0))
    assert corrupt("   ", random.Random(1))
    print("noise.py self-test passed")


if __name__ == "__main__":
    _self_test()
