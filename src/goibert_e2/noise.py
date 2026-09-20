"""Deterministic, bounded Vietnamese text noise for E2b/E2c.

Noise is selected independently for every record from four mutually exclusive
operations: unchanged (50%), accent removal (25%), a small word-boundary-safe
teen-code replacement (20%), or one internal-letter typo (5%).  The random
stream is derived from ``seed``, ``epoch`` and the stable example ID with a
cryptographic hash.  No module-level or global RNG is touched, so dataloader
worker order cannot silently change a corruption.

The input string is never normalised, stripped, truncated, or overwritten.  In
particular digits and punctuation are copied verbatim, and protected negation
words are never passed through the teen-code or typo operation.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence


NO_CHANGE_PROBABILITY = 0.50
ACCENT_PROBABILITY = 0.25
TEENCODE_PROBABILITY = 0.20
TYPO_PROBABILITY = 0.05

# Keep this intentionally small.  The map does not include "không" or any
# other negator: changing a negation word can change the task label.
TEENCODE_MAP: dict[str, str] = {
    "được": "dc",
    "với": "vs",
    "gì": "j",
    "mình": "mk",
    "biết": "bit",
    "rồi": "r",
    "bạn": "b",
    "ơi": "oi",
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

# These words are intentionally protected in all operations that select a
# token.  Accents are still removed by the accent-only operation, because that
# operation is a character-level robustness condition and does not delete a
# token or its negation semantics.
NEGATION_WORDS = frozenset(
    {
        "không",
        "ko",
        "chẳng",
        "chang",
        "chả",
        "cha",
        "chưa",
        "chua",
        "đừng",
        "dừng",
        "dung",
    }
)

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
LETTER_WORD_RE = re.compile(r"(?<!\w)[^\W\d_]+(?!\w)", flags=re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese combining marks while preserving every other char."""

    translated = text.translate(str.maketrans({"đ": "d", "Đ": "D"}))
    decomposed = unicodedata.normalize("NFD", translated)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


remove_accents = strip_diacritics


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _is_protected(word: str) -> bool:
    return word.casefold() in NEGATION_WORDS


def teencode(text: str, rng: random.Random, *, p_word: float = 0.3) -> str:
    """Replace known non-negating words using word boundaries and case.

    ``rng`` must be supplied by the caller so this helper cannot accidentally
    consume the process-global RNG.  It is compatible with the reference
    helper's interface and is useful for deterministic evaluation fixtures.
    """

    if not 0.0 <= p_word <= 1.0:
        raise ValueError("p_word must be between 0 and 1")

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = TEENCODE_MAP.get(word.casefold())
        if replacement is None or _is_protected(word) or rng.random() >= p_word:
            return word
        return _match_case(word, replacement)

    return WORD_RE.sub(replace, text)


def _shift_typo_character(character: str) -> str:
    """Return a nearby alphabetic character while preserving ASCII case."""

    alphabet = "aăâbcdđeêghiklmnoôơpqrstuvwxyz"
    lower = character.casefold()
    if lower not in alphabet:
        # A non-ASCII letter not represented in the compact alphabet is still
        # changed without touching digits/punctuation.
        replacement = "a" if lower != "a" else "e"
    else:
        replacement = alphabet[(alphabet.index(lower) + 1) % len(alphabet)]
    if character.isupper():
        return replacement.upper()
    return replacement


def one_internal_char_typo(text: str, rng: random.Random) -> str:
    """Make one internal-letter transposition/substitution, if possible."""

    candidates = [
        match
        for match in LETTER_WORD_RE.finditer(text)
        if len(match.group(0)) >= 3 and not _is_protected(match.group(0))
    ]
    if not candidates:
        return text
    match = candidates[rng.randrange(len(candidates))]
    word = match.group(0)
    # Internal positions exclude the first and last letter.  A swap is a
    # natural typo and changes no digit or punctuation.  If adjacent letters
    # happen to be equal, use a nearby-character substitution to guarantee an
    # actual edit.
    position = rng.randrange(1, len(word) - 1)
    chars = list(word)
    if chars[position] != chars[position + 1]:
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
    else:
        chars[position] = _shift_typo_character(chars[position])
    return text[: match.start()] + "".join(chars) + text[match.end() :]


def _rng_seed(seed: int, epoch: int, stable_id: object) -> int:
    """Derive a stable, collision-resistant integer seed."""

    payload = (
        f"goibert-e2-noise\0seed={int(seed)}\0epoch={int(epoch)}"
        f"\0stable_id_type={type(stable_id).__qualname__}\0stable_id={stable_id!s}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def make_rng(seed: int, epoch: int, stable_id: object) -> random.Random:
    """Create the record-local RNG; never mutates a global RNG."""

    return random.Random(_rng_seed(seed, epoch, stable_id))


def _operation(rng: random.Random) -> str:
    draw = rng.random()
    if draw < NO_CHANGE_PROBABILITY:
        return "none"
    if draw < NO_CHANGE_PROBABILITY + ACCENT_PROBABILITY:
        return "remove_accents"
    if draw < NO_CHANGE_PROBABILITY + ACCENT_PROBABILITY + TEENCODE_PROBABILITY:
        return "teencode"
    return "internal_char_typo"


@dataclass(frozen=True)
class NoiseResult:
    """One deterministic noisy view and its audit fields."""

    text: str
    operation: str
    changed: bool


def apply_noise(text: str, rng: random.Random) -> NoiseResult:
    """Apply exactly one operation using the supplied record-local RNG."""

    if not isinstance(text, str):
        raise TypeError("noise input must be a string")
    operation = _operation(rng)
    if operation == "none":
        noisy = text
    elif operation == "remove_accents":
        noisy = strip_diacritics(text)
    elif operation == "teencode":
        # p_word=1 makes the chosen operation meaningful while still replacing
        # only the small explicit vocabulary.  If no eligible word exists the
        # raw string is returned and the change-rate audit records that fact.
        noisy = teencode(text, rng, p_word=1.0)
    elif operation == "internal_char_typo":
        noisy = one_internal_char_typo(text, rng)
    else:  # pragma: no cover - _operation is closed over the four cases.
        raise RuntimeError(f"unknown noise operation: {operation}")
    return NoiseResult(text=noisy, operation=operation, changed=noisy != text)


def noisy_view(
    text: str,
    *,
    seed: int,
    epoch: int,
    stable_id: object,
) -> str:
    """Return the deterministic noisy view for one stable training record."""

    return apply_noise(text, make_rng(seed, epoch, stable_id)).text


def corrupt(
    text: str,
    seed: int | random.Random | None = None,
    epoch: int | None = None,
    stable_id: object | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Compatibility wrapper for deterministic and reference-style callers.

    New training code should pass keyword ``seed``, ``epoch`` and
    ``stable_id``.  A caller may instead supply an explicit ``rng`` for a
    local fixture; the global RNG is never used in either form.
    """

    if rng is None and isinstance(seed, random.Random):
        rng = seed
    if rng is None:
        if seed is None or epoch is None or stable_id is None:
            raise ValueError("corrupt requires seed, epoch and stable_id, or an explicit rng")
        rng = make_rng(int(seed), int(epoch), stable_id)
    return apply_noise(text, rng).text


@dataclass(frozen=True)
class NoiseBatch:
    """Batch output plus operation/change-rate metadata for runner logs."""

    texts: tuple[str, ...]
    operations: tuple[str, ...]
    changed: tuple[bool, ...]

    @property
    def change_rate(self) -> float:
        return float(sum(self.changed) / len(self.changed)) if self.changed else 0.0

    def to_dict(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for operation in self.operations:
            counts[operation] = counts.get(operation, 0) + 1
        return {
            "count": len(self.texts),
            "changed_count": int(sum(self.changed)),
            "change_rate": self.change_rate,
            "operation_counts": counts,
        }


def noisy_batch(
    texts: Sequence[str] | Iterable[str],
    stable_ids: Sequence[object] | Iterable[object],
    *,
    seed: int,
    epoch: int,
) -> NoiseBatch:
    """Build deterministic noisy views and return an auditable change rate."""

    text_values = list(texts)
    id_values = list(stable_ids)
    if len(text_values) != len(id_values):
        raise ValueError("texts and stable_ids must have equal lengths")
    results = [
        apply_noise(text, make_rng(seed, epoch, stable_id))
        for text, stable_id in zip(text_values, id_values)
    ]
    return NoiseBatch(
        texts=tuple(result.text for result in results),
        operations=tuple(result.operation for result in results),
        changed=tuple(result.changed for result in results),
    )


def noise_change_rate(
    texts: Sequence[str] | Iterable[str],
    stable_ids: Sequence[object] | Iterable[object],
    *,
    seed: int,
    epoch: int,
) -> float:
    """Convenience helper returning only the actual changed-row rate."""

    return noisy_batch(texts, stable_ids, seed=seed, epoch=epoch).change_rate


__all__ = [
    "ACCENT_PROBABILITY",
    "NEGATION_WORDS",
    "NoiseBatch",
    "NoiseResult",
    "TEENCODE_MAP",
    "TEENCODE_PROBABILITY",
    "TYPO_PROBABILITY",
    "WORD_RE",
    "apply_noise",
    "corrupt",
    "make_rng",
    "noise_change_rate",
    "noisy_batch",
    "noisy_view",
    "one_internal_char_typo",
    "remove_accents",
    "strip_diacritics",
    "teencode",
]
