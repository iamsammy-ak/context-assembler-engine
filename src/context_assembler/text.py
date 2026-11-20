"""Small text helpers: words, terms, sentences, identifiers and sentence-boundary trimming."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves please thanks thank hi hello ok okay also get got let us
    """.split()
)


def words(text: str) -> list[str]:
    """Lower-cased alphanumeric words."""
    return [word.lower() for word in _WORD.findall(text)]


@lru_cache(maxsize=65_536)
def s_stem(word: str) -> str:
    """Harman's S-stemmer: folds common English plurals (policies -> policy, invoices -> invoice)."""
    if len(word) > 3 and word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if len(word) > 2 and word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    """Content terms: words minus stop-words, plural-folded."""
    return [s_stem(word) for word in words(text) if word not in STOPWORDS]


def split_identifier(name: str) -> str:
    """``issue_refund`` / ``issueRefund`` / ``issue-refund`` -> ``issue refund``."""
    return " ".join(_CAMEL.sub(" ", name).replace("_", " ").replace("-", " ").split())


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation followed by whitespace, and on line breaks."""
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part and part.strip()]


def longest_fitting_prefix(
    sentences: list[str], fits: Callable[[str], bool], *, suffix: str = " …", joiner: str = " "
) -> str | None:
    """The longest run of leading sentences (plus ``suffix`` when cut short) that ``fits``.

    Binary search over the number of sentences, so a long chunk costs O(log n) token counts. Returns
    ``None`` when not even the first sentence fits.
    """
    best: str | None = None
    low, high = 1, len(sentences)
    while low <= high:
        middle = (low + high) // 2
        text = joiner.join(sentences[:middle])
        if middle < len(sentences):
            text += suffix
        if fits(text):
            best, low = text, middle + 1
        else:
            high = middle - 1
    return best
