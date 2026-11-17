"""Global sensitive-word list parsing and exact-match redaction.

Masking uses literal (non-regex) substring matching against the original
archived text. Every hit in the title and body is replaced by a fixed-width
block of mask characters. When several words produce overlapping hits the
spans are merged into one continuous masked block, so a character is never
partially exposed or masked twice.

A publication snapshots both the word-list version and the masked text.
Because word-list replacements are single transactions and the version only
moves forward, a public page can be validated against the current list with a
simple version comparison.
"""
from __future__ import annotations

import secrets

from .domain import (
    EmptyWordListError,
    PublicationTargetError,
    WordListChangedError,
    WordListValidationError,
)
from .store import ArchiveStore

MAX_WORDS = 500
MAX_WORD_LENGTH = 200
MASK_CHAR = "█"
PUBLISH_RETRIES = 10


def parse_words(raw: str) -> list[str]:
    """Split a textarea payload into de-duplicated, stripped, non-empty words.

    One word per line. Lines are stripped of surrounding whitespace; blank
    lines and repeated words are ignored. Raises WordListValidationError when
    the limits are exceeded so the store is never asked to persist them.
    """
    text = raw or ""
    words: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        word = line.strip()
        if not word or word in seen:
            continue
        if len(word) > MAX_WORD_LENGTH:
            raise WordListValidationError(
                f"敏感词“{word[:20]}…”超过 {MAX_WORD_LENGTH} 字上限，未保存。"
            )
        seen.add(word)
        words.append(word)
    if len(words) > MAX_WORDS:
        raise WordListValidationError(
            f"敏感词数量超过 {MAX_WORDS} 个上限，未保存。"
        )
    return words


def find_spans(text: str, words: tuple[str, ...] | list[str]) -> list[tuple[int, int]]:
    """Return every literal hit as half-open [start, end) code-point spans."""
    spans: list[tuple[int, int]] = []
    for word in words:
        if not word:
            continue
        start = 0
        while True:
            index = text.find(word, start)
            if index < 0:
                break
            spans.append((index, index + len(word)))
            # Non-overlapping per word, mirroring str.replace semantics; spans
            # from different words are merged afterwards by merge_spans().
            start = index + len(word)
    return spans


def merge_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Union overlapping (and adjacent) hit spans into sorted merged spans."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def mask_text(text: str, words: tuple[str, ...] | list[str]) -> str:
    """Replace every exact-match hit with mask blocks, overlaps merged."""
    spans = merge_spans(find_spans(text, words))
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            chars[index] = MASK_CHAR
    return "".join(chars)


def redact(
    title: str, body: str, words: tuple[str, ...] | list[str]
) -> tuple[str, str]:
    return mask_text(title, words), mask_text(body, words)


def new_public_token() -> str:
    """Unguessable public-page identifier (no sequential/guessable ids)."""
    return secrets.token_urlsafe(24)


class PublicationService:
    """Orchestrates word-list-guarded, atomic publication writes."""

    def __init__(self, store: ArchiveStore):
        self.store = store

    def publish(self, number: str, revision: int) -> dict:
        """Publish one archived revision under the current word list.

        The archived version is immutable and the masked snapshot is computed
        in application code, then bound to the word-list version inside one
        database transaction. If the list changes in between, the store
        rejects the stale write and we retry against the new list.
        """
        record = self.store.get_version(number, revision)
        if record is None:
            raise PublicationTargetError("要发布的存档版本不存在。")

        last_error: WordListChangedError | None = None
        for _ in range(PUBLISH_RETRIES):
            word_list = self.store.get_word_list()
            if not word_list.words:
                raise EmptyWordListError(
                    "当前全局敏感词表为空，按规定不能发布脱敏公开页；"
                    "请先在“敏感词表”页录入至少一个敏感词后再发布。"
                )
            masked_title, masked_body = redact(
                record.title, record.body, word_list.words
            )
            try:
                return self.store.create_publication(
                    announcement_number=number,
                    revision=revision,
                    word_list_version=word_list.version,
                    masked_title=masked_title,
                    masked_body=masked_body,
                    token=new_public_token(),
                )
            except WordListChangedError as exc:
                # The list was replaced mid-publish: discard the stale masked
                # text and retry with the current list.
                last_error = exc
        assert last_error is not None
        raise last_error
