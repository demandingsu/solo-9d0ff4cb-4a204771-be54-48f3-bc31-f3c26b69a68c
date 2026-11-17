"""Crash safety for word-list and publication transactions.

A hard process exit before COMMIT must leave the previous word list and any
previously published page exactly as they were.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_script(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    script = tmp_path / "crash_writer.py"
    script.write_text(
        textwrap.dedent(
            """
            import os, sqlite3, sys
            sys.path.insert(0, sys.argv[1])
            from app.store import ArchiveStore
            db = sys.argv[2]
            store = ArchiveStore(db)
            store.replace_word_list(['甲方', '乙方'])
            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, "-I", str(script), str(ROOT), str(tmp_path / "w.db")],
        capture_output=True,
        text=True,
    )


def test_kill_during_word_list_replacement_keeps_old_list(tmp_path: Path):
    result = _run_script(
        tmp_path,
        """
        conn = sqlite3.connect(db, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        # Half of a replacement transaction, then a hard kill.
        conn.execute("DELETE FROM sensitive_words")
        conn.execute("INSERT INTO sensitive_words (word) VALUES ('丙方')")
        os._exit(137)
        """,
    )
    assert result.returncode == 137, result.stderr

    sys.path.insert(0, str(ROOT))
    from app.store import ArchiveStore

    store = ArchiveStore(tmp_path / "w.db")
    word_list = store.get_word_list()
    assert word_list.version == 1
    assert word_list.words == ("甲方", "乙方")


def test_kill_during_publication_insert_leaves_no_publication(tmp_path: Path):
    result = _run_script(
        tmp_path,
        """
        # Build one archived revision directly.
        import hashlib
        from app.domain import (
            AttachmentSnapshot, SourceSnapshot, VersionRecord,
        )
        blob = b'payload'
        snap = SourceSnapshot(
            number='PUB-1', title='甲方公告', body='甲方正文',
            body_sha256=hashlib.sha256('甲方正文'.encode()).hexdigest(),
            revision=1, body_revision=1, attachment_revision=1,
            retracted=False, source_url='u',
            attachment=AttachmentSnapshot(
                filename='a.txt', content_type='text/plain',
                size=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                data=blob))
        store.insert_version(
            VersionRecord.from_snapshot(snap, '2026-01-01T00:00:00Z'),
            outcome='created', detail='seed')

        conn = sqlite3.connect(db, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO publications (token, announcement_number, revision,"
            " word_list_version, masked_title, masked_body, created_at)"
            " VALUES ('tok', 'PUB-1', 1, 1, 't', 'b', '2026-01-01T00:00:00Z')"
        )
        os._exit(137)
        """,
    )
    assert result.returncode == 137, result.stderr

    sys.path.insert(0, str(ROOT))
    from app.store import ArchiveStore

    store = ArchiveStore(tmp_path / "w.db")
    assert store.list_publications() == []
    assert store.get_publication_by_token("tok") is None
    # The archived version itself is intact and still downloadable.
    record = store.get_version("PUB-1", 1)
    assert record is not None and record.attachment_blob == b"payload"
    assert store.get_word_list().words == ("甲方", "乙方")
