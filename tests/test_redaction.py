"""Sensitive-word redaction, word-list invalidation and publication rules."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.domain import (
    ArchiveStorageError,
    EmptyWordListError,
    WordList,
    WordListChangedError,
    WordListValidationError,
)
from app.main import create_app
from app.redaction import (
    find_spans,
    mask_text,
    merge_spans,
    parse_words,
    redact,
)
from app.store import ArchiveStore

NUMBER = "ZFCG-2026-0001"


def _client(tmp_path: Path, demo):
    app = create_app(
        db_path=str(tmp_path / "archive-redact.db"),
        source_base_url="http://demo.test",
        source_client=demo,
    )
    return app, TestClient(app)


# ------------------------------------------------------------ pure masking

def test_parse_words_strips_dedupes_and_validates():
    assert parse_words("  甲方 \n\n乙方\n甲方\n 乙方 \n") == ["甲方", "乙方"]
    assert parse_words("") == []
    assert parse_words("\n  \n\t\n") == []
    with pytest.raises(WordListValidationError):
        parse_words("\n".join(f"词{i}" for i in range(501)))
    with pytest.raises(WordListValidationError):
        parse_words("甲" * 201)


def test_find_spans_reports_every_exact_literal_hit():
    assert find_spans("abab", ["ab"]) == [(0, 2), (2, 4)]
    assert find_spans("abc", ["xyz"]) == []
    # Regex metacharacters are matched literally, not as a pattern.
    assert find_spans("a.b a.b", ["a.b"]) == [(0, 3), (4, 7)]
    assert find_spans("abc", [""]) == []


def test_merge_spans_unions_overlapping_and_nested_hits():
    assert merge_spans([(0, 2), (1, 3)]) == [(0, 3)]
    assert merge_spans([(5, 6), (0, 2), (1, 4)]) == [(0, 4), (5, 6)]
    assert merge_spans([(0, 2), (3, 5)]) == [(0, 2), (3, 5)]
    assert merge_spans([]) == []


def test_mask_text_masks_all_hits_and_merges_overlaps():
    assert mask_text("abcXYZabc", ["abc", "abcXYZ", "XYZabc"]) == "█" * 9
    # Every occurrence is covered, length preserved (one mask char per point).
    text = "市政道路照明设备采购，市政采购重复采购"
    out = mask_text(text, ["市政", "采购", "照明设备"])
    assert len(out) == len(text)
    assert "市政" not in out and "采购" not in out and "照明设备" not in out
    # 市政 x2 (4 chars) + 采购 x3 (6 chars) + 照明设备 x1 (4 chars) = 14.
    assert out.count("█") == 14
    # No words -> text returned unchanged.
    assert mask_text("保持原样", ()) == "保持原样"


def test_redact_covers_title_and_body_independently():
    title, body = redact(
        "标题含甲方与乙方", "正文再次提到甲方，甲方出现两次。", ["甲方", "乙方"]
    )
    assert "甲方" not in title and "乙方" not in title
    assert "甲方" not in body
    assert "正文再次提到" in body  # non-matching text stays intact


# ------------------------------------------------------------- word list

def test_default_word_list_is_empty(store: ArchiveStore):
    word_list = store.get_word_list()
    assert word_list == WordList(version=0, updated_at=None, words=())


def test_word_list_effective_change_bumps_version_and_noop_keeps_it(
    store: ArchiveStore,
):
    first = store.replace_word_list(["甲方", "乙方"])
    assert first.version == 1 and first.words == ("甲方", "乙方")

    # Same set, different order/whitespace handling happens in parse_words;
    # at store level an identical set is a no-op.
    again = store.replace_word_list(["乙方", "甲方"])
    assert again.version == 1 and again.words == ("甲方", "乙方")

    second = store.replace_word_list(["甲方", "丙方"])
    assert second.version == 2

    # Clearing to empty is still an effective change.
    emptied = store.replace_word_list([])
    assert emptied.version == 3 and emptied.words == ()


def test_word_list_write_failure_rolls_back_to_previous_list(
    store: ArchiveStore, monkeypatch
):
    store.replace_word_list(["甲方", "乙方"])
    real_connect = store._connect

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *params):
            if "INSERT INTO sensitive_words" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *params)

        def executemany(self, sql, *params):
            if "INSERT INTO sensitive_words" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.executemany(sql, *params)

    monkeypatch.setattr(store, "_connect", lambda: FailingConnection(real_connect()))
    with pytest.raises(ArchiveStorageError):
        store.replace_word_list(["丙方"])
    monkeypatch.setattr(store, "_connect", real_connect)

    current = store.get_word_list()
    assert current.version == 1
    assert current.words == ("甲方", "乙方")


# ------------------------------------------------------- HTTP publication

def test_empty_word_list_blocks_publishing(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    client.post("/capture", data={"number": NUMBER})

    version_page = client.get(f"/notice/{NUMBER}/version/1")
    assert "词表为空，按规定不得发布" in version_page.text
    assert "disabled" in version_page.text

    response = client.post(
        f"/notice/{NUMBER}/version/1/publish", follow_redirects=False
    )
    assert response.status_code == 303
    page = client.get(response.headers["location"])
    assert "敏感词表为空" in page.text
    assert app.state.store.list_publications() == []


def test_published_page_contains_only_complete_masked_text(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    client.post("/capture", data={"number": NUMBER})
    # Put a sensitive term into a second revision so body masking is visible.
    demo.post(
        f"/notice/{NUMBER}/edit",
        data={"title": "市政道路照明设备采购招标公告",
              "body": "采购人甲方联系电话13800000000，甲方要求保密。"},
        follow_redirects=False,
    )
    client.post("/capture", data={"number": NUMBER})
    client.post(
        "/words",
        data={"raw_words": "甲方\n13800000000\n照明设备\n市政"},
        follow_redirects=False,
    )

    response = client.post(
        f"/notice/{NUMBER}/version/2/publish", follow_redirects=False
    )
    assert response.status_code == 303
    token = app.state.store.list_publications()[0]["token"]
    assert len(token) >= 20  # unguessable, non-sequential

    page = client.get(f"/p/{token}")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    # Every sensitive occurrence in title and body is gone.
    for secret in ("甲方", "13800000000", "照明设备", "市政"):
        assert secret not in page.text
    assert "█" in page.text
    # The full text is still present: masking is character-for-character.
    record = app.state.store.get_version(NUMBER, 2)
    publication = app.state.store.get_publication_by_token(token)
    assert len(publication["masked_body"]) == len(record.body)
    assert len(publication["masked_title"]) == len(record.title)
    # No way back to the original or its attachment from the public page.
    assert 'href="' not in page.text
    assert "/attachment" not in page.text
    assert "/notice/" not in page.text
    assert record.title not in page.text


def test_publication_write_failure_keeps_previous_state(tmp_path, demo, monkeypatch):
    app, client = _client(tmp_path, demo)
    store = app.state.store
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "甲方"}, follow_redirects=False)
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)
    before = store.list_publications()
    assert len(before) == 1
    old_token = before[0]["token"]

    real_connect = store._connect

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *params):
            if "INSERT INTO publications" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *params)

    monkeypatch.setattr(store, "_connect", lambda: FailingConnection(real_connect()))
    # Direct service call surfaces the storage failure without partial rows.
    from app.redaction import PublicationService

    with pytest.raises(ArchiveStorageError):
        PublicationService(store).publish(NUMBER, 1)
    monkeypatch.setattr(store, "_connect", real_connect)

    after = store.list_publications()
    assert len(after) == 1 and after[0]["token"] == old_token
    # The old page remains readable: the failed write changed nothing visible.
    assert client.get(f"/p/{old_token}").status_code == 200


def test_effective_word_change_expires_old_page_and_republish_revives(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    store = app.state.store
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政\n照明设备"},
                follow_redirects=False)
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)
    old_token = store.list_publications()[0]["token"]
    assert client.get(f"/p/{old_token}").status_code == 200

    # Effective change: the previously published page dies immediately.
    client.post("/words", data={"raw_words": "市政\n照明设备\n采购"},
                follow_redirects=False)
    expired = client.get(f"/p/{old_token}")
    assert expired.status_code == 410
    assert "已失效" in expired.text
    assert "█" not in expired.text  # stale masked text must not be served
    assert 'href="' not in expired.text

    # A no-op resubmission does not revive or further change the version.
    version_before = store.get_word_list().version
    client.post("/words", data={"raw_words": "采购\n照明设备\n市政"},
                follow_redirects=False)
    assert store.get_word_list().version == version_before
    assert client.get(f"/p/{old_token}").status_code == 410

    # Re-publish under the new list: new token, old link stays dead.
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)
    new_token = store.list_publications()[0]["token"]
    assert new_token != old_token
    assert client.get(f"/p/{old_token}").status_code == 404
    revived = client.get(f"/p/{new_token}")
    assert revived.status_code == 200
    assert "采购" not in revived.text and "市政" not in revived.text

    # The registry labels states correctly.
    registry = client.get("/publications")
    assert "当前有效" in registry.text


def test_stale_publication_surfaced_through_store_helpers(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    store = app.state.store
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政"}, follow_redirects=False)
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)
    token = store.list_publications()[0]["token"]
    assert store.get_publication_by_token(token)["is_current"] is True

    client.post("/words", data={"raw_words": "市政\n绿化"},
                follow_redirects=False)
    assert store.get_publication_by_token(token)["is_current"] is False
    assert store.list_publications()[0]["is_current"] is False
    assert store.list_publications_for_version(NUMBER, 1)[0]["is_current"] is False


def test_publish_retries_when_list_moves_midway(tmp_path, demo, monkeypatch):
    app, client = _client(tmp_path, demo)
    store = app.state.store
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政"}, follow_redirects=False)

    real_create = store.create_publication
    calls = {"n": 0}

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate another request replacing the list between the
            # service's read and its write.
            store.replace_word_list(["市政", "采购"])
            raise WordListChangedError("stale")
        return real_create(**kwargs)

    monkeypatch.setattr(store, "create_publication", flaky_create)
    publication = app.state.publication_service.publish(NUMBER, 1)
    monkeypatch.setattr(store, "create_publication", real_create)

    assert calls["n"] == 2
    assert publication["word_list_version"] == store.get_word_list().version
    # The stored masked text matches the *current* list's complete result.
    record = store.get_version(NUMBER, 1)
    expected_title, expected_body = redact(
        record.title, record.body, store.get_word_list().words
    )
    assert publication["masked_title"] == expected_title
    assert publication["masked_body"] == expected_body
    assert client.get(f"/p/{publication['token']}").status_code == 200


def test_publish_unknown_revision_is_rejected(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政"}, follow_redirects=False)
    response = client.post(
        f"/notice/{NUMBER}/version/99/publish", follow_redirects=False
    )
    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert app.state.store.list_publications() == []


def test_unknown_token_is_404(tmp_path, demo):
    _, client = _client(tmp_path, demo)
    response = client.get("/p/does-not-exist")
    assert response.status_code == 404
    assert "公开页不存在" in response.text
    assert 'href="' not in response.text


def test_word_list_validation_error_is_shown_without_changing_state(
    tmp_path, demo
):
    app, client = _client(tmp_path, demo)
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政"}, follow_redirects=False)

    response = client.post(
        "/words",
        data={"raw_words": "\n".join(f"词{i}" for i in range(501))},
        follow_redirects=False,
    )
    assert response.status_code == 303
    page = client.get(response.headers["location"])
    assert "上限" in page.text
    # Rejected submission leaves the previous usable list in place.
    current = app.state.store.get_word_list()
    assert current.version == 1 and current.words == ("市政",)


def test_clearing_word_list_expires_pages_and_blocks_republish(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    store = app.state.store
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政\n采购"},
                follow_redirects=False)
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)
    token = store.list_publications()[0]["token"]
    assert client.get(f"/p/{token}").status_code == 200

    client.post("/words", data={"raw_words": ""}, follow_redirects=False)
    assert store.get_word_list().words == ()
    assert client.get(f"/p/{token}").status_code == 410

    response = client.post(
        f"/notice/{NUMBER}/version/1/publish", follow_redirects=False
    )
    page = client.get(response.headers["location"])
    assert "敏感词表为空" in page.text
    # Still only the stale row; nothing new was published.
    publications = store.list_publications()
    assert len(publications) == 1 and publications[0]["is_current"] is False


def test_demo_site_and_archive_downloads_remain_untouched(tmp_path, demo):
    app, client = _client(tmp_path, demo)
    client.post("/capture", data={"number": NUMBER})
    client.post("/words", data={"raw_words": "市政"}, follow_redirects=False)
    client.post(f"/notice/{NUMBER}/version/1/publish", follow_redirects=False)

    # The demo source still serves full original content.
    source = demo.get(f"/api/notices/{NUMBER}")
    assert source.status_code == 200
    assert "市政" in source.json()["body"]
    # Archive browsing keeps the original text and the attachment download.
    notice_page = client.get(f"/notice/{NUMBER}")
    assert notice_page.status_code == 200
    download = client.get(f"/notice/{NUMBER}/version/1/attachment")
    assert download.status_code == 200
    assert download.headers["x-content-sha256"]
    assert download.content == demo.get(
        f"/api/notices/{NUMBER}/attachment"
    ).content


def test_interleaved_publish_and_word_changes_only_expose_current_result(
    tmp_path, demo
):
    """Aggressive interleaving: a publisher and a word-list editor race.

    Every externally visible publication must either be gone or exactly match
    the current word list's complete masking result.
    """
    app, client = _client(tmp_path, demo)
    store = app.state.store
    service = app.state.publication_service
    client.post("/capture", data={"number": NUMBER})
    store.replace_word_list(["市政", "采购"])

    alternatives = [
        ["市政", "采购"],
        ["市政", "照明设备"],
        ["采购", "绿化", "市政"],
        ["照明设备"],
    ]
    errors: list[Exception] = []

    def publisher():
        for _ in range(40):
            try:
                service.publish(NUMBER, 1)
            except WordListChangedError as exc:
                # Under continuous edits every retry slot may be lost; the
                # important contract is that no stale page is ever visible.
                errors.append(exc)
            except EmptyWordListError as exc:  # pragma: no cover - lists stay full
                errors.append(exc)
            except ArchiveStorageError as exc:
                errors.append(exc)

    def editor():
        for index in range(40):
            try:
                store.replace_word_list(alternatives[index % len(alternatives)])
            except ArchiveStorageError as exc:  # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=publisher), threading.Thread(target=editor)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Storage errors are not allowed; a lost retry race is tolerated.
    assert not [
        exc for exc in errors if not isinstance(exc, WordListChangedError)
    ]

    # Once the editor stops, a final publication must settle on the current list.
    settled = service.publish(NUMBER, 1)
    word_list = store.get_word_list()
    publications = store.list_publications()
    # One revision has at most one publication row and it must match the list.
    assert len(publications) == 1
    only = publications[0]
    assert only["token"] == settled["token"]
    assert only["word_list_version"] == word_list.version
    assert only["is_current"] is True

    record = store.get_version(NUMBER, 1)
    expected_title, expected_body = redact(
        record.title, record.body, word_list.words
    )
    assert only["masked_title"] == expected_title
    assert only["masked_body"] == expected_body

    page = client.get(f"/p/{only['token']}")
    assert page.status_code == 200
    for word in word_list.words:
        assert word not in page.text

    # Original-archive browsing and attachment download are unaffected.
    assert client.get(f"/notice/{NUMBER}").status_code == 200
    download = client.get(f"/notice/{NUMBER}/version/1/attachment")
    assert download.status_code == 200 and download.content
