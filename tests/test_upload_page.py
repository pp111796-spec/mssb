"""upload_page의 순수 검증 로직(_validate) — 세션 ID 형식·중복 체크."""

from review_app.upload_page import _validate


def test_empty_session_id_has_no_error_yet():
    assert _validate("", set()) is None


def test_rejects_characters_outside_ascii_id_charset():
    assert _validate("아동1", set()) is not None
    assert _validate("s 1", set()) is not None


def test_accepts_ascii_id():
    assert _validate("s2026_11_01", set()) is None


def test_rejects_duplicate_session_id():
    assert _validate("s1", {"s1"}) is not None
