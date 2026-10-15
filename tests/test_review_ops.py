import pytest

from factories import DEFAULT_STEMS, make_session, stem_result, stem_timestamp
from mssb_coder import storage
from mssb_coder.review_ops import session_progress


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_DATA_DIR", str(tmp_path))


def test_progress_before_segmentation():
    storage.session_dir("s1")
    assert session_progress("s1").label == "구간분리 전"


def test_progress_needs_confirmation_then_tracks_changes():
    make_session(confirm=False, coded=False, synthesized=False)
    assert session_progress("s1").label == "구간 확인 필요"

    storage.confirm_segmentation("s1")
    assert session_progress("s1").label == "채점 진행 중 (0/2)"

    storage.save_stem_timestamps("s1", [DEFAULT_STEMS[0], stem_timestamp("spilled_juice", 20.0, 31.0, 60.0), DEFAULT_STEMS[2]])
    assert session_progress("s1").label == "구간이 바뀜 — 다시 확정 필요"


def test_progress_through_synthesis_and_review():
    make_session(synthesized=False)
    assert session_progress("s1").label == "종합 전"

    make_session()
    assert session_progress("s1").label == "종합 완료 — 검토 전"

    storage.save_session_synthesis("s1", storage.load_session_synthesis("s1"), reviewed=True)
    assert session_progress("s1").label == "검토 저장됨"


def test_effective_codings_prefer_reviewed_and_skip_stale_stems():
    make_session()
    reviewed = stem_result("lost_dog", rationale="검토자가 고친 문장이다.")
    storage.save_stem_coding("s1", "lost_dog", reviewed, reviewed=True)
    storage.save_stem_coding("s1", "candy_theft", stem_result("candy_theft"))  # 현재 구간에 없는 예전 파일

    results = storage.load_effective_stem_codings("s1")

    assert [r.stem_name for r in results] == ["spilled_juice", "lost_dog"]
    assert results[1].codes["empathy_warmth"].rationale == "검토자가 고친 문장이다."
