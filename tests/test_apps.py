"""두 Streamlit 앱을 화면 없이 실행해 보는 연기 테스트 — 오류 화면 없이 뜨고 영문 id가 안 보이는지."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from factories import make_session  # noqa: E402
from mssb_coder import storage  # noqa: E402
from mssb_coder.schema import CompletionStatus  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
REVIEW_APP = str(SRC / "review_app" / "app.py")
PUBLIC_APP = str(SRC / "public_app" / "app.py")


def _all_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "title", "header", "subheader", "info", "warning", "error", "success"):
        parts.extend(str(el.value) for el in getattr(at, kind))
    parts.extend(str(el.value) for el in at.text_area)
    return "\n".join(parts)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_DATA_DIR", str(tmp_path))
    return tmp_path


def test_review_app_shows_synthesis_without_errors(data_dir):
    make_session()

    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    assert not at.exception
    assert not at.error
    text = _all_text(at)
    assert "세션 종합 패턴 초안" in text
    assert "spilled_juice" not in text  # 수정 입력란 초기값도 한글로 치환됨


def test_review_app_drilldown_opens_stem_detail(data_dir):
    make_session()
    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    evidence_buttons = [b for b in at.button if ":evidence:" in (b.key or "")]
    assert len(evidence_buttons) == 3  # 같은 근거가 두 번 나와도 key 충돌 없음
    evidence_buttons[2].click().run()  # 「뽀삐가 없어졌어요」 근거

    assert not at.exception
    stem_select = next(s for s in at.selectbox if (s.key or "").endswith(":stem_select"))
    assert stem_select.value == "lost_dog"
    assert any("스템 상세" in str(s.value) for s in at.subheader)


def test_review_app_explains_broken_synthesis_file(data_dir):
    make_session()
    (storage.session_dir("s1") / "session_summary_ai_draft.json").write_text("{\"broken\": true}", encoding="utf-8")

    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    assert not at.exception
    assert any("읽지 못했습니다" in str(e.value) for e in at.error)


def test_review_app_without_data_dir_setting(monkeypatch):
    monkeypatch.delenv("MSSB_DATA_DIR", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    assert not at.exception
    assert any("MSSB_DATA_DIR" in str(e.value) for e in at.error)


def test_public_app_renders_published_session_in_korean():
    """results/adult_demo_01.json(예전에 게시된 파일, 패턴 서술에 영문 id가 섞여 있음)이 한글로만 보이는지."""
    at = AppTest.from_file(PUBLIC_APP, default_timeout=30)
    at.secrets["APP_PASSWORD"] = "pw"
    at.run()
    at.text_input[0].input("pw").run()

    assert not at.exception
    text = _all_text(at)
    assert "세션 종합 분석" in text
    for raw_id in ("big_rock", "candy_theft", "spilled_juice"):
        assert raw_id not in text, f"영문 id {raw_id}가 화면에 보임"


def test_public_app_rejects_wrong_password():
    at = AppTest.from_file(PUBLIC_APP, default_timeout=30)
    at.secrets["APP_PASSWORD"] = "pw"
    at.run()
    at.text_input[0].input("nope").run()

    assert not at.exception
    assert any("틀렸습니다" in str(e.value) for e in at.error)


def test_segmentation_page_confirms_segments(data_dir):
    make_session(confirm=False, coded=False, synthesized=False)

    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()
    assert not at.exception
    assert any("구간 확인" in str(t.value) for t in at.title)  # 종합 전 세션은 구간 확인 화면부터
    assert storage.segmentation_review_status("s1") == "unconfirmed"

    confirm = next(b for b in at.button if (b.key or "").endswith(":confirm_segments"))
    confirm.click().run()

    assert not at.exception
    assert storage.segmentation_review_status("s1") == "confirmed"
    assert any("확정했습니다" in str(s.value) for s in at.success)


def test_segmentation_page_asks_for_video_path_when_missing(data_dir):
    make_session(confirm=False, coded=False, synthesized=False)

    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    assert not at.exception
    assert any("원본 영상 경로" in str(t.label) for t in at.text_input)


def test_segmentation_page_blocks_invalid_examiner_segment_text(data_dir):
    make_session(confirm=False, coded=False, synthesized=False)
    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    examiner_input = next(t for t in at.text_input if ":examiner_text:" in (t.key or ""))
    examiner_input.input("10-5").run()

    assert not at.exception
    assert any("끝이 시작보다 뒤" in str(e.value) for e in at.error)
    confirm = next(b for b in at.button if (b.key or "").endswith(":confirm_segments"))
    assert confirm.disabled


def test_stem_edit_saves_reviewed_copy_used_by_report_then_reverts(data_dir):
    make_session()
    at = AppTest.from_file(REVIEW_APP, default_timeout=30).run()

    first_item = next(c for c in at.checkbox if ":present:empathy_warmth:" in (c.key or "") and c.value)
    first_item.uncheck().run()
    rationale = next(t for t in at.text_area if (t.key or "").endswith(":rationale:empathy_warmth"))
    rationale.input("아이는 인물을 돕지 않았다.").run()
    next(s for s in at.selectbox if (s.key or "").endswith(":completion")).set_value(CompletionStatus.PARTIAL).run()
    next(b for b in at.button if (b.key or "").endswith(":save")).click().run()

    assert not at.exception
    saved = storage.load_stem_coding("s1", "spilled_juice", reviewed=True)
    assert saved.completion_status == CompletionStatus.PARTIAL
    assert saved.codes["empathy_warmth"].present_count == 0
    assert saved.codes["empathy_warmth"].rationale == "아이는 인물을 돕지 않았다."
    assert storage.load_effective_stem_codings("s1")[0].codes["empathy_warmth"].present_count == 0
    assert any("검토본을 저장했습니다" in str(s.value) for s in at.success)

    next(b for b in at.button if (b.key or "").endswith(":revert")).click().run()

    assert not at.exception
    assert not storage.stem_reviewed_exists("s1", "spilled_juice")
    assert list((storage.session_dir("s1") / "coding").glob("spilled_juice_reviewed.*.bak.json"))
