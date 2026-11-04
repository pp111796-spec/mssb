"""두 Streamlit 앱을 화면 없이 실행해 보는 연기 테스트 — 오류 화면 없이 뜨고 영문 id가 안 보이는지."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from factories import make_session, synthesis  # noqa: E402
from mssb_coder import storage  # noqa: E402
from mssb_coder.claude_client import StructuredResult  # noqa: E402
from mssb_coder.schema import CompletionStatus, ResponseMode  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
REVIEW_APP = str(SRC / "review_app" / "app.py")
PUBLIC_APP = str(SRC / "public_app" / "app.py")


def _review_app():
    at = AppTest.from_file(REVIEW_APP, default_timeout=60)
    at.session_state["actor_name"] = "검사자가"
    return at


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

    at = _review_app().run()

    assert not at.exception
    assert not at.error
    text = _all_text(at)
    assert "세션 종합 패턴 초안" in text
    assert "spilled_juice" not in text  # 수정 입력란 초기값도 한글로 치환됨


def test_review_app_drilldown_opens_stem_detail(data_dir):
    make_session()
    at = _review_app().run()

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

    at = _review_app().run()

    assert not at.exception
    assert any("읽지 못했습니다" in str(e.value) for e in at.error)


def test_review_app_without_data_dir_setting(monkeypatch):
    monkeypatch.delenv("MSSB_DATA_DIR", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    at = _review_app().run()

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


def test_review_app_shows_new_session_form_when_no_sessions(data_dir):
    """세션이 하나도 없으면 예전처럼 CLI 안내문만 보여주는 게 아니라, 업로드 화면이 기본으로 열린다."""
    at = _review_app().run()

    assert not at.exception
    assert any("새 세션 만들기" in str(t.value) for t in at.title)
    assert any("세션 ID" in str(t.label) for t in at.text_input)
    new_session_button = next(b for b in at.button if "구간분리 시작" in (b.label or ""))
    assert new_session_button.disabled  # 영상도 세션 ID도 없으니 비활성


def test_review_app_can_switch_to_new_session_form_when_sessions_exist(data_dir):
    """기존 세션이 있을 때는 그 세션이 기본으로 열리되, 사이드바에서 '새 세션 만들기'로 바꿀 수 있다."""
    make_session()
    at = _review_app().run()
    assert any("세션 종합" in str(t.value) for t in at.title)  # 기본은 기존 세션(종합까지 끝남)

    session_select = next(s for s in at.sidebar.selectbox if s.label == "세션 선택")
    session_select.select("➕ 새 세션 만들기").run()

    assert not at.exception
    assert any("새 세션 만들기" in str(t.value) for t in at.title)


def test_segmentation_page_confirms_segments(data_dir):
    make_session(confirm=False, coded=False, synthesized=False)

    at = _review_app().run()
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

    at = _review_app().run()

    assert not at.exception
    assert any("원본 영상 경로" in str(t.label) for t in at.text_input)


def test_segmentation_page_blocks_invalid_examiner_segment_text(data_dir):
    make_session(confirm=False, coded=False, synthesized=False)
    at = _review_app().run()

    examiner_input = next(t for t in at.text_input if ":examiner_text:" in (t.key or ""))
    examiner_input.input("10-5").run()

    assert not at.exception
    assert any("끝이 시작보다 뒤" in str(e.value) for e in at.error)
    confirm = next(b for b in at.button if (b.key or "").endswith(":confirm_segments"))
    assert confirm.disabled


def _fake_synthesis_call(monkeypatch, text="다시 쓴 행동관찰이다."):
    calls = []

    def fake(stem_results, client=None):
        calls.append(stem_results)
        data = synthesis().model_dump(
            mode="json", include={"stems_included", "stems_administered_count", "recurring_patterns",
                                  "strengths", "support_areas", "recommendations"}
        )
        return StructuredResult(data=data | {"behavioral_observation": text}, model="claude-opus-5-5", usage={})

    monkeypatch.setattr("mssb_coder.pipeline.call_session_synthesis", fake)
    return calls


def _button(at, suffix):
    return next(b for b in at.button if (b.key or "").endswith(suffix))


def _fill_record(at):
    next(s for s in at.selectbox if (s.key or "").endswith(":completion")).set_value(CompletionStatus.PARTIAL)
    next(s for s in at.selectbox if (s.key or "").endswith(":mode")).set_value(ResponseMode.PROMPTED)
    next(n for n in at.number_input if (n.key or "").endswith(":prompts")).set_value(1)
    at.run()


def test_legacy_record_starts_unentered_and_blocks_confirmation(data_dir, monkeypatch):
    make_session()
    at = _review_app().run()

    assert next(s for s in at.selectbox if (s.key or "").endswith(":completion")).value is None
    assert _button(at, ":save").disabled
    assert not at.get("download_button") or not any("PDF" in str(el.label) for el in at.get("download_button"))


def test_ai_judged_record_is_prefilled_for_examiner_to_confirm(data_dir):
    make_session()
    stem = storage.load_stem_coding("s1", "spilled_juice").model_copy(
        update={
            "administration_source": "ai", "administration_note": "검사자가 한 번 되물었다.",
            "response_mode": ResponseMode.PROMPTED, "issue_prompt_count": 1,
        }
    )
    storage.save_stem_coding("s1", "spilled_juice", stem)

    at = _review_app().run()

    assert next(s for s in at.selectbox if (s.key or "").endswith(":mode")).value == ResponseMode.PROMPTED
    assert any("검사자가 한 번 되물었다" in str(m.value) for m in at.markdown)
    assert not _button(at, ":save").disabled


def test_editing_a_stem_rewrites_synthesis_immediately_then_revert_rewrites_again(data_dir, monkeypatch):
    make_session()
    calls = _fake_synthesis_call(monkeypatch)
    storage.save_session_synthesis("s1", storage.load_session_synthesis("s1"), reviewed=True)
    at = _review_app().run()

    _fill_record(at)
    rationale = next(t for t in at.text_area if (t.key or "").endswith(":rationale:empathy_warmth"))
    rationale.input("아이는 인물을 돕지 않았다.").run()
    _button(at, ":save").click().run()

    assert not at.exception
    saved = storage.load_stem_coding("s1", "spilled_juice", reviewed=True)
    assert saved.administration_source == "examiner" and saved.reviewed_by == "검사자가"
    assert saved.completion_status == CompletionStatus.PARTIAL
    assert len(calls) == 1
    assert calls[0][0].codes["empathy_warmth"].rationale == "아이는 인물을 돕지 않았다."  # 확정본이 종합 입력
    assert storage.load_session_synthesis("s1").behavioral_observation == "다시 쓴 행동관찰이다."
    assert storage.load_session_synthesis("s1").based_on == "reviewed_stem_results"
    assert not (storage.session_dir("s1") / "session_summary_reviewed.json").exists()  # 다시 검토해야 함
    assert any("다시 쓴 행동관찰이다." == t.value for t in at.text_area)  # 입력칸도 새 문장

    _button(at, ":revert").click().run()

    assert not at.exception
    assert not storage.stem_reviewed_exists("s1", "spilled_juice")
    assert list((storage.session_dir("s1") / "coding").glob("spilled_juice_reviewed.*.bak.json"))
    assert len(calls) == 2


def test_confirming_without_changes_does_not_call_claude(data_dir, monkeypatch):
    make_session()
    calls = _fake_synthesis_call(monkeypatch)
    stem = storage.load_stem_coding("s1", "spilled_juice").model_copy(update={"administration_source": "ai"})
    storage.save_stem_coding("s1", "spilled_juice", stem)
    at = _review_app().run()

    _button(at, ":save").click().run()

    assert not at.exception
    assert calls == []
    assert storage.load_stem_coding("s1", "spilled_juice", reviewed=True).reviewed_at


def test_failed_rewrite_keeps_pdf_blocked_and_offers_retry(data_dir, monkeypatch):
    make_session()

    def boom(stem_results, client=None):
        raise RuntimeError("API 오류")

    monkeypatch.setattr("mssb_coder.pipeline.call_session_synthesis", boom)
    at = _review_app().run()
    _fill_record(at)
    _button(at, ":save").click().run()

    assert not at.exception
    assert any("다시 쓰지 못했습니다" in str(e.value) for e in at.error)
    assert storage.load_session_synthesis("s1").stale
    assert _button(at, ":rewrite_synthesis")


def _make_ready_session():
    """PDF를 내려받을 수 있는 상태: 모든 이야기 확정, 종합 저장, 검사일·검사자 입력."""
    make_session()
    for name in ("spilled_juice", "lost_dog"):
        stem = storage.load_stem_coding("s1", name).model_copy(
            update={"administration_source": "examiner", "reviewed_at": "2026-11-04T00:00:00+00:00", "reviewed_by": "검사자가"}
        )
        storage.save_stem_coding("s1", name, stem, reviewed=True)
    synth = storage.load_session_synthesis("s1").model_copy(update={"test_date": "2026-11-01", "examiner_name": "검사자가"})
    storage.save_session_synthesis("s1", synth, reviewed=True)


def test_pdf_appears_only_when_everything_is_confirmed(data_dir):
    _make_ready_session()

    at = _review_app().run()

    assert not at.exception
    assert any("PDF" in str(el.label) for el in at.get("download_button"))
    assert any("삭제" in str(w.value) for w in at.warning)


def test_pdf_callback_deletes_media_and_logs(data_dir):
    from mssb_coder.report import report_blockers  # noqa: F401
    from review_app.synthesis_page import _after_pdf_download

    _make_ready_session()
    root = storage.session_dir("s1")
    for rel in ("upload/original.mp4.enc", "frames/spilled_juice/frame_000001.png", "preview/clip.mp4", "audio_segments/a.wav", "audio.wav"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")

    _after_pdf_download("s1")

    for rel in ("upload", "frames", "preview", "audio_segments", "audio.wav"):
        assert not (root / rel).exists()
    assert (root / "session_summary_reviewed.json").exists()
    assert (root / "coding" / "spilled_juice_reviewed.json").exists()
    assert storage.load_session_metadata("s1").media_deleted_at
    events = [r["event"] for r in storage.read_access_log("s1")]
    assert "pdf_download" in events and "media_deleted" in events


def test_review_app_requires_user_name_and_logs_open(data_dir):
    make_session()
    at = AppTest.from_file(REVIEW_APP, default_timeout=60).run()

    assert not at.exception
    assert any("사용자 이름" in str(i.value) for i in at.info)
    assert storage.read_access_log("s1") == []

    at.sidebar.text_input[0].input("검사자가").run()
    at.run()  # 다시 그려도 기록은 한 번만

    opens = [r for r in storage.read_access_log("s1") if r["event"] == "session_open"]
    assert len(opens) == 1 and opens[0]["actor"] == "검사자가" and opens[0]["os_user"]


def test_private_viewer_closed_when_password_not_configured():
    at = AppTest.from_file(PUBLIC_APP, default_timeout=30)
    at.secrets["APP_PASSWORD"] = ""  # PC의 .streamlit/secrets.toml 값과 상관없이 비어 있는 상태로 시험한다
    at.run()

    assert any("APP_PASSWORD" in str(e.value) for e in at.error)
    assert not at.text_input


def test_private_viewer_hides_pdf_for_unconfirmed_published_session():
    at = AppTest.from_file(PUBLIC_APP, default_timeout=60)
    at.secrets["APP_PASSWORD"] = "pw"
    at.run()
    at.text_input[0].input("pw").run()

    assert not at.exception
    assert not any("PDF" in str(el.label) for el in at.get("download_button"))
    assert any("확정 전" in str(w.value) for w in at.warning)
