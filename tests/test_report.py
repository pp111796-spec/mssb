from mssb_coder.coding_systems import load_coding_system
from mssb_coder.report import build_session_report_markdown
from mssb_coder.schema import (
    CategoryCode,
    CompletionStatus,
    ItemCode,
    RecurringPattern,
    ResponseMode,
    SessionSynthesis,
    SingleStoryNotableSignal,
    StemCodingResult,
    SynthesisEvidence,
)


def _synthesis() -> SessionSynthesis:
    return SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice", "lost_dog"],
        stems_administered_count=2,
        recurring_patterns=[
            RecurringPattern(
                category="avoidant_withdrawal",
                observed_in_stems=["spilled_juice", "lost_dog"],
                stems_administered_count_for_pattern=2,
                pattern_description="갈등을 회피하는 반응이 반복됨",
                evidence=[SynthesisEvidence(stem_name="spilled_juice", timestamp_s=12.3, note="화제 전환")],
            )
        ],
        single_story_notable_signals=[
            SingleStoryNotableSignal(stem_name="lost_dog", category="emotional_integration", note="유독 침착함")
        ],
        model_version="claude-opus-5",
    )


def _stem_result() -> StemCodingResult:
    coding_system = load_coding_system()
    codes = {}
    for category in coding_system.categories:
        items = {item.id: ItemCode(present=False, evidence=[]) for item in category.items}
        codes[category.id] = CategoryCode(items=items, present_count=0, rationale="테스트 근거")
    return StemCodingResult(
        session_id="s1",
        child_age_months=54,
        stem_name="spilled_juice",
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=codes,
        story_specific_notes_applied="—",
        model_version="claude-opus-5",
    )


def test_report_includes_disclaimer_and_patterns():
    report = build_session_report_markdown(_synthesis())
    assert "진단이 아니라" in report
    assert "회피/위축" in report  # category id가 한글 라벨로 치환됐는지
    assert "엎지러진 주스" in report  # 정상 실시된 스템 안내문의 stem id가 한글 라벨로 치환됐는지
    assert "spilled_juice" not in report
    assert "12.3초" not in report  # 종합 해석에는 근거 인용(타임스탬프)이 더 이상 없음


def test_report_includes_single_story_signals():
    report = build_session_report_markdown(_synthesis())
    assert "유독 침착함" in report
    assert "정서적 통합" in report


def test_report_includes_stem_interpretation_when_stem_results_given():
    report = build_session_report_markdown(_synthesis(), stem_results=[_stem_result()])
    assert "이야기별 해석" in report
    assert "엎지러진 주스" in report  # stem id(spilled_juice)가 한글 라벨로 치환됐는지
    assert "테스트 근거" in report
    assert "존재 " not in report  # 체크리스트("존재 N개") 형식이 빠졌는지
    assert "spilled_juice" not in report


def test_report_omits_stem_interpretation_when_no_stem_results():
    report = build_session_report_markdown(_synthesis())
    assert "이야기별 해석" not in report


def test_report_scrubs_english_ids_embedded_in_claude_free_text():
    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice", "lost_dog"],
        stems_administered_count=2,
        recurring_patterns=[
            RecurringPattern(
                category="avoidant_withdrawal",
                observed_in_stems=["spilled_juice", "lost_dog"],
                stems_administered_count_for_pattern=2,
                pattern_description="spilled_juice와 lost_dog 모두에서 회피 반응이 나타났다.",
                evidence=[],
            )
        ],
        model_version="claude-opus-5",
    )
    report = build_session_report_markdown(synthesis)
    assert "spilled_juice" not in report
    assert "lost_dog" not in report
    assert "엎지러진 주스" in report
    assert "뽀삐가 없어졌어요" in report


def test_report_handles_no_recurring_patterns():
    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    report = build_session_report_markdown(synthesis)
    assert "반복해서 나타난 패턴은 없었다" in report


def test_report_includes_new_sections_when_present():
    synthesis = _synthesis().model_copy(
        update={
            "behavioral_observation": "검사 내내 협조적이었다.",
            "strengths": "감정을 말로 표현하는 모습이 돋보였다.",
            "support_areas": "갈등 장면에서 화제를 돌리는 경향이 보였다.",
            "recommendations": ["가정에서 감정을 말로 표현해 보는 놀이를 함께 해 본다."],
        }
    )
    report = build_session_report_markdown(synthesis, stem_results=[_stem_result()])
    for heading in ["## 인적 사항", "## 검사 개요", "## 의뢰 사유 및 배경", "## 검사 중 행동관찰",
                    "## 강점과 지원이 필요한 영역", "## 제언", "## 해석상 유의점"]:
        assert heading in report
    assert "54개월" in report
    assert "(전체 종합 보고서에서 가져올 예정)" in report
    assert "감정을 말로 표현해 보는 놀이" in report


def test_report_omits_optional_sections_for_old_synthesis_without_new_fields():
    report = build_session_report_markdown(_synthesis())
    assert "## 검사 중 행동관찰" not in report
    assert "## 제언" not in report
    assert "## 강점과 지원이 필요한 영역" not in report
    assert "## 해석상 유의점" in report


def test_caveat_says_draft_until_examiner_saves_review():
    from factories import synthesis
    from mssb_coder.report import REVIEW_STATUS_DRAFT, REVIEW_STATUS_REVIEWED

    draft = build_session_report_markdown(synthesis())
    reviewed = build_session_report_markdown(synthesis().model_copy(update={"reviewed_at": "2026-10-26T00:00:00+00:00"}))

    assert REVIEW_STATUS_DRAFT in draft and REVIEW_STATUS_REVIEWED not in draft
    assert REVIEW_STATUS_REVIEWED in reviewed and REVIEW_STATUS_DRAFT not in reviewed


def test_report_header_uses_korean_date_and_no_scorer_jargon():
    from factories import stem_result, synthesis

    report = build_session_report_markdown(
        synthesis().model_copy(update={"generated_at": "2026-10-14T20:21:30.610292+00:00"}),
        stem_results=[stem_result("spilled_juice")],
    )

    assert "작성일: 2026년 10월 15일" in report  # 한국 시간으로 바꾸면 다음 날
    assert "T20:21" not in report
    assert "스템" not in report
    assert report.rstrip().endswith(".")
