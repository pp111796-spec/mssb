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
    assert "화제 전환" in report
    assert "spilled_juice" in report


def test_report_includes_single_story_signals():
    report = build_session_report_markdown(_synthesis())
    assert "유독 침착함" in report
    assert "정서적 통합" in report


def test_report_appendix_included_when_stem_results_given():
    report = build_session_report_markdown(_synthesis(), stem_results=[_stem_result()])
    assert "부록" in report
    assert "테스트 근거" in report


def test_report_omits_appendix_when_no_stem_results():
    report = build_session_report_markdown(_synthesis())
    assert "부록" not in report


def test_report_handles_no_recurring_patterns():
    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    report = build_session_report_markdown(synthesis)
    assert "발견된 반복 패턴 없음" in report
