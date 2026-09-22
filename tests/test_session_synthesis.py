import pytest

from mssb_coder.coding_systems import load_coding_system
from mssb_coder.schema import CategoryCode, CompletionStatus, ItemCode, ResponseMode, StemCodingResult
from mssb_coder.session_synthesis import build_synthesis_user_message


def _stem_result(stem_name: str) -> StemCodingResult:
    coding_system = load_coding_system()
    codes = {
        category.id: CategoryCode(
            items={item.id: ItemCode(present=False, evidence=[]) for item in category.items},
            present_count=0,
            rationale=f"{stem_name} 테스트",
        )
        for category in coding_system.categories
    }
    return StemCodingResult(
        session_id="s1",
        child_age_months=54,
        stem_name=stem_name,
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=codes,
        story_specific_notes_applied="—",
        model_version="claude-opus-5",
    )


def test_build_synthesis_message_includes_all_stems():
    message = build_synthesis_user_message([_stem_result("spilled_juice"), _stem_result("lost_dog")])
    assert "spilled_juice" in message
    assert "lost_dog" in message
    assert "present_count" in message


def test_build_synthesis_message_rejects_empty_list():
    with pytest.raises(ValueError):
        build_synthesis_user_message([])
