import pytest
from pydantic import ValidationError

from mssb_coder.coding_systems import load_coding_system
from mssb_coder.schema import (
    CategoryCode,
    CompletionStatus,
    Evidence,
    EvidenceSource,
    ExaminerSegment,
    ItemCode,
    ResponseMode,
    StemCodingResult,
    StemTimestamp,
)


def _minimal_codes() -> dict:
    """실제 mssb_5category.yaml 기준으로, 모든 항목 present=false인 최소 codes 딕셔너리."""
    coding_system = load_coding_system()
    codes = {}
    for category in coding_system.categories:
        items = {item.id: ItemCode(present=False, evidence=[]) for item in category.items}
        codes[category.id] = CategoryCode(items=items, present_count=0, rationale="테스트용 최소 예시")
    return codes


def test_stem_coding_result_round_trip():
    result = StemCodingResult(
        session_id="test_session_001",
        child_age_months=54,
        stem_name="spilled_juice",
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=_minimal_codes(),
        story_specific_notes_applied="—",
        model_version="claude-opus-5",
    )
    dumped = result.model_dump_json()
    restored = StemCodingResult.model_validate_json(dumped)
    assert restored.stem_name == "spilled_juice"
    assert restored.interpretation_layer == "1_single_story_observation"
    assert "진단이 아니라" in restored.disclaimer


def test_present_count_mismatch_rejected():
    with pytest.raises(ValidationError):
        CategoryCode(
            items={
                "sharing": ItemCode(present=True, evidence=[]),
                "empathy_help": ItemCode(present=False, evidence=[]),
            },
            present_count=0,  # 실제로 1개 present인데 0이라 주장 — 거부되어야 함
            rationale="일부러 틀린 값",
        )


def test_evidence_source_enum():
    ev = Evidence(timestamp_s=42.3, source=EvidenceSource.FACIAL_AU, note="눈썹 내림")
    assert ev.source == "facial_au"


def test_stem_timestamp_handoff_must_be_inside_range():
    with pytest.raises(ValidationError):
        StemTimestamp(
            stem_name="spilled_juice",
            start_s=10.0,
            end_s=40.0,
            handoff_timestamp_s=5.0,  # start_s보다 앞선 시각 — 거부되어야 함
            confidence=0.9,
            matched_cue="완성요청 문구",
        )


def test_stem_timestamp_additional_examiner_segment_inside_range():
    ts = StemTimestamp(
        stem_name="lost_dog",
        start_s=0.0,
        end_s=100.0,
        handoff_timestamp_s=20.0,
        additional_examiner_segments_s=[ExaminerSegment(start_s=50.0, end_s=55.0)],
        confidence=0.7,
        matched_cue="뽀삐가 다시 돌아왔어",
    )
    assert ts.additional_examiner_segments_s[0].start_s == 50.0
