"""여러 테스트가 함께 쓰는 가짜 세션 데이터."""

from mssb_coder import storage
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.schema import (
    CategoryCode,
    CompletionStatus,
    Evidence,
    EvidenceSource,
    ItemCode,
    RecurringPattern,
    ResponseMode,
    SessionMetadata,
    SessionSynthesis,
    StemCodingResult,
    StemTimestamp,
    StemWindow,
    SubjectType,
    SynthesisEvidence,
)


def stem_timestamp(name, start, handoff, end, scored=True) -> StemTimestamp:
    return StemTimestamp(
        stem_name=name, start_s=start, end_s=end, handoff_timestamp_s=handoff,
        confidence=0.9, matched_cue="완성요청 문구", scored=scored,
    )


DEFAULT_STEMS = [
    stem_timestamp("warm_up", 0.0, 10.0, 20.0, scored=False),
    stem_timestamp("spilled_juice", 20.0, 30.0, 60.0),
    stem_timestamp("lost_dog", 60.0, 70.0, 100.0),
]


def stem_result(stem_name="spilled_juice", session_id="s1", window: StemTimestamp | None = None, rationale="관련된 행동은 나타나지 않았다.") -> StemCodingResult:
    codes = {}
    for i, category in enumerate(load_coding_system().categories):
        items = {item.id: ItemCode(present=False, evidence=[]) for item in category.items}
        if i == 0:
            first = category.items[0].id
            items[first] = ItemCode(
                present=True,
                evidence=[Evidence(timestamp_s=35.0, source=EvidenceSource.TRANSCRIPT, note="주스를 닦았다")],
            )
        codes[category.id] = CategoryCode(
            items=items, present_count=1 if i == 0 else 0, rationale=rationale
        )
    return StemCodingResult(
        session_id=session_id,
        child_age_months=54,
        stem_name=stem_name,
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=codes,
        story_specific_notes_applied="—",
        model_version="claude-opus-5-5",
        source_window=StemWindow.of(window) if window else None,
        timestamp_basis="session_absolute" if window else None,
    )


def synthesis(session_id="s1") -> SessionSynthesis:
    return SessionSynthesis(
        session_id=session_id,
        stems_included=["spilled_juice", "lost_dog"],
        stems_administered_count=2,
        recurring_patterns=[
            RecurringPattern(
                category="empathy_warmth",
                observed_in_stems=["spilled_juice", "lost_dog"],
                stems_administered_count_for_pattern=2,
                pattern_description="아이는 spilled_juice 이야기에서 인물을 도왔다.",
                evidence=[
                    SynthesisEvidence(stem_name="spilled_juice", timestamp_s=35.0, note="도움"),
                    SynthesisEvidence(stem_name="spilled_juice", timestamp_s=35.0, note="같은 근거 반복"),
                ],
            )
        ],
        behavioral_observation="아이는 검사에 협조했다.",
        strengths="인물의 마음을 헤아렸다.",
        support_areas="갈등 장면에서 말을 줄였다.",
        recommendations=["아이의 이야기를 끝까지 들어 준다."],
        model_version="claude-opus-5-5",
    )


def make_session(session_id="s1", stems=DEFAULT_STEMS, confirm=True, coded=True, synthesized=True) -> None:
    """MSSB_DATA_DIR(테스트용 임시 폴더) 아래에 세션 하나를 만든다."""
    storage.save_session_metadata(
        SessionMetadata(session_id=session_id, child_age_months=54, child_gender="female", subject_type=SubjectType.ADULT)
    )
    storage.save_stem_timestamps(session_id, stems)
    if confirm:
        storage.confirm_segmentation(session_id)
    if coded:
        for stem in stems:
            if stem.scored:
                storage.save_stem_coding(session_id, stem.stem_name, stem_result(stem.stem_name, session_id, stem))
    if synthesized:
        storage.save_session_synthesis(session_id, synthesis(session_id))
