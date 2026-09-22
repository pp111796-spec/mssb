import pytest

from mssb_coder.coding_systems import load_coding_system
from mssb_coder.schema import (
    CategoryCode,
    CompletionStatus,
    ItemCode,
    ResponseMode,
    SessionSynthesis,
    StemCodingResult,
    StemTimestamp,
)
from mssb_coder import storage


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_DATA_DIR", str(tmp_path))


def _sample_stem_result(session_id="s1", stem_name="spilled_juice") -> StemCodingResult:
    coding_system = load_coding_system()
    codes = {
        category.id: CategoryCode(
            items={item.id: ItemCode(present=False, evidence=[]) for item in category.items},
            present_count=0,
            rationale="테스트",
        )
        for category in coding_system.categories
    }
    return StemCodingResult(
        session_id=session_id,
        child_age_months=54,
        stem_name=stem_name,
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=codes,
        story_specific_notes_applied="—",
        model_version="claude-opus-5",
    )


def test_get_data_dir_requires_env_var(monkeypatch):
    monkeypatch.delenv("MSSB_DATA_DIR", raising=False)
    with pytest.raises(storage.DataDirNotConfiguredError):
        storage.get_data_dir()


def test_stem_coding_round_trip():
    result = _sample_stem_result()
    storage.save_stem_coding("s1", "spilled_juice", result)
    loaded = storage.load_stem_coding("s1", "spilled_juice")
    assert loaded.stem_name == "spilled_juice"
    assert loaded.session_id == "s1"


def test_stem_coding_reviewed_is_separate_file():
    ai_draft = _sample_stem_result()
    storage.save_stem_coding("s1", "spilled_juice", ai_draft, reviewed=False)

    reviewed = ai_draft.model_copy(update={"story_specific_notes_applied": "사람이 수정함"})
    storage.save_stem_coding("s1", "spilled_juice", reviewed, reviewed=True)

    assert storage.load_stem_coding("s1", "spilled_juice", reviewed=False).story_specific_notes_applied == "—"
    assert storage.load_stem_coding("s1", "spilled_juice", reviewed=True).story_specific_notes_applied == "사람이 수정함"


def test_load_all_stem_codings_uses_ai_draft_by_default():
    storage.save_stem_coding("s1", "spilled_juice", _sample_stem_result(stem_name="spilled_juice"))
    storage.save_stem_coding("s1", "lost_dog", _sample_stem_result(stem_name="lost_dog"))
    results = storage.load_all_stem_codings("s1")
    assert {r.stem_name for r in results} == {"spilled_juice", "lost_dog"}


def test_stem_timestamps_round_trip():
    timestamps = [
        StemTimestamp(
            stem_name="spilled_juice",
            start_s=0.0,
            end_s=30.0,
            handoff_timestamp_s=15.0,
            confidence=0.95,
            matched_cue="완성요청 문구",
        )
    ]
    storage.save_stem_timestamps("s1", timestamps)
    loaded = storage.load_stem_timestamps("s1")
    assert loaded[0].stem_name == "spilled_juice"


def test_session_synthesis_round_trip():
    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    storage.save_session_synthesis("s1", synthesis)
    loaded = storage.load_session_synthesis("s1")
    assert loaded.based_on == "ai_draft_stem_results"


def test_build_session_export_bundle_uses_ai_draft_when_no_reviewed():
    from mssb_coder.bundle import parse_bundle_dict

    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    storage.save_session_synthesis("s1", synthesis, reviewed=False)
    storage.save_stem_coding("s1", "spilled_juice", _sample_stem_result())

    bundle = storage.build_session_export_bundle("s1")
    parsed_synthesis, parsed_stems = parse_bundle_dict(bundle)
    assert parsed_synthesis.session_id == "s1"
    assert len(parsed_stems) == 1


def test_build_session_export_bundle_prefers_reviewed():
    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    storage.save_session_synthesis("s1", synthesis, reviewed=False)
    reviewed = synthesis.model_copy(update={"stems_administered_count": 99})
    storage.save_session_synthesis("s1", reviewed, reviewed=True)

    bundle = storage.build_session_export_bundle("s1")
    assert bundle["synthesis"]["stems_administered_count"] == 99
