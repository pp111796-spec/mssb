import json
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


def test_stem_cv_timeline_round_trip():
    from mssb_coder.schema import CvDollSample, CvFacialSample, StemCvTimeline

    timeline = StemCvTimeline(
        session_id="s1",
        stem_name="spilled_juice",
        fps=2.0,
        facial=[CvFacialSample(t=0.0, aus={"AU06": 0.8}, head_pitch_deg=5.0, head_yaw_deg=2.0, landmark_confidence=0.9)],
        doll=[CvDollSample(t=0.0, doll_id="dog_doll", bbox=None)],
    )
    assert not storage.cv_timeline_exists("s1", "spilled_juice")
    storage.save_stem_cv_timeline("s1", "spilled_juice", timeline)
    assert storage.cv_timeline_exists("s1", "spilled_juice")

    loaded = storage.load_stem_cv_timeline("s1", "spilled_juice")
    assert loaded.facial[0].aus == {"AU06": 0.8}
    assert loaded.doll[0].bbox is None


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


def test_publish_session_bundle_to_repo_writes_json_file(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "REPO_RESULTS_DIR", tmp_path / "results")

    synthesis = SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )
    storage.save_session_synthesis("s1", synthesis)
    storage.save_stem_coding("s1", "spilled_juice", _sample_stem_result())

    path = storage.publish_session_bundle_to_repo("s1")
    assert path == tmp_path / "results" / "s1.json"
    assert path.exists()

    from mssb_coder.bundle import parse_bundle_dict
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    parsed_synthesis, parsed_stems = parse_bundle_dict(raw)
    assert parsed_synthesis.session_id == "s1"
    assert len(parsed_stems) == 1


def test_append_api_usage_adds_one_json_line_per_call():
    storage.append_api_usage("s1", {"call": "synthesis", "model": "claude-opus-5-5", "input_tokens": 10})
    storage.append_api_usage("s1", {"call": "stem_coding:spilled_juice", "input_tokens": 20})

    lines = (storage.session_dir("s1") / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["call"] for r in records] == ["synthesis", "stem_coding:spilled_juice"]
    assert all("at" in r for r in records)


def test_save_failed_response_keeps_raw_payload():
    path = storage.save_failed_response("s1", "synthesis", {"recurring_patterns": "잘못된 형식"})

    assert path.parent.name == "failed"
    assert json.loads(path.read_text(encoding="utf-8")) == {"recurring_patterns": "잘못된 형식"}


def _timestamps():
    return [
        StemTimestamp(
            stem_name="spilled_juice", start_s=0.0, end_s=30.0, handoff_timestamp_s=10.0,
            confidence=0.9, matched_cue="문구",
        )
    ]


def test_segmentation_review_status_tracks_confirmation_and_later_edits():
    storage.save_stem_timestamps("s1", _timestamps())
    assert storage.segmentation_review_status("s1") == "unconfirmed"

    storage.confirm_segmentation("s1")
    assert storage.segmentation_review_status("s1") == "confirmed"

    edited = [_timestamps()[0].model_copy(update={"end_s": 31.0})]
    storage.save_stem_timestamps("s1", edited)
    assert storage.segmentation_review_status("s1") == "changed"


def test_ai_draft_timestamps_are_kept_separately():
    storage.save_stem_timestamps("s1", _timestamps(), ai_draft=True)
    storage.save_stem_timestamps("s1", [_timestamps()[0].model_copy(update={"end_s": 31.0})])

    assert storage.load_stem_timestamps("s1", ai_draft=True)[0].end_s == 30.0
    assert storage.load_stem_timestamps("s1")[0].end_s == 31.0


def test_session_metadata_and_transcript_round_trip():
    from mssb_coder.schema import SessionMetadata, SubjectType
    from mssb_coder.transcribe import TranscriptSegment

    meta = SessionMetadata(
        session_id="s1", child_age_months=60, child_gender="male", subject_type=SubjectType.CHILD,
        video_path="C:/mssb_data/raw_videos/a.mp4",
    )
    storage.save_session_metadata(meta)
    assert storage.load_session_metadata("s1") == meta

    assert storage.load_transcript("s1") is None
    segments = [TranscriptSegment(start_s=1.0, end_s=2.0, text="안녕")]
    storage.save_transcript("s1", segments)
    assert storage.load_transcript("s1") == segments


def test_reviewed_synthesis_records_review_time_and_ai_draft_never_does():
    from factories import synthesis

    storage.save_session_synthesis("s1", synthesis().model_copy(update={"reviewed_at": "x"}))
    storage.save_session_synthesis("s1", synthesis(), reviewed=True)

    assert storage.load_session_synthesis("s1").reviewed_at is None
    assert storage.load_session_synthesis("s1", reviewed=True).reviewed_at


def test_old_reviewed_synthesis_without_review_time_counts_as_reviewed():
    from factories import synthesis

    path = storage.session_dir("s1") / "session_summary_reviewed.json"
    path.write_text(synthesis().model_dump_json(), encoding="utf-8")  # reviewed_at 필드가 생기기 전 파일

    assert storage.load_session_synthesis("s1", reviewed=True).reviewed_at
