"""파이프라인 흐름 테스트 — CV·전사·Claude 호출은 가짜로 바꿔 실제 설치 없이 돈다."""

import pytest

from mssb_coder import pipeline, storage
from mssb_coder.claude_client import StructuredResult
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.cv_timeline import StemCvFrames, to_timeline
from mssb_coder.schema import SessionMetadata, StemTimestamp, StemWindow, SubjectType
from mssb_coder.segmentation import load_story_stems
from mssb_coder.transcribe import TranscriptSegment

SESSION = "s1"


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_DATA_DIR", str(tmp_path))


def _stem(name, start, handoff, end, scored=True):
    return StemTimestamp(
        stem_name=name, start_s=start, end_s=end, handoff_timestamp_s=handoff,
        confidence=0.9, matched_cue="완성요청 문구", scored=scored,
    )


STEMS = [
    _stem("warm_up", 0.0, 10.0, 20.0, scored=False),
    _stem("spilled_juice", 20.0, 30.0, 60.0),
    _stem("lost_dog", 60.0, 70.0, 100.0),
]


def _metadata():
    return SessionMetadata(session_id=SESSION, child_age_months=54, child_gender="female", subject_type=SubjectType.ADULT)


def _setup_session(stems=STEMS, confirm=True):
    storage.save_session_metadata(_metadata())
    storage.save_stem_timestamps(SESSION, stems)
    storage.save_transcript(SESSION, [TranscriptSegment(start_s=35.0, end_s=37.0, text="주스를 닦았어요.")])
    if confirm:
        storage.confirm_segmentation(SESSION)


def _valid_coding_data():
    codes = {
        category.id: {
            "items": {item.id: {"present": False, "evidence": []} for item in category.items},
            "present_count": 0,
            "rationale": "관련된 행동은 나타나지 않았다.",
        }
        for category in load_coding_system().categories
    }
    return {"codes": codes, "story_specific_notes_applied": "—"}


class FakeCoder:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []

    def __call__(self, coding_system, **kwargs):
        self.calls.append(kwargs)
        if any(name in kwargs["story_specific_notes"] for name in self.fail_on):
            raise RuntimeError("가짜 실패")
        return StructuredResult(data=_valid_coding_data(), model="claude-opus-5-5", usage={"input_tokens": 1})


@pytest.fixture
def fake_cv(monkeypatch):
    calls = []

    def compute(video_path, metadata, stem, cv_config):
        calls.append(stem.stem_name)
        return StemCvFrames()

    monkeypatch.setattr(pipeline, "compute_stem_cv", compute)
    return calls


@pytest.fixture
def fake_coder(monkeypatch):
    coder = FakeCoder()
    monkeypatch.setattr(pipeline, "call_stem_coding", coder)
    # 실패 스템을 고를 수 있도록 story_specific_notes에 스템 id를 넣는다
    stems = load_story_stems()
    patched = type(stems)(
        completion_request_phrase=stems.completion_request_phrase,
        stems=tuple(type(s)(s.id, s.display_name_ko, s.scored, s.keywords, s.id) for s in stems.stems),
    )
    monkeypatch.setattr(pipeline, "load_story_stems", lambda: patched)
    return coder


def _no_transcribe(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("저장된 전사가 있으면 다시 전사하면 안 됨")

    monkeypatch.setattr(pipeline, "transcribe_session", fail)


# --- 구간분리 응답 정규화 ---------------------------------------------------------


def test_normalize_segments_rejects_unknown_stem_names():
    data = {"segments": [{**STEMS[1].model_dump(), "stem_name": "made_up_story"}]}
    with pytest.raises(ValueError, match="made_up_story"):
        pipeline.normalize_segments(data, load_story_stems())


def test_normalize_segments_allows_recording_margins_and_uses_yaml_scored():
    data = {
        "segments": [
            {**_stem("pre_session", 0.0, 5.0, 5.0, scored=False).model_dump(), "scored": True},
            {**STEMS[0].model_dump(), "scored": True},  # Claude가 warm_up을 채점 대상으로 잘못 표시
            {**STEMS[1].model_dump(), "scored": False},
        ]
    }

    timestamps = pipeline.normalize_segments(data, load_story_stems())

    assert [(t.stem_name, t.scored) for t in timestamps] == [
        ("pre_session", False), ("warm_up", False), ("spilled_juice", True),
    ]


# --- 구간 확정 확인 ---------------------------------------------------------------


def test_coding_refuses_unconfirmed_segments(fake_cv, fake_coder):
    _setup_session(confirm=False)
    with pytest.raises(pipeline.SegmentationNotConfirmedError, match="확정되지"):
        pipeline.run_coding_stage(None, _metadata())
    assert fake_coder.calls == []


def test_coding_refuses_segments_changed_after_confirmation(fake_cv, fake_coder):
    _setup_session()
    storage.save_stem_timestamps(SESSION, [STEMS[0], _stem("spilled_juice", 20.0, 31.0, 60.0), STEMS[2]])
    with pytest.raises(pipeline.SegmentationNotConfirmedError, match="바뀌었습니다"):
        pipeline.run_coding_stage(None, _metadata())


def test_cv_stage_also_requires_confirmation(fake_cv):
    _setup_session(confirm=False)
    with pytest.raises(pipeline.SegmentationNotConfirmedError):
        pipeline.run_cv_stage(None, _metadata())


# --- 채점: 실패 격리 · 이어하기 · 전사 재사용 -------------------------------------


def test_coding_records_window_and_session_time_and_reuses_transcript(fake_cv, fake_coder, monkeypatch):
    _setup_session()
    _no_transcribe(monkeypatch)

    results = pipeline.run_coding_stage(None, _metadata())

    assert [r.stem_name for r in results] == ["spilled_juice", "lost_dog"]
    saved = storage.load_stem_coding(SESSION, "spilled_juice")
    assert saved.source_window == StemWindow.of(STEMS[1])
    assert saved.timestamp_basis == "session_absolute"
    assert saved.model_version == "claude-opus-5-5"
    assert "[35.0s] 주스를 닦았어요." in fake_coder.calls[0]["transcript_text"]
    assert storage.load_stem_verbatim(SESSION, "spilled_juice").lines  # 채점 성공 뒤 축어록 저장
    usage_lines = (storage.session_dir(SESSION) / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(usage_lines) == 2


def test_one_failing_stem_does_not_stop_the_others(fake_cv, fake_coder):
    _setup_session()
    fake_coder.fail_on = {"spilled_juice"}

    with pytest.raises(pipeline.StageFailedError) as exc_info:
        pipeline.run_coding_stage(None, _metadata())

    assert set(exc_info.value.failures) == {"spilled_juice"}
    assert storage.load_stem_coding(SESSION, "lost_dog").stem_name == "lost_dog"
    with pytest.raises(FileNotFoundError):
        storage.load_stem_verbatim(SESSION, "spilled_juice")


def test_rerun_skips_stems_already_coded_with_same_window(fake_cv, fake_coder):
    _setup_session()
    fake_coder.fail_on = {"spilled_juice"}
    with pytest.raises(pipeline.StageFailedError):
        pipeline.run_coding_stage(None, _metadata())
    fake_coder.fail_on = set()
    fake_coder.calls.clear()

    pipeline.run_coding_stage(None, _metadata())

    assert [c["story_specific_notes"] for c in fake_coder.calls] == ["spilled_juice"]


def test_force_recodes_everything(fake_cv, fake_coder):
    _setup_session()
    pipeline.run_coding_stage(None, _metadata())
    fake_coder.calls.clear()

    pipeline.run_coding_stage(None, _metadata(), force=True)

    assert len(fake_coder.calls) == 2


def test_changed_window_triggers_recoding_of_that_stem_only(fake_cv, fake_coder):
    _setup_session()
    pipeline.run_coding_stage(None, _metadata())
    fake_coder.calls.clear()

    storage.save_stem_timestamps(SESSION, [STEMS[0], _stem("spilled_juice", 20.0, 32.0, 60.0), STEMS[2]])
    storage.confirm_segmentation(SESSION)
    pipeline.run_coding_stage(None, _metadata())

    assert [c["story_specific_notes"] for c in fake_coder.calls] == ["spilled_juice"]


# --- CV 수치 캐시 무효화 ------------------------------------------------------------


def test_cv_timeline_is_recomputed_when_window_changes(fake_cv):
    stem = STEMS[1]
    old_stem = _stem("spilled_juice", 20.0, 30.0, 50.0)
    timeline = to_timeline(SESSION, stem.stem_name, 2.0, StemCvFrames())
    storage.save_stem_cv_timeline(
        SESSION, stem.stem_name, timeline.model_copy(update={"source_window": StemWindow.of(old_stem)})
    )

    pipeline._load_or_compute_cv(None, _metadata(), stem, {})

    assert fake_cv == ["spilled_juice"]
    assert storage.load_stem_cv_timeline(SESSION, stem.stem_name).source_window == StemWindow.of(stem)


def test_cv_timeline_is_reused_when_window_matches_or_unknown(fake_cv):
    for name, window in (("spilled_juice", StemWindow.of(STEMS[1])), ("lost_dog", None)):
        timeline = to_timeline(SESSION, name, 2.0, StemCvFrames()).model_copy(update={"source_window": window})
        storage.save_stem_cv_timeline(SESSION, name, timeline)

    pipeline._load_or_compute_cv(None, _metadata(), STEMS[1], {})
    pipeline._load_or_compute_cv(None, _metadata(), STEMS[2], {})

    assert fake_cv == []


# --- 프레임·오디오 재추출 -----------------------------------------------------------


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    calls = []

    def frames(video_path, stem, out_dir, fps):
        calls.append(("frames", stem.stem_name))
        d = out_dir / stem.stem_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "frame_000001.png").write_bytes(b"")
        return d

    def audio(video_path, stem, out_dir):
        calls.append(("audio", stem.stem_name))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem.stem_name}_child_segment.wav"
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(pipeline.video_prep, "extract_stem_frames", frames)
    monkeypatch.setattr(pipeline.video_prep, "extract_stem_audio", audio)
    return calls


def test_media_reextracted_only_when_window_changes(fake_ffmpeg, tmp_path):
    video = tmp_path / "v.mp4"
    stem = STEMS[1]

    pipeline._ensure_stem_media(video, SESSION, stem)
    pipeline._ensure_stem_media(video, SESSION, stem)
    assert fake_ffmpeg == [("frames", "spilled_juice"), ("audio", "spilled_juice")]

    pipeline._ensure_stem_media(video, SESSION, _stem("spilled_juice", 20.0, 30.0, 55.0))
    assert fake_ffmpeg[2:] == [("frames", "spilled_juice"), ("audio", "spilled_juice")]


def test_media_without_child_turn_returns_no_audio(fake_ffmpeg, tmp_path):
    margin = _stem("pre_session", 0.0, 5.0, 5.0, scored=False)
    _, audio = pipeline._ensure_stem_media(tmp_path / "v.mp4", SESSION, margin)
    assert audio is None
    assert ("audio", "pre_session") not in fake_ffmpeg


def test_media_needs_video_when_frames_missing(fake_ffmpeg):
    with pytest.raises(FileNotFoundError, match="원본 영상"):
        pipeline._ensure_stem_media(None, SESSION, STEMS[1])


# --- 종합 입력 검증 -----------------------------------------------------------------


def test_synthesis_input_stops_when_a_scored_stem_is_missing(fake_cv, fake_coder):
    _setup_session()
    fake_coder.fail_on = {"lost_dog"}
    with pytest.raises(pipeline.StageFailedError):
        pipeline.run_coding_stage(None, _metadata())

    with pytest.raises(RuntimeError, match="lost_dog"):
        pipeline.load_codings_for_synthesis(SESSION)


def test_synthesis_input_stops_when_a_stem_was_coded_with_old_window(fake_cv, fake_coder):
    _setup_session()
    pipeline.run_coding_stage(None, _metadata())
    storage.save_stem_timestamps(SESSION, [STEMS[0], _stem("spilled_juice", 20.0, 33.0, 60.0), STEMS[2]])

    with pytest.raises(RuntimeError, match="구간이 바뀐 뒤"):
        pipeline.load_codings_for_synthesis(SESSION)


def test_synthesis_input_ignores_stale_files_of_other_stems(fake_cv, fake_coder):
    _setup_session()
    pipeline.run_coding_stage(None, _metadata())
    stale = storage.load_stem_coding(SESSION, "lost_dog").model_copy(update={"stem_name": "candy_theft"})
    storage.save_stem_coding(SESSION, "candy_theft", stale)

    results = pipeline.load_codings_for_synthesis(SESSION)

    assert [r.stem_name for r in results] == ["spilled_juice", "lost_dog"]
