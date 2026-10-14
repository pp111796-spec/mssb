from mssb_coder.facial_expression import FacialFrame
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.schema import StemTimestamp, VerbatimSpeaker
from mssb_coder.transcribe import TranscriptSegment
from mssb_coder.verbatim import build_stem_verbatim
from mssb_coder.voice_prosody import ProsodyEvent


def _stem(**overrides) -> StemTimestamp:
    defaults = dict(
        stem_name="spilled_juice",
        start_s=100.0,
        end_s=130.0,
        handoff_timestamp_s=110.0,
        confidence=0.95,
        matched_cue="완성요청 문구",
    )
    defaults.update(overrides)
    return StemTimestamp(**defaults)


def test_build_stem_verbatim_converts_offsets_correctly():
    stem = _stem()
    facial_frames = [FacialFrame(timestamp_s=2.0, au_intensities={"AU04": 1.0})]
    prosody_events = [ProsodyEvent(timestamp_s=3.0, kind="pitch_spike", zscore=2.5)]

    lines = build_stem_verbatim(
        stem, {"examiner": [], "child": []}, facial_frames, [], [], prosody_events
    )

    facial_line = next(l for l in lines if l.source == "facial_au")
    prosody_line = next(l for l in lines if l.source == "voice_prosody")
    assert facial_line.timestamp_s == 102.0  # stem.start_s(100) + 2.0
    assert prosody_line.timestamp_s == 113.0  # stem.handoff_timestamp_s(110) + 3.0


def test_build_stem_verbatim_orders_chronologically_by_float_not_string():
    stem = _stem()
    posture_events = [
        PostureEvent(timestamp_s=10.0, kind="torso_tilt", value=40.0),  # -> 110.0
        PostureEvent(timestamp_s=0.5, kind="sudden_movement", value=1.0),  # -> 100.5
    ]

    lines = build_stem_verbatim(stem, {"examiner": [], "child": []}, [], posture_events, [], [])

    timestamps = [line.timestamp_s for line in lines]
    assert timestamps == sorted(timestamps)
    assert timestamps == [100.5, 110.0]


def test_build_stem_verbatim_includes_speech_lines_with_speaker():
    stem = _stem()
    speaker_tagged = {
        "examiner": [TranscriptSegment(start_s=101.0, end_s=103.0, text="그 다음엔 어떻게 될까?")],
        "child": [TranscriptSegment(start_s=104.0, end_s=106.0, text="주스를 다시 채웠어요.")],
    }

    lines = build_stem_verbatim(stem, speaker_tagged, [], [], [], [])

    assert len(lines) == 2
    examiner_line = next(l for l in lines if l.speaker == VerbatimSpeaker.EXAMINER)
    child_line = next(l for l in lines if l.speaker == VerbatimSpeaker.CHILD)
    assert examiner_line.timestamp_s == 101.0  # 세션-절대, 추가 오프셋 없음
    assert examiner_line.text == "그 다음엔 어떻게 될까?"
    assert child_line.timestamp_s == 104.0
    assert child_line.kind == "speech"


def test_build_stem_verbatim_doll_event_label_uses_korean():
    stem = _stem()
    doll_events = [
        DollEvent(timestamp_s=5.0, kind="tracking_lost", doll_ids=("dog_doll",), note="3프레임 연속 탐지 실패")
    ]

    lines = build_stem_verbatim(stem, {"examiner": [], "child": []}, [], [], doll_events, [])

    assert len(lines) == 1
    assert "추적 소실" in lines[0].label_ko
    assert "강아지인형" in lines[0].label_ko
    assert "tracking_lost" not in lines[0].label_ko
    assert "dog_doll" not in lines[0].label_ko


def test_build_stem_verbatim_empty_input_returns_empty_list():
    stem = _stem()
    lines = build_stem_verbatim(stem, {"examiner": [], "child": []}, [], [], [], [])
    assert lines == []
