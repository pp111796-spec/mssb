from mssb_coder.cv_events import StemEvents, to_session_time
from mssb_coder.facial_expression import FacialEvent
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.schema import StemTimestamp, VerbatimSpeaker
from mssb_coder.transcribe import TranscriptSegment
from mssb_coder.verbatim import build_stem_verbatim
from mssb_coder.voice_prosody import ProsodyEvent

EMPTY_SPEECH = {"examiner": [], "child": []}


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


def test_to_session_time_uses_stem_start_for_frames_and_handoff_for_audio():
    events = StemEvents(
        facial=[FacialEvent(timestamp_s=2.0, au="AU12", tier="reliable", peak_probability=0.9, frames=2)],
        posture=[PostureEvent(timestamp_s=1.0, kind="torso_tilt", value=40.0)],
        doll=[DollEvent(timestamp_s=4.0, kind="tracking_lost", doll_ids=("dog_doll",))],
        prosody=[ProsodyEvent(timestamp_s=3.0, kind="pitch_high", zscore=2.5)],
        quality_notes=["메모"],
    )

    shifted = to_session_time(events, _stem())

    assert shifted.facial[0].timestamp_s == 102.0  # 스템 시작(100) + 2.0
    assert shifted.posture[0].timestamp_s == 101.0
    assert shifted.doll[0].timestamp_s == 104.0
    assert shifted.prosody[0].timestamp_s == 113.0  # 전환 시각(110) + 3.0
    assert shifted.quality_notes == ["메모"]
    assert events.facial[0].timestamp_s == 2.0  # 원본은 그대로


def test_build_stem_verbatim_keeps_given_session_times():
    facial_events = [FacialEvent(timestamp_s=102.0, au="AU12", tier="reliable", peak_probability=0.9, frames=2)]
    prosody_events = [ProsodyEvent(timestamp_s=113.0, kind="pitch_high", zscore=2.5)]

    lines = build_stem_verbatim(EMPTY_SPEECH, facial_events, [], [], prosody_events)

    assert [line.timestamp_s for line in lines] == [102.0, 113.0]


def test_build_stem_verbatim_orders_chronologically_by_float_not_string():
    posture_events = [
        PostureEvent(timestamp_s=110.0, kind="torso_tilt", value=40.0),
        PostureEvent(timestamp_s=100.5, kind="sudden_movement", value=1.0),
    ]

    lines = build_stem_verbatim(EMPTY_SPEECH, [], posture_events, [], [])

    assert [line.timestamp_s for line in lines] == [100.5, 110.0]


def test_build_stem_verbatim_includes_speech_lines_with_speaker():
    speaker_tagged = {
        "examiner": [TranscriptSegment(start_s=101.0, end_s=103.0, text="그 다음엔 어떻게 될까?")],
        "child": [TranscriptSegment(start_s=104.0, end_s=106.0, text="주스를 다시 채웠어요.")],
    }

    lines = build_stem_verbatim(speaker_tagged, [], [], [], [])

    assert len(lines) == 2
    examiner_line = next(l for l in lines if l.speaker == VerbatimSpeaker.EXAMINER)
    child_line = next(l for l in lines if l.speaker == VerbatimSpeaker.CHILD)
    assert examiner_line.timestamp_s == 101.0
    assert examiner_line.text == "그 다음엔 어떻게 될까?"
    assert child_line.timestamp_s == 104.0
    assert child_line.kind == "speech"


def test_build_stem_verbatim_doll_event_label_uses_korean():
    doll_events = [
        DollEvent(timestamp_s=105.0, kind="tracking_lost", doll_ids=("dog_doll",), note="3프레임 연속 탐지 실패")
    ]

    lines = build_stem_verbatim(EMPTY_SPEECH, [], [], doll_events, [])

    assert len(lines) == 1
    assert "추적 소실" in lines[0].label_ko
    assert "강아지인형" in lines[0].label_ko
    assert "tracking_lost" not in lines[0].label_ko
    assert "dog_doll" not in lines[0].label_ko


def test_build_stem_verbatim_empty_input_returns_empty_list():
    assert build_stem_verbatim(EMPTY_SPEECH, [], [], [], []) == []
