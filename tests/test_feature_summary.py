from mssb_coder.facial_expression import FacialEvent, FacialFrame
from mssb_coder.feature_summary import summarize_stem_features
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.voice_prosody import ProsodyEvent


def test_summarize_stem_features_orders_chronologically_and_flags_caution():
    facial_frames = [
        FacialFrame(timestamp_s=1.0, au_intensities={"AU06": 0.9}, head_pitch_deg=50.0, landmark_confidence=0.9),
        FacialFrame(timestamp_s=3.0, au_intensities={"AU12": 0.8}, head_pitch_deg=5.0, landmark_confidence=0.95),
    ]
    facial_events = [FacialEvent(timestamp_s=3.0, au="AU12", tier="reliable", peak_probability=0.8, frames=2, caution=True)]
    posture_events = [PostureEvent(timestamp_s=2.0, kind="torso_tilt", value=45.0, note="워밍업 대비")]
    doll_events = [
        DollEvent(timestamp_s=4.0, kind="tracking_lost", doll_ids=("dog_doll",), note="3프레임 연속 탐지 실패")
    ]
    prosody_events = [ProsodyEvent(timestamp_s=1.5, kind="pitch_high", zscore=2.3)]

    summary = summarize_stem_features(
        "spilled_juice", facial_frames, facial_events, posture_events, doll_events, prosody_events
    )

    body_lines = [l for l in summary.splitlines() if l.startswith("[")]
    timestamps = [float(l.split("s]")[0].lstrip("[")) for l in body_lines]
    assert timestamps == sorted(timestamps) == [1.5, 2.0, 3.0, 4.0]

    assert "[머리 회전 주의]" in summary
    assert "추적소실" in summary
    assert "인형 추적 소실 구간: 1건" in summary
    assert "평소보다 높음" in summary  # 음성 이벤트 방향이 한글로


def test_summarize_stem_features_marks_reference_tier():
    facial_events = [FacialEvent(timestamp_s=1.0, au="AU04", tier="reference", peak_probability=0.7, frames=2)]
    summary = summarize_stem_features("s", [], facial_events, [], [], [])
    assert "[참고용]" in summary
    assert "확률 0.70" in summary


def test_summarize_stem_features_includes_quality_notes():
    summary = summarize_stem_features("s", [], [], [], [], [], quality_notes=["자세 신호 사용 불가: 결측 40%"])
    assert "자세 신호 사용 불가" in summary


def test_summarize_stem_features_handles_empty_input():
    summary = summarize_stem_features("empty_stem", [], [], [], [], [])
    assert "감지된 비언어 이벤트 없음" in summary
    assert "표정 데이터 없음" in summary
