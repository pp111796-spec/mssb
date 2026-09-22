from mssb_coder.facial_expression import FacialFrame
from mssb_coder.feature_summary import summarize_stem_features
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.voice_prosody import ProsodyEvent


def test_summarize_stem_features_orders_chronologically_and_flags_low_confidence():
    facial_frames = [
        FacialFrame(timestamp_s=1.0, au_intensities={"AU04": 2.0}, head_pitch_deg=50.0, landmark_confidence=0.9),
        FacialFrame(timestamp_s=3.0, au_intensities={"AU12": 1.5}, head_pitch_deg=5.0, landmark_confidence=0.95),
    ]
    posture_events = [PostureEvent(timestamp_s=2.0, kind="torso_tilt", value=45.0)]
    doll_events = [
        DollEvent(timestamp_s=4.0, kind="tracking_lost", doll_ids=("dog_doll",), note="3프레임 연속 탐지 실패")
    ]
    prosody_events = [ProsodyEvent(timestamp_s=1.5, kind="pitch_spike", zscore=2.3)]

    summary = summarize_stem_features(
        "spilled_juice", facial_frames, posture_events, doll_events, prosody_events
    )

    lines = summary.splitlines()
    # 헤더 다음 본문에서 타임스탬프 오름차순인지 확인
    body_lines = [l for l in lines if l.startswith("[")]
    timestamps = [float(l.split("s]")[0].lstrip("[")) for l in body_lines]
    assert timestamps == sorted(timestamps)

    assert "[저신뢰]" in summary  # head_pitch_deg=50.0인 첫 프레임이 저신뢰로 표시돼야 함
    assert "추적소실" in summary
    assert "인형 추적 소실 구간: 1건" in summary


def test_summarize_stem_features_handles_empty_input():
    summary = summarize_stem_features("empty_stem", [], [], [], [])
    assert "감지된 비언어 이벤트 없음" in summary
    assert "표정 데이터 없음" in summary
