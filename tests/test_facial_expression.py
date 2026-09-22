from mssb_coder.facial_expression import (
    FacialFrame,
    _find_column,
    is_low_confidence,
)


def test_find_column_matches_case_insensitively():
    columns = ["Pitch", "Yaw", "Roll", "FaceScore"]
    assert _find_column(columns, "pitch") == "Pitch"
    assert _find_column(columns, "facescore") == "FaceScore"
    assert _find_column(columns, "nope") is None


def test_find_column_requires_all_keywords():
    columns = ["Face_Score", "Pitch"]
    assert _find_column(columns, "face", "score") == "Face_Score"


def test_is_low_confidence_head_pitch():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=40.0, landmark_confidence=0.9)
    assert is_low_confidence(frame) is True


def test_is_low_confidence_landmark():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=5.0, landmark_confidence=0.2)
    assert is_low_confidence(frame) is True


def test_is_confident_when_both_good():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=5.0, landmark_confidence=0.9)
    assert is_low_confidence(frame) is False
