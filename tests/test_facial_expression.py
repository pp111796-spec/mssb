from mssb_coder.facial_expression import (
    FacialFrame,
    FacialThresholds,
    _find_column,
    compute_facial_events,
    head_pose_level,
    is_low_confidence,
    low_confidence_ratio,
)


def _thresholds(**overrides) -> FacialThresholds:
    values = dict(
        presence_probability_min=0.5,
        min_consecutive_frames=2,
        reliable_aus=("AU06", "AU12"),
        reference_aus=("AU04",),
        head_pose_caution_deg=20.0,
        head_pose_exclude_deg=35.0,
        landmark_confidence_min=0.5,
        max_low_confidence_ratio=0.5,
    )
    values.update(overrides)
    return FacialThresholds(**values)


def _frame(t, aus=None, pitch=5.0, yaw=0.0, conf=0.9):
    return FacialFrame(
        timestamp_s=t, au_intensities=aus or {}, head_pitch_deg=pitch, landmark_confidence=conf, head_yaw_deg=yaw
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


def test_is_low_confidence_head_yaw():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=5.0, landmark_confidence=0.9, head_yaw_deg=40.0)
    assert is_low_confidence(frame) is True


def test_is_low_confidence_landmark():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=5.0, landmark_confidence=0.2)
    assert is_low_confidence(frame) is True


def test_is_confident_when_both_good():
    frame = FacialFrame(timestamp_s=1.0, head_pitch_deg=5.0, landmark_confidence=0.9)
    assert is_low_confidence(frame) is False


def test_head_pose_level_three_tiers():
    assert head_pose_level(_frame(0, pitch=10), 20.0, 35.0) == "ok"
    assert head_pose_level(_frame(0, pitch=25), 20.0, 35.0) == "caution"
    assert head_pose_level(_frame(0, yaw=36), 20.0, 35.0) == "exclude"


def test_facial_event_needs_probability_at_or_above_threshold_for_consecutive_frames():
    frames = [
        _frame(0.0, {"AU12": 0.9}),
        _frame(0.5, {"AU12": 0.4}),  # 기준 미만 → 연속 끊김
        _frame(1.0, {"AU12": 0.8}),
        _frame(1.5, {"AU12": 0.7}),
    ]
    events = compute_facial_events(frames, _thresholds())
    assert len(events) == 1
    assert events[0].au == "AU12" and events[0].timestamp_s == 1.0 and events[0].frames == 2
    assert events[0].tier == "reliable"
    assert abs(events[0].peak_probability - 0.8) < 1e-9


def test_single_frame_blip_is_not_an_event():
    assert compute_facial_events([_frame(0.0, {"AU06": 0.95}), _frame(0.5)], _thresholds()) == []


def test_excluded_aus_are_never_reported():
    frames = [_frame(0.0, {"AU15": 0.99}), _frame(0.5, {"AU15": 0.99})]  # AU15는 아동 평가에서 F1=0
    assert compute_facial_events(frames, _thresholds()) == []


def test_reference_tier_and_caution_flag():
    frames = [_frame(0.0, {"AU04": 0.8}, pitch=25), _frame(0.5, {"AU04": 0.8}, pitch=25)]
    events = compute_facial_events(frames, _thresholds())
    assert events[0].tier == "reference" and events[0].caution is True


def test_low_confidence_frames_break_runs_and_never_count():
    frames = [
        _frame(0.0, {"AU06": 0.9}),
        _frame(0.5, {"AU06": 0.9}, pitch=50),  # 머리 과다 회전 → 제외, 연속 끊김
        _frame(1.0, {"AU06": 0.9}),
    ]
    assert compute_facial_events(frames, _thresholds()) == []


def test_low_confidence_ratio():
    frames = [_frame(0.0), _frame(0.5, pitch=50), _frame(1.0, pitch=50), _frame(1.5)]
    assert abs(low_confidence_ratio(frames, _thresholds()) - 0.5) < 1e-9


def test_events_empty_when_thresholds_missing():
    assert compute_facial_events([_frame(0.0, {"AU06": 0.9})] * 3, None) == []


def test_thresholds_from_config_none_when_presence_probability_null():
    cfg = dict(
        presence_probability_min=None, min_consecutive_frames=2, reliable_aus=["AU06"], reference_aus=[],
        head_pose_caution_deg=20.0, head_pose_exclude_deg=35.0, landmark_confidence_min=0.5,
        max_low_confidence_ratio=0.5,
    )
    assert FacialThresholds.from_config(cfg) is None
