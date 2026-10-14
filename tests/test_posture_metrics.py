from mssb_coder.pose_analysis import PoseFrame
from mssb_coder.posture_metrics import (
    PostureThresholds,
    compute_movement_magnitude,
    compute_normalized_movement,
    compute_posture_baseline,
    compute_posture_events,
    compute_torso_length,
    compute_torso_tilt_deg,
    is_usable,
    missing_ratio,
)


def _thresholds(**overrides) -> PostureThresholds:
    values = dict(
        max_missing_ratio=0.20,
        baseline_min_frames=5,
        noise_floor_torso_ratio=0.10,
        sudden_movement_min_torso_ratio=0.15,
        sudden_movement_zscore=2.0,
        torso_tilt_min_delta_deg=10.0,
        torso_tilt_zscore=2.0,
        min_consecutive_frames=2,
    )
    values.update(overrides)
    return PostureThresholds(**values)


def _frame(t, shoulder_dx=0.0, facing=True, nose_dx=0.0):
    """어깨를 shoulder_dx만큼 옆으로 밀어 몸통을 기울인 프레임. 몸통 길이는 0.3."""
    return PoseFrame(
        timestamp_s=t,
        landmarks={
            "left_shoulder": (0.4 + shoulder_dx, 0.2), "right_shoulder": (0.6 + shoulder_dx, 0.2),
            "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
            "nose": (0.5 + nose_dx, 0.1),
        },
        is_facing_camera=facing,
    )


def _warm_up():
    """약간씩 흔들리는 워밍업(기준선 분산이 0이 되지 않도록)."""
    return [_frame(i * 0.5, shoulder_dx=0.01 * (i % 3)) for i in range(20)]


def test_torso_tilt_zero_when_upright():
    assert compute_torso_tilt_deg(_frame(1.0)) == 0.0


def test_torso_tilt_large_when_leaning():
    frame = PoseFrame(
        timestamp_s=1.0,
        landmarks={
            "left_shoulder": (0.7, 0.2), "right_shoulder": (0.9, 0.2),
            "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
        },
    )
    assert compute_torso_tilt_deg(frame) > 30


def test_torso_tilt_none_when_landmarks_missing():
    frame = PoseFrame(timestamp_s=1.0, landmarks={"left_shoulder": (0.4, 0.2)})
    assert compute_torso_tilt_deg(frame) is None


def test_torso_length_is_shoulder_to_hip_distance():
    assert abs(compute_torso_length(_frame(0.0)) - 0.3) < 1e-9


def test_movement_magnitude():
    prev = PoseFrame(timestamp_s=0.0, landmarks={"nose": (0.5, 0.5)})
    curr = PoseFrame(timestamp_s=0.1, landmarks={"nose": (0.6, 0.5)})
    assert abs(compute_movement_magnitude(prev, curr) - 0.1) < 1e-9


def test_normalized_movement_divides_by_torso_length():
    prev = _frame(0.0)
    curr = _frame(0.5, nose_dx=0.0)
    curr.landmarks["nose"] = (0.5 + 0.3, 0.1)  # 코만 0.3 이동 → 평균 이동 0.3/5, 몸통 0.3
    # 공통 랜드마크 5개 중 코만 0.3 이동 → 평균 0.06, 몸통 길이 0.3 → 0.2
    assert abs(compute_normalized_movement(prev, curr) - 0.2) < 1e-9


def test_non_facing_frames_are_unusable_and_counted_as_missing():
    frames = [_frame(0.0), _frame(0.5, facing=False), _frame(1.0), _frame(1.5)]
    assert is_usable(frames[0]) and not is_usable(frames[1])
    assert abs(missing_ratio(frames) - 0.25) < 1e-9


def test_baseline_none_when_too_few_usable_frames():
    assert compute_posture_baseline([_frame(0.0), _frame(0.5)], _thresholds()) is None


def test_posture_events_empty_when_baseline_or_thresholds_missing():
    """임계값(cv_thresholds_*.yaml)이 null이거나 기준선을 못 만든 상태에서도 크래시 없이 빈 리스트."""
    baseline = compute_posture_baseline(_warm_up(), _thresholds())
    frames = [_frame(0.0, shoulder_dx=0.5), _frame(0.5, shoulder_dx=0.5)]
    assert compute_posture_events(frames, baseline, None) == []
    assert compute_posture_events(frames, None, _thresholds()) == []


def test_torso_tilt_event_requires_consecutive_frames_and_min_delta():
    thresholds = _thresholds()
    baseline = compute_posture_baseline(_warm_up(), thresholds)

    one_frame = [_frame(0.0), _frame(0.5, shoulder_dx=0.3), _frame(1.0)]
    assert [e for e in compute_posture_events(one_frame, baseline, thresholds) if e.kind == "torso_tilt"] == []

    two_frames = [_frame(0.0), _frame(0.5, shoulder_dx=0.3), _frame(1.0, shoulder_dx=0.3), _frame(1.5)]
    tilt_events = [e for e in compute_posture_events(two_frames, baseline, thresholds) if e.kind == "torso_tilt"]
    assert len(tilt_events) == 1
    assert tilt_events[0].timestamp_s == 0.5


def test_sudden_movement_event_uses_torso_normalized_size():
    thresholds = _thresholds()
    baseline = compute_posture_baseline(_warm_up(), thresholds)
    frames = [_frame(0.0), _frame(0.5, shoulder_dx=0.2, nose_dx=0.4)]  # 큰 몸통 이동
    events = compute_posture_events(frames, baseline, thresholds)
    assert any(e.kind == "sudden_movement" for e in events)


def test_small_jitter_below_noise_floor_is_not_an_event():
    thresholds = _thresholds()
    baseline = compute_posture_baseline(_warm_up(), thresholds)
    frames = [_frame(0.0), _frame(0.5, nose_dx=0.01)]
    assert [e for e in compute_posture_events(frames, baseline, thresholds) if e.kind == "sudden_movement"] == []


def test_events_skip_frames_where_child_is_not_facing_camera():
    thresholds = _thresholds()
    baseline = compute_posture_baseline(_warm_up(), thresholds)
    frames = [_frame(0.0), _frame(0.5, shoulder_dx=0.3, facing=False), _frame(1.0, shoulder_dx=0.3, facing=False)]
    assert compute_posture_events(frames, baseline, thresholds) == []


def test_thresholds_from_config_none_when_any_value_null():
    cfg = dict(
        max_missing_ratio=0.2, baseline_min_frames=10, noise_floor_torso_ratio=0.1,
        sudden_movement_min_torso_ratio=0.15, sudden_movement_zscore=None,
        torso_tilt_min_delta_deg=10.0, torso_tilt_zscore=2.0, min_consecutive_frames=2,
    )
    assert PostureThresholds.from_config(cfg) is None
