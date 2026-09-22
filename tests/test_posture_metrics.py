from mssb_coder.pose_analysis import PoseFrame
from mssb_coder.posture_metrics import (
    compute_movement_magnitude,
    compute_posture_events,
    compute_torso_tilt_deg,
)


def test_torso_tilt_zero_when_upright():
    frame = PoseFrame(
        timestamp_s=1.0,
        landmarks={
            "left_shoulder": (0.4, 0.2),
            "right_shoulder": (0.6, 0.2),
            "left_hip": (0.4, 0.5),
            "right_hip": (0.6, 0.5),
        },
    )
    tilt = compute_torso_tilt_deg(frame)
    assert tilt == 0.0


def test_torso_tilt_large_when_leaning():
    frame = PoseFrame(
        timestamp_s=1.0,
        landmarks={
            "left_shoulder": (0.7, 0.2),
            "right_shoulder": (0.9, 0.2),
            "left_hip": (0.4, 0.5),
            "right_hip": (0.6, 0.5),
        },
    )
    tilt = compute_torso_tilt_deg(frame)
    assert tilt > 30


def test_torso_tilt_none_when_landmarks_missing():
    frame = PoseFrame(timestamp_s=1.0, landmarks={"left_shoulder": (0.4, 0.2)})
    assert compute_torso_tilt_deg(frame) is None


def test_movement_magnitude():
    prev = PoseFrame(timestamp_s=0.0, landmarks={"nose": (0.5, 0.5)})
    curr = PoseFrame(timestamp_s=0.1, landmarks={"nose": (0.6, 0.5)})
    assert abs(compute_movement_magnitude(prev, curr) - 0.1) < 1e-9


def test_compute_posture_events_handles_null_thresholds_without_crashing():
    """cv_thresholds_*.yaml의 임계값이 아직 null인 상태 그대로 파이프라인이 도는지 확인 —
    None을 비교연산에 그냥 넘기면 TypeError가 나므로, 이 동작이 실제로 막혀있는지가 중요하다."""
    frames = [
        PoseFrame(
            timestamp_s=0.0,
            landmarks={
                "left_shoulder": (0.9, 0.2), "right_shoulder": (1.1, 0.2),
                "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
            },
        ),
        PoseFrame(
            timestamp_s=0.1,
            landmarks={
                "left_shoulder": (0.9, 0.2), "right_shoulder": (1.1, 0.2),
                "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
            },
        ),
    ]
    events = compute_posture_events(frames, torso_tilt_avoidance_deg=None, sudden_movement_threshold=None)
    assert events == []


def test_compute_posture_events_detects_tilt_and_movement():
    frames = [
        PoseFrame(
            timestamp_s=0.0,
            landmarks={
                "left_shoulder": (0.4, 0.2), "right_shoulder": (0.6, 0.2),
                "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
                "nose": (0.5, 0.1),
            },
        ),
        PoseFrame(
            timestamp_s=0.1,
            landmarks={
                "left_shoulder": (0.9, 0.2), "right_shoulder": (1.1, 0.2),
                "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
                "nose": (0.9, 0.1),  # 큰 이동
            },
        ),
    ]
    events = compute_posture_events(
        frames, torso_tilt_avoidance_deg=30.0, sudden_movement_threshold=0.2
    )
    kinds = {e.kind for e in events}
    assert "torso_tilt" in kinds
    assert "sudden_movement" in kinds
