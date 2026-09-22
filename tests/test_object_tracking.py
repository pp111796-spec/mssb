from mssb_coder.object_tracking import DollFrame, detect_tracking_lost_segments, run_object_tracking


def _frame(t, bbox):
    return DollFrame(timestamp_s=t, doll_id="dog_doll", bbox=bbox)


def test_detects_lost_segment_in_middle():
    frames = [
        _frame(0.0, (0, 0, 1, 1)),
        _frame(0.5, None),
        _frame(1.0, None),
        _frame(1.5, None),
        _frame(2.0, (0, 0, 1, 1)),
    ]
    events = detect_tracking_lost_segments(frames, "dog_doll", min_consecutive_missing_frames=3)
    assert len(events) == 1
    assert events[0].kind == "tracking_lost"
    assert "3프레임" in events[0].note


def test_ignores_short_gaps_below_threshold():
    frames = [
        _frame(0.0, (0, 0, 1, 1)),
        _frame(0.5, None),
        _frame(1.0, (0, 0, 1, 1)),
    ]
    events = detect_tracking_lost_segments(frames, "dog_doll", min_consecutive_missing_frames=3)
    assert events == []


def test_detects_lost_segment_running_to_end():
    frames = [
        _frame(0.0, (0, 0, 1, 1)),
        _frame(0.5, None),
        _frame(1.0, None),
        _frame(1.5, None),
    ]
    events = detect_tracking_lost_segments(frames, "dog_doll", min_consecutive_missing_frames=3)
    assert len(events) == 1
    assert "구간 끝까지" in events[0].note


def test_run_object_tracking_returns_empty_when_no_weights(tmp_path):
    """마일스톤 0(인형 파인튜닝) 이전에는 가중치가 없어 조용히 빈 리스트를 반환해야 한다 —
    가짜 탐지(학습 안 된 범용 YOLO)보다 빈 결과가 낫다는 설계 원칙."""
    empty_weights_dir = tmp_path / "doll_yolo"
    empty_weights_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    result = run_object_tracking(frames_dir, empty_weights_dir)
    assert result == []
