from mssb_coder.object_tracking import (
    DollFrame,
    DollThresholds,
    compute_doll_events,
    detect_proximity_events,
    detect_sudden_movement_events,
    detect_tracking_lost_segments,
    fill_missing_detections,
    run_object_tracking,
)


def _frame(t, bbox, doll_id="dog_doll"):
    return DollFrame(timestamp_s=t, doll_id=doll_id, bbox=bbox)


def _box(cx, cy, size=0.1):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


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


def test_fill_missing_detections_records_failed_frames_as_none():
    """탐지 실패 프레임을 기록해야 tracking_lost가 실제로 발생할 수 있다 (이전에는 영원히 불가능했음)."""
    box = (0.1, 0.1, 0.2, 0.2)
    detections = [{"dog_doll": box}, {}, {}, {}, {"dog_doll": box}]
    frames = fill_missing_detections(detections, fps=2.0, min_seen_frames=2)

    assert [f.bbox is None for f in frames] == [False, True, True, True, False]
    assert [f.timestamp_s for f in frames] == [0.0, 0.5, 1.0, 1.5, 2.0]
    events = detect_tracking_lost_segments(frames, "dog_doll", min_consecutive_missing_frames=3)
    assert len(events) == 1


def test_fill_missing_detections_ignores_dolls_that_barely_appear():
    """한 번만 잡힌 인형(오탐 가능)이나 안 나온 인형은 '소실'로 만들지 않는다."""
    box = (0.1, 0.1, 0.2, 0.2)
    detections = [{"dog_doll": box, "pot": box}, {"dog_doll": box}, {}]
    frames = fill_missing_detections(detections, fps=2.0, min_seen_frames=2)
    assert {f.doll_id for f in frames} == {"dog_doll"}


def test_proximity_event_when_boxes_close_for_consecutive_frames():
    frames = []
    for t in (0.0, 0.5, 1.0):
        frames.append(_frame(t, _box(0.30, 0.5), "mother_doll"))
        frames.append(_frame(t, _box(0.35, 0.5), "child_doll"))  # 상자가 겹침
    events = detect_proximity_events(frames, width_ratio=1.5, min_consecutive_frames=2)
    assert len(events) == 1
    assert events[0].kind == "proximity_seeking"
    assert set(events[0].doll_ids) == {"mother_doll", "child_doll"}


def test_no_proximity_event_when_far_apart():
    frames = []
    for t in (0.0, 0.5, 1.0):
        frames.append(_frame(t, _box(0.1, 0.5), "mother_doll"))
        frames.append(_frame(t, _box(0.9, 0.5), "child_doll"))
    assert detect_proximity_events(frames, width_ratio=1.5, min_consecutive_frames=2) == []


def test_sudden_movement_event_when_center_jumps():
    frames = [_frame(0.0, _box(0.1, 0.1)), _frame(0.5, _box(0.9, 0.9))]  # 대각선의 약 80% 이동
    events = detect_sudden_movement_events(frames, diag_ratio=0.25)
    assert len(events) == 1 and events[0].kind == "sudden_movement"


def test_small_doll_movement_is_not_an_event():
    frames = [_frame(0.0, _box(0.50, 0.5)), _frame(0.5, _box(0.52, 0.5))]
    assert detect_sudden_movement_events(frames, diag_ratio=0.25) == []


def test_compute_doll_events_empty_without_thresholds():
    assert compute_doll_events([_frame(0.0, None)] * 5, None) == []


def test_compute_doll_events_combines_kinds_in_time_order():
    thresholds = DollThresholds(
        detection_conf_min=0.25, class_min_seen_frames=2, tracking_lost_min_frames=3,
        proximity_center_distance_width_ratio=1.5, proximity_min_consecutive_frames=2,
        sudden_movement_diag_ratio=0.25,
    )
    frames = [_frame(0.0, _box(0.1, 0.1)), _frame(0.5, _box(0.9, 0.9)), _frame(1.0, None), _frame(1.5, None), _frame(2.0, None)]
    events = compute_doll_events(frames, thresholds)
    assert [e.kind for e in events] == ["sudden_movement", "tracking_lost"]
    assert [e.timestamp_s for e in events] == sorted(e.timestamp_s for e in events)


def test_thresholds_from_config_none_when_any_value_null():
    cfg = dict(
        detection_conf_min=0.25, class_min_seen_frames=2, tracking_lost_min_frames=None,
        proximity_center_distance_width_ratio=1.5, proximity_min_consecutive_frames=2,
        sudden_movement_diag_ratio=0.25,
    )
    assert DollThresholds.from_config(cfg) is None
