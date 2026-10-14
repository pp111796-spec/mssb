import pytest

from mssb_coder.voice_prosody import (
    ProsodyFrame,
    ProsodyThresholds,
    compute_baseline,
    compute_prosody_baseline,
    compute_prosody_events,
    filter_valid_pitch,
)


def _thresholds(**overrides) -> ProsodyThresholds:
    values = dict(
        pitch_valid_min_hz=150.0,
        pitch_valid_max_hz=500.0,
        baseline_min_voiced_frames=10,
        window_s=1.0,
        window_min_voiced_ratio=0.3,
        zscore_notable_threshold=2.0,
    )
    values.update(overrides)
    return ProsodyThresholds(**values)


def _frames(start_s, count, pitch, loudness=0.5):
    return [
        ProsodyFrame(timestamp_s=start_s + i * 0.01, pitch_hz=pitch, loudness=loudness) for i in range(count)
    ]


def _varied_warm_up(count=100):
    """분산이 0이 아닌 워밍업 — 220Hz 근처에서 ±10Hz 흔들림."""
    return [
        ProsodyFrame(timestamp_s=i * 0.01, pitch_hz=220.0 + (10.0 if i % 2 else -10.0), loudness=0.5)
        for i in range(count)
    ]


def test_compute_baseline():
    frames = [ProsodyFrame(timestamp_s=i, pitch_hz=200.0, loudness=0.5) for i in range(5)]
    baseline = compute_baseline(frames)
    assert baseline.pitch_mean == 200.0
    assert baseline.loudness_mean == 0.5


def test_compute_baseline_requires_min_frames():
    with pytest.raises(ValueError):
        compute_baseline([ProsodyFrame(timestamp_s=0, pitch_hz=200.0, loudness=0.5)])


def test_filter_valid_pitch_drops_tracking_errors():
    frames = [
        ProsodyFrame(0.0, 220.0, 0.5),
        ProsodyFrame(0.01, 90.0, 0.5),  # 성인 남성 수준 — 아동 피치 범위 밖
        ProsodyFrame(0.02, 800.0, 0.5),  # 피치 추적 오류
    ]
    kept = filter_valid_pitch(frames, _thresholds())
    assert [f.pitch_hz for f in kept] == [220.0]


def test_prosody_baseline_none_when_too_few_voiced_frames():
    thresholds = _thresholds(baseline_min_voiced_frames=300)
    assert compute_prosody_baseline(_varied_warm_up(100), thresholds) is None


def test_prosody_baseline_built_when_enough_voiced_frames():
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(_varied_warm_up(100), thresholds)
    assert baseline is not None
    assert abs(baseline.pitch_mean - 220.0) < 1e-6


def test_window_event_detects_high_pitch_with_direction():
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(_varied_warm_up(100), thresholds)

    calm = _frames(0.0, 60, 220.0)  # 0~1초 창, 평소와 같음
    excited = _frames(1.0, 60, 400.0)  # 1~2초 창, 크게 높음
    events = compute_prosody_events(calm + excited, baseline, thresholds)

    assert len(events) == 1
    assert events[0].kind == "pitch_high"
    assert events[0].timestamp_s == 1.0  # 창 시작 시각
    assert events[0].zscore > 2.0


def test_window_event_detects_low_loudness():
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(
        [ProsodyFrame(i * 0.01, 220.0 + (10.0 if i % 2 else -10.0), 0.5 + (0.1 if i % 2 else -0.1)) for i in range(100)],
        thresholds,
    )
    quiet = _frames(0.0, 60, 220.0, loudness=0.1)
    events = compute_prosody_events(quiet, baseline, thresholds)
    assert any(e.kind == "loudness_low" for e in events)


def test_window_with_too_few_voiced_frames_is_skipped():
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(_varied_warm_up(100), thresholds)
    sparse = _frames(0.0, 10, 400.0)  # 1초 창에 10프레임뿐(최소 30프레임 필요)
    assert compute_prosody_events(sparse, baseline, thresholds) == []


def test_prosody_events_one_per_window_not_per_frame():
    """10ms 프레임마다 이벤트를 만들면 폭주한다 — 높은 구간이 3초여도 창당 하나씩만."""
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(_varied_warm_up(100), thresholds)
    events = compute_prosody_events(_frames(0.0, 300, 400.0), baseline, thresholds)
    assert len([e for e in events if e.kind == "pitch_high"]) == 3


def test_prosody_events_empty_when_thresholds_or_baseline_missing():
    """cv_thresholds_*.yaml 값이 null이거나 기준선을 못 만든 상태에서도 크래시 없이 빈 리스트."""
    thresholds = _thresholds(baseline_min_voiced_frames=50)
    baseline = compute_prosody_baseline(_varied_warm_up(100), thresholds)
    frames = _frames(0.0, 60, 400.0)
    assert compute_prosody_events(frames, baseline, None) == []
    assert compute_prosody_events(frames, None, thresholds) == []


def test_thresholds_from_config_none_when_any_value_null():
    cfg = dict(
        pitch_valid_min_hz=150.0, pitch_valid_max_hz=500.0, baseline_min_voiced_frames=300,
        window_s=1.0, window_min_voiced_ratio=0.3, zscore_notable_threshold=None,
    )
    assert ProsodyThresholds.from_config(cfg) is None
