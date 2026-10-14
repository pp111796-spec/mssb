import pytest

from mssb_coder.voice_prosody import ProsodyFrame, compute_baseline, compute_zscore_events


def test_compute_baseline():
    frames = [ProsodyFrame(timestamp_s=i, pitch_hz=200.0, loudness=0.5) for i in range(5)]
    baseline = compute_baseline(frames)
    assert baseline.pitch_mean == 200.0
    assert baseline.loudness_mean == 0.5


def test_compute_baseline_requires_min_frames():
    with pytest.raises(ValueError):
        compute_baseline([ProsodyFrame(timestamp_s=0, pitch_hz=200.0, loudness=0.5)])


def test_zscore_events_detects_spike():
    warm_up = [ProsodyFrame(timestamp_s=i, pitch_hz=200.0, loudness=0.5) for i in range(10)]
    baseline = compute_baseline(warm_up)

    child_frames = [
        ProsodyFrame(timestamp_s=20.0, pitch_hz=200.0, loudness=0.5),  # 평소와 같음 — 이벤트 아님
        ProsodyFrame(timestamp_s=21.0, pitch_hz=350.0, loudness=0.5),  # 피치 급상승
    ]
    events = compute_zscore_events(child_frames, baseline, notable_threshold=2.0)
    assert len(events) == 1
    assert events[0].kind == "pitch_spike"
    assert events[0].timestamp_s == 21.0


def test_zscore_events_returns_empty_when_threshold_is_none():
    """cv_thresholds_*.yaml의 zscore_notable_threshold가 아직 null인 상태에서도
    크래시 없이 빈 리스트를 반환해야 한다 (실제 파이프라인 실행에서 발견된 버그)."""
    warm_up = [ProsodyFrame(timestamp_s=i, pitch_hz=200.0, loudness=0.5) for i in range(10)]
    baseline = compute_baseline(warm_up)
    child_frames = [ProsodyFrame(timestamp_s=21.0, pitch_hz=350.0, loudness=0.5)]

    events = compute_zscore_events(child_frames, baseline, notable_threshold=None)
    assert events == []
