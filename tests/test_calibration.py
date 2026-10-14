from pathlib import Path

import yaml

from mssb_coder.calibration import diagnose_stem, format_report, percentiles, summarize_event_rates
from mssb_coder.cv_events import CvThresholdSet, compute_baselines
from mssb_coder.cv_timeline import StemCvFrames, from_timeline, to_timeline
from mssb_coder.facial_expression import FacialFrame
from mssb_coder.object_tracking import DollFrame
from mssb_coder.pose_analysis import PoseFrame
from mssb_coder.voice_prosody import ProsodyFrame

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _thresholds() -> CvThresholdSet:
    cfg = yaml.safe_load((CONFIG_DIR / "cv_thresholds_child.yaml").read_text(encoding="utf-8"))
    return CvThresholdSet.from_config(cfg)


def _pose(t, shoulder_dx=0.0):
    return PoseFrame(
        timestamp_s=t,
        landmarks={
            "left_shoulder": (0.4 + shoulder_dx, 0.2), "right_shoulder": (0.6 + shoulder_dx, 0.2),
            "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5), "nose": (0.5, 0.1),
        },
    )


def _warm_up() -> StemCvFrames:
    return StemCvFrames(
        pose=[_pose(i * 0.5, shoulder_dx=0.01 * (i % 3)) for i in range(30)],
        prosody=[ProsodyFrame(i * 0.01, 220.0 + (10.0 if i % 2 else -10.0), 0.5) for i in range(400)],
    )


def test_percentiles():
    values = [float(v) for v in range(1, 101)]
    p = percentiles(values)
    assert p[50] in (50.0, 51.0)
    assert p[99] >= 99.0
    assert percentiles([]) == {}


def test_diagnose_stem_reports_quality_counts_and_candidates():
    thresholds = _thresholds()
    baselines = compute_baselines(_warm_up(), thresholds)
    frames = StemCvFrames(
        facial=[FacialFrame(i * 0.5, {"AU12": 0.9}) for i in range(2)] + [FacialFrame(1.0, head_pitch_deg=50.0)],
        pose=[_pose(0.0), _pose(0.5, shoulder_dx=0.3), _pose(1.0, shoulder_dx=0.3), _pose(1.5)],
        prosody=[ProsodyFrame(i * 0.01, 420.0, 0.5) for i in range(100)],
        doll=[DollFrame(0.0, "dog_doll", (0.1, 0.1, 0.2, 0.2)), DollFrame(0.5, "dog_doll", None)],
    )
    d = diagnose_stem(
        "spilled_juice", frames, thresholds, baselines,
        stem_start_s=100.0, handoff_s=110.0, frames_dir=Path("frames/spilled_juice"),
    )

    assert d.facial_frames == 3
    assert abs(d.facial_low_confidence_ratio - 1 / 3) < 1e-9
    assert d.prosody_voiced_frames == 100
    assert d.doll_detected_ratio == 0.5
    assert d.event_counts["표정"] == 1
    assert d.event_counts["음성운율"] == 1
    assert d.tilt_percentiles and d.movement_percentiles
    # 후보 시각은 세션 절대 시각 + 해당 프레임 파일 경로(번호 = 시각*fps + 1)
    assert any("100.0s" in c and "frame_000001.png" in c for c in d.candidates)
    assert any("[음성]" in c and "110.0s" in c for c in d.candidates)


def test_format_report_marks_unmeasurable_signals_and_summaries():
    thresholds = _thresholds()
    baselines = compute_baselines(_warm_up(), thresholds)
    d = diagnose_stem("s", StemCvFrames(), thresholds, baselines)
    report = format_report([d])
    assert "# CV 임계값 보정 리포트" in report
    assert "측정 불가" in report
    assert summarize_event_rates([d]) == {"표정": 0, "자세": 0, "인형": 0, "음성운율": 0}
    assert summarize_event_rates([]) == {}


def test_cv_timeline_round_trip_preserves_values():
    frames = StemCvFrames(
        facial=[FacialFrame(0.5, {"AU06": 0.8}, head_pitch_deg=12.0, landmark_confidence=0.9, head_yaw_deg=7.0)],
        pose=[_pose(0.0)],
        prosody=[ProsodyFrame(0.01, 230.0, 0.4)],
        doll=[DollFrame(0.0, "dog_doll", (0.1, 0.2, 0.3, 0.4)), DollFrame(0.5, "dog_doll", None)],
    )
    restored = from_timeline(to_timeline("s1", "spilled_juice", 2.0, frames))

    assert restored.facial == frames.facial
    assert restored.pose[0].landmarks == frames.pose[0].landmarks
    assert restored.prosody == frames.prosody
    assert restored.doll == frames.doll
