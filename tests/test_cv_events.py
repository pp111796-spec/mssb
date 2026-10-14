from pathlib import Path

import yaml

from mssb_coder.cv_events import CvThresholdSet, compute_baselines, derive_events
from mssb_coder.cv_timeline import StemCvFrames
from mssb_coder.facial_expression import FacialFrame
from mssb_coder.object_tracking import DollFrame
from mssb_coder.pose_analysis import PoseFrame
from mssb_coder.voice_prosody import ProsodyFrame

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _thresholds(profile="child") -> CvThresholdSet:
    cfg = yaml.safe_load((CONFIG_DIR / f"cv_thresholds_{profile}.yaml").read_text(encoding="utf-8"))
    return CvThresholdSet.from_config(cfg)


def _pose(t, shoulder_dx=0.0, facing=True):
    return PoseFrame(
        timestamp_s=t,
        landmarks={
            "left_shoulder": (0.4 + shoulder_dx, 0.2), "right_shoulder": (0.6 + shoulder_dx, 0.2),
            "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5), "nose": (0.5, 0.1),
        },
        is_facing_camera=facing,
    )


def _warm_up_frames() -> StemCvFrames:
    return StemCvFrames(
        pose=[_pose(i * 0.5, shoulder_dx=0.01 * (i % 3)) for i in range(30)],
        prosody=[
            ProsodyFrame(i * 0.01, 220.0 + (10.0 if i % 2 else -10.0), 0.5) for i in range(400)
        ],
    )


def test_shipped_child_and_adult_configs_load_every_signal():
    """실제 yaml이 모든 신호에서 유효한 임계값 세트를 만들어야 한다 (null이 남아 신호가 꺼지지 않았는지)."""
    for profile in ("child", "adult"):
        t = _thresholds(profile)
        assert t.facial is not None and t.posture is not None and t.doll is not None and t.prosody is not None


def test_baselines_built_from_warm_up():
    baselines = compute_baselines(_warm_up_frames(), _thresholds())
    assert baselines.posture is not None and baselines.prosody is not None
    assert baselines.notes == ()


def test_baselines_explain_why_missing():
    baselines = compute_baselines(None, _thresholds())
    assert baselines.posture is None and baselines.prosody is None
    assert baselines.notes

    sparse = StemCvFrames(pose=[_pose(0.0)], prosody=[ProsodyFrame(0.0, 220.0, 0.5)])
    baselines = compute_baselines(sparse, _thresholds())
    assert baselines.posture is None and baselines.prosody is None
    assert len(baselines.notes) == 2


def test_derive_events_marks_facial_unusable_when_mostly_low_confidence():
    baselines = compute_baselines(_warm_up_frames(), _thresholds())
    frames = StemCvFrames(
        facial=[FacialFrame(timestamp_s=i * 0.5, au_intensities={"AU06": 0.9}, head_pitch_deg=50.0) for i in range(10)]
    )
    events = derive_events(frames, _thresholds(), baselines)
    assert events.facial == []
    assert any("표정 신호 사용 불가" in n for n in events.quality_notes)


def test_derive_events_marks_posture_unusable_when_many_frames_missing():
    baselines = compute_baselines(_warm_up_frames(), _thresholds())
    frames = StemCvFrames(pose=[_pose(i * 0.5, facing=(i % 2 == 0)) for i in range(10)])  # 절반이 비정면
    events = derive_events(frames, _thresholds(), baselines)
    assert events.posture == []
    assert any("자세 신호 사용 불가" in n for n in events.quality_notes)


def test_derive_events_notes_missing_doll_results():
    baselines = compute_baselines(_warm_up_frames(), _thresholds())
    events = derive_events(StemCvFrames(), _thresholds(), baselines)
    assert any("인형 추적 결과 없음" in n for n in events.quality_notes)


def test_derive_events_end_to_end_detects_each_signal():
    baselines = compute_baselines(_warm_up_frames(), _thresholds())
    pose = [_pose(0.0), _pose(0.5, shoulder_dx=0.3), _pose(1.0, shoulder_dx=0.3), _pose(1.5), _pose(2.0)]
    facial = [
        FacialFrame(timestamp_s=0.0, au_intensities={"AU12": 0.9}),
        FacialFrame(timestamp_s=0.5, au_intensities={"AU12": 0.9}),
        FacialFrame(timestamp_s=1.0),
        FacialFrame(timestamp_s=1.5),
        FacialFrame(timestamp_s=2.0),
    ]
    prosody = [ProsodyFrame(i * 0.01, 420.0, 0.5) for i in range(100)]  # 1초 창, 피치가 크게 높음
    doll = [DollFrame(t * 0.5, "dog_doll", None if t > 1 else (0.1, 0.1, 0.2, 0.2)) for t in range(6)]

    events = derive_events(StemCvFrames(facial=facial, pose=pose, prosody=prosody, doll=doll), _thresholds(), baselines)

    assert [e.au for e in events.facial] == ["AU12"]
    assert any(e.kind == "torso_tilt" for e in events.posture)
    assert [e.kind for e in events.prosody] == ["pitch_high"]
    assert any(e.kind == "tracking_lost" for e in events.doll)
