"""랜드마크 → 몸통각도/이동량 (파이프라인 7단계, pose_analysis.py의 출력을 입력으로 받음).

별도 모델이 아니라 MediaPipe 랜드마크 위에 직접 짜는 각도/이동량 계산 로직
(계획 "관절/자세 분석" 섹션, 참고: nilutpolkashyap/body_posture_analysis).

판정 기준은 절대값이 아니라 **그 아이의 워밍업 구간 대비 상대값(z-score)** 이다:
  - 절대 각도(예: 슬라우칭 30도)는 성인 사례뿐이고 아동 체격·연령·활동성 차이를 감당하지 못한다
    (문제점.md 6번). 음성운율(voice_prosody.py)이 이미 택한 "워밍업 대비" 원칙을 자세에도 쓴다.
  - 이동량은 몸통 길이(어깨중점~골반중점)로 나눠 카메라 거리·아동 크기 차이를 없앤다
    (영아 포즈 연구가 같은 방식으로 정규화함).
  - 아동(주 피사체)이 정면에 잡히지 않은 프레임(is_facing_camera=False)은 제외한다.
임계값 근거와 등급은 config/cv_thresholds_child.yaml 주석 참고. 값이 null이면 이벤트를 만들지 않는다.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from mssb_coder.pose_analysis import PoseFrame

_TILT_STDEV_FLOOR_DEG = 1.0
_MOVEMENT_STDEV_FLOOR_TORSO_RATIO = 0.02


@dataclass(frozen=True)
class PostureEvent:
    timestamp_s: float
    kind: str  # "torso_tilt" | "sudden_movement"
    value: float  # torso_tilt: 도(deg), sudden_movement: 몸통 길이 대비 이동 비율
    note: str = ""


@dataclass(frozen=True)
class PostureThresholds:
    max_missing_ratio: float
    baseline_min_frames: int
    noise_floor_torso_ratio: float
    sudden_movement_min_torso_ratio: float
    sudden_movement_zscore: float
    torso_tilt_min_delta_deg: float
    torso_tilt_zscore: float
    min_consecutive_frames: int

    @classmethod
    def from_config(cls, cfg: dict) -> "PostureThresholds | None":
        """cv_thresholds_*.yaml의 posture 블록. 값이 하나라도 null이면 None — 이벤트를 만들지 않는다."""
        keys = (
            "max_missing_ratio", "baseline_min_frames", "noise_floor_torso_ratio",
            "sudden_movement_min_torso_ratio", "sudden_movement_zscore",
            "torso_tilt_min_delta_deg", "torso_tilt_zscore", "min_consecutive_frames",
        )
        if any(cfg.get(k) is None for k in keys):
            return None
        return cls(**{k: cfg[k] for k in keys})


@dataclass(frozen=True)
class PostureBaseline:
    tilt_mean: float
    tilt_stdev: float
    movement_mean: float  # 몸통 길이 대비 비율
    movement_stdev: float


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def compute_torso_tilt_deg(frame: PoseFrame) -> float | None:
    """어깨 중점 → 골반 중점을 잇는 선이 수직축에서 몇 도 기울었는지.

    0도 = 똑바로 앉은 자세. 랜드마크가 없으면(가려짐 등) None.
    """
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in frame.landmarks for name in required):
        return None

    shoulder_mid = _midpoint(frame.landmarks["left_shoulder"], frame.landmarks["right_shoulder"])
    hip_mid = _midpoint(frame.landmarks["left_hip"], frame.landmarks["right_hip"])

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if dx == 0 and dy == 0:
        return 0.0
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle_from_vertical


def compute_torso_length(frame: PoseFrame) -> float | None:
    """어깨 중점~골반 중점 거리(0~1 정규화 좌표). 이동량 정규화의 기준 길이."""
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in frame.landmarks for name in required):
        return None
    shoulder_mid = _midpoint(frame.landmarks["left_shoulder"], frame.landmarks["right_shoulder"])
    hip_mid = _midpoint(frame.landmarks["left_hip"], frame.landmarks["right_hip"])
    length = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
    return length if length > 0 else None


def compute_movement_magnitude(prev: PoseFrame, curr: PoseFrame) -> float | None:
    """공통 랜드마크의 프레임간 평균 이동 거리 (0~1 정규화 좌표 기준)."""
    common = set(prev.landmarks) & set(curr.landmarks)
    if not common:
        return None
    total = 0.0
    for name in common:
        px, py = prev.landmarks[name]
        cx, cy = curr.landmarks[name]
        total += math.hypot(cx - px, cy - py)
    return total / len(common)


def compute_normalized_movement(prev: PoseFrame, curr: PoseFrame) -> float | None:
    """프레임 간 이동량을 두 프레임 몸통 길이의 평균으로 나눈 비율."""
    movement = compute_movement_magnitude(prev, curr)
    prev_len = compute_torso_length(prev)
    curr_len = compute_torso_length(curr)
    if movement is None or prev_len is None or curr_len is None:
        return None
    return movement / ((prev_len + curr_len) / 2)


def is_usable(frame: PoseFrame) -> bool:
    """아동이 정면에 잡혔고 몸통 랜드마크 4개가 모두 있는 프레임만 자세 분석에 쓴다."""
    return frame.is_facing_camera and compute_torso_tilt_deg(frame) is not None


def missing_ratio(frames: list[PoseFrame]) -> float:
    if not frames:
        return 1.0
    return sum(1 for f in frames if not is_usable(f)) / len(frames)


def _effective_movement(prev: PoseFrame, curr: PoseFrame, noise_floor: float) -> float | None:
    if not (is_usable(prev) and is_usable(curr)):
        return None
    movement = compute_normalized_movement(prev, curr)
    if movement is None:
        return None
    return 0.0 if movement < noise_floor else movement


def compute_posture_baseline(
    warm_up_frames: list[PoseFrame], thresholds: PostureThresholds
) -> PostureBaseline | None:
    """워밍업 구간 프레임으로 그 아동의 몸통 기울기·이동량 기준선을 만든다.

    사용 가능한 프레임이 baseline_min_frames보다 적으면 None — 이 경우 자세 이벤트는 만들지 않는다.
    """
    tilts = [t for t in (compute_torso_tilt_deg(f) for f in warm_up_frames if is_usable(f)) if t is not None]
    movements = [
        m
        for prev, curr in zip(warm_up_frames, warm_up_frames[1:])
        if (m := _effective_movement(prev, curr, thresholds.noise_floor_torso_ratio)) is not None
    ]
    if len(tilts) < thresholds.baseline_min_frames or len(movements) < 2:
        return None
    return PostureBaseline(
        tilt_mean=statistics.mean(tilts),
        tilt_stdev=max(statistics.pstdev(tilts), _TILT_STDEV_FLOOR_DEG),
        movement_mean=statistics.mean(movements),
        movement_stdev=max(statistics.pstdev(movements), _MOVEMENT_STDEV_FLOOR_TORSO_RATIO),
    )


def compute_posture_events(
    frames: list[PoseFrame],
    baseline: PostureBaseline | None,
    thresholds: PostureThresholds | None,
) -> list[PostureEvent]:
    """기준선(워밍업)이나 임계값이 없으면 조용히 빈 리스트 — 값을 지어내 채우지 않는다."""
    if baseline is None or thresholds is None:
        return []

    events: list[PostureEvent] = []

    run: list[tuple[float, float]] = []  # (timestamp_s, tilt)

    def _flush_tilt_run() -> None:
        if len(run) >= thresholds.min_consecutive_frames:
            events.append(
                PostureEvent(
                    timestamp_s=run[0][0],
                    kind="torso_tilt",
                    value=max(t for _, t in run),
                    note=f"워밍업 평균 {baseline.tilt_mean:.1f}도 대비 {len(run)}프레임 연속",
                )
            )

    for frame in frames:
        tilt = compute_torso_tilt_deg(frame) if is_usable(frame) else None
        if (
            tilt is not None
            and tilt - baseline.tilt_mean >= thresholds.torso_tilt_min_delta_deg
            and (tilt - baseline.tilt_mean) / baseline.tilt_stdev >= thresholds.torso_tilt_zscore
        ):
            run.append((frame.timestamp_s, tilt))
            continue
        _flush_tilt_run()
        run = []
    _flush_tilt_run()

    for prev, curr in zip(frames, frames[1:]):
        movement = _effective_movement(prev, curr, thresholds.noise_floor_torso_ratio)
        if movement is None or movement < thresholds.sudden_movement_min_torso_ratio:
            continue
        zscore = (movement - baseline.movement_mean) / baseline.movement_stdev
        if zscore >= thresholds.sudden_movement_zscore:
            events.append(
                PostureEvent(
                    timestamp_s=curr.timestamp_s,
                    kind="sudden_movement",
                    value=movement,
                    note=f"몸통 길이의 {movement:.0%} 이동",
                )
            )

    events.sort(key=lambda e: e.timestamp_s)
    return events
