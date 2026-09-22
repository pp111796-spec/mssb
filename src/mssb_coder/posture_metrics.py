"""랜드마크 → 몸통각도/이동량 (파이프라인 7단계, pose_analysis.py의 출력을 입력으로 받음).

별도 모델이 아니라 MediaPipe 랜드마크 위에 직접 짜는 각도/이동량 계산 로직
(계획 "관절/자세 분석" 섹션, 참고: nilutpolkashyap/body_posture_analysis).
이 파일은 순수 수학 계산이라 실제 영상 없이도 지금 구현·테스트 가능하다.

임계값(몇 도 이상이면 회피로 볼지 등)은 아직 미확정 — cv_thresholds_*.yaml의 null 자리표시자
참고, 문제점.md 3번(미해결) 그대로다. 이 모듈은 "각도를 계산하는 방법"만 확정하고,
"그 각도가 얼마면 이벤트로 볼지"는 호출하는 쪽(feature_summary.py)이 cv_thresholds에서 읽어 판단한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mssb_coder.pose_analysis import PoseFrame


@dataclass(frozen=True)
class PostureEvent:
    timestamp_s: float
    kind: str  # "torso_tilt" | "sudden_movement"
    value: float
    note: str = ""


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


def compute_posture_events(
    frames: list[PoseFrame],
    torso_tilt_avoidance_deg: float | None,
    sudden_movement_threshold: float | None,
) -> list[PostureEvent]:
    """cv_thresholds_*.yaml에서 읽은 임계값을 인자로 받아 이벤트 목록을 만든다.

    두 임계값 모두 아직 실측 전이라 YAML에 null로 남아있을 수 있다(문제점.md 3번, 미해결).
    각 임계값이 None이면 **그 종류의 이벤트만** 조용히 건너뛴다 — 값을 임의로 지어내 채우는
    대신, "아직 이 신호는 설정 안 됨"을 그대로 반영한다. 실측값이 채워지면 자동으로 활성화된다.
    """
    events: list[PostureEvent] = []

    if torso_tilt_avoidance_deg is not None:
        for frame in frames:
            tilt = compute_torso_tilt_deg(frame)
            if tilt is not None and tilt >= torso_tilt_avoidance_deg:
                events.append(
                    PostureEvent(timestamp_s=frame.timestamp_s, kind="torso_tilt", value=tilt)
                )

    if sudden_movement_threshold is not None:
        for prev, curr in zip(frames, frames[1:]):
            movement = compute_movement_magnitude(prev, curr)
            if movement is not None and movement >= sudden_movement_threshold:
                events.append(
                    PostureEvent(timestamp_s=curr.timestamp_s, kind="sudden_movement", value=movement)
                )

    return events
