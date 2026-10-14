"""OpenFace(1순위)/Py-Feat(대체)/PyAFAR(subject_type=child) → AU 시계열 (파이프라인 6단계).

OpenFace는 Windows 바이너리를 수동 설치해야 해서 이 환경에서 자동화하지 못했다.
그래서 **Py-Feat을 1차 구현으로 채택**했다(계획이 "설치 마찰이 크면 즉시 전환"이라고
이미 명시해둔 경로). PyAFAR(subject_type=child)은 여전히 미구현 — 문제점.md 5번(AU 스키마
불일치 위험)이 그대로 남아있고, Py-Feat 자체도 성인 데이터 기반이라 아동 적용 시 같은 도메인
격차를 안고 있다(계획 "성인 시연 영상의 활용 범위" 표 참고).

Py-Feat의 정확한 AU 컬럼 이름을 하드코딩하지 않고 `Fex.aus`/`Fex.poses` 프로퍼티로 동적으로
가져온다 — 버전마다 AU 세트가 달라질 수 있어, 이름을 하드코딩하면 조용히 틀린 데이터를 만들
위험이 있기 때문이다. 단, 이 구현은 실제 얼굴이 있는 이미지로 끝까지 검증하지는 못했다
(합성 테스트 이미지로는 얼굴 탐지 자체가 안 됨) — 마일스톤 A에서 실제 영상으로 첫 검증할 것.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mssb_coder.schema import SubjectType


@dataclass(frozen=True)
class FacialFrame:
    timestamp_s: float
    # Py-Feat xgb AU 모델이 내는 보정된 확률(0~1) — OpenFace식 0~5 강도가 아니다. 예: {"AU06": 0.82}
    au_intensities: dict[str, float] = field(default_factory=dict)
    head_pitch_deg: float = 0.0  # 고개 숙임 각도(절댓값) — 클수록 아래를 봄
    landmark_confidence: float = 1.0  # 0~1
    head_yaw_deg: float = 0.0  # 좌우 회전 각도(절댓값)


DEFAULT_HEAD_POSE_EXCLUDE_DEG = 35.0
DEFAULT_LANDMARK_CONFIDENCE_MIN = 0.5


def head_pose_level(frame: FacialFrame, caution_deg: float, exclude_deg: float) -> str:
    """머리 방향이 15~20도를 넘으면 AU 검출 성능이 급락한다(Toser 2016) — 단계별로 신뢰도를 나눈다."""
    angle = max(abs(frame.head_pitch_deg), abs(frame.head_yaw_deg))
    if angle >= exclude_deg:
        return "exclude"
    if angle >= caution_deg:
        return "caution"
    return "ok"


def is_low_confidence(
    frame: FacialFrame,
    exclude_deg: float = DEFAULT_HEAD_POSE_EXCLUDE_DEG,
    min_landmark_confidence: float = DEFAULT_LANDMARK_CONFIDENCE_MIN,
) -> bool:
    """handoff 이후 구간 표정 신뢰도 저하 위험의 완화책 — 문제점.md 옛 8번(해결 반영)."""
    return (
        max(abs(frame.head_pitch_deg), abs(frame.head_yaw_deg)) >= exclude_deg
        or frame.landmark_confidence < min_landmark_confidence
    )


@dataclass(frozen=True)
class FacialThresholds:
    presence_probability_min: float
    min_consecutive_frames: int
    reliable_aus: tuple[str, ...]
    reference_aus: tuple[str, ...]
    head_pose_caution_deg: float
    head_pose_exclude_deg: float
    landmark_confidence_min: float
    max_low_confidence_ratio: float

    @classmethod
    def from_config(cls, cfg: dict) -> "FacialThresholds | None":
        """cv_thresholds_*.yaml의 facial_au 블록. 핵심 값이 null이면 None — 이벤트를 만들지 않는다."""
        if cfg.get("presence_probability_min") is None or not cfg.get("reliable_aus"):
            return None
        return cls(
            presence_probability_min=cfg["presence_probability_min"],
            min_consecutive_frames=cfg["min_consecutive_frames"],
            reliable_aus=tuple(cfg["reliable_aus"]),
            reference_aus=tuple(cfg.get("reference_aus") or ()),
            head_pose_caution_deg=cfg["head_pose_caution_deg"],
            head_pose_exclude_deg=cfg["head_pose_exclude_deg"],
            landmark_confidence_min=cfg["landmark_confidence_min"],
            max_low_confidence_ratio=cfg["max_low_confidence_ratio"],
        )


@dataclass(frozen=True)
class FacialEvent:
    timestamp_s: float
    au: str
    tier: str  # "reliable" | "reference"
    peak_probability: float
    frames: int
    caution: bool = False  # 머리가 많이 돌아간 프레임이 섞여 있음


def low_confidence_ratio(frames: list[FacialFrame], thresholds: FacialThresholds) -> float:
    if not frames:
        return 1.0
    low = sum(
        1
        for f in frames
        if is_low_confidence(f, thresholds.head_pose_exclude_deg, thresholds.landmark_confidence_min)
    )
    return low / len(frames)


def compute_facial_events(
    frames: list[FacialFrame], thresholds: FacialThresholds | None
) -> list[FacialEvent]:
    """AU 확률이 기준 이상인 프레임이 min_consecutive_frames 이상 이어질 때만 이벤트로 남긴다.

    신뢰도가 낮은 프레임(머리 과다 회전·얼굴 검출 불량)은 연속 구간을 끊는다. 채점 근거로 신뢰할 수 있는
    AU(reliable)와 참고용 AU(reference)만 대상 — 나머지 AU는 아동 대상 평가에서 정확도가 낮아 제외.
    """
    if thresholds is None:
        return []

    def _event_from_run(au: str, tier: str, run: list[FacialFrame]) -> FacialEvent | None:
        if len(run) < thresholds.min_consecutive_frames:
            return None
        return FacialEvent(
            timestamp_s=run[0].timestamp_s,
            au=au,
            tier=tier,
            peak_probability=max(f.au_intensities[au] for f in run),
            frames=len(run),
            caution=any(
                head_pose_level(f, thresholds.head_pose_caution_deg, thresholds.head_pose_exclude_deg)
                == "caution"
                for f in run
            ),
        )

    events: list[FacialEvent] = []
    tiers = [(au, "reliable") for au in thresholds.reliable_aus] + [
        (au, "reference") for au in thresholds.reference_aus
    ]
    for au, tier in tiers:
        run: list[FacialFrame] = []
        for frame in frames:
            usable = not is_low_confidence(
                frame, thresholds.head_pose_exclude_deg, thresholds.landmark_confidence_min
            )
            if usable and frame.au_intensities.get(au, 0.0) >= thresholds.presence_probability_min:
                run.append(frame)
                continue
            event = _event_from_run(au, tier, run)
            if event is not None:
                events.append(event)
            run = []
        event = _event_from_run(au, tier, run)
        if event is not None:
            events.append(event)

    events.sort(key=lambda e: e.timestamp_s)
    return events


def _find_column(columns, *keywords: str) -> str | None:
    """대소문자·버전에 안 흔들리게, 키워드를 포함하는 첫 컬럼명을 찾는다."""
    for col in columns:
        lowered = col.lower()
        if all(k in lowered for k in keywords):
            return col
    return None


def run_facial_expression(frames_dir: Path, subject_type: SubjectType, fps: float = 2.0) -> list[FacialFrame]:
    """Py-Feat Detectorv1로 frames_dir의 frame_%06d.png들을 처리한다.

    subject_type=child는 아직 별도 아동 특화 모델(PyAFAR)로 분기하지 않는다 — Py-Feat은
    성인 기준 모델이므로, 이 함수를 아동 영상에 그대로 쓰면 문제점.md 5·6번 위험을 그대로 안는다.
    """
    from feat import Detectorv1  # noqa: PLC0415

    frame_paths = [str(p) for p in sorted(Path(frames_dir).glob("frame_*.png"))]
    if not frame_paths:
        return []

    detector = Detectorv1(device="cpu")
    result = detector.detect(frame_paths, data_type="image", progress_bar=False)

    au_columns = list(result.aus.columns)
    pose_columns = list(result.poses.columns) if hasattr(result, "poses") else []
    pitch_col = _find_column(pose_columns, "pitch")
    yaw_col = _find_column(pose_columns, "yaw")
    face_score_col = _find_column(result.columns, "facescore") or _find_column(result.columns, "face", "score")

    frames: list[FacialFrame] = []
    for i, (_, row) in enumerate(result.iterrows()):
        au_intensities = {col: float(row[col]) for col in au_columns if not (row[col] != row[col])}  # NaN 제외
        head_pitch = abs(float(row[pitch_col])) if pitch_col else 0.0
        head_yaw = abs(float(row[yaw_col])) if yaw_col else 0.0
        # 얼굴을 못 찾은 프레임은 pose/score가 NaN — 그대로 두면 비교가 조용히 False가 되어
        # 신뢰도 게이트를 통과해버리므로 명시적으로 신뢰도 0으로 만든다.
        confidence = float(row[face_score_col]) if face_score_col else 1.0
        if confidence != confidence:
            confidence = 0.0
        if head_pitch != head_pitch:
            head_pitch = 0.0
        if head_yaw != head_yaw:
            head_yaw = 0.0
        frames.append(
            FacialFrame(
                timestamp_s=i / fps,
                au_intensities=au_intensities,
                head_pitch_deg=head_pitch,
                landmark_confidence=confidence,
                head_yaw_deg=head_yaw,
            )
        )
    return frames
