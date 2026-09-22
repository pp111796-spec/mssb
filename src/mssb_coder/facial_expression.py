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
    au_intensities: dict[str, float] = field(default_factory=dict)  # 예: {"AU04": 2.1}
    head_pitch_deg: float = 0.0  # 고개 숙임 각도 — 클수록 아래를 봄
    landmark_confidence: float = 1.0  # 0~1


LOW_CONFIDENCE_HEAD_PITCH_DEG = 35.0  # TODO: cv_thresholds_*.yaml로 옮기고 실측으로 조정 (마일스톤 A)
LOW_CONFIDENCE_LANDMARK_THRESHOLD = 0.5  # TODO: 위와 동일


def is_low_confidence(frame: FacialFrame) -> bool:
    """handoff 이후 구간 표정 신뢰도 저하 위험의 완화책 — 문제점.md 옛 8번(해결 반영)."""
    return (
        frame.head_pitch_deg >= LOW_CONFIDENCE_HEAD_PITCH_DEG
        or frame.landmark_confidence < LOW_CONFIDENCE_LANDMARK_THRESHOLD
    )


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
    face_score_col = _find_column(result.columns, "facescore") or _find_column(result.columns, "face", "score")

    frames: list[FacialFrame] = []
    for i, (_, row) in enumerate(result.iterrows()):
        au_intensities = {col: float(row[col]) for col in au_columns if not (row[col] != row[col])}  # NaN 제외
        head_pitch = abs(float(row[pitch_col])) if pitch_col else 0.0
        confidence = float(row[face_score_col]) if face_score_col else 1.0
        frames.append(
            FacialFrame(
                timestamp_s=i / fps,
                au_intensities=au_intensities,
                head_pitch_deg=head_pitch,
                landmark_confidence=confidence,
            )
        )
    return frames
