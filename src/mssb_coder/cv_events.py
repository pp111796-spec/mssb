"""CV 원본 프레임 + 임계값 + 워밍업 기준선 → 이벤트·품질 메모 (Claude 호출 없는 순수 로직).

파이프라인(run_coding_stage)과 scripts/calibrate_cv.py가 같은 이 경로로 이벤트를 만든다.
"측정 불가"(결측·저신뢰·기준선 없음)와 "아무 일 없었다"를 구분하려고, 신호별 사용 가능 여부를
quality_notes로 함께 돌려준다 — 이 메모가 feature_summary.py 머리말에 들어간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mssb_coder.cv_timeline import StemCvFrames
from mssb_coder.facial_expression import (
    FacialEvent,
    FacialThresholds,
    compute_facial_events,
    low_confidence_ratio,
)
from mssb_coder.object_tracking import DollEvent, DollThresholds, compute_doll_events
from mssb_coder.posture_metrics import (
    PostureBaseline,
    PostureEvent,
    PostureThresholds,
    compute_posture_baseline,
    compute_posture_events,
    missing_ratio,
)
from mssb_coder.voice_prosody import (
    ProsodyBaseline,
    ProsodyEvent,
    ProsodyThresholds,
    compute_prosody_baseline,
    compute_prosody_events,
)


@dataclass(frozen=True)
class CvThresholdSet:
    facial: FacialThresholds | None
    posture: PostureThresholds | None
    doll: DollThresholds | None
    prosody: ProsodyThresholds | None

    @classmethod
    def from_config(cls, cfg: dict) -> "CvThresholdSet":
        """cv_thresholds_{adult,child}.yaml 전체 dict. 블록의 핵심 값이 null이면 그 신호는 None(= 판정 안 함)."""
        return cls(
            facial=FacialThresholds.from_config(cfg["facial_au"]),
            posture=PostureThresholds.from_config(cfg["posture"]),
            doll=DollThresholds.from_config(cfg["doll_tracking"]),
            prosody=ProsodyThresholds.from_config(cfg["voice_prosody"]),
        )


@dataclass(frozen=True)
class CvBaselines:
    posture: PostureBaseline | None
    prosody: ProsodyBaseline | None
    notes: tuple[str, ...] = ()


@dataclass
class StemEvents:
    facial: list[FacialEvent] = field(default_factory=list)
    posture: list[PostureEvent] = field(default_factory=list)
    doll: list[DollEvent] = field(default_factory=list)
    prosody: list[ProsodyEvent] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)


def compute_baselines(warm_up: StemCvFrames | None, thresholds: CvThresholdSet) -> CvBaselines:
    """워밍업 구간 원본 프레임으로 자세·음성 기준선을 만든다. 만들 수 없으면 None과 사유를 함께 돌려준다."""
    if warm_up is None:
        return CvBaselines(None, None, ("워밍업 구간이 없어 자세·음성 기준선을 만들 수 없음",))

    notes: list[str] = []
    posture = compute_posture_baseline(warm_up.pose, thresholds.posture) if thresholds.posture else None
    if thresholds.posture and posture is None:
        notes.append("워밍업에서 자세 기준선을 만들기에 사용 가능한 프레임이 부족함")
    prosody = (
        compute_prosody_baseline(warm_up.prosody, thresholds.prosody) if thresholds.prosody else None
    )
    if thresholds.prosody and prosody is None:
        notes.append("워밍업 유성음이 부족해 음성운율 기준선을 만들 수 없음")
    return CvBaselines(posture, prosody, tuple(notes))


def derive_events(
    frames: StemCvFrames, thresholds: CvThresholdSet, baselines: CvBaselines
) -> StemEvents:
    """스템 1편의 원본 프레임에서 이벤트를 만든다. 신호가 사용 불가면 이벤트를 비우고 사유를 메모한다."""
    out = StemEvents()

    if thresholds.facial is not None and frames.facial:
        ratio = low_confidence_ratio(frames.facial, thresholds.facial)
        if ratio > thresholds.facial.max_low_confidence_ratio:
            out.quality_notes.append(
                f"표정 신호 사용 불가: 저신뢰 프레임 {ratio:.0%} (기준 {thresholds.facial.max_low_confidence_ratio:.0%} 초과)"
            )
        else:
            out.facial = compute_facial_events(frames.facial, thresholds.facial)

    if thresholds.posture is not None and frames.pose:
        missing = missing_ratio(frames.pose)
        if missing > thresholds.posture.max_missing_ratio:
            out.quality_notes.append(
                f"자세 신호 사용 불가: 결측·비정면 프레임 {missing:.0%} (기준 {thresholds.posture.max_missing_ratio:.0%} 초과)"
            )
        elif baselines.posture is None:
            out.quality_notes.append("자세 신호 판정 안 함: 워밍업 기준선 없음")
        else:
            out.posture = compute_posture_events(frames.pose, baselines.posture, thresholds.posture)

    if thresholds.prosody is not None and frames.prosody:
        if baselines.prosody is None:
            out.quality_notes.append("음성운율 판정 안 함: 워밍업 기준선 없음")
        else:
            out.prosody = compute_prosody_events(frames.prosody, baselines.prosody, thresholds.prosody)

    if not frames.doll:
        out.quality_notes.append("인형 추적 결과 없음: 가중치가 없거나 인형이 탐지되지 않음")
    else:
        out.doll = compute_doll_events(frames.doll, thresholds.doll)

    return out
