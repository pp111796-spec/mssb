"""저장된 CV 타임라인(cv/<stem>_cv.json) → 임계값 보정용 데이터 품질 진단·분포 리포트.

Claude를 호출하지 않고 디스크의 원본 수치만 읽는다. 목적은 두 가지:
  1) 신호별 "측정 가능했나"를 숫자로 보여준다 (얼굴 저신뢰 비율, 자세 결측률, 유성음 프레임 수, 인형 탐지율)
  2) 현재 임계값으로 이벤트가 몇 개 나오는지, 분포(p50/p90/p95/p99)가 어떤지 보여주고,
     사람이 육안으로 확인할 후보 순간(프레임 이미지 경로 포함)을 뽑아준다.
임계값은 이 리포트를 보고 사람이 정한다 — 이 모듈은 값을 자동으로 확정하지 않는다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from mssb_coder.cv_events import CvBaselines, CvThresholdSet, derive_events
from mssb_coder.cv_timeline import StemCvFrames
from mssb_coder.facial_expression import low_confidence_ratio
from mssb_coder.posture_metrics import (
    compute_normalized_movement,
    compute_torso_tilt_deg,
    is_usable,
    missing_ratio,
)


def percentiles(values: list[float], points: tuple[int, ...] = (50, 90, 95, 99)) -> dict[int, float]:
    """단순 최근접 순위 방식 백분위수. 값이 없으면 빈 dict."""
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        result[p] = ordered[index]
    return result


@dataclass
class StemDiagnosis:
    stem_name: str
    facial_frames: int = 0
    facial_low_confidence_ratio: float | None = None
    pose_frames: int = 0
    pose_missing_ratio: float | None = None
    prosody_voiced_frames: int = 0
    doll_frames: int = 0
    doll_detected_ratio: float | None = None
    event_counts: dict[str, int] = field(default_factory=dict)
    tilt_percentiles: dict[int, float] = field(default_factory=dict)
    movement_percentiles: dict[int, float] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)


def diagnose_stem(
    stem_name: str,
    frames: StemCvFrames,
    thresholds: CvThresholdSet,
    baselines: CvBaselines,
    stem_start_s: float = 0.0,
    handoff_s: float = 0.0,
    frames_dir: Path | None = None,
    fps: float = 2.0,
    candidates_per_signal: int = 5,
) -> StemDiagnosis:
    """스템 1편의 데이터 품질·분포·이벤트 수·육안 확인 후보를 계산한다.

    stem_start_s/handoff_s를 주면 후보 시각을 세션 절대 시각으로 표시한다 (표정·자세·인형은 스템 기준,
    음성은 handoff 기준 상대 시각이라 각각 더해준다).
    """
    d = StemDiagnosis(stem_name=stem_name)

    d.facial_frames = len(frames.facial)
    if thresholds.facial is not None and frames.facial:
        d.facial_low_confidence_ratio = low_confidence_ratio(frames.facial, thresholds.facial)

    d.pose_frames = len(frames.pose)
    if frames.pose:
        d.pose_missing_ratio = missing_ratio(frames.pose)

    d.prosody_voiced_frames = len(frames.prosody)

    d.doll_frames = len(frames.doll)
    if frames.doll:
        d.doll_detected_ratio = sum(1 for f in frames.doll if f.bbox is not None) / len(frames.doll)

    tilts = [t for t in (compute_torso_tilt_deg(f) for f in frames.pose if is_usable(f)) if t is not None]
    movements = [
        m
        for prev, curr in zip(frames.pose, frames.pose[1:])
        if is_usable(prev) and is_usable(curr) and (m := compute_normalized_movement(prev, curr)) is not None
    ]
    d.tilt_percentiles = percentiles(tilts)
    d.movement_percentiles = percentiles(movements)

    events = derive_events(frames, thresholds, baselines)
    d.quality_notes = list(events.quality_notes)
    d.event_counts = {
        "표정": len(events.facial),
        "자세": len(events.posture),
        "인형": len(events.doll),
        "음성운율": len(events.prosody),
    }

    def _frame_hint(t_rel: float) -> str:
        absolute = stem_start_s + t_rel
        if frames_dir is None:
            return f"{absolute:.1f}s"
        index = int(round(t_rel * fps)) + 1
        return f"{absolute:.1f}s → {frames_dir / f'frame_{index:06d}.png'}"

    for e in sorted(events.facial, key=lambda e: -e.peak_probability)[:candidates_per_signal]:
        d.candidates.append(f"[표정] {e.au} 확률 {e.peak_probability:.2f} {e.frames}프레임 — {_frame_hint(e.timestamp_s)}")
    for e in sorted(events.posture, key=lambda e: -e.value)[:candidates_per_signal]:
        d.candidates.append(f"[자세] {e.kind} {e.value:.2f} — {_frame_hint(e.timestamp_s)}")
    for e in events.doll[:candidates_per_signal]:
        d.candidates.append(f"[인형] {e.kind} {','.join(e.doll_ids)} — {_frame_hint(e.timestamp_s)}")
    for e in sorted(events.prosody, key=lambda e: -abs(e.zscore))[:candidates_per_signal]:
        absolute = handoff_s + e.timestamp_s
        d.candidates.append(f"[음성] {e.kind} z={e.zscore:+.2f} — {absolute:.1f}s (오디오 구간 {e.timestamp_s:.1f}s)")
    return d


def format_report(diagnoses: list[StemDiagnosis]) -> str:
    """사람이 읽는 Markdown 리포트."""
    lines = ["# CV 임계값 보정 리포트", ""]
    for d in diagnoses:
        lines.append(f"## {d.stem_name}")
        lines.append(
            f"- 표정: 프레임 {d.facial_frames}개, 저신뢰 비율 "
            + (f"{d.facial_low_confidence_ratio:.0%}" if d.facial_low_confidence_ratio is not None else "측정 불가")
        )
        lines.append(
            f"- 자세: 프레임 {d.pose_frames}개, 결측·비정면 비율 "
            + (f"{d.pose_missing_ratio:.0%}" if d.pose_missing_ratio is not None else "측정 불가")
        )
        if d.tilt_percentiles:
            lines.append("  - 몸통 기울기(도) 분포: " + ", ".join(f"p{p}={v:.1f}" for p, v in d.tilt_percentiles.items()))
        if d.movement_percentiles:
            lines.append(
                "  - 프레임 간 이동(몸통 길이 대비) 분포: "
                + ", ".join(f"p{p}={v:.2f}" for p, v in d.movement_percentiles.items())
            )
        lines.append(f"- 음성: 유성음 프레임 {d.prosody_voiced_frames}개")
        lines.append(
            f"- 인형: 기록 {d.doll_frames}개, 탐지율 "
            + (f"{d.doll_detected_ratio:.0%}" if d.doll_detected_ratio is not None else "측정 불가(가중치 없음)")
        )
        lines.append("- 현재 임계값으로 나온 이벤트 수: " + ", ".join(f"{k} {v}" for k, v in d.event_counts.items()))
        for note in d.quality_notes:
            lines.append(f"  - ⚠️ {note}")
        if d.candidates:
            lines.append("- 육안 확인 후보:")
            lines.extend(f"  - {c}" for c in d.candidates)
        lines.append("")
    return "\n".join(lines)


def summarize_event_rates(diagnoses: list[StemDiagnosis]) -> dict[str, float]:
    """스템당 평균 이벤트 수 — 임계값이 너무 느슨/빡빡한지 대략 가늠하는 용도."""
    if not diagnoses:
        return {}
    keys = {k for d in diagnoses for k in d.event_counts}
    return {k: statistics.mean(d.event_counts.get(k, 0) for d in diagnoses) for k in keys}
