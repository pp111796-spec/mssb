"""표정+자세+인형위치+음성운율 4개 시계열 → 타임스탬프 텍스트 요약 (파이프라인 8단계).

Claude에는 원본 영상/이미지를 절대 보내지 않으므로, 이 텍스트 요약이 비언어 신호가
AI 코딩(10단계)에 닿는 유일한 통로다 — 이 모듈의 출력 품질이 곧 CV 파이프라인 전체의 체감 가치다.

두 가지 완화책을 여기서 구현한다 (문제점.md 옛 8·9번, 계획에 반영됨):
  - 표정 저신뢰 구간 비율이 높으면 그 사실을 텍스트에 명시하고, 자세·음성 비중을 강조
  - 인형 추적 소실 구간(tracking_lost)을 조용히 누락시키지 않고 명시
또한 신호별로 "사용 가능 여부"(결측·저신뢰·기준선 없음)를 quality_notes로 받아 머리말에 적는다 —
빈 이벤트가 "아무 일 없었다"가 아니라 "측정 불가"일 수 있다는 걸 Claude가 구분하게 하려는 것.

이 모듈은 순수 텍스트 조합 로직이라 외부 의존성 없이 지금 실제로 테스트 가능하다.
"""

from __future__ import annotations

from mssb_coder.facial_expression import (
    FacialEvent,
    FacialFrame,
    FacialThresholds,
    is_low_confidence,
    low_confidence_ratio,
)
from mssb_coder.labels import au_gloss, doll_event_kind_label, posture_event_kind_label, prosody_event_kind_label
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.voice_prosody import ProsodyEvent

LOW_CONFIDENCE_RATIO_WARNING_THRESHOLD = 0.3  # 임계값 설정이 없을 때 쓰는 기본 경고 비율


def _facial_confidence_summary(
    facial_frames: list[FacialFrame], thresholds: FacialThresholds | None
) -> str:
    if not facial_frames:
        return "표정 데이터 없음."
    if thresholds is not None:
        ratio = low_confidence_ratio(facial_frames, thresholds)
        warn_at = thresholds.max_low_confidence_ratio
    else:
        ratio = sum(1 for f in facial_frames if is_low_confidence(f)) / len(facial_frames)
        warn_at = LOW_CONFIDENCE_RATIO_WARNING_THRESHOLD
    low_conf_count = round(ratio * len(facial_frames))
    line = f"표정 저신뢰 프레임 비율: {ratio:.0%} ({low_conf_count}/{len(facial_frames)})"
    if ratio >= warn_at:
        line += (
            " — 이 구간은 표정 신호를 낮은 신뢰도로 다루고, 같은 시간대의 자세·음성 신호를 "
            "더 비중 있게 참고할 것."
        )
    return line


def _facial_event_lines(events: list[FacialEvent]) -> list[str]:
    lines = []
    for e in events:
        tags = ""
        if e.tier == "reference":
            tags += " [참고용]"
        if e.caution:
            tags += " [머리 회전 주의]"
        lines.append(
            f"[{e.timestamp_s:.1f}s] (표정){tags} {au_gloss(e.au)} 확률 {e.peak_probability:.2f}, "
            f"{e.frames}프레임 지속"
        )
    return lines


def _posture_event_lines(events: list[PostureEvent]) -> list[str]:
    return [
        f"[{e.timestamp_s:.1f}s] (자세) {posture_event_kind_label(e.kind)}"
        + (f" — {e.note}" if e.note else "")
        for e in events
    ]


def _doll_event_lines(events: list[DollEvent]) -> list[str]:
    lines = []
    for e in events:
        dolls = ", ".join(e.doll_ids)
        kind_ko = doll_event_kind_label(e.kind)
        if e.kind == "tracking_lost":
            lines.append(f"[{e.timestamp_s:.1f}s] (인형추적) [추적소실] {dolls} — {e.note}")
        else:
            note = f" — {e.note}" if e.note else ""
            lines.append(f"[{e.timestamp_s:.1f}s] (인형추적) {kind_ko}: {dolls}{note}")
    return lines


def _prosody_event_lines(events: list[ProsodyEvent]) -> list[str]:
    return [
        f"[{e.timestamp_s:.1f}s] (음성운율) {prosody_event_kind_label(e.kind)} z={e.zscore:+.2f}"
        + (f" — {e.note}" if e.note else "")
        for e in events
    ]


def summarize_stem_features(
    stem_name: str,
    facial_frames: list[FacialFrame],
    facial_events: list[FacialEvent],
    posture_events: list[PostureEvent],
    doll_events: list[DollEvent],
    prosody_events: list[ProsodyEvent],
    quality_notes: list[str] | None = None,
    facial_thresholds: FacialThresholds | None = None,
) -> str:
    """AI 코딩 프롬프트(10단계)에 그대로 삽입되는 비언어 신호 텍스트 요약."""

    header = [
        f"=== {stem_name} 비언어 신호 요약 ===",
        _facial_confidence_summary(facial_frames, facial_thresholds),
    ]
    header.extend(quality_notes or [])

    tracking_lost_events = [e for e in doll_events if e.kind == "tracking_lost"]
    if tracking_lost_events:
        header.append(f"인형 추적 소실 구간: {len(tracking_lost_events)}건 (아래 목록에 [추적소실]로 표시)")

    timed_lines: list[tuple[float, str]] = []
    for events, render in (
        (facial_events, _facial_event_lines),
        (posture_events, _posture_event_lines),
        (doll_events, _doll_event_lines),
        (prosody_events, _prosody_event_lines),
    ):
        timed_lines.extend(zip((e.timestamp_s for e in events), render(events)))
    timed_lines.sort(key=lambda pair: pair[0])
    all_lines = [line for _, line in timed_lines]

    if not all_lines:
        body = ["(이 구간에서 감지된 비언어 이벤트 없음 — 전사문 위주로 판단할 것)"]
    else:
        body = all_lines

    return "\n".join(header + [""] + body)
