"""표정+자세+인형위치+음성운율 4개 시계열 → 타임스탬프 텍스트 요약 (파이프라인 8단계).

Claude에는 원본 영상/이미지를 절대 보내지 않으므로, 이 텍스트 요약이 비언어 신호가
AI 코딩(10단계)에 닿는 유일한 통로다 — 이 모듈의 출력 품질이 곧 CV 파이프라인 전체의 체감 가치다.

두 가지 완화책을 여기서 구현한다 (문제점.md 옛 8·9번, 계획에 반영됨):
  - 표정 저신뢰 구간 비율이 높으면 그 사실을 텍스트에 명시하고, 자세·음성 비중을 강조
  - 인형 추적 소실 구간(tracking_lost)을 조용히 누락시키지 않고 명시

이 모듈은 순수 텍스트 조합 로직이라 외부 의존성 없이 지금 실제로 테스트 가능하다.
"""

from __future__ import annotations

from mssb_coder.facial_expression import FacialFrame, is_low_confidence
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.voice_prosody import ProsodyEvent

LOW_CONFIDENCE_RATIO_WARNING_THRESHOLD = 0.3  # 이 비율 이상이면 텍스트에 경고 문구 추가


def _facial_confidence_summary(facial_frames: list[FacialFrame]) -> str:
    if not facial_frames:
        return "표정 데이터 없음."
    low_conf_count = sum(1 for f in facial_frames if is_low_confidence(f))
    ratio = low_conf_count / len(facial_frames)
    line = f"표정 저신뢰 프레임 비율: {ratio:.0%} ({low_conf_count}/{len(facial_frames)})"
    if ratio >= LOW_CONFIDENCE_RATIO_WARNING_THRESHOLD:
        line += (
            " — 이 구간은 표정 신호를 낮은 신뢰도로 다루고, 같은 시간대의 자세·음성 신호를 "
            "더 비중 있게 참고할 것."
        )
    return line


def _facial_event_lines(facial_frames: list[FacialFrame]) -> list[str]:
    lines = []
    for frame in facial_frames:
        if not frame.au_intensities:
            continue
        conf_tag = " [저신뢰]" if is_low_confidence(frame) else ""
        au_desc = ", ".join(f"{au}={val:.1f}" for au, val in sorted(frame.au_intensities.items()))
        lines.append(f"[{frame.timestamp_s:.1f}s] (facial){conf_tag} {au_desc}")
    return lines


def _posture_event_lines(events: list[PostureEvent]) -> list[str]:
    return [
        f"[{e.timestamp_s:.1f}s] (posture) {e.kind}={e.value:.1f}" + (f" — {e.note}" if e.note else "")
        for e in events
    ]


def _doll_event_lines(events: list[DollEvent]) -> list[str]:
    lines = []
    for e in events:
        dolls = ", ".join(e.doll_ids)
        if e.kind == "tracking_lost":
            lines.append(f"[{e.timestamp_s:.1f}s] (doll_tracking) [추적소실] {dolls} — {e.note}")
        else:
            value_str = f"={e.value:.2f}" if e.value is not None else ""
            lines.append(f"[{e.timestamp_s:.1f}s] (doll_tracking) {e.kind}{value_str}: {dolls}")
    return lines


def _prosody_event_lines(events: list[ProsodyEvent]) -> list[str]:
    return [f"[{e.timestamp_s:.1f}s] (voice_prosody) {e.kind} z={e.zscore:+.2f}" for e in events]


def summarize_stem_features(
    stem_name: str,
    facial_frames: list[FacialFrame],
    posture_events: list[PostureEvent],
    doll_events: list[DollEvent],
    prosody_events: list[ProsodyEvent],
) -> str:
    """AI 코딩 프롬프트(10단계)에 그대로 삽입되는 비언어 신호 텍스트 요약."""

    header = [f"=== {stem_name} 비언어 신호 요약 ===", _facial_confidence_summary(facial_frames)]

    tracking_lost_events = [e for e in doll_events if e.kind == "tracking_lost"]
    if tracking_lost_events:
        header.append(f"인형 추적 소실 구간: {len(tracking_lost_events)}건 (아래 목록에 [추적소실]로 표시)")

    all_lines = (
        _facial_event_lines(facial_frames)
        + _posture_event_lines(posture_events)
        + _doll_event_lines(doll_events)
        + _prosody_event_lines(prosody_events)
    )

    def _sort_key(line: str) -> float:
        # "[12.3s] ..." 형식에서 타임스탬프만 뽑아 정렬
        return float(line.split("s]", 1)[0].lstrip("["))

    all_lines.sort(key=_sort_key)

    if not all_lines:
        body = ["(이 구간에서 감지된 비언어 이벤트 없음 — 전사문 위주로 판단할 것)"]
    else:
        body = all_lines

    return "\n".join(header + [""] + body)
