"""검토 앱의 판단 로직 — Streamlit과 분리해 테스트할 수 있게 둔다 (화면은 src/review_app)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from pydantic import ValidationError

from mssb_coder import storage
from mssb_coder.schema import ExaminerSegment, StemTimestamp
from mssb_coder.segmentation import StoryStemsConfig
from mssb_coder.transcribe import TranscriptSegment

# 녹화 앞뒤 여백 구간 (pipeline.EDGE_SEGMENT_NAMES와 같은 값 — pipeline을 import하면 CV 모듈까지 끌려온다)
EDGE_SEGMENT_LABELS = {"pre_session": "녹화 시작 여백", "post_session": "녹화 끝 여백"}
LOW_CONFIDENCE = 0.7


def stem_choices(story_stems: StoryStemsConfig) -> dict[str, str]:
    """구간 표에서 고를 수 있는 이야기 {id: 표시 이름} — 실시 순서대로, 앞뒤 여백 포함."""
    choices = {"pre_session": EDGE_SEGMENT_LABELS["pre_session"]}
    choices.update({s.id: s.display_name_ko for s in story_stems.stems})
    choices["post_session"] = EDGE_SEGMENT_LABELS["post_session"]
    return choices


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors()[0]
    return str(first.get("msg", "")).removeprefix("Value error, ")


def validate_stem_timestamps(
    rows: list[dict],
    examiner_segments: dict[str, list[tuple[float, float]]],
    story_stems: StoryStemsConfig,
) -> tuple[list[StemTimestamp], list[str]]:
    """검토 화면에서 고친 구간 표를 검증한다. (시작 시각 순으로 정렬한 구간, 한글 오류 목록)을 돌려준다.

    오류가 하나라도 있으면 저장·확정하지 않는다. 채점 여부는 사람이 고를 수 없고 yaml을 따른다.
    """
    choices = stem_choices(story_stems)
    scored_by_id = {s.id: s.scored for s in story_stems.stems}
    errors: list[str] = []
    timestamps: list[StemTimestamp] = []

    names = [row.get("stem_name") or "" for row in rows]
    for name, count in Counter(n for n in names if n).items():
        if count > 1:
            errors.append(f"{choices.get(name, name)}: 같은 이야기가 {count}번 들어 있습니다. 하나만 남기세요.")

    for position, row in enumerate(rows, start=1):
        name = row.get("stem_name") or ""
        if not name:
            errors.append(f"{position}번째 줄: 이야기를 고르세요.")
            continue
        if name not in choices:
            errors.append(f"{position}번째 줄: 이야기 목록에 없는 이름입니다 ({name}).")
            continue
        label = choices[name]
        start, handoff, end = (_number(row.get(k)) for k in ("start_s", "handoff_timestamp_s", "end_s"))
        if None in (start, handoff, end):
            errors.append(f"{label}: 시작·전환·끝 시각을 모두 숫자로 입력하세요.")
            continue
        try:
            timestamps.append(
                StemTimestamp(
                    stem_name=name,
                    start_s=start,
                    end_s=end,
                    handoff_timestamp_s=handoff,
                    additional_examiner_segments_s=[
                        ExaminerSegment(start_s=s, end_s=e) for s, e in examiner_segments.get(name, [])
                    ],
                    confidence=_number(row.get("confidence")) if _number(row.get("confidence")) is not None else 1.0,
                    matched_cue=str(row.get("matched_cue") or "사람이 추가함"),
                    scored=scored_by_id.get(name, False),
                )
            )
        except ValidationError as exc:
            errors.append(f"{label}: {_validation_message(exc)}")

    timestamps.sort(key=lambda t: t.start_s)
    for previous, current in zip(timestamps, timestamps[1:]):
        if current.start_s < previous.end_s:
            errors.append(
                f"{choices[previous.stem_name]}와 {choices[current.stem_name]}의 구간이 겹칩니다 "
                f"({previous.end_s:.1f}초 > {current.start_s:.1f}초)."
            )
    return timestamps, errors


def format_examiner_segments(segments: list[tuple[float, float]]) -> str:
    return "; ".join(f"{start:.1f}-{end:.1f}" for start, end in segments)


def parse_examiner_segments(text: str) -> tuple[list[tuple[float, float]], str | None]:
    """'95.0-105.5; 120-128' → [(95.0, 105.5), (120.0, 128.0)]. 형식이 틀리면 (빈 목록, 한글 오류)."""
    segments: list[tuple[float, float]] = []
    for part in text.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = part.replace("~", "-").split("-")
        if len(pieces) != 2:
            return [], f"'{part}'는 '시작-끝' 형식이 아닙니다."
        start, end = (_number(p.strip()) for p in pieces)
        if start is None or end is None:
            return [], f"'{part}'에 숫자가 아닌 값이 있습니다."
        if start >= end:
            return [], f"'{part}'의 끝이 시작보다 뒤여야 합니다."
        segments.append((start, end))
    return sorted(segments), None


@dataclass(frozen=True)
class PreviewLine:
    timestamp_s: float
    speaker: str  # "검사자" | "아동" | "구간 밖"
    text: str


def transcript_preview(
    transcript: list[TranscriptSegment], stem: StemTimestamp, margin_s: float = 20.0
) -> list[PreviewLine]:
    """구간 앞뒤 margin_s초까지의 발화를, 지금 경계로 나눈 화자와 함께 보여준다 — 경계가 맞는지 눈으로 확인용."""

    def speaker(seg: TranscriptSegment) -> str:
        if not stem.start_s <= seg.start_s < stem.end_s:
            return "구간 밖"
        if seg.start_s < stem.handoff_timestamp_s:
            return "검사자"
        if any(extra.start_s <= seg.start_s < extra.end_s for extra in stem.additional_examiner_segments_s):
            return "검사자"
        return "아동"

    return [
        PreviewLine(seg.start_s, speaker(seg), seg.text)
        for seg in transcript
        if stem.start_s - margin_s <= seg.start_s < stem.end_s + margin_s
    ]


@dataclass(frozen=True)
class SessionProgress:
    label: str  # 사이드바에 보여줄 한 줄 상태
    has_timestamps: bool
    segmentation_status: str  # "unconfirmed" | "changed" | "confirmed" | "none"
    coded_count: int
    scored_count: int
    has_synthesis: bool
    synthesis_reviewed: bool


def session_progress(session_id: str) -> SessionProgress:
    d = storage.session_dir(session_id)
    has_timestamps = (d / "stem_timestamps.json").exists()
    status = storage.segmentation_review_status(session_id) if has_timestamps else "none"
    scored = [t.stem_name for t in storage.load_stem_timestamps(session_id) if t.scored] if has_timestamps else []
    coded = sum(1 for name in scored if (d / "coding" / f"{name}_ai_codes.json").exists())
    has_synthesis = (d / "session_summary_ai_draft.json").exists()
    synthesis_reviewed = (d / "session_summary_reviewed.json").exists()

    if not has_timestamps:
        label = "구간분리 전"
    elif status == "unconfirmed":
        label = "구간 확인 필요"
    elif status == "changed":
        label = "구간이 바뀜 — 다시 확정 필요"
    elif coded < len(scored):
        label = f"채점 진행 중 ({coded}/{len(scored)})"
    elif not has_synthesis:
        label = "종합 전"
    elif not synthesis_reviewed:
        label = "종합 완료 — 검토 전"
    else:
        label = "검토 저장됨"
    return SessionProgress(
        label=label,
        has_timestamps=has_timestamps,
        segmentation_status=status,
        coded_count=coded,
        scored_count=len(scored),
        has_synthesis=has_synthesis,
        synthesis_reviewed=synthesis_reviewed,
    )
