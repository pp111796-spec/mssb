"""faster-whisper 래퍼 — 세션 전체 전사 (파이프라인 2단계).

화자 태깅(검사자/아동 구분)은 여기서 하지 않는다. segmentation.py가 반환하는
handoff_timestamp_s / additional_examiner_segments_s 기준으로 단순 분기만 한다
(계획 "음성 — 화자 태깅" 섹션 — 별도 문자열 매칭을 두지 않기로 한 설계 결정).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mssb_coder.schema import ExaminerSegment, StemTimestamp


@dataclass(frozen=True)
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str


def transcribe_session(audio_path: Path, model_size: str = "large-v3") -> list[TranscriptSegment]:
    """faster-whisper로 세션 전체를 통으로 전사한다. 아직 스템별로 자르지 않은 상태.

    lazy import: faster-whisper가 설치돼 있지 않아도 이 모듈을 import할 수 있게 한다
    (requirements.txt에는 있지만, 지금 이 세션에서는 아직 설치하지 않음).
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(str(audio_path), language="ko", word_timestamps=False)
    return [
        TranscriptSegment(start_s=seg.start, end_s=seg.end, text=seg.text.strip())
        for seg in segments
    ]


def tag_speakers(
    transcript: list[TranscriptSegment], stem: StemTimestamp
) -> dict[str, list[TranscriptSegment]]:
    """handoff_timestamp_s / additional_examiner_segments_s를 기준으로 examiner/child 분기.

    별도 문자열 재검색 없음 — segmentation.py가 이미 찾아준 경계를 그대로 쓴다.
    """

    def _is_examiner_segment(seg: TranscriptSegment) -> bool:
        if seg.start_s < stem.handoff_timestamp_s:
            return True
        return any(
            extra.start_s <= seg.start_s < extra.end_s
            for extra in stem.additional_examiner_segments_s
        )

    stem_segments = [seg for seg in transcript if stem.start_s <= seg.start_s < stem.end_s]
    examiner_segments = [seg for seg in stem_segments if _is_examiner_segment(seg)]
    child_segments = [seg for seg in stem_segments if not _is_examiner_segment(seg)]
    return {"examiner": examiner_segments, "child": child_segments}
