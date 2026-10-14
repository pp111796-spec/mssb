"""발화 전사 + 4개 행동 신호(표정/자세/인형추적/음성운율)를 하나의 축어록으로 결합.

시간축이 모달리티마다 다르다는 게 이 모듈의 핵심 문제다:
  - 화자태깅된 발화 세그먼트(transcribe.tag_speakers 출력): 이미 세션-절대 시각
  - 표정·자세·인형추적 프레임/이벤트: 스템-상대 시각 (video_prep.extract_stem_frames가
    stem.start_s부터 프레임을 뽑기 때문 — stem.start_s + timestamp_s로 변환)
  - 음성운율 이벤트: handoff-상대 시각 (video_prep.extract_stem_audio가 아동 구간만,
    즉 stem.handoff_timestamp_s부터 오디오를 뽑기 때문 — stem.handoff_timestamp_s +
    timestamp_s로 변환)

Claude 호출 없는 순수 결합 로직이라 영상 없이도 지금 테스트 가능하다 (feature_summary.py와
같은 성격). feature_summary.py의 "[Ns]" 문자열 파싱식 정렬은 여기서는 쓰지 않고, 실제
float 필드로 정렬한다.
"""

from __future__ import annotations

from mssb_coder.facial_expression import FacialFrame, is_low_confidence
from mssb_coder.labels import au_gloss, doll_event_kind_label, doll_id_label, load_doll_labels, posture_event_kind_label, prosody_event_kind_label
from mssb_coder.object_tracking import DollEvent
from mssb_coder.posture_metrics import PostureEvent
from mssb_coder.schema import EvidenceSource, StemTimestamp, StemVerbatimRecord, VerbatimLine, VerbatimSpeaker
from mssb_coder.transcribe import TranscriptSegment
from mssb_coder.voice_prosody import ProsodyEvent


def build_stem_verbatim(
    stem: StemTimestamp,
    speaker_tagged: dict[str, list[TranscriptSegment]],
    facial_frames: list[FacialFrame],
    posture_events: list[PostureEvent],
    doll_events: list[DollEvent],
    prosody_events: list[ProsodyEvent],
) -> list[VerbatimLine]:
    """스템 1편의 발화+행동 신호를 세션-절대 시각 기준 하나의 정렬된 목록으로 합친다."""
    doll_labels = load_doll_labels()
    lines: list[VerbatimLine] = []

    for speaker_key, speaker_enum in (("examiner", VerbatimSpeaker.EXAMINER), ("child", VerbatimSpeaker.CHILD)):
        for seg in speaker_tagged.get(speaker_key, []):
            lines.append(
                VerbatimLine(timestamp_s=seg.start_s, kind="speech", speaker=speaker_enum, text=seg.text)
            )

    for frame in facial_frames:
        if not frame.au_intensities:
            continue
        conf_tag = " [저신뢰]" if is_low_confidence(frame) else ""
        au_desc = ", ".join(au_gloss(au) for au in sorted(frame.au_intensities))
        lines.append(
            VerbatimLine(
                timestamp_s=stem.start_s + frame.timestamp_s,
                kind="behavior",
                source=EvidenceSource.FACIAL_AU,
                label_ko=f"표정 변화{conf_tag}: {au_desc}",
            )
        )

    for event in posture_events:
        lines.append(
            VerbatimLine(
                timestamp_s=stem.start_s + event.timestamp_s,
                kind="behavior",
                source=EvidenceSource.POSTURE,
                label_ko=posture_event_kind_label(event.kind) + (f" — {event.note}" if event.note else ""),
                raw_value=event.value,
            )
        )

    for event in doll_events:
        dolls = ", ".join(doll_id_label(doll_labels, doll_id) for doll_id in event.doll_ids)
        label = f"{doll_event_kind_label(event.kind)}: {dolls}" if dolls else doll_event_kind_label(event.kind)
        if event.note:
            label += f" — {event.note}"
        lines.append(
            VerbatimLine(
                timestamp_s=stem.start_s + event.timestamp_s,
                kind="behavior",
                source=EvidenceSource.DOLL_TRACKING,
                label_ko=label,
                raw_value=event.value,
            )
        )

    for event in prosody_events:
        lines.append(
            VerbatimLine(
                timestamp_s=stem.handoff_timestamp_s + event.timestamp_s,
                kind="behavior",
                source=EvidenceSource.VOICE_PROSODY,
                label_ko=f"{prosody_event_kind_label(event.kind)} (z={event.zscore:+.2f})",
                raw_value=event.zscore,
            )
        )

    lines.sort(key=lambda line: line.timestamp_s)
    return lines


def render_verbatim_markdown(record: StemVerbatimRecord) -> str:
    """축어록 다운로드 버튼(review_app)용 Markdown — 시간순으로 발화와 행동신호를 함께 나열."""
    out = [f"# 축어록 — {record.stem_name}", ""]
    for line in record.lines:
        if line.kind == "speech":
            speaker_ko = "검사자" if line.speaker == VerbatimSpeaker.EXAMINER else "아동"
            out.append(f"[{line.timestamp_s:.1f}s] **{speaker_ko}**: {line.text}")
        else:
            out.append(f"[{line.timestamp_s:.1f}s] 🔹 {line.label_ko}")
    return "\n".join(out)
