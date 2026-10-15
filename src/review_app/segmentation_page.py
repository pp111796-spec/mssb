"""「1. 구간 확인」 화면 — AI가 나눈 이야기 구간을 확인·수정하고 확정한다 (파이프라인 4단계).

확정(segmentation_review.json)해야 cv·coding 단계를 실행할 수 있다. 확정 뒤 구간을 고치면
그 이야기만 다음 실행 때 다시 분석·채점된다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mssb_coder import storage
from mssb_coder.review_ops import (
    LOW_CONFIDENCE,
    format_examiner_segments,
    parse_examiner_segments,
    stem_choices,
    transcript_preview,
    validate_stem_timestamps,
)
from mssb_coder.schema import StemTimestamp
from review_app.common import FILE_FORMAT_ERRORS, show_file_error, story_stems, widget_key
from review_app.video_panel import render_clip

COLUMNS = {
    "stem": "이야기",
    "start": "시작(초)",
    "handoff": "검사자→아동 전환(초)",
    "end": "끝(초)",
    "confidence": "신뢰도",
    "flag": "확인 필요",
    "changed": "AI 원안과 다름",
    "cue": "근거 문구",
}


def _differs(stem: StemTimestamp, ai_draft: dict[str, StemTimestamp]) -> bool:
    original = ai_draft.get(stem.stem_name)
    if original is None:
        return True
    return (original.start_s, original.handoff_timestamp_s, original.end_s) != (
        stem.start_s, stem.handoff_timestamp_s, stem.end_s
    )


def _table(saved: list[StemTimestamp], ai_draft: dict[str, StemTimestamp], choices: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                COLUMNS["stem"]: choices.get(t.stem_name, t.stem_name),
                COLUMNS["start"]: t.start_s,
                COLUMNS["handoff"]: t.handoff_timestamp_s,
                COLUMNS["end"]: t.end_s,
                COLUMNS["confidence"]: t.confidence,
                COLUMNS["flag"]: "⚠" if t.confidence < LOW_CONFIDENCE else "",
                COLUMNS["changed"]: "✎" if ai_draft and _differs(t, ai_draft) else "",
                COLUMNS["cue"]: t.matched_cue,
            }
            for t in saved
        ],
        columns=list(COLUMNS.values()),
    )


def _status_banner(session_id: str) -> None:
    status = storage.segmentation_review_status(session_id)
    if status == "confirmed":
        st.success("구간이 확정되어 있습니다. 고친 뒤 다시 확정하면 바뀐 이야기만 다음 실행 때 다시 분석·채점됩니다.")
    elif status == "changed":
        st.warning("확정한 뒤 구간이 바뀌었습니다. 다시 확정해야 다음 단계를 실행할 수 있습니다.")
    else:
        st.info("아직 확정하지 않았습니다. 구간을 확인하고 맨 아래 '구간 확정'을 누르세요.")


def _examiner_key(session_id: str) -> str:
    return widget_key(session_id, "examiner_segments")


def _examiner_segments(session_id: str, saved: list[StemTimestamp]) -> dict[str, list[tuple[float, float]]]:
    """검사자 재개입 구간 — 위젯 상태가 아니라 별도 상태에 둬서 다른 이야기를 보다 와도 입력이 남게 한다."""
    key = _examiner_key(session_id)
    if key not in st.session_state:
        st.session_state[key] = {
            t.stem_name: [(seg.start_s, seg.end_s) for seg in t.additional_examiner_segments_s] for t in saved
        }
    return st.session_state[key]


def _reset_editors(session_id: str) -> None:
    prefix = widget_key(session_id, "")
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        if key != widget_key(session_id, "flash"):
            del st.session_state[key]


def _rows_from_editor(edited: pd.DataFrame, id_by_label: dict[str, str]) -> list[dict]:
    def text(value):
        return value if isinstance(value, str) else None

    return [
        {
            "stem_name": id_by_label.get(text(record[COLUMNS["stem"]]) or ""),
            "start_s": record[COLUMNS["start"]],
            "handoff_timestamp_s": record[COLUMNS["handoff"]],
            "end_s": record[COLUMNS["end"]],
            "confidence": record[COLUMNS["confidence"]],
            "matched_cue": text(record[COLUMNS["cue"]]),
        }
        for record in edited.to_dict("records")
    ]


def render(session_id: str) -> None:
    stems_cfg = story_stems()
    choices = stem_choices(stems_cfg)
    id_by_label = {label: stem_id for stem_id, label in choices.items()}

    st.title(f"구간 확인 — {session_id}")
    try:
        saved = storage.load_stem_timestamps(session_id)
    except FileNotFoundError:
        st.info(
            "구간분리 결과가 아직 없습니다. 먼저 다음 명령을 실행하세요.\n\n"
            f"`python scripts\\run_pipeline.py --session {session_id} --video <영상 경로> "
            "--child-age-months <개월> --child-gender <female|male> --stage segmentation`"
        )
        return
    except FILE_FORMAT_ERRORS as exc:
        show_file_error("구간(stem_timestamps.json)", exc)
        return
    try:
        ai_draft = {t.stem_name: t for t in storage.load_stem_timestamps(session_id, ai_draft=True)}
    except (FileNotFoundError, *FILE_FORMAT_ERRORS):
        ai_draft = {}

    flash = st.session_state.pop(widget_key(session_id, "flash"), None)
    if flash:
        st.success(flash)
    _status_banner(session_id)
    st.caption(
        "시각은 모두 녹화 시작부터의 초입니다. '확인 필요(⚠)'는 AI가 자신 없어 한 구간이니 먼저 보세요. "
        "AI가 놓친 이야기는 표 맨 아래 빈 줄에 추가하고, 잘못 잡은 줄은 왼쪽 칸을 체크한 뒤 삭제하세요."
    )

    edited = st.data_editor(
        _table(saved, ai_draft, choices),
        key=widget_key(session_id, "segment_editor"),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            COLUMNS["stem"]: st.column_config.SelectboxColumn(options=list(choices.values()), required=True),
            COLUMNS["start"]: st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
            COLUMNS["handoff"]: st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
            COLUMNS["end"]: st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
            COLUMNS["confidence"]: st.column_config.NumberColumn(disabled=True, format="%.2f"),
            COLUMNS["flag"]: st.column_config.TextColumn(disabled=True),
            COLUMNS["changed"]: st.column_config.TextColumn(disabled=True),
            COLUMNS["cue"]: st.column_config.TextColumn(disabled=True),
        },
    )
    rows = _rows_from_editor(edited, id_by_label)
    examiner_segments = _examiner_segments(session_id, saved)

    st.divider()
    _render_detail(session_id, rows, examiner_segments, choices)

    st.divider()
    timestamps, errors = validate_stem_timestamps(rows, examiner_segments, stems_cfg)
    parse_error = st.session_state.get(widget_key(session_id, "examiner_parse_error"))
    if parse_error:
        errors.append(parse_error)
    if errors:
        st.error("저장하기 전에 다음을 고치세요:\n\n" + "\n".join(f"- {e}" for e in errors))

    save_col, confirm_col = st.columns(2)
    with save_col:
        if st.button("💾 임시 저장", disabled=bool(errors), key=widget_key(session_id, "save_segments")):
            storage.save_stem_timestamps(session_id, timestamps)
            _reset_editors(session_id)
            st.session_state[widget_key(session_id, "flash")] = "구간을 저장했습니다. 확정은 아직 하지 않았습니다."
            st.rerun()
    with confirm_col:
        if st.button("✅ 구간 확정", type="primary", disabled=bool(errors), key=widget_key(session_id, "confirm_segments")):
            storage.save_stem_timestamps(session_id, timestamps)
            storage.confirm_segmentation(session_id)
            _reset_editors(session_id)
            st.session_state[widget_key(session_id, "flash")] = (
                "구간을 확정했습니다. 이제 다음 명령으로 분석·채점을 실행할 수 있습니다: "
                f"python scripts\\run_pipeline.py --session {session_id} --stage coding"
            )
            st.rerun()


def _render_detail(
    session_id: str,
    rows: list[dict],
    examiner_segments: dict[str, list[tuple[float, float]]],
    choices: dict[str, str],
) -> None:
    st.header("구간 자세히 보기")
    candidates = {}
    for row in rows:
        values = (row["start_s"], row["handoff_timestamp_s"], row["end_s"])
        if row["stem_name"] and all(isinstance(v, (int, float)) and not pd.isna(v) for v in values):
            candidates[choices[row["stem_name"]]] = row
    if not candidates:
        st.caption("시각이 모두 입력된 이야기가 없어 자세히 볼 수 없습니다.")
        return

    label = st.selectbox("이야기", list(candidates), key=widget_key(session_id, "detail_stem"))
    row = candidates[label]
    stem_name, start, handoff, end = row["stem_name"], row["start_s"], row["handoff_timestamp_s"], row["end_s"]

    video_col, text_col = st.columns([3, 2])
    with video_col:
        jump_options = {"시작": start, "검사자→아동 전환": handoff, "끝 10초 전": max(start, end - 10)}
        jump = st.radio("재생 위치", list(jump_options), horizontal=True, key=widget_key(session_id, "jump", stem_name))
        render_clip(session_id, start, end, jump_options[jump], key=f"seg_clip_{stem_name}")

    with text_col:
        st.subheader("전사문 미리보기")
        transcript = storage.load_transcript(session_id)
        if transcript is None:
            st.caption(
                "저장된 전사(transcript.json)가 없는 예전 세션이라 미리보기를 보여줄 수 없습니다. "
                "coding 단계를 한 번 실행하면 전사가 저장됩니다."
            )
        elif start < end:
            preview_stem = StemTimestamp(
                stem_name=stem_name, start_s=start, end_s=end, handoff_timestamp_s=min(max(handoff, start), end),
                confidence=1.0, matched_cue="", scored=False,
            )
            lines = transcript_preview(transcript, preview_stem)
            if not lines:
                st.caption("이 구간 근처에 전사된 발화가 없습니다.")
            for line in lines:
                text = f"[{line.timestamp_s:.1f}s] **{line.speaker}**: {line.text}"
                (st.caption if line.speaker == "구간 밖" else st.markdown)(text)

    st.subheader("아이 차례 도중 검사자가 다시 말한 구간")
    st.caption(
        "대부분 비워 둡니다. 예: 「뽀삐가 없어졌어요」에서 검사자가 재회 질문을 덧붙인 경우. "
        "'시작-끝' 형식으로 쓰고 여러 개는 세미콜론(;)으로 나눕니다. 예: 95.0-105.5; 120-128"
    )
    text_key = widget_key(session_id, "examiner_text", stem_name)
    error_key = widget_key(session_id, "examiner_parse_error")

    def _on_change() -> None:
        segments, error = parse_examiner_segments(st.session_state[text_key])
        if error:
            st.session_state[error_key] = f"{choices[stem_name]} 검사자 재개입 구간: {error}"
        else:
            st.session_state.pop(error_key, None)
            examiner_segments[stem_name] = segments

    st.text_input(
        "검사자 재개입 구간",
        value=format_examiner_segments(examiner_segments.get(stem_name, [])),
        key=text_key,
        on_change=_on_change,
        label_visibility="collapsed",
    )
