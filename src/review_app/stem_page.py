"""스템(이야기 1편) 상세 화면 — 검사자가 AI 채점을 확인·수정하고 근거 시점의 영상을 본다.

저장하면 <stem>_reviewed.json이 생기고, 보고서·PDF·내보내기·게시는 이 검토본을 우선 쓴다.
세션 종합(패턴 서술)은 AI 원본으로 만든 것이라 스템 수정이 자동 반영되지 않는다.
"""

from __future__ import annotations

import streamlit as st

from mssb_coder import storage
from mssb_coder.labels import (
    category_label,
    completion_status_label,
    evidence_source_label,
    item_label,
    response_mode_label,
    stem_label,
)
from mssb_coder.review_ops import StemEdits, apply_stem_edits, can_play_evidence
from mssb_coder.schema import CompletionStatus, ResponseMode, StemTimestamp, VerbatimSpeaker
from mssb_coder.verbatim import render_verbatim_markdown
from review_app.common import FILE_FORMAT_ERRORS, coding_system, plain, show_file_error, story_stems, widget_key
from review_app.video_panel import render_clip

TONE_HINT = (
    "이 문장은 학부모 보고서의 '이야기별 해석'에 그대로 실립니다. 하다체로 쓰고(~했다), 아이나 행동을 주어로 "
    "한 능동형으로 쓰세요. '부재로 코딩했다' 같은 채점자 용어나 표정 코드, '두 항목'처럼 개수로 부르는 말 대신 "
    "관찰된 행동을 직접 쓰세요."
)


def _current_window(session_id: str, stem_name: str) -> StemTimestamp | None:
    try:
        return next((t for t in storage.load_stem_timestamps(session_id) if t.stem_name == stem_name), None)
    except (FileNotFoundError, *FILE_FORMAT_ERRORS):
        return None


def _reset_stem_widgets(session_id: str, stem_name: str) -> None:
    prefix = widget_key(session_id, "stem", stem_name, "")
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[key]


def render_stem_detail(session_id: str, stem_name: str) -> None:
    cs, stems = coding_system(), story_stems()
    k = lambda *parts: widget_key(session_id, "stem", stem_name, *parts)  # noqa: E731

    reviewed = storage.stem_reviewed_exists(session_id, stem_name)
    try:
        result = storage.load_stem_coding(session_id, stem_name, reviewed=reviewed)
    except FileNotFoundError:
        st.warning("이 스템의 AI 채점 결과를 찾을 수 없습니다.")
        return
    except FILE_FORMAT_ERRORS as exc:
        show_file_error("스템 채점 결과", exc)
        return

    st.subheader(f"스템 상세 — {stem_label(stems, stem_name)}")
    flash = st.session_state.pop(k("flash"), None)
    if flash:
        st.success(flash)
    st.caption("검토본(사람이 고친 결과)을 보고 있습니다." if reviewed else "AI 원본을 보고 있습니다. 고친 뒤 저장하면 검토본이 됩니다.")
    st.info(result.disclaimer)

    window = _current_window(session_id, stem_name)
    play_key = k("play_at")
    if window is not None:
        with st.expander("🎬 이 이야기 영상", expanded=play_key in st.session_state):
            render_clip(session_id, window.start_s, window.end_s, st.session_state.get(play_key), key=f"stem_clip_{stem_name}")
    if result.timestamp_basis != "session_absolute":
        st.caption("이 결과는 근거 시각 기준이 기록되기 전에 만들어져 근거별 영상 재생을 할 수 없습니다. 다시 채점하면 쓸 수 있습니다.")

    _render_verbatim(session_id, stem_name)

    st.markdown("**실시 기록 (부록 B)** — 사람이 기록하는 값입니다. AI 채점 때는 기본값이 들어 있습니다.")
    b1, b2, b3 = st.columns(3)
    statuses = list(CompletionStatus)
    modes = list(ResponseMode)
    with b1:
        completion = st.selectbox(
            "완료상태", statuses, index=statuses.index(result.completion_status),
            format_func=completion_status_label, key=k("completion"),
        )
    with b2:
        mode = st.selectbox(
            "자발/프롬프트후", modes, index=modes.index(result.response_mode),
            format_func=response_mode_label, key=k("mode"),
        )
    with b3:
        prompt_count = st.number_input(
            "쟁점프롬프트 횟수", min_value=0, max_value=2, value=result.issue_prompt_count, step=1, key=k("prompts")
        )

    present: dict[tuple[str, str], bool] = {}
    rationale: dict[str, str] = {}
    excluded: set[tuple[str, str, int]] = set()
    for category_id, category_code in result.codes.items():
        label = category_label(cs, category_id)
        with st.expander(f"{label} — 존재 항목 {category_code.present_count}개", expanded=False):
            rationale[category_id] = st.text_area(
                "판단 이유", value=plain(category_code.rationale), key=k("rationale", category_id), help=TONE_HINT
            )
            for item_id, item in category_code.items.items():
                present[(category_id, item_id)] = st.checkbox(
                    item_label(cs, category_id, item_id), value=item.present, key=k("present", category_id, item_id)
                )
                for index, ev in enumerate(item.evidence):
                    text_col, play_col, drop_col = st.columns([6, 1, 1])
                    with text_col:
                        st.caption(f"[{ev.timestamp_s:.1f}s] ({evidence_source_label(ev.source)}) {plain(ev.note)}")
                    with play_col:
                        if st.button(
                            "▶", key=k("play", category_id, item_id, index), help="이 시점부터 영상 재생",
                            disabled=not can_play_evidence(result, window, ev.timestamp_s),
                        ):
                            st.session_state[play_key] = ev.timestamp_s
                            st.rerun()
                    with drop_col:
                        if st.checkbox("제외", key=k("drop", category_id, item_id, index), help="이 근거를 검토본에서 뺍니다"):
                            excluded.add((category_id, item_id, index))

    st.caption("스템을 고쳐도 위쪽 세션 종합 서술은 자동으로 바뀌지 않습니다. 필요하면 종합 서술도 직접 고치세요.")
    save_col, revert_col = st.columns(2)
    with save_col:
        if st.button("💾 이 이야기 검토본 저장", type="primary", key=k("save")):
            edits = StemEdits(
                completion_status=completion, response_mode=mode, issue_prompt_count=int(prompt_count),
                present=present, rationale=rationale, excluded_evidence=excluded,
            )
            storage.save_stem_coding(session_id, stem_name, apply_stem_edits(result, edits), reviewed=True)
            _reset_stem_widgets(session_id, stem_name)  # 제외한 근거가 빠져 순번이 바뀌므로 위젯을 새로 만든다
            st.session_state[k("flash")] = "검토본을 저장했습니다. 보고서·내보내기·게시에 이 내용이 쓰입니다."
            st.rerun()
    with revert_col:
        if reviewed and st.button("↩ AI 원본으로 되돌리기", key=k("revert"), help="검토본은 지우지 않고 백업 파일로 옮깁니다"):
            backup = storage.revert_stem_review(session_id, stem_name)
            _reset_stem_widgets(session_id, stem_name)
            st.session_state[k("flash")] = f"AI 원본으로 되돌렸습니다. 이전 검토본은 {backup.name}에 남아 있습니다."
            st.rerun()


def _render_verbatim(session_id: str, stem_name: str) -> None:
    with st.expander("축어록 (언어+행동분석)", expanded=False):
        try:
            verbatim = storage.load_stem_verbatim(session_id, stem_name)
        except FileNotFoundError:
            st.caption("축어록 데이터 없음 — 이 세션은 재분석이 필요합니다.")
            return
        except FILE_FORMAT_ERRORS as exc:
            show_file_error("축어록", exc)
            return
        for line in verbatim.lines:
            if line.kind == "speech":
                speaker_ko = "검사자" if line.speaker == VerbatimSpeaker.EXAMINER else "아동"
                st.write(f"[{line.timestamp_s:.1f}s] **{speaker_ko}**: {line.text or ''}")
            else:
                st.caption(f"[{line.timestamp_s:.1f}s] 🔹 {line.label_ko or ''}")
        st.download_button(
            "📄 축어록 다운로드 (.md)",
            data=render_verbatim_markdown(verbatim),
            file_name=f"{session_id}_{stem_name}_verbatim.md",
            mime="text/markdown",
            key=widget_key(session_id, "stem", stem_name, "verbatim_dl"),
        )
