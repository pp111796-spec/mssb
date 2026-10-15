"""스템(이야기 1편) 상세 화면 — 검사자용 체크리스트·근거·축어록."""

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
from mssb_coder.schema import VerbatimSpeaker
from mssb_coder.verbatim import render_verbatim_markdown
from review_app.common import FILE_FORMAT_ERRORS, coding_system, plain, show_file_error, story_stems, widget_key


def render_stem_detail(session_id: str, stem_name: str) -> None:
    cs, stems = coding_system(), story_stems()
    st.subheader(f"스템 상세 — {stem_label(stems, stem_name)}")
    try:
        result = storage.load_stem_coding(
            session_id, stem_name, reviewed=storage.stem_reviewed_exists(session_id, stem_name)
        )
    except FileNotFoundError:
        st.warning("이 스템의 AI 채점 결과를 찾을 수 없습니다.")
        return
    except FILE_FORMAT_ERRORS as exc:
        show_file_error("스템 채점 결과", exc)
        return

    st.caption(
        f"완료상태: {completion_status_label(result.completion_status)} · "
        f"자발/프롬프트후: {response_mode_label(result.response_mode)} "
        f"· 쟁점프롬프트 횟수: {result.issue_prompt_count}"
    )
    st.info(result.disclaimer)

    with st.expander("축어록 (언어+행동분석)", expanded=False):
        try:
            verbatim = storage.load_stem_verbatim(session_id, stem_name)
        except FileNotFoundError:
            st.caption("축어록 데이터 없음 — 이 세션은 재분석이 필요합니다.")
        except FILE_FORMAT_ERRORS as exc:
            show_file_error("축어록", exc)
        else:
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
                key=widget_key(session_id, stem_name, "verbatim_dl"),
            )

    for category_id, category_code in result.codes.items():
        label = category_label(cs, category_id)
        with st.expander(f"{label} — 존재 항목 {category_code.present_count}개", expanded=False):
            st.write(plain(category_code.rationale))
            for item_id, item in category_code.items.items():
                mark = "✅" if item.present else "—"
                st.write(f"{mark} **{item_label(cs, category_id, item_id)}**")
                for ev in item.evidence:
                    st.caption(f"  [{ev.timestamp_s:.1f}s] ({evidence_source_label(ev.source)}) {plain(ev.note)}")
