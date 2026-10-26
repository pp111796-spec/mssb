"""「2. 결과 검토」 화면 — 세션 종합 초안이 1차 화면, 근거 클릭 시 스템 상세로 드릴다운."""

from __future__ import annotations

import streamlit as st

from mssb_coder import storage
from mssb_coder.labels import category_label, stem_label
from mssb_coder.report import build_session_report_markdown
from review_app.common import (
    FILE_FORMAT_ERRORS,
    coding_system,
    plain,
    report_pdf_bytes,
    show_file_error,
    story_stems,
    widget_key,
)
from review_app.stem_page import render_stem_detail


def fit_height(text: str, chars_per_line: int) -> int:
    """글이 스크롤 없이 다 보이도록 입력칸 높이(px)를 글자 수로 어림잡는다. 한글은 폭이 넓어 줄당 글자 수를 작게 잡는다."""
    lines = sum(max(1, -(-len(line) // chars_per_line)) for line in (text or "").split("\n"))
    return max(68, min(lines * 22 + 20, 600))


def render(session_id: str) -> None:
    cs, stems = coding_system(), story_stems()
    reviewed = (storage.session_dir(session_id) / "session_summary_reviewed.json").exists()
    try:
        synthesis = storage.load_session_synthesis(session_id, reviewed=reviewed)
    except FileNotFoundError:
        st.title(session_id)
        st.warning("세션 종합 결과가 아직 없습니다. 파이프라인의 coding·synthesis 단계를 먼저 실행하세요.")
        return
    except FILE_FORMAT_ERRORS as exc:
        st.title(session_id)
        show_file_error("세션 종합", exc)
        return

    try:
        stem_results = storage.load_effective_stem_codings(session_id)
    except FILE_FORMAT_ERRORS as exc:
        show_file_error("스템 채점 결과", exc)
        stem_results = []

    st.title(f"세션 종합 패턴 초안 — {session_id}")
    included = ", ".join(stem_label(stems, s) for s in synthesis.stems_included)
    summary_line = f"정상 실시된 스템 {synthesis.stems_administered_count}개 (포함: {included})"
    if synthesis.stems_missing_or_partial:
        missing = ", ".join(stem_label(stems, s) for s in synthesis.stems_missing_or_partial)
        summary_line += f" · 미실시/부분실시: {missing}"
    st.caption(summary_line)
    with st.expander("AI 초안 안내문"):
        st.write(synthesis.disclaimer)

    report_markdown = build_session_report_markdown(
        synthesis, stem_results=stem_results, coding_system=cs, story_stems=stems
    )
    if not reviewed:
        st.caption("아직 종합 검토본을 저장하지 않아, 보고서의 '해석상 유의점'에 '검사자가 아직 검토하지 않았다'고 적힙니다.")

    st.download_button(
        "📑 보고서 PDF 다운로드",
        data=report_pdf_bytes(report_markdown, f"MSSB 세션 결과 — {session_id}"),
        file_name=f"{session_id}_mssb_report.pdf",
        mime="application/pdf",
        type="primary",
        key=widget_key(session_id, "report_pdf"),
    )

    # 본문 3단: 행동관찰·단발성 | 반복 패턴 | 강점·지원·제언 — 한 화면에 모두 보이게
    left, middle, right = st.columns([1, 1.3, 1])

    with left:
        st.header("검사 중 행동관찰")
        observation_text = plain(synthesis.behavioral_observation)
        edited_observation = st.text_area(
            "행동관찰 (수정 가능)", value=observation_text,
            height=fit_height(observation_text, chars_per_line=34),
            key=widget_key(session_id, "behavioral_observation"),
        )
        if synthesis.single_story_notable_signals:
            st.header("단발성 신호 (패턴 아님)")
            for signal in synthesis.single_story_notable_signals:
                label = category_label(cs, signal.category)
                st.write(f"**{stem_label(stems, signal.stem_name)}** ({label}): {plain(signal.note)}")

    edited_patterns = []
    with middle:
        st.header("반복되는 패턴")
        for i, pattern in enumerate(synthesis.recurring_patterns):
            label = category_label(cs, pattern.category)
            st.subheader(f"{label} — {len(pattern.observed_in_stems)}/{pattern.stems_administered_count_for_pattern}개 이야기")
            description = plain(pattern.pattern_description)
            edited_desc = st.text_area(
                "패턴 서술 (수정 가능)", value=description, height=fit_height(description, chars_per_line=44),
                key=widget_key(session_id, "pattern", i), label_visibility="collapsed",
            )
            edited_patterns.append(pattern.model_copy(update={"pattern_description": edited_desc}))

            # 근거 버튼은 한 줄에 3개씩
            for row_start in range(0, len(pattern.evidence), 3):
                row = pattern.evidence[row_start:row_start + 3]
                for j, (col, ev) in enumerate(zip(st.columns(3), row), start=row_start):
                    with col:
                        if st.button(
                            f"{stem_label(stems, ev.stem_name)} · {ev.timestamp_s:.1f}s",
                            key=widget_key(session_id, "evidence", i, j),
                            help=plain(ev.note) or None,
                        ) and ev.stem_name in {r.stem_name for r in stem_results}:
                            st.session_state[widget_key(session_id, "stem_select")] = ev.stem_name

    with right:
        st.header("강점과 지원이 필요한 영역")
        strengths_text = plain(synthesis.strengths)
        edited_strengths = st.text_area(
            "강점 (수정 가능)", value=strengths_text, height=fit_height(strengths_text, chars_per_line=34),
            key=widget_key(session_id, "strengths"),
        )
        support_text = plain(synthesis.support_areas)
        edited_support_areas = st.text_area(
            "지원이 필요한 영역 (수정 가능)", value=support_text, height=fit_height(support_text, chars_per_line=34),
            key=widget_key(session_id, "support_areas"),
        )

        st.header("제언")
        st.caption("AI가 작성한 초안입니다. 진단·위험 판단이 섞이지 않았는지 반드시 확인하고 수정하세요.")
        recommendations_text = "\n".join(plain(r) for r in synthesis.recommendations)
        edited_recommendations_text = st.text_area(
            "제언 (한 줄에 하나씩, 수정 가능)", value=recommendations_text,
            height=fit_height(recommendations_text, chars_per_line=34),
            key=widget_key(session_id, "recommendations"),
        )

        if st.button(
            "종합 초안 저장", type="primary", key=widget_key(session_id, "save_synthesis"),
            help="session_summary_reviewed.json으로 저장합니다.",
        ):
            updated = synthesis.model_copy(
                update={
                    "recurring_patterns": edited_patterns,
                    "behavioral_observation": edited_observation,
                    "strengths": edited_strengths,
                    "support_areas": edited_support_areas,
                    "recommendations": [
                        line.strip() for line in edited_recommendations_text.splitlines() if line.strip()
                    ],
                }
            )
            storage.save_session_synthesis(session_id, updated, reviewed=True)
            st.success("저장했습니다.")

    st.divider()
    st.header("이야기별 상세 검토")
    names = [r.stem_name for r in stem_results]
    if not names:
        st.caption("스템 채점 결과가 없습니다.")
        return
    st.caption("위 근거 버튼을 누르거나 아래에서 이야기를 고르면 채점 근거를 확인하고 고칠 수 있습니다.")
    selected = st.selectbox(
        "이야기",
        names,
        format_func=lambda n: stem_label(stems, n) + (" (검토본)" if storage.stem_reviewed_exists(session_id, n) else ""),
        key=widget_key(session_id, "stem_select"),
    )
    render_stem_detail(session_id, selected)
