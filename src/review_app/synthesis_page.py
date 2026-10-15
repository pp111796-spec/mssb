"""「2. 결과 검토」 화면 — 세션 종합 초안이 1차 화면, 근거 클릭 시 스템 상세로 드릴다운."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mssb_coder import git_publish, storage
from mssb_coder.labels import category_label, stem_label
from mssb_coder.report import build_session_report_markdown
from review_app.common import FILE_FORMAT_ERRORS, coding_system, plain, show_file_error, story_stems, widget_key
from review_app.stem_page import render_stem_detail

REPO_ROOT = Path(__file__).resolve().parents[2]


def publish_to_shared_site(session_id: str) -> tuple[bool, str]:
    """results/<session_id>.json을 만들고 그 파일만 커밋·푸시 — 사업단 공유 사이트가 읽는 곳.

    영상은 다루지 않는다 — storage.publish_session_bundle_to_repo가 만드는 처리된 JSON만 올린다.
    """
    try:
        path = storage.publish_session_bundle_to_repo(session_id)
    except (OSError, *FILE_FORMAT_ERRORS) as exc:
        return False, f"게시할 결과 파일을 만들지 못했습니다.\n{exc}"
    outcome = git_publish.publish_file(REPO_ROOT, path, f"세션 결과 게시: {session_id}")
    return outcome.ok, outcome.message


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
    st.info(synthesis.disclaimer)
    st.caption(
        f"정상 실시된 스템: {synthesis.stems_administered_count}개 "
        f"(포함: {', '.join(stem_label(stems, s) for s in synthesis.stems_included)})"
    )
    if synthesis.stems_missing_or_partial:
        st.warning(
            "미실시/부분실시: "
            f"{', '.join(stem_label(stems, s) for s in synthesis.stems_missing_or_partial)}"
        )

    report_markdown = build_session_report_markdown(
        synthesis, stem_results=stem_results, coding_system=cs, story_stems=stems
    )
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "📄 분석 결과 보고서 다운로드 (.md)",
            data=report_markdown,
            file_name=f"{session_id}_mssb_report.md",
            mime="text/markdown",
            key=widget_key(session_id, "report_md"),
        )
    with dl_col2:
        try:
            bundle_json = json.dumps(storage.build_session_export_bundle(session_id), ensure_ascii=False, indent=2)
        except FILE_FORMAT_ERRORS as exc:
            show_file_error("내보내기용 결과", exc)
        else:
            st.download_button(
                "💾 결과 파일만 내보내기 (.json)",
                data=bundle_json,
                file_name=f"{session_id}_mssb_bundle.json",
                mime="application/json",
                help="영상은 포함되지 않습니다 — 이미 처리된 결과(JSON)만 담습니다.",
                key=widget_key(session_id, "bundle_json"),
            )
    with dl_col3:
        if st.button(
            "🚀 사업단 공유 사이트에 게시",
            help="results/ 폴더에 이 세션 결과 파일만 커밋·푸시합니다. 공유 사이트에 자동 반영됩니다.",
            key=widget_key(session_id, "publish"),
        ):
            with st.spinner("게시 중..."):
                ok, message = publish_to_shared_site(session_id)
            (st.success if ok else st.error)(message)

    st.header("검사 중 행동관찰")
    edited_observation = st.text_area(
        "행동관찰 (수정 가능)", value=plain(synthesis.behavioral_observation),
        key=widget_key(session_id, "behavioral_observation"),
    )

    edited_patterns = []
    st.header("반복되는 패턴")
    for i, pattern in enumerate(synthesis.recurring_patterns):
        label = category_label(cs, pattern.category)
        st.subheader(f"{label} — {len(pattern.observed_in_stems)}/{pattern.stems_administered_count_for_pattern}개 이야기")
        edited_desc = st.text_area(
            "패턴 서술 (수정 가능)", value=plain(pattern.pattern_description), key=widget_key(session_id, "pattern", i)
        )
        edited_patterns.append(pattern.model_copy(update={"pattern_description": edited_desc}))

        cols = st.columns(len(pattern.evidence) or 1)
        for j, (col, ev) in enumerate(zip(cols, pattern.evidence)):
            with col:
                if st.button(
                    f"{stem_label(stems, ev.stem_name)} · {ev.timestamp_s:.1f}s",
                    key=widget_key(session_id, "evidence", i, j),
                    help=plain(ev.note) or None,
                ) and ev.stem_name in {r.stem_name for r in stem_results}:
                    st.session_state[widget_key(session_id, "stem_select")] = ev.stem_name

    if synthesis.single_story_notable_signals:
        st.header("단발성 신호 (패턴 아님)")
        for signal in synthesis.single_story_notable_signals:
            label = category_label(cs, signal.category)
            st.write(f"**{stem_label(stems, signal.stem_name)}** ({label}): {plain(signal.note)}")

    st.header("강점과 지원이 필요한 영역")
    edited_strengths = st.text_area(
        "강점 (수정 가능)", value=plain(synthesis.strengths), key=widget_key(session_id, "strengths")
    )
    edited_support_areas = st.text_area(
        "지원이 필요한 영역 (수정 가능)", value=plain(synthesis.support_areas), key=widget_key(session_id, "support_areas")
    )

    st.header("제언")
    st.warning("제언은 AI가 작성한 초안입니다. 진단·위험 판단이 섞이지 않았는지 반드시 확인하고 수정하세요.")
    edited_recommendations_text = st.text_area(
        "제언 (한 줄에 하나씩, 수정 가능)",
        value="\n".join(plain(r) for r in synthesis.recommendations),
        key=widget_key(session_id, "recommendations"),
    )

    if st.button("종합 초안 저장 (session_summary_reviewed.json)", key=widget_key(session_id, "save_synthesis")):
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
