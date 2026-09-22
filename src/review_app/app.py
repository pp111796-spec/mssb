"""Streamlit 검토 UI — 세션 종합 초안이 1차 화면, 스템별 화면은 드릴다운.

계획 "사람 검토(Human-in-the-loop) 설계" 섹션을 그대로 구현. 영상 재생 위젯은
아직 없다 — 파이프라인이 세션별 원본 영상 경로를 저장하는 부분이 마일스톤 A에서
채워져야 붙일 수 있어서, 지금은 텍스트·CV 지표·근거 타임스탬프까지만 보여준다.

실행: streamlit run src/review_app/app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from dotenv import load_dotenv

from mssb_coder import storage
from mssb_coder.report import build_session_report_markdown
from mssb_coder.schema import SessionSynthesis

load_dotenv()

st.set_page_config(page_title="MSSB AI 코딩 검토", layout="wide")


REPO_ROOT = Path(__file__).resolve().parents[2]


def publish_to_shared_site(session_id: str) -> tuple[bool, str]:
    """results/<session_id>.json을 만들고 git add/commit/push — 사업단 공유 사이트가 읽는 곳.

    영상은 이 함수가 다루는 대상이 아니다 — storage.publish_session_bundle_to_repo가 만드는
    이미 처리된 JSON 결과만 커밋한다. 저장소가 private이라도 이 파일은 git 히스토리에
    영구히 남는다는 점을 화면에 함께 안내한다(호출하는 쪽에서 표시).
    """
    path = storage.publish_session_bundle_to_repo(session_id)
    rel_path = path.relative_to(REPO_ROOT)
    try:
        subprocess.run(
            ["git", "add", str(rel_path)], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
        commit = subprocess.run(
            ["git", "commit", "-m", f"세션 결과 게시: {session_id}"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
            return False, f"커밋 실패:\n{commit.stderr or commit.stdout}"
        subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return True, "게시 완료 — 1~2분 뒤 공유 사이트에 반영됩니다."
    except subprocess.CalledProcessError as exc:
        return False, f"게시 실패:\n{exc.stderr or exc.stdout}"


def list_sessions() -> list[str]:
    try:
        sessions_root = storage.get_data_dir() / "sessions"
    except storage.DataDirNotConfiguredError:
        return []
    if not sessions_root.exists():
        return []
    return sorted(p.name for p in sessions_root.iterdir() if p.is_dir())


def render_stem_detail(session_id: str, stem_name: str) -> None:
    st.subheader(f"스템 상세 — {stem_name}")
    try:
        reviewed_exists = (storage.session_dir(session_id) / "coding" / f"{stem_name}_reviewed.json").exists()
        result = storage.load_stem_coding(session_id, stem_name, reviewed=reviewed_exists)
    except FileNotFoundError:
        st.warning("이 스템의 AI 채점 결과를 찾을 수 없습니다.")
        return

    st.caption(
        f"완료상태: {result.completion_status.value} · 자발/프롬프트후: {result.response_mode.value} "
        f"· 쟁점프롬프트 횟수: {result.issue_prompt_count}"
    )
    st.info(result.disclaimer)

    for category_id, category_code in result.codes.items():
        with st.expander(f"{category_id} — 존재 항목 {category_code.present_count}개", expanded=False):
            st.write(category_code.rationale)
            for item_id, item in category_code.items.items():
                mark = "✅" if item.present else "—"
                st.write(f"{mark} **{item_id}**")
                for ev in item.evidence:
                    st.caption(f"  [{ev.timestamp_s:.1f}s] ({ev.source.value}) {ev.note}")


def render_session_synthesis(session_id: str) -> None:
    reviewed_path = storage.session_dir(session_id) / "session_summary_reviewed.json"
    synthesis = storage.load_session_synthesis(session_id, reviewed=reviewed_path.exists())

    st.title(f"세션 종합 패턴 초안 — {session_id}")
    st.info(synthesis.disclaimer)
    st.caption(
        f"정상 실시된 스템: {synthesis.stems_administered_count}개 "
        f"(포함: {', '.join(synthesis.stems_included)})"
    )
    if synthesis.stems_missing_or_partial:
        st.warning(f"미실시/부분실시: {', '.join(synthesis.stems_missing_or_partial)}")

    try:
        stem_results = storage.load_all_stem_codings(session_id)
    except FileNotFoundError:
        stem_results = []
    report_markdown = build_session_report_markdown(synthesis, stem_results=stem_results)
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "📄 분석 결과 보고서 다운로드 (.md)",
            data=report_markdown,
            file_name=f"{session_id}_mssb_report.md",
            mime="text/markdown",
        )
    with dl_col2:
        bundle = storage.build_session_export_bundle(session_id)
        st.download_button(
            "💾 결과 파일만 내보내기 (.json)",
            data=json.dumps(bundle, ensure_ascii=False, indent=2),
            file_name=f"{session_id}_mssb_bundle.json",
            mime="application/json",
            help="영상은 포함되지 않습니다 — 이미 처리된 결과(JSON)만 담습니다.",
        )
    with dl_col3:
        if st.button("🚀 사업단 공유 사이트에 게시", help="results/ 폴더에 저장 후 git push — 공유 사이트에 자동 반영됩니다."):
            with st.spinner("게시 중..."):
                ok, message = publish_to_shared_site(session_id)
            (st.success if ok else st.error)(message)

    edited_patterns = []
    st.header("반복되는 패턴")
    for i, pattern in enumerate(synthesis.recurring_patterns):
        st.subheader(f"{pattern.category} — {len(pattern.observed_in_stems)}/{pattern.stems_administered_count_for_pattern}개 이야기")
        edited_desc = st.text_area(
            "패턴 서술 (수정 가능)", value=pattern.pattern_description, key=f"pattern_{i}"
        )
        edited_patterns.append(pattern.model_copy(update={"pattern_description": edited_desc}))

        cols = st.columns(len(pattern.evidence) or 1)
        for col, ev in zip(cols, pattern.evidence):
            with col:
                if st.button(f"{ev.stem_name} · {ev.timestamp_s:.1f}s", key=f"ev_{i}_{ev.stem_name}_{ev.timestamp_s}"):
                    st.session_state["drilldown_stem"] = ev.stem_name

    if synthesis.single_story_notable_signals:
        st.header("단발성 신호 (패턴 아님)")
        for signal in synthesis.single_story_notable_signals:
            st.write(f"**{signal.stem_name}** ({signal.category}): {signal.note}")

    if st.button("종합 초안 저장 (session_summary_reviewed.json)"):
        updated = synthesis.model_copy(update={"recurring_patterns": edited_patterns})
        storage.save_session_synthesis(session_id, updated, reviewed=True)
        st.success("저장했습니다.")

    st.divider()
    drilldown_stem = st.session_state.get("drilldown_stem")
    if drilldown_stem:
        render_stem_detail(session_id, drilldown_stem)
    else:
        st.caption("위 근거 버튼을 클릭하면 해당 스템의 상세(AI 채점 근거)가 여기 표시됩니다.")


def main() -> None:
    st.sidebar.title("세션 목록")
    sessions = list_sessions()
    if not sessions:
        st.sidebar.warning("MSSB_DATA_DIR에 세션이 없습니다. .env의 MSSB_DATA_DIR을 확인하세요.")
        st.title("MSSB AI 코딩 검토")
        st.write("아직 처리된 세션이 없습니다 — `scripts/run_pipeline.py`로 세션을 먼저 만드세요.")
        return

    selected = st.sidebar.selectbox("세션 선택", sessions)
    if selected != st.session_state.get("selected_session"):
        st.session_state["selected_session"] = selected
        st.session_state.pop("drilldown_stem", None)

    try:
        render_session_synthesis(selected)
    except FileNotFoundError:
        st.title(f"{selected}")
        st.warning(
            "session_summary_ai_draft.json이 아직 없습니다 — "
            "run_pipeline.py의 synthesis 단계까지 먼저 실행하세요."
        )


if __name__ == "__main__":
    main()
