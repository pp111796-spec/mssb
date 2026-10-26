"""Streamlit 검토 앱 (로컬 전용) — 세션 선택과 화면 전환만 담당한다.

화면:
  「1. 구간 확인」 segmentation_page — AI가 나눈 이야기 구간을 확인·수정하고 확정 (파이프라인 4단계)
  「2. 결과 검토」 synthesis_page — 세션 종합 초안 검토, 스템 상세 드릴다운, 보고서·게시

실행: streamlit run src/review_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from dotenv import load_dotenv

from mssb_coder import storage
from mssb_coder.review_ops import session_progress
from review_app import segmentation_page, synthesis_page
from review_app.common import FILE_FORMAT_ERRORS, coding_system, story_stems

load_dotenv()

st.set_page_config(page_title="MSSB AI 코딩 검토", layout="wide")

PAGES = ("1. 구간 확인", "2. 결과 검토")

# 스크롤을 줄이기 위한 촘촘한 레이아웃 — 여백·제목 크기·입력칸 간격을 줄인다
COMPACT_CSS = """
<style>
.block-container { padding: 1rem 1.5rem 0.5rem 1.5rem; max-width: 100%; }
h1 { font-size: 1.4rem !important; line-height: 1.5 !important; padding: 0 !important; }
h2 { font-size: 1.1rem !important; line-height: 1.5 !important; padding: 0 !important; }
h3 { font-size: 0.98rem !important; line-height: 1.5 !important; padding: 0 !important; }
[data-testid="stHeadingWithActionElements"] { margin-bottom: 0.35rem; }
[data-testid="stVerticalBlock"] { gap: 0.45rem; }
.stTextArea textarea { font-size: 0.86rem; line-height: 1.4; }
.stTextArea label p, .stCaption, [data-testid="stCaptionContainer"] { font-size: 0.78rem; }
[data-testid="stAlert"] { padding: 0.4rem 0.7rem; font-size: 0.82rem; }
.stButton button, .stDownloadButton button { padding: 0.15rem 0.6rem; min-height: 0; font-size: 0.8rem; }
hr { margin: 0.5rem 0; }
</style>
"""


def list_sessions() -> list[str] | None:
    """세션 폴더 이름 목록. MSSB_DATA_DIR이 설정되지 않았으면 None."""
    try:
        sessions_root = storage.get_data_dir() / "sessions"
    except storage.DataDirNotConfiguredError:
        return None
    if not sessions_root.exists():
        return []
    return sorted(p.name for p in sessions_root.iterdir() if p.is_dir())


def main() -> None:
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)
    try:
        coding_system()
        story_stems()
    except FileNotFoundError as exc:
        st.error(f"설정 파일을 찾을 수 없습니다. 저장소의 config 폴더를 확인하세요.\n\n{exc}")
        st.stop()

    st.sidebar.title("세션 목록")
    sessions = list_sessions()
    if sessions is None:
        st.title("MSSB AI 코딩 검토")
        st.error(".env에 MSSB_DATA_DIR(세션 데이터 폴더)이 설정되지 않았습니다. 예: MSSB_DATA_DIR=C:\\mssb_data")
        return
    if not sessions:
        st.title("MSSB AI 코딩 검토")
        st.write("아직 처리된 세션이 없습니다 — `scripts/run_pipeline.py --stage segmentation`으로 세션을 먼저 만드세요.")
        return

    selected = st.sidebar.selectbox("세션 선택", sessions)

    has_synthesis = False
    try:
        progress = session_progress(selected)
        has_synthesis = progress.has_synthesis
        st.sidebar.caption(f"상태: {progress.label}")
    except FILE_FORMAT_ERRORS:
        st.sidebar.caption("상태: 구간 파일을 읽지 못함")

    # 종합까지 끝난 세션은 결과 검토부터, 아니면 구간 확인부터 연다 (세션별로 사용자가 고른 화면은 유지)
    page = st.sidebar.radio("화면", PAGES, index=1 if has_synthesis else 0, key=f"page:{selected}")
    if page == PAGES[0]:
        segmentation_page.render(selected)
    else:
        synthesis_page.render(selected)


if __name__ == "__main__":
    main()
