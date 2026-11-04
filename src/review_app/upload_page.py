"""「새 세션 만들기」 화면 — 영상 업로드 + 세션 정보 입력 → 구간분리(파이프라인 1~3단계)까지 자동 실행.

이전에는 이 단계를 터미널에서 scripts\\run_pipeline.py로만 돌릴 수 있었다. 이 화면은 그 CLI 호출을
대신해, 검사자가 터미널 없이 영상만 올리면 구간분리까지 끝난 세션이 바로 생기게 한다.

mssb_coder.pipeline은 CV 라이브러리(Py-Feat/ultralytics/opensmile)까지 모듈 최상단에서 import하므로,
앱이 켜질 때마다 그 무게를 짊어지지 않도록 실제로 버튼을 눌렀을 때만 지연 import한다
(review_app.common.report_pdf_bytes의 지연 import와 같은 이유).
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from mssb_coder import storage, video_vault
from mssb_coder.schema import SessionMetadata, SubjectType
from review_app.common import SESSION_SELECT_KEY, list_sessions, log_event

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "m4v"]


def _validate(session_id: str, existing: set[str]) -> str | None:
    if not session_id:
        return None  # 아직 입력 전 — 버튼을 비활성화할 뿐, 오류로 보여주지 않음
    if not _SESSION_ID_RE.match(session_id):
        return "세션 ID는 영문·숫자·-·_ 만 쓸 수 있습니다 (아동 실명을 쓰지 마세요)."
    if session_id in existing:
        return f"이미 있는 세션 ID입니다: {session_id}"
    return None


def render() -> None:
    st.title("새 세션 만들기")
    st.caption(
        "영상을 올리고 세션 정보를 입력하면 오디오 추출 → 전사 → Claude 구간분리까지 자동으로 진행됩니다. "
        "구간분리 결과는 다음 화면(「1. 구간 확인」)에서 사람이 확인·수정합니다."
    )

    session_id = st.text_input(
        "세션 ID", placeholder="예: s2026_11_01", help="영문/숫자/-/_ 만 사용하세요. 아동 실명을 쓰지 마세요."
    )
    uploaded = st.file_uploader("검사 영상 업로드", type=_VIDEO_TYPES)

    col1, col2 = st.columns(2)
    with col1:
        child_age_months = st.number_input("아동 개월 수", min_value=1, max_value=240, value=54, step=1)
        child_gender = st.selectbox(
            "아동 성별 (여행 떠나기 화자 판별용)",
            ["female", "male"],
            format_func=lambda v: "여아" if v == "female" else "남아",
        )
    with col2:
        subject_type = st.selectbox(
            "대상 유형",
            ["child", "adult"],
            format_func=lambda v: "실제 아동" if v == "child" else "성인 대역 (배선 검증용)",
        )
        st.caption("실제 아동 영상이면 '실제 아동'을 고르세요 — 표정·자세 분석 임계값 프로파일이 달라집니다.")

    existing = {s for s in (list_sessions() or [])}
    error = _validate(session_id, existing)
    if error:
        st.error(error)

    ready = bool(session_id) and uploaded is not None and not error
    st.warning("⚠️ 아래 버튼을 누르면 Claude API를 실제로 1회 호출합니다 (구간분리, 비용 발생).")
    if st.button("세션 만들고 구간분리 시작", type="primary", disabled=not ready):
        _create_and_segment(session_id, uploaded, int(child_age_months), child_gender, subject_type)


def _create_and_segment(session_id: str, uploaded, child_age_months: int, child_gender: str, subject_type: str) -> None:
    from mssb_coder.pipeline import run_segmentation_stage  # noqa: PLC0415

    session_dir = storage.session_dir(session_id)
    upload_dir = session_dir / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"original{Path(uploaded.name).suffix or '.mp4'}{video_vault.ENCRYPTED_SUFFIX}"
    with st.spinner("영상을 암호화해 저장하는 중..."):
        uploaded.seek(0)
        video_vault.encrypt_upload(uploaded, video_path)
    log_event("session_create", session_id)

    metadata = SessionMetadata(
        session_id=session_id,
        child_age_months=child_age_months,
        child_gender=child_gender,
        subject_type=SubjectType(subject_type),
    )
    try:
        with st.spinner("오디오 추출 + 전사 + Claude 구간분리 중... (영상 길이에 따라 여러 분 걸릴 수 있습니다)"):
            run_segmentation_stage(video_path, metadata)
    except Exception as exc:  # noqa: BLE001 — ffmpeg/전사/Claude 중 어디서 나든 사람이 읽을 메시지로 보여줌
        st.error(f"구간분리에 실패했습니다: {exc}")
        return

    st.session_state[SESSION_SELECT_KEY] = session_id
    st.success(f"세션 '{session_id}' 구간분리 완료 — 「1. 구간 확인」 화면으로 이동합니다.")
    st.rerun()
