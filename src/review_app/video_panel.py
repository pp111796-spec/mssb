"""구간 미리보기 영상 — 원본 전체 대신 구간만 자른 저화질 클립을 재생한다 (구간 확인·스템 상세 공용)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mssb_coder import storage, video_prep
from mssb_coder.schema import SessionMetadata, SubjectType
from review_app.common import FILE_FORMAT_ERRORS, widget_key


def session_video_path(session_id: str) -> Path | None:
    if not storage.session_metadata_exists(session_id):
        return None
    try:
        metadata = storage.load_session_metadata(session_id)
    except FILE_FORMAT_ERRORS:
        return None
    if not metadata.video_path:
        return None
    path = Path(metadata.video_path)
    return path if path.exists() else None


def _render_video_setup(session_id: str) -> None:
    """영상 경로(와 예전 세션이라 session.json이 없으면 아동 정보)를 입력받아 session.json에 저장한다."""
    has_metadata = storage.session_metadata_exists(session_id)
    current = storage.load_session_metadata(session_id) if has_metadata else None
    if current and current.video_path:
        st.warning(f"저장된 영상 경로에 파일이 없습니다: {current.video_path}")
    else:
        st.info("미리보기를 보려면 이 세션의 원본 영상 경로를 한 번 입력하세요.")

    with st.form(widget_key(session_id, "video_setup")):
        video_path = st.text_input(
            "원본 영상 경로", value=current.video_path if current and current.video_path else "",
            placeholder=r"예: C:\mssb_data\raw_videos\세션영상.mp4",
        )
        if current is None:
            st.caption("이 세션은 세션 정보 파일(session.json)이 없는 예전 세션이라 아동 정보도 함께 입력해야 합니다.")
            age = st.number_input("아동 연령(개월)", min_value=1, max_value=240, value=54, step=1)
            gender = st.radio("아동 성별", ["female", "male"], format_func=lambda g: "여아" if g == "female" else "남아", horizontal=True)
            subject = st.radio(
                "영상 종류", ["adult", "child"], horizontal=True,
                format_func=lambda s: "성인 대역(배선 확인용)" if s == "adult" else "실제 아동",
            )
        submitted = st.form_submit_button("저장")

    if not submitted:
        return
    path = Path(video_path.strip().strip('"'))
    if not path.is_file():
        st.error(f"그 경로에 파일이 없습니다: {path}")
        return
    if current is None:
        metadata = SessionMetadata(
            session_id=session_id, child_age_months=int(age), child_gender=gender,
            subject_type=SubjectType(subject), video_path=str(path),
        )
    else:
        metadata = current.model_copy(update={"video_path": str(path)})
    storage.save_session_metadata(metadata)
    st.rerun()


def render_clip(session_id: str, start_s: float, end_s: float, jump_to_s: float | None, key: str) -> None:
    """[start_s, end_s] 구간(앞뒤 여백 포함) 클립을 보여주고, jump_to_s(세션 시각)부터 재생한다."""
    video_path = session_video_path(session_id)
    if video_path is None:
        _render_video_setup(session_id)
        return

    out_dir = storage.session_dir(session_id) / "preview"
    clip_path, clip_start, _ = video_prep.preview_clip_path(start_s, end_s, out_dir)
    if not clip_path.exists():
        st.caption("이 구간의 미리보기 클립이 아직 없습니다. 만들면 다음부터는 바로 열립니다.")
        if not st.button("🎬 미리보기 클립 만들기 (수십 초 걸릴 수 있음)", key=widget_key(session_id, key, "make_clip")):
            return
        with st.spinner("클립을 만드는 중..."):
            try:
                video_prep.extract_preview_clip(video_path, start_s, end_s, out_dir)
            except (video_prep.FfmpegNotFoundError, video_prep.FfmpegError) as exc:
                st.error(str(exc))
                return

    target = start_s if jump_to_s is None else jump_to_s
    st.video(str(clip_path), start_time=max(0, int(target - clip_start)))
    st.caption(f"클립은 세션 {clip_start:.1f}초부터 시작합니다 (구간 앞뒤 {video_prep.PREVIEW_MARGIN_S:.0f}초 포함).")
