"""검토 앱 화면들이 함께 쓰는 설정 로더와 표시 도우미."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError

from mssb_coder.coding_systems import CodingSystem, load_coding_system
from mssb_coder.labels import scrub_english_ids
from mssb_coder.segmentation import StoryStemsConfig, load_story_stems

# 저장된 JSON이 깨졌거나 예전 형식일 때 나는 오류들
FILE_FORMAT_ERRORS = (ValidationError, json.JSONDecodeError, UnicodeDecodeError)


@st.cache_resource
def coding_system() -> CodingSystem:
    return load_coding_system()


@st.cache_resource
def story_stems() -> StoryStemsConfig:
    return load_story_stems()


def plain(text: str | None) -> str:
    """Claude 자유 텍스트에 섞인 영문 id·표정 코드를 한글로 바꿔 화면에 보여줄 문장으로 만든다."""
    return scrub_english_ids(text or "", coding_system(), story_stems())


def widget_key(session_id: str, *parts) -> str:
    """세션마다 다른 위젯 key — 세션을 바꿨을 때 이전 세션의 입력 내용이 남지 않게 한다."""
    return ":".join([session_id, *(str(p) for p in parts)])


def show_file_error(what: str, exc: Exception) -> None:
    st.error(
        f"{what} 파일을 읽지 못했습니다. 예전 형식이거나 손상된 파일일 수 있습니다. "
        f"파이프라인의 해당 단계를 다시 실행하면 새로 만들어집니다.\n\n세부 내용: {exc}"
    )
