"""공개 배포용 결과 뷰어 — Streamlit Community Cloud 등 인터넷에 올리는 앱.

**로컬 분석 도구(src/review_app/app.py)와 완전히 분리돼 있다.** 이 앱은 MSSB_DATA_DIR에
전혀 접근하지 않는다 — 영상 분석·전사·AI 채점은 전부 로컬 컴퓨터에서 끝낸 뒤, 그 결과를
"세션 내보내기" 번들(JSON)로 다운로드해서 여기 업로드해야만 볼 수 있다. **영상 원본은
이 앱에 절대 올라오지 않는다** — 애초에 업로드 대상이 아니다.

비밀번호 게이트: Streamlit Cloud 앱 설정 > Secrets에 다음을 추가해야 한다.
    APP_PASSWORD = "원하는 비밀번호"
비밀번호가 설정돼 있지 않으면 이 앱은 접근을 막고 안내만 보여준다(열어두지 않음).

저장 안 함: 업로드한 파일은 그 브라우저 세션에만 존재한다. 이 앱은 어디에도 데이터를
쓰지 않는다 — 새로고침하거나 다시 방문하면 다시 업로드해야 한다(의도된 설계). 민감한
아동 평가 결과를 제3자 무료 클라우드에 계속 쌓아두지 않기 위한 선택.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from mssb_coder.bundle import BundleParseError, parse_bundle_dict
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.report import build_session_report_markdown

st.set_page_config(page_title="MSSB 분석 결과", layout="wide", page_icon="📋")


def _configured_password() -> str | None:
    try:
        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


def check_password() -> bool:
    configured = _configured_password()
    if not configured:
        st.error(
            "관리자가 아직 비밀번호를 설정하지 않았습니다 (Streamlit Cloud > 앱 설정 > "
            "Secrets에 APP_PASSWORD 추가 필요). 설정 전까지는 아무도 접근할 수 없습니다."
        )
        st.stop()

    if st.session_state.get("password_correct"):
        return True

    entered = st.text_input("비밀번호", type="password")
    if entered:
        if entered == configured:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False


def render_synthesis_readonly(synthesis, stem_results, coding_system) -> None:
    st.title(f"세션 종합 분석 — {synthesis.session_id}")
    st.info(synthesis.disclaimer)
    st.caption(
        f"정상 실시된 스템: {synthesis.stems_administered_count}개 "
        f"(포함: {', '.join(synthesis.stems_included) or '없음'})"
    )
    if synthesis.stems_missing_or_partial:
        st.warning(f"미실시/부분실시: {', '.join(synthesis.stems_missing_or_partial)}")

    report_markdown = build_session_report_markdown(
        synthesis, stem_results=stem_results, coding_system=coding_system
    )
    st.download_button(
        "📄 보고서 다운로드 (.md)",
        data=report_markdown,
        file_name=f"{synthesis.session_id}_mssb_report.md",
        mime="text/markdown",
    )

    def _category_label(category_id: str) -> str:
        for category in coding_system.categories:
            if category.id == category_id:
                return category.display_name_ko
        return category_id

    st.header("반복되는 패턴")
    if not synthesis.recurring_patterns:
        st.write("발견된 반복 패턴 없음")
    for pattern in synthesis.recurring_patterns:
        label = _category_label(pattern.category)
        st.subheader(
            f"{label} — {len(pattern.observed_in_stems)}/"
            f"{pattern.stems_administered_count_for_pattern}개 이야기"
        )
        st.write(pattern.pattern_description)
        for ev in pattern.evidence:
            st.caption(f"근거: {ev.stem_name} ({ev.timestamp_s:.1f}초) — {ev.note}")

    if synthesis.single_story_notable_signals:
        st.header("단발성 신호 (패턴 아님)")
        for signal in synthesis.single_story_notable_signals:
            st.write(f"**{signal.stem_name}** ({_category_label(signal.category)}): {signal.note}")

    if stem_results:
        st.header("이야기별 상세 (부록)")
        for result in sorted(stem_results, key=lambda r: r.stem_name):
            with st.expander(f"{result.stem_name} ({result.completion_status.value})"):
                for category_id, category_code in result.codes.items():
                    label = _category_label(category_id)
                    present_items = [
                        item_id for item_id, item in category_code.items.items() if item.present
                    ]
                    st.write(f"**{label}** (존재 {category_code.present_count}개): {', '.join(present_items) or '없음'}")
                    if category_code.rationale:
                        st.caption(category_code.rationale)


def main() -> None:
    if not check_password():
        return

    st.title("📋 MSSB 분석 결과 뷰어")
    st.caption(
        "로컬에서 처리된 결과만 업로드해서 볼 수 있습니다. 영상 원본은 이 사이트에 올라오지 "
        "않으며, 업로드한 내용은 저장되지 않고 이 세션이 끝나면 사라집니다."
    )

    uploaded = st.file_uploader(
        "세션 내보내기 파일 업로드 (로컬 앱의 '🌐 공개 사이트 업로드용 내보내기' 버튼으로 받은 .json)",
        type=["json"],
    )
    if uploaded is None:
        st.stop()

    try:
        raw = json.loads(uploaded.read().decode("utf-8"))
        synthesis, stem_results = parse_bundle_dict(raw)
    except (BundleParseError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f"파일을 읽을 수 없습니다: {exc}")
        st.stop()
        return

    coding_system = load_coding_system()
    render_synthesis_readonly(synthesis, stem_results, coding_system)


if __name__ == "__main__":
    main()
