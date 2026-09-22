"""세션 종합 결과를 사람이 읽기 쉬운 Markdown 보고서로 만든다 (Streamlit 다운로드 버튼용).

원본 JSON(`session_summary_*.json`, `<stem>_ai_codes.json`)은 이미 MSSB_DATA_DIR에
그대로 저장돼 있다 — 이 모듈은 그걸 대체하지 않고, 제출·공유용으로 보기 편하게
재구성만 한다. 계획이 명시한 "해석 층위2까지만" 제약을 지키기 위해, 원본 JSON에 없는
문장(진단·위험분류 등)을 새로 만들지 않고 있는 필드만 그대로 옮겨 적는다.
"""

from __future__ import annotations

from mssb_coder.coding_systems import CodingSystem, load_coding_system
from mssb_coder.schema import SessionSynthesis, StemCodingResult


def _category_label(coding_system: CodingSystem, category_id: str) -> str:
    for category in coding_system.categories:
        if category.id == category_id:
            return category.display_name_ko
    return category_id


def build_session_report_markdown(
    synthesis: SessionSynthesis,
    stem_results: list[StemCodingResult] | None = None,
    coding_system: CodingSystem | None = None,
) -> str:
    coding_system = coding_system or load_coding_system()
    lines: list[str] = [
        f"# MSSB 세션 종합 분석 보고서 — {synthesis.session_id}",
        "",
        f"생성 시각: {synthesis.generated_at}  ",
        f"모델 버전: {synthesis.model_version}  ",
        f"해석 층위: {synthesis.interpretation_layer}",
        "",
        f"> {synthesis.disclaimer}",
        "",
        f"**정상 실시된 스템**: {synthesis.stems_administered_count}개 "
        f"({', '.join(synthesis.stems_included) or '없음'})",
    ]
    if synthesis.stems_missing_or_partial:
        lines.append(f"**미실시/부분실시**: {', '.join(synthesis.stems_missing_or_partial)}")
    lines.append("")

    lines.append("## 반복되는 패턴")
    if not synthesis.recurring_patterns:
        lines.append("(발견된 반복 패턴 없음)")
    for pattern in synthesis.recurring_patterns:
        label = _category_label(coding_system, pattern.category)
        lines.append(
            f"### {label} — {len(pattern.observed_in_stems)}/"
            f"{pattern.stems_administered_count_for_pattern}개 이야기"
        )
        lines.append(pattern.pattern_description)
        lines.append("")
        if pattern.evidence:
            lines.append("근거:")
            for ev in pattern.evidence:
                note = f": {ev.note}" if ev.note else ""
                lines.append(f"- {ev.stem_name} ({ev.timestamp_s:.1f}초){note}")
        lines.append("")

    if synthesis.single_story_notable_signals:
        lines.append("## 단발성 신호 (패턴 아님)")
        for signal in synthesis.single_story_notable_signals:
            label = _category_label(coding_system, signal.category)
            lines.append(f"- **{signal.stem_name}** ({label}): {signal.note}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(synthesis.not_included_by_design)

    if stem_results:
        lines.append("")
        lines.append("## 부록 — 이야기별 상세 채점 (AI 원본 초안)")
        for result in sorted(stem_results, key=lambda r: r.stem_name):
            lines.append(
                f"\n### {result.stem_name} "
                f"(완료상태: {result.completion_status.value}, {result.response_mode.value})"
            )
            for category_id, category_code in result.codes.items():
                label = _category_label(coding_system, category_id)
                present_items = [
                    item_id for item_id, item in category_code.items.items() if item.present
                ]
                lines.append(
                    f"- **{label}** (존재 {category_code.present_count}개): "
                    f"{', '.join(present_items) or '없음'}"
                )
                if category_code.rationale:
                    lines.append(f"  - 근거: {category_code.rationale}")

    return "\n".join(lines)
