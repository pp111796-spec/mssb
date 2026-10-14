"""세션 종합 결과를 학부모가 받는 실제 심리검사보고서 형태의 Markdown으로 만든다.

원본 JSON(`session_summary_*.json`, `<stem>_ai_codes.json`)은 이미 MSSB_DATA_DIR에
그대로 저장돼 있다 — 이 모듈은 그걸 대체하지 않고, 두 가지 산출물(이야기별 해석/종합 해석)로
재구성만 한다. 계획이 명시한 "해석 층위2까지만" 제약을 지키기 위해, 원본 JSON에 없는
문장(진단·위험분류 등)을 새로 만들지 않고 있는 필드만 그대로 옮겨 적는다.

범주·항목 체크리스트와 타임스탬프 근거 인용은 이 문서에서 뺀다 — 그런 감사용 상세는
검사자용 검토 화면(review_app의 스템 상세 드릴다운)에만 남긴다. 여기서는 Claude가 이미 쓴
rationale/pattern_description 문장을 그대로(또는 이어붙여서) 서술형 해석문으로만 보여준다.
"""

from __future__ import annotations

from mssb_coder.coding_systems import CodingSystem, load_coding_system
from mssb_coder.labels import category_label, scrub_english_ids, stem_label
from mssb_coder.schema import ResponseMode, SessionSynthesis, StemCodingResult
from mssb_coder.segmentation import StoryStemsConfig, load_story_stems


def _completion_sentence(result: StemCodingResult) -> str:
    if result.response_mode == ResponseMode.SPONTANEOUS:
        return "아이는 이 이야기를 검사자의 프롬프트 없이 자발적으로 완성했다."
    return (
        f"아이는 이 이야기를 검사자의 프롬프트(쟁점 프롬프트 {result.issue_prompt_count}회) "
        "후에 완성했다."
    )


def _stem_interpretation_paragraph(
    coding_system: CodingSystem, story_stems: StoryStemsConfig, result: StemCodingResult
) -> str:
    """5범주 rationale(이미 Claude가 쓴 완결된 문장)을 이어붙여 한 단락으로 — 추가 Claude 호출 없음."""
    rationales = [
        scrub_english_ids(result.codes[category.id].rationale, coding_system, story_stems)
        for category in coding_system.categories
        if category.id in result.codes and result.codes[category.id].rationale
    ]
    return " ".join(rationales)


def build_session_report_markdown(
    synthesis: SessionSynthesis,
    stem_results: list[StemCodingResult] | None = None,
    coding_system: CodingSystem | None = None,
    story_stems: StoryStemsConfig | None = None,
) -> str:
    coding_system = coding_system or load_coding_system()
    story_stems = story_stems or load_story_stems()
    lines: list[str] = [
        f"# MSSB 세션 결과 — {synthesis.session_id}",
        "",
        f"생성 시각: {synthesis.generated_at}",
        "",
        f"> {synthesis.disclaimer}",
        "",
        f"**정상 실시된 스템**: {synthesis.stems_administered_count}개 "
        f"({', '.join(stem_label(story_stems, s) for s in synthesis.stems_included) or '없음'})",
    ]
    if synthesis.stems_missing_or_partial:
        lines.append(
            "**미실시/부분실시**: "
            f"{', '.join(stem_label(story_stems, s) for s in synthesis.stems_missing_or_partial)}"
        )
    lines.append("")

    if stem_results:
        lines.append("## 이야기별 해석")
        for result in sorted(stem_results, key=lambda r: r.stem_name):
            lines.append(f"\n### {stem_label(story_stems, result.stem_name)}")
            lines.append(_completion_sentence(result))
            lines.append(_stem_interpretation_paragraph(coding_system, story_stems, result))
        lines.append("")

    lines.append("## 종합 해석")
    lines.append("")
    if not synthesis.recurring_patterns:
        lines.append("이야기 전반에 걸쳐 반복적으로 관찰된 패턴은 없었다.")
        lines.append("")
    for pattern in synthesis.recurring_patterns:
        label = category_label(coding_system, pattern.category)
        lines.append(
            f"**{label}** 영역은 전체 {pattern.stems_administered_count_for_pattern}개 이야기 중 "
            f"{len(pattern.observed_in_stems)}개에서 일관되게 관찰되었다."
        )
        lines.append("")
        lines.append(scrub_english_ids(pattern.pattern_description, coding_system, story_stems))
        lines.append("")

    if synthesis.single_story_notable_signals:
        lines.append(
            "이 외에, 특정 이야기에서만 두드러지게 나타나 반복 패턴으로는 분류되지 않지만 "
            "참고할 만한 신호가 다음과 같이 관찰되었다."
        )
        lines.append("")
        for signal in synthesis.single_story_notable_signals:
            label = category_label(coding_system, signal.category)
            note = scrub_english_ids(signal.note, coding_system, story_stems)
            lines.append(
                f"- {stem_label(story_stems, signal.stem_name)} 이야기에서는 {label} 영역과 관련하여 "
                f'"{note}"가 관찰되었다.'
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(synthesis.not_included_by_design)

    return "\n".join(lines)
