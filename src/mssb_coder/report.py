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


PLACEHOLDER = "(전체 종합 보고서에서 가져올 예정)"

TEST_OVERVIEW = (
    "이 검사는 인형으로 짧은 이야기의 시작 부분을 보여 준 뒤, 아이가 이야기를 어떻게 이어 가는지 "
    "살펴보는 놀이 형식의 검사이다. 아이가 이야기 속 인물의 마음과 행동, 갈등이 풀리는 방식을 "
    "어떻게 표현하는지를 통해 아이가 관계와 감정을 어떻게 이해하고 있는지 살펴본다."
)

INTERPRETATION_CAVEAT = (
    "이 보고서는 한 번의 검사 장면에서 관찰된 내용에 근거한 것으로, 아이의 전체 모습을 대표하지 않을 수 있다. "
    "내용은 인공지능이 작성한 초안을 검사자가 검토해 확정한 것이며, 제언은 참고용 안내이다."
)


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

    age_months = stem_results[0].child_age_months if stem_results else None
    lines.append("## 인적 사항")
    lines.append("")
    lines.append(f"- 아동 연령: {f'{age_months}개월' if age_months else PLACEHOLDER}")
    lines.append(f"- 검사일: {PLACEHOLDER}")
    lines.append(f"- 검사자: {PLACEHOLDER}")
    lines.append("")

    lines.append("## 검사 개요")
    lines.append("")
    lines.append(TEST_OVERVIEW)
    lines.append("")

    lines.append("## 의뢰 사유 및 배경")
    lines.append("")
    lines.append(PLACEHOLDER)
    lines.append("")

    if synthesis.behavioral_observation:
        lines.append("## 검사 중 행동관찰")
        lines.append("")
        lines.append(scrub_english_ids(synthesis.behavioral_observation, coding_system, story_stems))
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

    if synthesis.strengths or synthesis.support_areas:
        lines.append("## 강점과 지원이 필요한 영역")
        lines.append("")
        if synthesis.strengths:
            lines.append(f"**강점** {scrub_english_ids(synthesis.strengths, coding_system, story_stems)}")
            lines.append("")
        if synthesis.support_areas:
            lines.append(f"**지원이 필요한 영역** {scrub_english_ids(synthesis.support_areas, coding_system, story_stems)}")
            lines.append("")

    if synthesis.recommendations:
        lines.append("## 제언")
        lines.append("")
        for recommendation in synthesis.recommendations:
            lines.append(f"- {scrub_english_ids(recommendation, coding_system, story_stems)}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 해석상 유의점")
    lines.append("")
    lines.append(INTERPRETATION_CAVEAT)
    lines.append("")
    lines.append(synthesis.not_included_by_design)

    return "\n".join(lines)
