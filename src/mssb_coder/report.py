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

from datetime import date, datetime, timedelta, timezone

from mssb_coder.coding_systems import CodingSystem, load_coding_system
from mssb_coder.labels import category_label, scrub_english_ids, stem_label
from mssb_coder.schema import CompletionStatus, ResponseMode, SessionSynthesis, StemCodingResult
from mssb_coder.segmentation import StoryStemsConfig, load_story_stems


KST = timezone(timedelta(hours=9))

PLACEHOLDER = "(전체 종합 보고서에서 가져올 예정)"
NOT_ENTERED = "(미입력)"

TEST_OVERVIEW = (
    "이 검사는 인형으로 짧은 이야기의 시작 부분을 보여 준 뒤, 아이가 이야기를 어떻게 이어 가는지 "
    "살펴보는 놀이 형식의 검사이다. 아이가 이야기 속 인물의 마음과 행동, 갈등이 풀리는 방식을 "
    "어떻게 표현하는지를 통해 아이가 관계와 감정을 어떻게 이해하고 있는지 살펴본다."
)

INTERPRETATION_CAVEAT = "이 보고서는 한 번의 검사 장면에서 관찰한 내용에 근거하므로 아이의 전체 모습을 대표하지 않을 수 있다."
REVIEW_STATUS_REVIEWED = "인공지능이 작성한 초안을 검사자가 검토해 확정했다. 제언은 참고용 안내이다."
REVIEW_STATUS_DRAFT = "인공지능이 작성한 초안이며, 검사자가 아직 검토하지 않았다. 제언은 참고용 안내이다."


def _korean_date(iso_timestamp: str) -> str:
    """'2026-10-14T10:21:30.61+00:00' → '2026년 10월 14일' (한국 시간 기준). 형식이 다르면 원문 그대로."""
    try:
        moment = datetime.fromisoformat(iso_timestamp).astimezone(KST)
    except (TypeError, ValueError):
        return iso_timestamp
    return f"{moment.year}년 {moment.month}월 {moment.day}일"


def _korean_day(iso_date: str | None) -> str:
    """검사일 'YYYY-MM-DD' → 'YYYY년 M월 D일'. 비었거나 형식이 다르면 미입력 표시 또는 원문."""
    if not iso_date:
        return NOT_ENTERED
    try:
        day = date.fromisoformat(iso_date)
    except ValueError:
        return iso_date
    return f"{day.year}년 {day.month}월 {day.day}일"


def stem_is_confirmed(result: StemCodingResult) -> bool:
    return result.administration_source == "examiner" and result.reviewed_at is not None


def report_blockers(
    synthesis: SessionSynthesis, stem_results: list[StemCodingResult], story_stems: StoryStemsConfig
) -> list[str]:
    """보고서를 내보내기 전에 남은 일 목록. 비어 있어야 PDF를 내보낼 수 있고 '확정' 문구가 나간다."""
    blockers: list[str] = []
    by_name = {r.stem_name: r for r in stem_results}
    missing = [s for s in synthesis.stems_included if s not in by_name]
    if missing or not stem_results:
        names = ", ".join(stem_label(story_stems, s) for s in missing)
        blockers.append(f"채점 결과가 없는 이야기가 있습니다{f': {names}' if names else ''}.")
    unconfirmed = [r.stem_name for r in stem_results if not stem_is_confirmed(r)]
    if unconfirmed:
        names = ", ".join(stem_label(story_stems, s) for s in unconfirmed)
        blockers.append(f"검사자가 확인·확정하지 않은 이야기: {names}.")
    if synthesis.stale:
        blockers.append("이야기 수정 내용이 종합 서술에 반영되지 않았습니다. 종합을 다시 쓰세요.")
    elif synthesis.reviewed_at is None:
        blockers.append("종합 서술을 검토해 저장하지 않았습니다.")
    if not synthesis.test_date:
        blockers.append("검사일을 입력하지 않았습니다.")
    if not synthesis.examiner_name:
        blockers.append("검사자를 입력하지 않았습니다.")
    return blockers


def _completion_sentence(result: StemCodingResult) -> str:
    """실시 기록 문장. 검사자가 확정하기 전에는 사실로 단정하지 않도록 쓰지 않는다."""
    if not stem_is_confirmed(result):
        return ""
    if result.completion_status == CompletionStatus.NOT_ADMINISTERED:
        return "이 이야기는 실시하지 않았다."
    prefix = "아이는 이 이야기를 일부만 이어 갔다. " if result.completion_status == CompletionStatus.PARTIAL else ""
    if result.response_mode == ResponseMode.SPONTANEOUS:
        return prefix + "아이는 이 이야기를 검사자의 프롬프트 없이 자발적으로 이어 갔다."
    return prefix + f"아이는 검사자의 프롬프트(쟁점 프롬프트 {result.issue_prompt_count}회) 후에 이 이야기를 이어 갔다."


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
        f"작성일: {_korean_date(synthesis.reviewed_at or synthesis.generated_at)}",
        "",
        f"> {synthesis.disclaimer}",
        "",
        f"**실시한 이야기**: {synthesis.stems_administered_count}편 "
        f"({', '.join(stem_label(story_stems, s) for s in synthesis.stems_included) or '없음'})",
    ]
    if synthesis.stems_missing_or_partial:
        lines.append(
            "**실시하지 않았거나 일부만 실시한 이야기**: "
            f"{', '.join(stem_label(story_stems, s) for s in synthesis.stems_missing_or_partial)}"
        )
    lines.append("")

    age_months = stem_results[0].child_age_months if stem_results else None
    lines.append("## 인적 사항")
    lines.append("")
    lines.append(f"- 아동 연령: {f'{age_months}개월' if age_months else PLACEHOLDER}")
    lines.append(f"- 검사일: {_korean_day(synthesis.test_date)}")
    lines.append(f"- 검사자: {synthesis.examiner_name or NOT_ENTERED}")
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
        for result in stem_results:  # 호출 쪽이 넘긴 순서가 실시 순서 (storage.load_effective_stem_codings)
            lines.append(f"\n### {stem_label(story_stems, result.stem_name)}")
            lines.append(_completion_sentence(result))
            lines.append(_stem_interpretation_paragraph(coding_system, story_stems, result))
        lines.append("")

    lines.append("## 종합 해석")
    lines.append("")
    if not synthesis.recurring_patterns:
        lines.append("이야기 전반에 걸쳐 반복해서 나타난 패턴은 없었다.")
        lines.append("")
    for pattern in synthesis.recurring_patterns:
        label = category_label(coding_system, pattern.category)
        lines.append(
            f"**{label}** 영역은 전체 {pattern.stems_administered_count_for_pattern}개 이야기 중 "
            f"{len(pattern.observed_in_stems)}개에서 나타났다."
        )
        lines.append("")
        lines.append(scrub_english_ids(pattern.pattern_description, coding_system, story_stems))
        lines.append("")

    if synthesis.single_story_notable_signals:
        lines.append(
            "이 외에, 특정 이야기에서만 두드러져 반복 패턴으로 분류하지 않았지만 "
            "참고할 만한 신호는 다음과 같다."
        )
        lines.append("")
        for signal in synthesis.single_story_notable_signals:
            label = category_label(coding_system, signal.category)
            note = scrub_english_ids(signal.note, coding_system, story_stems)
            lines.append(
                f"- {stem_label(story_stems, signal.stem_name)} 이야기({label} 영역): {note}"
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
    confirmed = not report_blockers(synthesis, stem_results or [], story_stems)
    review_status = REVIEW_STATUS_REVIEWED if confirmed else REVIEW_STATUS_DRAFT
    lines.append(f"{INTERPRETATION_CAVEAT} {review_status}")
    lines.append("")
    scope_note = synthesis.not_included_by_design.strip()
    lines.append(scope_note if scope_note.endswith(".") else scope_note + ".")

    return "\n".join(lines)
