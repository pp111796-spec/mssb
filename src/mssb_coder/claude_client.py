"""Claude API 호출 공통 로직 — 텍스트만 전송, structured outputs(JSON) 파싱.

원본 이미지/영상은 절대 보내지 않는다 (계획 전체의 핵심 전제). 시스템 프롬프트에
비진단·해석 층위2 제약을 항상 포함해, 매뉴얼 §10.1·§2.2의 오용금지 목록을 코드 수준에서 강제한다.

프롬프트 구성 함수(build_*)는 순수 문자열 조합이라 API 키 없이 지금 테스트 가능하다.
실제 API 호출(call_stem_coding)은 ANTHROPIC_API_KEY가 있어야 동작한다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from mssb_coder.coding_systems import CodingSystem, build_claude_tool_schema, to_structured_output_schema
from mssb_coder.labels import response_mode_label
from mssb_coder.schema import STANDARD_DISCLAIMER, ResponseMode

MODEL_VERSION = "claude-opus-5-5"
# Opus 5.5의 기본 effort는 medium — 채점 품질을 위해 명시적으로 high를 쓴다.
EFFORT = "high"
# thinking이 항상 켜져 있고 max_tokens에 포함되므로 넉넉히 둔다 (스트리밍이라 타임아웃 걱정 없음).
MAX_TOKENS = 64000
MAX_RETRIES = 4
# 안전 분류기가 오탐으로 거절하면 Anthropic이 권장하는 모델로 같은 요청을 다시 돌린다.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredResult:
    data: dict
    model: str  # 실제로 응답한 모델 (fallback이 돌았으면 원래 모델과 다를 수 있음)
    usage: dict = field(default_factory=dict)


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic  # noqa: PLC0415 — SDK 없이도 이 모듈을 import할 수 있게

        _client = anthropic.Anthropic(api_key=_require_api_key(), max_retries=MAX_RETRIES)
    return _client


def _usage_dict(usage) -> dict:
    return {
        name: getattr(usage, name, None) or 0
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    }


def call_structured(
    *, system_prompt: str, user_message: str, schema: dict, call_name: str, client=None
) -> StructuredResult:
    """Claude를 한 번 호출해 schema에 맞는 JSON(dict)을 받는다. 세 호출(구간분리·스템채점·종합) 공통.

    pydantic 검증은 호출 쪽 책임 — 여기서는 응답이 잘리거나 거절되거나 JSON이 아닌 경우만 걸러낸다.
    """
    client = client or _get_client()
    with client.beta.messages.stream(
        model=MODEL_VERSION,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message}],
        output_config={
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": to_structured_output_schema(schema)},
        },
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise ClaudeCallError(f"{call_name}: Claude가 응답을 거절했습니다 (분류: {category or '알 수 없음'}).")
    if response.stop_reason == "max_tokens":
        raise ClaudeCallError(f"{call_name}: 응답이 최대 길이({MAX_TOKENS} 토큰)에서 잘렸습니다.")

    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        raise ClaudeCallError(f"{call_name}: 응답에 JSON 본문이 없습니다 (stop_reason={response.stop_reason!r}).")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaudeCallError(f"{call_name}: 응답이 올바른 JSON이 아닙니다 ({exc}).") from exc
    if not isinstance(data, dict):
        raise ClaudeCallError(f"{call_name}: 응답 JSON의 최상위가 객체가 아닙니다.")

    return StructuredResult(data=data, model=response.model, usage=_usage_dict(response.usage))

NON_DIAGNOSTIC_CONSTRAINT = (
    "이 결과는 진단·위험분류·임상적 판단이 아니다. 해석은 '이야기 전반의 패턴' 층위를 넘지 않는다 "
    "(매뉴얼 §10.1의 해석 층위 4단계 중 층위 2까지만). 다른 자료와의 통합이나 아동학대 여부 판단 같은 "
    "층위 3·4에 해당하는 내용은 절대 만들지 않는다. 모든 응답에 다음 면책 문구가 있다고 가정하고 "
    f"작성한다: \"{STANDARD_DISCLAIMER}\""
)


# 보고서에 그대로 실리는 자유 텍스트(rationale 등)의 문체 지시 — .claude/skills/report-tone 원칙과 맞춘다.
REPORT_TONE_GUIDE = (
    "보고서에 그대로 실리는 문장(rationale·pattern_description 등)은 다음 문체 원칙을 지키세요.\n"
    "1) 하다체 서술문으로 끝맺으세요('~했다', '~이다', '~보인다'). '~됨', '~않음', '~부재'처럼 명사형으로 "
    "끝내는 메모체는 쓰지 마세요.\n"
    "2) 코드·변인명을 쓰지 마세요. 영문 id, 표정 코드(AU 번호), '부재로 코딩했다'·'존재로 채점했다' 같은 "
    "채점자용 표현 대신 관찰된 말과 행동을 직접 서술하세요.\n"
    "3) 수동태·물주구문 같은 번역체를 쓰지 말고 아이나 행동을 주어로 한 능동형으로 쓰세요"
    "('~가 관찰되었다' 대신 '~했다').\n"
    "4) 전문가끼리 읽는 보고서입니다. 같은 뜻을 되풀이해 분량을 늘리지 마세요.\n"
    "5) 문장을 매끄럽게 만들려고 용어를 바꿔 쓰지 말고, 같은 대상은 같은 용어로 반복하세요.\n"
    "6) 항목을 '두 항목', '5개 항목'처럼 개수로 부르지 마세요. 독자는 어떤 항목이 있는지 모르므로 해당 "
    "행동을 직접 서술하세요(예: '공격성과 관련된 행동은 관찰되지 않았다').\n"
)


def build_stem_coding_system_prompt(coding_system: CodingSystem) -> str:
    """스템당 1회 호출(10단계)의 시스템 프롬프트. cache_control로 캐싱 대상."""
    category_lines = []
    for category in coding_system.categories:
        item_names = ", ".join(item.display_name_ko for item in category.items)
        category_lines.append(f"- {category.display_name_ko}: {item_names}")

    return (
        "당신은 MSSB(MacArthur Story Stem Battery) 이야기 줄기 1편의 아동 반응을 채점하는 "
        "보조 도구입니다. 아래 5범주의 세부 항목마다 존재/부재를 판단하고, 각 항목 판단의 근거를 "
        "반드시 텍스트로 남기세요 (근거가 없으면 검토자가 신뢰할 수 없습니다).\n\n"
        "5범주 및 세부 항목:\n" + "\n".join(category_lines) + "\n\n"
        "중요: 코드의 의미는 이야기마다 다를 수 있습니다 — 제공되는 story_specific_notes를 반드시 "
        "반영하세요. 제공되는 child_age_months를 참고해 그 연령대의 정상 발달 범위를 감안하세요 "
        "(예: 어린 연령이 딜레마의 한쪽 측면만 다루는 것은 정상일 수 있음).\n\n"
        f"{REPORT_TONE_GUIDE}\n"
        f"{NON_DIAGNOSTIC_CONSTRAINT}"
    )


def build_stem_coding_user_message(
    *,
    transcript_text: str,
    nonverbal_summary: str,
    story_specific_notes: str,
    child_age_months: int,
    response_mode: str,
    issue_prompt_count: int,
) -> str:
    """10단계 메시지 본문 — 해당 스템 전사문 + 비언어 요약 + 스템 고유 주의사항 + 아동 연령."""
    return (
        f"아동 연령: {child_age_months}개월\n"
        f"완료 방식: {response_mode_label(ResponseMode(response_mode))} "
        f"(쟁점 프롬프트 사용 횟수: {issue_prompt_count})\n"
        f"이 스템의 코드 의미 주의사항:\n{story_specific_notes}\n\n"
        f"--- 전사문 (아동 발화 구간) ---\n{transcript_text}\n\n"
        f"--- 비언어 신호 요약 ---\n{nonverbal_summary}\n"
    )


def _require_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. .env에 키를 넣으세요."
        )
    return api_key


def call_stem_coding(
    coding_system: CodingSystem,
    *,
    transcript_text: str,
    nonverbal_summary: str,
    story_specific_notes: str,
    child_age_months: int,
    response_mode: str,
    issue_prompt_count: int,
    client=None,
) -> StructuredResult:
    """스템 1편을 채점한다. 호출 쪽이 결과의 data를 schema.StemCodingResult로 검증할 것."""
    return call_structured(
        system_prompt=build_stem_coding_system_prompt(coding_system),
        user_message=build_stem_coding_user_message(
            transcript_text=transcript_text,
            nonverbal_summary=nonverbal_summary,
            story_specific_notes=story_specific_notes,
            child_age_months=child_age_months,
            response_mode=response_mode,
            issue_prompt_count=issue_prompt_count,
        ),
        schema=build_claude_tool_schema(coding_system)["input_schema"],
        call_name="스템 채점",
        client=client,
    )
