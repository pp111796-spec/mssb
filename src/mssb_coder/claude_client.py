"""Claude API 호출 공통 로직 — 텍스트만 전송, tool_use 파싱.

원본 이미지/영상은 절대 보내지 않는다 (계획 전체의 핵심 전제). 시스템 프롬프트에
비진단·해석 층위2 제약을 항상 포함해, 매뉴얼 §10.1·§2.2의 오용금지 목록을 코드 수준에서 강제한다.

프롬프트 구성 함수(build_*)는 순수 문자열 조합이라 API 키 없이 지금 테스트 가능하다.
실제 API 호출(call_stem_coding)은 ANTHROPIC_API_KEY가 있어야 동작한다.
"""

from __future__ import annotations

import os

from mssb_coder.coding_systems import CodingSystem, build_claude_tool_schema
from mssb_coder.schema import STANDARD_DISCLAIMER

MODEL_VERSION = "claude-opus-5"

NON_DIAGNOSTIC_CONSTRAINT = (
    "이 결과는 진단·위험분류·임상적 판단이 아니다. 해석은 '이야기 전반의 패턴' 층위를 넘지 않는다 "
    "(매뉴얼 §10.1의 해석 층위 4단계 중 층위 2까지만). 다른 자료와의 통합이나 아동학대 여부 판단 같은 "
    "층위 3·4에 해당하는 내용은 절대 만들지 않는다. 모든 응답에 다음 면책 문구가 있다고 가정하고 "
    f"작성한다: \"{STANDARD_DISCLAIMER}\""
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
        f"완료 방식: {response_mode} (쟁점 프롬프트 사용 횟수: {issue_prompt_count})\n"
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
) -> dict:
    """Claude를 호출해 tool_use 결과(dict)를 반환한다. 호출 쪽이 schema.StemCodingResult로 검증할 것.

    lazy import: anthropic SDK가 설치돼 있지 않아도 이 모듈을 import할 수 있다.
    """
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=_require_api_key())
    tool = build_claude_tool_schema(coding_system)
    system_prompt = build_stem_coding_system_prompt(coding_system)
    user_message = build_stem_coding_user_message(
        transcript_text=transcript_text,
        nonverbal_summary=nonverbal_summary,
        story_specific_notes=story_specific_notes,
        child_age_months=child_age_months,
        response_mode=response_mode,
        issue_prompt_count=issue_prompt_count,
    )

    response = client.messages.create(
        model=MODEL_VERSION,
        max_tokens=4096,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError("Claude 응답에 tool_use 블록이 없음")
