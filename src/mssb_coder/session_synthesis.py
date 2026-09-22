"""스템별 AI 채점 원본 전체 → Claude(텍스트만)로 세션 종합 패턴 초안 생성 (파이프라인 12단계).

입력은 사람이 검토하기 전의 AI 원본 드래프트 전체(based_on: "ai_draft_stem_results") —
검토 시간 단축이 목적이라는 사용자 결정에 따른 의도적 트레이드오프
(계획 "세션 종합 패턴 초안 JSON 스키마" 설계 원칙 참고).
"""

from __future__ import annotations

from mssb_coder.claude_client import NON_DIAGNOSTIC_CONSTRAINT
from mssb_coder.schema import StemCodingResult

SYNTHESIS_SYSTEM_PROMPT = (
    "당신은 MSSB 세션 하나에서 나온 여러 이야기 스템의 AI 채점 결과를 종합해, 이야기 전반에 걸쳐 "
    "반복되는 패턴을 서술하는 도구입니다.\n\n"
    "단순 점수 평균을 내지 마세요 — 코드의 의미는 이야기마다 다를 수 있습니다 (매뉴얼 §2.3). "
    "어떤 범주가 몇 개 이야기에서 어떤 식으로 반복됐는지, 그리고 한 이야기에서만 두드러진 신호는 "
    "무엇인지(패턴이 아니라 단발성 신호로 별도 표시) 근거(스템명+타임스탬프)와 함께 정리하세요. "
    "분모(전체 유효 스템 수, 그 패턴에 사용된 유효 스템 수)를 반드시 함께 표기하세요 — "
    "\"3개 이야기에서 반복\"이 10개 중 3개인지 5개 중 3개인지는 해석이 완전히 다릅니다.\n\n"
    f"{NON_DIAGNOSTIC_CONSTRAINT}"
)


def build_synthesis_user_message(stem_results: list[StemCodingResult]) -> str:
    if not stem_results:
        raise ValueError("종합할 스템 채점 결과가 비어 있음")

    blocks = []
    for result in stem_results:
        category_lines = []
        for category_id, category_code in result.codes.items():
            present_items = [
                item_id for item_id, item in category_code.items.items() if item.present
            ]
            category_lines.append(
                f"  - {category_id}: present_count={category_code.present_count} "
                f"({', '.join(present_items) or '없음'}) — {category_code.rationale}"
            )
        blocks.append(
            f"### {result.stem_name} (완료상태: {result.completion_status.value}, "
            f"{result.response_mode.value})\n" + "\n".join(category_lines)
        )

    return "--- 스템별 AI 채점 원본 (전체) ---\n\n" + "\n\n".join(blocks)


SYNTHESIS_TOOL = {
    "name": "submit_session_synthesis",
    "description": "여러 스템의 AI 채점 결과를 종합해 세션 전체 패턴 초안을 만든다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stems_included": {"type": "array", "items": {"type": "string"}},
            "stems_missing_or_partial": {"type": "array", "items": {"type": "string"}},
            "stems_administered_count": {"type": "integer", "minimum": 0},
            "recurring_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "observed_in_stems": {"type": "array", "items": {"type": "string"}},
                        "stems_administered_count_for_pattern": {"type": "integer", "minimum": 0},
                        "pattern_description": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "stem_name": {"type": "string"},
                                    "timestamp_s": {"type": "number"},
                                    "note": {"type": "string"},
                                },
                                "required": ["stem_name", "timestamp_s"],
                            },
                        },
                    },
                    "required": [
                        "category", "observed_in_stems", "stems_administered_count_for_pattern",
                        "pattern_description", "evidence",
                    ],
                },
            },
            "single_story_notable_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stem_name": {"type": "string"},
                        "category": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["stem_name", "category", "note"],
                },
            },
        },
        "required": ["stems_included", "stems_administered_count", "recurring_patterns"],
    },
}


def call_session_synthesis(stem_results: list[StemCodingResult]) -> dict:
    """Claude를 호출해 세션 종합 결과(dict)를 반환한다. 호출 쪽이 schema.SessionSynthesis로 검증할 것.

    interpretation_layer/based_on/disclaimer 같은 고정 상수 필드는 Claude에게 묻지 않는다 —
    pipeline.py의 run_synthesis_stage가 이 함수의 반환값에 그 상수들을 덧씌워 검증한다.
    """
    import anthropic  # noqa: PLC0415

    from mssb_coder.claude_client import MODEL_VERSION, _require_api_key  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=_require_api_key())
    user_message = build_synthesis_user_message(stem_results)

    response = client.messages.create(
        model=MODEL_VERSION,
        max_tokens=4096,
        system=[{"type": "text", "text": SYNTHESIS_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[SYNTHESIS_TOOL],
        tool_choice={"type": "tool", "name": SYNTHESIS_TOOL["name"]},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise RuntimeError("Claude 응답에 tool_use 블록이 없음")
