"""전체 전사문 → Claude(텍스트만)로 스템 구간 자동 분리·라벨링 (파이프라인 3단계).

핵심 앵커는 완성요청 고정 문구("자 이제 OO차례야...")다. 화자 태깅(검사자/아동 구분)은
여기서 만드는 handoff_timestamp_s / additional_examiner_segments_s를 transcribe.py가
그대로 갖다 쓴다 — 별도 문자열 매칭 경로를 두지 않는다 (계획 "음성 — 화자 태깅" 섹션의
설계 수정 이력 참고: 별도 매칭이 오히려 이 단계보다 더 취약했음).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_STORY_STEMS_PATH = Path(__file__).resolve().parents[2] / "config" / "story_stems.yaml"
DEFAULT_STORY_ORDER_PATH = Path(__file__).resolve().parents[2] / "config" / "story_order.yaml"
DEFAULT_SEGMENTATION_EXAMPLES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "segmentation_examples.yaml"
)


@dataclass(frozen=True)
class StemDefinition:
    id: str
    display_name_ko: str
    scored: bool
    keywords: tuple[str, ...]
    story_specific_notes: str


@dataclass(frozen=True)
class StoryStemsConfig:
    completion_request_phrase: str
    stems: tuple[StemDefinition, ...]  # 워밍업 → 10편 → 윈드다운 순서


def load_story_stems(path: Path | None = None) -> StoryStemsConfig:
    raw = yaml.safe_load((path or DEFAULT_STORY_STEMS_PATH).read_text(encoding="utf-8"))

    def _to_def(entry: dict) -> StemDefinition:
        return StemDefinition(
            id=entry["id"],
            display_name_ko=entry["display_name_ko"],
            scored=entry.get("scored", True),
            keywords=tuple(entry.get("keywords", [])),
            story_specific_notes=str(entry.get("story_specific_notes", "—")).strip(),
        )

    stems = [_to_def(raw["warm_up"])]
    stems.extend(_to_def(s) for s in raw["stems"])
    stems.append(_to_def(raw["wind_down"]))

    return StoryStemsConfig(
        completion_request_phrase=raw["completion_request_phrase"],
        stems=tuple(stems),
    )


def load_fixed_order(path: Path | None = None) -> list[str]:
    raw = yaml.safe_load((path or DEFAULT_STORY_ORDER_PATH).read_text(encoding="utf-8"))
    return raw.get("fixed_order", []) or []


def load_segmentation_examples(path: Path | None = None) -> list[dict]:
    """사람이 검토 화면에서 handoff 지점을 고친 사례들 — few-shot 예시로 프롬프트에 포함.

    파일이 아직 없으면(첫 세션) 빈 리스트 — 통계적 재학습이 아니라 점진적으로 쌓이는 예시 목록.
    """
    p = path or DEFAULT_SEGMENTATION_EXAMPLES_PATH
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return raw.get("examples", []) if raw else []


def build_segmentation_system_prompt(story_stems: StoryStemsConfig) -> str:
    stem_lines = [
        f"- {s.id} ({s.display_name_ko}){' [채점 안 함]' if not s.scored else ''}: "
        f"키워드 {', '.join(s.keywords) or '(없음)'}"
        for s in story_stems.stems
    ]
    return (
        "당신은 MSSB 검사 세션 전체 전사문을, 워밍업 → 10개 이야기 스템 → 윈드다운 구간으로 "
        "자동 분리하는 도구입니다.\n\n"
        f"모든 구간은 다음 완성요청 고정 문구로 끝납니다 (주 앵커): "
        f"\"{story_stems.completion_request_phrase}\" (OO는 실제 아동 이름으로 치환되어 있을 수 있음)\n\n"
        "이야기 목록과 키워드:\n" + "\n".join(stem_lines) + "\n\n"
        "각 구간에 대해 다음을 반환하세요: stem_name, start_s, end_s, "
        "handoff_timestamp_s(그 구간 안에서 검사자의 완성요청이 끝나고 아이 차례가 시작되는 시점), "
        "additional_examiner_segments_s(아이 차례 도중 검사자가 다시 끼어드는 구간이 있으면, "
        "대부분은 빈 배열), confidence(0~1), matched_cue(어떤 문구로 이 구간을 찾았는지).\n\n"
        "녹화 시작 시점이 아동 입장보다 먼저일 수 있습니다 — 세션 맨 앞의 여백은 워밍업 이전 구간으로 "
        "별도 처리하고 스템으로 포함하지 마세요."
    )


def build_segmentation_user_message(
    full_transcript_text: str,
    fixed_order: list[str] | None = None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    parts = []
    if fixed_order:
        parts.append(f"이 기관의 고정 실시 순서(참고용 힌트): {', '.join(fixed_order)}")
    if few_shot_examples:
        parts.append("과거 세션에서 사람이 수정한 handoff 판단 예시 (참고용):")
        for ex in few_shot_examples:
            parts.append(f"  - {ex}")
    parts.append(f"--- 전체 세션 전사문 ---\n{full_transcript_text}")
    return "\n\n".join(parts)


SEGMENTATION_TOOL = {
    "name": "submit_segmentation",
    "description": "전체 세션 전사문을 워밍업·10개 이야기 스템·윈드다운 구간으로 분리한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stem_name": {"type": "string"},
                        "start_s": {"type": "number"},
                        "end_s": {"type": "number"},
                        "handoff_timestamp_s": {"type": "number"},
                        "additional_examiner_segments_s": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_s": {"type": "number"},
                                    "end_s": {"type": "number"},
                                },
                                "required": ["start_s", "end_s"],
                            },
                        },
                        "confidence": {"type": "number"},
                        "matched_cue": {"type": "string"},
                        "scored": {"type": "boolean"},
                    },
                    "required": [
                        "stem_name", "start_s", "end_s", "handoff_timestamp_s",
                        "confidence", "matched_cue", "scored",
                    ],
                },
            }
        },
        "required": ["segments"],
    },
}


def call_segmentation(story_stems: StoryStemsConfig, full_transcript_text: str) -> list[dict]:
    """Claude를 호출해 구간별 dict 목록을 반환한다. 호출 쪽이 schema.StemTimestamp로 검증할 것."""
    import anthropic  # noqa: PLC0415

    from mssb_coder.claude_client import MODEL_VERSION, _require_api_key  # noqa: PLC0415

    client = anthropic.Anthropic(api_key=_require_api_key())
    system_prompt = build_segmentation_system_prompt(story_stems)
    fixed_order = load_fixed_order()
    few_shot_examples = load_segmentation_examples()
    user_message = build_segmentation_user_message(
        full_transcript_text, fixed_order=fixed_order, few_shot_examples=few_shot_examples
    )

    response = client.messages.create(
        model=MODEL_VERSION,
        max_tokens=4096,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        tools=[SEGMENTATION_TOOL],
        tool_choice={"type": "tool", "name": SEGMENTATION_TOOL["name"]},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input["segments"]

    raise RuntimeError("Claude 응답에 tool_use 블록이 없음")
