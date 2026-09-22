"""Load config/coding_systems/mssb_5category.yaml and build the Claude tool_use schema for it.

파이프라인 9단계("코딩 체계 로드")와 10단계("AI 코딩 호출")가 쓰는 모듈.
YAML은 사람이 읽고 고치는 소스, 이 모듈이 그걸 Anthropic tool_use의 input_schema(JSON Schema)로
변환한다 — 5범주 세부 항목 체크리스트를 하나의 tool로 정의해 스템당 1회 호출로 5범주 값을 동시에 받는다
(계획 "기술 스택" 섹션의 "구조화 출력" 항목).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CODING_SYSTEM_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "coding_systems" / "mssb_5category.yaml"
)


@dataclass(frozen=True)
class CodingItem:
    id: str
    display_name_ko: str


@dataclass(frozen=True)
class CodingCategory:
    id: str
    display_name_ko: str
    items: tuple[CodingItem, ...]


@dataclass(frozen=True)
class CodingSystem:
    interpretation_layer_ceiling: str
    categories: tuple[CodingCategory, ...]
    aggregation_method: str

    def category_ids(self) -> list[str]:
        return [c.id for c in self.categories]


def load_coding_system(path: Path | None = None) -> CodingSystem:
    path = path or DEFAULT_CODING_SYSTEM_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    categories = tuple(
        CodingCategory(
            id=cat_id,
            display_name_ko=cat["display_name_ko"],
            items=tuple(
                CodingItem(id=item["id"], display_name_ko=item["display_name_ko"])
                for item in cat["items"]
            ),
        )
        for cat_id, cat in raw["categories"].items()
    )

    return CodingSystem(
        interpretation_layer_ceiling=raw["interpretation_layer_ceiling"],
        categories=categories,
        aggregation_method=raw["aggregation"]["method"],
    )


def build_claude_tool_schema(coding_system: CodingSystem) -> dict:
    """스템당 1회 Claude 호출에 쓰이는 tool_use 정의 (claude_client.py가 사용).

    반환값의 input_schema.properties.codes는 schema.py의 StemCodingResult.codes와
    구조적으로 일치해야 한다 (pydantic 검증 통과가 이 스키마의 목적).
    """

    evidence_schema = {
        "type": "object",
        "properties": {
            "timestamp_s": {"type": "number"},
            "source": {
                "type": "string",
                "enum": ["transcript", "facial_au", "posture", "doll_tracking", "voice_prosody"],
            },
            "note": {"type": "string"},
        },
        "required": ["timestamp_s", "source"],
    }

    codes_properties: dict[str, dict] = {}
    for category in coding_system.categories:
        item_properties = {
            item.id: {
                "type": "object",
                "description": item.display_name_ko,
                "properties": {
                    "present": {"type": "boolean"},
                    "evidence": {"type": "array", "items": evidence_schema},
                },
                "required": ["present", "evidence"],
            }
            for item in category.items
        }
        codes_properties[category.id] = {
            "type": "object",
            "description": category.display_name_ko,
            "properties": {
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": [item.id for item in category.items],
                },
                "present_count": {"type": "integer", "minimum": 0},
                "rationale": {"type": "string"},
            },
            "required": ["items", "present_count", "rationale"],
        }

    return {
        "name": "submit_stem_coding",
        "description": (
            "MSSB 스템 1편의 5범주 세부 항목 체크리스트를 존재/부재(+근거)로 채점한다. "
            f"해석은 '{coding_system.interpretation_layer_ceiling}' 층위를 넘지 않는다 "
            "(진단·위험분류·임상판단 금지, 매뉴얼 §10.1)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "object",
                    "properties": codes_properties,
                    "required": coding_system.category_ids(),
                },
                "story_specific_notes_applied": {"type": "string"},
            },
            "required": ["codes", "story_specific_notes_applied"],
        },
    }
