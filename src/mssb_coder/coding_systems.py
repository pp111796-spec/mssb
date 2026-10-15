"""Load config/coding_systems/mssb_5category.yaml and build the JSON Schema Claude must answer in.

파이프라인 9단계("코딩 체계 로드")와 10단계("AI 코딩 호출")가 쓰는 모듈.
YAML은 사람이 읽고 고치는 소스, 이 모듈이 그걸 JSON Schema로 변환한다 — 5범주 세부 항목 체크리스트를
스키마 하나로 정의해 스템당 1회 호출로 5범주 값을 동시에 받는다. 실제 호출은 structured outputs
(output_config.format)로 보내며, to_structured_output_schema가 그 형식에 맞게 다듬는다.
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


# structured outputs(output_config.format)가 받지 않는 제약 키워드 — 값 검증은 pydantic이 다시 한다.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength"}
)


def to_structured_output_schema(schema: dict) -> dict:
    """tool input_schema를 structured outputs용으로 바꾼다.

    모든 object에 additionalProperties:false를 붙이고 지원하지 않는 제약 키워드를 뺀다.
    description은 남긴다 — rationale의 문체 지시처럼 모델이 읽어야 하는 안내가 들어 있다.
    """

    def convert(node, is_properties_map: bool = False):
        if isinstance(node, list):
            return [convert(v) for v in node]
        if not isinstance(node, dict):
            return node
        if is_properties_map:  # 키가 사용자 정의 속성 이름이라 키워드로 걸러내면 안 된다
            return {name: convert(sub) for name, sub in node.items()}
        out = {
            key: convert(value, is_properties_map=(key == "properties"))
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out

    return convert(schema)


def build_claude_tool_schema(coding_system: CodingSystem) -> dict:
    """스템당 1회 Claude 호출의 응답 스키마 정의 (claude_client.py가 input_schema를 사용).

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
                "rationale": {
                    "type": "string",
                    "description": "이 범주의 판단 근거를 하다체 서술문 1~3문장으로. 메모체·코드명·항목 개수 표현 금지",
                },
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
