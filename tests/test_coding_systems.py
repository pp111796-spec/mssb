from mssb_coder.coding_systems import build_claude_tool_schema, load_coding_system, to_structured_output_schema
from mssb_coder.schema import CategoryId
from mssb_coder.segmentation import SEGMENTATION_TOOL
from mssb_coder.session_synthesis import SYNTHESIS_TOOL
import typing

import pytest

_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength"}


def _walk_schema_nodes(node):
    """properties 맵의 키(속성 이름)는 건너뛰고 스키마 노드만 순회한다."""
    if isinstance(node, list):
        for v in node:
            yield from _walk_schema_nodes(v)
    elif isinstance(node, dict):
        yield node
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    yield from _walk_schema_nodes(sub)
            else:
                yield from _walk_schema_nodes(value)


def _all_input_schemas():
    return {
        "stem_coding": build_claude_tool_schema(load_coding_system())["input_schema"],
        "segmentation": SEGMENTATION_TOOL["input_schema"],
        "synthesis": SYNTHESIS_TOOL["input_schema"],
    }


@pytest.mark.parametrize("name", ["stem_coding", "segmentation", "synthesis"])
def test_structured_output_schema_is_accepted_shape(name):
    converted = to_structured_output_schema(_all_input_schemas()[name])
    for node in _walk_schema_nodes(converted):
        assert not (_UNSUPPORTED & node.keys()), f"{name}: 지원 안 되는 키워드 {_UNSUPPORTED & node.keys()}"
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False


def test_structured_output_schema_keeps_descriptions_and_property_names():
    converted = to_structured_output_schema(_all_input_schemas()["stem_coding"])
    category = converted["properties"]["codes"]["properties"]["empathy_warmth"]
    assert "하다체" in category["properties"]["rationale"]["description"]
    assert "present_count" in category["properties"]
    assert category["required"] == ["items", "present_count", "rationale"]


def test_structured_output_schema_does_not_mutate_input():
    original = SYNTHESIS_TOOL["input_schema"]
    to_structured_output_schema(original)
    assert "additionalProperties" not in original
    assert original["properties"]["stems_administered_count"]["minimum"] == 0


def test_load_real_coding_system_matches_schema_category_ids():
    coding_system = load_coding_system()
    expected_ids = set(typing.get_args(CategoryId))
    assert set(coding_system.category_ids()) == expected_ids


def test_load_real_coding_system_has_items_per_category():
    coding_system = load_coding_system()
    for category in coding_system.categories:
        assert len(category.items) >= 1


def test_build_claude_tool_schema_structure():
    coding_system = load_coding_system()
    tool = build_claude_tool_schema(coding_system)

    assert tool["name"] == "submit_stem_coding"
    codes_props = tool["input_schema"]["properties"]["codes"]["properties"]
    assert set(codes_props.keys()) == set(coding_system.category_ids())

    empathy = codes_props["empathy_warmth"]
    item_props = empathy["properties"]["items"]["properties"]
    assert "sharing" in item_props
    assert item_props["sharing"]["properties"]["present"]["type"] == "boolean"
