from mssb_coder.coding_systems import build_claude_tool_schema, load_coding_system
from mssb_coder.schema import CategoryId
import typing


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
