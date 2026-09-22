import pytest

from mssb_coder.bundle import BundleParseError, build_bundle_dict, parse_bundle_dict
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.schema import (
    CategoryCode,
    CompletionStatus,
    ItemCode,
    ResponseMode,
    SessionSynthesis,
    StemCodingResult,
)


def _stem_result() -> StemCodingResult:
    coding_system = load_coding_system()
    codes = {}
    for category in coding_system.categories:
        items = {item.id: ItemCode(present=False, evidence=[]) for item in category.items}
        codes[category.id] = CategoryCode(items=items, present_count=0, rationale="테스트")
    return StemCodingResult(
        session_id="s1",
        child_age_months=54,
        stem_name="spilled_juice",
        completion_status=CompletionStatus.COMPLETE,
        response_mode=ResponseMode.SPONTANEOUS,
        codes=codes,
        story_specific_notes_applied="—",
        model_version="claude-opus-5",
    )


def _synthesis() -> SessionSynthesis:
    return SessionSynthesis(
        session_id="s1",
        stems_included=["spilled_juice"],
        stems_administered_count=1,
        recurring_patterns=[],
        model_version="claude-opus-5",
    )


def test_build_and_parse_bundle_round_trip():
    bundle = build_bundle_dict(_synthesis(), [_stem_result()])
    synthesis, stem_results = parse_bundle_dict(bundle)
    assert synthesis.session_id == "s1"
    assert len(stem_results) == 1
    assert stem_results[0].stem_name == "spilled_juice"


def test_parse_bundle_missing_synthesis_raises():
    with pytest.raises(BundleParseError):
        parse_bundle_dict({"stem_results": []})


def test_parse_bundle_invalid_synthesis_raises():
    with pytest.raises(BundleParseError):
        parse_bundle_dict({"synthesis": {"not": "valid"}})


def test_parse_bundle_no_stem_results_key_defaults_to_empty():
    bundle = {"synthesis": _synthesis().model_dump(mode="json")}
    synthesis, stem_results = parse_bundle_dict(bundle)
    assert stem_results == []
