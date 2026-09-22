import pytest

from mssb_coder.claude_client import (
    build_stem_coding_system_prompt,
    build_stem_coding_user_message,
    _require_api_key,
)
from mssb_coder.coding_systems import load_coding_system


def test_system_prompt_includes_all_categories_and_disclaimer():
    coding_system = load_coding_system()
    prompt = build_stem_coding_system_prompt(coding_system)
    for category in coding_system.categories:
        assert category.display_name_ko in prompt
    assert "진단" in prompt


def test_user_message_includes_all_fields():
    message = build_stem_coding_user_message(
        transcript_text="주스를 다시 채웠어요.",
        nonverbal_summary="[12.0s] (posture) torso_tilt=40.0",
        story_specific_notes="공감/순응은 이 이야기에서 긍정적 신호.",
        child_age_months=54,
        response_mode="spontaneous",
        issue_prompt_count=0,
    )
    assert "54개월" in message
    assert "주스를 다시 채웠어요" in message
    assert "torso_tilt" in message


def test_require_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _require_api_key()


def test_require_api_key_returns_value(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert _require_api_key() == "sk-test-123"
