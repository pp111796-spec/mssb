import json
from types import SimpleNamespace

import pytest

from mssb_coder.claude_client import (
    FALLBACK_BETA,
    MODEL_VERSION,
    REPORT_TONE_GUIDE,
    ClaudeCallError,
    build_stem_coding_system_prompt,
    build_stem_coding_user_message,
    call_structured,
    _require_api_key,
)
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.session_synthesis import SYNTHESIS_SYSTEM_PROMPT

_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}, "required": ["n"]}


class _FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class FakeClient:
    """client.beta.messages.stream(...)만 흉내 내는 가짜 클라이언트 — 보낸 인자를 기록한다."""

    def __init__(self, response):
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self._response = response

    def _stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._response)


def _response(text=None, stop_reason="end_turn", model=MODEL_VERSION, stop_details=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=50, cache_read_input_tokens=None, cache_creation_input_tokens=20
    )
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, model=model, usage=usage, stop_details=stop_details
    )


def _call(client):
    return call_structured(system_prompt="sys", user_message="msg", schema=_SCHEMA, call_name="테스트", client=client)


def test_call_structured_sends_opus55_structured_output_request():
    client = FakeClient(_response(json.dumps({"n": 3})))

    result = _call(client)

    assert result.data == {"n": 3}
    sent = client.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert "tool_choice" not in sent and "tools" not in sent and "thinking" not in sent
    assert sent["output_config"]["effort"] == "high"
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert "minimum" not in fmt["schema"]["properties"]["n"]
    assert sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_call_structured_reports_actual_model_and_usage():
    client = FakeClient(_response(json.dumps({"n": 1}), model="claude-opus-5"))

    result = _call(client)

    assert result.model == "claude-opus-5"  # fallback이 돈 경우 실제 모델이 기록돼야 함
    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 20,
    }


def test_call_structured_raises_on_refusal_with_category():
    client = FakeClient(_response(stop_reason="refusal", stop_details=SimpleNamespace(category="bio")))
    with pytest.raises(ClaudeCallError, match="bio"):
        _call(client)


def test_call_structured_raises_on_max_tokens():
    client = FakeClient(_response('{"n": ', stop_reason="max_tokens"))
    with pytest.raises(ClaudeCallError, match="잘렸"):
        _call(client)


@pytest.mark.parametrize("text", [None, "not json", "[1, 2]"])
def test_call_structured_rejects_missing_or_invalid_json(text):
    client = FakeClient(_response(text))
    with pytest.raises(ClaudeCallError):
        _call(client)


def test_both_report_prompts_carry_tone_guide():
    assert REPORT_TONE_GUIDE in build_stem_coding_system_prompt(load_coding_system())
    assert REPORT_TONE_GUIDE in SYNTHESIS_SYSTEM_PROMPT


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
