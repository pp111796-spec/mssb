from mssb_coder.segmentation import (
    build_segmentation_system_prompt,
    build_segmentation_user_message,
    load_fixed_order,
    load_segmentation_examples,
    load_story_stems,
)


def test_load_real_story_stems_has_12_entries_in_order():
    config = load_story_stems()
    assert len(config.stems) == 12
    assert config.stems[0].id == "warm_up"
    assert config.stems[0].scored is False
    assert config.stems[-1].id == "wind_down"
    assert config.stems[-1].scored is False
    scored_ids = [s.id for s in config.stems if s.scored]
    assert len(scored_ids) == 10
    assert "reunion" in scored_ids
    assert "lost_dog" in scored_ids


def test_completion_request_phrase_present():
    config = load_story_stems()
    assert "인형을 잡고 보여줄래" in config.completion_request_phrase


def test_build_system_prompt_includes_all_stems():
    config = load_story_stems()
    prompt = build_segmentation_system_prompt(config)
    for stem in config.stems:
        assert stem.id in prompt


def test_build_user_message_includes_transcript_and_fixed_order():
    message = build_segmentation_user_message(
        "검사자: 안녕 (주인공)아...", fixed_order=["spilled_juice", "lost_dog"]
    )
    assert "안녕" in message
    assert "spilled_juice" in message


def test_load_fixed_order_defaults_to_empty():
    assert load_fixed_order() == []


def test_load_segmentation_examples_missing_file_returns_empty(tmp_path):
    assert load_segmentation_examples(tmp_path / "does_not_exist.yaml") == []
