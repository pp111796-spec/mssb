from mssb_coder.coding_systems import load_coding_system
from mssb_coder.labels import scrub_english_ids
from mssb_coder.segmentation import load_story_stems


def test_scrub_english_ids_replaces_stem_and_category_ids():
    coding_system = load_coding_system()
    story_stems = load_story_stems()
    text = (
        "spilled_juice와 lost_dog에서 avoidant_withdrawal 양상이 반복됐다 "
        "(empathy_warmth는 두 이야기 모두에서 존재)."
    )
    scrubbed = scrub_english_ids(text, coding_system, story_stems)

    assert "spilled_juice" not in scrubbed
    assert "lost_dog" not in scrubbed
    assert "avoidant_withdrawal" not in scrubbed
    assert "empathy_warmth" not in scrubbed
    assert "엎지러진 주스" in scrubbed
    assert "뽀삐가 없어졌어요" in scrubbed
    assert "회피/위축" in scrubbed
    assert "공감/온화" in scrubbed


def test_scrub_english_ids_does_not_touch_ordinary_korean_text():
    coding_system = load_coding_system()
    story_stems = load_story_stems()
    text = "아이는 주스를 쏟은 뒤 당황했지만 금방 침착해졌다."

    assert scrub_english_ids(text, coding_system, story_stems) == text
