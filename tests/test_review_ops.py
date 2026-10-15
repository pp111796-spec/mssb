import pytest

from factories import DEFAULT_STEMS, make_session, stem_result, stem_timestamp
from mssb_coder import storage
from mssb_coder.review_ops import session_progress


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MSSB_DATA_DIR", str(tmp_path))


def test_progress_before_segmentation():
    storage.session_dir("s1")
    assert session_progress("s1").label == "구간분리 전"


def test_progress_needs_confirmation_then_tracks_changes():
    make_session(confirm=False, coded=False, synthesized=False)
    assert session_progress("s1").label == "구간 확인 필요"

    storage.confirm_segmentation("s1")
    assert session_progress("s1").label == "채점 진행 중 (0/2)"

    storage.save_stem_timestamps("s1", [DEFAULT_STEMS[0], stem_timestamp("spilled_juice", 20.0, 31.0, 60.0), DEFAULT_STEMS[2]])
    assert session_progress("s1").label == "구간이 바뀜 — 다시 확정 필요"


def test_progress_through_synthesis_and_review():
    make_session(synthesized=False)
    assert session_progress("s1").label == "종합 전"

    make_session()
    assert session_progress("s1").label == "종합 완료 — 검토 전"

    storage.save_session_synthesis("s1", storage.load_session_synthesis("s1"), reviewed=True)
    assert session_progress("s1").label == "검토 저장됨"


def test_effective_codings_prefer_reviewed_and_skip_stale_stems():
    make_session()
    reviewed = stem_result("lost_dog", rationale="검토자가 고친 문장이다.")
    storage.save_stem_coding("s1", "lost_dog", reviewed, reviewed=True)
    storage.save_stem_coding("s1", "candy_theft", stem_result("candy_theft"))  # 현재 구간에 없는 예전 파일

    results = storage.load_effective_stem_codings("s1")

    assert [r.stem_name for r in results] == ["spilled_juice", "lost_dog"]
    assert results[1].codes["empathy_warmth"].rationale == "검토자가 고친 문장이다."


# --- 구간 표 검증 ------------------------------------------------------------------

from mssb_coder.review_ops import transcript_preview, validate_stem_timestamps  # noqa: E402
from mssb_coder.segmentation import load_story_stems  # noqa: E402
from mssb_coder.transcribe import TranscriptSegment  # noqa: E402


def _row(name, start, handoff, end, **extra):
    return {"stem_name": name, "start_s": start, "handoff_timestamp_s": handoff, "end_s": end, **extra}


def _validate(rows, examiner=None):
    return validate_stem_timestamps(rows, examiner or {}, load_story_stems())


def test_valid_rows_are_sorted_and_take_scored_from_yaml():
    timestamps, errors = _validate([
        _row("spilled_juice", 20, 30, 60, confidence=0.4, matched_cue="문구"),
        _row("warm_up", 5, 12, 20),
        _row("pre_session", 0, 5, 5),  # 아동 차례 없는 여백 구간은 허용
    ])

    assert errors == []
    assert [(t.stem_name, t.scored) for t in timestamps] == [
        ("pre_session", False), ("warm_up", False), ("spilled_juice", True),
    ]
    assert timestamps[1].matched_cue == "사람이 추가함" and timestamps[1].confidence == 1.0
    assert timestamps[2].confidence == 0.4


def test_overlap_duplicate_and_missing_values_are_reported_in_korean():
    _, errors = _validate([
        _row("warm_up", 0, 10, 25),
        _row("spilled_juice", 20, 30, 60),
        _row("lost_dog", 60, None, 100),
        _row("lost_dog", 100, 110, 120),
        _row("", 1, 2, 3),
    ])

    joined = "\n".join(errors)
    assert "겹칩니다" in joined
    assert "2번 들어 있습니다" in joined
    assert "숫자로 입력" in joined
    assert "이야기를 고르세요" in joined


def test_rule_violation_message_comes_from_schema():
    _, errors = _validate([_row("spilled_juice", 20, 60, 60)])
    assert len(errors) == 1
    assert errors[0].endswith("채점 대상 이야기인데 아동 차례가 없음 (전환 시각 = 끝 시각)")
    assert not errors[0].startswith("spilled_juice")  # 영문 id가 아니라 한글 이야기 이름으로 안내


def test_examiner_segments_must_fall_inside_the_stem():
    _, errors = _validate([_row("lost_dog", 60, 70, 100)], {"lost_dog": [(95.0, 105.0)]})
    assert errors and "벗어남" in errors[0]


def test_transcript_preview_labels_speakers_by_current_boundaries():
    stem = stem_timestamp("spilled_juice", 20.0, 30.0, 60.0)
    transcript = [
        TranscriptSegment(4.0, 5.0, "너무 이른 말"),  # 시작(20) - 여백(15) = 5초보다 앞
        TranscriptSegment(10.0, 11.0, "앞 여백"),
        TranscriptSegment(25.0, 26.0, "자 이제 네 차례야"),
        TranscriptSegment(35.0, 36.0, "주스를 닦았어요"),
        TranscriptSegment(65.0, 66.0, "뒤 여백"),
    ]

    lines = transcript_preview(transcript, stem, margin_s=15.0)

    assert [(l.speaker, l.text) for l in lines] == [
        ("구간 밖", "앞 여백"), ("검사자", "자 이제 네 차례야"), ("아동", "주스를 닦았어요"), ("구간 밖", "뒤 여백"),
    ]


def test_preview_clip_command_cuts_only_the_window_at_low_resolution(tmp_path):
    from mssb_coder.video_prep import preview_clip_command

    cmd = preview_clip_command("ffmpeg", tmp_path / "v.mp4", 10.0, 70.0, tmp_path / "c.mp4")

    assert cmd[cmd.index("-ss") + 1] == "10.00"
    assert cmd[cmd.index("-t") + 1] == "60.00"
    assert "scale=-2:360" in cmd


def test_parse_and_format_examiner_segments():
    from mssb_coder.review_ops import format_examiner_segments, parse_examiner_segments

    assert parse_examiner_segments("120-128; 95~105.5") == ([(95.0, 105.5), (120.0, 128.0)], None)
    assert parse_examiner_segments("") == ([], None)
    assert parse_examiner_segments("abc-5")[1] is not None
    assert parse_examiner_segments("10-5")[1] is not None
    assert format_examiner_segments([(95.0, 105.5)]) == "95.0-105.5"


def test_apply_stem_edits_recounts_and_drops_evidence():
    from mssb_coder.review_ops import StemEdits, apply_stem_edits
    from mssb_coder.schema import CompletionStatus, ResponseMode

    result = stem_result("spilled_juice")
    first_item = next(iter(result.codes["empathy_warmth"].items))
    second_item = list(result.codes["empathy_warmth"].items)[1]
    edits = StemEdits(
        completion_status=CompletionStatus.PARTIAL,
        response_mode=ResponseMode.PROMPTED,
        issue_prompt_count=2,
        present={("empathy_warmth", second_item): True},
        rationale={"empathy_warmth": "  아이는 인물을 도왔다.  "},
        excluded_evidence={("empathy_warmth", first_item, 0)},
    )

    edited = apply_stem_edits(result, edits)

    category = edited.codes["empathy_warmth"]
    assert category.present_count == 2  # 원래 존재 1개 + 새로 체크 1개
    assert category.items[first_item].evidence == []
    assert category.rationale == "아이는 인물을 도왔다."
    assert (edited.completion_status, edited.response_mode, edited.issue_prompt_count) == (
        CompletionStatus.PARTIAL, ResponseMode.PROMPTED, 2,
    )
    assert result.codes["empathy_warmth"].present_count == 1  # 원본은 그대로


def test_can_play_evidence_only_for_session_time_inside_window():
    from mssb_coder.review_ops import can_play_evidence

    window = stem_timestamp("spilled_juice", 20.0, 30.0, 60.0)
    new_result = stem_result("spilled_juice", window=window)
    old_result = stem_result("spilled_juice")

    assert can_play_evidence(new_result, window, 35.0)
    assert not can_play_evidence(new_result, window, 0.0)
    assert not can_play_evidence(old_result, window, 35.0)
    assert not can_play_evidence(new_result, None, 35.0)
