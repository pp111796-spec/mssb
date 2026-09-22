from mssb_coder.schema import ExaminerSegment, StemTimestamp
from mssb_coder.transcribe import TranscriptSegment, tag_speakers


def test_tag_speakers_splits_by_handoff():
    stem = StemTimestamp(
        stem_name="spilled_juice",
        start_s=0.0,
        end_s=30.0,
        handoff_timestamp_s=10.0,
        confidence=0.9,
        matched_cue="완성요청 문구",
    )
    transcript = [
        TranscriptSegment(start_s=1.0, end_s=9.0, text="검사자 대사"),
        TranscriptSegment(start_s=12.0, end_s=20.0, text="아동 발화"),
    ]
    tagged = tag_speakers(transcript, stem)
    assert [s.text for s in tagged["examiner"]] == ["검사자 대사"]
    assert [s.text for s in tagged["child"]] == ["아동 발화"]


def test_tag_speakers_handles_additional_examiner_segment():
    stem = StemTimestamp(
        stem_name="lost_dog",
        start_s=0.0,
        end_s=60.0,
        handoff_timestamp_s=10.0,
        additional_examiner_segments_s=[ExaminerSegment(start_s=30.0, end_s=35.0)],
        confidence=0.8,
        matched_cue="완성요청 문구",
    )
    transcript = [
        TranscriptSegment(start_s=12.0, end_s=20.0, text="아동: 뽀삐가 사라졌어요"),
        TranscriptSegment(start_s=31.0, end_s=34.0, text="검사자: 뽀삐가 다시 돌아왔어"),
        TranscriptSegment(start_s=36.0, end_s=40.0, text="아동: 그래서 안아줬어요"),
    ]
    tagged = tag_speakers(transcript, stem)
    assert [s.text for s in tagged["child"]] == ["아동: 뽀삐가 사라졌어요", "아동: 그래서 안아줬어요"]
    assert [s.text for s in tagged["examiner"]] == ["검사자: 뽀삐가 다시 돌아왔어"]


def test_tag_speakers_excludes_segments_outside_stem_range():
    stem = StemTimestamp(
        stem_name="spilled_juice",
        start_s=10.0,
        end_s=20.0,
        handoff_timestamp_s=15.0,
        confidence=0.9,
        matched_cue="완성요청 문구",
    )
    transcript = [TranscriptSegment(start_s=100.0, end_s=105.0, text="다른 스템 발화")]
    tagged = tag_speakers(transcript, stem)
    assert tagged["examiner"] == []
    assert tagged["child"] == []
