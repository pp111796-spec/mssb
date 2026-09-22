"""세션 1건 end-to-end 실행 — 계획 "파이프라인 단계" 1~13단계를 그대로 코드로 옮긴 것.

각 단계의 실제 구현 상태 (README.md "지금 상태" 절과 함께 최신으로 유지할 것):
  - 오디오/프레임 추출(ffmpeg), 전사(faster-whisper), 자세(MediaPipe), 음성운율(openSMILE),
    Claude 3개 호출(구간분리·스템채점·세션종합) — 실제 구현됨. ANTHROPIC_API_KEY만 있으면 동작.
  - 인형/소품 추적(YOLO) — 코드는 실제 구현됐지만, 마일스톤 0(라벨링·파인튜닝)이 없으면
    가중치 파일이 없어 빈 결과를 반환함(의도된 동작 — 가짜 탐지보다 나음).
  - 표정 분석(OpenFace/PyAFAR/Py-Feat) — 아직 미구현.

4단계(사람 빠른 확인)는 이 파일이 하지 않는다 — Streamlit 앱(src/review_app/app.py)의 몫이다.
그래서 run_pipeline은 "3단계까지 자동 실행 후 저장"과 "확정된 stem_timestamps.json을 입력으로
5단계부터 재개"를 별도 함수로 나눈다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from mssb_coder import storage, video_prep
from mssb_coder.claude_client import call_stem_coding
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.facial_expression import run_facial_expression
from mssb_coder.feature_summary import summarize_stem_features
from mssb_coder.object_tracking import run_object_tracking
from mssb_coder.pose_analysis import run_pose_analysis
from mssb_coder.posture_metrics import compute_posture_events
from mssb_coder.schema import SessionMetadata, SessionSynthesis, StemCodingResult, StemTimestamp
from mssb_coder.segmentation import call_segmentation, load_story_stems
from mssb_coder.session_synthesis import call_session_synthesis
from mssb_coder.transcribe import tag_speakers, transcribe_session
from mssb_coder.voice_prosody import run_voice_prosody

logger = logging.getLogger("mssb_coder.pipeline")

CV_THRESHOLDS_DIR = Path(__file__).resolve().parents[2] / "config"


def _load_cv_thresholds(subject_type) -> dict:
    filename = f"cv_thresholds_{subject_type.value}.yaml"
    return yaml.safe_load((CV_THRESHOLDS_DIR / filename).read_text(encoding="utf-8"))


def run_segmentation_stage(video_path: Path, metadata: SessionMetadata) -> list[StemTimestamp]:
    """1~3단계: 오디오 추출 → 전체 전사 → Claude 구간분리. stem_timestamps.json에 저장."""
    logger.info("2단계: 오디오 추출 + 전체 전사")
    audio_path = video_prep.extract_audio(video_path, storage.session_dir(metadata.session_id) / "audio.wav")
    transcript = transcribe_session(audio_path)
    full_text = "\n".join(f"[{seg.start_s:.1f}s] {seg.text}" for seg in transcript)

    logger.info("3단계: Claude 구간분리")
    story_stems = load_story_stems()
    raw_segments = call_segmentation(story_stems, full_text)
    timestamps = [StemTimestamp.model_validate(seg) for seg in raw_segments]

    storage.save_stem_timestamps(metadata.session_id, timestamps)
    logger.info("stem_timestamps.json 저장 완료 — 4단계(사람 확인)는 Streamlit에서 진행할 것")
    return timestamps


def run_coding_stage(video_path: Path, metadata: SessionMetadata) -> list[StemCodingResult]:
    """5~11단계: 사람이 확정한 stem_timestamps.json을 입력으로, 스템별 AI 채점까지 자동 진행.

    이 단계까지는 사람이 개입하지 않는다 (계획 파이프라인 10단계 설명 참고).
    """
    timestamps = storage.load_stem_timestamps(metadata.session_id)
    cv_thresholds = _load_cv_thresholds(metadata.subject_type)
    if cv_thresholds["posture"]["torso_tilt_avoidance_deg"] is None:
        logger.warning(
            "cv_thresholds_%s.yaml의 posture 임계값이 아직 null임 — "
            "자세 이벤트(torso_tilt/sudden_movement) 없이 진행 (문제점.md 3번 참고)",
            metadata.subject_type.value,
        )
    coding_system = load_coding_system()
    story_stems = {s.id: s for s in load_story_stems().stems}

    session_audio_path = storage.session_dir(metadata.session_id) / "audio.wav"
    transcript = transcribe_session(session_audio_path)

    results: list[StemCodingResult] = []
    for stem in timestamps:
        if not stem.scored:
            continue
        logger.info("스템 %s 처리 시작", stem.stem_name)

        stem_frames_dir = storage.session_dir(metadata.session_id) / "frames" / stem.stem_name
        video_prep.extract_stem_frames(video_path, stem, stem_frames_dir.parent)
        child_audio_path = video_prep.extract_stem_audio(
            video_path, stem, storage.session_dir(metadata.session_id) / "audio_segments"
        )

        facial_frames = run_facial_expression(stem_frames_dir, metadata.subject_type)
        pose_frames = run_pose_analysis(stem_frames_dir)
        posture_events = compute_posture_events(
            pose_frames,
            torso_tilt_avoidance_deg=cv_thresholds["posture"]["torso_tilt_avoidance_deg"],
            sudden_movement_threshold=cv_thresholds["posture"]["sudden_movement_threshold"],
        )
        doll_frames = run_object_tracking(
            stem_frames_dir, Path(__file__).resolve().parents[2] / "models" / "doll_yolo"
        )
        prosody_frames = run_voice_prosody(child_audio_path)

        nonverbal_summary = summarize_stem_features(
            stem.stem_name, facial_frames, posture_events, doll_frames, prosody_frames
        )

        speaker_tagged = tag_speakers(transcript, stem)
        transcript_text = "\n".join(seg.text for seg in speaker_tagged["child"])

        stem_def = story_stems[stem.stem_name]
        raw_result = call_stem_coding(
            coding_system,
            transcript_text=transcript_text,
            nonverbal_summary=nonverbal_summary,
            story_specific_notes=stem_def.story_specific_notes,
            child_age_months=metadata.child_age_months,
            response_mode="spontaneous",  # TODO: 실제로는 사람이 기록해야 함 (§6.6)
            issue_prompt_count=0,  # TODO: 위와 동일
        )
        result = StemCodingResult.model_validate(
            raw_result
            | {
                "session_id": metadata.session_id,
                "child_age_months": metadata.child_age_months,
                "stem_name": stem.stem_name,
                "completion_status": "complete",  # TODO: 실제로는 사람이 기록해야 함
                "response_mode": "spontaneous",
                "issue_prompt_count": 0,
                "model_version": "claude-opus-5",
            }
        )
        storage.save_stem_coding(metadata.session_id, stem.stem_name, result)
        results.append(result)

    return results


def run_synthesis_stage(metadata: SessionMetadata) -> SessionSynthesis:
    """12단계: 스템별 AI 채점 원본 전체 → 세션 종합 패턴 초안."""
    stem_results = storage.load_all_stem_codings(metadata.session_id)  # 원본(ai_codes) 사용
    raw_synthesis = call_session_synthesis(stem_results)
    synthesis = SessionSynthesis.model_validate(
        raw_synthesis | {"session_id": metadata.session_id, "model_version": "claude-opus-5"}
    )
    storage.save_session_synthesis(metadata.session_id, synthesis)
    return synthesis


def run_pipeline(video_path: Path, metadata: SessionMetadata) -> SessionSynthesis:
    """전체 1~13단계 실행 (4단계 사람 확인 제외 — 그 지점은 CLI/Streamlit이 일시정지해야 함)."""
    run_segmentation_stage(video_path, metadata)
    logger.warning(
        "구간분리 결과를 Streamlit에서 확인한 뒤 run_coding_stage()부터 이어서 실행하세요."
    )
    run_coding_stage(video_path, metadata)
    return run_synthesis_stage(metadata)
