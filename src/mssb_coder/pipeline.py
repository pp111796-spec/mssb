"""세션 1건 end-to-end 실행 — 계획 "파이프라인 단계" 1~13단계를 그대로 코드로 옮긴 것.

각 단계의 실제 구현 상태 (README.md "지금 상태" 절과 함께 최신으로 유지할 것):
  - 오디오/프레임 추출(ffmpeg), 전사(faster-whisper), 자세(MediaPipe), 음성운율(openSMILE),
    Claude 3개 호출(구간분리·스템채점·세션종합) — 실제 구현됨. ANTHROPIC_API_KEY만 있으면 동작.
  - 인형/소품 추적(YOLO) — 코드는 실제 구현됐지만, 마일스톤 0(라벨링·파인튜닝)이 없으면
    가중치 파일이 없어 빈 결과를 반환함(의도된 동작 — 가짜 탐지보다 나음).
  - 표정 분석(Py-Feat) — 코드는 있으나 실제 아동 얼굴로는 아직 검증 전.

4단계(사람 빠른 확인)는 이 파일이 하지 않는다 — Streamlit 앱(src/review_app/app.py)의 몫이다.
그래서 run_pipeline은 "3단계까지 자동 실행 후 저장"과 "확정된 stem_timestamps.json을 입력으로
5단계부터 재개"를 별도 함수로 나눈다.

5~7단계(CV 분석)는 run_cv_stage로 따로 돌릴 수 있다 — Claude를 호출하지 않고 원본 수치를
cv/<stem>_cv.json에 저장하므로, 임계값을 바꿔가며 보정(scripts/calibrate_cv.py)할 때 비용이 들지 않는다.
run_coding_stage는 저장된 타임라인이 있으면 재사용한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from mssb_coder import storage, video_prep
from mssb_coder.claude_client import call_stem_coding
from mssb_coder.coding_systems import load_coding_system
from mssb_coder.cv_events import CvThresholdSet, compute_baselines, derive_events
from mssb_coder.cv_timeline import StemCvFrames, from_timeline, to_timeline
from mssb_coder.facial_expression import run_facial_expression
from mssb_coder.feature_summary import summarize_stem_features
from mssb_coder.object_tracking import run_object_tracking
from mssb_coder.pose_analysis import run_pose_analysis
from mssb_coder.schema import (
    SessionMetadata,
    SessionSynthesis,
    StemCodingResult,
    StemTimestamp,
    StemVerbatimRecord,
)
from mssb_coder.segmentation import call_segmentation, load_story_stems
from mssb_coder.session_synthesis import call_session_synthesis
from mssb_coder.transcribe import tag_speakers, transcribe_session
from mssb_coder.verbatim import build_stem_verbatim
from mssb_coder.voice_prosody import run_voice_prosody

logger = logging.getLogger("mssb_coder.pipeline")

CV_THRESHOLDS_DIR = Path(__file__).resolve().parents[2] / "config"
DOLL_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "models" / "doll_yolo"
FRAME_FPS = 2.0  # video_prep.extract_stem_frames / pose / facial / object_tracking이 모두 같은 값을 써야 함


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


def _ensure_stem_media(
    video_path: Path | None, metadata: SessionMetadata, stem: StemTimestamp
) -> tuple[Path, Path]:
    """스템의 프레임 폴더·아동 오디오 구간이 이미 디스크에 있으면 재사용하고, 없을 때만 ffmpeg로 추출한다."""
    session_dir = storage.session_dir(metadata.session_id)
    frames_dir = session_dir / "frames" / stem.stem_name
    child_audio_path = session_dir / "audio_segments" / f"{stem.stem_name}_child_segment.wav"

    if not any(frames_dir.glob("frame_*.png")):
        if video_path is None:
            raise FileNotFoundError(f"{stem.stem_name}: 프레임이 없고 --video도 주어지지 않음")
        video_prep.extract_stem_frames(video_path, stem, frames_dir.parent, fps=FRAME_FPS)
    if not child_audio_path.exists():
        if video_path is None:
            raise FileNotFoundError(f"{stem.stem_name}: 아동 오디오 구간이 없고 --video도 주어지지 않음")
        child_audio_path = video_prep.extract_stem_audio(video_path, stem, child_audio_path.parent)
    return frames_dir, child_audio_path


def compute_stem_cv(
    video_path: Path | None, metadata: SessionMetadata, stem: StemTimestamp, cv_config: dict
) -> StemCvFrames:
    """스템 1편의 CV 4종 원본 프레임을 계산한다 (임계값은 아직 적용하지 않음)."""
    frames_dir, child_audio_path = _ensure_stem_media(video_path, metadata, stem)
    doll_cfg = cv_config["doll_tracking"]
    return StemCvFrames(
        facial=run_facial_expression(frames_dir, metadata.subject_type, fps=FRAME_FPS),
        pose=run_pose_analysis(
            frames_dir,
            fps=FRAME_FPS,
            landmark_visibility_min=cv_config["posture"]["landmark_visibility_min"] or 0.5,
        ),
        prosody=run_voice_prosody(child_audio_path),
        doll=run_object_tracking(
            frames_dir,
            DOLL_WEIGHTS_DIR,
            fps=FRAME_FPS,
            conf_min=doll_cfg["detection_conf_min"] or 0.25,
            class_min_seen_frames=doll_cfg["class_min_seen_frames"] or 2,
        ),
    )


def _load_or_compute_cv(
    video_path: Path | None, metadata: SessionMetadata, stem: StemTimestamp, cv_config: dict
) -> StemCvFrames:
    if storage.cv_timeline_exists(metadata.session_id, stem.stem_name):
        return from_timeline(storage.load_stem_cv_timeline(metadata.session_id, stem.stem_name))
    frames = compute_stem_cv(video_path, metadata, stem, cv_config)
    storage.save_stem_cv_timeline(
        metadata.session_id, stem.stem_name, to_timeline(metadata.session_id, stem.stem_name, FRAME_FPS, frames)
    )
    return frames


def _cv_target_stems(timestamps: list[StemTimestamp]) -> list[StemTimestamp]:
    """채점 대상 스템 + 기준선용 워밍업."""
    return [t for t in timestamps if t.scored or t.stem_name == "warm_up"]


def run_cv_stage(video_path: Path | None, metadata: SessionMetadata, recompute: bool = False) -> None:
    """5~7단계만: 표정·자세·인형·음성운율 원본 수치를 스템별로 계산해 cv/<stem>_cv.json에 저장.

    Claude를 호출하지 않는다 — 임계값 보정(scripts/calibrate_cv.py)을 비용 없이 반복하기 위한 단계.
    프레임·오디오가 이미 디스크에 있으면 video_path 없이도 돌 수 있다. recompute=True면 기존 타임라인을 덮어쓴다.
    """
    timestamps = storage.load_stem_timestamps(metadata.session_id)
    cv_config = _load_cv_thresholds(metadata.subject_type)
    for stem in _cv_target_stems(timestamps):
        if not recompute and storage.cv_timeline_exists(metadata.session_id, stem.stem_name):
            logger.info("스템 %s: 저장된 CV 타임라인 재사용", stem.stem_name)
            continue
        logger.info("스템 %s: CV 분석", stem.stem_name)
        try:
            frames = compute_stem_cv(video_path, metadata, stem, cv_config)
        except FileNotFoundError as exc:
            logger.warning("스템 %s 건너뜀: %s", stem.stem_name, exc)
            continue
        storage.save_stem_cv_timeline(
            metadata.session_id, stem.stem_name, to_timeline(metadata.session_id, stem.stem_name, FRAME_FPS, frames)
        )


def run_coding_stage(video_path: Path | None, metadata: SessionMetadata) -> list[StemCodingResult]:
    """5~11단계: 사람이 확정한 stem_timestamps.json을 입력으로, 스템별 AI 채점까지 자동 진행.

    이 단계까지는 사람이 개입하지 않는다 (계획 파이프라인 10단계 설명 참고).
    """
    timestamps = storage.load_stem_timestamps(metadata.session_id)
    cv_config = _load_cv_thresholds(metadata.subject_type)
    thresholds = CvThresholdSet.from_config(cv_config)
    coding_system = load_coding_system()
    story_stems = {s.id: s for s in load_story_stems().stems}

    session_audio_path = storage.session_dir(metadata.session_id) / "audio.wav"
    transcript = transcribe_session(session_audio_path)

    warm_up = next((t for t in timestamps if t.stem_name == "warm_up"), None)
    warm_up_frames = _load_or_compute_cv(video_path, metadata, warm_up, cv_config) if warm_up else None
    baselines = compute_baselines(warm_up_frames, thresholds)
    for note in baselines.notes:
        logger.warning("%s — 해당 신호는 이벤트 없이 진행", note)

    results: list[StemCodingResult] = []
    for stem in timestamps:
        if not stem.scored:
            continue
        logger.info("스템 %s 처리 시작", stem.stem_name)

        frames = _load_or_compute_cv(video_path, metadata, stem, cv_config)
        events = derive_events(frames, thresholds, baselines)

        nonverbal_summary = summarize_stem_features(
            stem.stem_name,
            frames.facial,
            events.facial,
            events.posture,
            events.doll,
            events.prosody,
            quality_notes=list(baselines.notes) + events.quality_notes,
            facial_thresholds=thresholds.facial,
        )

        speaker_tagged = tag_speakers(transcript, stem)
        transcript_text = "\n".join(seg.text for seg in speaker_tagged["child"])

        verbatim_lines = build_stem_verbatim(
            stem, speaker_tagged, events.facial, events.posture, events.doll, events.prosody
        )
        storage.save_stem_verbatim(
            metadata.session_id,
            stem.stem_name,
            StemVerbatimRecord(session_id=metadata.session_id, stem_name=stem.stem_name, lines=verbatim_lines),
        )

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
