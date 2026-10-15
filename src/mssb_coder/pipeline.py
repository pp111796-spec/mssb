"""세션 1건 실행 — 계획 "파이프라인 단계" 1~13단계를 단계별 함수로 옮긴 것.

각 단계의 실제 구현 상태 (README.md "지금 상태" 절과 함께 최신으로 유지할 것):
  - 오디오/프레임 추출(ffmpeg), 전사(faster-whisper), 자세(MediaPipe), 음성운율(openSMILE),
    Claude 3개 호출(구간분리·스템채점·세션종합) — 실제 구현됨. ANTHROPIC_API_KEY만 있으면 동작.
  - 인형/소품 추적(YOLO) — 코드는 실제 구현됐지만, 마일스톤 0(라벨링·파인튜닝)이 없으면
    가중치 파일이 없어 빈 결과를 반환함(의도된 동작 — 가짜 탐지보다 나음).
  - 표정 분석(Py-Feat) — 코드는 있으나 실제 아동 얼굴로는 아직 검증 전.

4단계(사람 확인)는 검토 앱(src/review_app)의 몫이다. 구간분리 이후 단계(cv·coding)는 사람이
구간을 확정(segmentation_review.json)하지 않았거나 확정 뒤 구간이 바뀌었으면 실행을 거부한다 —
구간이 틀리면 그 뒤의 분석과 채점이 모두 엉뚱한 이야기에 붙기 때문이다.

5~7단계(CV 분석)는 run_cv_stage로 따로 돌릴 수 있다 — Claude를 호출하지 않고 원본 수치를
cv/<stem>_cv.json에 저장하므로, 임계값을 바꿔가며 보정(scripts/calibrate_cv.py)할 때 비용이 들지 않는다.
저장된 프레임·오디오·CV 수치·채점 결과는 만들 때의 구간을 함께 기록하고, 구간이 바뀐 이야기만 다시 계산한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypeVar

import yaml
from pydantic import ValidationError

from mssb_coder import storage, video_prep
from mssb_coder.claude_client import StructuredResult, call_stem_coding
from mssb_coder.coding_systems import CodingSystem, load_coding_system
from mssb_coder.cv_events import CvBaselines, CvThresholdSet, compute_baselines, derive_events, to_session_time
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
    StemWindow,
)
from mssb_coder.segmentation import StemDefinition, StoryStemsConfig, call_segmentation, load_story_stems
from mssb_coder.session_synthesis import call_session_synthesis
from mssb_coder.transcribe import TranscriptSegment, tag_speakers, transcribe_session
from mssb_coder.verbatim import build_stem_verbatim
from mssb_coder.voice_prosody import run_voice_prosody

logger = logging.getLogger("mssb_coder.pipeline")

T = TypeVar("T")

CV_THRESHOLDS_DIR = Path(__file__).resolve().parents[2] / "config"
DOLL_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "models" / "doll_yolo"
FRAME_FPS = 2.0  # video_prep.extract_stem_frames / pose / facial / object_tracking이 모두 같은 값을 써야 함

# Claude가 녹화 앞뒤 여백을 이 이름의 구간으로 돌려준다 — 이야기 목록에는 없지만 채점 제외 구간으로 허용한다.
EDGE_SEGMENT_NAMES = ("pre_session", "post_session")


class SegmentationNotConfirmedError(RuntimeError):
    pass


class StageFailedError(RuntimeError):
    """일부 스템만 실패한 경우 — 나머지 스템 결과는 저장된 상태로 끝난다."""

    def __init__(self, failures: dict[str, str]):
        self.failures = failures
        lines = "\n".join(f"  - {stem}: {reason}" for stem, reason in failures.items())
        super().__init__(f"{len(failures)}개 이야기 처리에 실패했습니다. 다시 실행하면 실패한 이야기만 이어서 처리합니다.\n{lines}")


def _record_usage(session_id: str, call_name: str, response: StructuredResult) -> None:
    storage.append_api_usage(session_id, {"call": call_name, "model": response.model, **response.usage})


def _validate_or_keep(session_id: str, call_name: str, payload: dict, validate: Callable[[dict], T]) -> T:
    """Claude 응답을 스키마로 검증하고, 실패하면 원본을 failed/에 남긴 뒤 다시 던진다."""
    try:
        return validate(payload)
    except (ValueError, KeyError, TypeError):  # pydantic ValidationError는 ValueError의 하위 클래스
        path = storage.save_failed_response(session_id, call_name, payload)
        logger.error("%s 응답이 스키마 검증에 실패함 — 원본 보존: %s", call_name, path)
        raise


def _load_cv_thresholds(subject_type) -> dict:
    filename = f"cv_thresholds_{subject_type.value}.yaml"
    return yaml.safe_load((CV_THRESHOLDS_DIR / filename).read_text(encoding="utf-8"))


def _cfg(value, default):
    """yaml 값이 null일 때만 기본값을 쓴다 (`value or default`는 0도 기본값으로 바꿔버린다)."""
    return default if value is None else value


# --- 1~3단계: 구간분리 ---------------------------------------------------------


def normalize_segments(data: dict, story_stems: StoryStemsConfig) -> list[StemTimestamp]:
    """Claude 구간분리 응답 → StemTimestamp 목록. 이야기 목록에 없는 이름은 거부하고, 채점 여부는 yaml을 따른다."""
    known: dict[str, StemDefinition] = {s.id: s for s in story_stems.stems}
    timestamps: list[StemTimestamp] = []
    unknown: list[str] = []
    for segment in data["segments"]:
        name = segment["stem_name"]
        if name in known:
            scored = known[name].scored
        elif name in EDGE_SEGMENT_NAMES:
            scored = False
        else:
            unknown.append(name)
            continue
        timestamps.append(StemTimestamp.model_validate({**segment, "scored": scored}))
    if unknown:
        raise ValueError(f"이야기 목록(story_stems.yaml)에 없는 구간 이름: {', '.join(unknown)}")
    return timestamps


def run_segmentation_stage(video_path: Path, metadata: SessionMetadata) -> list[StemTimestamp]:
    """1~3단계: 오디오 추출 → 전체 전사 → Claude 구간분리. 세션 정보·전사·구간을 저장한다."""
    session_id = metadata.session_id
    storage.save_session_metadata(metadata.model_copy(update={"video_path": str(Path(video_path).resolve())}))

    logger.info("2단계: 오디오 추출 + 전체 전사")
    audio_path = video_prep.extract_audio(video_path, storage.session_dir(session_id) / "audio.wav")
    transcript = transcribe_session(audio_path)
    storage.save_transcript(session_id, transcript)
    full_text = "\n".join(f"[{seg.start_s:.1f}s] {seg.text}" for seg in transcript)

    logger.info("3단계: Claude 구간분리")
    story_stems = load_story_stems()
    response = call_segmentation(story_stems, full_text)
    _record_usage(session_id, "segmentation", response)
    timestamps = _validate_or_keep(
        session_id, "segmentation", response.data, lambda data: normalize_segments(data, story_stems)
    )

    if (storage.session_dir(session_id) / "stem_timestamps.json").exists():
        backup = storage.backup_stem_timestamps(session_id)
        logger.warning("기존 구간(사람이 고친 내용일 수 있음)을 %s로 백업하고 새 구간분리로 바꿉니다", backup)
    storage.save_stem_timestamps(session_id, timestamps, ai_draft=True)
    storage.save_stem_timestamps(session_id, timestamps)
    logger.info("구간 저장 완료 — 검토 앱 「1. 구간 확인」에서 확인하고 확정하세요")
    return timestamps


def _require_confirmed_segments(session_id: str) -> list[StemTimestamp]:
    status = storage.segmentation_review_status(session_id)
    if status == "unconfirmed":
        raise SegmentationNotConfirmedError(
            "구간이 아직 확정되지 않았습니다. 검토 앱 「1. 구간 확인」에서 구간을 확인하고 '구간 확정'을 누른 뒤 다시 실행하세요."
        )
    if status == "changed":
        raise SegmentationNotConfirmedError(
            "확정한 뒤 구간(stem_timestamps.json)이 바뀌었습니다. 검토 앱에서 다시 확정한 뒤 실행하세요."
        )
    return storage.load_stem_timestamps(session_id)


# --- 5~7단계: CV 분석 ----------------------------------------------------------


def _ensure_stem_media(
    video_path: Path | None, session_id: str, stem: StemTimestamp
) -> tuple[Path, Path | None]:
    """스템의 프레임 폴더와 아동 오디오 구간을 준비한다.

    이미 있고 같은 구간에서 뽑은 것이면 재사용, 구간이 바뀌었으면 다시 뽑는다.
    아동 차례가 없는 구간(전환 시각 = 끝 시각)은 오디오 경로 대신 None을 돌려준다.
    """
    session_dir = storage.session_dir(session_id)
    frames_dir = session_dir / "frames" / stem.stem_name
    child_audio_path = session_dir / "audio_segments" / f"{stem.stem_name}_child_segment.wav"
    window = StemWindow.of(stem)
    saved_window = storage.load_media_window(session_id, stem.stem_name)
    has_frames = any(frames_dir.glob("frame_*.png"))

    stale = has_frames and saved_window is not None and saved_window != window
    if has_frames and saved_window is None:
        logger.warning("%s: 어떤 구간에서 뽑았는지 기록이 없는 예전 프레임을 재사용함", stem.stem_name)
    if stale:
        logger.info("%s: 구간이 바뀌어 프레임·오디오를 다시 추출함", stem.stem_name)
        child_audio_path.unlink(missing_ok=True)

    if not has_frames or stale:
        if video_path is None:
            raise FileNotFoundError(f"{stem.stem_name}: 프레임을 새로 뽑아야 하는데 원본 영상 경로가 없음")
        video_prep.extract_stem_frames(video_path, stem, frames_dir.parent, fps=FRAME_FPS)
        storage.save_media_window(session_id, stem.stem_name, window)

    if stem.handoff_timestamp_s >= stem.end_s:
        return frames_dir, None
    if not child_audio_path.exists():
        if video_path is None:
            raise FileNotFoundError(f"{stem.stem_name}: 아동 오디오를 새로 뽑아야 하는데 원본 영상 경로가 없음")
        child_audio_path = video_prep.extract_stem_audio(video_path, stem, child_audio_path.parent)
    return frames_dir, child_audio_path


def compute_stem_cv(
    video_path: Path | None, metadata: SessionMetadata, stem: StemTimestamp, cv_config: dict
) -> StemCvFrames:
    """스템 1편의 CV 4종 원본 프레임을 계산한다 (임계값은 아직 적용하지 않음)."""
    frames_dir, child_audio_path = _ensure_stem_media(video_path, metadata.session_id, stem)
    doll_cfg = cv_config["doll_tracking"]
    return StemCvFrames(
        facial=run_facial_expression(frames_dir, metadata.subject_type, fps=FRAME_FPS),
        pose=run_pose_analysis(
            frames_dir,
            fps=FRAME_FPS,
            landmark_visibility_min=_cfg(cv_config["posture"]["landmark_visibility_min"], 0.5),
        ),
        prosody=run_voice_prosody(child_audio_path) if child_audio_path else [],
        doll=run_object_tracking(
            frames_dir,
            DOLL_WEIGHTS_DIR,
            fps=FRAME_FPS,
            conf_min=_cfg(doll_cfg["detection_conf_min"], 0.25),
            class_min_seen_frames=_cfg(doll_cfg["class_min_seen_frames"], 2),
        ),
    )


def _load_or_compute_cv(
    video_path: Path | None,
    metadata: SessionMetadata,
    stem: StemTimestamp,
    cv_config: dict,
    recompute: bool = False,
) -> StemCvFrames:
    """저장된 CV 수치가 같은 구간 것이면 재사용하고, 구간이 바뀌었거나 recompute면 다시 계산해 저장한다."""
    session_id = metadata.session_id
    window = StemWindow.of(stem)
    if not recompute and storage.cv_timeline_exists(session_id, stem.stem_name):
        timeline = storage.load_stem_cv_timeline(session_id, stem.stem_name)
        if timeline.source_window == window:
            return from_timeline(timeline)
        if timeline.source_window is None:
            logger.warning(
                "%s: 어떤 구간으로 계산했는지 기록이 없는 예전 CV 수치를 재사용함 (다시 계산하려면 --recompute-cv)",
                stem.stem_name,
            )
            return from_timeline(timeline)
        logger.info("%s: 구간이 바뀌어 CV 수치를 다시 계산함", stem.stem_name)

    frames = compute_stem_cv(video_path, metadata, stem, cv_config)
    timeline = to_timeline(session_id, stem.stem_name, FRAME_FPS, frames).model_copy(update={"source_window": window})
    storage.save_stem_cv_timeline(session_id, stem.stem_name, timeline)
    return frames


def _cv_target_stems(timestamps: list[StemTimestamp]) -> list[StemTimestamp]:
    """채점 대상 스템 + 기준선용 워밍업."""
    return [t for t in timestamps if t.scored or t.stem_name == "warm_up"]


def _video_path_for(video_path: Path | None, metadata: SessionMetadata) -> Path | None:
    if video_path is not None:
        return video_path
    return Path(metadata.video_path) if metadata.video_path else None


def run_cv_stage(video_path: Path | None, metadata: SessionMetadata, recompute: bool = False) -> None:
    """5~7단계만: 표정·자세·인형·음성운율 원본 수치를 스템별로 계산해 cv/<stem>_cv.json에 저장.

    Claude를 호출하지 않는다 — 임계값 보정(scripts/calibrate_cv.py)을 비용 없이 반복하기 위한 단계.
    """
    timestamps = _require_confirmed_segments(metadata.session_id)
    video_path = _video_path_for(video_path, metadata)
    cv_config = _load_cv_thresholds(metadata.subject_type)
    for stem in _cv_target_stems(timestamps):
        logger.info("스템 %s: CV 분석", stem.stem_name)
        try:
            _load_or_compute_cv(video_path, metadata, stem, cv_config, recompute=recompute)
        except (FileNotFoundError, video_prep.FfmpegError) as exc:
            logger.warning("스템 %s 건너뜀: %s", stem.stem_name, exc)


# --- 8~11단계: 스템 채점 -------------------------------------------------------


def _load_or_transcribe(session_id: str) -> list[TranscriptSegment]:
    transcript = storage.load_transcript(session_id)
    if transcript is not None:
        return transcript
    audio_path = storage.session_dir(session_id) / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"{audio_path}가 없습니다. segmentation 단계를 먼저 실행하세요.")
    logger.warning("저장된 전사가 없는 예전 세션이라 다시 전사함 (결과는 transcript.json으로 저장)")
    transcript = transcribe_session(audio_path)
    storage.save_transcript(session_id, transcript)
    return transcript


def format_child_transcript(segments: list[TranscriptSegment]) -> str:
    """Claude에 보내는 아동 발화 — 근거 시각을 지어내지 않도록 세션 절대 시각을 붙인다."""
    return "\n".join(f"[{seg.start_s:.1f}s] {seg.text}" for seg in segments)


def _existing_coding_matches(session_id: str, stem: StemTimestamp) -> StemCodingResult | None:
    try:
        existing = storage.load_stem_coding(session_id, stem.stem_name)
    except FileNotFoundError:
        return None
    return existing if existing.source_window == StemWindow.of(stem) else None


def _code_one_stem(
    *,
    video_path: Path | None,
    metadata: SessionMetadata,
    stem: StemTimestamp,
    stem_def: StemDefinition,
    cv_config: dict,
    thresholds: CvThresholdSet,
    baselines: CvBaselines,
    transcript: list[TranscriptSegment],
    coding_system: CodingSystem,
) -> StemCodingResult:
    session_id = metadata.session_id
    frames = _load_or_compute_cv(video_path, metadata, stem, cv_config)
    events = to_session_time(derive_events(frames, thresholds, baselines), stem)

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

    response = call_stem_coding(
        coding_system,
        transcript_text=format_child_transcript(speaker_tagged["child"]),
        nonverbal_summary=nonverbal_summary,
        story_specific_notes=stem_def.story_specific_notes,
        child_age_months=metadata.child_age_months,
        response_mode="spontaneous",  # 검토 앱에서 사람이 바로잡는 값 (§6.6)
        issue_prompt_count=0,
    )
    _record_usage(session_id, f"stem_coding:{stem.stem_name}", response)
    result = _validate_or_keep(
        session_id,
        f"stem_coding_{stem.stem_name}",
        response.data
        | {
            "session_id": session_id,
            "child_age_months": metadata.child_age_months,
            "stem_name": stem.stem_name,
            "completion_status": "complete",
            "response_mode": "spontaneous",
            "issue_prompt_count": 0,
            "model_version": response.model,
            "source_window": StemWindow.of(stem).model_dump(),
            "timestamp_basis": "session_absolute",
        },
        StemCodingResult.model_validate,
    )
    storage.save_stem_coding(session_id, stem.stem_name, result)
    storage.save_stem_verbatim(
        session_id,
        stem.stem_name,
        StemVerbatimRecord(
            session_id=session_id,
            stem_name=stem.stem_name,
            lines=build_stem_verbatim(speaker_tagged, events.facial, events.posture, events.doll, events.prosody),
        ),
    )
    return result


def run_coding_stage(
    video_path: Path | None, metadata: SessionMetadata, force: bool = False
) -> list[StemCodingResult]:
    """5~11단계: 확정된 구간으로 스템별 AI 채점까지 진행한다.

    같은 구간으로 이미 채점한 스템은 건너뛴다(force=True면 전부 다시). 한 스템이 실패해도 나머지는
    계속 처리하고, 끝에 실패 목록을 StageFailedError로 알린다.
    """
    session_id = metadata.session_id
    timestamps = _require_confirmed_segments(session_id)
    video_path = _video_path_for(video_path, metadata)
    cv_config = _load_cv_thresholds(metadata.subject_type)
    thresholds = CvThresholdSet.from_config(cv_config)
    coding_system = load_coding_system()
    story_stems = {s.id: s for s in load_story_stems().stems}
    transcript = _load_or_transcribe(session_id)

    warm_up = next((t for t in timestamps if t.stem_name == "warm_up"), None)
    warm_up_frames = None
    if warm_up is not None:
        try:
            warm_up_frames = _load_or_compute_cv(video_path, metadata, warm_up, cv_config)
        except (FileNotFoundError, video_prep.FfmpegError) as exc:
            logger.warning("워밍업 CV를 준비하지 못함 (%s)", exc)
    baselines = compute_baselines(warm_up_frames, thresholds)
    for note in baselines.notes:
        logger.warning("%s — 해당 신호는 이벤트 없이 진행", note)

    results: list[StemCodingResult] = []
    failures: dict[str, str] = {}
    for stem in timestamps:
        if not stem.scored:
            continue
        if not force:
            existing = _existing_coding_matches(session_id, stem)
            if existing is not None:
                logger.info("스템 %s: 같은 구간으로 이미 채점함 — 건너뜀 (다시 하려면 --force)", stem.stem_name)
                results.append(existing)
                continue
        logger.info("스템 %s 처리 시작", stem.stem_name)
        try:
            results.append(
                _code_one_stem(
                    video_path=video_path,
                    metadata=metadata,
                    stem=stem,
                    stem_def=story_stems[stem.stem_name],
                    cv_config=cv_config,
                    thresholds=thresholds,
                    baselines=baselines,
                    transcript=transcript,
                    coding_system=coding_system,
                )
            )
        except Exception as exc:  # noqa: BLE001 — CV 라이브러리·API 오류가 무엇이든 다른 스템은 계속 처리
            logger.exception("스템 %s 처리 실패", stem.stem_name)
            failures[stem.stem_name] = f"{type(exc).__name__}: {exc}"

    if failures:
        raise StageFailedError(failures)
    return results


# --- 12단계: 세션 종합 ---------------------------------------------------------


def load_codings_for_synthesis(session_id: str) -> list[StemCodingResult]:
    """현재 구간의 채점 대상 스템 결과(AI 원본)만 모은다. 빠졌거나 예전 구간으로 채점된 스템이 있으면 멈춘다."""
    timestamps = storage.load_stem_timestamps(session_id)
    results: list[StemCodingResult] = []
    missing: list[str] = []
    outdated: list[str] = []
    for stem in timestamps:
        if not stem.scored:
            continue
        try:
            result = storage.load_stem_coding(session_id, stem.stem_name)
        except FileNotFoundError:
            missing.append(stem.stem_name)
            continue
        if result.source_window is not None and result.source_window != StemWindow.of(stem):
            outdated.append(stem.stem_name)
        results.append(result)
    problems = []
    if missing:
        problems.append(f"채점 결과가 없는 이야기: {', '.join(missing)}")
    if outdated:
        problems.append(f"구간이 바뀐 뒤 다시 채점하지 않은 이야기: {', '.join(outdated)}")
    if problems:
        raise RuntimeError("종합을 멈춥니다. coding 단계를 먼저 끝내세요.\n" + "\n".join(problems))
    return results


def run_synthesis_stage(metadata: SessionMetadata) -> SessionSynthesis:
    """12단계: 스템별 AI 채점 원본 전체 → 세션 종합 패턴 초안."""
    stem_results = load_codings_for_synthesis(metadata.session_id)
    response = call_session_synthesis(stem_results)
    _record_usage(metadata.session_id, "synthesis", response)
    synthesis = _validate_or_keep(
        metadata.session_id,
        "synthesis",
        response.data | {"session_id": metadata.session_id, "model_version": response.model},
        SessionSynthesis.model_validate,
    )
    storage.save_session_synthesis(metadata.session_id, synthesis)
    return synthesis
