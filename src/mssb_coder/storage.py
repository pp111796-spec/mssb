"""세션 JSON 읽기/쓰기.

저장 위치는 항상 MSSB_DATA_DIR 환경변수가 가리키는, OneDrive 동기화 밖의 폴더다
(계획 "데이터 저장 위치" 섹션 — 이 리포 자체가 OneDrive 동기화 경로 안에 있어서
원본 영상·세션 데이터를 여기 두면 안 된다). 이 모듈은 그 규칙을 단순 편의가 아니라
강제 조건으로 취급 — MSSB_DATA_DIR이 없으면 조용히 기본값으로 넘어가지 않고 에러를 낸다.

디렉토리 레이아웃:
  <MSSB_DATA_DIR>/sessions/<session_id>/
    session.json                            (아동 연령·성별·subject_type·원본 영상 경로)
    transcript.json                         (전체 전사 — 코딩 단계가 다시 전사하지 않도록)
    stem_timestamps_ai_draft.json           (Claude 구간분리 원안)
    stem_timestamps.json                    (사람이 확인·수정한 구간)
    segmentation_review.json                (구간 확정 기록 — 확정 이후 구간이 바뀌면 무효)
    frames/<stem_name>/source_window.json   (추출한 프레임·오디오의 구간)
    api_usage.jsonl / failed/               (Claude 호출 사용량 / 검증 실패 원본)
    coding/<stem_name>_ai_codes.json
    coding/<stem_name>_reviewed.json        (선택, 사람이 스템 상세를 고친 경우만)
    coding/<stem_name>_verbatim.json        (축어록 — 언어+행동분석 통합)
    cv/<stem_name>_cv.json                  (표정·자세·음성·인형 원본 수치 — 임계값 보정용)
    session_summary_ai_draft.json
    session_summary_reviewed.json
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

from mssb_coder.bundle import build_bundle_dict
from mssb_coder.schema import (
    SessionMetadata,
    SessionSynthesis,
    StemCodingResult,
    StemCvTimeline,
    StemTimestamp,
    StemVerbatimRecord,
    StemWindow,
    utc_now_iso,
)
from mssb_coder.transcribe import TranscriptSegment

ModelT = TypeVar("ModelT", bound=BaseModel)


class DataDirNotConfiguredError(RuntimeError):
    pass


def get_data_dir() -> Path:
    raw = os.environ.get("MSSB_DATA_DIR")
    if not raw:
        raise DataDirNotConfiguredError(
            "MSSB_DATA_DIR 환경변수가 설정되지 않았습니다. "
            ".env에 OneDrive 동기화 밖의 경로(예: C:\\mssb_data)를 지정하세요."
        )
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_dir(session_id: str) -> Path:
    d = get_data_dir() / "sessions" / session_id
    (d / "coding").mkdir(parents=True, exist_ok=True)
    return d


def _write_model(path: Path, model: BaseModel) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(model.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)  # 원자적 교체 — 쓰다 만 파일이 남지 않게
    return path


def _read_model(path: Path, model_cls: type[ModelT]) -> ModelT:
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


# --- stem_timestamps.json (파이프라인 3~4단계) ---------------------------------


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return path


def _timestamps_path(session_id: str, ai_draft: bool) -> Path:
    name = "stem_timestamps_ai_draft.json" if ai_draft else "stem_timestamps.json"
    return session_dir(session_id) / name


def save_stem_timestamps(session_id: str, timestamps: list[StemTimestamp], ai_draft: bool = False) -> Path:
    """ai_draft=True는 Claude가 처음 나눈 원안 — 사람이 고친 stem_timestamps.json과 비교·감사용으로 따로 둔다."""
    text = "[\n" + ",\n".join(t.model_dump_json(indent=2, ensure_ascii=False) for t in timestamps) + "\n]"
    return _write_text_atomic(_timestamps_path(session_id, ai_draft), text)


def backup_stem_timestamps(session_id: str) -> Path:
    """구간분리를 다시 돌리기 전에 현재 구간(사람이 고친 내용일 수 있음)을 시각 붙은 파일로 남긴다."""
    src = _timestamps_path(session_id, ai_draft=False)
    stamp = utc_now_iso().replace(":", "").replace("+", "_")
    dst = src.with_name(f"stem_timestamps.{stamp}.bak.json")
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def load_stem_timestamps(session_id: str, ai_draft: bool = False) -> list[StemTimestamp]:
    raw = json.loads(_timestamps_path(session_id, ai_draft).read_text(encoding="utf-8"))
    return [StemTimestamp.model_validate(item) for item in raw]


# --- segmentation_review.json (4단계 사람 확인 — 확정 여부) -------------------------


def _timestamps_fingerprint(timestamps: list[StemTimestamp]) -> str:
    canonical = json.dumps([t.model_dump(mode="json") for t in timestamps], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def confirm_segmentation(session_id: str) -> Path:
    """현재 stem_timestamps.json을 사람이 확정했다고 기록한다 — 이후 구간이 바뀌면 확정이 풀린 것으로 본다."""
    record = {
        "confirmed_at": utc_now_iso(),
        "timestamps_sha256": _timestamps_fingerprint(load_stem_timestamps(session_id)),
    }
    return _write_text_atomic(
        session_dir(session_id) / "segmentation_review.json", json.dumps(record, ensure_ascii=False, indent=2)
    )


def segmentation_review_status(session_id: str) -> Literal["unconfirmed", "changed", "confirmed"]:
    path = session_dir(session_id) / "segmentation_review.json"
    if not path.exists():
        return "unconfirmed"
    record = json.loads(path.read_text(encoding="utf-8"))
    current = _timestamps_fingerprint(load_stem_timestamps(session_id))
    return "confirmed" if record.get("timestamps_sha256") == current else "changed"


# --- session.json / transcript.json (구간분리 단계에서 저장, 이후 단계가 재사용) ----------


def save_session_metadata(metadata: SessionMetadata) -> Path:
    return _write_model(session_dir(metadata.session_id) / "session.json", metadata)


def load_session_metadata(session_id: str) -> SessionMetadata:
    return _read_model(session_dir(session_id) / "session.json", SessionMetadata)


def session_metadata_exists(session_id: str) -> bool:
    return (session_dir(session_id) / "session.json").exists()


def save_transcript(session_id: str, segments: list[TranscriptSegment]) -> Path:
    payload = [{"start_s": s.start_s, "end_s": s.end_s, "text": s.text} for s in segments]
    return _write_text_atomic(
        session_dir(session_id) / "transcript.json", json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_transcript(session_id: str) -> list[TranscriptSegment] | None:
    """저장된 전체 전사. 없으면 None (예전 세션 — 호출 쪽이 다시 전사한다)."""
    path = session_dir(session_id) / "transcript.json"
    if not path.exists():
        return None
    return [TranscriptSegment(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


# --- frames/<stem>/source_window.json (추출한 프레임·오디오가 어떤 구간 것인지) -----------


def _media_window_path(session_id: str, stem_name: str) -> Path:
    return session_dir(session_id) / "frames" / stem_name / "source_window.json"


def save_media_window(session_id: str, stem_name: str, window: StemWindow) -> Path:
    return _write_model(_media_window_path(session_id, stem_name), window)


def load_media_window(session_id: str, stem_name: str) -> StemWindow | None:
    path = _media_window_path(session_id, stem_name)
    return _read_model(path, StemWindow) if path.exists() else None


# --- <stem>_ai_codes.json / <stem>_reviewed.json (파이프라인 10·13단계) --------


def save_stem_coding(session_id: str, stem_name: str, result: StemCodingResult, reviewed: bool = False) -> Path:
    suffix = "reviewed" if reviewed else "ai_codes"
    return _write_model(session_dir(session_id) / "coding" / f"{stem_name}_{suffix}.json", result)


def load_stem_coding(session_id: str, stem_name: str, reviewed: bool = False) -> StemCodingResult:
    suffix = "reviewed" if reviewed else "ai_codes"
    return _read_model(session_dir(session_id) / "coding" / f"{stem_name}_{suffix}.json", StemCodingResult)


def load_all_stem_codings(session_id: str, reviewed: bool = False) -> list[StemCodingResult]:
    """12단계(세션 종합) 입력 — 명시적으로 ai_codes(원본)를 기본으로 쓴다 (based_on 설계 원칙 참고)."""
    suffix = "reviewed" if reviewed else "ai_codes"
    coding_dir = session_dir(session_id) / "coding"
    results = []
    for path in sorted(coding_dir.glob(f"*_{suffix}.json")):
        results.append(_read_model(path, StemCodingResult))
    return results


def stem_reviewed_exists(session_id: str, stem_name: str) -> bool:
    return (session_dir(session_id) / "coding" / f"{stem_name}_reviewed.json").exists()


def load_effective_stem_codings(session_id: str) -> list[StemCodingResult]:
    """보고서·번들·게시용 스템 결과 — 스템마다 사람 검토본이 있으면 그것을, 없으면 AI 원본을 쓴다.

    stem_timestamps.json이 있으면 그 안의 채점 대상 스템만 모은다(예전 실행에서 남은 다른 스템 파일 제외).
    세션 종합의 입력은 이 함수가 아니라 AI 원본이다(pipeline.load_codings_for_synthesis).
    """
    coding_dir = session_dir(session_id) / "coding"
    if (session_dir(session_id) / "stem_timestamps.json").exists():
        names = [t.stem_name for t in load_stem_timestamps(session_id) if t.scored]
    else:
        names = sorted(p.name.removesuffix("_ai_codes.json") for p in coding_dir.glob("*_ai_codes.json"))
    results = []
    for name in names:
        try:
            results.append(load_stem_coding(session_id, name, reviewed=stem_reviewed_exists(session_id, name)))
        except FileNotFoundError:
            continue
    return results


# --- <stem>_verbatim.json (축어록 — 언어+행동분석 통합, 파이프라인 5~10단계 중 생성) ----------


def save_stem_verbatim(session_id: str, stem_name: str, record: StemVerbatimRecord) -> Path:
    return _write_model(session_dir(session_id) / "coding" / f"{stem_name}_verbatim.json", record)


def load_stem_verbatim(session_id: str, stem_name: str) -> StemVerbatimRecord:
    return _read_model(session_dir(session_id) / "coding" / f"{stem_name}_verbatim.json", StemVerbatimRecord)


# --- api_usage.jsonl / failed/ (Claude 호출 기록) ----------------------------------------------


def append_api_usage(session_id: str, record: dict) -> Path:
    """Claude 호출 1회의 토큰 사용량을 한 줄(JSON)로 덧붙인다 — 세션당 실제 비용 확인용."""
    path = session_dir(session_id) / "api_usage.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": utc_now_iso(), **record}, ensure_ascii=False) + "\n")
    return path


def save_failed_response(session_id: str, call_name: str, payload) -> Path:
    """스키마 검증에 실패한 Claude 응답 원본을 보존한다 — 같은 호출을 다시 돌리지 않고도 원인을 볼 수 있게."""
    stamp = utc_now_iso().replace(":", "").replace("+", "_")
    path = session_dir(session_id) / "failed" / f"{call_name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --- cv/<stem>_cv.json (CV 원본 수치 타임라인 — 임계값 보정·이벤트 재계산용) -----------------


def _cv_path(session_id: str, stem_name: str) -> Path:
    return session_dir(session_id) / "cv" / f"{stem_name}_cv.json"


def save_stem_cv_timeline(session_id: str, stem_name: str, timeline: StemCvTimeline) -> Path:
    return _write_model(_cv_path(session_id, stem_name), timeline)


def load_stem_cv_timeline(session_id: str, stem_name: str) -> StemCvTimeline:
    return _read_model(_cv_path(session_id, stem_name), StemCvTimeline)


def cv_timeline_exists(session_id: str, stem_name: str) -> bool:
    return _cv_path(session_id, stem_name).exists()


# --- session_summary_*.json (파이프라인 12·13단계) -----------------------------


def save_session_synthesis(session_id: str, result: SessionSynthesis, reviewed: bool = False) -> Path:
    filename = "session_summary_reviewed.json" if reviewed else "session_summary_ai_draft.json"
    return _write_model(session_dir(session_id) / filename, result)


def load_session_synthesis(session_id: str, reviewed: bool = False) -> SessionSynthesis:
    filename = "session_summary_reviewed.json" if reviewed else "session_summary_ai_draft.json"
    return _read_model(session_dir(session_id) / filename, SessionSynthesis)


# --- 공개 사이트 업로드용 내보내기 번들 -----------------------------------------


def build_session_export_bundle(session_id: str) -> dict:
    """공개 사이트(src/public_app/app.py)에 업로드할 JSON 번들을 만든다.

    사람이 검토·수정한 reviewed 버전이 있으면 그걸, 없으면 AI 원본을 쓴다 — 검토를
    마친 뒤 내보내는 게 자연스러운 흐름이기 때문. 영상 원본은 이 함수가 다루는
    대상이 아니다 — 애초에 이미 구조화된 JSON 결과만 읽는다.
    """
    reviewed_exists = (session_dir(session_id) / "session_summary_reviewed.json").exists()
    synthesis = load_session_synthesis(session_id, reviewed=reviewed_exists)
    return build_bundle_dict(synthesis, load_effective_stem_codings(session_id))


# --- 사업단 공유 사이트 게시 (results/ — git 저장소 안, MSSB_DATA_DIR과 다름) ----

REPO_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def publish_session_bundle_to_repo(session_id: str) -> Path:
    """번들을 저장소의 results/<session_id>.json으로 저장한다 (git add/commit/push는 호출 쪽 책임).

    MSSB_DATA_DIR이 아니라 이 코드 저장소 안에 쓴다 — 공개 사이트(src/public_app/app.py)가
    같은 저장소에서 이 폴더를 읽어 보여주기 때문. 영상은 절대 여기 들어가지 않는다 —
    build_session_export_bundle이 만드는, 이미 처리된 JSON 결과만 저장한다.
    """
    bundle = build_session_export_bundle(session_id)
    REPO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPO_RESULTS_DIR / f"{session_id}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
