"""세션 JSON 읽기/쓰기.

저장 위치는 항상 MSSB_DATA_DIR 환경변수가 가리키는, OneDrive 동기화 밖의 폴더다
(계획 "데이터 저장 위치" 섹션 — 이 리포 자체가 OneDrive 동기화 경로 안에 있어서
원본 영상·세션 데이터를 여기 두면 안 된다). 이 모듈은 그 규칙을 단순 편의가 아니라
강제 조건으로 취급 — MSSB_DATA_DIR이 없으면 조용히 기본값으로 넘어가지 않고 에러를 낸다.

디렉토리 레이아웃:
  <MSSB_DATA_DIR>/sessions/<session_id>/
    stem_timestamps.json
    coding/<stem_name>_ai_codes.json
    coding/<stem_name>_reviewed.json        (선택, 사람이 스템 상세를 고친 경우만)
    session_summary_ai_draft.json
    session_summary_reviewed.json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from mssb_coder.bundle import build_bundle_dict
from mssb_coder.schema import SessionSynthesis, StemCodingResult, StemTimestamp

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


def save_stem_timestamps(session_id: str, timestamps: list[StemTimestamp]) -> Path:
    path = session_dir(session_id) / "stem_timestamps.json"
    path.write_text(
        "[\n" + ",\n".join(t.model_dump_json(indent=2, ensure_ascii=False) for t in timestamps) + "\n]",
        encoding="utf-8",
    )
    return path


def load_stem_timestamps(session_id: str) -> list[StemTimestamp]:
    import json

    path = session_dir(session_id) / "stem_timestamps.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [StemTimestamp.model_validate(item) for item in raw]


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
    stem_results = load_all_stem_codings(session_id)
    return build_bundle_dict(synthesis, stem_results)
