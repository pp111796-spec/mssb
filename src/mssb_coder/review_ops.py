"""검토 앱의 판단 로직 — Streamlit과 분리해 테스트할 수 있게 둔다 (화면은 src/review_app)."""

from __future__ import annotations

from dataclasses import dataclass

from mssb_coder import storage


@dataclass(frozen=True)
class SessionProgress:
    label: str  # 사이드바에 보여줄 한 줄 상태
    has_timestamps: bool
    segmentation_status: str  # "unconfirmed" | "changed" | "confirmed" | "none"
    coded_count: int
    scored_count: int
    has_synthesis: bool
    synthesis_reviewed: bool


def session_progress(session_id: str) -> SessionProgress:
    d = storage.session_dir(session_id)
    has_timestamps = (d / "stem_timestamps.json").exists()
    status = storage.segmentation_review_status(session_id) if has_timestamps else "none"
    scored = [t.stem_name for t in storage.load_stem_timestamps(session_id) if t.scored] if has_timestamps else []
    coded = sum(1 for name in scored if (d / "coding" / f"{name}_ai_codes.json").exists())
    has_synthesis = (d / "session_summary_ai_draft.json").exists()
    synthesis_reviewed = (d / "session_summary_reviewed.json").exists()

    if not has_timestamps:
        label = "구간분리 전"
    elif status == "unconfirmed":
        label = "구간 확인 필요"
    elif status == "changed":
        label = "구간이 바뀜 — 다시 확정 필요"
    elif coded < len(scored):
        label = f"채점 진행 중 ({coded}/{len(scored)})"
    elif not has_synthesis:
        label = "종합 전"
    elif not synthesis_reviewed:
        label = "종합 완료 — 검토 전"
    else:
        label = "검토 저장됨"
    return SessionProgress(
        label=label,
        has_timestamps=has_timestamps,
        segmentation_status=status,
        coded_count=coded,
        scored_count=len(scored),
        has_synthesis=has_synthesis,
        synthesis_reviewed=synthesis_reviewed,
    )
