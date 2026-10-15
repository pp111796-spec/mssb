#!/usr/bin/env python
"""CLI: python scripts/run_pipeline.py --session <id> --stage <단계> ...

순서: segmentation → 검토 앱 「1. 구간 확인」에서 확정 → (cv) → coding → synthesis.
사람 확인을 건너뛰지 않도록 단계를 한 번에 몰아 실행하는 옵션은 두지 않는다.
segmentation 단계에서 저장한 session.json(연령·성별·영상 경로)이 있으면 이후 단계는 --session만으로 된다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from mssb_coder import storage  # noqa: E402
from mssb_coder.pipeline import (  # noqa: E402
    SegmentationNotConfirmedError,
    StageFailedError,
    run_coding_stage,
    run_cv_stage,
    run_segmentation_stage,
    run_synthesis_stage,
)
from mssb_coder.schema import SessionMetadata, SubjectType  # noqa: E402

logger = logging.getLogger("run_pipeline")


def resolve_metadata(args: argparse.Namespace) -> SessionMetadata | None:
    """CLI 값과 저장된 session.json을 합친다. CLI에 연령·성별이 있으면 그 값으로 session.json을 갱신한다."""
    saved = storage.load_session_metadata(args.session) if storage.session_metadata_exists(args.session) else None
    video_path = str(args.video.resolve()) if args.video else (saved.video_path if saved else None)

    if args.child_age_months and args.child_gender:
        subject_type = args.subject_type or (saved.subject_type.value if saved else "adult")
        metadata = SessionMetadata(
            session_id=args.session,
            child_age_months=args.child_age_months,
            child_gender=args.child_gender,
            subject_type=SubjectType(subject_type),
            video_path=video_path,
        )
        if saved and saved.model_dump(exclude={"video_path"}) != metadata.model_dump(exclude={"video_path"}):
            logger.warning("저장된 세션 정보와 다른 값이 주어져 새 값으로 바꿉니다: %s → %s", saved, metadata)
        storage.save_session_metadata(metadata)
        return metadata

    if saved is None:
        return None
    if args.video and saved.video_path != video_path:
        saved = saved.model_copy(update={"video_path": video_path})
        storage.save_session_metadata(saved)
    return saved


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MSSB AI 코딩 파이프라인 실행")
    parser.add_argument("--session", required=True, help="세션 ID (파일명에 아동 실명 쓰지 말 것)")
    parser.add_argument("--video", type=Path, help="원본 영상 경로 (segmentation에 필요, 이후엔 저장된 경로 사용)")
    parser.add_argument("--child-age-months", type=int, help="아동 개월 수 (처음 한 번만, 이후 session.json 재사용)")
    parser.add_argument("--child-gender", choices=["male", "female"], help="departure 스템 화자 판별용")
    parser.add_argument(
        "--subject-type", choices=["adult", "child"], default=None,
        help="adult=성인 대역(마일스톤 A), child=실제 아동(마일스톤 B). 생략하면 저장된 값, 처음이면 adult",
    )
    parser.add_argument(
        "--stage", choices=["segmentation", "cv", "coding", "synthesis"], required=True,
        help="segmentation 실행 후 검토 앱에서 구간을 확정해야 cv·coding을 실행할 수 있다. "
        "cv는 Claude 호출 없이 표정·자세·인형·음성운율 원본 수치만 저장 (임계값 보정용)",
    )
    parser.add_argument(
        "--recompute-cv", action="store_true",
        help="cv 단계에서 저장된 수치가 있어도 다시 계산",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="coding 단계에서 같은 구간으로 이미 채점한 이야기도 다시 채점 (Claude 호출 비용 발생)",
    )
    args = parser.parse_args()

    metadata = resolve_metadata(args)
    if metadata is None:
        parser.error("이 세션의 session.json이 없어 --child-age-months와 --child-gender가 필요함")

    try:
        if args.stage == "segmentation":
            if args.video is None:
                parser.error("--video는 segmentation 단계에 필요함")
            run_segmentation_stage(args.video, metadata)
        elif args.stage == "cv":
            run_cv_stage(args.video, metadata, recompute=args.recompute_cv)
        elif args.stage == "coding":
            run_coding_stage(args.video, metadata, force=args.force)
        elif args.stage == "synthesis":
            run_synthesis_stage(metadata)
    except (SegmentationNotConfirmedError, StageFailedError, FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
