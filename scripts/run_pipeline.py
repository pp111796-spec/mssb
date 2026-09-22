#!/usr/bin/env python
"""CLI: python scripts/run_pipeline.py --session <id> --video <path> ...

계획 "설정 절차" 섹션의 실행 명령을 실제로 구현한 것. --stage로 4단계(사람 확인) 전후를 나눠 실행한다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from mssb_coder.pipeline import run_coding_stage, run_segmentation_stage, run_synthesis_stage  # noqa: E402
from mssb_coder.schema import SessionMetadata, SubjectType  # noqa: E402


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MSSB AI 코딩 파이프라인 실행")
    parser.add_argument("--session", required=True, help="세션 ID (파일명에 아동 실명 쓰지 말 것)")
    parser.add_argument("--video", type=Path, help="session_video.mp4 경로 (segmentation/coding 단계에 필요)")
    parser.add_argument("--child-age-months", type=int, help="아동 개월 수")
    parser.add_argument("--child-gender", choices=["male", "female"], help="departure 스템 화자 판별용")
    parser.add_argument(
        "--subject-type", choices=["adult", "child"], default="adult",
        help="adult=성인 대역(마일스톤 A), child=실제 아동(마일스톤 B)",
    )
    parser.add_argument(
        "--stage", choices=["segmentation", "coding", "synthesis", "all"], default="all",
        help="segmentation 실행 후 Streamlit에서 4단계(사람 확인)를 거치고 coding부터 이어서 실행",
    )
    args = parser.parse_args()

    metadata = None
    if args.child_age_months and args.child_gender:
        metadata = SessionMetadata(
            session_id=args.session,
            child_age_months=args.child_age_months,
            child_gender=args.child_gender,
            subject_type=SubjectType(args.subject_type),
        )

    if args.stage in ("segmentation", "all"):
        if args.video is None:
            parser.error("--video는 segmentation 단계에 필요함")
        if metadata is None:
            parser.error("--child-age-months와 --child-gender는 segmentation 단계에 필요함")
        run_segmentation_stage(args.video, metadata)

    if args.stage in ("coding", "all"):
        if args.video is None or metadata is None:
            parser.error("--video, --child-age-months, --child-gender는 coding 단계에 필요함")
        run_coding_stage(args.video, metadata)

    if args.stage in ("synthesis", "all"):
        if metadata is None:
            parser.error("--child-age-months와 --child-gender는 synthesis 단계에 필요함")
        run_synthesis_stage(metadata)


if __name__ == "__main__":
    main()
