#!/usr/bin/env python
"""CLI: python scripts/calibrate_cv.py --session <id> [--session <id> ...] --subject-type child

저장된 cv/<stem>_cv.json(run_pipeline.py --stage cv의 결과)만 읽어 임계값 보정용 리포트를 만든다.
Claude를 호출하지 않고, 영상이 없어도 된다. 리포트는 MSSB_DATA_DIR 아래에만 쓴다 (저장소에 넣지 않음).

임계값은 이 리포트의 분포와 "육안 확인 후보" 프레임을 사람이 직접 보고
config/cv_thresholds_{adult,child}.yaml에 반영한다 — 이 스크립트는 값을 자동으로 확정하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from mssb_coder import storage  # noqa: E402
from mssb_coder.calibration import diagnose_stem, format_report, summarize_event_rates  # noqa: E402
from mssb_coder.cv_events import CvThresholdSet, compute_baselines  # noqa: E402
from mssb_coder.cv_timeline import from_timeline  # noqa: E402
from mssb_coder.pipeline import FRAME_FPS, _load_cv_thresholds  # noqa: E402
from mssb_coder.schema import SubjectType  # noqa: E402


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="CV 임계값 보정 리포트 (Claude 호출 없음)")
    parser.add_argument("--session", action="append", required=True, help="세션 ID (여러 번 지정 가능)")
    parser.add_argument("--subject-type", choices=["adult", "child"], default="child")
    args = parser.parse_args()

    thresholds = CvThresholdSet.from_config(_load_cv_thresholds(SubjectType(args.subject_type)))

    for session_id in args.session:
        timestamps = storage.load_stem_timestamps(session_id)
        by_name = {t.stem_name: t for t in timestamps}

        warm_up_frames = None
        if "warm_up" in by_name and storage.cv_timeline_exists(session_id, "warm_up"):
            warm_up_frames = from_timeline(storage.load_stem_cv_timeline(session_id, "warm_up"))
        baselines = compute_baselines(warm_up_frames, thresholds)

        diagnoses = []
        for stem in timestamps:
            if not (stem.scored or stem.stem_name == "warm_up"):
                continue
            if not storage.cv_timeline_exists(session_id, stem.stem_name):
                print(f"[건너뜀] {session_id}/{stem.stem_name}: cv 타임라인 없음 — run_pipeline.py --stage cv 먼저 실행")
                continue
            frames = from_timeline(storage.load_stem_cv_timeline(session_id, stem.stem_name))
            diagnoses.append(
                diagnose_stem(
                    stem.stem_name,
                    frames,
                    thresholds,
                    baselines,
                    stem_start_s=stem.start_s,
                    handoff_s=stem.handoff_timestamp_s,
                    frames_dir=storage.session_dir(session_id) / "frames" / stem.stem_name,
                    fps=FRAME_FPS,
                )
            )

        report = format_report(diagnoses)
        rates = summarize_event_rates(diagnoses)
        report += "\n## 스템당 평균 이벤트 수\n" + "\n".join(f"- {k}: {v:.1f}" for k, v in sorted(rates.items())) + "\n"
        out_path = storage.session_dir(session_id) / "cv_calibration_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
