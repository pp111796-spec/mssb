"""파인튜닝된 YOLO+ByteTrack → 인형·소품별 위치/이동 시계열 (파이프라인 7-1단계).

실제 탐지(YOLO 추론)는 마일스톤 0(인형 라벨링·파인튜닝)이 끝나야 의미가 있다 —
지금은 데이터 구조와, 탐지 소실을 플래그로 남기는 순수 로직만 구현한다
(문제점.md 옛 9번 완화책, config/doll_labels.yaml의 tracking_lost_flag 참고).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DollFrame:
    timestamp_s: float
    doll_id: str  # config/doll_labels.yaml의 class id (예: "dog_doll")
    bbox: tuple[float, float, float, float] | None  # (x1, y1, x2, y2), 탐지 실패 시 None
    track_id: int | None = None


@dataclass(frozen=True)
class DollEvent:
    timestamp_s: float
    kind: str  # "proximity_seeking" | "sudden_movement" | "tracking_lost"
    doll_ids: tuple[str, ...]
    value: float | None = None
    note: str = ""


def detect_tracking_lost_segments(
    frames: list[DollFrame], doll_id: str, min_consecutive_missing_frames: int
) -> list[DollEvent]:
    """특정 인형의 경계상자가 min_consecutive_missing_frames 이상 연속으로 비면 이벤트로 남긴다.

    탐지 자체가 비는 문제(포옹으로 가려짐, 던지기로 블러)는 이 함수가 고치지 못한다 —
    다만 "조용히 비지 않고 드러나게" 만드는 게 이 함수의 역할이다.
    """
    events: list[DollEvent] = []
    doll_frames = [f for f in frames if f.doll_id == doll_id]

    run_start: DollFrame | None = None
    run_len = 0
    for frame in doll_frames:
        if frame.bbox is None:
            if run_start is None:
                run_start = frame
            run_len += 1
        else:
            if run_start is not None and run_len >= min_consecutive_missing_frames:
                events.append(
                    DollEvent(
                        timestamp_s=run_start.timestamp_s,
                        kind="tracking_lost",
                        doll_ids=(doll_id,),
                        note=f"{run_len}프레임 연속 탐지 실패 ({run_start.timestamp_s:.1f}s~{frame.timestamp_s:.1f}s)",
                    )
                )
            run_start = None
            run_len = 0

    if run_start is not None and run_len >= min_consecutive_missing_frames:
        events.append(
            DollEvent(
                timestamp_s=run_start.timestamp_s,
                kind="tracking_lost",
                doll_ids=(doll_id,),
                note=f"{run_len}프레임 연속 탐지 실패 (구간 끝까지)",
            )
        )

    return events


import logging

logger = logging.getLogger(__name__)


def run_object_tracking(frames_dir, model_weights_path) -> list[DollFrame]:
    """파인튜닝된 YOLO+ByteTrack으로 인형·소품을 추적한다.

    model_weights_path(models/doll_yolo/*.pt)가 아직 없으면(마일스톤 0 라벨링·파인튜닝을
    아직 안 했으면) 조용히 빈 리스트를 반환한다 — 학습 안 된 범용 YOLO로 인형을 억지로
    "탐지"하면 근거 없는 가짜 신호를 만들게 되므로, 차라리 이 모달리티가 비어 있다는 걸
    명확히 하는 쪽을 택했다 (feature_summary.py가 "이 구간에서 감지된 비언어 이벤트 없음"으로
    표시하며, 문제점.md의 "근거 신뢰성" 원칙과 같은 맥락).
    """
    from pathlib import Path  # noqa: PLC0415

    model_weights_path = Path(model_weights_path)
    weight_files = sorted(model_weights_path.glob("*.pt")) if model_weights_path.is_dir() else (
        [model_weights_path] if model_weights_path.exists() else []
    )
    if not weight_files:
        logger.warning(
            "인형 YOLO 가중치를 찾을 수 없음 (%s) — 마일스톤 0 미완료. "
            "인형 추적 없이 진행합니다.", model_weights_path,
        )
        return []

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(weight_files[0]))
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))

    results: list[DollFrame] = []
    fps = 2.0  # video_prep.extract_stem_frames와 반드시 같은 값이어야 함
    for i, frame_path in enumerate(frame_paths):
        timestamp_s = i / fps
        detections = model.track(str(frame_path), persist=True, verbose=False)
        boxes = detections[0].boxes if detections else None
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            doll_id = model.names[class_id]
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            results.append(
                DollFrame(
                    timestamp_s=timestamp_s,
                    doll_id=doll_id,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
    return results
