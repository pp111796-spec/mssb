"""파인튜닝된 YOLO → 인형·소품별 위치/이동 시계열 (파이프라인 7-1단계).

실제 탐지(YOLO 추론)는 마일스톤 0(인형 라벨링·파인튜닝)이 끝나야 의미가 있다 —
가중치가 없으면 빈 결과를 반환한다 (config/doll_labels.yaml의 tracking_lost_flag 참고).

설계 결정:
  - 인형은 서로 다른 클래스(엄마·아빠·강아지…)라 프레임 간 ID 추적 없이 **프레임별 탐지**만으로
    충분하다. 프레임을 2fps로 뽑아서 30fps 기준 추적기(track_buffer=30 등)는 프레임 간 이동이 커서
    ID가 끊기기 쉽기도 하다.
  - bbox는 이미지 크기로 나눈 0~1 정규화 좌표다 — 해상도·카메라 구도가 달라도 임계값이 같은 의미를 갖는다.
  - 탐지 실패 프레임도 bbox=None으로 **기록**한다. 안 그러면 detect_tracking_lost_segments가 찾을
    "연속 소실" 자체가 존재하지 않는다. 단, 그 스템에서 한 번도(또는 class_min_seen_frames 미만)
    탐지되지 않은 인형은 "이 이야기에 안 나온 인형"으로 보고 소실 프레임을 만들지 않는다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2), 0~1 정규화 좌표
_FRAME_DIAGONAL = math.sqrt(2)


@dataclass(frozen=True)
class DollFrame:
    timestamp_s: float
    doll_id: str  # config/doll_labels.yaml의 class id (예: "dog_doll")
    bbox: BBox | None  # 탐지 실패 시 None
    track_id: int | None = None


@dataclass(frozen=True)
class DollEvent:
    timestamp_s: float
    kind: str  # "proximity_seeking" | "sudden_movement" | "tracking_lost"
    doll_ids: tuple[str, ...]
    value: float | None = None
    note: str = ""


@dataclass(frozen=True)
class DollThresholds:
    detection_conf_min: float
    class_min_seen_frames: int
    tracking_lost_min_frames: int
    proximity_center_distance_width_ratio: float
    proximity_min_consecutive_frames: int
    sudden_movement_diag_ratio: float

    @classmethod
    def from_config(cls, cfg: dict) -> "DollThresholds | None":
        keys = (
            "detection_conf_min", "class_min_seen_frames", "tracking_lost_min_frames",
            "proximity_center_distance_width_ratio", "proximity_min_consecutive_frames",
            "sudden_movement_diag_ratio",
        )
        if any(cfg.get(k) is None for k in keys):
            return None
        return cls(**{k: cfg[k] for k in keys})


def _center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def fill_missing_detections(
    detections_per_frame: list[dict[str, BBox]], fps: float, min_seen_frames: int
) -> list[DollFrame]:
    """프레임별 {doll_id: bbox} 탐지 결과를 DollFrame 목록으로 바꾸고, 탐지 실패 프레임을 bbox=None으로 채운다.

    min_seen_frames번 이상 탐지된 인형만 "이 스템에 등장한 인형"으로 본다.
    """
    seen_counts: dict[str, int] = {}
    for detections in detections_per_frame:
        for doll_id in detections:
            seen_counts[doll_id] = seen_counts.get(doll_id, 0) + 1
    present = sorted(d for d, n in seen_counts.items() if n >= min_seen_frames)

    frames: list[DollFrame] = []
    for i, detections in enumerate(detections_per_frame):
        for doll_id in present:
            frames.append(DollFrame(timestamp_s=i / fps, doll_id=doll_id, bbox=detections.get(doll_id)))
    return frames


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


def detect_proximity_events(
    frames: list[DollFrame], width_ratio: float, min_consecutive_frames: int
) -> list[DollEvent]:
    """두 인형의 상자가 겹치거나 중심 거리가 큰 상자 너비의 width_ratio배 이내인 상태가 이어지면 이벤트.

    문헌에 기준이 없는 추정값(C등급)이라 실제 아동 영상 분포를 보고 조정할 것.
    """
    by_time: dict[float, dict[str, BBox]] = {}
    for f in frames:
        if f.bbox is not None:
            by_time.setdefault(f.timestamp_s, {})[f.doll_id] = f.bbox

    runs: dict[tuple[str, str], list[tuple[float, float]]] = {}  # pair -> [(t, distance_ratio)]
    events: list[DollEvent] = []

    def _flush(pair: tuple[str, str]) -> None:
        run = runs.pop(pair, [])
        if len(run) >= min_consecutive_frames:
            events.append(
                DollEvent(
                    timestamp_s=run[0][0],
                    kind="proximity_seeking",
                    doll_ids=pair,
                    value=min(r for _, r in run),
                    note=f"{len(run)}프레임 연속 근접",
                )
            )

    for t in sorted(by_time):
        boxes = by_time[t]
        ids = sorted(boxes)
        pairs_seen_now = set()
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = boxes[a_id], boxes[b_id]
                (ax, ay), (bx, by) = _center(a), _center(b)
                reference_width = max(_width(a), _width(b)) or 1e-9
                ratio = math.hypot(ax - bx, ay - by) / reference_width
                if _boxes_overlap(a, b) or ratio <= width_ratio:
                    pair = (a_id, b_id)
                    pairs_seen_now.add(pair)
                    runs.setdefault(pair, []).append((t, ratio))
        for pair in [p for p in runs if p not in pairs_seen_now]:
            _flush(pair)
    for pair in list(runs):
        _flush(pair)

    events.sort(key=lambda e: e.timestamp_s)
    return events


def detect_sudden_movement_events(frames: list[DollFrame], diag_ratio: float) -> list[DollEvent]:
    """한 인형의 중심이 연속한 두 프레임 사이에 프레임 대각선의 diag_ratio 이상 이동하면 이벤트(던지기 등 추정)."""
    events: list[DollEvent] = []
    by_doll: dict[str, list[DollFrame]] = {}
    for f in frames:
        by_doll.setdefault(f.doll_id, []).append(f)

    for doll_id, doll_frames in by_doll.items():
        for prev, curr in zip(doll_frames, doll_frames[1:]):
            if prev.bbox is None or curr.bbox is None:
                continue
            (px, py), (cx, cy) = _center(prev.bbox), _center(curr.bbox)
            displacement = math.hypot(cx - px, cy - py) / _FRAME_DIAGONAL
            if displacement >= diag_ratio:
                events.append(
                    DollEvent(
                        timestamp_s=curr.timestamp_s,
                        kind="sudden_movement",
                        doll_ids=(doll_id,),
                        value=displacement,
                        note=f"프레임 대각선의 {displacement:.0%} 이동",
                    )
                )
    events.sort(key=lambda e: e.timestamp_s)
    return events


def compute_doll_events(frames: list[DollFrame], thresholds: DollThresholds | None) -> list[DollEvent]:
    """소실·근접·급이동 이벤트를 모아 시간순으로 반환. 임계값이 없으면 빈 리스트."""
    if thresholds is None or not frames:
        return []
    events: list[DollEvent] = []
    for doll_id in sorted({f.doll_id for f in frames}):
        events.extend(detect_tracking_lost_segments(frames, doll_id, thresholds.tracking_lost_min_frames))
    events.extend(
        detect_proximity_events(
            frames,
            thresholds.proximity_center_distance_width_ratio,
            thresholds.proximity_min_consecutive_frames,
        )
    )
    events.extend(detect_sudden_movement_events(frames, thresholds.sudden_movement_diag_ratio))
    events.sort(key=lambda e: e.timestamp_s)
    return events


def run_object_tracking(
    frames_dir, model_weights_path, fps: float = 2.0, conf_min: float = 0.25, class_min_seen_frames: int = 2
) -> list[DollFrame]:
    """파인튜닝된 YOLO로 프레임별 인형·소품 탐지 (ID 추적 없음 — 모듈 docstring 참고).

    model_weights_path(models/doll_yolo/*.pt)가 아직 없으면(마일스톤 0 라벨링·파인튜닝을
    아직 안 했으면) 조용히 빈 리스트를 반환한다 — 학습 안 된 범용 YOLO로 인형을 억지로
    "탐지"하면 근거 없는 가짜 신호를 만들게 되므로, 차라리 이 모달리티가 비어 있다는 걸
    명확히 하는 쪽을 택했다 (feature_summary.py가 "이 구간에서 감지된 비언어 이벤트 없음"으로
    표시하며, 문제점.md의 "근거 신뢰성" 원칙과 같은 맥락).

    fps는 video_prep.extract_stem_frames와 반드시 같은 값이어야 한다.
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

    detections_per_frame: list[dict[str, BBox]] = []
    for frame_path in frame_paths:
        results = model.predict(str(frame_path), conf=conf_min, verbose=False)
        best: dict[str, tuple[BBox, float]] = {}  # doll_id -> (bbox, confidence)
        if results:
            result = results[0]
            height, width = result.orig_shape
            for box in result.boxes:
                doll_id = model.names[int(box.cls[0])]
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
                confidence = float(box.conf[0])
                # 같은 클래스가 여러 개 잡히면 확신도가 가장 높은 것 하나만 (인형은 클래스당 1개)
                if doll_id not in best or confidence > best[doll_id][1]:
                    best[doll_id] = (bbox, confidence)
        detections_per_frame.append({d: v[0] for d, v in best.items()})

    return fill_missing_detections(detections_per_frame, fps, class_min_seen_frames)
