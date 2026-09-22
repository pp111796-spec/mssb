"""MediaPipe Pose → 프레임별 랜드마크 (파이프라인 7단계).

주 피사체 선별 규칙: 검사자는 뒤통수·어깨만 프레임 구석에 살짝 걸리는 촬영 구도라
"정면을 향한 인물 = 아동"이라는 단순 규칙이 대부분 통한다 (계획 "관절/자세 분석" 섹션).
여기서는 얼굴 랜드마크(코) 검출 신뢰도를 그 규칙의 대리 지표로 쓴다 — 검사자는
카메라에 얼굴이 거의 안 잡히니 코 visibility가 낮게 나올 것으로 기대된다. 이 대리 지표
자체가 실측으로 검증된 적은 없다 — 마일스톤 A에서 실제 영상으로 확인할 것.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# MediaPipe Pose의 33개 랜드마크 중 자세 분석(posture_metrics.py)에 직접 쓰는 것만 우선 정의.
# 전체 33개+손 21개x2는 실제 연동 시 mediapipe 라이브러리의 landmark enum을 그대로 씀.
KEY_LANDMARK_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_wrist",
    "right_wrist",
)


@dataclass(frozen=True)
class PoseFrame:
    timestamp_s: float
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (x, y), 0~1 정규화 좌표
    is_facing_camera: bool = True  # 주 피사체(아동) 선별 결과


FACE_VISIBILITY_THRESHOLD = 0.5  # TODO: 실측 후 조정 — 코 visibility가 이 값 미만이면 "카메라를 안 보는 인물"

# mediapipe PoseLandmark enum index -> 이 모듈이 쓰는 이름
_LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_wrist": 15,
    "right_wrist": 16,
}


def run_pose_analysis(frames_dir: Path, fps: float = 2.0) -> list[PoseFrame]:
    """frames_dir(video_prep.extract_stem_frames의 출력)에 있는 frame_%06d.png들을 순서대로 처리.

    fps는 프레임 추출 때 쓴 값과 같아야 timestamp_s가 정확하다 (video_prep.extract_stem_frames 참고).
    """
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415

    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    results: list[PoseFrame] = []

    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        for i, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detection = pose.process(image_rgb)

            timestamp_s = i / fps
            if detection.pose_landmarks is None:
                results.append(PoseFrame(timestamp_s=timestamp_s, landmarks={}, is_facing_camera=False))
                continue

            lm = detection.pose_landmarks.landmark
            landmarks = {
                name: (lm[idx].x, lm[idx].y)
                for name, idx in _LANDMARK_INDEX.items()
                if lm[idx].visibility >= 0.1
            }
            is_facing_camera = lm[_LANDMARK_INDEX["nose"]].visibility >= FACE_VISIBILITY_THRESHOLD
            results.append(
                PoseFrame(timestamp_s=timestamp_s, landmarks=landmarks, is_facing_camera=is_facing_camera)
            )

    return results
