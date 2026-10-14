"""MediaPipe Pose → 프레임별 랜드마크 (파이프라인 7단계).

주 피사체 선별 규칙: 검사자는 뒤통수·어깨만 프레임 구석에 살짝 걸리는 촬영 구도라
"정면을 향한 인물 = 아동"이라는 단순 규칙이 대부분 통한다 (계획 "관절/자세 분석" 섹션).
여기서는 얼굴 랜드마크(코) 검출 신뢰도를 그 규칙의 대리 지표로 쓴다 — 검사자는
카메라에 얼굴이 거의 안 잡히니 코 visibility가 낮게 나올 것으로 기대된다. 이 대리 지표
자체가 실측으로 검증된 적은 없다 — 마일스톤 A에서 실제 영상으로 확인할 것.

**mediapipe 1.0의 API 변경(실측으로 확인)**: 예전에 흔히 쓰던 `mp.solutions.pose.Pose(...)`
(Solutions API)가 mediapipe 1.0부터 완전히 제거됐다 — `AttributeError: module 'mediapipe'
has no attribute 'solutions'`. 새 Tasks API(`mediapipe.tasks.vision.PoseLandmarker`)로
다시 짰다. 이 API는 모델 파일(.task)을 명시적으로 내려받아야 하는데, openSMILE 때와 똑같은
이유로 OneDrive 한글 경로에 모델을 두면 네이티브 라이브러리가 못 연다 — 그래서
MSSB_DATA_DIR(ASCII 경로) 아래에 내려받는다.
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


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ascii_safe_model_path() -> Path:
    """openSMILE과 같은 이유로, 모델 파일을 MSSB_DATA_DIR(ASCII 경로) 아래 캐싱한다."""
    import urllib.request  # noqa: PLC0415

    from mssb_coder.storage import get_data_dir  # noqa: PLC0415

    model_path = get_data_dir() / "models" / "mediapipe" / "pose_landmarker_lite.task"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


def select_primary_pose(poses: list, nose_index: int = 0):
    """검출된 사람이 여럿이면 코 visibility가 가장 높은(=카메라를 가장 정면으로 보는) 사람을 아동으로 본다.

    num_poses 기본값(1)이면 검사자가 먼저 검출될 수 있어서 2명까지 받아 고른다.
    """
    return max(poses, key=lambda lm: lm[nose_index].visibility)


def run_pose_analysis(
    frames_dir: Path, fps: float = 2.0, landmark_visibility_min: float = 0.5
) -> list[PoseFrame]:
    """frames_dir(video_prep.extract_stem_frames의 출력)에 있는 frame_%06d.png들을 순서대로 처리.

    fps는 프레임 추출 때 쓴 값과 같아야 timestamp_s가 정확하다 (video_prep.extract_stem_frames 참고).
    landmark_visibility_min 미만인 랜드마크는 버린다 (미취학 MediaPipe 연구도 0.5 미만을 무효 처리).
    """
    import mediapipe as mp  # noqa: PLC0415

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_ascii_safe_model_path())),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=2,
    )

    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    results: list[PoseFrame] = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame_path in enumerate(frame_paths):
            timestamp_s = i / fps
            mp_image = mp.Image.create_from_file(str(frame_path))
            detection = landmarker.detect(mp_image)

            if not detection.pose_landmarks:
                results.append(PoseFrame(timestamp_s=timestamp_s, landmarks={}, is_facing_camera=False))
                continue

            lm = select_primary_pose(detection.pose_landmarks, _LANDMARK_INDEX["nose"])
            landmarks = {
                name: (lm[idx].x, lm[idx].y)
                for name, idx in _LANDMARK_INDEX.items()
                if lm[idx].visibility >= landmark_visibility_min
            }
            is_facing_camera = lm[_LANDMARK_INDEX["nose"]].visibility >= FACE_VISIBILITY_THRESHOLD
            results.append(
                PoseFrame(timestamp_s=timestamp_s, landmarks=landmarks, is_facing_camera=is_facing_camera)
            )

    return results
