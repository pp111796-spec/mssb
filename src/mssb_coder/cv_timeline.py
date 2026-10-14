"""CV 4종(표정·자세·음성·인형) 프레임 객체 ↔ 저장용 StemCvTimeline 변환.

타임라인은 임계값을 적용하기 전의 원본 수치다 — 그래서 임계값을 바꿔도 영상 분석(무거운 부분)을
다시 돌릴 필요 없이 이벤트만 새로 계산할 수 있다 (scripts/calibrate_cv.py가 이 파일들을 읽는다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mssb_coder.facial_expression import FacialFrame
from mssb_coder.object_tracking import DollFrame
from mssb_coder.pose_analysis import PoseFrame
from mssb_coder.schema import (
    CvDollSample,
    CvFacialSample,
    CvPoseSample,
    CvProsodySample,
    StemCvTimeline,
)
from mssb_coder.voice_prosody import ProsodyFrame


@dataclass
class StemCvFrames:
    facial: list[FacialFrame] = field(default_factory=list)
    pose: list[PoseFrame] = field(default_factory=list)
    prosody: list[ProsodyFrame] = field(default_factory=list)
    doll: list[DollFrame] = field(default_factory=list)


def to_timeline(session_id: str, stem_name: str, fps: float, frames: StemCvFrames) -> StemCvTimeline:
    return StemCvTimeline(
        session_id=session_id,
        stem_name=stem_name,
        fps=fps,
        facial=[
            CvFacialSample(
                t=f.timestamp_s,
                aus=f.au_intensities,
                head_pitch_deg=f.head_pitch_deg,
                head_yaw_deg=f.head_yaw_deg,
                landmark_confidence=f.landmark_confidence,
            )
            for f in frames.facial
        ],
        pose=[
            CvPoseSample(t=p.timestamp_s, landmarks=p.landmarks, is_facing_camera=p.is_facing_camera)
            for p in frames.pose
        ],
        prosody=[
            CvProsodySample(t=p.timestamp_s, pitch_hz=p.pitch_hz, loudness=p.loudness)
            for p in frames.prosody
        ],
        doll=[
            CvDollSample(t=d.timestamp_s, doll_id=d.doll_id, bbox=d.bbox, track_id=d.track_id)
            for d in frames.doll
        ],
    )


def from_timeline(timeline: StemCvTimeline) -> StemCvFrames:
    return StemCvFrames(
        facial=[
            FacialFrame(
                timestamp_s=s.t,
                au_intensities=s.aus,
                head_pitch_deg=s.head_pitch_deg,
                landmark_confidence=s.landmark_confidence,
                head_yaw_deg=s.head_yaw_deg,
            )
            for s in timeline.facial
        ],
        pose=[
            PoseFrame(timestamp_s=s.t, landmarks=s.landmarks, is_facing_camera=s.is_facing_camera)
            for s in timeline.pose
        ],
        prosody=[
            ProsodyFrame(timestamp_s=s.t, pitch_hz=s.pitch_hz, loudness=s.loudness)
            for s in timeline.prosody
        ],
        doll=[
            DollFrame(timestamp_s=s.t, doll_id=s.doll_id, bbox=s.bbox, track_id=s.track_id)
            for s in timeline.doll
        ],
    )
