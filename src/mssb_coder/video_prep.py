"""ffmpeg 래퍼 — 오디오 추출 + (구간 확정 후) 스템별 프레임 추출 (파이프라인 2·5단계).

시스템에 ffmpeg이 설치돼 있어야 한다 (winget install Gyan.FFmpeg — 계획 "설정 절차" 참고).
이 모듈은 순수 subprocess 호출이라 추가 파이썬 의존성이 없다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from mssb_coder.schema import StemTimestamp


class FfmpegNotFoundError(RuntimeError):
    pass


def _ffmpeg_bin() -> str:
    bin_path = shutil.which("ffmpeg")
    if bin_path is None:
        raise FfmpegNotFoundError(
            "ffmpeg을 찾을 수 없습니다. `winget install Gyan.FFmpeg`로 설치한 뒤 PATH를 확인하세요."
        )
    return bin_path


def extract_audio(video_path: Path, out_path: Path, sample_rate: int = 16000) -> Path:
    """전체 세션 오디오를 하나의 wav로 추출 (2단계 — faster-whisper 입력용, 16kHz mono가 표준)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(), "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def extract_stem_frames(
    video_path: Path,
    stem: StemTimestamp,
    out_dir: Path,
    fps: float = 2.0,
) -> Path:
    """확정된 스템 구간(start_s~end_s)에서 CV 분석용 프레임을 샘플링 (5단계).

    fps는 프레임 추출 밀도 — 표정/자세 변화를 놓치지 않으면서도 용량을 아끼는 절충값.
    실제 적정값은 마일스톤 A에서 CV 산출물을 보며 조정할 것 (cv_thresholds.yaml TODO와 별개 사안).
    """
    stem_out_dir = out_dir / stem.stem_name
    stem_out_dir.mkdir(parents=True, exist_ok=True)
    duration = stem.end_s - stem.start_s
    cmd = [
        _ffmpeg_bin(), "-y",
        "-ss", str(stem.start_s), "-i", str(video_path), "-t", str(duration),
        "-vf", f"fps={fps}",
        str(stem_out_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return stem_out_dir


def extract_stem_audio(video_path: Path, stem: StemTimestamp, out_dir: Path) -> Path:
    """스템 구간의 아동 발화 구간(handoff_timestamp_s 이후)만 별도 wav로 추출 (voice_prosody.py 입력).

    검사자 구간을 섞지 않는 이유는 계획 "음성 운율(목소리 톤) 분석" 섹션 참고 —
    handoff 탐지가 음성운율 분석의 전제조건이다.
    """
    out_path = out_dir / f"{stem.stem_name}_child_segment.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    child_start = stem.handoff_timestamp_s
    duration = stem.end_s - child_start
    cmd = [
        _ffmpeg_bin(), "-y",
        "-ss", str(child_start), "-i", str(video_path), "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
