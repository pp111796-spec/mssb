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


class FfmpegError(RuntimeError):
    pass


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-5:])
        raise FfmpegError(f"ffmpeg {what} 실패 (종료코드 {result.returncode}):\n{tail}")


def extract_audio(video_path: Path, out_path: Path, sample_rate: int = 16000) -> Path:
    """전체 세션 오디오를 하나의 wav로 추출 (2단계 — faster-whisper 입력용, 16kHz mono가 표준)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(), "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        str(out_path),
    ]
    _run_ffmpeg(cmd, "전체 오디오 추출")
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
    # 이전 구간에서 뽑은 프레임이 남아 있으면 구간이 짧아졌을 때 섞여 분석된다
    for old_frame in stem_out_dir.glob("frame_*.png"):
        old_frame.unlink()
    duration = stem.end_s - stem.start_s
    cmd = [
        _ffmpeg_bin(), "-y",
        "-ss", str(stem.start_s), "-i", str(video_path), "-t", str(duration),
        "-vf", f"fps={fps}",
        str(stem_out_dir / "frame_%06d.png"),
    ]
    _run_ffmpeg(cmd, f"{stem.stem_name} 프레임 추출")
    return stem_out_dir


PREVIEW_MARGIN_S = 10.0


def preview_clip_command(ffmpeg: str, video_path: Path, start_s: float, end_s: float, out_path: Path) -> list[str]:
    """검토 앱용 저화질(360p) 미리보기 클립 — 원본 전체를 브라우저에 올리지 않으려고 구간만 자른다."""
    return [
        ffmpeg, "-y",
        "-ss", f"{start_s:.2f}", "-i", str(video_path), "-t", f"{end_s - start_s:.2f}",
        "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
        str(out_path),
    ]


def preview_clip_path(start_s: float, end_s: float, out_dir: Path) -> tuple[Path, float, float]:
    """구간 앞뒤 PREVIEW_MARGIN_S초를 붙인 미리보기 클립의 (경로, 클립 시작 세션 시각, 클립 끝 세션 시각)."""
    clip_start = max(0.0, start_s - PREVIEW_MARGIN_S)
    clip_end = end_s + PREVIEW_MARGIN_S
    return out_dir / f"clip_{clip_start:.1f}_{clip_end:.1f}.mp4", clip_start, clip_end


def extract_preview_clip(video_path: Path, start_s: float, end_s: float, out_dir: Path) -> tuple[Path, float]:
    """미리보기 클립을 만들고 (클립 경로, 클립이 시작하는 세션 시각)을 돌려준다. 이미 있으면 다시 만들지 않는다."""
    out_path, clip_start, clip_end = preview_clip_path(start_s, end_s, out_dir)
    if not out_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.stem + ".tmp.mp4")
        _run_ffmpeg(preview_clip_command(_ffmpeg_bin(), video_path, clip_start, clip_end, tmp_path), "미리보기 클립 생성")
        tmp_path.replace(out_path)
    return out_path, clip_start


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
    _run_ffmpeg(cmd, f"{stem.stem_name} 아동 구간 오디오 추출")
    return out_path
