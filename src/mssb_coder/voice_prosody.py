"""openSMILE(eGeMAPS) → 피치/음량/떨림 등 음향 특징 시계열 (파이프라인 7-2단계).

handoff_timestamp_s 이후(아동 구간)에만 적용한다 — 검사자 목소리가 섞이면
"검사자 톤"이 "아동 정서 신호"로 잘못 해석될 위험 (계획 "음성 운율" 섹션).

절대 임계값 대신 세션 내 상대 정규화(z-score)를 쓴다 — 아동은 기본 피치 자체가
성인과 다르므로, 그 아동 자신의 워밍업 구간 평균 대비 얼마나 튀었는지를 본다.
이 정규화 로직은 순수 통계 계산이라 지금 실제로 구현·테스트 가능하다.
실제 openSMILE 호출(eGeMAPS 추출)은 마일스톤 A에서 구현한다.
"""

from __future__ import annotations

import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProsodyFrame:
    timestamp_s: float
    pitch_hz: float
    loudness: float


@dataclass(frozen=True)
class ProsodyEvent:
    timestamp_s: float
    kind: str  # "pitch_spike" | "loudness_spike"
    zscore: float
    note: str = ""


@dataclass(frozen=True)
class ProsodyBaseline:
    pitch_mean: float
    pitch_stdev: float
    loudness_mean: float
    loudness_stdev: float


def compute_baseline(warm_up_frames: list[ProsodyFrame]) -> ProsodyBaseline:
    """워밍업 구간 프레임으로 아동 개인의 기준선을 계산 (config의 baseline_segment: warm_up)."""
    if len(warm_up_frames) < 2:
        raise ValueError("기준선 계산에는 최소 2프레임 이상의 워밍업 구간 데이터가 필요함")
    pitches = [f.pitch_hz for f in warm_up_frames]
    loudness = [f.loudness for f in warm_up_frames]
    return ProsodyBaseline(
        pitch_mean=statistics.mean(pitches),
        pitch_stdev=statistics.pstdev(pitches) or 1e-6,
        loudness_mean=statistics.mean(loudness),
        loudness_stdev=statistics.pstdev(loudness) or 1e-6,
    )


def compute_zscore_events(
    frames: list[ProsodyFrame], baseline: ProsodyBaseline, notable_threshold: float
) -> list[ProsodyEvent]:
    events: list[ProsodyEvent] = []
    for frame in frames:
        pitch_z = (frame.pitch_hz - baseline.pitch_mean) / baseline.pitch_stdev
        if abs(pitch_z) >= notable_threshold:
            events.append(
                ProsodyEvent(timestamp_s=frame.timestamp_s, kind="pitch_spike", zscore=pitch_z)
            )
        loudness_z = (frame.loudness - baseline.loudness_mean) / baseline.loudness_stdev
        if abs(loudness_z) >= notable_threshold:
            events.append(
                ProsodyEvent(timestamp_s=frame.timestamp_s, kind="loudness_spike", zscore=loudness_z)
            )
    return events


def _ascii_safe_egemaps_config_path() -> str:
    """openSMILE의 네이티브 라이브러리는 설정파일 경로를 ASCII로만 인코딩한다.

    이 프로젝트가 OneDrive의 한글 경로("...OneDrive - 건양대학교\\바탕 화면...") 안에 있어서,
    opensmile 패키지가 pip으로 설치된 site-packages 경로(=.venv도 같은 한글 경로 하위)를
    그대로 쓰면 `UnicodeEncodeError: 'ascii' codec can't encode characters...`로 죽는다
    (opensmile/core/lib.py의 `bytes(config_file, "ascii")` — 실측으로 확인함).

    해결책: eGeMAPSv02 설정 파일(및 include 대상 전체)을 MSSB_DATA_DIR(ASCII 경로로 지정된
    OneDrive 밖 폴더) 아래로 한 번 복사해두고, 그 복사본 경로를 feature_set에 직접 넘긴다.
    """
    import opensmile  # noqa: PLC0415
    from mssb_coder.storage import get_data_dir  # noqa: PLC0415

    source_root = Path(opensmile.Smile.__new__(opensmile.Smile).default_config_root)
    dest_root = get_data_dir() / "opensmile_config"

    if not dest_root.exists():
        shutil.copytree(source_root, dest_root)

    return str(dest_root / (opensmile.FeatureSet.eGeMAPSv02.value + ".conf"))


def run_voice_prosody(child_segment_wav_path) -> list[ProsodyFrame]:
    """openSMILE eGeMAPSv02 LLD(프레임 단위 저수준 기술자)에서 피치·음량을 뽑는다.

    eGeMAPS는 피치를 27.5Hz 기준 세미톤으로 내보낸다 — 여기서 Hz로 환산해 반환한다
    (필드명 pitch_hz와 상위 문서의 "피치(F0)" 설명을 그대로 지키기 위함).
    세미톤 값이 0 이하인 프레임(무성음/침묵 구간)은 통계를 왜곡하므로 제외한다.
    """
    import opensmile  # noqa: PLC0415

    smile = opensmile.Smile(
        feature_set=_ascii_safe_egemaps_config_path(),
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    df = smile.process_file(str(child_segment_wav_path))

    frames: list[ProsodyFrame] = []
    for (_file, start, _end), row in df.iterrows():
        semitone = row["F0semitoneFrom27.5Hz_sma3nz"]
        if semitone <= 0:
            continue  # 무성음/침묵 구간
        pitch_hz = 27.5 * (2 ** (semitone / 12))
        frames.append(
            ProsodyFrame(
                timestamp_s=start.total_seconds(),
                pitch_hz=pitch_hz,
                loudness=row["Loudness_sma3"],
            )
        )
    return frames
