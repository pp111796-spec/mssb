"""openSMILE(eGeMAPS) → 피치/음량/떨림 등 음향 특징 시계열 (파이프라인 7-2단계).

handoff_timestamp_s 이후(아동 구간)에만 적용한다 — 검사자 목소리가 섞이면
"검사자 톤"이 "아동 정서 신호"로 잘못 해석될 위험 (계획 "음성 운율" 섹션).

절대 임계값 대신 세션 내 상대 정규화(z-score)를 쓴다 — 아동은 기본 피치 자체가
성인과 다르므로, 그 아동 자신의 워밍업 구간 평균 대비 얼마나 튀었는지를 본다.
이 정규화 로직은 순수 통계 계산이라 지금 실제로 구현·테스트 가능하다.
실제 openSMILE 호출(eGeMAPS 추출)은 마일스톤 A에서 구현한다.
"""

from __future__ import annotations

import math
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
    kind: str  # "pitch_high" | "pitch_low" | "loudness_high" | "loudness_low"
    zscore: float
    note: str = ""


@dataclass(frozen=True)
class ProsodyBaseline:
    pitch_mean: float
    pitch_stdev: float
    loudness_mean: float
    loudness_stdev: float


# [C] 문헌 근거 없는 공학적 하한 — 워밍업 목소리가 거의 일정하면 표준편차가 0에 가까워져
# 작은 차이도 z-score가 수십~수백으로 튄다. posture_metrics의 _*_STDEV_FLOOR와 같은 목적.
_PITCH_STDEV_FLOOR_HZ = 5.0
_LOUDNESS_STDEV_FLOOR = 0.05


def compute_baseline(warm_up_frames: list[ProsodyFrame]) -> ProsodyBaseline:
    """워밍업 구간 프레임으로 아동 개인의 기준선을 계산 (config의 baseline_segment: warm_up)."""
    usable = [f for f in warm_up_frames if math.isfinite(f.pitch_hz) and math.isfinite(f.loudness)]
    if len(usable) < 2:
        raise ValueError("기준선 계산에는 최소 2프레임 이상의 워밍업 구간 데이터가 필요함")
    pitches = [f.pitch_hz for f in usable]
    loudness = [f.loudness for f in usable]
    return ProsodyBaseline(
        pitch_mean=statistics.mean(pitches),
        pitch_stdev=max(statistics.pstdev(pitches), _PITCH_STDEV_FLOOR_HZ),
        loudness_mean=statistics.mean(loudness),
        loudness_stdev=max(statistics.pstdev(loudness), _LOUDNESS_STDEV_FLOOR),
    )


FRAME_STEP_S = 0.01  # openSMILE LLD 프레임 간격(10ms)


@dataclass(frozen=True)
class ProsodyThresholds:
    pitch_valid_min_hz: float
    pitch_valid_max_hz: float
    baseline_min_voiced_frames: int
    window_s: float
    window_min_voiced_ratio: float
    zscore_notable_threshold: float

    @classmethod
    def from_config(cls, cfg: dict) -> "ProsodyThresholds | None":
        """cv_thresholds_*.yaml의 voice_prosody 블록. 값이 하나라도 null이면 None — 이벤트를 만들지 않는다."""
        keys = (
            "pitch_valid_min_hz", "pitch_valid_max_hz", "baseline_min_voiced_frames",
            "window_s", "window_min_voiced_ratio", "zscore_notable_threshold",
        )
        if any(cfg.get(k) is None for k in keys):
            return None
        return cls(**{k: cfg[k] for k in keys})


def filter_valid_pitch(frames: list[ProsodyFrame], thresholds: ProsodyThresholds) -> list[ProsodyFrame]:
    """아동 목소리로 타당한 피치 범위 밖의 프레임은 피치 추적 오류로 보고 버린다."""
    return [
        f for f in frames
        if thresholds.pitch_valid_min_hz <= f.pitch_hz <= thresholds.pitch_valid_max_hz
        and math.isfinite(f.loudness)
    ]


def compute_prosody_baseline(
    warm_up_frames: list[ProsodyFrame], thresholds: ProsodyThresholds
) -> ProsodyBaseline | None:
    """워밍업 유성음이 baseline_min_voiced_frames보다 적으면 None — 기준선이 불안정하니 음성 신호를 끈다."""
    voiced = filter_valid_pitch(warm_up_frames, thresholds)
    if len(voiced) < max(thresholds.baseline_min_voiced_frames, 2):
        return None
    return compute_baseline(voiced)


def compute_prosody_events(
    frames: list[ProsodyFrame], baseline: ProsodyBaseline | None, thresholds: ProsodyThresholds | None
) -> list[ProsodyEvent]:
    """창(window_s) 단위 중앙값 z-score로 판정한다 — 10ms 프레임마다 이벤트를 만들면 폭주하기 때문.

    기준선이나 임계값이 없으면(cv_thresholds_*.yaml 미확정, 문제점.md 3번) 조용히 빈 리스트를
    반환한다 — posture_metrics.compute_posture_events와 같은 원칙(값을 지어내지 않음).
    kind에 방향(높음/낮음)을 담는다: pitch_high/pitch_low/loudness_high/loudness_low.
    """
    if baseline is None or thresholds is None:
        return []

    windows: dict[int, list[ProsodyFrame]] = {}
    for frame in filter_valid_pitch(frames, thresholds):
        windows.setdefault(int(frame.timestamp_s // thresholds.window_s), []).append(frame)

    min_frames = thresholds.window_min_voiced_ratio * thresholds.window_s / FRAME_STEP_S
    events: list[ProsodyEvent] = []
    for index in sorted(windows):
        window = windows[index]
        if len(window) < min_frames:
            continue
        start_s = index * thresholds.window_s
        pitch_z = statistics.median(
            (f.pitch_hz - baseline.pitch_mean) / baseline.pitch_stdev for f in window
        )
        loudness_z = statistics.median(
            (f.loudness - baseline.loudness_mean) / baseline.loudness_stdev for f in window
        )
        for name, z in (("pitch", pitch_z), ("loudness", loudness_z)):
            if abs(z) >= thresholds.zscore_notable_threshold:
                events.append(
                    ProsodyEvent(
                        timestamp_s=start_s,
                        kind=f"{name}_{'high' if z > 0 else 'low'}",
                        zscore=z,
                        note=f"{thresholds.window_s:g}초 창 유성음 {len(window)}프레임 중앙값",
                    )
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
        # 임시 폴더에 다 복사한 뒤 이름을 바꾼다 — 복사 도중 끊기면 반쪽 설정이 영구히 재사용되기 때문
        tmp_root = dest_root.with_name(dest_root.name + ".tmp")
        shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.copytree(source_root, tmp_root)
        tmp_root.replace(dest_root)

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
        if not semitone > 0 or not math.isfinite(row["Loudness_sma3"]):
            continue  # 무성음/침묵 구간 (NaN도 여기서 걸러진다)
        pitch_hz = 27.5 * (2 ** (semitone / 12))
        frames.append(
            ProsodyFrame(
                timestamp_s=start.total_seconds(),
                pitch_hz=pitch_hz,
                loudness=row["Loudness_sma3"],
            )
        )
    return frames
