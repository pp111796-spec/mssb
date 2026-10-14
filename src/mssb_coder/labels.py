"""영문 id/enum → 한글 표시 라벨 변환 — report.py/review_app/public_app이 공유.

카테고리·세부항목·스템은 이미 YAML에 display_name_ko가 있으므로(coding_systems.py,
segmentation.py) 이 모듈은 그걸 조회하는 얇은 헬퍼만 제공한다. CompletionStatus/
ResponseMode/EvidenceSource와 CV 이벤트 kind(posture/doll/prosody), AU 코드는 YAML
소스가 없어 이 파일 안에 직접 매핑을 둔다.

어떤 조회 함수든 매칭 실패 시 예외를 내지 않고 원본 id/코드를 그대로 반환한다 —
라벨 누락은 화면 표시 문제일 뿐, 파이프라인을 멈출 이유가 아니기 때문이다
(report.py의 기존 _category_label과 같은 원칙).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from mssb_coder.coding_systems import CodingSystem
from mssb_coder.schema import CompletionStatus, EvidenceSource, ResponseMode
from mssb_coder.segmentation import StoryStemsConfig

DEFAULT_DOLL_LABELS_PATH = Path(__file__).resolve().parents[2] / "config" / "doll_labels.yaml"


def category_label(coding_system: CodingSystem, category_id: str) -> str:
    for category in coding_system.categories:
        if category.id == category_id:
            return category.display_name_ko
    return category_id


def item_label(coding_system: CodingSystem, category_id: str, item_id: str) -> str:
    for category in coding_system.categories:
        if category.id != category_id:
            continue
        for item in category.items:
            if item.id == item_id:
                return item.display_name_ko
    return item_id


def stem_label(story_stems: StoryStemsConfig, stem_name: str) -> str:
    for stem in story_stems.stems:
        if stem.id == stem_name:
            return stem.display_name_ko
    return stem_name


_COMPLETION_STATUS_KO = {
    CompletionStatus.COMPLETE: "완료",
    CompletionStatus.PARTIAL: "부분 실시",
    CompletionStatus.NOT_ADMINISTERED: "미실시",
}

_RESPONSE_MODE_KO = {
    ResponseMode.SPONTANEOUS: "자발적",
    ResponseMode.PROMPTED: "프롬프트 후",
}


def scrub_english_ids(text: str, coding_system: CodingSystem, story_stems: StoryStemsConfig) -> str:
    """Claude가 자유 텍스트(rationale/pattern_description/evidence note)에 영문 스템id·범주id·
    enum값(spontaneous 등)을 그대로 echo한 경우를 한글 라벨로 치환한다 — 프롬프트 수정 전에
    이미 생성·저장된 텍스트에도 API 재호출 없이 적용되는 사후 안전망.
    """
    if not text:
        return text
    replacements = {s.id: s.display_name_ko for s in story_stems.stems}
    replacements.update({c.id: c.display_name_ko for c in coding_system.categories})
    replacements.update({mode.value: ko for mode, ko in _RESPONSE_MODE_KO.items()})
    replacements.update({status.value: ko for status, ko in _COMPLETION_STATUS_KO.items()})
    for raw_id, ko in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        # 일반 \b는 안 됨 — 한글 문자가 Python 정규식에서 \w로 취급돼, "juice와"처럼 영문 id
        # 바로 뒤에 한글이 붙으면 경계로 인식되지 않는다. ASCII 식별자 문자만 경계 판정에 쓴다.
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(raw_id)}(?![A-Za-z0-9_])"
        text = re.sub(pattern, ko, text)
    text = _AU_RUN_RE.sub(_plain_au_run, text)
    for jargon, plain in _JARGON_KO.items():
        text = text.replace(jargon, plain)
    return text


# "AU23/AU24", "AU04+AU20", "AU23=0.7, AU24=0.6"처럼 이어 붙은 AU 코드 묶음 — 학부모가 읽는 글에는
# 코드·수치 대신 얼굴에서 보인 모습만 남긴다.
_AU_ONE = r"AU\d{2}(?:\s*=\s*[\d.]+)?"
_AU_RUN_RE = re.compile(
    rf"{_AU_ONE}(?:\s*(?:[/+,·]|및|와|과)\s*{_AU_ONE})*(?P<tail>\s*(?:등|중심|동반|상승|표정))?"
)

_JARGON_KO = {
    "긴장 계열": "긴장과 관련된",
    "자기수정": "말을 고쳐 말하는 모습",
}


def _plain_au_run(match: re.Match) -> str:
    names: list[str] = []
    for code in re.findall(r"AU\d{2}", match.group(0)):
        name = _AU_GLOSS.get(code, "").split("(")[0]
        if name and name not in names:
            names.append(name)
    tail = match.group("tail") or ""
    body = "·".join(names) if names else "표정 변화"
    # 뒤에 '표정'이 이미 붙어 있으면 그대로 두고, 아니면 '표정'을 덧붙여 명사구로 만든다.
    suffix = "" if tail.strip() == "표정" else " 표정"
    return body + suffix + (" " if tail and tail[0].isspace() else "") + tail.strip()


_EVIDENCE_SOURCE_KO = {
    EvidenceSource.TRANSCRIPT: "발화 전사",
    EvidenceSource.FACIAL_AU: "표정",
    EvidenceSource.POSTURE: "자세",
    EvidenceSource.DOLL_TRACKING: "인형 추적",
    EvidenceSource.VOICE_PROSODY: "음성 운율",
}


def completion_status_label(status: CompletionStatus) -> str:
    return _COMPLETION_STATUS_KO.get(status, status.value)


def response_mode_label(mode: ResponseMode) -> str:
    return _RESPONSE_MODE_KO.get(mode, mode.value)


def evidence_source_label(source: EvidenceSource) -> str:
    return _EVIDENCE_SOURCE_KO.get(source, source.value)


_POSTURE_KIND_KO = {
    "torso_tilt": "몸통 기울임",
    "sudden_movement": "갑작스러운 움직임",
}

_DOLL_KIND_KO = {
    "proximity_seeking": "인형에게 다가감",
    "sudden_movement": "갑작스러운 움직임",
    "tracking_lost": "인형 추적 소실",
}

_PROSODY_KIND_KO = {
    "pitch_high": "음높이 평소보다 높음",
    "pitch_low": "음높이 평소보다 낮음",
    "loudness_high": "목소리 평소보다 큼",
    "loudness_low": "목소리 평소보다 작음",
}


def posture_event_kind_label(kind: str) -> str:
    return _POSTURE_KIND_KO.get(kind, kind)


def doll_event_kind_label(kind: str) -> str:
    return _DOLL_KIND_KO.get(kind, kind)


def prosody_event_kind_label(kind: str) -> str:
    return _PROSODY_KIND_KO.get(kind, kind)


_AU_GLOSS = {
    "AU01": "눈썹 안쪽 올림",
    "AU02": "눈썹 바깥쪽 올림",
    "AU04": "눈썹 찌푸림",
    "AU05": "윗눈꺼풀 올림",
    "AU06": "뺨 올림",
    "AU07": "눈꺼풀 조임",
    "AU09": "코 주름",
    "AU10": "윗입술 올림",
    "AU12": "입꼬리 올림(웃음)",
    "AU14": "보조개(입꼬리 당김)",
    "AU15": "입꼬리 내림",
    "AU17": "턱 올림",
    "AU20": "입 가로 당김(공포·긴장)",
    "AU23": "입술 조임",
    "AU24": "입술 누름",
    "AU25": "입 벌림",
    "AU26": "턱 벌림",
    "AU28": "입술 빨기",
    "AU45": "눈 깜빡임",
}


def au_gloss(au_code: str) -> str:
    gloss = _AU_GLOSS.get(au_code)
    return f"{gloss}({au_code})" if gloss else f"표정 변화({au_code})"


@dataclass(frozen=True)
class DollLabelsConfig:
    classes: dict[str, str]  # doll_id -> display_name_ko


def load_doll_labels(path: Path | None = None) -> DollLabelsConfig:
    raw = yaml.safe_load((path or DEFAULT_DOLL_LABELS_PATH).read_text(encoding="utf-8"))
    classes = {entry["id"]: entry["display_name_ko"] for entry in raw.get("classes", [])}
    return DollLabelsConfig(classes=classes)


def doll_id_label(doll_labels: DollLabelsConfig, doll_id: str) -> str:
    return doll_labels.classes.get(doll_id, doll_id)
