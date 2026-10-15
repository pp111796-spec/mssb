"""Pydantic models for the MSSB AI coding pipeline.

Mirrors the JSON schemas in the plan document (iterative-wondering-ocean.md):
  - "스템별 코딩 결과 JSON 스키마"
  - "세션 종합 패턴 초안 JSON 스키마"
  - segmentation.py's stem_timestamps.json output (파이프라인 3단계)

These models are the pydantic-validation gate the plan requires after every
Claude call (pipeline steps 3, 10, 12) before anything is written to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

STANDARD_DISCLAIMER = (
    "이 결과는 진단이 아니라 이해를 돕기 위한 선별·탐색 목적이며, "
    "정확한 평가는 전문가 상담을 통해 받으실 수 있습니다."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubjectType(str, Enum):
    """cv_thresholds_{adult,child}.yaml / facial_expression.py 도구 스위치 기준 (계획 '설정 프로파일 분리')."""

    ADULT = "adult"
    CHILD = "child"


class CompletionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_ADMINISTERED = "not_administered"


class ResponseMode(str, Enum):
    SPONTANEOUS = "spontaneous"
    PROMPTED = "prompted"


class EvidenceSource(str, Enum):
    TRANSCRIPT = "transcript"
    FACIAL_AU = "facial_au"
    POSTURE = "posture"
    DOLL_TRACKING = "doll_tracking"
    VOICE_PROSODY = "voice_prosody"


# ---------------------------------------------------------------------------
# 3단계: 스템 구간 자동 분리 결과 (stem_timestamps.json)
# ---------------------------------------------------------------------------


class ExaminerSegment(BaseModel):
    """아이 차례 도중 검사자가 다시 끼어드는 구간 (예: 뽀삐 재회 즉흥 질문)."""

    start_s: float
    end_s: float


class StemTimestamp(BaseModel):
    stem_name: str
    start_s: float
    end_s: float
    handoff_timestamp_s: float = Field(
        description="검사자의 완성요청이 끝나고 아이 차례가 시작되는 지점"
    )
    additional_examiner_segments_s: list[ExaminerSegment] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    matched_cue: str = Field(description="어떤 문구/키워드로 이 구간을 찾았는지 (감사용)")
    scored: bool = Field(
        default=True, description="워밍업/윈드다운 등 채점 대상이 아니면 False"
    )

    @model_validator(mode="after")
    def _check_ordering(self) -> "StemTimestamp":
        if not self.start_s < self.end_s:
            raise ValueError(f"{self.stem_name}: 끝 시각이 시작 시각보다 뒤여야 함")
        if not (self.start_s <= self.handoff_timestamp_s <= self.end_s):
            raise ValueError(f"{self.stem_name}: 검사자→아동 전환 시각은 시작과 끝 사이여야 함")
        # 채점 대상인데 아동 차례 길이가 0이면 분석할 게 없다 (녹화 앞뒤 여백 같은 비채점 구간은 허용)
        if self.scored and self.handoff_timestamp_s == self.end_s:
            raise ValueError(f"{self.stem_name}: 채점 대상 이야기인데 아동 차례가 없음 (전환 시각 = 끝 시각)")
        for seg in self.additional_examiner_segments_s:
            if not (self.start_s <= seg.start_s < seg.end_s <= self.end_s):
                raise ValueError(
                    f"{self.stem_name}: additional_examiner_segments_s가 스템 구간을 벗어남"
                )
        return self


class StemWindow(BaseModel):
    """CV 수치·채점 결과가 어떤 구간으로 만들어졌는지 — 구간이 바뀌면 다시 계산해야 하는지 판단하는 기준."""

    start_s: float
    end_s: float
    handoff_timestamp_s: float

    @classmethod
    def of(cls, stem: StemTimestamp) -> "StemWindow":
        return cls(start_s=stem.start_s, end_s=stem.end_s, handoff_timestamp_s=stem.handoff_timestamp_s)


# ---------------------------------------------------------------------------
# 10단계: 스템별 AI 코딩 결과 (<stem>_ai_codes.json)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    timestamp_s: float
    source: EvidenceSource
    note: str = ""


class ItemCode(BaseModel):
    present: bool
    evidence: list[Evidence] = Field(default_factory=list)


class CategoryCode(BaseModel):
    items: dict[str, ItemCode]
    present_count: int = Field(ge=0)
    rationale: str

    @model_validator(mode="after")
    def _check_present_count(self) -> "CategoryCode":
        actual = sum(1 for item in self.items.values() if item.present)
        if actual != self.present_count:
            raise ValueError(
                f"present_count({self.present_count})가 실제 존재 항목 수({actual})와 다름"
            )
        return self


# mssb_5category.yaml의 category id와 반드시 일치해야 함 (coding_systems.py가 강제)
CategoryId = Literal[
    "empathy_warmth",
    "performance_anxiety",
    "avoidant_withdrawal",
    "dysregulated_aggression",
    "emotional_integration",
]


class StemCodingResult(BaseModel):
    session_id: str
    child_age_months: int = Field(gt=0, description="매뉴얼 §10.2·§11 — 연령 기대치 감안 채점에 필수")
    stem_name: str
    completion_status: CompletionStatus
    response_mode: ResponseMode
    issue_prompt_count: int = Field(default=0, ge=0, le=2, description="쟁점 프롬프트는 이야기당 최대 2회 (§6.6)")
    codes: dict[CategoryId, CategoryCode]
    story_specific_notes_applied: str = Field(
        description="해당 스템의 코드 의미 주의사항 원문 (감사용, story_stems.yaml의 story_specific_notes)"
    )
    interpretation_layer: Literal["1_single_story_observation"] = "1_single_story_observation"
    disclaimer: str = STANDARD_DISCLAIMER
    model_version: str
    # 아래 두 필드는 예전 결과 파일엔 없다 — None이면 "확인 불가"로 다룬다.
    source_window: StemWindow | None = None  # 채점에 쓴 구간 (구간이 같으면 재채점 생략)
    timestamp_basis: Literal["session_absolute"] | None = None  # 근거 시각이 세션 절대 시각인지
    generated_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# 12단계: 세션 종합 패턴 초안 (session_summary_ai_draft.json)
# ---------------------------------------------------------------------------


class SynthesisEvidence(BaseModel):
    stem_name: str
    timestamp_s: float
    note: str = ""


class RecurringPattern(BaseModel):
    category: CategoryId
    observed_in_stems: list[str]
    stems_administered_count_for_pattern: int = Field(
        ge=0, description="이 패턴 판단에 사용된 유효 스템 수 (분모) — 문제점.md 옛 3번 해결"
    )
    pattern_description: str
    evidence: list[SynthesisEvidence]


class SingleStoryNotableSignal(BaseModel):
    stem_name: str
    category: CategoryId
    note: str


class SessionSynthesis(BaseModel):
    session_id: str
    stems_included: list[str]
    stems_missing_or_partial: list[str] = Field(default_factory=list)
    stems_administered_count: int = Field(ge=0, description="세션 전체에서 정상 실시된 스템 수 (분모)")
    recurring_patterns: list[RecurringPattern]
    single_story_notable_signals: list[SingleStoryNotableSignal] = Field(default_factory=list)
    # 보고서 섹션용 서술 필드 — 예전에 저장된 JSON(필드 없음)도 읽혀야 하므로 전부 기본값을 둔다.
    behavioral_observation: str = Field(
        default="", description="검사 중 행동관찰: 검사 태도·협조도·비언어 단서(표정/자세/목소리/인형 조작) 서술"
    )
    strengths: str = Field(default="", description="이야기 전반에서 잘 나타난 강점 서술")
    support_areas: str = Field(default="", description="살펴보거나 지원이 필요해 보이는 영역 서술 (진단 아님)")
    recommendations: list[str] = Field(
        default_factory=list, description="가정·기관에서의 지원 제언 (AI 초안, 검사자 확인 필수 — 진단·위험분류 아님)"
    )
    not_included_by_design: str = (
        "이 요약은 다른 자료(부모 보고·행동관찰 등)와의 통합이나 임상적 진단·위험 분류를 포함하지 않는다"
    )
    interpretation_layer: Literal["2_cross_story_pattern"] = "2_cross_story_pattern"
    based_on: Literal["ai_draft_stem_results"] = "ai_draft_stem_results"
    disclaimer: str = STANDARD_DISCLAIMER
    model_version: str
    generated_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# 세션 메타데이터 (1단계 입력)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 축어록(언어+행동분석 통합) — <stem>_verbatim.json
# ---------------------------------------------------------------------------


class VerbatimSpeaker(str, Enum):
    EXAMINER = "examiner"
    CHILD = "child"


class VerbatimLine(BaseModel):
    """축어록 한 줄 — 발화(speech) 또는 행동 신호(behavior) 중 하나."""

    timestamp_s: float  # 세션 절대 시각 (StemTimestamp.start_s와 같은 축)
    kind: Literal["speech", "behavior"]
    speaker: VerbatimSpeaker | None = None  # kind=="speech"일 때만
    text: str | None = None  # kind=="speech"일 때 전사 텍스트
    source: EvidenceSource | None = None  # kind=="behavior"일 때
    label_ko: str | None = None  # kind=="behavior"일 때 저장 시점에 완성된 한글 서술
    raw_value: float | None = None  # 감사용 원본 수치(각도/zscore 등) — 화면엔 label_ko만 노출


class StemVerbatimRecord(BaseModel):
    """스템 1편의 언어+행동 통합 축어록. 영상/이미지는 담지 않고, 이미 계산된 타임스탬프·텍스트만."""

    session_id: str
    stem_name: str
    lines: list[VerbatimLine]
    generated_at: str = Field(default_factory=utc_now_iso)
    # Claude 호출 아님(순수 결합 로직) — 버전 표시만 위한 상수
    model_version: Literal["rule_based_join_v1"] = "rule_based_join_v1"


# ---------------------------------------------------------------------------
# 스템별 CV 원본 수치 타임라인 — cv/<stem>_cv.json (기준값 보정·이벤트 재계산용, Claude 호출 없음)
# ---------------------------------------------------------------------------
# 시간 기준은 기존 모듈과 같다: 표정·자세·인형 = 스템 시작 기준 상대 초, 음성 = handoff 기준 상대 초.


class CvFacialSample(BaseModel):
    t: float
    aus: dict[str, float]  # AU 확률(0~1)
    head_pitch_deg: float
    head_yaw_deg: float
    landmark_confidence: float


class CvPoseSample(BaseModel):
    t: float
    landmarks: dict[str, tuple[float, float]]
    is_facing_camera: bool


class CvProsodySample(BaseModel):
    t: float
    pitch_hz: float
    loudness: float


class CvDollSample(BaseModel):
    t: float
    doll_id: str
    bbox: tuple[float, float, float, float] | None  # 0~1 정규화 좌표, 탐지 실패면 None
    track_id: int | None = None


class StemCvTimeline(BaseModel):
    session_id: str
    stem_name: str
    fps: float
    facial: list[CvFacialSample] = Field(default_factory=list)
    pose: list[CvPoseSample] = Field(default_factory=list)
    prosody: list[CvProsodySample] = Field(default_factory=list)
    doll: list[CvDollSample] = Field(default_factory=list)
    # 이 수치를 계산할 때의 구간 — 구간을 고친 뒤 예전 수치를 재사용하지 않기 위한 표시 (예전 파일엔 없음)
    source_window: StemWindow | None = None
    generated_at: str = Field(default_factory=utc_now_iso)


class SessionMetadata(BaseModel):
    session_id: str
    child_age_months: int = Field(gt=0)
    child_gender: Literal["male", "female"] = Field(
        description="departure(여행 떠나기) 스템의 화자 판별에 필요"
    )
    subject_type: SubjectType = Field(
        description="adult=성인 대역(마일스톤 A, 배선 검증용), child=실제 아동(마일스톤 B)"
    )
    video_path: str | None = Field(
        default=None, description="원본 영상 경로 (MSSB_DATA_DIR 쪽, 검토 앱 미리보기·재분석용)"
    )
