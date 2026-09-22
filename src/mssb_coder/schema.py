"""Pydantic models for the MSSB AI coding pipeline.

Mirrors the JSON schemas in the plan document (iterative-wondering-ocean.md):
  - "스템별 코딩 결과 JSON 스키마"
  - "세션 종합 패턴 초안 JSON 스키마"
  - segmentation.py's stem_timestamps.json output (파이프라인 3단계)

These models are the pydantic-validation gate the plan requires after every
Claude tool_use call (pipeline steps 3, 10, 12) before anything is written to disk.
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
        if not (self.start_s <= self.handoff_timestamp_s <= self.end_s):
            raise ValueError(
                f"{self.stem_name}: handoff_timestamp_s는 start_s와 end_s 사이여야 함"
            )
        for seg in self.additional_examiner_segments_s:
            if not (self.start_s <= seg.start_s < seg.end_s <= self.end_s):
                raise ValueError(
                    f"{self.stem_name}: additional_examiner_segments_s가 스템 구간을 벗어남"
                )
        return self


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


class SessionMetadata(BaseModel):
    session_id: str
    child_age_months: int = Field(gt=0)
    child_gender: Literal["male", "female"] = Field(
        description="departure(여행 떠나기) 스템의 화자 판별에 필요"
    )
    subject_type: SubjectType = Field(
        description="adult=성인 대역(마일스톤 A, 배선 검증용), child=실제 아동(마일스톤 B)"
    )
