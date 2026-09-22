"""세션 결과를 "내보내기 번들" 하나로 묶고 다시 읽는 로직.

공개 사이트(src/public_app/app.py)는 MSSB_DATA_DIR에 접근하지 못한다 — 로컬에서 만든
이 번들 파일을 사람이 직접 업로드해야만 결과를 볼 수 있다. 영상 원본은 번들에 절대
포함되지 않는다(애초에 이 모듈이 다루는 대상이 아님) — synthesis·stem_results라는
이미 다 처리된 텍스트/구조화 결과만 담는다.

이 모듈 자체는 디스크(MSSB_DATA_DIR)에 의존하지 않는다 — 로컬 앱·공개 앱 양쪽에서
그대로 재사용할 수 있게 순수 직렬화/검증 로직만 둔다.
"""

from __future__ import annotations

from mssb_coder.schema import SessionSynthesis, StemCodingResult

BUNDLE_FORMAT_VERSION = 1


def build_bundle_dict(
    synthesis: SessionSynthesis, stem_results: list[StemCodingResult]
) -> dict:
    return {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "synthesis": synthesis.model_dump(mode="json"),
        "stem_results": [r.model_dump(mode="json") for r in stem_results],
    }


class BundleParseError(ValueError):
    pass


def parse_bundle_dict(raw: dict) -> tuple[SessionSynthesis, list[StemCodingResult]]:
    if "synthesis" not in raw:
        raise BundleParseError("번들에 'synthesis' 필드가 없음 — 세션 내보내기 파일이 맞는지 확인하세요.")
    try:
        synthesis = SessionSynthesis.model_validate(raw["synthesis"])
        stem_results = [
            StemCodingResult.model_validate(r) for r in raw.get("stem_results", [])
        ]
    except Exception as exc:  # pydantic ValidationError 등
        raise BundleParseError(f"번들 형식이 올바르지 않음: {exc}") from exc
    return synthesis, stem_results
