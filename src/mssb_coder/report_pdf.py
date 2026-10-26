"""학부모 보고서 PDF — report.build_session_report_markdown()의 결과를 그대로 PDF로 옮긴다.

보고서 내용(섹션·문장)은 report.py가 정하고, 이 모듈은 모양만 담당한다. 보고서 마크다운이 쓰는
요소(#·##·### 제목, 문단, "- " 목록, **굵게**, "> " 안내 문구, "---" 구분선)만 처리한다.
한글은 저장소에 함께 둔 나눔고딕(assets/fonts, SIL OFL)으로 넣는다 — Streamlit Cloud에도 그대로 올라간다.
"""

from __future__ import annotations

import re
from pathlib import Path

from mssb_coder.coding_systems import CodingSystem
from mssb_coder.report import build_session_report_markdown
from mssb_coder.schema import SessionSynthesis, StemCodingResult
from mssb_coder.segmentation import StoryStemsConfig

FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
FONT = "NanumGothic"
BODY_SIZE = 10.5
LINE_HEIGHT = 6.5
HEADING_SIZES = {1: 17, 2: 13.5, 3: 11.5}
MUTED = (100, 100, 100)

_BOLD_SPLIT = re.compile(r"(\*\*[^*]+\*\*)")


def _new_pdf(title: str):
    from fpdf import FPDF  # noqa: PLC0415 — 보고서 화면만 쓰는 의존성이라 필요할 때 불러온다

    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font(FONT, "", str(FONT_DIR / "NanumGothic.ttf"))
    pdf.add_font(FONT, "B", str(FONT_DIR / "NanumGothicBold.ttf"))
    pdf.set_title(title)
    pdf.set_creator("MSSB AI 코딩 보조 도구")
    pdf.add_page()
    return pdf


def _write_inline(pdf, text: str, size: float = BODY_SIZE) -> None:
    """**굵게** 구간만 굵은 글꼴로 바꿔 가며 이어 쓴다 (줄바꿈은 fpdf가 처리)."""
    for part in _BOLD_SPLIT.split(text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        pdf.set_font(FONT, "B" if bold else "", size)
        pdf.write(LINE_HEIGHT, part[2:-2] if bold else part)
    pdf.ln(LINE_HEIGHT)


def markdown_to_pdf(markdown: str, title: str) -> bytes:
    pdf = _new_pdf(title)
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            pdf.ln(2)
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            level = len(heading.group(1))
            pdf.ln(4 if level > 1 else 0)
            pdf.set_font(FONT, "B", HEADING_SIZES[level])
            pdf.multi_cell(0, HEADING_SIZES[level] * 0.6, heading.group(2), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif line == "---":
            pdf.ln(3)
            y = pdf.get_y()
            pdf.set_draw_color(*MUTED)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(3)
        elif line.startswith("> "):
            pdf.set_text_color(*MUTED)
            _write_inline(pdf, line[2:], size=BODY_SIZE - 1)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith("- "):
            _write_inline(pdf, "• " + line[2:])
        else:
            _write_inline(pdf, line)
    return bytes(pdf.output())


def build_session_report_pdf(
    synthesis: SessionSynthesis,
    stem_results: list[StemCodingResult] | None = None,
    coding_system: CodingSystem | None = None,
    story_stems: StoryStemsConfig | None = None,
) -> bytes:
    markdown = build_session_report_markdown(
        synthesis, stem_results=stem_results, coding_system=coding_system, story_stems=story_stems
    )
    return markdown_to_pdf(markdown, title=f"MSSB 세션 결과 — {synthesis.session_id}")
