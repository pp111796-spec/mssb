import pytest

pytest.importorskip("fpdf")

from factories import stem_result, synthesis  # noqa: E402
from mssb_coder.report_pdf import build_session_report_pdf, markdown_to_pdf  # noqa: E402


def test_report_pdf_is_a_pdf_with_korean_font_embedded():
    pdf = build_session_report_pdf(synthesis(), [stem_result("spilled_juice"), stem_result("lost_dog")])

    assert pdf.startswith(b"%PDF")
    assert b"NanumGothic" in pdf
    assert len(pdf) > 10_000


def test_markdown_elements_render_without_errors():
    markdown = "\n".join([
        "# 제목", "", "> 안내 문구", "", "## 섹션", "", "**굵게** 이어지는 문장이다.", "- 목록 항목", "", "---",
        "### 작은 제목", "아주 긴 문장이다. " * 80,
    ])

    pdf = markdown_to_pdf(markdown, title="테스트")

    assert pdf.startswith(b"%PDF")
