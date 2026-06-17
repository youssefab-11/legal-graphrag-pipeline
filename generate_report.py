"""Generate architecture_report.pdf from architecture_report.md.

Simple Markdown-to-PDF converter using fpdf2. Keeps the report self-contained
and reproducible without requiring pandoc or LaTeX.
"""

import argparse
import logging
import re
from pathlib import Path

from fpdf import FPDF, XPos, YPos

from src.config.settings import settings

logger = logging.getLogger(__name__)


def sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters unsupported by core Latin-1 fonts."""
    replacements = {
        "\u2014": "--",  # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201C": '"',   # left double quote
        "\u201D": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00A0": " ",   # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop any remaining non-Latin1 characters to avoid encoding errors
    return text.encode("latin-1", "ignore").decode("latin-1")


class PDFReport(FPDF):
    """Simple PDF report renderer using built-in core fonts."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Helvetica", "", 11)

    def add_heading(self, text: str, level: int) -> None:
        """Add a Markdown heading with appropriate size."""
        sizes = {1: 18, 2: 14, 3: 12}
        size = sizes.get(level, 11)
        self.set_font("Helvetica", "B", size)
        self.ln(4)
        self.cell(0, 8, sanitize_for_pdf(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 11)

    def add_paragraph(self, text: str) -> None:
        """Add a paragraph of body text."""
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, sanitize_for_pdf(text))
        self.ln(2)

    def add_code_block(self, lines: list) -> None:
        """Add a code block with monospace font."""
        self.set_font("Courier", "", 9)
        for line in lines:
            self.cell(0, 5, sanitize_for_pdf(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 11)
        self.ln(2)


def markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    """Convert a Markdown file to a basic PDF."""
    text = input_path.read_text(encoding="utf-8")
    pdf = PDFReport()

    lines = text.splitlines()
    code_buffer: list = []
    in_code = False

    for line in lines:
        stripped = line.strip()

        # Code block toggle
        if stripped.startswith("```"):
            if in_code and code_buffer:
                pdf.add_code_block(code_buffer)
                code_buffer = []
            in_code = not in_code
            continue

        if in_code:
            code_buffer.append(line)
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            pdf.add_heading(heading_match.group(2), level)
            continue

        # Horizontal rule
        if stripped == "---":
            pdf.ln(2)
            continue

        # Empty line
        if not stripped:
            pdf.ln(2)
            continue

        # Bold text cleanup
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        pdf.add_paragraph(clean)

    if code_buffer:
        pdf.add_code_block(code_buffer)

    pdf.output(str(output_path))
    logger.info("Generated PDF report: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate architecture report PDF")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("architecture_report.md"),
        help="Input Markdown file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("architecture_report.pdf"),
        help="Output PDF file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=settings.LOG_LEVEL)
    markdown_to_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
