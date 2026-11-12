"""text extraction for text-based cv pdfs. fails loudly instead of guessing."""

import logging
import re
from dataclasses import dataclass

import pymupdf

from src.schemas import ScreeningError

logger = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 15
MAX_TEXT_CHARS = 60_000
MIN_TEXT_CHARS = 50


class PDFParseError(ScreeningError):
    """raised with a code: empty_file, too_large, unsupported_format, corrupt_pdf,
    encrypted_pdf, empty_document, too_many_pages, scanned_pdf, no_text,
    unreadable_text, too_long."""


@dataclass(frozen=True)
class ParsedPDF:
    text: str
    page_count: int
    char_count: int


def extract_text(data: bytes) -> ParsedPDF:
    """return the embedded text of a pdf, or raise PDFParseError. never ocr, never invent."""
    # step 1: cheap checks before touching the parser
    if not data:
        raise PDFParseError("empty_file", "the uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise PDFParseError("too_large", f"file exceeds {MAX_BYTES // (1024 * 1024)} MB.")
    if b"%PDF-" not in data[:1024]:
        raise PDFParseError("unsupported_format", "file is not a pdf (missing %PDF header).")

    # step 2: open
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
        raise PDFParseError("corrupt_pdf", "the pdf could not be opened; it may be corrupt.") from exc

    with doc:
        if doc.needs_pass:
            raise PDFParseError("encrypted_pdf", "the pdf is password protected.")
        if doc.page_count == 0:
            raise PDFParseError("empty_document", "the pdf has no pages.")
        if doc.page_count > MAX_PAGES:
            raise PDFParseError("too_many_pages", f"pdf has more than {MAX_PAGES} pages.")

        # step 3: read text and count embedded images (used to tell scanned from blank)
        try:
            pages = [page.get_text("text") for page in doc]
            image_count = sum(len(page.get_images()) for page in doc)
        except Exception as exc:  # pymupdf raises assorted types on damaged pages
            raise PDFParseError("corrupt_pdf", "text extraction failed; the pdf may be damaged.") from exc
        page_count = doc.page_count

    text = "\n".join(pages).strip()
    visible_chars = len(re.sub(r"\s+", "", text))

    # step 4: classify unusable results
    if visible_chars < MIN_TEXT_CHARS:
        if image_count > 0:
            raise PDFParseError(
                "scanned_pdf",
                "no extractable text but the pdf contains images; it looks scanned. "
                "ocr is not supported, please upload a text-based pdf.",
            )
        raise PDFParseError("no_text", "the pdf contains no extractable text.")
    if text.count("�") / max(len(text), 1) > 0.2:
        raise PDFParseError("unreadable_text", "extracted text is mostly unreadable characters.")
    if len(text) > MAX_TEXT_CHARS:
        raise PDFParseError("too_long", f"extracted text exceeds {MAX_TEXT_CHARS} characters.")

    # only sizes are logged, never content
    logger.info("pdf parsed: pages=%d chars=%d", page_count, len(text))
    return ParsedPDF(text=text, page_count=page_count, char_count=len(text))
