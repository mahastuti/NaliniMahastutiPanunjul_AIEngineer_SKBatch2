from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_pdf() -> bytes:
    return (ROOT / "samples" / "synthetic_cv.pdf").read_bytes()


@pytest.fixture
def sample_jd() -> str:
    return (ROOT / "samples" / "job_description.txt").read_text(encoding="utf-8")


def make_text_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=10)
    return doc.tobytes()


@pytest.fixture
def text_pdf_factory():
    return make_text_pdf


@pytest.fixture
def scanned_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


@pytest.fixture
def blank_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page()
    return doc.tobytes()


@pytest.fixture
def encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "secret content " * 10)
    return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
