import pytest

from src.pdf_parser import MAX_BYTES, PDFParseError, extract_text


def test_extracts_text_from_text_pdf(sample_pdf):
    parsed = extract_text(sample_pdf)
    assert parsed.page_count == 1
    assert "Data Analyst" in parsed.text
    assert parsed.char_count == len(parsed.text)


@pytest.mark.parametrize(
    "data, code",
    [
        (b"", "empty_file"),
        (b"just some plain text, not a pdf at all", "unsupported_format"),
        (b"%PDF-1.7\nthis is not a real pdf body", "corrupt_pdf"),
        (b"%PDF-" + b"0" * (MAX_BYTES + 1), "too_large"),
    ],
)
def test_rejects_bad_bytes(data, code):
    with pytest.raises(PDFParseError) as exc:
        extract_text(data)
    assert exc.value.code == code


def test_detects_scanned_pdf(scanned_pdf):
    with pytest.raises(PDFParseError) as exc:
        extract_text(scanned_pdf)
    assert exc.value.code == "scanned_pdf"


def test_detects_blank_pdf(blank_pdf):
    with pytest.raises(PDFParseError) as exc:
        extract_text(blank_pdf)
    assert exc.value.code == "no_text"


def test_detects_encrypted_pdf(encrypted_pdf):
    with pytest.raises(PDFParseError) as exc:
        extract_text(encrypted_pdf)
    assert exc.value.code == "encrypted_pdf"


def test_short_text_is_not_accepted(text_pdf_factory):
    with pytest.raises(PDFParseError) as exc:
        extract_text(text_pdf_factory("too short"))
    assert exc.value.code == "no_text"
