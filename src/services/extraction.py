from io import BytesIO

from pypdf import PdfReader

from src.core.custom_exception import ExtractionError
from src.schemas.ingestion import FileType


def _extract_pdf(content: bytes) -> str:
    """Extract and combine text from a pdf"""
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ExtractionError("Invalid or corrupted PDF") from exc

    if reader.is_encrypted:
        raise ExtractionError("Encrypted PDFs are not supported")

    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            raise ExtractionError(f"Failed to read page {page_number}") from exc

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def _extract_txt(content: bytes) -> str:
    """Decode a text file."""
    try:
        return content.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return content.decode("latin-1").strip()


def extract_text(
    content: bytes,
    file_type: FileType,
) -> str:
    """Extract text from a supported document."""
    if file_type == FileType.PDF:
        text = _extract_pdf(content)
    elif file_type == FileType.TXT:
        text = _extract_txt(content)
    else:
        raise ExtractionError(f"Unsupported file type: {file_type}")

    if not text:
        raise ExtractionError(
            "No extractable text found (file may be empty or a scanned document)"
        )

    return text
