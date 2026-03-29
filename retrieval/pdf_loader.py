"""Yerel PDF dosyalarindan sayfa duzeyinde metin kayitlari yukleme araclari.PDF’lerden metni sayfa sayfa ceker."""

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from config import PDF_DIR


class PageRecord(TypedDict):
    """Tek bir PDF sayfasindan cikarilan yapilandirilmis metin."""

    document: str
    page: int
    text: str


def load_pdf_pages(pdf_dir: Path = PDF_DIR) -> list[PageRecord]:
    """Bir klasordeki tum PDF dosyalarini yukler ve bos olmayan sayfa kayitlarini dondurur."""

    records: list[PageRecord] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        records.extend(load_pdf_file(pdf_path))

    return records


def load_pdf_file(pdf_path: Path) -> list[PageRecord]:
    """Tek bir PDF dosyasini yukler ve metni sayfa sayfa cikarir."""

    reader = PdfReader(str(pdf_path))
    records: list[PageRecord] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        records.append(
            {
                "document": pdf_path.name,
                "page": page_number,
                "text": text,
            }
        )

    return records
