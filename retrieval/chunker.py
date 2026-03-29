"""Sayfa duzeyindeki PDF kayitlari icin metni kucuk parcalara ayirir."""

from typing import TypedDict

from config import CHUNK_OVERLAP, CHUNK_SIZE
from retrieval.pdf_loader import PageRecord


MIN_CHUNK_LENGTH = 150


class ChunkRecord(TypedDict):
    """Kaynak metaverisi tasiyan yapilandirilmis parca."""

    document: str
    page: int
    chunk_id: str
    text: str


def chunk_page_records(
    records: list[PageRecord],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """Sayfa kayitlarini kararli metaveri ile okunabilir parcalara ayirir."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunked_records: list[ChunkRecord] = []

    for record in records:
        text_chunks = _chunk_text(record["text"], chunk_size, chunk_overlap)
        for index, chunk_text in enumerate(text_chunks, start=1):
            chunked_records.append(
                {
                    "document": record["document"],
                    "page": record["page"],
                    "chunk_id": f'{record["document"]}-p{record["page"]}-c{index}',
                    "text": chunk_text,
                }
            )

    return chunked_records


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Bir sayfayi cumle farkindaligi olan ve overlap iceren parcalara ayirir."""

    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    segments = _split_segments(normalized_text)
    chunks: list[str] = []
    current = ""

    for segment in segments:
        if len(segment) > chunk_size:
            chunks = _flush_current(chunks, current)
            current = ""
            for piece in _split_long_segment(segment, chunk_size):
                chunks = _append_chunk(chunks, piece)
            continue

        candidate = segment if not current else f"{current} {segment}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks = _flush_current(chunks, current)
        current = ""
        overlap_text = _tail_overlap(chunks[-1], chunk_overlap) if chunks else ""
        current = segment if not overlap_text else f"{overlap_text} {segment}"

        if len(current) > chunk_size:
            for piece in _split_long_segment(current, chunk_size):
                chunks = _append_chunk(chunks, piece)
            current = ""

    chunks = _flush_current(chunks, current)
    return chunks


def _flush_current(chunks: list[str], current: str) -> list[str]:
    """Mevcut parcayi yararli metin iceriyorsa listeye ekler."""

    if current.strip():
        chunks = _append_chunk(chunks, current)
    return chunks


def _append_chunk(chunks: list[str], text: str) -> list[str]:
    """Bir parcayi ekler ve cok kucuk kalan kisimlari onceki parcayla birlestirir."""

    cleaned = text.strip()
    if not cleaned:
        return chunks

    if len(cleaned) < MIN_CHUNK_LENGTH and chunks:
        chunks[-1] = f"{chunks[-1]} {cleaned}".strip()
        return chunks

    chunks.append(cleaned)
    return chunks


def _normalize_text(text: str) -> str:
    """Paragraf sinirlarini korurken bosluk yapisini normalize eder."""

    paragraphs = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return "\n".join(paragraphs).strip()


def _split_segments(text: str) -> list[str]:
    """Metni paragraf farkindaligi olan cumle benzeri bolumlere ayirir."""

    segments: list[str] = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        buffer = ""
        for char in paragraph:
            buffer += char
            if char in ".!?;:" and len(buffer.strip()) >= 40:
                segments.append(buffer.strip())
                buffer = ""

        if buffer.strip():
            segments.append(buffer.strip())

    return segments


def _split_long_segment(text: str, chunk_size: int) -> list[str]:
    """Cok uzun bir bolumu sirayi bozmadan kelimelere gore ayirir."""

    words = text.split()
    if not words:
        return []

    parts: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        parts.append(current.strip())
        current = word

    if current.strip():
        parts.append(current.strip())

    return parts


def _tail_overlap(text: str, overlap_size: int) -> str:
    """Sonraki parca icin kelime sinirina hizali bir kuyruk kesiti dondurur."""

    if overlap_size <= 0:
        return ""
    if len(text) <= overlap_size:
        return text.strip()

    tail = text[-overlap_size:]
    split_at = tail.find(" ")
    if split_at == -1:
        return tail.strip()
    return tail[split_at + 1 :].strip()
