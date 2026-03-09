import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import docx
import fitz


@dataclass(frozen=True)
class ChunkConfig:
    """Konfigurasi untuk pengaturan ukuran dan overlap chunk."""
    size: int = 5
    overlap: int = 1
    delimiter: str = r"(?<=[.!?])\s+"


class DocumentReader:
    """Menangani pembacaan berbagai format file ke dalam teks mentah."""

    @staticmethod
    def from_txt(path: Path) -> str:
        """Membaca file teks dengan penanganan error karakter."""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def from_docx(path: Path) -> str:
        """Mengekstrak teks dari paragraf file Word."""
        doc = docx.Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    
    @staticmethod # <--- Tambahkan ini
    def from_pdf(path: Path) -> str: 
        """Mengekstrak teks dari setiap halaman PDF."""
        import fitz
        text = ""
        with fitz.open(str(path)) as doc:
            for page in doc:
                text += page.get_text()
        return text

def extract_text(file_path: Path) -> str:
    """
    Router untuk menentukan pembaca file berdasarkan ekstensi.
    """
    readers: dict[str, Callable[[Path], str]] = {
        ".txt": DocumentReader.from_txt,
        ".docx": DocumentReader.from_docx,
        ".pdf": DocumentReader.from_pdf,
    }

    extension = file_path.suffix.lower()
    if extension not in readers:
        supported = ", ".join(readers.keys())
        raise ValueError(f"Format '{extension}' tidak didukung. Gunakan: {supported}")

    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan di: {file_path}")

    return readers[extension](file_path)


def split_to_sentences(text: str, delimiter: str) -> list[str]:
    """Memecah teks menjadi daftar kalimat bersih menggunakan regex."""
    return [s.strip() for s in re.split(delimiter, text) if s.strip()]


def chunk_text(text: str, config: ChunkConfig = ChunkConfig()) -> list[str]:
    """
    Memecah teks menjadi chunks dengan sistem overlap 
    untuk menjaga konteks antar potongan.
    """
    if not text.strip():
        return []

    sentences = split_to_sentences(text, config.delimiter)
    chunks: list[str] = []
    
    # Menentukan langkah pergeseran (step) berdasarkan overlap
    step = max(1, config.size - config.overlap)

    for i in range(0, len(sentences), step):
        # Mengambil potongan kalimat sesuai ukuran window
        window = sentences[i : i + config.size]
        chunks.append(" ".join(window))
        
        # Berhenti jika window terakhir sudah mencapai ujung kalimat
        if i + config.size >= len(sentences):
            break

    return chunks


def process_document(file_path: Path, config: ChunkConfig = ChunkConfig()) -> list[str]:
    """
    Fungsi utama untuk memproses file langsung menjadi chunks.
    """
    try:
        content = extract_text(file_path)
        return chunk_text(content, config)
    except Exception as e:
        # Menambah konteks pada error untuk memudahkan debugging
        raise RuntimeError(f"Gagal memproses dokumen {file_path.name}: {e}") from e