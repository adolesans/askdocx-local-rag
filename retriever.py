import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
MIN_SIMILARITY_THRESHOLD = 0.0

# Manajemen model

def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Memuat model Sentence-Transformer ke memori.
    
    Dipilih 'all-MiniLM-L6-v2' karena sangat efisien (hanya ~22MB) namun 
    akurat untuk mencari kemiripan semantik pada CPU lokal.
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as error:
        raise RuntimeError(
            f"Gagal memuat model embedding '{model_name}': {error}"
        ) from error


# Proses embedding teks dan retrieval


def generate_text_embedding(
    text: str, 
    embedding_model: SentenceTransformer
) -> NDArray[np.float32]:
    """
    Mengubah satu kalimat/pertanyaan menjadi vektor berdimensi tetap.
    
    Normalisasi L2 diterapkan agar panjang vektor menjadi 1, sehingga 
    kalkulasi kemiripan nantinya cukup menggunakan Dot Product sederhana.
    """
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embedding.astype(np.float32)


def generate_batch_embeddings(
    text_list: List[str], 
    embedding_model: SentenceTransformer
) -> NDArray[np.float32]:
    """
    Mengubah daftar teks menjadi matriks vektor secara massal (Batch).
    
    Menggunakan batching jauh lebih cepat daripada looping manual karena 
    memanfaatkan optimasi internal pada pustaka Sentence-Transformers.
    """
    if not text_list:
        return np.array([], dtype=np.float32)

    embeddings = embedding_model.encode(
        text_list,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)


# Logika Cosine Similarity

def calculate_cosine_similarity(
    query_vector: NDArray[np.float32], 
    document_vector: NDArray[np.float32]
) -> float:
    """
    Menghitung skor kemiripan antara dua vektor secara manual.
    Rumus: (A · B) / (||A|| * ||B||)
    """
    dot_product = np.dot(query_vector, document_vector)
    
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(document_vector)

    # Menghindari pembagian dengan nol jika ditemukan vektor kosong
    if query_norm == 0 or doc_norm == 0:
        return 0.0

    score = dot_product / (query_norm * doc_norm)
    return float(score)


def calculate_all_similarity_scores(
    query_vector: NDArray[np.float32], 
    document_embedding_matrix: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Menghitung skor kemiripan antara satu query terhadap seluruh dokumen sekaligus.
    
    Memanfaatkan perkalian matriks NumPy yang sudah teroptimasi pada level 
    rendah (BLAS) untuk kecepatan maksimal.
    """
    # Operasi Linear Algebra: Matriks @ Vektor
    scores = np.dot(document_embedding_matrix, query_vector)
    return scores.astype(np.float32)


# LOGIKA RETRIEVAL(PENCARIAN)

def find_top_k_indices(
    similarity_scores: NDArray[np.float32], 
    top_k: int
) -> NDArray[np.int64]:
    """
    Mengambil indeks dari skor tertinggi dalam urutan menurun (descending).
    """
    # argsort secara default mengurutkan terkecil ke terbesar, maka perlu dibalik [::-1]
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    limit = min(top_k, len(similarity_scores))
    return sorted_indices[:limit]


def retrieve_relevant_chunks(
    query_text: str,
    document_chunks: List[str],
    embedding_model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[str], List[float]]:
    """
    Fungsi Orkestrator Utama untuk melakukan Retrieval.
    
    Alur:
    1. Embed Query & Chunks -> 2. Hitung Similarity -> 3. Sorting -> 4. Return Results
    """
    if not document_chunks:
        raise ValueError("Dokumen kosong. Tidak ada data yang bisa di-retrieve.")

    # 1. Transformasi Teks ke Vektor
    query_vec = generate_text_embedding(query_text, embedding_model)
    chunk_matrix = generate_batch_embeddings(document_chunks, embedding_model)

    # 2. Kalkulasi Skor Kemiripan
    scores = calculate_all_similarity_scores(query_vec, chunk_matrix)

    # 3. Pengurutan Hasil Terbaik
    best_indices = find_top_k_indices(scores, top_k)

    # 4. Perakitan Hasil Akhir
    results = [document_chunks[idx] for idx in best_indices]
    final_scores = [round(float(scores[idx]), 4) for idx in best_indices]

    return results, final_scores