import tempfile
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Mengimpor modul internal yang sudah dirapikan
from document_processor import process_document
from llm_engine import DEFAULT_MODEL_PATH, generate_answer_with_context, load_llm_model
from retriever import load_embedding_model, retrieve_relevant_chunks
from schemas import AskResponse, TokenUsage


# Konfigurasi Aplikasi

ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf"}  # file yang didukung untuk diunggah

app = FastAPI(
    title="AskDocx API",
    description="Sistem RAG lokal yang transparan menggunakan FastAPI, NumPy, dan LLM lokal.",
    version="1.0.0"
)


# DEPENDENCY INJECTION (CACHE MODELS)

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Memuat model embedding satu kali saat aplikasi berjalan (Singleton)."""
    try:
        return load_embedding_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Gagal memuat embedding model: {e}"
        )

@lru_cache(maxsize=1)
def get_llm_model() -> Llama:
    """Memuat model LLM lokal satu kali saat aplikasi berjalan (Singleton)."""
    try:
        # Pastikan DEFAULT_MODEL_PATH mengarah ke folder models/model.gguf
        return load_llm_model(model_path=DEFAULT_MODEL_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Gagal memuat LLM: {e}"
        )

# ENDPOINTS

@app.get("/status", tags=["System"])  
def check_system_status():             
    """Mengecek apakah AskDocx sudah online."""
    return {
        "status": "Online",
        "service": "AskDocx Assistant",
        "message": "Sistem berjalan dengan lancar."
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Tanya Jawab Berdasarkan Dokumen",
    tags=["RAG"]
)
async def ask_question(
    question: Annotated[str, Form(description="Pertanyaan untuk AI")],
    document: Annotated[UploadFile, File(description="File dokumen (.txt / .docx / .pdf)")],
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
    llm: Llama = Depends(get_llm_model),
    top_k: int = Form(default=3)
) -> AskResponse:
    """
    Menerima dokumen dan pertanyaan, memproses RAG, dan mengembalikan jawaban.
    """
    
    # 1. Validasi Input Dasar
    if not question.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Pertanyaan tidak boleh kosong.")
    
    file_ext = Path(document.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, 
            f"Format {file_ext} tidak didukung. Gunakan .txt atau .docx"
        )

    # 2. Simpan file sementara dengan aman
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await document.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)
            
        # 3. Jalankan Pipeline RAG
        
        # A. Ingestion & Chunking (Nama fungsi disesuaikan: process_document)
        chunks = process_document(temp_path)
        
        if not chunks:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Dokumen kosong atau teks tidak terbaca.")

        # B. Retrieval
        retrieved_chunks, scores = retrieve_relevant_chunks(
            query_text=question,
            document_chunks=chunks,
            embedding_model=embedding_model,
            top_k=top_k
        )

        # C. Generation (LLM)
        llm_result = generate_answer_with_context(
            user_question=question,
            retrieved_chunks=retrieved_chunks,
            llm_instance=llm
        )

        # 4. Kembalikan Response Terstruktur
        return AskResponse(
            answer=llm_result["answer"],
            retrieved_documents=retrieved_chunks,
            similarity_scores=scores,
            prompt_used=llm_result["prompt_used"],
            # Mapping ke TokenUsageEstimation sesuai file schemas.py
            token_usage_estimation=TokenUsage(**llm_result["token_usage"])
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[ERROR] RAG Pipeline failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Kesalahan internal saat memproses dokumen: {str(e)}"
        )
    
    finally:
        # Selalu bersihkan file sementara agar penyimpanan tidak penuh
        if temp_path and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    # Reload=True hanya disarankan untuk tahap development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)