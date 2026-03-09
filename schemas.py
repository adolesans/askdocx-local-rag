from pydantic import BaseModel, Field

# SUB-SCHEMA (Estimasi penggunaan token)

class TokenUsage(BaseModel):
    prompt_tokens: int = Field(
        description="Jumlah perkiraan token dalam instruksi (context + query)."
    )
    completion_tokens: int = Field(
        description="Jumlah perkiraan token dalam jawaban yang dihasilkan AI."
    )
    total_tokens: int = Field(
        description="Total penggunaan token (Input + Output)."
    )



# SCHEMA UTAMA: Respons jawaban dengan konteks (Explainable RAG Response)

class AskResponse(BaseModel):
    answer: str = Field(
        description="Jawaban akhir yang dirangkai oleh AI."
    )
    retrieved_documents: list[str] = Field(
        description="Potongan teks asli dari dokumen yang dijadikan referensi."
    )
    similarity_scores: list[float] = Field(
        description="Skor kemiripan (0.0 - 1.0) untuk setiap dokumen yang diambil."
    )
    prompt_used: str = Field(
        description="Instruksi lengkap yang dikirim ke AI untuk keperluan audit."
    )
    token_usage_estimation: TokenUsage = Field(
        description="Detail statistik penggunaan token."
    )


# SCHEMA: Penanganan Error

class ErrorResponse(BaseModel):
    error_code: str = Field(
        description="Kode unik error (misal: 'FILE_NOT_SUPPORTED')."
    )
    error_message: str = Field(
        description="Pesan penjelasan masalah yang ramah bagi pengguna."
    )
    detail: str = Field(
        default="", 
        description="Informasi teknis tambahan untuk kebutuhan debugging."
    )