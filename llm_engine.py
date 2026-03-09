import os
from typing import Optional, Dict, Any

from llama_cpp import Llama

# Konfigurasi LLM


DEFAULT_MODEL_PATH: str = os.getenv("GGUF_MODEL_PATH", "./models/model.gguf")
DEFAULT_CONTEXT_WINDOW_SIZE: int = 4096      # Ukuran konteks maksimum dalam token
DEFAULT_MAX_NEW_TOKENS: int = 512            # Batas token untuk output LLM
DEFAULT_GPU_LAYERS: int = 0                  # Layer yang di-offload ke GPU (0 = CPU only)
DEFAULT_TEMPERATURE: float = 0.1             # Rendah = output lebih deterministik/faktual
DEFAULT_REPEAT_PENALTY: float = 1.15         # Penalti untuk mengurangi pengulangan kata/frasa dalam output
DEFAULT_TOP_P_SAMPLING: float = 0.9          # Nucleus sampling threshold
AVERAGE_CHARS_PER_TOKEN: int = 4             # Estimasi kasar: 1 token ≈ 4 karakter


# Template Prompt untuk RAG

RAG_SYSTEM_INSTRUCTION: str = """You are a precise and helpful assistant. 
Your task is to answer the user's question STRICTLY based on the provided context.
If the answer is not found in the context, clearly state that the information is not available in the provided document.
Do NOT use any external knowledge. Be concise and accurate. 
IMPORTANT: You must ALWAYS answer in the SAME LANGUAGE as the user's question (e.g., if the user asks in Indonesian, answer in Indonesian)."""

RAG_PROMPT_TEMPLATE: str = """### System:
{system_instruction}

### Context from Document:
{retrieved_context}

### User Question:
{user_question}

### Answer:"""

# Manajemen model LLM lokal menggunakan llama-cpp-python

def load_llm_model(
    model_path: str = DEFAULT_MODEL_PATH,
    context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
    gpu_layers: int = DEFAULT_GPU_LAYERS,
) -> Llama:
    """
    Memuat model GGUF lokal ke memori menggunakan llama-cpp-python.

    Mengapa verbose=False:
    Output verbose llama.cpp sangat panjang dan mencemari log aplikasi.
    Error yang relevan tetap di-raise sebagai Python exception.

    Mengapa n_ctx dikonfigurasi saat loading (bukan saat inferensi):
    llama.cpp mengalokasikan KV-cache berdasarkan n_ctx saat model dimuat.
    Mengubahnya memerlukan reload model, jadi lebih baik ditetapkan di awal.

    Args:
        model_path: Path absolut atau relatif ke file .gguf
        context_window_size: Ukuran konteks maksimum dalam token
        gpu_layers: Jumlah layer transformer yang di-offload ke GPU VRAM

    Returns:
        Instance Llama yang siap digunakan untuk inferensi

    Raises:
        FileNotFoundError: Jika file model .gguf tidak ditemukan
        RuntimeError: Jika model gagal dimuat karena alasan lain
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model GGUF tidak ditemukan di: '{model_path}'. "
            f"Pastikan Anda sudah mengunduh model dan mengatur GGUF_MODEL_PATH."
        )

    try:
        llm_instance = Llama(
            model_path=model_path,
            n_ctx=context_window_size,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        return llm_instance
    except Exception as loading_error:
        raise RuntimeError(
            f"Gagal memuat model dari '{model_path}': {loading_error}"
        ) from loading_error


def format_context_from_chunks(retrieved_chunks: list[str]) -> str:
    """
    Memformat daftar chunks menjadi satu blok konteks yang terstruktur.

    Mengapa memberi nomor pada setiap passage:
    Penomoran membantu LLM merujuk sumber spesifik jika diperlukan,
    dan memudahkan debugging ketika melihat prompt_used dalam respons.

    Args:
        retrieved_chunks: Daftar chunks teks yang relevan dari retriever

    Returns:
        String konteks terformat dengan penomoran setiap passage
    """
    formatted_passages = [
        f"[Passage {passage_number}]: {chunk_text}"
        for passage_number, chunk_text in enumerate(retrieved_chunks, start=1)
    ]
    return "\n\n".join(formatted_passages)


def build_rag_prompt(
    user_question: str,
    retrieved_chunks: list[str],
) -> str:
    """
    Merakit prompt lengkap untuk RAG menggunakan template yang telah didefinisikan.

    Mengapa struktur System / Context / Question / Answer:
    Format ini mengikuti pola instruksi yang dikenali oleh sebagian besar
    model instruksi (Mistral, Llama, dll.) dan terbukti menghasilkan respons
    yang lebih patuh terhadap batasan konteks.

    Args:
        user_question: Pertanyaan asli dari pengguna
        retrieved_chunks: Chunks dokumen yang paling relevan

    Returns:
        String prompt lengkap yang siap dikirim ke LLM
    """
    formatted_context = format_context_from_chunks(retrieved_chunks)

    complete_prompt = RAG_PROMPT_TEMPLATE.format(
        system_instruction=RAG_SYSTEM_INSTRUCTION,
        retrieved_context=formatted_context,
        user_question=user_question,
    )
    return complete_prompt

# Estimasi Token

def estimate_token_count(text: str) -> int:
    """
    Memperkirakan jumlah token dari sebuah teks.

    Mengapa estimasi kasar (bukan tokenizer asli):
    Memanggil tokenizer asli memerlukan overhead tambahan dan akses ke model.
    Aturan praktis '4 karakter per token' adalah standar industri yang cukup
    akurat untuk teks bahasa Inggris (+/- 15% error margin).

    Args:
        text: Teks yang akan diestimasi jumlah tokennya

    Returns:
        Perkiraan jumlah token sebagai integer
    """
    return max(1, len(text) // AVERAGE_CHARS_PER_TOKEN)


def calculate_token_usage(
    prompt_text: str,
    completion_text: str
) -> Dict[str, int]:
    """
    Menghitung estimasi penggunaan token untuk prompt dan completion.

    Mengapa mengembalikan dict (bukan tuple):
    Dictionary dengan key yang jelas lebih ekspresif dan mudah dipetakan
    ke schema Pydantic TokenUsageEstimation.

    Args:
        prompt_text: Teks prompt yang dikirim ke LLM
        completion_text: Teks jawaban yang dihasilkan LLM

    Returns:
        Dictionary dengan key: prompt_tokens, completion_tokens, total_tokens
    """
    prompt_token_count = estimate_token_count(prompt_text)
    completion_token_count = estimate_token_count(completion_text)
    total_token_count = prompt_token_count + completion_token_count

    return {
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": total_token_count,
    }

# Inferensi LLM

def run_llm_inference(
    llm_instance: Llama,
    prompt_text: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P_SAMPLING,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
) -> str:
    """
    Menjalankan inferensi pada model LLM dan mengekstrak teks jawaban.

    Mengapa temperature rendah (0.1) sebagai default untuk RAG:
    Dalam konteks RAG, kita menginginkan jawaban yang faktual dan deterministik
    berdasarkan dokumen, bukan jawaban kreatif. Temperature rendah meminimalkan
    'halusinasi' dan menjaga model tetap dekat dengan konteks yang diberikan.

    Args:
        llm_instance: Instance Llama yang sudah dimuat
        prompt_text: Prompt lengkap yang sudah dirakit
        max_new_tokens: Batas maksimum token yang dihasilkan
        temperature: Tingkat 'kreativitas' output (0.0 = deterministik)
        top_p: Nucleus sampling - hanya pertimbangkan token dengan prob kumulatif >= p

    Returns:
        Teks jawaban yang dihasilkan oleh LLM

    Raises:
        RuntimeError: Jika inferensi gagal
    """
    try:
        llm_response = llm_instance(
            prompt_text,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            echo=False,  
            stop=["### User", "### Question", "\n\n\n", "User:", "<|eot_id|>", "###", "Question:"],  # Stop tokens
        )

        generated_answer = llm_response["choices"][0]["text"].strip()
        return generated_answer

    except Exception as inference_error:
        raise RuntimeError(
            f"Inferensi LLM gagal: {inference_error}"
        ) from inference_error


# Fungsi Utama (Entry Point Model)

def generate_answer_with_context(
    user_question: str,
    retrieved_chunks: list[str],
    llm_instance: Llama,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """
    Fungsi: merakit prompt dan menghasilkan jawaban dari LLM.

    Mengapa mengembalikan dict lengkap (bukan hanya jawaban):
    Transparansi adalah inti dari 'Explainable RAG'. Mengembalikan prompt_used
    dan token_usage memungkinkan pengguna memverifikasi dan mengaudit
    bagaimana jawaban dihasilkan.

    Args:
        user_question: Pertanyaan dari pengguna
        retrieved_chunks: Chunks dokumen paling relevan dari retriever
        llm_instance: Instance Llama yang sudah dimuat
        max_new_tokens: Batas token untuk output

    Returns:
        Dictionary berisi:
        - answer: Teks jawaban dari LLM
        - prompt_used: Prompt lengkap yang digunakan
        - token_usage: Estimasi penggunaan token
    """
    complete_prompt = build_rag_prompt(user_question, retrieved_chunks)
    generated_answer = run_llm_inference(llm_instance, complete_prompt, max_new_tokens)
    token_usage_data = calculate_token_usage(complete_prompt, generated_answer)

    return {
        "answer": generated_answer,
        "prompt_used": complete_prompt,
        "token_usage": token_usage_data,
    }