"""
index.py — Sprint 1: Build RAG Index
====================================
Pipeline: đọc tài liệu → preprocess → chunk theo section/paragraph →
embed (OpenAI) → lưu ChromaDB (cosine).

Yêu cầu lab: mỗi chunk có metadata tối thiểu source, section, effective_date;
embedding query sau này phải dùng cùng model với index (text-embedding-3-small).

Gợi ý đọc file theo luồng:
- `preprocess_document()` -> (text đã clean, metadata chuẩn)
- `chunk_document()` / `_split_by_size()` -> danh sách chunk nhỏ + overlap
- `get_embedding()` -> biến text -> vector (OpenAI hoặc local)
- `build_index()` -> upsert toàn bộ chunk vào ChromaDB để `rag_answer.py` query
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import logging
from transformers import logging as transformers_logging

load_dotenv()

# Tắt cảnh báo từ thư viện transformers
transformers_logging.set_verbosity_error()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rag_lab"

# Chunking:
# - LLM làm việc với token, nhưng ở lab ta thao tác trên ký tự.
# - Heuristic phổ biến: ~4 ký tự ~ 1 token (xấp xỉ; VN/EN khác nhau).
# - Với lab nhỏ, chọn chunk vừa phải để retrieval “trúng đoạn” và prompt không bị dài.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Embedding model:
# - Bắt buộc: query embedding (trong `rag_answer.py`) phải dùng cùng model/provider với indexing,
#   nếu không similarity search sẽ sai (vector space khác nhau).
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# =============================================================================
# OPENAI CLIENT (singleton qua cache — tránh tạo client lặp lại)
# =============================================================================


@lru_cache(maxsize=1)
def _openai_client():
    # Tạo client một lần để tránh overhead và dễ kiểm soát cấu hình.
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY trong .env. Sao chép .env.example → .env và điền key."
        )
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _local_embedding_model():
    """Khởi tạo model SentenceTransformer một lần."""
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    return SentenceTransformer(model_name)


def get_embedding(text: str) -> List[float]:
    """
    Vector hóa một đoạn text bằng OpenAI Embeddings.

    Lưu ý: Model và nhà cung cấp phải trùng với bước query trong rag_answer.py,
    nếu không similarity search sẽ không có nghĩa.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("get_embedding: text rỗng.")

    # Provider switch (để lab có thể chạy theo 2 hướng):
    # - openai: dùng OpenAI embeddings (cần OPENAI_API_KEY)
    # - local: dùng sentence-transformers chạy local (không cần key, nhưng query cũng phải dùng cùng model)
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")

    if provider == "openai":
        client = _openai_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return list(response.data[0].embedding)
    else:
        # Option B — Sentence Transformers (chạy local)
        model = _local_embedding_model()
        return model.encode(text).tolist()


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma chỉ chấp nhận metadata dạng scalar: str / int / float / bool.
    Ta “sanitize” để:
    - tránh lỗi upsert (ví dụ metadata lỡ là list/dict),
    - đảm bảo filter/inspect sau này ổn định.
    """
    out: Dict[str, Any] = {}
    for key, val in meta.items():
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            out[str(key)] = val
        else:
            out[str(key)] = str(val)
    return out


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================


def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    # Regex patterns for metadata
    patterns = {
        "source": re.compile(r"^Source:\s*(.*)", re.IGNORECASE),
        "department": re.compile(r"^Department:\s*(.*)", re.IGNORECASE),
        "effective_date": re.compile(r"^Effective Date:\s*(.*)", re.IGNORECASE),
        "access": re.compile(r"^Access:\s*(.*)", re.IGNORECASE),
    }

    for line in lines:
        stripped = line.strip()
        if not header_done:
            # Check for metadata matches
            matched = False
            for key, pattern in patterns.items():
                match = pattern.match(stripped)
                if match:
                    metadata[key] = match.group(1).strip()
                    matched = True
                    break
            
            if matched:
                continue
            
            # Start of content markers
            if stripped.startswith("===") or (stripped and stripped.isupper()):
                header_done = True
                content_lines.append(line)
            elif stripped == "":
                continue
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    # Normalize whitespace: max 2 consecutive newlines, remove trailing whitespace
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# =============================================================================


def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Split theo heading pattern "=== ... ==="
    # Dùng regex để bắt cả heading để biết section đang ở đâu
    parts = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in parts:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict[str, Any],
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks bằng cách ghép paragraph.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    
    current_chunk_text = ""
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
            
        # Nếu cộng thêm paragraph này mà vượt quá chunk_chars, thì lưu chunk hiện tại
        if current_chunk_text and len(current_chunk_text) + len(para) > chunk_chars:
            chunks.append({
                "text": current_chunk_text.strip(),
                "metadata": {**base_metadata, "section": section},
            })
            
            # Tính toán overlap: lấy đoạn cuối của chunk hiện tại
            # Hoặc đơn giản hơn: lấy paragraph trước đó làm overlap nếu nó không quá dài
            overlap_text = current_chunk_text[-overlap_chars:] if len(current_chunk_text) > overlap_chars else current_chunk_text
            current_chunk_text = overlap_text + "\n\n" + para
        else:
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para
                
    # Thêm chunk cuối cùng
    if current_chunk_text:
        chunks.append({
            "text": current_chunk_text.strip(),
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """Đọc toàn bộ .txt → chunk → embed OpenAI → upsert Chroma (persistent)."""
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Index mới: xóa collection cũ để tránh trùng id/metadata lệch phiên bản
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            emb = get_embedding(chunk["text"])
            meta = _sanitize_metadata(chunk["metadata"])
            
            all_ids.append(chunk_id)
            all_embeddings.append(emb)
            all_documents.append(chunk["text"])
            all_metadatas.append(meta)

    if all_ids:
        # Upsert theo batch (Chroma xử lý tốt batch lớn, nhưng ta có thể chia nếu cần)
        # Ở đây có ~50-100 chunks, upsert 1 lần là ổn.
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )

    print(f"\nHoàn thành! Tổng số chunks đã index: {len(all_ids)}")
    print(f"Collection: {COLLECTION_NAME} | DB: {db_dir}")

    # Làm mới cache BM25 trong rag_answer (nếu đã import) để hybrid search khớp corpus mới
    try:
        from rag_answer import invalidate_bm25_cache

        invalidate_bm25_cache()
    except Exception:
        pass


# =============================================================================
# STEP 4: INSPECT / DEBUG
# =============================================================================


def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """In n chunk mẫu để kiểm tra metadata và ranh giới nội dung."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(
            zip(results["documents"] or [], results["metadatas"] or [])
        ):
            print(f"[Chunk {i + 1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {doc[:200]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """Thống kê nhanh phủ metadata trên toàn bộ index."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        metas = results["metadatas"] or []

        print(f"\nTổng chunks: {len(metas)}")
        departments: Dict[str, int] = {}
        missing_date = 0
        for meta in metas:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            ed = meta.get("effective_date", "unknown")
            if ed in ("unknown", "", None):
                missing_date += 1

        print("Phân bố theo department:")
        for dept, count in sorted(departments.items()):
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date (unknown): {missing_date}")
    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index (OpenAI + Chroma)")
    print("=" * 60)

    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    print("\n--- Test preprocess + chunking (không gọi API) ---")
    if doc_files:
        filepath = doc_files[0]
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i + 1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:180]}...")

    print("\n--- Build Full Index (EMBEDDING_PROVIDER=local hoặc openai) ---")
    build_index()
    list_chunks(n=3)
    inspect_metadata_coverage()

    print("\nSprint 1: hoàn tất pipeline index + inspect.")
