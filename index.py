"""
index.py — Sprint 1: Build RAG Index
====================================
Pipeline: đọc tài liệu → preprocess → chunk theo section/paragraph →
embed (OpenAI) → lưu ChromaDB (cosine).

Yêu cầu lab: mỗi chunk có metadata tối thiểu source, section, effective_date;
embedding query sau này phải dùng cùng model với index (text-embedding-3-small).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "rag_lab"

# Chunk: ước lượng token ≈ len(chars)/4 (gợi ý slide 300–500 tokens)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# OpenAI Embeddings — giữ đồng bộ với rag_answer.retrieve_dense()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# =============================================================================
# OPENAI CLIENT (singleton qua cache — tránh tạo client lặp lại)
# =============================================================================


@lru_cache(maxsize=1)
def _openai_client():
    """Khởi tạo client OpenAI một lần cho cả index và (nếu cần) các module khác."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY trong .env. Sao chép .env.example → .env và điền key."
        )
    return OpenAI(api_key=api_key)


def get_embedding(text: str) -> List[float]:
    """
    Vector hóa một đoạn text bằng OpenAI Embeddings.

    Lưu ý: Model và nhà cung cấp phải trùng với bước query trong rag_answer.py,
    nếu không similarity search sẽ không có nghĩa.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("get_embedding: text rỗng.")

    client = _openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return list(response.data[0].embedding)


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma chỉ chấp nhận metadata dạng str / int / float / bool.
    Chuẩn hóa để tránh lỗi upsert và để filter ổn định.
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
    Tách metadata chuẩn (Source, Department, Effective Date, Access) và phần nội dung.

    Cấu trúc file mẫu:
      - Vài dòng tiêu đề (bỏ qua trước khối metadata)
      - Các dòng Key: Value
      - Tuỳ file: dòng ghi chú (ví dụ alias tên tài liệu) — được giữ lại trong nội dung
      - Các section bắt đầu bằng '=== ... ==='
    """
    lines = raw_text.strip().split("\n")

    # Tách phần trước section đầu tiên (để không đánh mất đoạn giữa header và ===)
    first_section_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("===")),
        len(lines),
    )
    head, body = lines[:first_section_idx], lines[first_section_idx:]

    metadata: Dict[str, Any] = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }

    meta_prefixes = ("Source:", "Department:", "Effective Date:", "Access:")
    for line in head:
        if line.startswith("Source:"):
            metadata["source"] = line.replace("Source:", "", 1).strip()
        elif line.startswith("Department:"):
            metadata["department"] = line.replace("Department:", "", 1).strip()
        elif line.startswith("Effective Date:"):
            metadata["effective_date"] = line.replace("Effective Date:", "", 1).strip()
        elif line.startswith("Access:"):
            metadata["access"] = line.replace("Access:", "", 1).strip()

    # Dòng tiêu đề đứng trước dòng Source — bỏ qua; phần còn lại (ghi chú, v.v.) giữ
    first_meta_idx = next(
        (i for i, ln in enumerate(head) if ln.startswith("Source:")),
        0,
    )
    preamble_lines: List[str] = []
    for i, line in enumerate(head):
        if i < first_meta_idx and line.strip():
            continue
        if any(line.startswith(p) for p in meta_prefixes):
            continue
        if line.strip() == "":
            continue
        preamble_lines.append(line)

    cleaned_text = "\n".join(preamble_lines + body)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return {"text": cleaned_text, "metadata": metadata}


# =============================================================================
# STEP 2: CHUNK
# =============================================================================


def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chia theo heading === ... === trước; section dài thì cắt theo đoạn + overlap."""
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks: List[Dict[str, Any]] = []

    # Tách giữ lại delimiter để biết ranh giới section (có thể xuống dòng trong heading)
    sections = re.split(r"(===[\s\S]*?===)", text)
    current_section = "General"
    current_section_text = ""

    for part in sections:
        stripped = part.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            if current_section_text.strip():
                chunks.extend(
                    _split_by_size(
                        current_section_text.strip(),
                        base_metadata=base_metadata,
                        section=current_section,
                    )
                )
            current_section = stripped.strip("=").strip()
            current_section_text = ""
        else:
            current_section_text += part

    if current_section_text.strip():
        chunks.extend(
            _split_by_size(
                current_section_text.strip(),
                base_metadata=base_metadata,
                section=current_section,
            )
        )

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict[str, Any],
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Ghép paragraph (\n\n) tới gần chunk_chars; giữ overlap giữa hai chunk liên tiếp.

    Paragraph cực dài: cắt cửa sổ ký tự, bước nhảy (window - overlap).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    raw_chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if buf:
            raw_chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for para in paragraphs:
        if len(para) > chunk_chars:
            flush_buf()
            start = 0
            while start < len(para):
                end = min(start + chunk_chars, len(para))
                raw_chunks.append(para[start:end])
                if end >= len(para):
                    break
                nxt = end - overlap_chars
                start = nxt if nxt > start else end
        else:
            extra = len(para) + (2 if buf else 0)
            if buf_len + extra <= chunk_chars:
                buf.append(para)
                buf_len += extra
            else:
                flush_buf()
                buf = [para]
                buf_len = len(para)

    flush_buf()

    if not raw_chunks:
        raw_chunks = [text[:chunk_chars]]

    # Overlap: chunk kế tiếp bắt đầu bằng đuôi chunk trước (chuẩn RAG)
    merged_texts: List[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev = merged_texts[-1]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        nxt = raw_chunks[i]
        combined = tail + "\n\n" + nxt
        if len(combined) > chunk_chars * 2:
            merged_texts.append(nxt)
        else:
            merged_texts.append(combined)

    return [
        {
            "text": t,
            "metadata": {**base_metadata, "section": section, "chunk_part": idx},
        }
        for idx, t in enumerate(merged_texts)
    ]


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

    total_chunks = 0
    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            emb = get_embedding(chunk["text"])
            meta = _sanitize_metadata(chunk["metadata"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[chunk["text"]],
                metadatas=[meta],
            )
            total_chunks += 1

    print(f"\nHoàn thành! Tổng số chunks đã index: {total_chunks}")
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

    print("\n--- Build Full Index (cần OPENAI_API_KEY) ---")
    build_index()
    list_chunks(n=3)
    inspect_metadata_coverage()

    print("\nSprint 1: hoàn tất pipeline index + inspect.")
