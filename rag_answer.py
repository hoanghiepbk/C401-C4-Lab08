"""
rag_answer.py — Sprint 2 + 3: Retrieval & Grounded Answer
=========================================================
- Sprint 2: dense retrieval (Chroma + cùng embedding OpenAI), grounded generation.
- Sprint 3 (variant lab): hybrid = dense + BM25 + Reciprocal Rank Fusion (một biến so với baseline).

Module nhập embedding/collection từ index.py để đảm bảo cùng model embedding.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Cùng nguồn sự thật với index: model embedding + đường dẫn DB
from index import CHROMA_DB_DIR, COLLECTION_NAME, get_embedding

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10
TOP_K_SELECT = 3

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# RRF / hybrid
RRF_K = 60
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4

# BM25 index cache (build 1 lần sau khi có collection)
_BM25_CACHE: Optional[Tuple[Any, List[str], List[str], List[Dict[str, Any]]]] = None


def invalidate_bm25_cache() -> None:
    """Gọi sau khi build_index() trong cùng tiến trình để BM25 khớp corpus mới."""
    global _BM25_CACHE
    _BM25_CACHE = None

# Cross-encoder (chỉ load khi use_rerank=True)
_RERANKER = None


def _openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thiếu OPENAI_API_KEY trong .env.")
    return OpenAI(api_key=key)


# =============================================================================
# CHROMA HELPERS
# =============================================================================


def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_collection(COLLECTION_NAME)


def _distance_to_score(distance: Optional[float]) -> float:
    """
    Chroma `cosine` trả về khoảng cách cosine; đổi sang điểm similarity gần (0,1).

    Thực tế embedding OpenAI + Chroma: d thường nằm trong [0, 2] nhưng với vector chuẩn hóa
    thường thấy quanh [0, 1]. Ta dùng score = max(0, 1 - d).
    """
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


# =============================================================================
# RETRIEVAL — DENSE
# =============================================================================


def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Vector search: embed query (cùng hàm với index) → query Chroma cosine.
    Mỗi phần tử: id, text, metadata, score.
    """
    collection = _get_collection()
    q_emb = get_embedding(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = _distance_to_score(dist)
        out.append(
            {
                "id": cid,
                "text": doc,
                "metadata": meta or {},
                "score": score,
            }
        )
    return out


# =============================================================================
# RETRIEVAL — SPARSE (BM25) + HYBRID (RRF)
# =============================================================================


def _load_bm25_corpus() -> Tuple[Any, List[str], List[str], List[Dict[str, Any]]]:
    """Đọc toàn bộ chunk từ Chroma để dựng BM25 (keyword)."""
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE

    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    data = collection.get(include=["documents", "metadatas", "ids"])
    ids = data["ids"] or []
    documents = data["documents"] or []
    metas = data["metadatas"] or []

    # Token đơn giản: chữ/số + ký tự Latin/VN — đủ cho lab (mã lỗi, P1, SLA, ...)
    tokenized: List[List[str]] = []
    for d in documents:
        tokens = re.findall(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[\wÀ-ỹ]+", (d or "").lower())
        tokenized.append(tokens)

    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE = (bm25, ids, documents, metas)
    return _BM25_CACHE


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """BM25 trên toàn corpus đã index — mạnh mã lỗi / mã ưu tiên / từ khóa hiếm."""
    bm25, ids, documents, metas = _load_bm25_corpus()
    tokens = re.findall(
        r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[\wÀ-ỹ]+",
        (query or "").lower(),
    )
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    out: List[Dict[str, Any]] = []
    for i in ranked:
        s = float(scores[i])
        out.append(
            {
                "id": ids[i],
                "text": documents[i],
                "metadata": metas[i] or {},
                "score": s,
            }
        )
    return out


def _reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int,
    k_rrf: int = RRF_K,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    RRF: hợp nhất hai thứ hạng mà không cần chuẩn hóa score khác thang (dense vs BM25).
    """
    rrf_scores: Dict[str, float] = {}
    id_to_row: Dict[str, Dict[str, Any]] = {}

    for rank, row in enumerate(dense_results):
        cid = row["id"]
        id_to_row[cid] = row
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight * (1.0 / (k_rrf + rank + 1))

    for rank, row in enumerate(sparse_results):
        cid = row["id"]
        id_to_row.setdefault(cid, row)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight * (1.0 / (k_rrf + rank + 1))

    ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    merged: List[Dict[str, Any]] = []
    for cid in ranked_ids:
        base = dict(id_to_row[cid])
        base["score"] = float(rrf_scores[cid])
        base["metadata"] = dict(base.get("metadata") or {})
        merged.append(base)
    return merged


def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
) -> List[Dict[str, Any]]:
    """Kết hợp dense + BM25 bằng RRF (Sprint 3 — variant khuyến nghị cho corpus mixed)."""
    dense = retrieve_dense(query, top_k=top_k)
    sparse = retrieve_sparse(query, top_k=top_k)
    if not sparse:
        return dense
    return _reciprocal_rank_fusion(
        dense,
        sparse,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )


# =============================================================================
# RERANK (tuỳ chọn — dùng khi use_rerank=True)
# =============================================================================


def _get_cross_encoder():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """Cross-encoder: chấm lại (query, passage) sau khi đã search rộng."""
    if not candidates:
        return []
    model = _get_cross_encoder()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for chunk, sc in ranked[:top_k]:
        row = dict(chunk)
        row["score"] = float(sc)
        out.append(row)
    return out


# =============================================================================
# QUERY TRANSFORM (Sprint 3 — stub mở rộng; mặc định không đổi query)
# =============================================================================


def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """Giữ interface lab; có thể mở rộng LLM expansion sau này."""
    return [query]


# =============================================================================
# GENERATION
# =============================================================================


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Đánh số [1],[2],... để model trích dẫn nhất quán."""
    context_parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score:
            header += f" | score={float(score):.3f}"
        context_parts.append(f"{header}\n{text}")
    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Evidence-only + abstain tiếng Việt khi không có chứng cứ (đặc biệt mã lỗi / chi tiết không có trong context).
    """
    return f"""Bạn là trợ lý nội bộ. Chỉ được trả lời dựa trên ngữ cảnh đã trích dẫn bên dưới.

Quy tắc bắt buộc:
1) Nếu ngữ cảnh KHÔNG chứa thông tin đủ để trả lời (ví dụ: mã lỗi không xuất hiện, không có điều khoản liên quan), trả lời CHÍNH XÁC một câu:
   "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này."
2) Không suy đoán, không dùng kiến thức bên ngoài ngữ cảnh.
3) Khi trả lời được, trích dẫn nguồn bằng số trong ngoặc như [1], [2] tương ứng đoạn ngữ cảnh.
4) Trả lời ngắn gọn, rõ ràng, cùng ngôn ngữ với câu hỏi.

Câu hỏi: {query}

Ngữ cảnh:
{context_block}

Trả lời:"""


def call_llm(prompt: str) -> str:
    """Chat completions OpenAI, temperature thấp để ổn định khi chấm điểm."""
    client = _openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=700,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline: retrieve → (tuỳ chọn rerank) → build prompt → LLM.

    Abstain khi retrieval quá yếu: không gọi LLM, trả về câu chuẩn lab.
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            print(
                f"  [{i + 1}] score={c.get('score', 0):.3f} | "
                f"{c.get('metadata', {}).get('source', '?')}"
            )

    if not candidates:
        msg = "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này."
        return {
            "query": query,
            "answer": msg,
            "sources": [],
            "chunks_used": [],
            "config": config,
        }

    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select/rerank: {len(candidates)} chunks")

    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)
    answer = call_llm(prompt)

    sources = sorted(
        {c["metadata"].get("source", "unknown") for c in candidates if c.get("metadata")}
    )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SO SÁNH STRATEGIES (Sprint 3)
# =============================================================================


def compare_retrieval_strategies(query: str) -> None:
    """In nhanh baseline dense vs hybrid cho cùng một query."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    for strategy in ("dense", "hybrid"):
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline (OpenAI)")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",
    ]

    print("\n--- Baseline dense ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n--- Variant hybrid (Sprint 3) ---")
    try:
        q = "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"
        compare_retrieval_strategies(q)
    except Exception as e:
        print(f"Lỗi: {e}")
