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
from functools import lru_cache
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


@lru_cache(maxsize=1)
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
    # Chroma does not accept "ids" inside include; ids are returned by default.
    data = collection.get(include=["documents", "metadatas"])
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
    """
    Sinh biến thể truy vấn để tăng recall (đặc biệt alias/viết tắt/mã lỗi).
    Trả về danh sách đã loại trùng và giữ nguyên thứ tự ưu tiên.
    """
    q = (query or "").strip()
    if not q:
        return []

    if strategy == "none":
        return [q]

    out: List[str] = [q]
    lower_q = q.lower()

    # Alias/domain terms cho bộ tài liệu lab (VN + EN)
    alias_map = {
        "Approval Matrix": ["approval matrix", "ACCESS CONTROL SOP", "ma trận phê duyệt"],
        "sla": ["service level agreement", "cam kết mức dịch vụ"],
        "hoàn tiền": ["refund", "refund policy", "chính sách hoàn tiền"],
        "helpdesk": ["it helpdesk", "hỗ trợ kỹ thuật", "it support"],
        "cấp quyền": ["access control", "phân quyền", "approval matrix"],
        "level 3": ["l3", "quyền level 3", "mức quyền 3"],
        "ticket": ["incident", "request", "phiếu hỗ trợ"],
        "p1": ["priority 1", "mức ưu tiên 1", "độ ưu tiên p1"],
        "p2": ["priority 2", "mức ưu tiên 2", "độ ưu tiên p2"],
        "p3": ["priority 3", "mức ưu tiên 3", "độ ưu tiên p3"],
        "p4": ["priority 4", "mức ưu tiên 4", "độ ưu tiên p4"],
    }

    for term, expansions in alias_map.items():
        if term in lower_q:
            out.extend(expansions)

    # Bắt mã lỗi/chuỗi kỹ thuật để thêm truy vấn chính xác dạng token
    # Ví dụ: ERR-403-AUTH -> "ERR 403 AUTH", "error 403 auth"
    code_like = re.findall(r"[A-Za-z]{2,}-\d{2,}(?:-[A-Za-z0-9]+)*", q)
    for code in code_like:
        spaced = code.replace("-", " ")
        out.append(spaced)
        out.append(f"error {spaced.lower()}")

    # Câu hỏi thời lượng/SLA hay gặp -> thêm signal retrieval theo intent
    if any(k in lower_q for k in ("bao lâu", "thời gian", "trong bao nhiêu", "deadline")):
        out.append("thời gian xử lý")
        out.append("response time")

    # Khử trùng lặp theo lowercase, giữ thứ tự
    deduped: List[str] = []
    seen = set()
    for item in out:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped


def _merge_query_variants_results(
    all_results: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Gộp kết quả retrieval từ nhiều biến thể query theo max score mỗi chunk id.
    """
    best_by_id: Dict[str, Dict[str, Any]] = {}
    for results in all_results:
        for row in results:
            cid = row["id"]
            old = best_by_id.get(cid)
            if old is None or float(row.get("score", 0.0)) > float(old.get("score", 0.0)):
                best_by_id[cid] = row
    return sorted(best_by_id.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)


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
    query_transform: str = "none",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    skip_generation: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline: retrieve → (tuỳ chọn rerank) → build prompt → LLM.

    Abstain khi retrieval quá yếu: không gọi LLM, trả về câu chuẩn lab.
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "query_transform": query_transform,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Retrieve ---
    query_variants = transform_query(query, strategy=query_transform)
    if not query_variants:
        query_variants = [query]

    retrieved_per_variant: List[List[Dict[str, Any]]] = []
    for qv in query_variants:
        if retrieval_mode == "dense":
            retrieved = retrieve_dense(qv, top_k=top_k_search)
        elif retrieval_mode == "sparse":
            retrieved = retrieve_sparse(qv, top_k=top_k_search)
        elif retrieval_mode == "hybrid":
            retrieved = retrieve_hybrid(qv, top_k=top_k_search)
        else:
            raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")
        retrieved_per_variant.append(retrieved)

    candidates = _merge_query_variants_results(retrieved_per_variant)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Query variants ({len(query_variants)}): {query_variants}")
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

    # Giữ danh sách candidates gốc để fallback khi LLM "abstain nhầm".
    # Lý do: đôi khi top_k_select quá nhỏ, phần evidence quan trọng nằm ở vị trí 4-8,
    # khiến model nghĩ "không đủ dữ liệu" dù corpus có thông tin.
    all_candidates = candidates

    if use_rerank:
        selected = rerank(query, all_candidates, top_k=top_k_select)
    else:
        selected = all_candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select/rerank: {len(selected)} chunks")

    # Cho phép "retrieve-only" để phục vụ grading/judge mà không tốn LLM call.
    if skip_generation:
        sources = sorted(
            {c["metadata"].get("source", "unknown") for c in selected if c.get("metadata")}
        )
        return {
            "query": query,
            "answer": "",
            "sources": sources,
            "chunks_used": selected,
            "config": config,
        }

    context_block = build_context_block(selected)
    prompt = build_grounded_prompt(query, context_block)
    answer = call_llm(prompt)

    # Fallback 1 lần nếu model trả abstain nhưng ta còn nhiều candidate chưa đưa vào context.
    abstain_msg = "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này."
    if (
        answer.strip() == abstain_msg
        and len(all_candidates) > len(selected)
        and len(selected) < 8
    ):
        retry_k = min(max(len(selected) * 2, 6), 8, len(all_candidates))
        if use_rerank:
            retry_selected = rerank(query, all_candidates, top_k=retry_k)
        else:
            retry_selected = all_candidates[:retry_k]

        if verbose:
            print(f"[RAG] Retry with wider context: {retry_k} chunks")

        retry_context = build_context_block(retry_selected)
        retry_prompt = build_grounded_prompt(query, retry_context)
        retry_answer = call_llm(retry_prompt)

        # Chấp nhận retry nếu nó không còn abstain (hoặc đơn giản là khác câu cũ).
        if retry_answer.strip() != abstain_msg:
            selected = retry_selected
            answer = retry_answer

    sources = sorted(
        {c["metadata"].get("source", "unknown") for c in selected if c.get("metadata")}
    )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": selected,
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

def compare_retrieval_strategies_expansion(query: str) -> None:
    """In nhanh query_transform none vs expansion cho cùng một query."""
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    for query_transform in ("none", "expansion"):
        print(f"\n--- query_transform: {query_transform} ---")
        try:
            result = rag_answer(query, retrieval_mode="dense", query_transform=query_transform, verbose=False)
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

    print("\n--- Variant hybrid/ query_transform expansion (Sprint 3) ---")
    try:
        q = "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"
        compare_retrieval_strategies(q)
        compare_retrieval_strategies_expansion(q)
    except Exception as e:
        print(f"Lỗi: {e}")
