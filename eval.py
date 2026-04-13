"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Chạy bộ câu hỏi, chấm 4 metrics, so sánh baseline vs variant.

Faithfulness / Answer relevance / Completeness: LLM-as-Judge (gpt-4o-mini, JSON).
Context recall: heuristic so khớp expected_sources với metadata chunk (programmatic).
"""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

# Variant Sprint 3 — chỉ đổi MỘT biến so với baseline: dense → hybrid (RRF + BM25)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "variant_hybrid_rrf",
}

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")


def _openai_eval_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thiếu OPENAI_API_KEY — cần để chạy LLM-as-Judge trong eval.py.")
    return OpenAI(api_key=key)


def score_subjective_with_llm(
    query: str,
    answer: str,
    chunks_used: List[Dict[str, Any]],
    expected_answer: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Một lần gọi API chấm faithfulness, answer relevance, completeness (thang 1–5).

    Dùng retrieved chunks làm ground-truth cho faithfulness; expected_answer làm
    tham chiếu mềm cho completeness (không phải trích word-for-word).
    """
    context_blob = "\n\n---\n\n".join(
        (c.get("text") or "")[:8000] for c in chunks_used
    )
    if not context_blob.strip():
        context_blob = "(Không có chunk retrieval — abstain hoặc lỗi pipeline.)"

    schema_hint = """Trả về JSON duy nhất với các khóa:
{
  "faithfulness": {"score": <1-5>, "notes": "<ngắn>"},
  "relevance": {"score": <1-5>, "notes": "<ngắn>"},
  "completeness": {"score": <1-5>, "notes": "<ngắn>"}
}"""

    prompt = f"""Bạn là giám khảo độc lập cho pipeline RAG nội bộ.

{schema_hint}

Tiêu chí:
- faithfulness: Mức độ câu trả lời chỉ dựa trên CONTEXT đã trích (không bịa). 5 = hoàn toàn bám context.
- relevance: Câu trả lời có đúng trọng tâm QUERY không. 5 = trả lời trực tiếp, đúng ý.
- completeness: So với EXPECTED (ý chính), câu trả lời có bao phủ đủ không. 5 = đủ ý chính hoặc abstain đúng khi expected nói không có thông tin.

QUERY:
{query}

CONTEXT (retrieved):
{context_blob}

ANSWER (model):
{answer}

EXPECTED (tham chiếu giảng viên, có thể dài hơn answer):
{expected_answer}
"""

    client = _openai_eval_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "{}").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    def _pick(key: str) -> Dict[str, Any]:
        block = data.get(key) or {}
        sc = block.get("score")
        try:
            score_int = int(sc) if sc is not None else None
        except (TypeError, ValueError):
            score_int = None
        if score_int is not None:
            score_int = max(1, min(5, score_int))
        return {"score": score_int, "notes": str(block.get("notes", ""))[:500]}

    return _pick("faithfulness"), _pick("relevance"), _pick("completeness")


def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Giữ interface lab — nên gọi score_subjective_with_llm() từ runner để tránh lặp API."""
    return {
        "score": None,
        "notes": "Dùng score_subjective_with_llm() trong run_scorecard.",
    }


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    return {
        "score": None,
        "notes": "Dùng score_subjective_with_llm() trong run_scorecard.",
    }


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5

    TODO Sprint 4:
    1. Lấy danh sách source từ chunks_used
    2. Kiểm tra xem expected_sources có trong retrieved sources không
    3. Tính recall score
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    def _norm_source(s: str) -> str:
        s = (s or "").strip().replace("\\", "/")
        while "//" in s:
            s = s.replace("//", "/")
        return s.lower()

    retrieved_sources = {
        _norm_source(c.get("metadata", {}).get("source", ""))
        for c in chunks_used
    }
    retrieved_sources.discard("")  # tránh nhiễu nếu metadata thiếu source

    # TODO: Kiểm tra matching theo partial path (vì source paths có thể khác format)
    found = 0
    missing = []
    for expected in expected_sources:
        exp_norm = _norm_source(expected)
        exp_file = exp_norm.split("/")[-1]
        # Ưu tiên: match theo file name exact; fallback: substring (alias đường dẫn)
        matched = any((r.split("/")[-1] == exp_file) for r in retrieved_sources) or any(
            exp_file in r for r in retrieved_sources
        )
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    # Convert recall (0..1) sang thang 1..5, clamp để không ra 0
    score_1_5 = int(round(recall * 5))
    score_1_5 = max(1, min(5, score_1_5))

    return {
        "score": score_1_5,
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    return {
        "score": None,
        "notes": "Dùng score_subjective_with_llm() trong run_scorecard.",
    }


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

    TODO Sprint 4:
    1. Load test_questions từ data/test_questions.json
    2. Với mỗi câu hỏi:
       a. Gọi rag_answer() với config tương ứng
       b. Chấm 4 metrics
       c. Lưu kết quả
    3. Tính average scores
    4. In bảng kết quả
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        recall = score_context_recall(chunks_used, expected_sources)

        if answer.startswith("PIPELINE_NOT_IMPLEMENTED") or answer.startswith("ERROR:"):
            faith = {"score": None, "notes": "Pipeline lỗi / chưa implement."}
            relevance = {"score": None, "notes": faith["notes"]}
            complete = {"score": None, "notes": faith["notes"]}
        else:
            try:
                faith, relevance, complete = score_subjective_with_llm(
                    query=query,
                    answer=answer,
                    chunks_used=chunks_used,
                    expected_answer=expected_answer,
                )
            except Exception as exc:
                err = f"Judge lỗi: {exc}"
                faith = {"score": None, "notes": err}
                relevance = {"score": None, "notes": err}
                complete = {"score": None, "notes": err}

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(
            f"\nAverage {metric}: {avg:.2f}"
            if avg is not None
            else f"\nAverage {metric}: N/A (chưa chấm)"
        )

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.

    TODO Sprint 4:
    Điền vào bảng sau để trình bày trong báo cáo:

    | Metric          | Baseline | Variant | Delta |
    |-----------------|----------|---------|-------|
    | Faithfulness    |   ?/5    |   ?/5   |  +/?  |
    | Answer Relevance|   ?/5    |   ?/5   |  +/?  |
    | Context Recall  |   ?/5    |   ?/5   |  +/?  |
    | Completeness    |   ?/5    |   ?/5   |  +/?  |

    Câu hỏi cần trả lời:
    - Variant tốt hơn baseline ở câu nào? Vì sao?
    - Biến nào (chunking / hybrid / rerank) đóng góp nhiều nhất?
    - Có câu nào variant lại kém hơn baseline không? Tại sao?
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh đơn giản
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.

    TODO Sprint 4: Cập nhật template này theo kết quả thực tế của nhóm.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")

        # In preview
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    baseline_results: List[Dict[str, Any]] = []
    variant_results: List[Dict[str, Any]] = []

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    print("Lưu ý: Cần index + OPENAI_API_KEY để chạy scorecard.")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )

        # Save scorecard
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")

    except NotImplementedError:
        print("Pipeline chưa implement. Hoàn thành Sprint 2 trước.")
        baseline_results = []
    except Exception as exc:
        print(f"Baseline scorecard lỗi: {exc}")
        baseline_results = []

    # --- Chạy Variant (hybrid RRF — Sprint 3) ---
    print("\n--- Chạy Variant ---")
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
        (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
        print(f"\nScorecard variant lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")
    except Exception as exc:
        print(f"Variant scorecard lỗi: {exc}")

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv",
        )

    print("\n\nGợi ý Sprint 4 (sau khi chạy xong pipeline):")
    print("  - Điền docs/architecture.md, docs/tuning-log.md theo kết quả thực tế.")
    print("  - Hoàn thiện báo cáo cá nhân trong reports/individual/.")
