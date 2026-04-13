"""
score_grading_run.py — Faithfulness scoring for logs/grading_run.json
===============================================================
Chấm metric "faithfulness" (1-5) cho từng entry trong logs/grading_run.json,
dựa trên retrieved chunks (re-fetch bằng pipeline retrieve-only) và answer đã log.

Output JSON per item:
  {"score": <int>, "reason": "<string>"}

Usage:
  python score_grading_run.py --input logs/grading_run.json --output logs/grading_run_faithfulness.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from rag_answer import rag_answer


def _openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (required to score faithfulness).")
    return OpenAI(api_key=key)


def _build_chunks_blob(chunks: List[Dict[str, Any]], char_limit: int = 12000) -> str:
    parts: List[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        text = (c.get("text") or "").strip()
        meta = c.get("metadata") or {}
        source = str(meta.get("source", "unknown"))
        section = str(meta.get("section", ""))
        header = f"[{i}] {source}" + (f" | {section}" if section else "")
        block = f"{header}\n{text}".strip()
        if not block:
            continue
        if used + len(block) + 2 > char_limit:
            remaining = max(0, char_limit - used - 2)
            if remaining > 200:
                parts.append(block[:remaining] + "\n...(truncated)")
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts) if parts else "(no retrieved chunks)"


def judge_faithfulness(chunks_blob: str, answer: str, model: str) -> Dict[str, Any]:
    prompt = f"""Given these retrieved chunks: {chunks_blob}
             And this answer: {answer}
             Rate the faithfulness on a scale of 1-5.
             5 = completely grounded in the provided context (including when the answer correctly says the context is insufficient).
             1 = answer contains information not in the context OR says "insufficient context" when the provided chunks actually contain the requested information.
             Output JSON: {{'score': <int>, 'reason': '<string>'}}"""

    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    score = data.get("score")
    try:
        score_int = int(score)
    except Exception:
        score_int = None
    if score_int is not None:
        score_int = max(1, min(5, score_int))

    reason = str(data.get("reason", ""))[:800]
    return {"score": score_int, "reason": reason}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="logs/grading_run.json")
    p.add_argument("--grading", default="data/grading_questions.json", help="Optional, for id->question fallback")
    p.add_argument("--output", default="logs/grading_run_faithfulness.json")
    p.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"))
    p.add_argument("--top-k-search", type=int, default=10)
    p.add_argument("--top-k-select", type=int, default=3)
    p.add_argument("--use-rerank", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    if not isinstance(runs, list):
        raise ValueError("grading_run.json must be a list.")

    # Optional: map id -> question from grading_questions.json
    id_to_question: Dict[str, str] = {}
    grading_path = Path(args.grading)
    if grading_path.exists():
        with open(grading_path, "r", encoding="utf-8") as f:
            gqs = json.load(f)
        if isinstance(gqs, list):
            for q in gqs:
                qid = q.get("id")
                qq = q.get("question") or q.get("query")
                if qid and qq:
                    id_to_question[str(qid)] = str(qq)

    out: List[Dict[str, Any]] = []
    for item in runs:
        qid = str(item.get("id", ""))
        question = item.get("question") or id_to_question.get(qid) or ""
        answer = str(item.get("answer", ""))
        retrieval_mode = str(item.get("retrieval_mode", "hybrid"))

        # Re-fetch retrieved chunks (retrieve-only) so scoring is grounded in actual context.
        retrieved = rag_answer(
            query=question,
            retrieval_mode=retrieval_mode,
            top_k_search=args.top_k_search,
            top_k_select=args.top_k_select,
            use_rerank=args.use_rerank,
            skip_generation=True,
            verbose=False,
        )
        chunks_used = retrieved.get("chunks_used") or []
        chunks_blob = _build_chunks_blob(chunks_used)

        judge = judge_faithfulness(chunks_blob=chunks_blob, answer=answer, model=args.judge_model)
        out.append(
            {
                "id": qid,
                "question": question,
                "retrieval_mode": retrieval_mode,
                "answer": answer,
                "judge": judge,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path} ({len(out)} items)")


if __name__ == "__main__":
    main()

