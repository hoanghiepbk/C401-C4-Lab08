"""
run_grading.py — Generate logs/grading_run.json for SCORING.md
=============================================================
Chạy pipeline với data/grading_questions.json (được public lúc 17:00)
và lưu log theo format yêu cầu.

Usage:
  python run_grading.py --input data/grading_questions.json --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rag_answer import rag_answer


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("grading_questions.json phải là một list JSON objects.")
    return data


def main() -> None:
    # Windows console đôi khi dùng encoding không phải UTF-8 (ví dụ cp1252),
    # khiến print tiếng Việt bị UnicodeEncodeError. Ta ép stdout/stderr về utf-8 nếu có thể.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/grading_questions.json", help="Path tới grading_questions.json")
    p.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"], help="retrieval_mode")
    p.add_argument("--top-k-search", type=int, default=10)
    p.add_argument("--top-k-select", type=int, default=3)
    p.add_argument("--use-rerank", action="store_true")
    p.add_argument("--output", default="logs/grading_run.json", help="Path output log json")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {input_path}")

    questions = _load_questions(input_path)

    out: List[Dict[str, Any]] = []
    for q in questions:
        qid = q.get("id")
        question = q.get("question") or q.get("query")
        if not qid or not question:
            raise ValueError(f"Câu hỏi thiếu id/question: {q}")

        result = rag_answer(
            question,
            retrieval_mode=args.mode,
            top_k_search=args.top_k_search,
            top_k_select=args.top_k_select,
            use_rerank=args.use_rerank,
            verbose=False,
        )

        out.append(
            {
                "id": qid,
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "chunks_retrieved": len(result.get("chunks_used") or []),
                "retrieval_mode": (result.get("config") or {}).get("retrieval_mode", args.mode),
                "timestamp": datetime.now().isoformat(),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved log: {output_path} ({len(out)} questions)")


if __name__ == "__main__":
    main()

