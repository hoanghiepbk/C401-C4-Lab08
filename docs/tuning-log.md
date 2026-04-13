# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 13/04/2026  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = "gpt-4o-mini"
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 5.00 /5 |
| Answer Relevance | 4.40 /5 |
| Context Recall | 2.78 /5 |
| Completeness | 4.10 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
> q09 (ERR-403-AUTH) - relevance = 2, recall = None. Vì Dense thường bỏ lỡ kết quả tra cứu mã code/lỗi chính xác.
> q10 (hoàn tiền khách hàng VIP) - relevance = 2, recall = 1 vì dense khó tìm kết nối keyword "khách hàng VIP" với ngoại lệ.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 13/04/2026  
**Biến thay đổi:** retrieval_mode (dense -> hybrid)  
**Lý do chọn biến này:**
> Dựa vào baseline, q09 (mã lỗi ERR-403-AUTH) và q10 (từ khóa "VIP") không tìm được hoặc ra điểm thấp, đây là các dạng câu hỏi cần bắt keyword chính xác (exact match). Hybrid Search sẽ kết hợp sức mạnh Semantic của Dense và Exact Math của BM25.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # hoặc biến khác
# Các tham số còn lại giữ nguyên như baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 5.00/5 | 5.00/5 | +0.00 |
| Answer Relevance | 4.40/5 | 4.20/5 | -0.20 |
| Context Recall | 2.78/5 | 2.78/5 | +0.00 |
| Completeness | 4.10/5 | 3.90/5 | -0.20 |

**Nhận xét:**
> Variant 1 (hybrid) cải thiện được câu q10 (từ relevance 2 -> 4) do BM25 bắt keyword "VIP" tốt hơn.
> Tuy nhiên lại làm giảm điểm q04 (từ relevance 5 -> 2) và q09 (relevance 2 -> 1). Điều này xảy ra do ranking RRF chưa được tinh chỉnh chuẩn xác, dẫn đến một số document semantic tốt bị tụt hạng.

**Kết luận:**
> Variant 1 (hybrid RRF) chưa đem lại hiệu quả tốt hơn so với baseline tổng thể (Relevance và Completeness đều giảm).
> Cần phương pháp rerank thực thụ hoặc trọng số lai tốt hơn thay vì RRF mặc định của Chroma/Langchain.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

> TODO (Sprint 4): Điền sau khi hoàn thành evaluation.

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Context Recall luôn ở mức rất thấp (trung bình ~2.78), tức là search ra đúng tài liệu nhưng không có khả năng chứa mảnh thông tin cụ thể (có thể do chunk nhỏ quá hoặc cắt đoạn).

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Việc cải thiện ranking system (chọn document top) sẽ có tác động lớn nhất do Retriever đang lấy lên sai/không đầy đủ context.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Bật `use_rerank = True` kết hợp `top_k_search = 15` để tăng số lượng candidate, sau đó bắt LLM rà soát và rank lại trước khi generation.
