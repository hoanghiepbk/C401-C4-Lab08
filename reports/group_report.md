# Báo Cáo Nhóm — Lab Day 08: Full RAG Pipeline

**Tên nhóm:** RAG Masters
**Thành viên:**
| Tên | Vai trò | Email |
|-----|---------|-------|
| Phạm Hữu Hoàng Hiệp (2A202600415) | Tech Lead | hiep.phh@example.com |
| Đặng Tiến Dũng (2A202600024) | Retrieval Owner | dung.dt@example.com |
| Phạm Việt Cường | Eval Owner | cuong.pv@example.com |
| Phạm Trần Thanh Lâm | Documentation Owner | lam.ptt@example.com |

**Ngày nộp:** 13/04/2026  
**Repo:** C401-C4-Lab08  
**Độ dài khuyến nghị:** 600–900 từ

---

## 1. Pipeline nhóm đã xây dựng (150–200 từ)

**Chunking decision:**
Nhóm sử dụng cấu hình `chunk_size` = 400 (khoảng 1600 ký tự) và `overlap` = 80 (khoảng 320 ký tự). Phương pháp cắt chunk kết hợp dựa trên cấu trúc Heading (`=== ... ===`) và Paragraph để đảm bảo các điều khoản chính sách không bị cắt đôi. Mỗi chunk được đính kèm các metadata quan trọng như `source`, `section`, `effective_date`, `department`, và `access` để dễ dàng cho việc tracking và xây dựng context.

**Embedding model:**
Nhóm sử dụng local model `AITeamVN/Vietnamese_Embedding` thông qua thư viện SentenceTransformers, lưu trữ vector vào ChromaDB với khoảng cách đo là Cosine Distance. Model được chọn vì hỗ trợ tốt tiếng Việt, dung lượng gọn nhẹ nhưng vẫn đáp ứng được nhu cầu semantic search.

**Retrieval variant (Sprint 3):**
Nhóm đã chọn thực hiện **Hybrid Search (kết hợp Dense Retrieval và Sparse BM25)** sử dụng Reciprocal Rank Fusion (RRF). Việc sử dụng Hybrid nhằm khắc phục điểm yếu của Dense model trong việc bắt chính xác các keyword (ví dụ mã lỗi ERR-403-AUTH, các thuật ngữ cụ thể như VIP) đồng thời vẫn tận dụng được khả năng hiểu ngữ nghĩa của Dense model.

---

## 2. Quyết định kỹ thuật quan trọng nhất (200–250 từ)

**Quyết định:** Chuyển đổi từ Dense Retrieval thuần sang Hybrid Search (Dense + BM25) ở Sprint 3.

**Bối cảnh vấn đề:**
Sau khi hoàn thành Sprint 2 và tiến hành evaluation, nhóm nhận thấy metric `Context Recall` của pipeline rất thấp (đạt 2.78/5). Các câu hỏi liên quan đến từ khóa cụ thể hoặc mã số chính xác (ví dụ "mã lỗi ERR-403-AUTH" hay từ khóa "VIP") thường bị Dense Retrieval bỏ lỡ do mô hình tập trung quá nhiều vào ý nghĩa tổng thể mà không matching được các token đặc thù. Điều này dẫn đến Answer Relevance cũng bị giảm sút.

**Các phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| Tinh chỉnh Chunking | Dễ dàng implement, giúp context đầy đủ hơn | Vẫn không giải quyết triệt để lỗi keyword matching, dễ bị "lost in the middle" nếu chunk lớn |
| Áp dụng Reranker | Giúp rank lại kết quả tốt hơn, chất lượng thường cao | Tốn kém tài nguyên tính toán, tăng độ trễ (latency) của pipeline đáng kể |
| Áp dụng Hybrid Search (BM25 + Dense) | Giải quyết được lỗi keyword matching, tận dụng được sức mạnh của cả semantic và sparse search | Cần tinh chỉnh thuật toán hợp nhất (RRF) để kết quả trả về không bị sai lệch |

**Phương án đã chọn và lý do:**
Nhóm quyết định triển khai **Hybrid Search**. Hệ thống nội bộ như IT Helpdesk và Chính sách (Policies) chứa rất nhiều các thuật ngữ chuyên ngành (VPN, SSO, JIRA, P1, Flash Sale...). BM25 cực kỳ mạnh trong việc tìm kiếm các keyword này, kết hợp với khả năng hiểu ngữ nghĩa của Dense tạo thành một giải pháp cân bằng giữa hiệu năng và độ chính xác mà không tốn chi phí quá cao cho mô hình Rerank.

**Bằng chứng từ scorecard/tuning-log:**
Trong `docs/tuning-log.md`, nhóm ghi nhận câu hỏi có chứa từ khóa "VIP" ban đầu có `Relevance` = 2 (Dense search). Khi dùng Hybrid Search, BM25 đã bắt trúng keyword và đẩy điểm `Relevance` lên 4. Mặc dù tổng thể Hybrid (RRF) chưa hoàn hảo do thiết lập RRF mặc định (Answer Relevance tổng bị tụt một chút), nhưng Hybrid đã cho thấy khả năng vá được các lỗ hổng keyword matching.

---

## 3. Kết quả grading questions (100–150 từ)

**Ước tính điểm raw:** 88 / 98

**Câu tốt nhất:** ID: `gq06` — Lý do: Pipeline đã xử lý xuất sắc một câu hỏi yêu cầu kỹ năng Cross-doc multi-hop (kết hợp `access_control_sop.txt` và `sla_p1_2026.txt`). Pipeline lấy được cả thông tin sự cố P1 (2am) và quy trình cấp quyền tạm thời tối đa 24h, sau đó tổng hợp được câu trả lời hoàn hảo mà không bị thiếu ý.

**Câu fail:** ID: `gq05` — Root cause: Lỗi nằm ở bước Retrieval. Câu hỏi yêu cầu thông tin về Admin Access (Level 4), nhưng retrieval lại ưu tiên các đoạn chunk nói về quyền truy cập thông thường (Level 1/2) do có chung keyword "cấp quyền". Hệ quả là LLM sinh ra thông tin quy trình duyệt 1 ngày làm việc (sai so với 5 ngày thực tế của Admin). 

**Câu gq07 (abstain):** Pipeline đã xử lý rất tốt khả năng anti-hallucination. Mặc dù truy xuất được thông tin về SLA P1, nhưng vì tài liệu không đề cập đến việc "phạt bao nhiêu", LLM đã tuân thủ prompt grounding và trả lời chính xác: "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này."

---

## 4. A/B Comparison — Baseline vs Variant (150–200 từ)

**Biến đã thay đổi (chỉ 1 biến):** `retrieval_mode` (từ `dense` sang `hybrid` với RRF)

| Metric | Baseline | Variant 1 | Delta |
|--------|---------|---------|-------|
| Faithfulness | 5.00/5 | 5.00/5 | +0.00 |
| Answer Relevance | 4.40/5 | 4.20/5 | -0.20 |
| Context Recall | 2.78/5 | 2.78/5 | +0.00 |
| Completeness | 4.10/5 | 3.90/5 | -0.20 |

**Kết luận:**
Kết quả Variant (Hybrid Search) có chỉ số Faithfulness và Context Recall giữ nguyên, nhưng lại bị giảm nhẹ ở `Answer Relevance` (-0.20) và `Completeness` (-0.20).
Dù Hybrid khắc phục được điểm mù keyword (như câu hỏi có keyword "VIP" được cải thiện), nhưng do trọng số Reciprocal Rank Fusion (RRF) mặc định chưa được tối ưu, một số document có giá trị ngữ nghĩa cao lại bị đẩy xuống dưới các document match keyword nhưng ít quan trọng. Kết quả là chất lượng một số câu trả lời khác bị kéo xuống. Điều này chứng tỏ chỉ ghép Dense + BM25 là chưa đủ, mà cần phải tinh chỉnh công thức ranking.

---

## 5. Phân công và đánh giá nhóm (100–150 từ)

**Phân công thực tế:**

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Phạm Hữu Hoàng Hiệp | Thiết kế hệ thống, hỗ trợ các thành viên, code base `index.py`, quản lý repo | 1, 2 |
| Đặng Tiến Dũng | Viết `rag_answer.py`, thiết lập prompt grounding, triển khai Variant Hybrid Search (Sprint 3) | 2, 3 |
| Phạm Việt Cường | Hoàn thiện code `eval.py`, thiết lập metrics đánh giá (LLM-as-a-judge, Recall score), debug | 4 |
| Phạm Trần Thanh Lâm | Xây dựng dataset, chuẩn bị tài liệu nội bộ, viết documentation (`architecture.md`, `tuning-log.md`) | 1, 4 |

**Điều nhóm làm tốt:**
Các thành viên đều hoàn thành tốt vai trò của mình. Code chạy ổn định không lỗi, pipeline bắt được tốt lỗi Hallucination (thể hiện qua câu gq07). Bộ metrics trong `eval.py` được xây dựng chi tiết giúp đánh giá khách quan.

**Điều nhóm làm chưa tốt:**
Thiếu thời gian để tuning trọng số cho RRF khi implement Hybrid. Chưa tích hợp Reranker vào để xem tác động của nó với những context bị "lost" khi search. Điểm Context Recall vẫn còn thấp.

---

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì? (50–100 từ)

Nếu có thêm thời gian, nhóm sẽ triển khai thêm một Variant dùng **Cross-Encoder (Reranker)** áp dụng sau bước Retrieval (lấy `top_k_search` = 20 và `top_k_select` = 3).
Dựa trên `scorecard`, Context Recall bị thấp do candidate lấy về không chứa đáp án (ví dụ câu gq05 bị nhầm level quyền truy cập). Reranker sẽ đánh giá trực tiếp độ phù hợp giữa query và từng chunk thay vì tính khoảng cách vector, giúp đẩy đúng chunk có chứa thông tin "Level 4 Admin" lên top đầu.

---

*File này lưu tại: `reports/group_report.md`*  
*Commit sau 18:00 được phép theo SCORING.md*