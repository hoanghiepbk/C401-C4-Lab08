# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phạm Hữu Hoàng Hiệp  
**Vai trò trong nhóm:** Tech Lead  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong vai trò Tech Lead, mình chịu trách nhiệm “nối” toàn bộ pipeline chạy end-to-end và đảm bảo repo đáp ứng đúng deliverables. Cụ thể, mình chốt cấu trúc repo theo README (tách rõ `index.py` / `rag_answer.py` / `eval.py`), thống nhất các tham số baseline/variant (dense vs hybrid RRF) để đảm bảo A/B chỉ đổi một biến. Mình trực tiếp rà và sửa các lỗi ảnh hưởng chấm điểm: tạo script xuất log `logs/grading_run.json` theo format trong `SCORING.md`, sửa lỗi tính delta trong `compare_ab()` và chuẩn hóa thang điểm context recall. Ngoài ra, mình sửa logic generation để giảm “abstain nhầm”: khi LLM trả đúng câu “Không đủ dữ liệu…” nhưng pipeline vẫn còn nhiều chunk candidates chưa đưa vào prompt, hệ thống sẽ retry 1 lần với context rộng hơn; nhờ đó các câu policy có ngoại lệ rõ (ví dụ Flash Sale + đã kích hoạt) không bị bỏ sót evidence. Cuối cùng mình chạy sanity check (compile) để đảm bảo code không crash trước khi demo.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này mình hiểu rõ hơn mối quan hệ “chunking ↔ retrieval ↔ grounded generation”. Trước đây mình nghĩ chỉ cần tăng top-k là đủ, nhưng khi làm end-to-end mới thấy chunking quyết định trực tiếp việc query có “chạm” đúng đoạn evidence hay không. Nếu chunk quá dài, embedding bị loãng và context block dài làm LLM dễ bỏ sót ý; nếu chunk quá ngắn, evidence bị vỡ và cần nhiều chunk hơn để đủ ngữ cảnh. Mình cũng hiểu rõ giá trị của hybrid retrieval: dense mạnh về ngữ nghĩa, còn BM25 lại mạnh ở từ khóa hiếm/mã lỗi/alias. RRF giúp hợp nhất theo thứ hạng mà không phải “cưỡng ép” chuẩn hóa hai thang điểm khác nhau (cosine distance vs BM25 score), nên rất phù hợp cho corpus lab có cả policy tự nhiên lẫn keyword kỹ thuật.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Điều khiến mình ngạc nhiên là evaluation có thể “trông đúng” nhưng thực ra sai do mismatch nhỏ ở metadata/source. Ví dụ expected_sources trong test set dùng đường dẫn kiểu `support/sla-p1-2026.pdf`, còn chunk metadata khi index lại là file `.txt` và có thể khác dấu (`sla_p1_2026.txt`). Nếu context recall match theo chuỗi thô, recall sẽ về 0 dù retrieval đã lấy đúng tài liệu — dẫn đến kết luận tuning sai. Debug phần này mất thời gian vì pipeline retrieval/generation vẫn chạy bình thường, chỉ metric bị lệch. Khó khăn thứ hai là tính ổn định của LLM-as-judge: cùng prompt nhưng output có thể thay đổi theo lần chạy (dù temperature=0), nên mình ưu tiên đảm bảo metric programmatic (context recall) đúng logic và log output đầy đủ để audit, thay vì chỉ nhìn “điểm tổng” rồi kết luận vội.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** q07 — “Approval Matrix để cấp quyền hệ thống là tài liệu nào?”

**Phân tích:**
Baseline dùng dense retrieval. Với dạng query alias/tên cũ (“Approval Matrix”), dense đôi khi kéo về chunk có nhắc đến từ “Approval” nhưng không đúng “tên mới” của tài liệu, khiến generation trả lời thiếu ý quan trọng (expected muốn mapping: “Approval Matrix for System Access” đã đổi tên thành “Access Control SOP”). Lỗi chính nằm ở retrieval: semantic search có thể ưu tiên đoạn mô tả quy trình cấp quyền hơn là đoạn nêu tên/alias tài liệu. Variant hybrid (dense + BM25 + RRF) kỳ vọng cải thiện vì BM25 đánh mạnh vào keyword “Approval Matrix”, giúp kéo đúng chunk có tiêu đề/alias vào top-k. Tuy nhiên, nếu chunking không giữ lại phần alias nằm gần header hoặc section title bị cắt rời, hybrid vẫn có thể chỉ trả tên cũ mà không nêu “tên mới”. Bài học ở đây là: cải thiện retrieval (hybrid) cần đi kèm chunking/metadata tốt (giữ tiêu đề/alias trong chunk) và prompt phải ép trả lời theo evidence (“tài liệu hiện tại tên gì”) để đạt completeness cao.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Mình sẽ làm 2 cải tiến cụ thể. (1) Thêm caching cho LLM-as-judge theo khóa (query+answer+context+expected) để chạy lại scorecard nhanh và ổn định hơn, giảm chi phí và tránh thay đổi điểm do judge variability. (2) Bổ sung một ngưỡng “retrieval đủ mạnh” trước khi gọi LLM (ví dụ dựa trên top score hoặc diversity nguồn) để giảm hallucination ở các câu không có thông tin trong docs; vì rubric grading phạt nặng nếu bịa, nên abstain đúng còn quan trMìnhọng hơn trả lời dài nhưng thiếu chứng cứ.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*

