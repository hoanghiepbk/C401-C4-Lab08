# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phạm Việt Cường  
**Vai trò trong nhóm:** Eval Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500-800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong vai trò Eval Owner, mình chịu trách nhiệm xây bộ đánh giá và đảm bảo kết quả có thể audit được. Mình phụ trách 4 phần chính: (1) chuẩn hóa test questions thành file có `id`, `question`, `expected_answer`, `expected_sources`; (2) định nghĩa expected evidence để đo context recall thay vì chỉ nhìn câu trả lời cuối; (3) chạy scorecard cho 2 cấu hình trong `eval.py` (`baseline_dense` và `variant_hybrid_rrf`); (4) xuất bảng A/B để nhóm dùng trực tiếp trong báo cáo.  
Kết quả được lưu nhất quán trong `results/scorecard_baseline.md`, `results/scorecard_variant.md`, và `results/ab_comparison.csv`. Nhờ đó team có thể nhìn theo từng câu thay vì tranh luận cảm tính. Ví dụ ở baseline, điểm trung bình là Faithfulness 5.00, Relevance 4.60, Context Recall 2.78, Completeness 4.30; còn variant giảm ở Relevance (3.90) và Completeness (3.80), cho thấy đổi sang hybrid chưa chắc đã tốt nếu truy hồi chưa ổn định.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Điều mình hiểu rõ hơn là evaluation phải tách retrieval quality và generation quality. Trước đây mình hay nhìn câu trả lời nghe “đúng” là cho qua, nhưng khi đo bằng expected evidence thì thấy nhiều câu trả lời đúng bề mặt nhưng lấy sai nguồn. Trong kết quả của nhóm, cả baseline và variant đều có cụm lỗi lặp lại: `q02`, `q04`, `q05`, `q08`, `q10` có context recall = 1 vì thiếu expected source (ví dụ thiếu `policy/refund-v4.pdf` hoặc `support/helpdesk-faq.md`).  
Điểm thứ hai là A/B chỉ có ý nghĩa khi đổi đúng một biến. Trong `eval.py`, team giữ nguyên `top_k_search=10`, `top_k_select=3`, `use_rerank=False`, chỉ đổi `retrieval_mode` từ dense sang hybrid. Cách làm này giúp mình kết luận chính xác hơn: vấn đề chính của variant không nằm ở judge hay prompt, mà ở việc hybrid đang kéo nhầm context ở vài câu khó.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất là có câu “đúng ý” nhưng vẫn bị chấm thấp ở completeness/relevance do bằng chứng retrieval không khớp. Ví dụ `q04` (Refund): baseline trả lời tương đối đúng policy sản phẩm số (F=5, R=5, C=4), nhưng variant lại tụt mạnh (F=3, R=2, C=2) vì model đi theo hướng “không đủ dữ liệu”. Đây không phải lỗi chấm điểm ngẫu nhiên mà phản ánh context vào prompt đã yếu.  
Mình cũng gặp bài toán ở nhóm câu “Insufficient Context” như `q09`. Baseline có Relevance=2, Completeness=2; variant còn thấp hơn (1 và 1). Điều này cho thấy pipeline biết abstain, nhưng chưa hướng dẫn người dùng tốt (thiếu next-step hoặc giả thuyết có kiểm soát). Từ góc nhìn Eval Owner, đây là vùng rủi ro thực tế: sản phẩm có thể “an toàn” nhưng chưa “hữu ích”.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q04` — “Sản phẩm kỹ thuật số có được hoàn tiền không?”

**Phân tích:**

Đây là câu mình dùng để kiểm tra độ bền của pipeline vì liên quan trực tiếp đến chính sách và ngoại lệ. Trong `results/ab_comparison.csv`, baseline trả lời: sản phẩm kỹ thuật số không được hoàn tiền, kèm ngoại lệ khi có lỗi nhà sản xuất; điểm là F=5, R=5, Rc=1, C=4. Variant thì trả lời “Không đủ dữ liệu”, dẫn đến F=3, R=2, Rc=1, C=2.  
Điểm đáng chú ý là **context recall đều thấp (1)** ở cả hai cấu hình do thiếu expected source `policy/refund-v4.pdf`, nhưng baseline vẫn giữ chất lượng generation cao hơn variant. Điều này gợi ý baseline dense dù không match đúng expected source vẫn lấy được chunk “gần đúng” để trả lời, còn hybrid ở câu này bị nhiễu và chọn phương án an toàn quá mức.  
Theo mình, lỗi chính nằm ở retrieval + ranking hơn là prompt generation. Nếu retrieval không đưa được chunk có câu “digital goods là ngoại lệ không hoàn tiền” vào top chunk, mô hình rất dễ chuyển sang abstain. Bài học rút ra cho team: với domain policy, cần tăng độ ưu tiên cho nguồn chuẩn (document-level boost hoặc metadata filter theo `policy/*`) trước khi kết luận hybrid tốt hơn dense.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Mình sẽ làm hai cải tiến cụ thể từ kết quả hiện tại. Thứ nhất, thêm “source-priority retrieval” cho nhóm tài liệu chính sách vì các lỗi recall tập trung ở Refund (`q02`, `q04`, `q10`). Thứ hai, thêm checklist hậu xử lý cho câu abstain: khi model trả “không đủ dữ liệu”, hệ thống phải kiểm tra lại có chunk policy liên quan trong top-k-search hay chưa trước khi chốt câu trả lời. Hai thay đổi này nhắm trực tiếp vào các điểm rơi mà scorecard đã chỉ ra.

---
