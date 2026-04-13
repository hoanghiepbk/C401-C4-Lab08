# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Phạm Trần Thanh Lâm
**Vai trò trong nhóm:** Documentation Owner  
**Ngày nộp:** 13/04/2026  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> Mô tả cụ thể phần bạn đóng góp vào pipeline:
> - Sprint nào bạn chủ yếu làm? Tôi đảm nhận Sprint 4
> - Cụ thể bạn implement hoặc quyết định điều gì? Viết và chạy test cases
> - Công việc của bạn kết nối với phần của người khác như thế nào? Tôi phối hợp với các thanh viên khác để implement code, sau đó viết report architecture.md và tuning-log.md


_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

> Chọn 1-2 concept từ bài học mà bạn thực sự hiểu rõ hơn sau khi làm lab.
> Ví dụ: chunking, hybrid retrieval, grounded prompt, evaluation loop.
> Giải thích bằng ngôn ngữ của bạn — không copy từ slide.
chunking: chia đoạn văn bản nguồn thành những đoạn văn bản nhỏ hơn để thành context đưa vào LLm để gen ra câu trả lời.
Hybrid retrieval: Kết hợp 2 kỹ thuật tìm kiếm: sparse(TF-IDF, Bm25,...) và Dense retrieval giúp giảm tiêu tốn tài nguyên của Dense retrieval và tăng độ chính xác của sparse retrieval
_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

> Điều gì xảy ra không đúng kỳ vọng?
> Lỗi nào mất nhiều thời gian debug nhất?
> Giả thuyết ban đầu của bạn là gì và thực tế ra sao?
Không có
_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

Đánh giá q07: Lỗi truy xuất tên tài liệu (Alias)

Vấn đề (Baseline): Dense Retrieval bị trượt vì ưu tiên "ngữ nghĩa" (mô tả quy trình) thay vì "từ khóa" (tên cũ/alias của tài liệu).

Kỳ vọng ở Hybrid: Kết hợp thêm BM25 để "bắt" chính xác keyword (ví dụ: "Approval Matrix"), giúp kéo đúng tài liệu vào top-K.

Action Items (Khắc phục triệt để):

Dùng Hybrid Retrieval: Kết hợp Dense + BM25.

Tối ưu Chunking/Metadata: Đảm bảo Tiêu đề (tên mới) và Alias (tên cũ) không bị cắt rời, phải nằm trong cùng một chunk hoặc được gán vào metadata.

Siết lại Prompt: Ép LLM phải chỉ ra "tên hiện tại" của tài liệu dựa trên evidence truy xuất được.
_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Mình sẽ làm 2 cải tiến cụ thể. (1) Thêm caching cho LLM-as-judge theo khóa (query+answer+context+expected) để chạy lại scorecard nhanh và ổn định hơn, giảm chi phí và tránh thay đổi điểm do judge variability. (2) Bổ sung một ngưỡng “retrieval đủ mạnh” trước khi gọi LLM (ví dụ dựa trên top score hoặc diversity nguồn) để giảm hallucination ở các câu không có thông tin trong docs; vì rubric grading phạt nặng nếu bịa, nên abstain đúng còn quan trMìnhọng hơn trả lời dài nhưng thiếu chứng cứ.

_________________

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
