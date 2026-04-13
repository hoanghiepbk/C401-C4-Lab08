# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Đặng Tiến Dũng - 2A202600024  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

- Em làm vai trò Retrieval Owner nên tập trung từ Sprint 1 đến Sprint 3. 
- Ở Sprint 1, em chỉnh lại bước preprocess trong `index.py` để metadata header được parse ổn định hơn (`Department`, `Effective Date`, `Access`) thay vì bị rơi về `unknown`. Em cũng đảm bảo metadata được sanitize trước khi upsert Chroma để tránh lỗi kiểu dữ liệu. Ở Sprint 2, em kiểm tra lại pipeline retrieve → prompt → answer trong `rag_answer.py`, đặc biệt là logic abstain khi không đủ chứng cứ. Ở Sprint 3, em triển khai query transform theo hướng practical: mở rộng alias/keyword và chuẩn hóa query có mã lỗi để tăng recall. Ngoài ra, em thêm hướng fallback khi không tìm thấy câu trả lời: trả về hotline/contact phù hợp theo department. 
- Phần của em kết nối trực tiếp với Eval Owner vì mọi thay đổi retrieval đều được phản ánh vào scorecard/A-B comparison.


## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Sau lab này, em hiểu rõ hơn hai ý chính:
- chunking theo cấu trúc tài liệu và retrieval theo intent. Trước đây em nghĩ chunking chỉ cần cắt theo độ dài là đủ, nhưng khi làm thực tế, em thấy ranh giới section và metadata ảnh hưởng mạnh đến khả năng truy xuất đúng đoạn. Nếu parse header sai hoặc chunk không giữ ngữ cảnh section, hệ thống vẫn “trả lời” nhưng dễ thiếu chứng cứ.
- Ý thứ hai là query transform chính xác là công cụ điều hướng retrieval. Ví dụ với query chứa mã lỗi hoặc từ mơ hồ, transform giúp tạo các biến thể có tín hiệu tốt hơn cho dense/sparse search. Em cũng hiểu grounded prompt đúng nghĩa là giới hạn model trong phạm vi context: chất lượng câu trả lời phụ thuộc retrieval trước, không thể “cầu may” ở bước generation.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Khó khăn lớn nhất của em là lỗi “đúng code nhưng sai dữ liệu”: hệ thống chạy bình thường, không crash, nhưng metadata coverage báo toàn `unknown`. Lúc đầu em giả thuyết do lúc upsert metadata bị mất hoặc sanitize sai kiểu, nhưng debug kỹ thì nguyên nhân nằm ở preprocess: logic kết thúc header quá sớm khi gặp dòng tiêu đề in hoa, làm parser bỏ qua các dòng `Department` và `Effective Date`. Đây là lỗi khá “im lặng” vì không tạo exception. Điểm ngạc nhiên là chỉ cần metadata sai một bước nhỏ thì đánh giá downstream giảm rõ (filter/analysis theo department không dùng được, fallback khó chính xác). 
Em rút ra rằng với RAG, không nên chỉ test bằng “có ra câu trả lời không”, mà cần test chất lượng index (coverage metadata, section integrity) như một checkpoint bắt buộc.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:**

Đây là câu hỏi khó vì người dùng dùng tên cũ (alias) của tài liệu thay vì tên hiện tại trong corpus. Ở baseline dense, hệ thống đôi khi không lấy đúng chunk mở đầu của `access-control-sop.md`, dẫn đến câu trả lời mơ hồ hoặc thiếu phần “đổi tên tài liệu”. Điểm bị ảnh hưởng chủ yếu là context recall và citation precision: model có thể trả lời gần đúng, nhưng chưa chắc bám đúng chứng cứ chứa câu “trước đây có tên Approval Matrix...”.

Khi chạy variant, em cải thiện ở retrieval bằng query transform: thêm biến thể như “access control sop”, “ma trận phê duyệt”, và kết hợp hybrid retrieval để BM25 bắt tốt keyword alias. Sau đó, rerank/select giúp ưu tiên chunk có tín hiệu trực tiếp về mapping tên cũ–tên mới. Kết quả thực tế trong scorecard tốt hơn baseline vì câu trả lời ổn định hơn: hệ thống nêu rõ “Approval Matrix for System Access” nay là “Access Control SOP”, đồng thời trích dẫn đúng nguồn `it/access-control-sop.md`. Với em, đây là minh chứng rõ nhất rằng tuning retrieval đúng chỗ có tác động lớn hơn chỉnh prompt đơn thuần.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Nếu có thêm thời gian, em sẽ làm hai cải tiến cụ thể:
-  Thứ nhất, thêm metadata-aware reranking: ưu tiên chunk có `department` khớp intent và section chứa từ khóa “Liên hệ/Hỗ trợ”, vì eval cho thấy fallback contact vẫn có lúc trả nguồn chưa tối ưu.
- Thứ hai, em muốn xây bộ test regression nhỏ cho preprocess/chunking (đặc biệt parser header) để tránh lỗi coverage lặp lại sau mỗi lần refactor. Điều này giúp pipeline ổn định hơn trước khi tinh chỉnh model/prompt.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
