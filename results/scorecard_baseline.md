# Scorecard: baseline_dense
Generated: 2026-04-13 16:28

## Summary

Total questions: 10
Questions scored: 10

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 5.00/5 |
| Relevance | 4.60/5 |
| Context Recall | 2.78/5 |
| Completeness | 4.30/5 |

## Weakest Questions (by total score)

| ID | Category | Total(F+R+Rc+C) | Query |
|----|----------|----------------|-------|
| q09 | Insufficient Context | 9 | ERR-403-AUTH là lỗi gì và cách xử lý? |
| q10 | Refund | 13 | Nếu cần hoàn tiền khẩn cấp cho khách hàng VIP, quy trình có khác không? |
| q04 | Refund | 15 | Sản phẩm kỹ thuật số có được hoàn tiền không? |
| q08 | HR Policy | 15 | Nhân viên được làm remote tối đa mấy ngày mỗi tuần? |
| q02 | Refund | 16 | Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày? |

## Retrieval Issues (low context recall)

| ID | Recall Score | Recall Notes |
|----|-------------|-------------|
| q02 | 1 | Retrieved: 0/1 expected sources. Missing: ['policy/refund-v4.pdf'] |
| q04 | 1 | Retrieved: 0/1 expected sources. Missing: ['policy/refund-v4.pdf'] |
| q05 | 1 | Retrieved: 0/1 expected sources. Missing: ['support/helpdesk-faq.md'] |
| q08 | 1 | Retrieved: 0/1 expected sources. Missing: ['hr/leave-policy-2026.pdf'] |
| q10 | 1 | Retrieved: 0/1 expected sources. Missing: ['policy/refund-v4.pdf'] |

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| q01 | SLA | 5 | 5 | 5 | 5 | Câu trả lời hoàn toàn dựa trên thông tin trong con |
| q02 | Refund | 5 | 5 | 1 | 5 | Câu trả lời hoàn toàn bám sát vào thông tin trong  |
| q03 | Access Control | 5 | 5 | 5 | 5 | Câu trả lời hoàn toàn dựa trên thông tin trong con |
| q04 | Refund | 5 | 5 | 1 | 4 | Câu trả lời hoàn toàn dựa trên thông tin trong con |
| q05 | IT Helpdesk | 5 | 5 | 1 | 5 | Câu trả lời hoàn toàn bám sát thông tin trong cont |
| q06 | SLA | 5 | 5 | 5 | 5 | Câu trả lời hoàn toàn dựa trên context đã trích. |
| q07 | Access Control | 5 | 5 | 5 | 5 | Câu trả lời hoàn toàn bám sát vào context đã trích |
| q08 | HR Policy | 5 | 5 | 1 | 4 | Câu trả lời hoàn toàn bám sát thông tin trong cont |
| q09 | Insufficient Context | 5 | 2 | None | 2 | Câu trả lời hoàn toàn dựa trên thông tin có trong  |
| q10 | Refund | 5 | 4 | 1 | 3 | Câu trả lời hoàn toàn dựa trên thông tin trong con |
