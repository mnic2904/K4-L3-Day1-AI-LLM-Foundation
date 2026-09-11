# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature tăng từ 0.0 lên 1.5, câu trả lời thường trở nên đa dạng và sáng tạo hơn, nhưng cũng có thể ít ổn định và dễ lan man hơn. Ở temperature thấp, model có xu hướng đưa ra câu trả lời nhất quán và tập trung hơn; ở temperature cao, cách diễn đạt và lựa chọn thông tin có thể thay đổi nhiều hơn giữa các lần gọi.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ chọn temperature khoảng 0–0.2 cho chatbot hỗ trợ khách hàng. Khoảng giá trị thấp này giúp phản hồi của chatbot luôn chính xác, nhất quán và bám sát tài liệu hướng dẫn. Đồng thời, nó giúp ngăn chặn sự bịa đặt thông tin (hallucination) về chính sách hay giá cả của doanh nghiệp.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

> ***Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini: GPT-4o có giá output khoảng 0.010 USD/1K token, trong khi GPT-4o-mini là 0.0006 USD/1K token, nên GPT-4o đắt hơn khoảng 16.67 lần cho cùng số token đầu ra. Với 10.000 người dùng, mỗi người 3 lần gọi/ngày và mỗi lần 350 token, tổng output khoảng 10,5 triệu token/ngày. GPT-4o xứng đáng khi cần xử lý yêu cầu phức tạp, cần chất lượng và khả năng suy luận cao; GPT-4o-mini phù hợp với các tác vụ đơn giản, số lượng lớn như FAQ, phân loại câu hỏi hoặc trả lời hỗ trợ cơ bản.***

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Khi dùng system prompt dành cho giáo viên tiểu học, câu trả lời thường ngắn gọn và dễ hiểu hơn, sử dụng từ vựng đơn giản và các ví dụ gần gũi với trẻ em. Ngược lại, system prompt dành cho chuyên gia tài chính khiến câu trả lời có xu hướng dài và chuyên sâu hơn, sử dụng nhiều thuật ngữ kỹ thuật và giải thích blockchain dưới góc nhìn công nghệ hoặc tài chính. Như vậy, system prompt có ảnh hưởng rõ rệt đến vai trò, cách diễn đạt, mức độ chi tiết và loại ví dụ mà model lựa chọn. Tuy nhiên, system prompt định hướng hành vi chứ không đảm bảo tuyệt đối model luôn tuân thủ mọi yêu cầu.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với đoạn văn tiếng Việt khoảng 100 từ, số token đo bằng tiktoken và số token ước lượng theo công thức số từ / 0.75 chênh nhau khoảng 6,82%. Nguyên nhân là token không tương đương trực tiếp với một từ; tokenizer có thể chia một từ thành nhiều token tùy theo ngôn ngữ và cách biểu diễn. Tiếng Việt có dấu và đặc điểm mã hóa ký tự khiến cùng một lượng thông tin có thể sử dụng nhiều token hơn tiếng Anh trong một số trường hợp. Vì vậy, khi tính chi phí API cho người dùng Việt Nam, nên dựa trên số token thực tế thay vì chỉ ước lượng bằng số từ.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất khi model tạo ra câu trả lời dài hoặc cần nhiều thời gian xử lý, vì người dùng có thể bắt đầu đọc ngay khi những token đầu tiên được tạo ra thay vì phải chờ toàn bộ câu trả lời. Streaming không làm model xử lý nhanh hơn mà chủ yếu làm giảm thời gian chờ cảm nhận của người dùng. Ngược lại, non-streaming phù hợp với các tác vụ có kết quả ngắn hoặc khi ứng dụng cần nhận toàn bộ response trước khi xử lý tiếp.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giảm áp lực lên API khi hệ thống đang quá tải bằng cách tăng dần thời gian chờ giữa các lần retry, chẳng hạn 0,1 giây, 0,2 giây rồi 0,4 giây. Nếu hàng nghìn client cùng retry với delay cố định, chúng có thể gửi request lại gần như cùng một thời điểm, tạo ra một đợt tải mới ngay khi server vừa phục hồi. Exponential backoff giúp phân tán các lần retry theo thời gian và cho server thêm thời gian xử lý tình trạng quá tải.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *System prompt:

Bạn là một trợ lý học tập AI thân thiện và kiên nhẫn. Hãy trả lời bằng tiếng Việt, giải thích rõ ràng theo từng bước và ưu tiên ví dụ đơn giản khi người học chưa hiểu khái niệm. Với câu hỏi kỹ thuật, hãy đưa ra câu trả lời chính xác và có thể áp dụng được. Không bịa thông tin; nếu không chắc chắn, hãy nói rõ giới hạn của mình. Ưu tiên câu trả lời ngắn gọn nhưng đủ để người học hiểu và có thể tự áp dụng.

Giải thích lựa chọn từ ngữ:

- "Trả lời bằng tiếng Việt" giúp trợ lý sử dụng ngôn ngữ phù hợp với người học, từ đó dễ hiểu và thuận tiện hơn khi học các khái niệm khó.
- "Giải thích rõ ràng theo từng bước" giúp biến các vấn đề phức tạp thành những phần nhỏ dễ theo dõi, đặc biệt hữu ích khi học lập trình hoặc các khái niệm kỹ thuật.
- "Không bịa thông tin; nếu không chắc chắn, hãy nói rõ giới hạn" giúp giảm nguy cơ trợ lý đưa ra thông tin sai và khuyến khích người học kiểm chứng những nội dung chưa chắc chắn.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: trợ lý hiện chỉ giữ lại tối đa 3 lượt hội thoại gần nhất, tương đương 6 message. Vì vậy, nếu cuộc trò chuyện kéo dài, trợ lý có thể quên những thông tin quan trọng được đề cập ở các lượt đầu. 
- Cải thiện đề xuất: bổ sung cơ chế tóm tắt lịch sử hội thoại. Khi history vượt quá giới hạn, hệ thống có thể dùng model để tóm tắt những nội dung cũ thành một đoạn summary ngắn, sau đó giữ summary cùng với 3 lượt hội thoại gần nhất khi gửi request mới. Cách này giúp giảm số token sử dụng nhưng vẫn duy trì được ngữ cảnh quan trọng của cuộc trò chuyện trong thời gian dài.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
