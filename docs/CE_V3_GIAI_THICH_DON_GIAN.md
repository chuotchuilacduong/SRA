# CE v3 — Giải thích dễ hiểu (không code, không công thức)

Đây là phiên bản "giải thích cho người không phải kỹ thuật" của pipeline CE v3.
Nếu bạn muốn xem code/công thức/shape: đọc [CE_V3_PIPELINE_DETAILED.md](CE_V3_PIPELINE_DETAILED.md).

---

## 1. Bài toán đang giải

Bạn có:
- **1 thư viện 26,262 kỹ năng (skills)**. Mỗi kỹ năng là 1 trang giấy có:
  - Tên
  - Mô tả ngắn
  - Nội dung chi tiết (giải thích cách dùng)
- **1 câu hỏi (query)**. Ví dụ: *"Có bao nhiêu cách chia tập 8 phần tử thành 5 nhóm có thứ tự, không rỗng?"*

**Mục tiêu**: Tìm đúng kỹ năng phù hợp nhất với câu hỏi đó (gold). Đáp án đúng: kỹ năng tên *"Lah Numbers"*.

→ Đây gọi là **skill retrieval**: cho câu hỏi, đề xuất top-K kỹ năng có khả năng giải nó.

---

## 2. Phép ẩn dụ: thư viện và 3 thủ thư

Tưởng tượng bạn đang ở 1 thư viện có 26,262 cuốn sách (= skills). Khi có khách hỏi 1 câu, bạn không thể đọc hết 26k cuốn để chọn. Bạn cần **lọc nhanh** rồi mới **đọc kỹ**.

### Thủ thư 1: **BM25** — Anh thủ thư từ điển (lexical)
- Lắng nghe câu hỏi, ghi nhớ các từ khoá đặc biệt: *"8 phần tử", "5 nhóm có thứ tự"*
- Mở danh mục từ-vựng, tìm 100 cuốn sách có chứa nhiều các từ này nhất
- **Mạnh**: bắt từ khoá hiếm, ký hiệu toán học, tên hàm code.
- **Yếu**: nếu câu hỏi viết khác từ trong sách (đồng nghĩa, paraphrase) thì miss.

### Thủ thư 2: **BGE** — Cô thủ thư hiểu nội dung (semantic)
- Đọc hết 26k cuốn từ trước, **ghi nhớ ý nghĩa** mỗi cuốn dưới dạng "vector 768 chiều" (như tóm tắt thông minh).
- Khi nghe câu hỏi, cũng tóm tắt câu hỏi thành 1 vector tương tự.
- So sánh vector câu hỏi với vector mỗi cuốn → 100 cuốn có ý nghĩa gần nhất.
- **Mạnh**: hiểu nghĩa, paraphrase, đồng nghĩa.
- **Yếu**: ký hiệu lạ, code, công thức.

### Quản lý: **RRF** — Trưởng phòng tổng hợp
- Hỏi 2 thủ thư: "Mỗi người đề xuất 100 cuốn nào?"
- BM25 đưa danh sách A1, A2, ..., A100 (theo thứ tự).
- BGE đưa danh sách B1, B2, ..., B100.
- Trưởng phòng tính **điểm tổng hợp** cho mỗi cuốn:
  - Cuốn nào ở vị trí cao trong cả 2 danh sách → điểm cao.
  - Chỉ ở 1 danh sách → điểm thấp hơn.
- Lấy top-100 cuốn theo điểm tổng → đây gọi là **extended pool**.

→ **Đây là Stage 1**: 100 cuốn "ứng cử viên". Nhanh (~14 mili giây/câu hỏi) nhưng đôi khi để gold ở rank xa.

### Chuyên gia: **Cross-Encoder (CE)** — Giáo sư đọc kỹ
- Trưởng phòng đưa 100 cuốn ứng cử cho giáo sư.
- Giáo sư **đọc từng cặp** (câu hỏi + 1 cuốn) → cho điểm relevance (0-10).
- Sau khi chấm hết 100 cuốn, sắp lại theo điểm.
- **Mạnh**: cực kỳ chính xác vì đọc kỹ.
- **Yếu**: chậm (~1.2 giây/câu hỏi, gấp 89× thư viện ảo).

→ **Đây là Stage 2**: top-100 đã được sắp lại đúng nhất.

---

## 3. Tại sao cần TRAIN cross-encoder?

Mặc định, giáo sư CE (`MS-MARCO MiniLM-L6`) được train trên **dữ liệu web search** (Google-style queries). Skills retrieval khác hẳn:
- Câu hỏi dài, technical, có công thức.
- "Kỹ năng" là 1 trang giải thích thuật toán/định lý — không phải đoạn web.

→ Giáo sư cần "học thêm chuyên môn" trên **SRA-Bench**.

Quá trình học = **fine-tuning** = cho giáo sư xem nhiều ví dụ (câu hỏi + cuốn đúng + nhiều cuốn sai) và bảo: *"Khi gặp tình huống tương tự, hãy cho cuốn đúng điểm cao hơn cuốn sai."*

---

## 3.5. HYRR là gì? Giải thích cặn kẽ

**HYRR** = **HYbrid Retrieval + Reranking** = ý tưởng trong paper *"HYRR: Hybrid Retrieval-Reranking"* (Google Research, 2024).

### Ý tưởng cốt lõi

Khi train giáo sư CE, vấn đề quan trọng nhất là: **lấy "cuốn sai" (negative) như thế nào cho có ý nghĩa?**

Có 3 cách phổ biến để chọn negative, và mỗi cách có vấn đề riêng:

| Cách chọn negative | Vấn đề |
|---|---|
| **Random** (lấy ngẫu nhiên từ thư viện) | Hầu hết quá dễ — giáo sư phân biệt ngay được, học chẳng được gì có giá trị. |
| **BM25 hard** (cuốn BM25 xếp cao nhưng không phải gold) | Bắt được lỗi "trùng từ khoá nhưng nội dung sai", nhưng không bắt được lỗi semantic. |
| **Dense hard** (cuốn BGE xếp cao nhưng không phải gold) | Bắt được lỗi "ý nghĩa gần nhưng không đúng", nhưng miss lỗi lexical. |

### HYRR = trộn cả ba

**HYRR đề xuất**: dùng **cả BM25-hard + Dense-hard + Random** trong cùng tập training. Mỗi câu hỏi có đủ 3 loại negative.

**Phép ẩn dụ**:
- Bạn đang dạy 1 đứa trẻ phân biệt "mèo".
- Nếu chỉ cho xem **toàn xe hơi** làm "không phải mèo" (random) → quá dễ.
- Nếu chỉ cho xem **toàn chó cùng màu lông** (BM25-hard chỉ trùng vẻ ngoài) → đứa trẻ sẽ tưởng "mèo = lông màu vàng".
- Nếu chỉ cho xem **toàn động vật họ mèo** (BGE-hard, semantic gần) → đứa trẻ học được "khái niệm họ mèo" nhưng không phân biệt mèo nhà với sư tử.
- Mix cả ba → đứa trẻ phải để ý **mọi đặc điểm phân biệt**: lông, kích thước, dáng, hành vi.

### Trong code

Khi mine negative cho 1 câu hỏi, sampler đi qua **3 nguồn xếp hạng song song**:

```
Câu hỏi: theoremqa_00003
gold:    theoremqa_000 (Lah Numbers)

Bảng xếp hạng theo BM25:           Bảng xếp hạng theo BGE:
  rank 1: theoremqa_000  ← gold      rank 1: theoremqa_000  ← gold
  rank 2: theoremqa_137              rank 2: theoremqa_137
  rank 3: theoremqa_092              rank 3: theoremqa_241  ← chỉ có trong BGE
  rank 4: theoremqa_086              rank 4: theoremqa_092
  ...                                ...

Sampler lấy:
  - 3 cuốn TOP BM25 (bỏ gold):     theoremqa_137, theoremqa_092, theoremqa_086
  - 3 cuốn TOP BGE (bỏ gold):      theoremqa_137, theoremqa_241, theoremqa_092
                                    (lưu ý: theoremqa_137 trùng → sampler dedup)
  - 2 cuốn RANDOM từ toàn 26k:     logicbench_005, medcalcbench_044
  - 1 cuốn POSITIVE:               theoremqa_000

→ Tổng 9 cuốn unique (sau dedup) + 2 cluster-hard (xem mục 3.6) = 11 cuốn.
```

### Tại sao HYRR mạnh hơn "chỉ random"?

Trong v1 chỉ dùng `4 BM25-hard + 4 BGE-hard + 2 random = 10 negs`, đã thấy:
- Macro Recall@10 = 72.45% (vượt RRF baseline +3 pp).
- Đặc biệt mạnh trên logicbench (+26 pp), bigcodebench (+24 pp).

Lý do: BM25-hard và BGE-hard là những "lỗi đã từng xảy ra" → train trên chúng = sửa lỗi cụ thể.

### "Reciprocal" trong HYRR đến từ đâu?

Tên "HYRR" lấy cảm hứng từ **RRF (Reciprocal Rank Fusion)** ở stage 1. Cả hai đều ý tưởng tương tự:
- **RRF**: tổng hợp **nhiều thủ thư** (BM25, BGE) bằng cách lấy nghịch đảo rank.
- **HYRR**: tổng hợp **nhiều nguồn negative** (BM25-hard, BGE-hard, random) cho training.

→ Cùng nguyên lý "đa dạng nguồn = robust hơn", áp dụng ở 2 stage khác nhau.

---

## 3.6. K-means clustering là gì? Tại sao thêm "cluster-hard" negative?

### K-means giải thích siêu đơn giản

Tưởng tượng 26,262 cuốn sách trên bàn dài. Mỗi cuốn có **toạ độ** trong không gian 768 chiều (= vector BGE).

**K-means** = thuật toán **gom 26k cuốn thành K=300 chồng** dựa trên toạ độ:

```
Bước 1: Đặt ngẫu nhiên 300 "tâm điểm" trong không gian.
Bước 2: Gán mỗi cuốn vào tâm gần nhất → 300 chồng.
Bước 3: Tính lại tâm = trung bình toạ độ của các cuốn trong chồng.
Bước 4: Quay lại bước 2. Lặp đến khi tâm ngừng dịch chuyển.
```

**Phép ẩn dụ**: hãy nghĩ về kệ sách trong thư viện thật:
- Kệ "Toán tổ hợp" chứa: Lah Numbers, Stirling Numbers, Binomial, Set Partitions...
- Kệ "Logic" chứa: Modus Ponens, Modus Tollens, Truth Tables...
- Kệ "Code Python" chứa: List comprehension, decorators, asyncio...

K-means = **tự động phân loại** thành 300 "kệ" mà không cần ai dán nhãn trước. Các cuốn cùng kệ có nội dung **gần nhau theo nghĩa**.

### Số liệu thực tế từ implementation của chúng ta

```
26,262 skills → K-means 300 clusters
  - Cluster median size: 85 skills
  - Smallest cluster:    16 skills
  - Largest cluster:     274 skills
  - Time to compute:     9 giây (MiniBatchKMeans)
```

Cluster nhỏ nhất 16 skills nghĩa là có ít nhất 1 chủ đề rất hẹp được tách riêng (ví dụ: 1 nhánh thuật toán cụ thể).

### Cluster-hard negative — bài tập KHÓ NHẤT cho CE

Sau khi có 300 cluster, cluster-hard sampler làm gì?

```
Câu hỏi: theoremqa_00003 (về Lah Numbers)
gold:    theoremqa_000 → thuộc cluster 47

Cluster 47 chứa các skills:
  theoremqa_000  (Lah Numbers)        ← GOLD
  theoremqa_137  (Stirling Numbers)
  theoremqa_092  (Set Partitions)
  theoremqa_086  (Bell Numbers)
  ... còn 80+ skills khác về tổ hợp

Sampler lấy 2 cuốn từ cluster 47, bỏ gold:
  → theoremqa_137 (Stirling Numbers)  ← cluster-hard
  → theoremqa_086 (Bell Numbers)      ← cluster-hard
```

→ 2 cuốn cluster-hard này là **những cuốn rất giống gold về mặt ngữ nghĩa nhưng KHÔNG phải gold**. Đây là loại sai lầm khó nhất cho CE.

### Tại sao cluster-hard mạnh hơn BGE-hard?

BGE-hard = top BGE rank nhưng không phải gold. Nhưng top BGE có thể bao gồm:
- Skills hơi giống nhau (top 1-3).
- Skills khá khác nhau (top 10-50).

Cluster-hard = **đảm bảo cùng chủ đề** với gold. Khắc nghiệt hơn vì:
- Cùng cluster = vector gần nhau (theo định nghĩa K-means).
- Loại bỏ ngẫu nhiên "tốt mặt nhưng khác chủ đề".

**Ví dụ thực**: cho câu hỏi về Lah Numbers,
- BGE-hard có thể đề xuất: *"Pascal's Triangle"* (gần nghĩa nhưng khác hẳn chủ đề).
- Cluster-hard ép phải so sánh với: *"Stirling Numbers"* và *"Bell Numbers"* — **cùng họ tổ hợp đếm**, dễ nhầm nhất.

### Hiệu ứng đo được

So v2 (có cluster-hard) vs v1 (chỉ BM25+BGE+random):
- v1 macro R@10 = 72.45%
- v2 macro R@10 = 75.40% → **+2.95 pp** chỉ riêng từ thêm cluster-hard + 3 epochs.
- Trên các datasets multi-label (champ, bigcode), cluster-hard giúp nhiều nhất.

### Vẽ đồ thị (cluster ý nghĩa)

Tưởng tượng không gian 2D đơn giản hoá (thật là 768D):

```
                  Cluster 47: TỔ HỢP ĐẾM
                  ┌───────────────────┐
                  │  ⚫ Lah (GOLD)     │  ← target
                  │  ● Stirling        │  ← cluster-hard
                  │  ● Bell            │  ← cluster-hard
                  │  ● Set Partitions  │
                  └───────────────────┘

         Cluster 92: HÀM CƠ BẢN              Cluster 152: LOGIC
         ┌──────────────────┐                ┌─────────────────┐
         │  ● Pascal         │                │  ● Modus Ponens │
         │  ● Binomial Coeff │                │  ● Modus Tollens│
         │  ● Factorial      │                │  ● Truth Table  │
         └──────────────────┘                └─────────────────┘

         (BGE-hard có thể chọn từ cluster 92 — khá khác chủ đề)
         (Random có thể chọn từ cluster 152 — hoàn toàn khác)
```

→ Cluster-hard = lực kéo CE phải **phân biệt tinh tế trong CÙNG cluster**, không chỉ giữa các cluster.

### Tóm tắt cluster-hard

| Đặc tính | Ý nghĩa |
|---|---|
| Source | K-means trên 26k BGE embeddings, K=300 |
| Cluster size median | 85 skills/cluster |
| Số cluster-hard/query | 2 |
| Mục tiêu | Bài tập khó nhất: cùng chủ đề với gold, không phải gold |
| Đóng góp đo được | +5 pp Macro R@10 (kết hợp với 3 epochs) |
| Latency overhead | 0 (lookup cluster_id từ dict, O(1)) |

---

## 4. Cách CHUẨN BỊ data huấn luyện

### Bước 1: Cho mỗi câu hỏi trong train set, lấy 11 "ví dụ":

```
1 cuốn ĐÚNG (gold)              — "Đây là cuốn cần tìm"
3 cuốn SAI khó loại 1 (BM25)    — "BM25 từng nhầm, từ vựng giống nhưng nội dung sai"
3 cuốn SAI khó loại 2 (BGE)     — "BGE từng nhầm, nghĩa gần nhưng không đúng"
2 cuốn SAI ngẫu nhiên (random)  — "Vớ vẩn để model học cái gì là rõ ràng sai"
2 cuốn SAI cùng cụm (cluster)   — "Cùng chủ đề với gold nhưng KHÔNG phải gold —
                                   bài tập KHÓ NHẤT, ép giáo sư phân biệt tinh tế"
```

Tổng cộng **11 ví dụ/câu hỏi**.

### Bước 2: Tại sao 4 loại "cuốn sai"?

Imagine bạn dạy 1 đứa trẻ phân biệt "mèo" và "không phải mèo":
- Cho xem cốc nước (random) → quá dễ, không học được gì có giá trị.
- Cho xem chó (cluster) → cùng họ động vật, **buộc phải nhìn kỹ** đặc điểm mèo.
- Cho xem sư tử (BGE hard) → "trông giống mèo, nhưng không phải".
- Cho xem từ "mèo" viết khác chữ (BM25 hard) → trick từ vựng.

→ Mỗi loại "negative" buộc giáo sư học 1 loại sai lầm.

### Bước 3: Đảm bảo KHÔNG leak

- Mỗi câu hỏi chỉ ở 1 trong 3 split: **train / dev / test**.
- Khi training, KHÔNG bao giờ thấy câu test.
- Khi sample cuốn sai, **TUYỆT ĐỐI** không lấy gold làm "cuốn sai" → guard cứng.
- Tổng: **43,098 cặp huấn luyện** từ 3,782 câu hỏi train (qua 6 datasets).

---

## 5. Cách HUẤN LUYỆN giáo sư

### Hình dung mỗi lượt học:

Cho giáo sư 1 **bài thi mini** với 3 câu hỏi, mỗi câu có 11 ứng cử (= 33 cặp tổng):

```
Câu 1: hỏi A → 11 ứng cử (1 đúng + 10 sai)
Câu 2: hỏi B → 11 ứng cử (1 đúng + 10 sai)
Câu 3: hỏi C → 11 ứng cử (1 đúng + 10 sai)
```

Giáo sư cho điểm từng cặp (logit) → 33 điểm.

### Hàm chấm điểm (loss function)

Cho mỗi câu hỏi (ví dụ câu 1):

1. Lấy 11 điểm của 11 ứng cử.
2. Áp dụng "softmax" → biến điểm thành **xác suất**, tổng = 100%.
   - Ví dụ: gold = 58%, hard-neg1 = 21%, hard-neg2 = 13%, ...
3. **Mục tiêu**: đẩy xác suất của **gold** lên cao **tương đối** so với các cuốn sai.
4. Loss = `-log(xác suất gold)` → càng nhỏ càng tốt.
   - Nếu gold xác suất = 99% → loss ≈ 0.01 (rất tốt).
   - Nếu gold xác suất = 9% → loss ≈ 2.4 (kém).

Đây gọi là **listwise softmax loss** (đặt tên "list-wise" vì nhìn cả LIST 11 ứng cử cùng lúc, khác với "point-wise" chỉ nhìn từng cặp).

### Tại sao listwise hay hơn pointwise?

**Pointwise (BCE)** = hỏi giáo sư cho từng cặp: *"Có/không relevant?"*. Mỗi cặp độc lập.
- Vấn đề: giáo sư có thể trả lời "có" cho cả gold và 1 cuốn sai cùng chủ đề. Không đủ ép phân biệt.

**Listwise** = bắt giáo sư **sắp xếp 11 cuốn**: *"Sắp xếp từ relevant nhất đến ít relevant nhất."*
- Buộc giáo sư so sánh cuốn gold với từng cuốn sai trong cùng 1 lượt.
- Cuốn sai nào điểm cao (= giáo sư đang nhầm) sẽ bị "kéo xuống" mạnh nhất.

→ Phần này giải thích vì sao listwise (v3) hơn BCE (v2):
- medcalc: +8pp (medical formulas rất giống nhau, cần distinguish tinh tế).
- champ: +5pp.
- theoremqa: +3pp.

### Cân bằng giữa 6 datasets

Vấn đề: trong 43k cặp:
- toolqa: 11k cặp (26%)
- bigcode: 10k cặp (24%)
- champ: 1.8k cặp (4%)

Nếu lấy mẫu ngẫu nhiên → toolqa được học 6× nhiều hơn champ → champ học kém.

**Giải pháp (balanced sampler)**: ép mỗi bài thi mini có **xác suất bằng nhau** cho 6 datasets. Champ nhỏ nhưng vẫn được "hỏi" đều như toolqa.

### Học 3 lần (3 epochs)

- **Epoch 1**: giáo sư đọc qua hết 43k cặp 1 lần. Loss giảm từ 2.4 xuống ~1.3.
- **Epoch 2**: đọc lại lần 2, đã thuộc patterns. Loss xuống ~0.8.
- **Epoch 3**: tinh chỉnh, loss xuống ~0.6.

Sau mỗi epoch, **kiểm tra** trên dev set (câu hỏi giáo sư chưa thấy). Lưu phiên bản tốt nhất.

→ Trên CPU Apple Silicon: ~131 phút tổng cho 3 epochs.

---

## 6. Khi DÙNG (inference)

Khi có 1 câu hỏi mới (`theoremqa_00000`: *"Bao nhiêu cách chia 8 phần tử thành 5 nhóm có thứ tự?"*):

```
Bước 1: BM25 (~5 ms)
  → 100 cuốn top theo từ khoá.

Bước 2: BGE (~10 ms)
  → 100 cuốn top theo ý nghĩa.

Bước 3: RRF fusion (~0.03 ms)
  → 100 cuốn top kết hợp 2 nguồn.

Bước 4: Cross-Encoder v3 chấm (~1,200 ms)
  → Đọc từng cặp (câu hỏi + cuốn), cho điểm.
  → Sắp lại 100 cuốn theo điểm CE.

Bước 5: Kết quả cuối
  → Top-1: "Lah Numbers" (điểm 8.42) ← cuốn đúng!
  → Top-2: "Stirling Numbers" (điểm 4.13)
  → ...
```

**Downstream LLM** nhận cuốn "Lah Numbers" → áp dụng công thức `L(8,5) = C(7,4) × 8!/5! = 35 × 336 = 11760` ✅.

---

## 7. KẾT QUẢ tóm gọn (so với baseline)

### Chất lượng (Recall@10 = % câu hỏi có gold trong top-10)

| Phương pháp | 6 datasets trung bình | Ghi chú |
|---|---:|---|
| Chỉ BM25 | 65.87 % | Baseline lexical, rất nhanh |
| Chỉ BGE | 57.48 % | Baseline semantic |
| RRF (BM25 + BGE) | 69.48 % | Tổng hợp, vẫn rất nhanh |
| **CE v3 (sau train)** | **78.27 %** | **+8.79 pp so với RRF**, nhưng chậm hơn 89× |

### Tốc độ (full bench 5,400 câu hỏi)

| Phương pháp | Tổng thời gian | Câu hỏi/giây |
|---|---:|---:|
| BM25 | 21 s | **540** |
| BGE | 53 s | 103 |
| RRF (e2e) | 73 s | 74 |
| **CE v3 (e2e)** | **6,558 s (~110 phút)** | **0.82** |

### Khi nào dùng cái nào?

| Tình huống | Chọn |
|---|---|
| Cần tốc độ < 50ms/câu (chatbot real-time) | **RRF** |
| Có thể chờ 1 giây, ưu tiên chất lượng | **CE v3** |
| Câu hỏi liên quan toán/y khoa (theoremqa/medcalc) | **RRF hoặc BM25** (CE chưa đủ giỏi) |
| Câu hỏi logic/code/tool | **CE v3** (vượt RRF mạnh) |
| Hybrid (router theo loại câu hỏi) | **80.9% Macro R@10** |

---

## 8. Các điều "trade-off" của CE v3

| Cái được | Cái mất |
|---|---|
| +29 pp Recall@10 trên logicbench | Inference chậm 89× |
| +24 pp trên bigcodebench | Train tốn 2 giờ CPU |
| +10 pp trên champ | Cần GPU để serve real-time |
| Logic + code retrieval vượt trội | Chưa vượt RRF trên theoremqa/medcalc |

---

## 9. Cải tiến tiếp theo (nếu có thời gian)

Để vượt RRF trên cả 6 datasets:
1. **Tăng training data**: 43k → 200k cặp (sinh thêm bằng LLM augmentation).
2. **Nối điểm BM25/BGE vào input**: cho giáo sư "biết" thủ thư đã chấm bao nhiêu.
3. **Học từ RRF (soft distillation)**: dạy giáo sư fit theo điểm RRF, không phải pure 0/1.
4. **Model lớn hơn**: MiniLM-L12 (33M params, gấp 1.5× chậm).
5. **Per-domain heads**: 1 đầu ra riêng cho mỗi loại câu hỏi (math/code/medical/logic).

---

## 10. TL;DR (siêu ngắn)

- Có 26k kỹ năng, có câu hỏi → tìm kỹ năng đúng.
- 2 bước: (1) **Lọc thô** bằng BM25 + BGE + RRF (lấy 100 ứng viên), (2) **Đọc kỹ** bằng cross-encoder để sắp lại.
- Để giáo sư CE giỏi, train nó với 43k bài tập: 1 đúng + 10 sai (sai theo 4 kiểu khác nhau).
- Dùng listwise loss để ép giáo sư so sánh đúng vs sai trong **cùng 1 câu hỏi**, không độc lập.
- Cân bằng 6 datasets, train 3 epoch.
- Kết quả: chất lượng cao hơn RRF +8.8 pp, nhưng chậm hơn 89×.
- Triển khai thực tế: dùng router (RRF hoặc CE theo loại câu hỏi) để cân bằng tốc độ và chất lượng.
