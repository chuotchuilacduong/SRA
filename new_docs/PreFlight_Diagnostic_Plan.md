# Pre-Flight Diagnostic Plan — UCB-ProbeHYRR vs CE-HYRR

> **Mục tiêu của tài liệu này**
> Đây là một *runbook chạy được ngay* dành cho Claude Code. Nó **KHÔNG** chạy toàn bộ pipeline UCB-ProbeHYRR.
> Nó chỉ trả lời một câu hỏi duy nhất, bằng chi phí rẻ nhất có thể, **trước khi** đầu tư build pipeline:
>
> > *"Trên dataset này, liệu UCB-ProbeHYRR có chỗ nào để thắng được CE-HYRR/Method7 thường hay không?"*
>
> Nếu câu trả lời là KHÔNG (headroom ≈ 0) → dừng lại, xem lại dataset, không build pipeline.
> Nếu câu trả lời là CÓ → tiến hành pipeline đầy đủ với tự tin rằng phép so sánh sẽ có ý nghĩa.

---

## 0. Nguyên tắc nền (đọc trước khi code)

1. **Recall@50 đã bão hòa** (M4 ≈ Method7 ≈ 91.9). Do đó *không* dùng Recall/nDCG để chứng minh lợi thế. Lợi thế chỉ nằm ở chỗ **"liên quan ngữ nghĩa" ≠ "thực sự có ích"**.
2. Phương pháp mới được thiết kế để chiếm 4 loại "headroom":
   - **No-load opportunity** — query không cần skill nào.
   - **Harmful exposure** — CE-HYRR load skill làm hỏng một đáp án vốn đúng.
   - **False friend** — skill rank cao nhưng utility ≤ 0.
   - **Oracle gap** — gold skill còn dư địa cải thiện so với skill mà CE-HYRR chọn.
3. Diagnostic này **đo trực tiếp 4 con số đó** trên một mẫu nhỏ, bằng verifier-grounded counterfactual probes. Nếu cả 4 ≈ 0, phương pháp mới về lý thuyết không thể thắng.
4. Mọi nhãn phải đến từ **verifier V**, không từ LLM tự chấm chính nó.
5. Generator `G` và verifier `V` ở đây phải **giống hệt** cái sẽ dùng trong pipeline chính (nếu không, headroom đo được không chuyển sang được).

---

## 1. Tiền điều kiện (Claude Code kiểm tra trước khi chạy)

### 1.1 File dữ liệu cần có

| File | Mô tả | Bắt buộc |
|---|---|---|
| `data/m4_top50.jsonl` | top-50 ứng viên của M4 cho mỗi query (xem schema mục 10.1 của Architecture v2) | ✅ |
| `data/skill_corpus.jsonl` | `{skill_id, skill_title, skill_description, skill_content}` | ✅ |
| `data/queries.jsonl` | `{qid, query, gold_skill_ids, gold_answer, task_type, dataset_id}` | ✅ |
| `data/ce_hyrr_top1.jsonl` (hoặc `method7_top1.jsonl`) | `{qid, ce_top1_skill_id}` — top-1 của baseline cần so sánh | ⚠️ xem 1.3 |

### 1.2 Mô hình & công cụ

```text
G = Qwen2.5:7B-Instruct   (qua vLLM khuyến nghị, hoặc Ollama)
V = verifier theo task_type:
      code   → execution + unit test (sandbox)
      math   → numeric checker (exact / tolerance)
      mcq    → exact option match
      toolqa → structured answer match
      open   → LLM-judge (đánh dấu weak_label, loại khỏi kết luận chính)
```

### 1.3 Nếu CE-HYRR chưa được train

Diagnostic vẫn chạy được. Dùng baseline thay thế theo thứ tự ưu tiên:
1. **CE-HYRR / Method7 đã train** → tốt nhất, đo đúng đối thủ.
2. **Method7 β=0.7 nếu đã có** → xấp xỉ tốt.
3. **M4 top-1** → cận dưới (lower bound). Headroom đo trên M4 top-1 sẽ ≥ headroom thật của CE-HYRR, nên nếu *M4 top-1 đã không có headroom* thì CE-HYRR chắc chắn cũng không → kết luận no-go vẫn vững. Nếu M4 top-1 có headroom, cần train CE-HYRR rồi đo lại để chốt.

> **Claude Code:** ghi rõ trong report baseline nào đã được dùng.

---

## 2. Kiến trúc thư mục đầu ra

```text
preflight/
├── config.yaml                  # tham số (sample size, seed, model, ngưỡng)
├── scripts/
│   ├── 00_check_recall.py       # xác minh Recall@50 bão hòa (không cần LLM)
│   ├── 01_sample.py             # chọn mẫu phân tầng
│   ├── 02_build_prompts.py      # dựng prompt cho 3 (hoặc 5) probe / query
│   ├── 03_run_probes.py         # gọi G, có cache
│   ├── 04_verify.py             # chạy V trên mọi generation
│   ├── 05_metrics.py            # tính 4 headroom + CI + per-stratum
│   └── 06_report.py             # sinh report markdown + bảng go/no-go
├── cache/
│   └── generations.sqlite       # cache (qid, skill_id, prompt_hash) → output
└── out/
    ├── sample.jsonl
    ├── probes.jsonl             # 1 dòng / lần probe
    ├── verified.jsonl
    ├── headroom_metrics.json
    └── REPORT.md                # kết quả + quyết định go/no-go
```

---

## 3. Trình tự thực nghiệm (6 bước)

### Bước 0 — Xác minh giả định bão hòa Recall@50  *(rẻ, KHÔNG cần LLM)*

Mục đích: toàn bộ luận điểm dựa trên "Recall@50 đã bão hòa". Xác nhận điều này đúng trên *dữ liệu thật của bạn* trước khi làm gì khác.

```python
# 00_check_recall.py — logic cốt lõi
# Với mỗi query: gold ∈ top50 của M4? gold ∈ top50 của CE-HYRR?
def recall_at_k(rows, k):
    hit = sum(any(g in [c["skill_id"] for c in r["topk"][:k]]
                  for g in r["gold_skill_ids"]) for r in rows)
    return hit / len(rows)

# Báo cáo: Recall@50 (M4) vs Recall@50 (CE-HYRR), và Recall@1/@5/@10
```

**Diễn giải:**
- Nếu `|Recall@50(M4) − Recall@50(CE-HYRR)| < ~1pp` → giả định đúng, đi tiếp.
- Nếu CE-HYRR cao hơn M4 nhiều ở Recall@50 → pool M4 đang mất coverage; cân nhắc đổi pool generator trước khi nói về utility.

⏱️ **Thời gian: 2–5 phút** (chỉ đọc file + đếm).

---

### Bước 1 — Chọn mẫu phân tầng

Mục đích: ~200 query, **phân tầng** để mỗi regime đủ số lượng (tránh mẫu toàn query "dễ").

```python
# 01_sample.py — phân tầng theo vị trí gold trong M4 top50
# Stratum A: gold ∉ top50            (cận trên của abstain headroom)
# Stratum B: gold = rank 1            (CE đã đúng, ít headroom)
# Stratum C: gold ∈ rank 2..10
# Stratum D: gold ∈ rank 11..50
# Stratum E: query không có gold_skill (ứng viên no-load)
#
# Mục tiêu: mỗi stratum >= 30 query (đủ để ước lượng tỉ lệ với CI ~±0.18).
# Nếu một stratum thiếu, lấy hết những gì có và ghi chú trong report.
TARGET_PER_STRATUM = 40
SEED = 13
```

> Lý do 30–40/stratum: để tỉ lệ kiểu "harmful exposure 15%" có khoảng tin cậy đủ hẹp để ra quyết định. Tổng ~150–200 query.

⏱️ **Thời gian: < 1 phút.**

---

### Bước 2 — Dựng prompt cho các probe

Với mỗi query trong mẫu, dựng **3 prompt bắt buộc** (MVP) và tuỳ chọn thêm 2:

```text
BẮT BUỘC (3 probe / query):
  P_no    : G(q)                      → v_no
  P_gold  : G(q + gold_skill)         → v_gold   (chỉ khi có gold)
  P_ce    : G(q + ce_top1_skill)      → v_ce     (skill CE-HYRR chọn)

TUỲ CHỌN — "extended" (đo trực tiếp false-friend trong top-10):
  P_r2    : G(q + skill rank 2 của CE-HYRR)
  P_r3    : G(q + skill rank 3 của CE-HYRR)
```

Format prompt phải khớp với cách incorporation thật trong pipeline (full skill content injection, mục 4.1 / 12.2). Dùng **một template duy nhất** cho mọi điều kiện, chỉ thay phần skill, để không gây confound.

⏱️ **Thời gian: 1–2 phút.**

---

### Bước 3 — Chạy probes (phần tốn kém nhất)

```python
# 03_run_probes.py
# - Cache theo key = hash(prompt). KHÔNG gọi lại nếu đã có trong cache/generations.sqlite
# - Dùng vLLM offline batching nếu có (throughput cao gấp nhiều lần Ollama tuần tự)
# - max_new_tokens vừa đủ cho task (code/math thường ≤ 512)
# - temperature = 0 (greedy) để verifier ổn định và tái lập được
```

**Số lần gọi G:**

```text
MVP   : ~200 query × 3 probe  ≈ 600 generations
        (query không gold thì bỏ P_gold → có thể ít hơn)
Extended: ~200 query × 5 probe ≈ 1000 generations
```

⏱️ **Thời gian (xem chi tiết mục 5):**
- vLLM batched, 1×GPU 24GB (vd 4090/A100): **~10–25 phút** cho 600 gen.
- Ollama tuần tự, 1×GPU: **~60–110 phút** cho 600 gen.

---

### Bước 4 — Chạy verifier

```python
# 04_verify.py — định tuyến theo task_type
# code   : chạy unit test trong sandbox (timeout mỗi case)
# math   : so khớp số (exact / tolerance)
# mcq    : khớp đáp án
# open   : LLM-judge → weak_label (loại khỏi kết luận chính, chỉ tham khảo)
# Output: verifier_score ∈ {0,1} + verifier_type ('strong'/'weak')
```

⏱️ **Thời gian:**
- Rule/numeric/mcq: **vài giây tổng cộng**, gần như tức thì.
- Code execution: **5–20 phút** tuỳ số test & timeout (chạy song song được).
- LLM-judge: thêm ~1 generation/mẫu → cộng tương đương một phần Bước 3.

---

### Bước 5 — Tính 4 headroom metric + CI + per-stratum

```python
# 05_metrics.py — utility = v_s - v_no
# Tính trên TOÀN mẫu và TỪNG stratum, kèm bootstrap 95% CI (B=2000).

# 1) No-load opportunity   = P(v_no == 1)
# 2) Harmful exposure (CE) = P(v_no == 1 AND v_ce == 0)        # CE phá đáp án đúng
# 3) False-friend rate     = P(rank_ce_top1 cao AND utility_ce <= 0)
#      (extended: P(tồn tại skill trong top-10 CE có rank cao & utility <= 0))
# 4) Oracle gap            = mean(v_gold) - mean(v_ce)          # chỉ trên query có gold

# Bổ sung hữu ích:
#   CE-utility win rate    = P(utility_ce > 0)   # CE top-1 thực sự giúp bao nhiêu %
#   Need-external rate     = P(v_no == 0 AND v_gold == 1)
```

⏱️ **Thời gian: < 1 phút.**

---

### Bước 6 — Sinh report + quyết định Go/No-Go

```python
# 06_report.py → out/REPORT.md
# Bảng: metric | toàn mẫu (CI) | per-stratum | cờ go/no-go
```

⏱️ **Thời gian: < 1 phút.**

---

## 4. Tiêu chí quyết định Go / No-Go

Áp ngưỡng trên **toàn mẫu** (và xem per-stratum để biết lợi thế đến từ đâu).

| Headroom metric | 🟢 Go (có cơ hội rõ) | 🟡 Cân nhắc | 🔴 No-go |
|---|---|---|---|
| Harmful exposure (CE) | ≥ 8% | 3–8% | < 3% |
| No-load opportunity | ≥ 15% | 5–15% | < 5% |
| False-friend rate | ≥ 15% | 7–15% | < 7% |
| Oracle gap | ≥ 5pp | 2–5pp | < 2pp |
| CE-utility win rate | ≤ 70% (còn nhiều chỗ sai) | 70–85% | > 90% (CE đã gần tối ưu) |

**Quy tắc kết luận:**

```text
GO        : >= 2 metric ở vùng 🟢  → build pipeline đầy đủ, kỳ vọng gain có cơ sở.
CONDITIONAL: 1 metric 🟢 hoặc nhiều 🟡 → build nhưng thu hẹp claim (vd chỉ tập trung
             vào no-load + harmful-exposure, không hứa cải thiện trên mọi regime).
NO-GO     : tất cả 🔴  → KHÔNG build. Utility ≈ relevance trên dataset này. Hành động:
             đổi/khó hoá dataset, thêm task không cần skill, hoặc thêm skill gây nhiễu.
```

> **Lưu ý quan trọng:** nếu baseline dùng ở Bước 1.3 là **M4 top-1** (không phải CE-HYRR thật) và kết quả ra GO, bạn vẫn cần train CE-HYRR rồi chạy lại Bước 3–6 với `ce_hyrr_top1.jsonl` để chốt — vì CE-HYRR có thể đã tự lấp một phần headroom mà M4 top-1 để lộ.

---

## 5. Ước tính thời gian tổng

Giả định **~200 query, MVP 3 probe (~600 generations)**, Qwen2.5:7B, một GPU 24GB.

### Kịch bản A — vLLM offline batching (khuyến nghị)

| Bước | Việc | Thời gian |
|---|---|---|
| Setup | cài vLLM, load model, chuẩn data | 20–40 phút (một lần) |
| 0 | check Recall@50 | 2–5 phút |
| 1 | sampling | < 1 phút |
| 2 | build prompts | 1–2 phút |
| 3 | chạy 600 generations (batched) | 10–25 phút |
| 4 | verify (rule/math/mcq) | < 5 phút |
| 4 | verify (code execution) | +5–20 phút |
| 5 | metrics + CI | < 1 phút |
| 6 | report | < 1 phút |
| **Tổng (rule-based V)** | | **~40–80 phút** (sau setup) |
| **Tổng (code-execution V)** | | **~50–100 phút** (sau setup) |

### Kịch bản B — Ollama tuần tự (không batch)

| Bước | Thời gian |
|---|---|
| 3 — 600 generations tuần tự (~8s/gen) | **60–110 phút** |
| Các bước khác | như trên |
| **Tổng** | **~1.5–2.5 giờ** (sau setup) |

### Kịch bản C — Extended (5 probe, ~1000 gen) + LLM-judge

Cộng thêm ~0.6–1× thời gian Bước 3. Tổng thực tế **~2–4 giờ**.

> **Yếu tố ảnh hưởng lớn nhất tới thời gian:** (1) batching hay không, (2) `max_new_tokens`, (3) verifier code-execution có timeout dài không. Cache làm cho mọi lần chạy lại gần như tức thì.

**Khuyến nghị thực tế:** chạy Kịch bản A, MVP 3-probe, verifier rule/execution. Đặt mục tiêu **có REPORT.md trong vòng ~1.5 giờ kể từ lúc bắt đầu** (gồm cả setup). Nếu kết quả ở ranh giới 🟡, mới nâng lên Extended.

---

## 6. Checklist tránh confound (Claude Code tự kiểm trước khi kết luận)

- [ ] Cùng một template prompt cho cả 3 điều kiện (no / gold / ce), chỉ khác phần skill.
- [ ] Cùng `G`, cùng `temperature=0`, cùng `max_new_tokens` cho mọi probe.
- [ ] Cùng verifier `V` cho mọi điều kiện; ghi `verifier_type` (strong/weak).
- [ ] Query mẫu KHÔNG trùng với query sẽ dùng train/UCB sau này (ghi lại danh sách qid đã dùng).
- [ ] Mẫu được phân tầng, không phải random thuần (nếu random thuần, ghi rõ và đừng đọc per-stratum).
- [ ] Open-ended/LLM-judge bị loại khỏi kết luận go/no-go chính (chỉ tham khảo).
- [ ] Report ghi rõ baseline nào (CE-HYRR thật / Method7 / M4 top-1).
- [ ] Bootstrap CI được báo cáo, không chỉ điểm ước lượng.

---

## 7. Lệnh chạy (Claude Code thực thi tuần tự)

```bash
# 0. cài đặt
pip install vllm jsonlines numpy scipy pyyaml tqdm   # + bộ verifier của bạn

# 1. chạy tuần tự
python scripts/00_check_recall.py   --config config.yaml
python scripts/01_sample.py         --config config.yaml
python scripts/02_build_prompts.py  --config config.yaml
python scripts/03_run_probes.py     --config config.yaml   # phần lâu nhất, có cache
python scripts/04_verify.py         --config config.yaml
python scripts/05_metrics.py        --config config.yaml
python scripts/06_report.py         --config config.yaml

# 2. đọc kết quả
cat out/REPORT.md
```

`config.yaml` gợi ý:

```yaml
seed: 13
sample:
  target_per_stratum: 40
model:
  name: "Qwen2.5-7B-Instruct"
  backend: "vllm"          # hoặc "ollama"
  temperature: 0.0
  max_new_tokens: 512
probe:
  mode: "mvp"              # "mvp" (3) hoặc "extended" (5)
baseline:
  source: "ce_hyrr"        # "ce_hyrr" | "method7" | "m4_top1"
  file: "data/ce_hyrr_top1.jsonl"
thresholds:                # ngưỡng go/no-go ở mục 4
  harmful_exposure_go: 0.08
  no_load_go: 0.15
  false_friend_go: 0.15
  oracle_gap_go: 0.05
verifier:
  code_timeout_s: 10
  bootstrap_iters: 2000
```

---

## 8. Đầu ra cuối cùng cần xem

`out/REPORT.md` phải trả lời gọn:

```text
1. Recall@50 có bão hòa không?           (Bước 0)
2. 4 headroom metric + CI là bao nhiêu?  (Bước 5)
3. Lợi thế nằm ở stratum nào?            (per-stratum)
4. Quyết định: GO / CONDITIONAL / NO-GO  (mục 4)
5. Nếu GO: nên hứa gain ở những regime nào trong paper.
```

Chỉ khi report kết luận **GO** (hoặc **CONDITIONAL** với phạm vi claim đã thu hẹp) thì mới chuyển sang build pipeline UCB-ProbeHYRR đầy đủ.
