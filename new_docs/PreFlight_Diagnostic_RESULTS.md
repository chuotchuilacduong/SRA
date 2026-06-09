# PreFlight Diagnostic — KẾT QUẢ & PHÂN TÍCH

> Thực thi runbook [`PreFlight_Diagnostic_Plan.md`](PreFlight_Diagnostic_Plan.md) trên dữ liệu thật.
> Câu hỏi: *"Trên dataset này, UCB-ProbeHYRR có chỗ nào để thắng được CE-HYRR thường không?"*
>
> **Kết luận ngắn: 🟢 GO** — nhưng chỉ kết luận được trên **logicbench + medcalcbench** (single-shot). toolqa và bigcodebench bị loại vì lý do *kỹ thuật harness*, không phải vì hết headroom (xem §6).

Ngày chạy: 2026-06-08. Artifacts: `results/preflight/{headroom_metrics.json, probes.jsonl, probe_cache.jsonl, run_full.log}`.

---

## 1. Cấu hình thực nghiệm (đã kiểm chứng chạy được)

| Thành phần | Giá trị thực tế |
|---|---|
| Generator `G` | **Qwen2.5:7B-Instruct** qua Ollama (`localhost:11434`, M3 Max/Metal), `temperature=0`, `max_tokens=512` |
| Verifier `V` | `sragents.evaluate` — checker tất định theo dataset (logicbench=exact MCQ, medcalcbench=numeric, toolqa=structured) |
| Baseline đối thủ | **CE-HYRR = CE v3 rerank top-1** (`results/rerank/test-kmeans_M30-*-ce-v3-top20.json`) — đối thủ thật, không phải proxy |
| Engine | `DirectEngine` (single-shot) — *quan trọng cho việc loại toolqa, xem §6* |
| Mẫu | **222 query**, phân tầng theo rank gold trong M4 pool: A_absent 42 · B_rank1 60 · C_rank2_10 60 · D_rank11_50 60 |
| Probe/query | 3 bắt buộc: `v_no=G(q)`, `v_gold=G(q+gold)`, `v_ce=G(q+ce_top1)`; có cache |
| Datasets chạy được | logicbench, toolqa, medcalcbench |

> Checklist confound (plan §6): cùng template, cùng G/temperature/max_tokens, cùng V, mẫu phân tầng, baseline ghi rõ là CE v3 top-1, bootstrap CI 2000. ✅

---

## 2. Bước 0 — Recall@50 đã bão hòa? (không cần LLM)

M4 và CE-HYRR cùng rerank chung **một pool RRF top-100** ⇒ trần coverage (Recall@100) **giống hệt nhau**; rerank chỉ đổi thứ tự.

| dataset (test) | M4 R@10 | M4 R@50 | M4 R@100 (trần) | CE-HYRR R@10 |
|---|---:|---:|---:|---:|
| logicbench | 42.8 | 71.7 | 75.7 | 59.2 |
| toolqa | 60.8 | 88.5 | 94.4 | 71.7 |
| medcalcbench | 80.0 | 99.1 | 99.1 | 80.0 |

Macro full-bench đã biết: **M4 R@50 = 91.9 ≈ CE v3 R@50 = 91.4** → giả định bão hòa **đúng**. Reranking không thêm coverage; lợi thế (nếu có) chỉ nằm ở **utility**. ✅ Đi tiếp.

*(Lưu ý: logicbench coverage thấp — R@100 chỉ 75.7% → nhiều query gold nằm ngoài top-50 = vùng "abstain".)*

---

## 3. Kết quả 4 headroom metric

### 3a. Toàn mẫu gộp (n=222) — *có lẫn toolqa degenerate*

| Metric | Giá trị | 95% CI | Cờ |
|---|---:|---|:--:|
| No-load opportunity | **29.3%** | [23.4, 35.6] | 🟢 GO |
| Harmful exposure (CE) | **7.7%** | [4.5, 11.3] | 🟡 MAYBE |
| False-friend rate | **55.0%** | [48.2, 61.7] | 🟢 GO |
| Oracle gap | **11.3pp** | [6.3, 16.7] | 🟢 GO |
| *CE-utility win rate* | *13.5%* | [9.0, 18.0] | (CE top-1 giúp rất ít) |
| *Need-external rate* | *22.5%* | [17.6, 27.9] | |

→ 3 GO ⇒ **GO**. Nhưng `harmful_exposure` chỉ MAYBE vì bị **toolqa kéo xuống** (toolqa = 0%, xem §6).

### 3b. **Subset hợp lệ — logicbench + medcalcbench (n=142)** ⭐ đây là số nên đọc

Sau khi loại toolqa (lý do §6), **cả 4 metric đều xanh đậm**:

| Metric | Giá trị | 95% CI | Cờ |
|---|---:|---|:--:|
| No-load opportunity | **45.8%** | [37.3, 53.5] | 🟢 GO |
| Harmful exposure (CE) | **12.0%** | [7.0, 17.6] | 🟢 GO |
| False-friend (raw) | **59.9%** | [52.1, 68.3] | 🟢 GO |
| False-friend (chặt: gold *có* work nhưng CE chọn vô dụng) | **41.5%** | [33.1, 50.0] | 🟢 GO |
| Oracle gap | **17.6pp** (v_gold 72.5 vs v_ce 54.9) | [9.9, 25.4] | 🟢 GO |
| CE-utility win rate | 21.1% | [14.1, 27.5] | (CE top-1 chỉ giúp 1/5 query) |
| Need-external rate | 35.2% | [28.2, 43.0] | |

**Bằng chứng trực tiếp nhất:** trong các query *need-external* (model sai khi không có skill nhưng gold cứu được, n=50), **CE-HYRR chỉ cứu được 56%** [42, 70] → **CE chọn sai skill ở 44% query mà lẽ ra cứu được**. Đây chính là headroom mà reranker utility-aware nhắm tới.

---

## 4. Theo stratum (gộp n=222) — headroom tăng khi gold càng khó rank

| Stratum (rank gold trong M4) | n | no-load | harmful | false-friend | CE-util win |
|---|---:|---:|---:|---:|---:|
| B_rank1 (gold ở top-1) | 60 | 21.7% | 3.3% | 16.7% | **26.7%** |
| C_rank2_10 | 60 | 25.0% | 10.0% | 55.0% | 11.7% |
| D_rank11_50 | 60 | 40.0% | 11.7% | **75.0%** | 3.3% |
| A_absent (gold ngoài top-50) | 42 | 31.0% | 4.8% | 81.0% | 11.9% |

→ Khi gold dễ (rank1), CE làm tốt (false-friend thấp, win cao). Càng xuống sâu (rank 11-50) CE càng chọn sai → **đúng vùng UCB-ProbeHYRR khai thác**. Stratum A (gold vắng) = vùng *abstain/no-load*, không reranker nào cứu được.

---

## 5. Theo dataset

| dataset | n | no-load | harmful | false-friend | CE-util win | oracle gap | v_no/v_gold/v_ce |
|---|---:|---:|---:|---:|---:|---:|---|
| **logicbench** | 80 | 58.8% | 13.8% | 52.5% | 22.5% | 5.0pp | 0.59 / 0.73 / 0.68 |
| **medcalcbench** | 62 | 29.0% | 9.7% | 69.4% | 19.4% | **33.9pp** | 0.29 / 0.73 / 0.39 |
| toolqa *(loại)* | 80 | 0% | 0% | 46.3%* | 0% | 0pp | **0.00 / 0.00 / 0.00** |

- **logicbench**: headroom đến từ **no-load** (59% query không cần skill!) + harmful exposure (CE phá 14% câu đúng).
- **medcalcbench**: headroom đến từ **oracle gap khổng lồ 33.9pp** — gold skill nâng accuracy từ 39% (CE chọn) lên 73%; CE đang chọn sai skill nghiêm trọng.

---

## 6. ⚠️ Caveat hợp lệ (đọc kỹ trước khi tin con số)

1. **toolqa bị loại — lỗi harness, KHÔNG phải hết headroom.** Cả 80 probe toolqa đều `v_no=v_gold=v_ce=0` (kể cả khi tiêm *gold* skill). Nguyên nhân: toolqa là task **agentic tool-execution** (LoadDB/FilterDB/GetValue…), nhưng tôi probe bằng `DirectEngine` single-shot — model sinh "Action/Observation" nhưng **không có vòng thực thi tool**, nên tự bịa "Observation: ..." và luôn sai. Đây là *mismatch engine*, không phải verifier hỏng cũng không phải Qwen quá yếu. → Muốn chẩn đoán toolqa phải chạy lại bằng engine `react`/`tool_loop` + môi trường ToolQA thật.
2. **bigcodebench bị loại — verifier crash trên macOS.** Sandbox execution dùng `multiprocessing spawn` + closure không pickle được (`untrusted_check.<locals>._isolated_execute`). Cần Linux hoặc context `fork` để chạy.
3. **theoremqa/champ** — ban đầu hoãn vì nghi "verifier yếu", **nhưng kiểm tra lại thì verifier là rule-based numeric/symbolic tất định** (không phải LLM-judge) → đã chạy bổ sung, xem **§9**. Kết luận: cả hai đều GO cả 3 hướng.
4. **Nhãn đặc thù Qwen2.5:7B.** Generator mạnh hơn sẽ nâng `v_gold`/`v_ce`, giảm no-load & false-friend → **headroom phụ thuộc generator**. Tín hiệu *bền* nhất: no-load, oracle-gap, harmful. `false_friend(raw) 60%` lẫn một phần "Qwen yếu"; dùng bản chặt **false_friend(gold-works) = 41.5%** cho an toàn.
5. **Quy mô**: n=142 (subset hợp lệ), **1 seed**, `temperature=0`. CI đã báo cáo, đủ hẹp để quyết định nhưng nên xác nhận lại với generator pipeline thật.

---

## 7. Quyết định Go / No-Go

Theo ngưỡng plan §4 (GO nếu ≥2 metric 🟢):

```
Subset hợp lệ (logicbench+medcalcbench): no-load 🟢 · harmful 🟢 · false-friend 🟢 · oracle-gap 🟢
=> 4/4 GO  =>  QUYẾT ĐỊNH: 🟢 GO
```

**Lợi thế nằm ở đâu (nên hứa gì trong paper):**
- ✅ **No-load / abstain**: ~46% query không cần skill (logicbench 59%) — controller load/no-load có chỗ thắng rõ.
- ✅ **Harmful-exposure reduction**: CE phá 12% câu vốn đúng — risk head có giá trị.
- ✅ **Skill selection / oracle-gap**: gold > CE-pick 17.6pp (medcalc 33.9pp); trong query cứu được, CE bỏ lỡ 44%.
- ✅ **Regime rank ≥ 2**: headroom tăng mạnh khi gold không ở top-1 (false-friend 55-75%).
- ❌ **Đừng hứa** cải thiện trên task agentic (toolqa) hay Recall/nDCG (đã bão hòa).

---

## 8. Việc cần làm trước khi build full pipeline

1. **Chạy lại toolqa bằng engine agentic** (`react`/`tool_loop`) + môi trường ToolQA → mới biết headroom thật ở đó.
2. **Chạy bigcodebench trên Linux** (hoặc sửa sandbox sang `fork`) để có dataset code.
3. **Xác nhận bằng generator của pipeline chính** nếu khác Qwen2.5:7B (nhãn utility không tự chuyển generator).
4. (Tùy) tăng mẫu + nhiều seed cho logicbench/medcalc để siết CI trước khi chốt claim số trong paper.
5. Vì đã **GO**: tiến hành pipeline đầy đủ, nhưng **thu hẹp claim** vào 3 trục: no-load, harmful-exposure, skill-selection/oracle-gap (không hứa mọi regime).

---

### Phụ lục — cách tái lập

```bash
export OLLAMA đang chạy, model qwen2.5:7b đã pull
/Users/hiro/miniconda3/envs/linearag311/bin/python -m experiments.probehyrr_validation.preflight \
  --datasets logicbench toolqa medcalcbench \
  --per-stratum 20 --workers 4 --bootstrap 2000 --seed 13 \
  --model qwen2.5:7b --api-base http://localhost:11434/v1
```
Runner: [`experiments/probehyrr_validation/preflight.py`](../experiments/probehyrr_validation/preflight.py). Probe có cache → chạy lại gần như tức thì.

---

## 9. Mở rộng — CHAMP & TheoremQA: có đủ mạnh để hứa 3 hướng không?

**Câu hỏi:** CHAMP và TheoremQA (chưa chạy ở vòng đầu) có hỗ trợ 3 hướng *(no-load · harmful-exposure · oracle-gap)* mạnh như logicbench/medcalcbench không?

**Đính chính caveat cũ:** verifier của hai bộ này **tất định** (`theoremqa.py`: numeric/symbolic + latex2sympy + tolerance; `champ.py`: exact/numeric/sympy), KHÔNG phải LLM-judge. Đáp án ngắn ("239", "2^(n-1)", "11760", "1.0"). Engine single-shot **đúng** (đây là reasoning/math, không phải agentic như toolqa). → Dùng được.

**Kiểm tra floor (lo Qwen-7B yếu toán → floor như toolqa):** KHÔNG floor.

| dataset | n | v_no | v_gold | v_ce | any_correct |
|---|--:|--:|--:|--:|--:|
| champ | 43 | 0.33 | 0.42 | 0.28 | 0.54 |
| theoremqa | 56 | 0.32 | 0.41 | 0.30 | 0.55 |

`v_gold > v_no` (gold giúp thật), any_correct ~54% — sống khỏe, khác hẳn toolqa (0.00). Audit verifier bằng tay: gán nhãn đúng (vd theoremqa_00088: không skill→75 *sai*, +skill→88.33 *đúng* = ca need-external kinh điển). Nhiễu chính: ~13–15% output bị **cắt ở mốc 512 token** trên bài toán khó → giảm đều v ở mọi điều kiện, không lệch headroom.

### 9.1 — Kết quả 4 metric (95% CI bootstrap)

| Metric | CHAMP (n=43) | TheoremQA (n=56) |
|---|---|---|
| No-load opportunity | **32.6%** [18.6, 46.5] 🟢 | **32.1%** [19.6, 44.6] 🟢 |
| Harmful exposure (CE) | **14.0%** [4.7, 25.6] 🟢 | **16.1%** [7.1, 25.0] 🟢 |
| Oracle gap | **14.0pp** [−2.3, 30.2] 🟢* | **10.7pp** [0.0, 23.2] 🟢* |
| False-friend (raw) | 72.1% [58.1, 83.7] | 51.8% [39.3, 64.3] |
| False-friend (gold-works) | 32.6% [18.6, 46.5] | 23.2% [12.5, 35.7] |
| CE-utility win rate | 9.3% [2.3, 18.6] | 14.3% [5.4, 25.0] |
| CE-recovers \| need-external | 37.5% (miss **62%**), n=8 | 54.5% (miss **45%**), n=11 |

`*` oracle-gap: điểm ước lượng dương, nhưng vì n nhỏ + subset need-external nhỏ nên **CI chạm/qua 0** (champ [−2.3,30.2], theoremqa [0.0,23.2]) → đây là hướng *kém chắc nhất* ở 2 bộ này.

### 9.2 — So sánh trực tiếp 4 dataset single-shot (ngưỡng GO: no-load≥15%, harmful≥8%, oracle-gap≥5pp)

| Hướng | logicbench | medcalc | **champ** | **theoremqa** |
|---|--:|--:|--:|--:|
| No-load / abstain | 58.8% | 29.0% | **32.6%** | **32.1%** |
| Harmful-exposure | 13.8% | 9.7% | **14.0%** | **16.1%** |
| Oracle-gap (skill-sel) | 5.0pp | 33.9pp | **14.0pp** | **10.7pp** |
| → cờ go/no-go | 🟢🟢🟢 | 🟢🟢🟢 | 🟢🟢🟢* | 🟢🟢🟢* |

### 9.3 — Trả lời: **CÓ, đủ mạnh.**

- ✅ **No-load**: champ/theoremqa ~32% — nằm giữa medcalc (29%) và logicbench (59%), CI dương rõ. **Hứa được, ngang ngửa.**
- ✅ **Harmful-exposure**: champ 14% / theoremqa 16% — **cao hơn cả** logicbench (13.8%) và medcalc (9.7%), CI dương rõ. **Đây là bằng chứng harmful-exposure mạnh nhất trong 4 bộ.**
- ⚠️ **Oracle-gap / skill-selection**: điểm ước lượng tốt (champ 14pp, theoremqa 10.7pp, giữa logicbench 5pp và medcalc 34pp), **nhưng CI rộng do n nhỏ** (need-external champ chỉ n=8). → Hướng này *hứa được nhưng nên gom thêm dữ liệu* (nhất là champ) trước khi trích số vào paper.

**Kết luận:** cả CHAMP và TheoremQA đều **GO cả 3 hướng**, ngang logicbench/medcalcbench (và harmful-exposure còn nhỉnh hơn). Caveat: (1) n nhỏ (43/56) → CI oracle-gap rộng, nên tăng mẫu nếu muốn chốt số; (2) ~13–15% output bị cắt ở 512 token — chạy lại với `max_tokens=1024` sẽ siết nhiễu; (3) vẫn đặc thù Qwen2.5:7B, 1 seed.

→ **Bộ datasets để hứa 3 hướng giờ là 4: logicbench, medcalcbench, CHAMP, TheoremQA** (single-shot, verifier tất định). toolqa (agentic) và bigcodebench (sandbox macOS) vẫn ngoài phạm vi vì lý do kỹ thuật ở §6.

*Artifacts: `results/preflight_champ_theoremqa/{headroom_metrics.json, probes.jsonl, run.log}`.*

---

## 10. Re-run `max_tokens=1024` + tăng mẫu — siết CI oracle-gap

Chạy lại champ/theoremqa với `max_tokens=1024` (cache mới để thật sự sinh lại) + lấy **toàn bộ** test set. Mục tiêu: bỏ nhiễu truncation và siết CI oracle-gap (hướng kém chắc nhất ở §9).

### 10.1 — Trước/sau (per-dataset, 95% CI)

| dataset | run | n | any_correct | no-load | harmful | **oracle-gap** | need-ext n |
|---|---|--:|--:|--:|--:|--:|--:|
| champ | 512 | 43 | 0.53 | 32.6% | 14.0% | 14.0pp [−2.3, 30.2] | 8 |
| champ | **1024** | 44 | 0.52 | 31.8% | 13.6% | 13.6pp **[−2.3, 29.5]** | 8 |
| theoremqa | 512 | 56 | 0.55 | 32.1% | 16.1% | 10.7pp [−1.8, 21.4] | 11 |
| theoremqa | **1024** | **149** | **0.62** | 38.9% | 17.4% | **13.4pp [7.4, 20.1] ✓** | **33** |

### 10.2 — Đọc kết quả

- **theoremqa: thành công kép.** `max_tokens=1024` nâng any-correct 0.55→0.62 (xác nhận 512 *có* cắt bớt và làm tụt accuracy), và tăng mẫu 56→149 đưa need-external 11→33 → **CI oracle-gap [7.4, 20.1] giờ đã vượt hẳn 0** (trước là [−1.8, 21.4] chạm 0). → **Oracle-gap của theoremqa giờ vững thật sự.**
- **champ: không đổi & không thể siết.** 512↔1024 gần như y hệt (đáp án champ rất ngắn → vốn không truncate), và test set champ **chỉ có 44 query** (đã dùng hết). CI oracle-gap vẫn [−2.3, 29.5] chạm 0 vì need-external chỉ n=8. → **Đây là giới hạn dữ liệu cứng, không sửa được** — champ chỉ chắc ở no-load + harmful.

### 10.3 — Trả lời: 2 dataset còn lại có cần làm như thế không?

Quyết định bằng số liệu thật, **KHÔNG cần làm đồng loạt**:

| dataset | cần max_tokens=1024? | cần tăng mẫu (oracle-gap)? |
|---|---|---|
| **medcalcbench** | ❌ Không — truncation chỉ **4%**, output ngắn (538 ch). | ❌ Không — oracle-gap **33.9pp CI[21.0, 46.8]** đã rất chắc (need-ext n=30). |
| **logicbench** | ❌ Không — accuracy đã cao (any-correct 86%); 512 đủ. | ⚠️ *Tùy chọn, ROI thấp* — oracle-gap chỉ **5.0pp CI[−3.8, 15.0]** (cross 0), nhưng điểm ước lượng vốn bé nên thêm mẫu (test=152, mới dùng 80) **nhiều khả năng vẫn marginal**. logicbench mạnh ở no-load (59%) + harmful (13.8%), yếu bẩm sinh ở skill-selection. |

**Nguyên tắc rút ra:** `max_tokens=1024` chỉ cần cho dataset **thực sự bị truncate** (chỉ theoremqa). Tăng mẫu chỉ đáng làm khi **(a) còn data** và **(b) gap có thật** — đúng với theoremqa; champ hết data; logicbench gap quá nhỏ; medcalc đã đủ.

### 10.4 — Xếp hạng độ chắc của 3 hướng (sau khi siết)

| Hướng | Độ chắc (4 dataset single-shot) |
|---|---|
| **No-load / abstain** | 🟢🟢🟢🟢 chắc cả 4 (logic 59%, theoremqa 39%, champ 32%, medcalc 29%) |
| **Harmful-exposure** | 🟢🟢🟢🟢 chắc cả 4 (theoremqa 17%, champ 14%, logic 14%, medcalc 10%) — *bằng chứng mạnh nhất* |
| **Oracle-gap / skill-sel** | medcalc 34pp ✓ · theoremqa 13.4pp ✓ · champ 13.6pp (điểm tốt, *kẹt data*) · logicbench 5pp (yếu bẩm sinh) |

→ Hứa **no-load** và **harmful-exposure** thoải mái trên cả 4. Hứa **oracle-gap** dẫn dắt bằng **medcalc + theoremqa** (CI>0), champ là minh họa (caveat n nhỏ), **không over-claim trên logicbench**.

*Artifacts: `results/preflight_champ_theoremqa_mt1024/{headroom_metrics.json, probes.jsonl, run.log}` (theoremqa@1024 là số authoritative; thay cho theoremqa@512 ở §9).*
