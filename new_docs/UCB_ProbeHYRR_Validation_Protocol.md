# UCB-ProbeHYRR — Protocol kiểm chứng DATA trước khi chạy full

> **Câu hỏi cần trả lời:** *Trước khi bỏ chi phí lớn chạy full pipeline UCB-ProbeHYRR, làm sao test data để biết data (và model train từ nó) thực sự lợi thế hơn CrossEncoder HYRR thường?*

Tài liệu này đi kèm bộ script trong [`experiments/probehyrr_validation/`](../experiments/probehyrr_validation/). Kiến trúc gốc: [`UCB_ProbeHYRR_Architecture_v2.md`](UCB_ProbeHYRR_Architecture_v2.md).

---

## 0. Vấn đề phương pháp luận (vì sao không test "ngây thơ" được)

CE-HYRR và UCB-ProbeHYRR gán nhãn cho **hai mục tiêu khác nhau**:

| | CE-HYRR (hiện tại) | UCB-ProbeHYRR (mới) |
|---|---|---|
| Nhãn positive | gold skill (annotation) | skill có `utility = v(q,G(q+s)) − v(q,G(q)) > 0` |
| Nhãn negative | hard negative (BM25/BGE/cluster/random), **loại trừ gold** | harmful / false_friend / bad-candidate |
| Nguồn nhãn | relevance ngữ nghĩa | kết quả verifier downstream |

Hệ quả, hai sai lầm chí mạng khi đánh giá:

1. **Data mới chỉ có giá trị nếu nhãn utility PHÂN KỲ khỏi nhãn relevance.** Nếu "gold ⇔ helpful" gần như luôn đúng → data mới dư thừa, không cần chạy full.
2. **Recall@k / nDCG@k có thể cho FALSE NEGATIVE.** Phương pháp tối ưu *utility*, không phải *relevance*: nó có thể hạ bậc một gold mà generator không dùng được (`v_gold=0`) và nâng một non-gold thực sự giúp → **tụt Recall nhưng tăng task accuracy**. Doc kiến trúc tự dự đoán Recall@50 ~92 (bão hòa). ⇒ **Không bao giờ phán quyết bằng Recall/nDCG.**

Vì vậy protocol là một **cầu thang go/no-go** (rẻ → đắt): mỗi bậc là một điểm có thể *giết* dự án hoặc *bật đèn xanh* cho bậc đắt hơn. Hai bậc rẻ nhất (Gate 1, 2 — ≤5 probe/query trên vài trăm query) trả lời thẳng tiền đề cốt lõi trước khi train bất cứ thứ gì.

---

## 1. Tin tốt: toàn bộ "bộ máy probe" đã có sẵn

Không cần viết lại generator G hay verifier V — chỉ orchestrate code có sẵn:

| Khái niệm trong doc | Code tái dùng | Cách gọi |
|---|---|---|
| Generator `G` | `src/sragents/infer/engines/direct.py` (`DirectEngine`) qua `sragents.llm` | OpenAI-compatible client |
| `v_no = G(q)` | provider `none.py` | probe với `skill_ids=[]` |
| `v_gold = G(q+gold)` | provider `oracle.py` | probe với gold ids |
| `v_i = G(q+s_i)` | provider `topk.py` | probe với `[skill_id]` |
| Verifier `V` | `src/sragents/evaluate/datasets/*` → `{"correct": bool}` | `evaluate(raw_output, instance)` |
| M4 pool | `results/pool/hybrid_km_alpha30-*.json` (có `is_gold`, `rrf_rank`, `bm25_rank`, `bge_rank`, `cluster_id`, `cluster_affinity`) | đã trên đĩa |
| Splits chống leak | `results/splits/*-{query_gen,no_leak,clean}.json` | `common.load_split` |
| Metrics | `src/sragents/retrieve/metrics.py` | đã mở rộng (xem §3) |
| CE trainer | `src/sragents/train/{train_cross_encoder,negative_sampler,losses}.py` | Gate 5 tái dùng nguyên |

**Generator = Qwen2.5:7B local:** trỏ `OPENAI_API_BASE` tới endpoint Ollama/vLLM và truyền `--model qwen2.5:7b`. Nhãn utility là **đặc thù generator** → log `generator_model` mỗi probe, claim phải scoped (pitfall #2).

---

## 2. Cầu thang go/no-go (tóm tắt)

| Gate | Đo gì | Script | Chi phí LLM | Kill rule |
|---|---|---|---|---|
| **0** | Pool coverage & rank headroom (gold-in-top50, phân bố rank gold) | `gate0_pool_audit.py` | **0** | coverage<0.85 **hoặc** gold-top1>0.80 |
| **1** | Utility headroom chỉ bằng mandatory probes (`v_no`,`v_gold`) + human-audit verifier | `gate1_headroom.py` | ~2N | addressable_mass<0.15 **hoặc** gold_hurts>0.25 **hoặc** verifier_agree<0.85 |
| **2** | **Phân kỳ relevance↔utility** (lõi câu hỏi) | `gate2_disagreement.py` | ≤3N | label_disagreement<0.10 **hoặc** divergent_labels<0.08 |
| **3** | Build test set có nhãn utility + metrics mới; sanity oracle-utility | `build_utility_testset.py` | ~12·N_test | oracle gap < +5pp |
| **4** | UCB labels/100-probes vs random/top-rank | `gate4_probe_efficiency.py` | ~3·B·N (cache) | ratio<1.5 → bỏ UCB (không giết project) |
| **6** | Power/sizing (chạy *trước* probe Gate 5): N để phát hiện +2..+5pp | `paired_stats.py` | 0 | N cho +2pp > ngân sách full-run |
| **5** | **A/B pilot thật** CE-ProbeHYRR vs CE-HYRR budget-matched | `gate5_ab_pilot.py` | pilot probe (cache) + ~6 fine-tune | Utility@1 Δ≤0 **hoặc** HarmfulExposure xấu đi (CI loại 0) |

Mỗi gate trên chỉ chạy nếu các gate dưới đã pass ⇒ cam kết đắt nhất (Gate 5 + full run) được bảo vệ bởi 4 lớp kill rẻ hơn.

---

## 3. Metrics utility-aware (đã thêm vào `metrics.py`)

`compute_utility_metrics(results, utility_labels, ks)` và bản `_by_dataset` (dùng chung plumbing macro/micro). `utility_labels` là test set Gate 3: `instance_id → {v_no, v_gold, skills:{skill_id:{v,u,label}}}`.

- **Utility@1** = mean `1[u(top1_ranked) > 0]` — skill #1 reranker chọn có *thật sự giúp* không. ↑ tốt.
- **FalseFriend@10** = tỉ lệ top-10 (đã probe) có `u ≤ 0`. ↓ tốt.
- **HarmfulExposure** = P(top1 phá câu đúng | `v_no=1`). ↓ tốt.
- **CS-Gold@K** = P(top1 có `u>0` | gold ∈ top-K). ↑ tốt — "incorporation, không chỉ retrieval".
- Mỗi metric kèm `/coverage` (tỉ lệ ô có ground-truth) để biết support.

Recall/nDCG (`compute_retrieval_metrics`) và end-task (`infer --provider topk k=1` + `evaluate`) in **kèm bên cạnh** để minh họa điểm "false negative" — không dùng phán quyết.

---

## 4. Chi tiết từng Gate

### Gate 0 — Pool coverage & rank headroom (0 LLM)
- **Đo:** với mỗi dataset, gold-in-top-50 coverage và phân bố rank gold (top1 / 2-10 / 11-50 / absent). Coverage chính là trần của mọi reranker trên pool này.
- **Kill:** macro coverage<0.85 (pool quá yếu) **hoặc** gold-đã-top1>0.80 (không còn gì để rerank).
- **Chạy:** `python -m experiments.probehyrr_validation.gate0_pool_audit` (chạy thật được ngay, 0 call). Đối chiếu M4 Recall@50≈91.9 đã biết.

### Gate 1 — Utility headroom (mandatory probes)
- **Đo (sample ~80/dataset từ TEST split, stratified theo rank gold):** `need_external = P(v_no=0 & v_gold=1)`, `no_load = P(v_no=1)`, `gold_helps`, `gold_hurts`. `addressable_mass = need_external + no_load` = vùng mà utility khác "luôn load top-1".
- **Verifier audit:** xuất `gate1_audit_sample.csv` (100-200 mẫu) để con người chấm; báo agreement per-dataset. Quyết định chỉ trên 4 dataset strong-verifier.
- **Kill:** addressable_mass<0.15 (relevance≈utility) / gold_hurts>0.25 (generator không dùng được skill → nhãn nhiễu) / verifier_agree<0.85.
- **Chạy (cần endpoint):** `... gate1_headroom --model qwen2.5:7b --api-base <url> --run`.

### Gate 2 — Phân kỳ relevance↔utility (LÕI)
- **Đo:** probe M4 top-1/3/5 non-gold (dùng lại `v_no/v_gold` cache), `u=v_i−v_no`, gán nhãn `derive_label`. Tính `false_friend_rate`, `helpful_non_gold` (CE-HYRR sẽ gán nhầm thành NEGATIVE!), `harmful_rate`, và `label_disagreement = P((s∈gold) ≠ (u>0))`.
- **Kill:** label_disagreement<0.10 (≈trùng → data dư thừa) / (helpful_non_gold+false_friend)<0.08.
- **Phụ:** dump raw label counts để check imbalance (pitfall #5).

### Gate 3 — Test set có nhãn utility (chống false-negative)
- **Build:** trên TEST split (disjoint probe-train), probe **cố định M4 top-10** (policy-independent, KHÔNG UCB) + mandatory. Lưu `results/probehyrr/utility_testset.json` đúng contract của `compute_utility_metrics`.
- **Sanity STOP:** oracle-utility ranker phải > M4 ≥ +5pp macro Utility@1, nếu không thì không có signal để học.

### Gate 4 — UCB efficiency
- **Đo (trên probe-train, disjoint test):** useful-labels/100-probes của random (P0) vs top-rank (P1) vs bucket-UCB (P4), cùng budget B. Tái dùng cache.
- **Kill mềm:** ratio = UCB/max(random,top-rank). ≥1.5 → dùng UCB; 1.0-1.5 → dùng top-rank/round-robin (bỏ UCB, **không** giết project); <1.0 → UCB sai, debug.

### Gate 6 — Power / sizing (trước khi probe Gate 5)
- Dùng `addressable_mass` (Gate 1) + `discordant_rate` (≈ Gate 2 disagreement) để tính N cần thiết: `N_eval ≈ (z_{α/2}+z_β)²·p_d/δ²`; `N_raw = N_eval / addressable_mass`. Báo **MDE** cho N thực chạy (để null result đọc được).
- **Mốc tham khảo:** phát hiện +3pp (p_d≈0.15) cần ~1300 evaluable; +5pp cần ~470. Với addressable_mass~30% → probe ~1500 test + ~2000 train query (vẫn « full 5400×5 probe).
- **Kill:** nếu N để phát hiện +2pp vượt ngân sách full-run thì pilot không de-risk được mức khiêm tốn — chỉ commit nếu mục tiêu là mức lạc quan +5pp.

### Gate 5 — A/B pilot (quyết định)
- **2 arm, mọi thứ giữ nguyên, chỉ khác NHÃN:** cùng M4 pool / backbone MiniLM-L6 / splits / leakage filter / **matched số pair**. Arm B mine bằng `NegativeSampler` trên *cùng* query, subsample khớp N. Single-head BCE để cô lập tác động nhãn (multi-head để ablation sau).
- **2 so sánh:** *matched-query* (chỉ query có gold) và *superset* (Arm A thêm query no-load).
- **Đánh giá** trên test set Gate 3: cả 4 nhóm metric + end-task. **Thống kê paired:** McNemar + bootstrap CI, ≥3 seeds.
- **GO full-run iff:** macro Utility@1 Δ≥+2pp (CI loại 0) **và** McNemar p<0.05 end-task **và** HarmfulExposure không tăng **và** Recall@10 tụt ≤1pp.

---

## 5. Pitfalls (kiểm soát ở mọi gate)

1. **Verifier noise** → go/no-go chỉ trên 4 dataset strong-verifier (toolqa/logicbench/medcalcbench/bigcodebench); theoremqa/champ weak/audited. Lan truyền noise rate vào diễn giải CI Gate 5.
2. **Phụ thuộc Qwen2.5:7B** → log `generator_model`; claim scoped; tùy chọn đo transfer sang generator thứ 2.
3. **Gold vắng top-50** → loại khỏi CS-Gold và khỏi train Arm B; gắn `abstain`.
4. **Leak probe-train↔test** → disjoint `instance_id` qua splits; test set probe **cố định top-10**, không UCB.
5. **Label imbalance** → Gate 2 báo raw count; nếu harmful/false_friend hiếm → class-weight + báo per-class.
6. **Retrieval false-negative** → không phán quyết bằng Recall/nDCG; in kèm để minh họa phân kỳ.

---

## 6. Trình tự chạy (khi sẵn sàng — hiện tại mới là DESIGN)

```bash
# Gate 0 (free, chạy ngay)
python -m experiments.probehyrr_validation.gate0_pool_audit

# Gate 1 → 2 (cần Qwen2.5:7B local; ví dụ Ollama)
export OPENAI_API_BASE=http://localhost:11434/v1
python -m experiments.probehyrr_validation.gate1_headroom   --model qwen2.5:7b --run
python -m experiments.probehyrr_validation.gate2_disagreement --model qwen2.5:7b --run

# Gate 3 (test set) + Gate 4 (UCB) + Gate 6 (sizing) → Gate 5 (A/B)
python -m experiments.probehyrr_validation.build_utility_testset --model qwen2.5:7b --run
python -m experiments.probehyrr_validation.gate4_probe_efficiency --model qwen2.5:7b --run
python -m experiments.probehyrr_validation.gate5_ab_pilot --run
```

Tất cả gate (trừ Gate 0) mặc định **design-only**: không có `--run` thì chỉ in kế hoạch + decision rule, không gọi LLM.
