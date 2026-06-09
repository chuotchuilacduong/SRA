# UCB-ProbeHYRR: Thiết kế phương pháp mới cho Skill Retrieval Augmentation

**Mục tiêu paper:** thiết kế một hướng đủ khác biệt so với HYRR/semantic reranking thông thường bằng cách biến skill retrieval thành **counterfactual utility optimization**: mỗi skill không chỉ được đánh giá bởi semantic relevance, mà bởi **tác động nhân quả của nó lên độ đúng của output cuối**, so với `no_skill` và `oracle/gold_skill`.

**Tên đề xuất:** `UCB-ProbeHYRR` hoặc `Bandit-Guided Counterfactual Skill Utility Reranking`.

**One-line thesis:**

> Retrieval tốt chưa đủ cho SRA. Sau khi Method 7 đã lấy được top-K skills, hệ thống phải học thêm: task có cần external skill không, skill nào thật sự giúp output đúng, skill nào gây hại, và khi nào nên abstain/no-load. Chúng tôi dùng ProbeLLM-style verified failure probing + UCB budget allocation để tạo labels utility/risk/abstention, rồi fine-tune HYRR thành utility-aware reranker/controller.

---

## 1. Bối cảnh từ 2 markdown hiện có

### 1.1 Method 7 hiện tại đã mạnh về retrieval

Pipeline hiện tại:

```text
Stage 1:
  BM25 + BGE → RRF → KMeans α-blend

Stage 2:
  Cross-Encoder MiniLM-L6 rerank top-100

Post-hoc fusion:
  score* = 0.7 · CE_norm + 0.3 · Stage1_norm
```

Kết quả full SRA-Bench:

```text
Macro Recall@10 = 86.06
Macro nDCG@10  = 74.67
```

So với CE-only, β=0.7 fusion tăng:

```text
+3.42 pp Recall@10
+4.93 pp nDCG@10
```

**Interpretation:** Method 7 đã là một retrieval baseline rất mạnh. Nếu paper mới chỉ nói “thêm hard negatives cho CE để tăng R@10”, novelty sẽ yếu. Contribution nên chuyển sang **post-topK utility / incorporation**.

### 1.2 HYRR data hiện tại vẫn là semantic/hard-negative training

CE hiện tại dùng training group:

```text
1 positive
3 BM25 hard negatives
3 BGE hard negatives
2 random negatives
2 cluster hard negatives
```

Loss hiện tại:

```text
listwise softmax:
  gold skill phải có logit cao nhất trong group 11 candidates
```

Các negative này tốt cho semantic/ranking robustness, nhưng chúng chưa trả lời:

```text
Skill này có làm final answer đúng hơn không?
Skill này có gây nhiễu không?
Task này có cần external skill không?
Nếu gold skill không có trong top-K thì có nên load skill nào không?
```

---

## 2. Tại sao cần đổi objective?

### 2.1 Vấn đề cũ

HYRR-v0/Method7 học:

```text
relevance(q, skill)
```

Nhưng SRA cần:

```text
utility(q, skill, generator)
risk(q, skill, generator)
need_external(q)
gold_coverage(q, topK)
load_or_abstain(q, topK)
```

Semantic relevance và downstream utility không đồng nhất.

Ví dụ:

```text
Query q cần skill A.

Skill B có cùng topic, cùng keyword, score CE cao.
Nhưng khi đưa skill B vào LLM, LLM trả lời sai.

→ B là semantic hard negative.
→ Quan trọng hơn: B là verified false-friend / harmful negative.
```

### 2.2 Paper mới nên claim gì?

Claim yếu:

```text
We improve HYRR by adding ProbeLLM-generated negatives.
```

Claim này dễ bị reviewer xem là incremental negative mining.

Claim mạnh hơn:

```text
We formulate skill retrieval as counterfactual utility optimization:
each candidate skill is judged by its causal effect on downstream task correctness
relative to no-skill and oracle-skill executions.

We use verified failure modes to construct utility, risk, and abstention labels,
enabling a reranker/controller to decide not only which skill is relevant,
but whether any skill should be loaded.
```

---

## 3. Định vị novelty so với SOTA

### 3.1 So với HYRR

**HYRR** train reranker bằng hybrid retriever outputs như BM25 + neural retrieval để tạo robust reranker cho passage retrieval.

**UCB-ProbeHYRR khác ở:**

| HYRR | UCB-ProbeHYRR |
|---|---|
| Negative = hard vì được BM25/dense retrieve cao | Negative = hard vì được retrieve cao **và verifier chứng minh không giúp/gây hại** |
| Objective = relevance ranking | Objective = downstream utility/risk/abstention |
| Passage/document retrieval | Skill/capability retrieval |
| Không mô hình hóa no-load | Có `need_external` và `abstain/no-load` |

### 3.2 So với ProbeLLM

**ProbeLLM** dùng automated probing để khám phá failure modes của LLM, với hierarchical MCTS, verifiable test cases, failure-aware embeddings, boundary-aware induction.

**UCB-ProbeHYRR khác ở:**

| ProbeLLM | UCB-ProbeHYRR |
|---|---|
| Mục tiêu = diagnose failure modes | Mục tiêu = biến failure modes thành supervision cho reranker/controller |
| Output = failure clusters | Output = utility/risk/abstention labels + trained HYRR-v1 |
| MCTS-based probing | UCB/contextual-bandit probing đơn giản hơn, budget-aware |
| LLM failure diagnosis chung | Skill retrieval augmentation cụ thể |

### 3.3 So với Self-RAG / CRAG / Adaptive-RAG

Các hướng này chủ yếu xử lý RAG document retrieval:

| Method | Ý chính | Khác biệt của UCB-ProbeHYRR |
|---|---|---|
| Self-RAG | LLM học retrieve/generate/critique bằng reflection tokens | UCB-ProbeHYRR không train generator bằng reflection; nó train skill reranker/controller bằng verified utility labels |
| CRAG | Retrieval evaluator đánh giá chất lượng retrieved documents và trigger corrective actions | UCB-ProbeHYRR đánh giá skill utility/risk qua counterfactual execution, không chỉ document quality |
| Adaptive-RAG | Classifier chọn no-retrieval/single/multi-step retrieval dựa trên complexity | UCB-ProbeHYRR chọn load/no-load dựa trên verified no-skill vs skill outcomes |

### 3.4 So với utility-aware RAG

Các paper như SCARLet, LURE-RAG, Predicting Retrieval Utility đều chỉ ra rằng relevance không đủ, retrieved context cần được đánh giá theo utility cho generation.

**Khoảng trống còn lại:**

```text
Existing utility-aware RAG:
  document/passage utility cho factual RAG.

UCB-ProbeHYRR:
  skill/capability utility cho SRA,
  với no-skill / candidate-skill / oracle-skill counterfactuals,
  plus risk and abstention labels.
```

### 3.5 So với DeepSeek-R1 / Search-R1 / GRPO-style reasoning

DeepSeek-R1 dùng RL/GRPO với rule-based rewards để kích thích reasoning behavior. Search-R1/ReSearch mở rộng RL cho multi-turn search/retrieval reasoning.

**UCB-ProbeHYRR không nên claim là GRPO.**

Khác biệt:

```text
DeepSeek-R1/Search-R1:
  train LLM policy sinh reasoning/search trajectory.

UCB-ProbeHYRR:
  không train generator reasoning policy.
  train reranker/controller để chọn skill/no-skill dựa trên verified counterfactual labels.
```

Điểm tương đồng:

```text
Cả hai dùng verifiable reward/outcome.
```

Điểm khác chính:

```text
R1-style RL tối ưu policy generation.
UCB-ProbeHYRR tối ưu data acquisition + reranking/control.
```

---

## 4. Core idea: biến Stage 2 thành two-pass learning

### 4.1 Không thay Method7 ngay lập tức

Method7 hiện tại là `HYRR-v0`.

```text
HYRR-v0:
  relevance-oriented CE + β-blend reranker
```

Ta dùng HYRR-v0 để tạo candidate pools.

Sau đó:

```text
HYRR-v0 top-K
→ UCB-guided counterfactual probing
→ verified labels
→ train HYRR-v1 + controller
```

### 4.2 Hai vòng training

```text
Round 0:
  Train CE/HYRR-v0 bằng semantic/hybrid negatives hiện tại.

Round 1:
  Freeze v0.
  Dùng v0 lấy top-K trên train/dev queries.
  Dùng UCB để chọn probe nào đáng chạy.
  Chạy LLM + verifier.
  Tạo labels utility/risk/abstention.
  Fine-tune thành HYRR-v1 và train controller.

Inference:
  Chỉ dùng HYRR-v1 + controller.
  Không dùng UCB online.
```

**UCB là offline label acquisition module**, không phải online reranker.

---

## 5. Formalization

### 5.1 Ký hiệu

```text
q: query
s: skill candidate
S_K(q): top-K skills từ Method7/HYRR-v0
G(q): generator LLM
V(q, y): verifier, trả về correctness/score
s_gold: gold skill nếu có
```

### 5.2 Counterfactual outcomes

```text
y_0      = G(q)                         # no skill
y_s      = G(q, s)                      # candidate skill
y_gold   = G(q, s_gold)                 # oracle/gold skill, nếu có
```

Verifier:

```text
v_0      = V(q, y_0)
v_s      = V(q, y_s)
v_gold   = V(q, y_gold)
```

Binary version:

```text
v ∈ {0, 1}
```

Score version:

```text
v ∈ [0, 1]
```

### 5.3 Utility label

```text
utility(q, s) = v_s - v_0
```

Interpretation:

```text
utility > 0:
  skill helps

utility = 0:
  skill neutral

utility < 0:
  skill harms
```

### 5.4 Need label

```text
need_external(q) = 1
  if v_0 = 0 and v_gold = 1

need_external(q) = 0
  if v_0 = 1 and no candidate skill improves output
```

Ambiguous cases:

```text
v_0 = 0 and v_gold = 0:
  task may be unsolved even with gold skill
  mark as ambiguous / exclude from need training

v_0 = 1 and v_gold = 1:
  skill may be unnecessary
  mark as no-load unless skill improves robustness/format
```

### 5.5 Risk label

```text
risk(q, s) = 1
  if v_0 = 1 and v_s = 0
```

### 5.6 False-friend label

```text
false_friend(q, s) = 1
  if rank_v0(s) is high and utility(q, s) <= 0
```

For example:

```text
rank ≤ 10
β_score high
CE_score high
but v_s ≤ v_0
```

### 5.7 Abstention / no-load label

```text
abstain(q, S_K) = 1
  if gold skill absent from top-K
  and all probed candidate skills have utility ≤ 0

no_load(q) = 1
  if v_0 = 1
  and skill loading does not improve or harms
```

---

## 6. UCB-guided probing

### 6.1 Vì sao cần UCB?

Nếu K=100, probing toàn bộ là quá tốn:

```text
1 no_skill call
1 gold_skill call
100 candidate skill calls
```

Với 5,400 queries, chi phí sẽ rất lớn.

UCB giúp trả lời:

```text
Trong top-K/failure buckets, probe cái nào trước để tạo nhiều label hữu ích nhất?
```

### 6.2 Arm không nên là skill global

Không nên dùng mỗi skill corpus làm một arm, vì utility phụ thuộc query.

Nên dùng **probe strategy / failure-mode bucket** làm arm.

Arms đề xuất:

```text
A1: top_beta
  probe skill có β-score cao nhất

A2: ce_s1_conflict
  probe skill CE cao nhưng Stage1 thấp, hoặc ngược lại

A3: false_friend_suspect
  probe skill rank cao nhưng khác gold cluster / có high semantic overlap với nhiều distractors

A4: boundary_near_gold
  probe skill gần gold skill trong embedding/cluster nhưng không phải gold

A5: need_confusion
  probe skill cho query mà no_skill có khả năng đã đủ

A6: gold_absent_suspect
  probe top candidates khi gold không nằm trong top-K hoặc confidence thấp

A7: failure_cluster_specific
  probe từ failure cluster đã được ProbeLLM-style induction phát hiện
```

### 6.3 UCB score

Với mỗi arm `a`:

```text
N_a: số lần arm a đã được probe
R_a: tổng reward hữu ích
mean_a = R_a / N_a
```

UCB index:

```text
UCB(a) = mean_a + c * sqrt(log(t) / N_a)
```

Trong đó:

```text
t = tổng số probes đã chạy
c = exploration coefficient
```

Nếu dùng công thức UCB1 chuẩn:

```text
UCB(a) = mean_a + sqrt(2 log(t) / N_a)
```

### 6.4 Reward cho một probe

Reward không phải correctness trực tiếp. Reward là **label usefulness**.

Một probe có reward cao nếu tạo ra:

```text
helpful positive
harmful negative
false-friend negative
need/no-load label
abstention label
boundary negative
```

Reward đơn giản:

```text
reward = 1 nếu label ∈ {helpful, harmful, false_friend, abstain, boundary}
reward = 0 nếu neutral/uninformative
```

Reward tốt hơn:

```text
reward =
  1.0  nếu harmful hoặc false_friend
  0.8  nếu helpful non-gold skill
  0.7  nếu abstention/no-load
  0.5  nếu boundary negative
  0.0  nếu neutral
```

Reward research-quality:

```text
reward = label_value + validation_gain_estimate
```

Nhưng MVP nên dùng binary reward trước.

### 6.5 Contextual UCB extension

Vanilla UCB giả định mỗi arm có reward distribution cố định. Nhưng trong SRA, reward phụ thuộc query, dataset, skill type, score pattern.

Vì vậy, bản mạnh hơn nên là **contextual UCB**.

Context vector:

```text
x(q, s, topK) = [
  dataset_id,
  β_score,
  CE_score,
  Stage1_score,
  rank,
  CE-S1 gap,
  entropy(topK scores),
  score_margin(top1-top2),
  cluster distance to gold or top1,
  query length,
  skill length,
  skill type/domain
]
```

Action:

```text
choose probe arm / candidate skill
```

Reward:

```text
useful_label indicator
```

MVP:

```text
bucket-level UCB
```

Advanced:

```text
LinUCB / contextual UCB
```

---

## 7. Full algorithm

### 7.1 Offline data acquisition

```python
# Inputs:
# Q_train: train queries
# Retriever_v0: Method7/HYRR-v0
# Generator G
# Verifier V
# Budget B candidates per query

for q in Q_train:
    topK = Retriever_v0(q, K=100)

    # Mandatory probes
    y_no = G(q, skills=None)
    v_no = V(q, y_no)

    if gold_skill_available(q):
        y_gold = G(q, skill=gold_skill(q))
        v_gold = V(q, y_gold)
    else:
        v_gold = None

    # Build probe candidate buckets
    candidate_actions = build_probe_actions(q, topK, v_no, v_gold)

    for b in range(B):
        action = UCB.select(candidate_actions)
        s = action.skill

        y_s = G(q, skill=s)
        v_s = V(q, y_s)

        label = derive_label(
            q=q,
            skill=s,
            v_no=v_no,
            v_s=v_s,
            v_gold=v_gold,
            rank=s.rank,
            scores=s.scores
        )

        reward = label_usefulness(label)
        UCB.update(action.arm, reward)

        save_probe(q, s, y_s, v_s, label)
```

### 7.2 Label derivation

```python
def derive_label(q, skill, v_no, v_s, v_gold, rank, scores):
    utility = v_s - v_no

    if utility > 0:
        return "helpful"

    if v_no == 1 and v_s == 0:
        return "harmful"

    if rank <= 10 and utility <= 0 and scores.beta_high:
        return "false_friend"

    if v_gold is not None and v_no == 0 and v_gold == 1 and utility <= 0:
        return "bad_candidate_when_skill_needed"

    if v_no == 1 and utility <= 0:
        return "no_load_preferred"

    return "neutral"
```

### 7.3 Training HYRR-v1

Create groups:

```text
positive:
  gold skill if verified useful
  or candidate skill with utility > 0

negative:
  false_friend
  harmful
  boundary_negative
  no_load_competing_skill
```

Pairwise ranking loss:

```text
L_rank = max(0, m - score(q, s_pos) + score(q, s_neg))
```

Listwise loss:

```text
target distribution over group:
  helpful skills get high probability
  neutral low
  harmful/false-friend near zero
```

Multi-task heads:

```text
CE encoder → shared CLS vector

Heads:
  relevance_head(q, s)
  utility_head(q, s)
  risk_head(q, s)
  false_friend_head(q, s)
```

Loss:

```text
L = L_rank
  + λ_u * BCE(utility_helpful)
  + λ_r * BCE(risk)
  + λ_f * BCE(false_friend)
```

Final skill score:

```text
score_v1(q, s)
  = relevance_score(q, s)
  + α * utility_score(q, s)
  - γ * risk_score(q, s)
  - ρ * false_friend_score(q, s)
  + δ * β_score_v0(q, s)
```

### 7.4 Training controller

The controller is list-level, not pair-level.

Input:

```text
q
topK skill metadata
topK scores from HYRR-v1
score entropy
score margins
max utility
max risk
gold coverage features if training
```

Outputs:

```text
P_need_external(q, topK)
P_gold_covered(q, topK)
P_load(q, topK)
```

Loss:

```text
L_controller =
  BCE(need_external)
  + BCE(gold_covered)
  + BCE(load_or_abstain)
```

Online policy:

```python
topK = HYRR_v1(q)

p_need = controller.need(q, topK)
p_gold = controller.gold_covered(q, topK)

if p_need < tau_need:
    return G(q, skills=None)

if p_gold < tau_gold:
    return fallback_or_abstain(q)

selected = argmax_s [
    utility(q, s)
    - gamma * risk(q, s)
    + eta * relevance(q, s)
]

return G(q, skill=selected)
```

---

## 8. Failure-mode taxonomy

Initial taxonomy:

| Failure mode | Condition | Use for training |
|---|---|---|
| `retrieval_miss` | gold not in top-K | Stage1/Stage2 recall issue |
| `ranking_miss` | gold in top-K but low rank | reranker training |
| `selection_miss` | gold in top-K but selected skill wrong | utility/risk controller |
| `application_miss` | gold selected but answer wrong | generator/prompt issue |
| `false_loading` | no_skill correct, skill causes wrong | risk/no-load training |
| `need_miss` | no_skill wrong, gold correct, but no skill loaded | need training |
| `gold_absent_overconfidence` | gold absent but top score high | abstention/fallback training |
| `boundary_confusion` | similar skill differs in key condition/procedure | boundary negatives |

ProbeLLM-style clustering:

```text
1. Create failure-aware embedding:
   [query, topK summary, selected skill, model output, verifier result, label]

2. Cluster failures:
   HDBSCAN or KMeans.

3. For each cluster:
   central failures = representative cases
   boundary failures = close to non-failures
   nearby non-failures = contrastive examples

4. Summarize cluster with LLM:
   "This failure happens when..."
```

These cluster descriptions are **not labels by themselves**. They are used to build better arms/buckets for UCB and to explain errors in the paper.

---

## 9. Where exactly UCB fits

```text
Training time only:

HYRR-v0 / Method7
  → top-K candidates
  → UCB-Probe Scheduler
      chooses which skill/bucket to probe
  → LLM generation
  → verifier
  → utility/risk/abstention labels
  → HYRR-v1 + controller training

Inference time:

HYRR-v1
  → controller
  → selected skill or no-load
  → final answer
```

Do **not** use UCB online unless you want an expensive adaptive agent.

---

## 10. Baselines

### 10.1 Retrieval/reranking baselines

```text
B0: BM25
B1: BGE
B2: RRF
B3: RRF + KMeans α=0.7
B4: CE-v0
B5: Method7 β=0.7
B6: R-RRF
B7: HYRR-v1 without UCB labels
B8: HYRR-v1 with random probed labels
B9: HYRR-v1 with top-score probed labels
B10: HYRR-v1 with UCB-Probe labels
```

### 10.2 Skill incorporation baselines

```text
I0: no_skill
I1: gold_skill oracle
I2: Method7 top1 full skill
I3: Method7 topK full injection
I4: Method7 + LLM metadata selection
I5: Method7 + utility head only
I6: Method7 + need controller only
I7: Method7 + utility/risk controller
I8: HYRR-v1 + controller
```

### 10.3 Active probing baselines

```text
P0: probe random candidate
P1: probe top β-score candidates
P2: probe top uncertainty candidates
P3: round-robin failure buckets
P4: UCB bucket-level
P5: contextual UCB
```

---

## 11. Metrics

### 11.1 Retrieval metrics

```text
Recall@1 / @5 / @10
MRR
nDCG@10
```

### 11.2 Utility-aware metrics

```text
Utility@1:
  selected top skill has utility > 0

FalseFriend@10:
  percentage of top-10 skills with utility <= 0 but high relevance score

HarmfulExposure@K:
  percentage of exposed skills that make no_skill-correct cases wrong

Risk@1:
  probability top selected skill is harmful

NoLoad Accuracy:
  whether controller correctly chooses no skill when skill unnecessary
```

### 11.3 End-task metrics

```text
Final accuracy / EM / pass@1
CS-Gold@K:
  P(final answer correct | gold skill ∈ topK)

Oracle Gap:
  GoldSkillOracle - Method

Skill-Free Correct Preservation:
  P(final correct | no_skill correct)
```

### 11.4 Probing efficiency metrics

```text
Useful labels per 100 probes
Harmful negatives found per 100 probes
False-friend negatives found per 100 probes
Cost per useful label
Final accuracy per 1K LLM probe calls
```

---

## 12. Success criteria

Minimum success:

```text
1. UCB-Probe finds more useful labels per 100 probes than random/top-score probing.
2. HYRR-v1 trained with UCB labels improves Utility@1 over Method7.
3. HarmfulExposure decreases versus top1/topK full injection.
4. CS-Gold@10 improves:
   if gold skill is already in top-10, final answer correctness should increase.
5. Final task accuracy improves under same inference budget.
```

Strong paper-level success:

```text
+2–5 pp final task accuracy over Method7 + top1 injection.
+5–10 pp CS-Gold@10.
≥30% reduction in HarmfulExposure.
Useful-label yield > random probing by ≥1.5×.
No significant drop in Recall@10/nDCG@10.
```

---

## 13. Ablation plan

### 13.1 Model ablations

```text
A0: Method7 only
A1: Method7 + controller trained without counterfactual labels
A2: Method7 + utility head
A3: Method7 + utility + risk heads
A4: Method7 + utility + risk + no-load controller
A5: HYRR-v1 trained with failure-mode negatives
A6: HYRR-v1 + full controller
```

### 13.2 Label ablations

```text
L0: semantic HYRR negatives only
L1: + helpful positives
L2: + false-friend negatives
L3: + harmful negatives
L4: + abstention/no-load labels
L5: + boundary negatives
L6: all labels
```

### 13.3 UCB ablations

```text
U0: no UCB, random probe
U1: top β-score probe
U2: top uncertainty probe
U3: round-robin buckets
U4: vanilla UCB
U5: contextual UCB
```

### 13.4 Probe budget ablations

```text
B = 1, 2, 3, 5, 10 candidate probes per query
```

Report:

```text
label yield vs budget
final accuracy vs budget
HYRR-v1 Utility@1 vs budget
```

---

## 14. Expected trade-offs

### 14.1 Cost

Counterfactual probing costs extra LLM calls.

Mitigation:

```text
Use UCB to probe only high-yield candidates.
Cache all generations.
Start with top-5 not top-100.
Use smaller generator for probing if verifier is robust.
```

### 14.2 Verifier quality

If verifier is noisy, utility labels are noisy.

Mitigation:

```text
Prefer execution/rule-based verifiers.
Use LLM judge only for low-stakes auxiliary labels.
Human audit 100–200 samples.
Keep ambiguous labels separate.
```

### 14.3 Generator dependence

Utility is generator-specific.

```text
A skill helpful for GPT-4.1 may be neutral for Qwen/Llama.
```

Mitigation:

```text
Log generator_model_id.
Train generator-specific utility heads.
Evaluate cross-generator transfer explicitly.
```

### 14.4 Pairwise vs list-level mismatch

Pairwise CE cannot know if gold is absent from top-K.

Mitigation:

```text
Use pairwise HYRR-v1 for skill scoring.
Use list-level controller for need/gold_coverage/load-or-abstain.
```

### 14.5 Novelty risk

UCB itself is not novel. Hard-negative mining itself is not novel.

The novel part must be:

```text
budgeted counterfactual utility learning for skill retrieval augmentation
```

not merely:

```text
UCB + hard negatives
```

---

## 15. MVP implementation plan

### Phase 1 — Freeze Method7

Input:

```text
Method7 top-100 outputs with:
  skill_id
  β_score
  ce_score
  stage1_score
  rank
```

### Phase 2 — Mandatory probes

For each training query:

```text
Run no_skill.
Run gold_skill if gold exists.
Verify both.
```

Create:

```text
need_external preliminary label.
```

### Phase 3 — UCB candidate probing

Budget:

```text
B = 3 candidate probes/query
```

Arms:

```text
top_beta
ce_s1_conflict
boundary_near_gold
false_friend_suspect
need_confusion
```

Output:

```text
probe_logs.jsonl
```

Example:

```json
{
  "qid": "logicbench_00042",
  "skill_id": "skill_123",
  "arm": "boundary_near_gold",
  "v_no": 0,
  "v_skill": 0,
  "v_gold": 1,
  "label": "false_friend",
  "utility": 0,
  "reward": 1,
  "scores": {
    "beta": 0.91,
    "ce_norm": 0.88,
    "stage1_norm": 0.77
  }
}
```

### Phase 4 — Train controller first

Before retraining HYRR, train simple controller:

```text
Input:
  topK metadata + scores

Output:
  load/no-load
  selected skill
```

This tests the thesis cheaply.

### Phase 5 — Fine-tune HYRR-v1

Add utility/risk heads.

Training data:

```text
original HYRR groups
+ UCB-generated helpful/harmful/false-friend examples
```

Loss:

```text
listwise relevance loss
+ utility classification
+ risk classification
+ pairwise utility margin
```

### Phase 6 — Evaluate end-task, not only retrieval

Compare:

```text
Method7 top1 injection
Method7 topK injection
Method7 + controller
HYRR-v1 + controller
Oracle gold skill
No skill
```

---

## 16. Recommended paper framing

### Title candidates

```text
1. UCB-ProbeHYRR: Bandit-Guided Counterfactual Utility Learning for Skill Retrieval
2. From Relevance to Utility: Failure-Guided Skill Reranking for Agentic Retrieval
3. When Not to Retrieve Skills: Counterfactual Utility Optimization for Skill Retrieval Augmentation
```

### Abstract skeleton

```text
Skill Retrieval Augmentation enables agents to retrieve reusable external capabilities,
but high retrieval recall does not guarantee correct skill use. We show that a strong
hybrid cross-encoder reranker can retrieve gold skills in top-K while still failing
downstream because agents load unnecessary, harmful, or semantically plausible but
non-useful skills. We propose UCB-ProbeHYRR, a budgeted counterfactual probing framework
that uses no-skill, candidate-skill, and oracle-skill executions with verifiers to
derive utility, risk, and abstention labels. A UCB scheduler allocates limited probing
budget toward high-yield failure-mode regions. These labels fine-tune a HYRR-style
cross-encoder and train a list-level load/no-load controller. Experiments on SRA-Bench
show improved conditional success given gold@K, reduced harmful skill exposure, and
higher final task accuracy under the same inference budget.
```

### Contributions

```text
1. We introduce a counterfactual utility formulation for skill retrieval:
   a skill is useful if it causally improves verified downstream correctness over no-skill.

2. We propose UCB-guided failure-mode probing to efficiently acquire utility/risk/abstention
   labels from expensive LLM+verifier executions.

3. We train a utility-aware HYRR reranker and list-level controller that decide not only
   which skill is relevant, but whether any skill should be loaded.

4. We introduce evaluation metrics for post-topK skill incorporation:
   CS-Gold@K, HarmfulExposure, Utility@1, NoLoad Accuracy, and label yield per probe.
```

---

## 17. What to avoid claiming

Avoid:

```text
UCB is new.
ProbeLLM is our invention.
GRPO/R1-style reasoning is what we do.
Semantic reranking SOTA alone is solved.
```

Claim instead:

```text
We adapt bandit-guided probing to acquire verifier-grounded utility labels
for skill retrieval, a setting where semantic relevance and downstream usefulness diverge.
```

---

## 18. Verification checklist before writing paper

```text
[ ] Does UCB probing find more useful labels than random/top-score probing?
[ ] Are harmful/false-friend labels reliable under verifier audit?
[ ] Does controller improve final accuracy without retraining HYRR?
[ ] Does HYRR-v1 improve Utility@1 and reduce HarmfulExposure?
[ ] Does final end-task accuracy improve, not just nDCG?
[ ] Is the gain consistent across at least 4/6 SRA datasets?
[ ] Does method preserve Method7 Recall@10 within ≤1 pp?
[ ] Does ablation show UCB contributes beyond failure-mode labels alone?
[ ] Does ablation show utility/risk/no-load each contributes?
[ ] Are costs reported: probe calls, training time, inference latency?
```

---

## 19. Supported references

Use these references in the paper. They define the surrounding SOTA and the novelty boundary.

1. **SRA-Bench / Skill Retrieval Augmentation**  
   Weihang Su et al. *Skill Retrieval Augmentation for Agentic AI*. arXiv:2604.24594.  
   https://arxiv.org/abs/2604.24594

2. **ProbeLLM**  
   Yue Huang et al. *ProbeLLM: Automating Principled Diagnosis of LLM Failures*. arXiv:2602.12966.  
   https://arxiv.org/abs/2602.12966

3. **HYRR**  
   Jing Lu et al. *HYRR: Hybrid Infused Reranking for Passage Retrieval*. arXiv:2212.10528.  
   https://arxiv.org/abs/2212.10528

4. **UCB / Multi-Armed Bandit**  
   Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer. *Finite-time Analysis of the Multiarmed Bandit Problem*. Machine Learning, 2002.  
   https://link.springer.com/article/10.1023/A:1013689704352

5. **Self-RAG**  
   Akari Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.  
   https://arxiv.org/abs/2310.11511

6. **CRAG**  
   Shi-Qi Yan et al. *Corrective Retrieval Augmented Generation*. arXiv:2401.15884.  
   https://arxiv.org/abs/2401.15884

7. **Adaptive-RAG**  
   Soyeong Jeong et al. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. arXiv:2403.14403.  
   https://arxiv.org/abs/2403.14403

8. **SCARLet / Utility-based retrieval**  
   Yilong Xu et al. *Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models*. arXiv:2504.00573.  
   https://arxiv.org/abs/2504.00573

9. **LURE-RAG**  
   Manish Chandra et al. *LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG*. arXiv:2601.19535.  
   https://arxiv.org/abs/2601.19535

10. **Predicting Retrieval Utility**  
    Fangzheng Tian et al. *Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation*. arXiv:2601.14546.  
    https://arxiv.org/abs/2601.14546

11. **DeepSeek-R1**  
    DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.  
    https://arxiv.org/abs/2501.12948

12. **Search-R1**  
    Bowen Jin et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*. arXiv:2503.09516.  
    https://arxiv.org/abs/2503.09516

13. **Skill-RAG**  
    Kai Wei et al. *Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing*. arXiv:2604.15771.  
    https://arxiv.org/abs/2604.15771

---

## 20. Final recommendation

Best research direction:

```text
Do not frame the paper as "HYRR + better negatives".

Frame it as:
  "counterfactual utility learning for skill retrieval augmentation",
where UCB is the budget-aware data acquisition mechanism,
ProbeLLM is the failure-mode discovery inspiration,
HYRR is the reranker backbone,
and the controller solves the SRA-specific load/no-load bottleneck.
```

The method should be evaluated primarily on:

```text
final task correctness
CS-Gold@K
HarmfulExposure
NoLoad Accuracy
Utility@1
label yield per probe
```

not only Recall@K/nDCG.

If the results confirm the hypothesized gains, this has a credible novelty angle for a strong ML/NLP venue because it moves SRA from **semantic retrieval** to **verified utility-aware skill incorporation**.
