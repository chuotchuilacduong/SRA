# UCB-ProbeHYRR v2 — Architecture Design for Counterfactual Skill Utility Learning

> **Purpose of this redesign**  
> The previous draft correctly framed the research direction as counterfactual utility optimization, but it still centered the pipeline around `Method7 / HYRR-v0` as the first teacher. This v2 architecture makes the method cleaner and more novel: use **M4: RRF+KMeans top-50** as a cheap high-recall candidate generator, then train a new **CE-ProbeHYRR** directly from verifier-grounded counterfactual utility labels.

---

## 0. Executive decision

### Default architecture

```text
M4 RRF+KMeans top-50
  → UCB-guided counterfactual probing with local generator G, e.g. Qwen2.5:7B
  → task verifier V
  → utility / risk / false-friend / no-load / abstention labels
  → train CE-ProbeHYRR reranker
  → train list-level load/no-load controller
  → final skill selection and answer generation
```

### Why this is better than using Method7/HYRR-v0 as the probing teacher

Your Recall@50 table suggests that M4 and Method7 have almost identical top-50 coverage:

```text
M4 RRF+KMeans macro Recall@50 ≈ 91.9
M7 β=0.7 macro Recall@50 ≈ 91.88
```

So Method7 is not adding much **coverage** at top-50. Its main benefit is **ranking compression**, moving useful/gold skills from rank 11–50 into top-10/top-1.

Therefore, the new paper should not spend effort training a semantic CE-v0 first. Instead:

```text
M4 is the candidate pool generator.
CE-ProbeHYRR is trained once, directly for utility-aware reranking.
```

This gives a cleaner thesis:

> Starting from a cheap high-recall hybrid pool, we learn which retrieved skills are actually useful, harmful, or unnecessary through verifier-grounded counterfactual probing.

---

## 1. Research thesis

### Weak framing to avoid

```text
We improve HYRR by adding ProbeLLM-generated negatives.
```

This sounds like incremental hard-negative mining.

### Strong framing to use

```text
We formulate skill retrieval as budgeted counterfactual utility learning.
A candidate skill is not only ranked by semantic relevance, but by its verified causal effect on downstream correctness relative to no-skill and oracle-skill executions.
```

### One-line paper claim

> Semantic relevance is insufficient for Skill Retrieval Augmentation. We propose a bandit-guided counterfactual probing framework that learns whether a skill helps, harms, or should not be loaded, then trains a utility-aware skill reranker and load/no-load controller.

---

## 2. Full architecture overview

### 2.1 Training-time architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING-TIME PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

  Query q
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage A — Cheap high-recall candidate generation                     │
│                                                                     │
│   BM25(q) ─┐                                                        │
│            ├──► RRF fuse ─► KMeans α-blend ─► top-50 skills          │
│   BGE(q)  ─┘                                                        │
│                                                                     │
│   Default pool = M4 top-50                                           │
│   Reason: Recall@50 already near Method7, so CE-v0 is unnecessary    │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage B — Mandatory counterfactual probes                            │
│                                                                     │
│   y_no    = G(q)                                                     │
│   v_no    = V(q, y_no)                                               │
│                                                                     │
│   y_gold  = G(q + gold_skill), if available                          │
│   v_gold  = V(q, y_gold)                                             │
│                                                                     │
│   These establish:                                                   │
│     need_external(q)                                                 │
│     no_load(q)                                                       │
│     oracle upper bound                                               │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage C — UCB Probe Scheduler                                        │
│                                                                     │
│   Input: M4 top-50 + score features + probe history                  │
│   Action: choose candidate skill or failure bucket to probe           │
│   Objective: maximize useful labels per probe call                   │
│                                                                     │
│   UCB(a) = mean_reward(a) + c * sqrt(log(t) / N_a)                  │
└───────────────┬─────────────────────────────────────────────────────┘
                │ selected skill s_i
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage D — Candidate-skill execution                                  │
│                                                                     │
│   y_i = G(q + s_i)                                                   │
│   v_i = V(q, y_i)                                                    │
│                                                                     │
│   LabelBuilder derives:                                              │
│     utility(q, s_i) = v_i - v_no                                     │
│     risk(q, s_i)                                                     │
│     false_friend(q, s_i)                                             │
│     helpful / neutral / harmful                                      │
│     no_load / abstain                                                │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage E — Label Store                                                │
│                                                                     │
│   probe_logs.jsonl                                                   │
│   utility_pairs.jsonl                                                │
│   controller_labels.jsonl                                            │
│   failure_clusters.jsonl                                             │
└───────────────┬─────────────────────────────────────────────────────┘
                │
                ├────────────────────────────┐
                ▼                            ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│ Stage F1 — CE-ProbeHYRR        │   │ Stage F2 — List-level controller │
│                               │   │                                 │
│ Train pairwise/listwise CE    │   │ Train load/no-load/abstain       │
│ with utility/risk labels      │   │ from top-K distribution features │
└───────────────┬───────────────┘   └──────────────┬──────────────────┘
                │                                  │
                └─────────────────┬────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Output                                                               │
│                                                                     │
│   CE-ProbeHYRR reranker                                              │
│   Load/no-load/abstain controller                                    │
│   Failure-mode-aware negative dataset                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Inference-time architecture

UCB is not used online.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     INFERENCE-TIME PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘

  Query q
    │
    ▼
  M4 RRF+KMeans top-50
    │
    ▼
  CE-ProbeHYRR scores each candidate
    │
    ▼
  Controller sees q + top-K score distribution
    │
    ├── if no skill needed      → answer with no_skill
    ├── if gold likely absent   → abstain / fallback / retrieve more
    └── if skill needed         → select low-risk high-utility skill
                                      │
                                      ▼
                               G(q + selected_skill)
```

---

## 3. Why M4 top-50 should be the default candidate generator

### Observation

Your current result shows:

```text
M4 top-50 coverage ≈ M7 top-50 coverage
```

This implies:

```text
M4 is enough for candidate generation.
CE is still needed for reranking.
```

### Design consequence

Do not use this pipeline by default:

```text
train HYRR-v0 → use HYRR-v0 top-K → UCB probe → train HYRR-v1
```

Use this pipeline instead:

```text
M4 top-50 → UCB probe → train CE-ProbeHYRR directly
```

### Why this improves novelty

The paper no longer depends on a semantic CE teacher. The contribution becomes:

```text
A cheap high-recall retriever supplies candidates.
Verifier-grounded counterfactual probing supplies supervision.
The new reranker learns downstream utility, not semantic relevance alone.
```

---

## 4. Core modules

## 4.1 Candidate generator: M4 RRF+KMeans

### Input

```text
query q
skill corpus C
```

### Output

```text
S_50(q) = top-50 skills
```

### Stored features per candidate

```json
{
  "qid": "...",
  "skill_id": "...",
  "rank_m4": 7,
  "rrf_score": 0.031,
  "stage1_score": 0.82,
  "bm25_rank": 13,
  "bge_rank": 4,
  "cluster_id": 118,
  "cluster_affinity": 0.76,
  "skill_title": "...",
  "skill_description": "...",
  "skill_content": "..."
}
```

### Optional branch

Keep Method7 as a comparison baseline and optional teacher:

```text
M4 default: cheaper, cleaner, cold-start utility learning.
Method7 optional: stronger semantic reranking baseline.
```

---

## 4.2 Generator for probing: local lightweight LLM

Recommended default:

```text
G = Qwen2.5:7B-Instruct or similar local 7B model
```

Role:

```text
Generator/probe actor only.
```

Not role:

```text
Do not use G as the main verifier.
```

Reason:

```text
The label must come from task verifier V, not the same LLM that generated the answer.
```

---

## 4.3 Verifier V

The verifier should be as deterministic as possible.

| Dataset type | Preferred verifier |
|---|---|
| Code tasks | execution / unit tests |
| Math / numeric tasks | exact or tolerance-based numeric checker |
| MedCalc | formula/rule-based checker |
| Logic MCQ | exact option match |
| ToolQA | structured answer match |
| Open-ended theorem/explanation | use weak label or human-audited subset |

Label confidence:

```text
strong_label: execution/rule/exact verified
weak_label: LLM-judge or heuristic
ambiguous: exclude from main training or downweight
```

---

## 4.4 UCB Probe Scheduler

### What UCB chooses

UCB chooses **which probe action** to spend budget on.

A probe action can be:

```text
probe this candidate skill
probe this failure bucket
probe this score-conflict bucket
probe this boundary-near-gold bucket
```

### What UCB does not do

```text
UCB does not replace the reranker.
UCB does not run online at inference.
UCB does not decide the final answer.
```

### Why UCB is useful

Full probing is expensive:

```text
For each query:
  no_skill call
  gold_skill call
  K candidate-skill calls
```

With K=50 and thousands of queries, this is expensive. UCB reduces probing cost by allocating calls to high-yield regions.

---

## 5. UCB design

## 5.1 Simple bucket-level UCB, MVP

Each arm is a bucket.

```text
A1: top1_m4
A2: rank_2_to_5
A3: rank_6_to_20
A4: rank_21_to_50
A5: same_cluster_as_gold_non_gold
A6: high_bm25_low_bge
A7: high_bge_low_bm25
A8: score_gap_uncertain
A9: suspected_no_load
```

For each arm `a`:

```text
N_a = number of probes from arm a
R_a = total useful-label reward from arm a
mean_a = R_a / N_a
```

Selection:

```text
UCB(a) = mean_a + c * sqrt(log(t) / N_a)
```

where:

```text
t = total number of probes so far
c = exploration strength
```

Default:

```text
c = sqrt(2) or tuned on dev
```

Initialization:

```text
probe each arm at least once
```

---

## 5.2 Contextual UCB, advanced version

Bucket-level UCB ignores query context. Contextual UCB uses feature vector:

```text
x(q, s) = [
  dataset_id,
  rank_m4,
  rrf_score,
  bm25_rank,
  bge_rank,
  bm25_bge_gap,
  cluster_affinity,
  distance_to_gold_cluster_if_train,
  score_entropy_top50,
  query_length,
  skill_length,
  skill_domain,
  no_skill_correct_if_known
]
```

Action:

```text
choose candidate skill s to probe
```

Reward:

```text
useful_label(q, s)
```

Recommendation:

```text
Start with bucket-level UCB.
Only move to contextual UCB after MVP proves useful-label yield.
```

---

## 5.3 Reward definition

### Binary reward, MVP

```text
reward = 1 if label ∈ {
  helpful,
  harmful,
  false_friend,
  no_load_preferred,
  abstain,
  boundary_negative
}

reward = 0 if label ∈ {
  neutral,
  duplicate,
  ambiguous,
  unverifiable
}
```

### Weighted reward, stronger version

```text
harmful             → 1.0
false_friend        → 1.0
helpful             → 0.9
no_load_preferred   → 0.7
abstain             → 0.7
boundary_negative   → 0.6
neutral             → 0.0
unverifiable        → -0.2
```

### Development rule

Start with binary reward. Weighted reward can be an ablation.

---

## 6. Counterfactual label construction

## 6.1 Mandatory probes

For each training query:

```text
y_no = G(q)
v_no = V(q, y_no)
```

If gold skill exists:

```text
y_gold = G(q + s_gold)
v_gold = V(q, y_gold)
```

These two calls are not chosen by UCB. They are mandatory because they define the task-level counterfactual baseline.

---

## 6.2 Candidate probes

For a selected candidate skill `s_i`:

```text
y_i = G(q + s_i)
v_i = V(q, y_i)
```

Utility:

```text
u_i = utility(q, s_i) = v_i - v_no
```

---

## 6.3 Labels

### Helpful

```text
v_no = 0
v_i  = 1
```

Interpretation:

```text
skill_i improves correctness.
```

### Harmful

```text
v_no = 1
v_i  = 0
```

Interpretation:

```text
skill_i causes a previously correct answer to become wrong.
```

### False friend

```text
rank_m4(s_i) is high
u_i <= 0
v_gold = 1, if available
```

Interpretation:

```text
skill_i looks relevant but is not useful.
```

### Need external

```text
v_no = 0
v_gold = 1
```

Interpretation:

```text
the task needs external capability and the gold skill can help.
```

### No-load preferred

```text
v_no = 1
and no probed skill improves output
```

Interpretation:

```text
skill loading is unnecessary or risky.
```

### Abstain / fallback

```text
gold skill absent from top-50
or all probed candidates have utility <= 0 while task needs skill
```

Interpretation:

```text
top-50 is not trustworthy enough; retrieve more, abstain, or fallback.
```

---

## 7. Training CE-ProbeHYRR

## 7.1 Model architecture

Use a cross-encoder backbone, e.g. MiniLM or stronger CE.

```text
Input:
  [CLS] query [SEP] pack(skill) [SEP]

Shared encoder:
  h_cls

Heads:
  relevance_head(h_cls)      → scalar
  utility_head(h_cls)        → P(helpful)
  risk_head(h_cls)           → P(harmful)
  false_friend_head(h_cls)   → P(false_friend)
```

### Why multiple heads

```text
relevance: preserves retrieval capability
utility: learns downstream helpfulness
risk: avoids harmful skill loading
false_friend: penalizes semantically plausible distractors
```

---

## 7.2 Training group construction

For each query group, build:

```text
positive candidates:
  verified helpful skill
  gold skill only if verified helpful

negative candidates:
  false_friend
  harmful
  boundary negative
  original BM25/BGE/cluster hard negative
  random negative
```

Important rule:

```text
Do not blindly label gold skill as positive if G(q + gold_skill) fails.
```

Instead:

```text
gold skill with v_gold = 1 → verified positive
gold skill with v_gold = 0 → ambiguous for utility training, still useful for retrieval-only analysis
```

---

## 7.3 Losses

### Pairwise utility ranking loss

```text
L_pair = max(0, margin - score(q, s_pos) + score(q, s_neg))
```

where:

```text
s_pos = helpful skill
s_neg = harmful / false_friend / boundary negative
```

### Listwise utility loss

Assign target probability by utility label:

```text
helpful      → high target mass
neutral      → low target mass
harmful      → near zero
false_friend → near zero
```

### Multi-task classification loss

```text
L_cls =
  λ_u * BCE(helpful)
  + λ_r * BCE(harmful)
  + λ_f * BCE(false_friend)
```

### Total loss

```text
L_total =
  L_pair
  + λ_l * L_listwise
  + λ_u * L_helpful
  + λ_r * L_risk
  + λ_f * L_false_friend
```

---

## 7.4 Inference score

For each candidate skill:

```text
score_probehyrr(q, s) =
    w_rel  * relevance_score(q, s)
  + w_util * utility_score(q, s)
  - w_risk * risk_score(q, s)
  - w_ff   * false_friend_score(q, s)
  + w_m4   * M4_score(q, s)
```

Default starting weights:

```text
w_rel  = 0.4
w_util = 0.4
w_risk = 0.2
w_ff   = 0.2
w_m4   = 0.2
```

Tune on dev.

Reason to keep `M4_score`:

```text
M4 preserves lexical/dense/cluster robustness and protects against CE overfitting.
```

---

## 8. List-level controller

## 8.1 Why pairwise CE is not enough

A pairwise CE sees:

```text
(q, s_i)
```

It cannot reliably know:

```text
whether any skill is needed
whether gold is absent from the candidate set
whether top-K is globally low confidence
whether no-load is safer
```

So we need a list-level controller.

---

## 8.2 Controller input

```text
query features:
  dataset_id
  query length
  task type if available

top-K score features:
  max utility score
  max risk score
  max relevance score
  top1-top2 margin
  entropy of top-K scores
  number of high-risk candidates
  number of high-utility candidates
  M4 score dispersion

candidate metadata:
  top candidate titles/descriptions
  cluster diversity
```

---

## 8.3 Controller outputs

```text
P_need_external(q, topK)
P_gold_covered(q, topK)
P_load(q, topK)
```

Policy:

```python
if P_need_external < tau_need:
    return no_skill

if P_gold_covered < tau_gold:
    return abstain_or_fallback

selected = argmax_s(score_probehyrr(q, s))

if risk(selected) > tau_risk:
    return no_skill_or_fallback

return load(selected)
```

---

## 8.4 Controller training labels

```text
need_external = 1 if v_no = 0 and v_gold = 1
need_external = 0 if v_no = 1 and skills do not improve

load = 1 if at least one candidate skill has utility > 0
load = 0 if no_load_preferred

gold_covered = 1 if gold skill in top-50
gold_covered = 0 otherwise
```

Ambiguous labels should be downweighted or excluded.

---

## 9. ProbeLLM-style failure-mode mining

This component should be used **after initial probing**, not before.

### 9.1 Failure-aware record

For every failed or informative probe:

```json
{
  "qid": "...",
  "query": "...",
  "top50_summary": "...",
  "probed_skill": "...",
  "generator_output": "...",
  "verifier_result": 0,
  "v_no": 0,
  "v_gold": 1,
  "label": "false_friend",
  "failure_condition": "rank high but utility <= 0"
}
```

### 9.2 Clustering

```text
Embedding input:
  query + skill summary + output + verifier result + derived label

Clustering:
  HDBSCAN or KMeans

Cluster summarization:
  LLM summarizes recurring failure condition
```

### 9.3 Use of clusters

Failure clusters are used to:

```text
create new UCB arms
construct boundary negatives
explain failure modes in paper
stratify ablation/evaluation
```

They are not used as ground-truth labels without verifier support.

---

## 10. Data artifacts

### 10.1 `m4_top50.jsonl`

```json
{
  "qid": "...",
  "query": "...",
  "gold_skill_ids": ["..."],
  "top50": [
    {
      "skill_id": "...",
      "rank_m4": 1,
      "m4_score": 0.94,
      "bm25_rank": 5,
      "bge_rank": 2,
      "cluster_id": 12
    }
  ]
}
```

### 10.2 `probe_logs.jsonl`

```json
{
  "qid": "...",
  "skill_id": "...",
  "probe_type": "candidate_skill",
  "ucb_arm": "high_bge_low_bm25",
  "generator_model": "Qwen2.5-7B-Instruct",
  "answer": "...",
  "verifier_score": 1,
  "verifier_type": "execution_or_rule",
  "v_no": 0,
  "v_gold": 1,
  "utility": 1,
  "label": "helpful",
  "reward": 1
}
```

### 10.3 `utility_train_groups.jsonl`

```json
{
  "qid": "...",
  "query": "...",
  "group": [
    {"skill_id": "s1", "label": "helpful", "target": 1.0},
    {"skill_id": "s2", "label": "false_friend", "target": 0.0},
    {"skill_id": "s3", "label": "harmful", "target": 0.0},
    {"skill_id": "s4", "label": "semantic_negative", "target": 0.1}
  ]
}
```

### 10.4 `controller_labels.jsonl`

```json
{
  "qid": "...",
  "need_external": 1,
  "gold_covered_top50": 1,
  "load_label": 1,
  "no_load_label": 0,
  "abstain_label": 0,
  "label_confidence": "strong"
}
```

---

## 11. Algorithms

## 11.1 Data acquisition algorithm

```python
def build_counterfactual_dataset(queries, m4_retriever, generator, verifier, budget=3):
    ucb = BucketUCB(arms=[
        "top1_m4",
        "rank_2_to_5",
        "rank_6_to_20",
        "rank_21_to_50",
        "same_cluster_as_gold",
        "high_bm25_low_bge",
        "high_bge_low_bm25",
        "suspected_no_load",
    ])

    for q in queries:
        top50 = m4_retriever.retrieve(q, k=50)

        y_no = generator.answer(q, skill=None)
        v_no = verifier.score(q, y_no)

        if q.has_gold_skill:
            y_gold = generator.answer(q, skill=q.gold_skill)
            v_gold = verifier.score(q, y_gold)
        else:
            v_gold = None

        actions = build_actions(q, top50, v_no, v_gold)

        for _ in range(budget):
            action = ucb.select(actions)
            s = action.skill

            y_s = generator.answer(q, skill=s)
            v_s = verifier.score(q, y_s)

            label = derive_label(q, s, v_no, v_s, v_gold, action.features)
            reward = label_usefulness(label)

            ucb.update(action.arm, reward)
            save_probe(q, s, y_s, v_s, label, reward)
```

---

## 11.2 Label derivation algorithm

```python
def derive_label(q, s, v_no, v_s, v_gold, features):
    utility = v_s - v_no

    if utility > 0:
        return {
            "label": "helpful",
            "utility": utility,
            "risk": 0,
            "false_friend": 0,
        }

    if v_no == 1 and v_s == 0:
        return {
            "label": "harmful",
            "utility": utility,
            "risk": 1,
            "false_friend": 0,
        }

    if features["rank_m4"] <= 10 and utility <= 0:
        return {
            "label": "false_friend",
            "utility": utility,
            "risk": int(v_no == 1 and v_s == 0),
            "false_friend": 1,
        }

    if v_no == 1 and utility <= 0:
        return {
            "label": "no_load_preferred",
            "utility": utility,
            "risk": 0,
            "false_friend": 0,
        }

    if v_gold is not None and v_no == 0 and v_gold == 1 and utility <= 0:
        return {
            "label": "bad_candidate_when_skill_needed",
            "utility": utility,
            "risk": 0,
            "false_friend": int(features["rank_m4"] <= 20),
        }

    return {
        "label": "neutral",
        "utility": utility,
        "risk": 0,
        "false_friend": 0,
    }
```

---

## 12. Baselines

## 12.1 Retrieval baselines

```text
B0: BM25
B1: BGE
B2: RRF
B3: M4 RRF+KMeans
B4: CE-HYRR original
B5: Method7 β=0.7
B6: CE-ProbeHYRR without utility labels
B7: CE-ProbeHYRR with random-probed labels
B8: CE-ProbeHYRR with top-score-probed labels
B9: CE-ProbeHYRR with UCB-probed labels
```

## 12.2 Incorporation baselines

```text
I0: no_skill
I1: gold_skill oracle
I2: M4 top1 full skill
I3: Method7 top1 full skill
I4: Method7 topK full injection
I5: CE-ProbeHYRR top1 full skill
I6: CE-ProbeHYRR + controller
I7: CE-ProbeHYRR + controller + fallback/abstain
```

## 12.3 Probe policy baselines

```text
P0: random candidate probing
P1: top-rank probing
P2: round-robin bucket probing
P3: uncertainty probing
P4: bucket-level UCB
P5: contextual UCB
```

---

## 13. Metrics

## 13.1 Retrieval metrics

```text
Recall@1 / @5 / @10 / @50
nDCG@10
MRR
```

## 13.2 Utility-aware metrics

```text
Utility@1:
  top selected skill has utility > 0

FalseFriend@10:
  top-10 skills with high relevance but utility <= 0

HarmfulExposure:
  exposed skill makes no_skill-correct query wrong

NoLoad Accuracy:
  controller correctly avoids loading a skill when no skill is needed

Abstain Accuracy:
  controller abstains/fallbacks when gold is absent or top-K is unsafe
```

## 13.3 End-task metrics

```text
Final Accuracy / EM / Pass@1

CS-Gold@K:
  P(final answer correct | gold skill ∈ top-K)

Skill-Free Correct Preservation:
  P(final correct | no_skill correct)

Oracle Gap:
  GoldSkillOracle - Method
```

## 13.4 Probing efficiency metrics

```text
Useful labels per 100 probes
False-friend labels per 100 probes
Harmful labels per 100 probes
Cost per useful label
Final accuracy per 1K probe calls
```

---

## 14. Expected outcomes and prediction

These are hypotheses to test, not claims.

### Retrieval prediction

```text
Recall@50:
  should remain around 91.7–92.0 because M4 top-50 defines the pool.

Recall@10:
  likely 86–88 if utility labels are clean.
  likely 82–85 if labels are noisy.

nDCG@10:
  likely 75–77 if utility labels improve ordering.
```

### End-task prediction

```text
Final task accuracy:
  +2 to +5 points over Method7 top1 injection in Round 1.

HarmfulExposure:
  20–40% relative reduction if risk labels are reliable.

CS-Gold@50:
  should improve more than Recall@K because the method targets incorporation, not just retrieval.
```

### Where gains should appear

```text
Strongest gains:
  no-load decisions
  false-friend avoidance
  skill-needed cases where gold is in top-50 but not top-1

Weakest gains:
  cases where gold is absent from top-50
  cases where verifier is noisy
  cases where Qwen2.5:7B cannot use gold skill correctly
```

---

## 15. Ablation plan

### 15.1 Architecture ablation

```text
A0: M4 only
A1: Method7 β=0.7
A2: M4 → CE-ProbeHYRR, no controller
A3: M4 → CE-ProbeHYRR + controller
A4: Method7-v0 → UCB labels → HYRR-v1
A5: M4 cold-start → UCB labels → CE-ProbeHYRR
```

This directly tests:

```text
Do we need CE-v0 as teacher?
```

### 15.2 Label ablation

```text
L0: original semantic HYRR negatives only
L1: + helpful labels
L2: + false_friend labels
L3: + harmful labels
L4: + no_load labels
L5: + abstain labels
L6: all labels
```

### 15.3 UCB ablation

```text
U0: random probing
U1: top-rank probing
U2: round-robin buckets
U3: bucket-level UCB
U4: contextual UCB
```

### 15.4 Budget ablation

```text
B = 1, 2, 3, 5, 10 candidate probes/query
```

Report:

```text
label yield vs budget
final accuracy vs budget
training cost vs gain
```

---

## 16. Success criteria

### Minimum viable success

```text
1. UCB finds ≥1.5× more useful labels per 100 probes than random probing.
2. CE-ProbeHYRR improves Utility@1 over Method7.
3. Controller reduces HarmfulExposure relative to top1/topK injection.
4. Final accuracy improves under the same inference budget.
5. Recall@50 does not drop because candidate pool remains M4 top-50.
```

### Strong paper-level success

```text
+2–5 pp final task accuracy over Method7 top1 injection.
+5–10 pp CS-Gold@50.
≥30% relative reduction in HarmfulExposure.
No significant drop in Recall@10/nDCG@10.
UCB label yield > random/top-rank probing by ≥1.5×.
```

---

## 17. Trade-offs and risks

### 17.1 Probing cost

Risk:

```text
Counterfactual probing can be expensive.
```

Mitigation:

```text
Use local Qwen2.5:7B.
Use M4 top-50, not top-100.
Use UCB budget B=3 initially.
Cache all generations.
```

---

### 17.2 Verifier noise

Risk:

```text
Bad verifier creates bad utility labels.
```

Mitigation:

```text
Use execution/rule/exact verifiers where possible.
Downweight weak labels.
Human-audit 100–200 samples.
Report label confidence.
```

---

### 17.3 Generator dependence

Risk:

```text
Utility labels are specific to Qwen2.5:7B.
A skill helpful for Qwen may not be helpful for another generator.
```

Mitigation:

```text
Log generator_model_id.
Evaluate transfer to at least one stronger or different generator.
Train generator-specific controller if needed.
```

---

### 17.4 Cold-start M4 limitation

Risk:

```text
If gold is absent from M4 top-50, CE-ProbeHYRR cannot recover it.
```

Mitigation:

```text
Measure gold_absent_top50.
Use abstain/fallback for low-coverage cases.
Optionally increase pool to top-100 for low-confidence queries.
```

---

### 17.5 Novelty risk

Risk:

```text
Reviewer may view this as UCB + hard negatives.
```

Mitigation:

Frame method as:

```text
budgeted counterfactual utility learning for skill retrieval augmentation
```

not:

```text
hard negative mining
```

---

## 18. Implementation plan

## Phase 0 — Confirm pool choice

Run and report:

```text
M4 Recall@50
Method7 Recall@50
M4 Recall@10
Method7 Recall@10
```

Expected conclusion:

```text
M4 top-50 is enough for coverage.
CE-ProbeHYRR is needed for reranking.
```

---

## Phase 1 — Build M4 top-50 cache

Output:

```text
results/probehyrr/m4_top50_train.jsonl
results/probehyrr/m4_top50_dev.jsonl
results/probehyrr/m4_top50_test.jsonl
```

---

## Phase 2 — Run mandatory probes

For each train query:

```text
G(q)
G(q + gold_skill), if available
```

Output:

```text
results/probehyrr/mandatory_probes.jsonl
```

---

## Phase 3 — Run UCB candidate probing

Budget:

```text
B = 3 candidate probes/query
```

Output:

```text
results/probehyrr/ucb_probe_logs.jsonl
```

---

## Phase 4 — Build utility training groups

Output:

```text
results/probehyrr/utility_train_groups.jsonl
results/probehyrr/controller_labels.jsonl
```

---

## Phase 5 — Train CE-ProbeHYRR

Suggested file:

```text
src/sragents/train/train_probehyrr_ce.py
```

Suggested model output:

```text
models/ce_probehyrr_m4_ucb/
```

---

## Phase 6 — Train controller

Suggested file:

```text
src/sragents/train/train_skill_controller.py
```

Suggested model output:

```text
models/skill_load_controller/
```

---

## Phase 7 — Evaluate

Compare:

```text
M4 top1 injection
Method7 top1 injection
Method7 topK injection
CE-ProbeHYRR top1 injection
CE-ProbeHYRR + controller
Gold skill oracle
No skill
```

Report:

```text
Recall/nDCG
Utility@1
HarmfulExposure
NoLoad Accuracy
CS-Gold@50
Final task accuracy
Probe efficiency
```

---

## 19. Paper positioning

### Title candidates

```text
1. UCB-ProbeHYRR: Bandit-Guided Counterfactual Utility Learning for Skill Retrieval
2. From Relevance to Utility: Counterfactual Skill Reranking for Agentic Retrieval
3. When Not to Load Skills: Utility-Aware Skill Retrieval Augmentation
```

### Contributions

```text
1. Counterfactual skill utility formulation:
   a skill is useful if it causally improves verified downstream correctness over no-skill.

2. UCB-guided probing:
   a budget-aware label acquisition method for finding useful, harmful, false-friend, and no-load examples.

3. CE-ProbeHYRR:
   a multi-head cross-encoder reranker trained with utility/risk/false-friend supervision.

4. List-level controller:
   decides whether to load, no-load, abstain, or fallback based on top-K utility/risk distribution.

5. Evaluation protocol:
   Utility@1, HarmfulExposure, NoLoad Accuracy, CS-Gold@K, and useful-label yield per probe.
```

### Abstract skeleton

```text
Skill Retrieval Augmentation enables agents to dynamically load external capabilities,
but high top-K recall does not guarantee correct skill use. We observe that a cheap
hybrid RRF+KMeans retriever already provides near-saturated top-50 coverage, while
remaining failures arise from loading semantically plausible but non-useful or harmful
skills. We propose UCB-ProbeHYRR, a budgeted counterfactual utility learning framework
for skill retrieval. Starting from a high-recall hybrid candidate pool, a UCB scheduler
allocates limited LLM+verifier probe calls to candidate skills and failure buckets. The
resulting no-skill, candidate-skill, and oracle-skill executions produce utility, risk,
false-friend, no-load, and abstention labels. These labels train a multi-head cross-encoder
reranker and a list-level load/no-load controller. Experiments on SRA-Bench evaluate not
only Recall and nDCG, but also conditional success given gold@K, harmful exposure, no-load
accuracy, and final task correctness.
```

---

## 20. SOTA comparison table

| Line of work | What it does | Why it is not enough | UCB-ProbeHYRR difference |
|---|---|---|---|
| HYRR | Trains reranker from hybrid sparse+dense retrieval candidates | Optimizes passage relevance, not downstream skill utility | Uses verified utility/risk/no-load labels for skill reranking |
| SRA-Bench | Defines skill retrieval, incorporation, and execution benchmark | Reveals load/need bottleneck but does not solve it | Directly trains load/no-load and skill utility controller |
| ProbeLLM | Discovers structured LLM failure modes with verifiable probes | Diagnoses failures but does not train skill retriever | Converts verified failures into reranker/controller supervision |
| Self-RAG / CRAG / Adaptive-RAG | Adaptive document retrieval and correction | Focus on documents/passages, not reusable skills | Skill/capability utility with no-skill/candidate/oracle counterfactuals |
| Utility-aware RAG | Optimizes retrieved passage utility | Utility is usually document context utility | Utility is skill/capability effect on task execution |
| Search-R1 / DeepSeek-R1 style RL | Trains LLM reasoning/search policy | Requires RL on generator trajectories | Uses bandit probing for data acquisition and supervised reranker/controller training |

---

## 21. Supported references

Use these to verify claims and position novelty.

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

10. **DeepSeek-R1**  
    DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.  
    https://arxiv.org/abs/2501.12948

11. **Search-R1**  
    Bowen Jin et al. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*. arXiv:2503.09516.  
    https://arxiv.org/abs/2503.09516

12. **Skill-RAG**  
    Kai Wei et al. *Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing*. arXiv:2604.15771.  
    https://arxiv.org/abs/2604.15771

---

## 22. What changed from the previous markdown

| Previous draft | v2 redesign |
|---|---|
| Method7/HYRR-v0 is the default candidate generator | M4 RRF+KMeans top-50 is the default candidate generator |
| Two-pass training starts from trained semantic CE-v0 | Cold-start utility learning trains CE-ProbeHYRR directly |
| UCB probes Method7 top-K | UCB probes M4 top-50 buckets |
| Main claim can sound like HYRR + better negatives | Main claim is budgeted counterfactual utility learning |
| Controller is introduced after HYRR-v1 | Controller is a first-class module because load/no-load is central |
| Metrics include retrieval and utility | Metrics now emphasize final accuracy, CS-Gold@50, HarmfulExposure, NoLoad Accuracy |

---

## 23. Final recommendation

Use this as the main architecture:

```text
M4 top-50
  → Qwen2.5:7B counterfactual probes
  → verifier-grounded labels
  → UCB budget allocation
  → CE-ProbeHYRR utility reranker
  → list-level controller
  → selected skill / no-load / abstain
```

Do not make the paper about increasing Recall@50. That metric is already saturated.

Make the paper about:

```text
same high-recall candidate pool,
better utility-aware reranking,
safer load/no-load decisions,
higher final answer correctness.
```

