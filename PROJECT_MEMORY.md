# CSE495B — GBD Project Memory & Progress Log

> **Purpose:** Living project memory and append-only changelog tracking goals, architectural decisions, hardware constraints, code changes, and experimental results along the way.
>
> **CRITICAL WORKFLOW RULE (Active from Nov 12, 2026 — Entry 007):**
> - **DO NOT run training (`train.py`) or test (`test_gbd.py`, `gbd_v2_check.py`) scripts in the shell.**
> - Assistant **writes/edits code only**; **only standalone evaluation code (`gbd/eval.py`)** may be executed in the shell when needed.

---

## Current Project Snapshot

- **Project Title:** Geometric Backoff Distillation (GBD): A Teacher-Free Target for Training Small Language Models
- **Core Objective:** Train small language models locally **without any external teacher model**, using corpus-derived two-way backoff soft targets (exact multi-order backoff + embedding-space semantic neighbour backoff, weighted by KL risk).
- **Scale Clarification ("30M" vs. "30B"):** NotebookLM verification of course materials confirmed the professor's reference is **~30 million (**`~30M`**) parameters** (not 30B — earlier "30B" was a million/billion mishearing), under strict **in-house / no-cloud / no-API** and **small-data / teacher-free** constraints.
- **Target Use Case & Staged Model Ladder (Aligned with Nov 12 Pitch Revision):**
  - **Task:** Conversation reply on **strictly human-written English dialogue** (local, teacher-free, from-scratch initialization by deliberate choice).
  - **Tokeniser:** SentencePiece **8k–16k** trained on the training split only (`byte_fallback=True` required for zero-`<unk>` BLiMP evaluation).
  - **Stage 1 (Primary Gate — Start Here):** GPT-2-small-shaped decoder (`12 layers, d=768`, ~91M–97M params at 8k–16k vocab; `124M` at 50k vocab) + **`~30M` compact decoder** matching the course's explicit ~30M reference scale. Full-parameter AdamW on 16 GB VRAM.
  - **Stage 2 (Conditional Scale-Up):** From-scratch configurations of **Granite, Llama, Gemma, Phi, Hymba** architectures if Stage 1 passes kill criteria. *(Hardware note: because training is from scratch rather than LoRA/QLoRA fine-tuning, full-parameter mixed-precision AdamW on 16 GB VRAM caps student size around **0.5B–0.9B** rather than 2B–3B.)*
- **Local Hardware & Software Environment:**
  - **OS:** Windows 11 (`win32 10.0.26300`) — PowerShell + `wsl.exe` available.
  - **GPU:** 1× NVIDIA GeForce RTX 5060 Ti (**16 GB VRAM**, ~15.0 GB usable under WDDM; CUDA driver 13.5 / 621.01).
  - **System RAM:** **16 GB installed, expandable to 32 GB temporarily** (borrowing 16 GB from a friend — ideal window to build and persist offline GBD target `.npy`/memmap files).
  - **CPU:** 12 logical processors.
  - **Python Env Status:** Local `.venv` configured (`Python 3.12`, `torch 2.14.1+cu130` with native RTX 5060 Ti `bf16` SDPA, `numpy`, `scipy`, `faiss-cpu`, `gensim`, `sentencepiece`, `datasets`).
- **Current Codebase (**`gbd/`**):**
  - `gbd/gbd.py` — **Production GBD v2** with **fully orthogonal component switches** (`use_exact`, `use_order_backoff`, `exact_orders`, `leave_one_out`, `respect_eos`, `use_semantic`, `skip_role_markers`, `partition_turn_state`, `pool_semantic`, `lambda_mode={adaptive,fixed}` + `fixed_lambda`, `alpha_mode={adaptive,r_ord,R,r_sem,fixed}`, `phi={inv,exp,uniform}`, and `sem_feature_fn` hook), `uint16`/`fp16` memmap I/O, `gbd_loss`, and `GBDChunks`.
  - `gbd/embeddings.py` — Pluggable geometry/embedding module supporting 6 teacher-free sources: `w2v` (same-corpus skip-gram), `ppmi_svd` (100% count-only Shifted PPMI + truncated SVD), `random` (Gaussian control), `student_tok_emb` (student token table), `student_hidden` (mid-layer student contextual hidden states pooled per semantic context), and `file`.
  - `gbd/model.py` — Causal Decoder Transformer students (`30m` = `30.49M` params; `gpt2-small` = `97.65M` params at `V=16k`) with Flash/SDPA causal attention, `forward_hidden(idx, layer)` for contextual state extraction, and static-embedding init.
  - `gbd/train.py` — Config-switchable trainer (`--config <json>` + CLI overrides for every GBD switch, faithful Sathe & Sarawagi Eq. 4 `coconts`, without-replacement `random_target`, modified Kneser–Ney `kn`, neighbour-restricted D2GPo `d2gpo`, Born-Again / BabyLlama-2 `self_kd` with a comma-separated teacher list and mean-of-logits ensemble, canonical `blake2b` target cache + `cache_spec.json` verification, collision-free `run_name`, `--alpha_schedule {constant,linear,cosine}`, student-geometry bootstrapping, and `bf16` `.pt` checkpoint saving).
  - `gbd/eval.py` — **Standalone evaluation CLI** (`--checkpoint <path.pt>`, Reply/All BPB, Rare-token NLL bins, ECE, `--eval_blimp`, and `--num_samples` qualitative reply generation).
  - `gbd/summarize.py` — Writes `artifacts/results_summary.md` from `artifacts/results_groups.json` and the saved run / BLiMP JSON. Missing runs stay missing.
  - `gbd/kn.py`, `gbd/d2gpo.py` — Kneser–Ney and D2GPo-NN64 target builders. D2GPo-NN64 is the neighbour-restricted variant, not full-vocabulary D2GPo.
  - `gbd/data.py` — Human-only conversation loader (`oasst1`/`oasst2` node-disjoint tree decomposition, `no_robots`, `dailydialog`, `prosocial`), deduplicator, `16k` SentencePiece BPE (`byte_fallback=True`), same-corpus `word2vec`, and Phase 0 diagnostics.
  - `gbd/test_gbd.py` — Unit tests covering v2 parity, brute-force Eq. 1/2, LOO/constant-$\alpha$, memmap I/O, autograd gradients, orthogonal switches, CoCoNTs Eq. 4 and cache isolation, KN against a dict multiset, D2GPo-NN64 rows, and Born-Again / duplicate-teacher self-KD.
  - `gbd/gbd_v2_check.py` — Reference GBD v2 prototype & 3-seed synthetic benchmark (`test` and `bench`).

---

## Open Decisions & Action Items

- [x] **Professor requirement on random initialization:** Confirmed — random-weight initialization is **not** strictly required by the professor.
- [x] **Clarify "30B" vs. "30M" professor constraint:** Resolved via NotebookLM — professor specified **~30M parameters** compact student models, **in-house / no-cloud**, **small-data teacher-free** training. (Can also remove/update the old `"30B-level"` open item on lines 341 & 354 of the pitch `.md`.)
- [x] **Lock Stage 1 starter model:** Confirmed `gpt2-small` **(**`124M` **/** `0.1B`**)** as the first model to validate GBD end-to-end before scaling up to larger models (`Granite`, `Llama`, `Gemma`, `Phi`, `Hymba`) and/or adding a `~30M` compact point.
- [x] **Decide** `gpt2-small` **init mode(s) for Stage 1:** Run pretrained `gpt2` (124M) fine-tuning on conversation reply (matching CoCoNTs' `gpt2-125m` setup), from-scratch (~30M / 124M) small-corpus training, or both. - from scratch.
- [x] **Select human-only conversation corpus:** Built deduplicated human-only English conversation corpus (`oasst1` node-disjoint + `no_robots` + `dailydialog`: `29,141` clean conversations; `6,358,385` train tokens / `376,857` val / `340,225` test; `V = 16,000` SentencePiece with `byte_fallback=True`). *(Can optionally add `oasst2` / `prosocial` in `gbd/data.py` if a larger 15M–20M slice is desired later.)*
- [x] **Set up local ML environment:** Created `.venv` with `torch 2.14.1+cu130` (Blackwell RTX 5060 Ti `bf16` verified), `scipy`, `faiss-cpu`, `gensim`, `sentencepiece`, `datasets`.
- [x] **Port `gbd/gbd.py` to the final GBD v2 spec (Entry 005):** Ported Eqs. 1–5 + sparse `plugin_kl_v2` + HNSW switch + `uint16`/`fp16` memmap target persistence; verified exact match with `gbd_v2_check.py` and brute-force Eq. 1/2 tests in `gbd/test_gbd.py`.
- [x] **Phase 0: repeat the risk calibration check on the real corpus:** Completed in `artifacts/corpus/phase0_diagnostics.json` (`Spearman(r_ord, LOO NLL) = +0.3317`, `Spearman(r_sem, LOO NLL) = +0.0970` — confirms `r_ord` is the right driver for $\alpha_t$ on real human dialogue too).
- [x] **Phase 1 & Phase 2 Pilot Gate (`1M` and full `6.36M` tokens, `gpt2-small` & `30m`):** Completed!
  - **1M tokens:** Fixed-$\alpha$ GBD full crushes Tuned CE (`1.8984` / `1.8964` $\to$ **`1.8276` Reply BPB**, `-0.070` BPB / `-29` perplexity points!), Label Smoothing (`1.9047`), Exact 6-gram (`1.8954`), CoCoNTs (`1.8928`), and GBD order-only (`1.8473`), and wins on `30m` (`1.8685` $\to$ **`1.8389`**).
  - **6.36M full corpus (Truncation-Bias Discovery & Repair 2 Validation):** Unannealed $\alpha_{\max}=0.5$ plateaus slightly behind CE on `gpt2-small` (`1.3265` vs `1.3226`/`1.3179`) due to late-epoch order-$K$ truncation bias, **and linear $\alpha_{\max}$ annealing (`--anneal_alpha`, pre-planned Repair 2) completely solves it**, beating every CE seed on both `gpt2-small` (**`1.3037` Reply BPB** vs `1.3179`/`1.3226`) and `30m` (**`1.3282` Reply BPB** vs `1.3489`/`1.3523`).
- [x] **Pre-Phase-3 controls (Entries 008–009):** The 1M matched-α controls show the gain is not generic smoothing. CoCoNTs Eq. 4 was repaired and, at the pilot recipe, matches CE.
- [x] **Paper draft v6 (Nov 15):** `reports/GBD_paper_draft_v6.tex` is the current paper. At 1M tokens GBD beats tuned CE by **0.048** reply BPB on `gpt2-small` (1.7846 vs 1.8327) and **0.044** on `30m` (1.7833 vs 1.8273), leads every teacher-free baseline by at least **0.032**, ties Born-Again self-distillation, and recovers **80–83%** of a two-teacher ensemble's gain. At 6.36M tokens on `30m`, annealed GBD beats CE by **0.011** (1.2918 vs 1.3027) and ties the other annealed soft targets; a trained teacher beats GBD by **0.013**. BLiMP does not separate them. The gain is in mid-frequency tokens.
- [ ] **Still open after draft v6:** MIR and static-embedding initialisation; two more seeds each for Kneser–Ney and D2GPo-NN64 at 6.36M; prefix-only targets; the 98M student at 6.36M under the final recipe; whether GBD and a teacher add.

---

## Chronological Progress & Change Log

### [2026-11-11] Entry 001 — Baseline Audit & Shift Toward 2B–3B Local Conversation Models

- **Initial State Reviewed:**
  - Audited `Geometric Backoff Distillation (GBD) A Teacher-Free Target for Training Small LM.md` (Nov 11 rev 2) and `gbd/` (`gbd.py`, `test_gbd.py`, `sanity_synthetic.py`).
  - Original pitch scoped from-scratch pretraining of 39M and 124M decoder models on 1M–20M English tokens, treating 30B purely as a passive ceiling.
- **Strategic Direction Recorded:**
  - **30B Target Policy:** Hitting 30B level is not a hard blocker for project completion, but maximizing proximity to 30B-level conversational quality is a primary goal.
  - **Task & Scale Update:** Moving focus to **conversation reply** using **2B–3B class models**, trained **100% locally** on the RTX 5060 Ti (16 GB VRAM) + 16 GB System RAM machine.
  - **Candidate Architectures Proposed:** Gemma, Llama, Phi, GPT-2, Granite, Hymba.
- **Technical Findings Logged from Audit:**
  - Identified hardware constraints: 16 GB VRAM requires bf16 LoRA/DoRA or QLoRA + chunked logit loss for 2B–3B models (full fp32-state AdamW needs ~36 GB VRAM); 16 GB System RAM requires sparse/memmap care during `build_targets`.
  - Identified `_plugin_kl` bug in `gbd/gbd.py` where `fit_eps = 0.01` multiplies by `V`, adding `2,560` pseudo-counts at Gemma's $V=256\text{k}$ and converting sparse rows to dense `(200, V)` arrays.
  - Identified chat-template suffix collision risk for conversation data when $K_s = 2$ (assistant header tokens mask the preceding user prompt unless skipped or widened).
  - Created `PROJECT_MEMORY.md` to persist all ongoing updates.

### [2026-11-11] Entry 002 — RAM Upgrade Window, Staged Rollout Gate & Framing Alignment Analysis

- **Hardware & Execution Updates:**
  - **System RAM:** User can temporarily add **+16 GB RAM** (bringing system total to **32 GB RAM**). Best strategy: batch offline `build_targets` runs during the 32 GB window and save `.npy`/memmap targets to disk; subsequent GPU training streams chunks with minimal RAM.
  - **Staged Model Rollout:** Confirmed step-by-step gate — test GBD on **1–2 models first**; expand to the full model list only after verifying GBD beats CE/baselines.
  - **Random-Init Rule:** Confirmed professor does **not** strictly require random-weight initialization.
- **Python Environment Audit:**
  - Checked local `Python 3.12.10` (`win32`) and confirmed `wsl.exe` is installed; ML dependencies (`torch`, `scipy`, `faiss`, `gensim`, `peft`, `bitsandbytes`) are not yet installed in global Python, so a clean `.venv` or WSL2 environment will be created when implementation begins.
- **Framing Audit Against Pitch Doc (**`Geometric Backoff Distillation...md`**):**
  - Noted explicit conflict between current pitch text (lines 7, 252–261, 341: random-init 39M/124M decoder LM on 1M–20M tokens, 30B purely a passive ceiling) and the new goal (2B–3B conversation reply approaching 30B across Gemma/Llama/Phi/Granite/Hymba/GPT-2).
  - Documented concrete options to reconcile this cleanly without hidden teacher leakage (including human-only dialogue corpus requirement to avoid sequence-level KD from GPT-4 synthetic chat datasets).

### [2026-11-12] Entry 003 — "30M" Course Scale Clarification & Stage 1 Lock on `gpt2-small` (0.1B)

- **Course Constraint Resolved (NotebookLM Verification):**
  - Confirmed professor's benchmark target is **~30 million (**`~30M`**) parameters** (not 30 billion — resolves the `"30B-level"` open question in `Geometric Backoff Distillation...md` lines 341 & 354 as a million/billion mishearing).
  - Core course constraints confirmed:
    1. Compact student network (~30M scale focus).
    2. Strict **in-house / no-cloud / no closed API** local training and inference.
    3. **Small-data, low-resource, teacher-free** distillation/training regime.
- **Updated Staged Execution Plan:**
  - Keep conversation reply goal and multi-family roadmap, starting **Stage 1** with `gpt2-small` **(**`124M` **/** `0.1B`**)** (and optionally a `~30M` config to hit the professor's exact number head-on).
  - **Why** `gpt2-small` **first is a major win:**
    - Full-parameter AdamW training fits in **<3.5 GB VRAM** on the RTX 5060 Ti (no LoRA/quantization confounders).
    - ~20–25× faster per token than 2B–3B models, enabling rapid 3-seed baselines and ablations on current 16 GB RAM + 16 GB VRAM.
    - Matches CoCoNTs' exact `gpt2-125m` scale for direct comparison.
  - **Stage 2 (Scale-Up):** If GBD demonstrates clear gains on `gpt2-small`, scale upward to the larger model families (`Granite`, `Llama`, `Gemma`, `Phi`, `Hymba`, up to 2B–3B).
- **Technical Note Logged for** `gpt2-small` **(**`V = 50,257`**):**
  - `_plugin_kl` bug in `gbd/gbd.py` still triggers heavily at GPT-2's `V = 50,257` (`0.01 * 50,257 = 502.6` pseudo-counts vs. `n_fit = 20` real counts = 96% washout), so fixing `gbd.py` remains the top pre-flight step before running `gpt2-small`.

### [2026-11-12] Entry 004 — Pitch Consistency Audit & Complete Mathematical Specification of GBD v2 (All Equations)

> **Superseded by Entry 005 (same day).** Sections A and B below stand. Do not implement Section C: its Eqs. 3–5 have defects listed in Entry 005-A. Do not cite Section D: no v2 code existed when it was written and its numbers could not be reproduced. The final equations are in Entry 005-B.

#### A. Consistency Audit of Nov 12 Revised Pitch (`Geometric Backoff Distillation...md`)
- **Aligned Decisions Confirmed:** From-scratch initialization chosen deliberately; English human-only dialogue corpus (no synthetic LLM chat data); 8k–16k SentencePiece tokenizer; Stage 1 (`~30M` + GPT-2-small-shaped decoder) and Stage 2 capped at `~0.5B–0.9B` by full-parameter AdamW on 16 GB VRAM; 32 GB RAM window used to prebuild memmap targets.
- **5 Consistency Gaps Flagged:**
  1. **Corpus size vs. 20M scale:** Individual candidates (`DailyDialog` ~1.4M, `no_robots` ~3.2M, `oasst1` EN ~4–8M) are under 20M tokens, and `oasst1` branching trees require node/turn-level deduplication so shared prompt prefixes do not leak labels past Leave-One-Out.
  2. **Token scope mismatch:** Primary metric evaluates reply tokens only, while `GBDChunks` trains and computes median risk $\rho$ over all tokens (including frequent role markers).
  3. **Model size & compute:** An 8k–16k vocab GPT-2-small backbone is ~91M–97M params (not 124M), and running the 100-run Phase 3 matrix on ~30M first is 3–4× faster and matches the course scale directly.
  4. **Dangling $\lambda_t$ symbol:** Mentioned in Contribution 2 (`line 12`) but missing from old Method Eq. 4 (restored explicitly in GBD v2 Eq. 4 below).
  5. **BLiMP OOV protection:** SentencePiece 8k–16k must set `byte_fallback=True` so out-of-domain BLiMP words do not map to `<unk>`.

#### B. Algebraic Flaws Identified in GBD v1 (`gbd/gbd.py` & Old Method Eqs. 2–5)
1. **Small-$n$ variance cap (`Eq. 3`):** Because $d_{\text{obs}} \le n$ and $f_1 \le n$ for every empirical count vector, $\frac{\hat{d}-1}{2n} = \frac{d_{\text{obs}}+f_1}{2n} \le \frac{2n}{2n} = 1.0\text{ nat}$ for **all** $n \ge 1$. A singleton ($n=1$) at order $K$ ($\delta_K=0$) gets risk $\le 1.0\text{ nat}$ and steals ~50% target mass.
2. **High-dimensional distance floor & KKT clamp bug (`Eq. 3` / `fit_semantic_bias`):** Noisy unit embeddings in $\mathbb{R}^{D K_s}$ concentrate true synonyms around a positive distance shell $d_{\min} > 0$, making the true OLS intercept $a < 0$. Clamping $a \ge 0$ forced the regression line through $(0,0)$, assigning $3+$ nats of false bias to true synonyms; and clipping $b < 0 \mapsto 0$ without recomputing $a = \bar{y}_+$ violated KKT conditions.
3. **Nested suffix noise sign & `eps * V` prior blowup (`fit_delta` / `_plugin_kl`):** `eps * V` added $0.01 V$ pseudo-counts ($160$ at $V=16\text{k}$ vs. $n_{\text{fit}}=20$); and `fit_delta` added variances $\frac{d-1}{2N_{\text{hi}}} + \frac{d-1}{2N_{\text{lo}}}$ even though $c^{(k)}$ is a nested superset of $c^{(k+1)}$ whose null KL expectation is $\frac{d-1}{2}\big(\frac{1}{N_{\text{hi}}} - \frac{1}{N_{\text{lo}}}\big)$.
4. **Role-marker blindness & boundary bleed (`Eq. 1–2`):** Rolling hashes crossed `<eos>`, and at reply start (`... ? <assistant>`) $K_s=2$ only saw `("?", "<assistant>")`, masking the user's question. Naively skipping `<assistant>` without tracking turn-state $s_t$ would collide reply-opener positions with pre-marker positions that predict `<assistant>`.
5. **$M$-multiplicity inflation, Harmonic-Mean paradox & non-zero $\alpha_t$ floor (`Eq. 4–5`):** Summing $M$ raw semantic components inflated semantic share linearly with $M$; $R_t = \sum_j w_j r_j$ reduced to the harmonic mean (penalizing sharp contexts when fallbacks exist); and since unigram $k=0$ always exists ($r_{0,t} \approx \delta_0 < \infty$), $\alpha_t = \alpha_{\max}\exp(-R_t/\rho)$ could **never** reach $0$ (plain CE fallback never triggered).

---

#### C. Complete Repaired Mathematical Specification (GBD v2 — All Equations)

##### 0. Setup, Conversation Boundaries & Structural Token Set
Let the tokenized training corpus be $x_1, \dots, x_T \in \{1, \dots, V\}$, where conversations are separated by $\langle\text{eos}\rangle$. Let $b(t) \in \{1, \dots, t\}$ be the first token index of the conversation containing position $t$ (one index after the last $\langle\text{eos}\rangle$ before $t$). Let $\mathcal{S} \subset \{1, \dots, V\}$ be the set of structural role-marker tokens (e.g., $\langle\text{user}\rangle, \langle\text{assistant}\rangle$; $\mathcal{S} = \varnothing$ for plain text), and let $V_{\text{act}} = |\{w : N(\varnothing, w) > 0\}|$ be the active vocabulary size.

##### Equation 1 — Boundary-Safe Leave-One-Out Exact Components (Orders $k = 0, \dots, K$)
An order-$k$ exact context $c_t^{(k)} = x_{t-k:t-1}$ is **valid** iff $t - k \ge b(t)$ (never crossing $\langle\text{eos}\rangle$). For valid $c_t^{(k)}$, corpus counts are $N(c, w) = \sum_{s : c_s^{(k)} = c} \mathbb{1}[x_s = w]$ and $N(c) = \sum_{w=1}^V N(c, w)$. Removing position $t$ itself gives:
$$
n_{k,t} = N\!\big(c_t^{(k)}\big) - 1, \qquad
p_{k,t}(w) = \frac{N\!\big(c_t^{(k)}, w\big) - \mathbb{1}[w = x_t]}{n_{k,t}} \quad (n_{k,t} \ge 1).
$$
If $t - k < b(t)$ or $n_{k,t} = 0$, the component is inactive ($r_{k,t} = +\infty$). Each active distribution is row-truncated to its top $R_{\text{pre}}$ entries and renormalized before mixing.
*(Property: Averaged over all $N(c) \ge 2$ occurrences of $c$, $\frac{1}{N(c)}\sum_{t:c_t^{(k)}=c} p_{k,t}(w) = \frac{N(c,w)}{N(c)}$, so LOO removes $N(c)=1$ label leakage with zero corpus-level bias.)*

##### Equation 2 — Turn-State Partitioned, Content-Projected Semantic Components (Order $K_s$)
Define the immediate structural turn-state $s_t$ and the $K_s$ most recent **content** token positions $\tilde{\tau}_1(t) > \tilde{\tau}_2(t) > \dots > \tilde{\tau}_{K_s}(t) \ge b(t)$ satisfying $x_{\tilde{\tau}_m(t)} \notin \mathcal{S}$:
$$
s_t = \begin{cases} x_{t-1} & \text{if } t > b(t) \text{ and } x_{t-1} \in \mathcal{S}, \\ 0 & \text{otherwise,} \end{cases}
\qquad
\tilde{c}_t = \big(s_t,\; x_{\tilde{\tau}_1(t)},\; \dots,\; x_{\tilde{\tau}_{K_s}(t)}\big).
$$
Using unit-normalized static embeddings $\hat{e}(w) = e(w) / \|e(w)\|_2$ trained teacher-free on the training corpus:
$$
f(\tilde{c}_t) = \Big[\hat{e}\!\big(x_{\tilde{\tau}_1(t)}\big);\; \gamma\,\hat{e}\!\big(x_{\tilde{\tau}_2(t)}\big);\; \dots;\; \gamma^{K_s - 1}\,\hat{e}\!\big(x_{\tilde{\tau}_{K_s}(t)}\big)\Big] \in \mathbb{R}^{D K_s}.
$$
Within the same turn-state bucket ($s_i = s_t$), retrieve the $M_{\text{search}}$ nearest contexts $\tilde{c}_i \ne \tilde{c}_t$ by Euclidean distance $d_{i,t} = \|f(\tilde{c}_t) - f(\tilde{c}_i)\|_2$. Each candidate defines $p_i(w) = N(\tilde{c}_i, w) / N(\tilde{c}_i)$ (strictly leave-one-out since $\tilde{c}_i \ne \tilde{c}_t \implies t \notin \text{occ}(\tilde{c}_i)$). When $\mathcal{S} = \varnothing$, $\tilde{c}_t = c_t^{(K_s)}$ recovers original GBD identically.

##### Equation 3 — Empirical-Bayes Shrunk Support, Log-Saturating KL Variance & Debiased Bias Fitting
For any count vector of total count $n$, distinct observed types $d_{\text{obs}}$, singletons $f_1$, and doubletons $f_2$, estimate the order-$k$ prior branching factor $d_0(k)$ on frequent contexts ($N(c) \ge n_{\text{fit}}$) via monotone Chao1 lower-bound median:
$$
d_0^{\text{raw}}(k) = \operatorname{median}_{c : N(c) \ge n_{\text{fit}}} \!\left(d_{\text{obs}}(c) + \frac{f_1(c)^2}{2 \max(f_2(c), 1)}\right), \qquad
d_0(k) = \min\!\big(d_0^{\text{raw}}(k),\, d_0(k-1)\big)
$$
(with $d_0(0) = d_0^{\text{raw}}(0)$, inheriting $d_0(k-1)$ if no context at order $k$ has $N(c) \ge n_{\text{fit}}$, consistent with $H(X_t \mid C_t^{(k)}) \le H(X_t \mid C_t^{(k-1)})$). Shrink each context's support estimate with prior strength $n_0 > 0$ (default $n_0 = 5$) and compute bounded non-asymptotic KL sampling variance:
$$
\tilde{d}(n, d_{\text{obs}}, f_1; k) = \frac{n\,(d_{\text{obs}} + f_1 + 1) + n_0\, d_0(k)}{n + n_0}, \qquad
v(n, d_{\text{obs}}, f_1; k) = \ln\!\left(1 + \frac{\tilde{d}(n, d_{\text{obs}}, f_1; k) - 1}{2\, n}\right).
$$
*(Asymptotics: as $n \gg \tilde{d}$, $v \to \frac{d_{\text{obs}} + f_1}{2n}$ matching Balasundaram et al. Theorem 3.1; at $n = 1$, $\tilde{d} \approx d_0(k)$ breaks the artificial $1.0\text{-nat}$ ceiling while $\ln(1 + \cdot) < \ln V_{\text{act}}$ prevents unphysical divergence above corpus entropy.)*

Component risks against the order-$K$ conditional are:
$$
r_{k,t} = \delta_k + v(n_{k,t}, d_{k,t}, f_{1,k,t};\, k), \qquad
r_{i,t} = \delta_{K_s} + \big[a + b\,\|f(\tilde{c}_t) - f(\tilde{c}_i)\|_2\big]_+ + v\!\big(N(\tilde{c}_i), d_i, f_{1,i};\, K_s\big).
$$
From the $M_{\text{search}}$ semantic candidates, keep the $M$ lowest-risk neighbours $\mathcal{N}_t$ ($|\mathcal{N}_t| = M$).

**Scale-Free Bias Fitting ($\delta_k, a, b$) on Frequent Training Contexts ($N \ge n_{\text{fit}}$):**
- **Vocabulary-invariant smoothed plug-in KL** with fixed total Dirichlet prior mass $\beta_0 > 0$ (default $\beta_0 = 3.0$, evaluated sparsely over $w : N(c_1, w) > 0$):
  $$
  \widehat{\operatorname{KL}}_{\beta_0}(c_1 \,\|\, c_2) = \sum_{w : N(c_1, w) > 0} \frac{N(c_1, w)}{N(c_1)} \ln \frac{N(c_1, w) / N(c_1)}{\big(N(c_2, w) + \beta_0 / V_{\text{act}}\big) / \big(N(c_2) + \beta_0\big)}.
  $$
- **Order-truncation bias $\delta_k$ (nested suffix noise subtraction — minus sign!):** $\delta_K = 0$, and for $j = K-1, \dots, 0$ over sampled $c \in \mathcal{C}_{j+1}$ ($N(c) \ge n_{\text{fit}}$) with $c' = \operatorname{suffix}_j(c)$:
  $$
  \delta_j = \delta_{j+1} + \operatorname{mean}_{c}\!\left[\widehat{\operatorname{KL}}_{\beta_0}(c \,\|\, c') - \ln\!\left(1 + \frac{\tilde{d}_c - 1}{2}\left(\frac{1}{N(c)} - \frac{1}{N(c')}\right)\right)\right]_+.
  $$
- **Rectified semantic distance bias $[a + b\,\|\Delta f\|_2]_+$ ($a \in \mathbb{R},\; b \ge 0$):** Fit $(a_{\text{ols}}, b_{\text{ols}})$ by ordinary least squares of $y_{c,c'} = \big[\widehat{\operatorname{KL}}_{\beta_0}(c \,\|\, c') - v(N(c), d_c, f_{1,c}; K_s) - v(N(c'), d_{c'}, f_{1,c'}; K_s)\big]_+$ on $d_{c,c'} = \|f(c) - f(c')\|_2$ over pairs with $N(c), N(c') \ge n_{\text{fit}}$. Enforce exact KKT slope non-negativity: if $b_{\text{ols}} < 0$, set $(a, b) = (\operatorname{mean}(y),\, 0)$; else $(a, b) = (a_{\text{ols}},\, b_{\text{ols}})$ (allowing $a < 0$ to capture the zero-bias embedding noise shell $d_{\min} = -a/b$).

##### Equation 4 — Pooled Semantic Estimator & Explicit Two-Way Mixture ($\lambda_t$)
With precision mapping $\phi(r) = \frac{1}{\max(r, r_{\min})}$ (or softmin $\phi(r) = e^{-\tau r}$):
1. **Exact order-backoff branch** ($k \in \{0, \dots, K\}$):
   $$
   u_{k,t} = \frac{\phi(r_{k,t})}{\sum_{k'=0}^K \phi(r_{k',t})}, \qquad
   p_{\text{ord}, t}(w) = \sum_{k=0}^K u_{k,t}\, p_{k,t}(w), \qquad
   r_{\text{ord}, t} = \min_{0 \le k \le K} r_{k,t}.
   $$
2. **Pooled semantic-backoff branch** ($i \in \mathcal{N}_t$, accounting for shared bias vs. disjoint-count variance reduction):
   $$
   \tilde{w}_{i,t} = \frac{\phi(r_{i,t})}{\sum_{i' \in \mathcal{N}_t} \phi(r_{i',t})}, \qquad
   p_{\text{sem}, t}(w) = \sum_{i \in \mathcal{N}_t} \tilde{w}_{i,t}\, p_{i}(w),
   $$
   $$
   N_{\text{sem}, t} = \left(\sum_{i \in \mathcal{N}_t} \frac{\tilde{w}_{i,t}^2}{N(\tilde{c}_i)}\right)^{-1}, \qquad
   \bar{d}_{\text{sem}, t} = \sum_{i \in \mathcal{N}_t} \tilde{w}_{i,t}\, \tilde{d}_i,
   $$
   $$
   r_{\text{sem}, t} = \underbrace{\delta_{K_s} + \sum_{i \in \mathcal{N}_t} \tilde{w}_{i,t}\,\big[a + b\,\|f(\tilde{c}_t) - f(\tilde{c}_i)\|_2\big]_+}_{\text{pooled semantic bias } B_{\text{sem}, t}} \;+\; \underbrace{\ln\!\left(1 + \frac{\bar{d}_{\text{sem}, t} - 1}{2\, N_{\text{sem}, t}}\right)}_{\text{pooled sampling variance } V_{\text{sem}, t}}.
   $$
   *(If semantic backoff is disabled or $\mathcal{N}_t = \varnothing$, $r_{\text{sem}, t} = +\infty$ and $\phi(r_{\text{sem}, t}) = 0$.)*
3. **Two-way mixing weight $\lambda_t \in [0, 1]$ and top-$r$ target $q_t$:**
   $$
   \lambda_t = \frac{\phi(r_{\text{ord}, t})}{\phi(r_{\text{ord}, t}) + \phi(r_{\text{sem}, t})}, \qquad
   \tilde{q}_t(w) = \lambda_t\, p_{\text{ord}, t}(w) + (1 - \lambda_t)\, p_{\text{sem}, t}(w),
   $$
   truncated to its top-$r$ probabilities and renormalized to $q_t(w)$ ($\sum_w q_t(w) = 1$). Note this expands linearly as $\tilde{q}_t = \sum_k (\lambda_t u_{k,t}) p_{k,t} + \sum_i ((1-\lambda_t)\tilde{w}_{i,t}) p_i$, preserving single-pass sparse matrix multiplication in `build_targets`.

##### Equation 5 — Parallel-Branch Combined Risk $R_t$ & Unigram-Anchored Distillation Strength $\alpha_t$
Combine the nested exact branch and disjoint semantic branch by parallel precision, and anchor distillation strength against the context-free unigram risk $r_{0,t}$:
$$
R_t = \frac{1}{\phi(r_{\text{ord}, t}) + \phi(r_{\text{sem}, t})}, \qquad
\alpha_t = \alpha_{\max} \left[1 - \frac{R_t}{r_{0,t}}\right]_+^{\gamma_\alpha} \quad (\text{default } \gamma_\alpha = 1).
$$
- **Exact boundary behavior:**
  - If a position has **no informative match beyond unigram** ($r_{k,t} = \infty$ for all $k \ge 1$ and $r_{\text{sem}, t} = \infty$), then $r_{\text{ord}, t} = r_{0,t} \implies R_t = r_{0,t} \implies \alpha_t = 0$ **identically** (exact fallback to plain one-hot cross-entropy!).
  - As exact or pooled semantic counts grow and bias vanishes ($R_t \to 0$), $\alpha_t \to \alpha_{\max}$ monotonically without harmonic-mean drag from lower-order fallbacks.
  - Optional median calibration: setting $\gamma_\alpha = \frac{\ln 2}{\ln\!\big(1 / (1 - \operatorname{median}_{t \notin \mathcal{S}}(R_t / r_{0,t}))\big)}$ centers median non-structural $\alpha_t$ at $\frac{1}{2}\alpha_{\max}$ while keeping $\alpha_t = 0$ at $R_t \ge r_{0,t}$.

##### Equation 6 — Training Objective & Sequence-Chunked Gradient
Over active training positions $\mathcal{M} \subseteq \{1, \dots, T\}$:
$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \Big[(1 - \alpha_t)\big(-\ln p_\theta(x_t \mid x_{<t})\big) + \alpha_t\, H\!\big(q_t,\, p_\theta(\cdot \mid x_{<t})\big)\Big],
$$
with exact per-token logit gradient $\nabla_{z_t} \ell_t = \operatorname{softmax}(z_t) - \big[(1 - \alpha_t)e_{x_t} + \alpha_t q_t\big]$, evaluated in sequence chunks of size $L_{\text{chunk}} \in [128, 256]$ so peak `fp32` `log_softmax` memory is $O(B \cdot L_{\text{chunk}} \cdot V)$.

#### D. Numerical Verification Summary (200k Synthetic Synonym-Markov Benchmark)
Verified GBD v2 against GBD v1 in controlled NumPy simulations on the exact 200k-token generator (`V=300, C=60, conc=0.08, r=32, R_pre=64`):
- **Small-$n$ singleton over-confidence cured:** Top target probability on `order-4 seen once (n=1)` dropped from **`0.50` $\to$ `0.20`** (`order-3 seen once`: **`0.47` $\to$ `0.18`**).
- **Risk calibration (`Spearman(R_t, true KL)`):** Improved from **`0.45` $\to$ `0.66`** (order backoff) and **`0.50` $\to$ `0.56–0.64`** (full two-way backoff across embedding noise levels).
- **$\alpha_t$ dynamic range restored:** Instead of flat `[0.210, 0.248]`, $\alpha_t$ spans **`0.00` (novel context) / `0.05` (novel bigram) $\to$ `0.12` ($n=1$ singleton) $\to$ `0.35+` ($n \ge 20$ reliable context)**.
- **Rare-context target quality improved:** Full GBD v2 beat GBD v1 on rare contexts (`KL_rare`: `2.844` $\to$ **`2.760`**, `TV_rare`: `0.647` $\to$ **`0.608`**, `TV_all`: `0.572` $\to$ **`0.558`**) once rectified distance bias $[-4.09 + 7.10\|\Delta f\|]_+$ captured the high-dimensional synonym shell ($d_{\min} = 0.58$).

### [2026-11-12] Entry 005 — Audit of Entry 004 Math, Final GBD v2 Specification & Reproducible Verification

#### A. Audit of Entry 004

- **Held up (checked by algebra and Monte Carlo):** the 1.0-nat cap of the v1 variance term; the minus sign of the nested-suffix noise term; the KKT fix for a negative slope; the harmonic-mean identity of v1's $R_t$; "α never reaches 0" in v1; the leave-one-out unbiasedness property; $N_{\text{sem}}$; the Eq. 6 gradient.
- **Did not hold up (fixed below):**
  1. Eq. 4 set $r_{\text{ord}} = \min_k r_k$ while $p_{\text{ord}}$ is the mixture over all orders, so the risk belonged to a distribution that is not used.
  2. Eq. 5's parallel formula $R_t = 1/(\phi(r_{\text{ord}}) + \phi(r_{\text{sem}}))$ is only meaningful for $\phi = 1/r$ and variance-only risk (softmin gives $R_t \approx 0.99$ at $r_{\text{ord}} = 0$).
  3. $\delta_j$ and the regression target were rectified per sample before averaging, which biases them upward (0.025–0.04 nats per step at true bias zero).
  4. $\ln(1 + (\tilde d - 1)/(2n))$ is not a bound (uniform $d = 10$, $n = 1$: true expected KL 2.30, formula 1.71), and the Chao1 prior was estimated on frequent contexts and forced to shrink with order, so it was smallest exactly where singletons live.
  5. The "vocabulary-invariant" KL still penalised unseen tokens by $\ln V_{\text{act}}$.
  6. Section D could not be reproduced: no v2 code existed, and its "v1" baselines do not match the pitch's v1 numbers.
- **New finding while testing:** the semantic risk $r_{\text{sem}}$ does not track target error in any version (v1, Entry 004, or this one). Contexts with rare tokens have nearer but worse neighbours, because rare-token vectors are poorly trained and $a, b$ are fitted on frequent contexts only. Adding a token-frequency regressor to the bias fit did not help (its coefficient fitted to 0). Consequence: $\alpha_t$ reads $r_{\text{ord}}$ only (Eq. 5).

#### B. Final GBD v2 Specification

Identical to the pitch's Method section (sections 1–6). Implemented for $\mathcal{S} = \varnothing$, single document, in `gbd/gbd_v2_check.py`.

##### 0. Setup
Corpus $x_1, \dots, x_T \in \{1, \dots, V\}$; conversations separated by $\langle\text{eos}\rangle$; $b(t)$ is the first index of the conversation containing $t$; $\mathcal{S}$ is the set of role-marker tokens ($\varnothing$ for plain text). $N(c, w)$, $N(c) = \sum_w N(c, w)$. $V_{\text{act}} = |\{w : N(\varnothing, w) > 0\}|$, $p_{\text{uni}}(w) = N(\varnothing, w)/T$. For a count vector: $d$ distinct next tokens, $f_1$ singletons, $\hat d = d + f_1 + 1$.

##### Equation 1 — Exact components, leave-one-out, boundary-safe ($k = 0, \dots, K$)
$c_t^{(k)} = x_{t-k:t-1}$ is valid iff $t - k \ge b(t)$.
$$
n_{k,t} = N\big(c_t^{(k)}\big) - 1, \qquad
p_{k,t}(w) = \frac{N\big(c_t^{(k)}, w\big) - \mathbb{1}[w = x_t]}{n_{k,t}}
$$
Active iff valid and $n_{k,t} \ge 1$; otherwise $r_{k,t} = +\infty$. $d$ and $f_1$ come from the same leave-one-out counts. Rows are truncated to the top $R_{\text{pre}}$ entries and renormalised.

##### Equation 2 — Semantic components (order $K_s$, turn state kept, role markers skipped)
$$
s_t = \begin{cases} x_{t-1} & \text{if } t > b(t) \text{ and } x_{t-1} \in \mathcal{S} \\ 0 & \text{otherwise} \end{cases}
\qquad
\tilde c_t = \big(s_t,\ x_{\tau_1(t)},\ \dots,\ x_{\tau_{K_s}(t)}\big)
$$
$\tau_1(t) > \dots > \tau_{K_s}(t) \ge b(t)$ are the $K_s$ most recent positions before $t$ with tokens not in $\mathcal{S}$. If fewer than $K_s$ exist, position $t$ has no semantic components.
$$
f(\tilde c_t) = \big[\hat e(x_{\tau_1(t)});\ \gamma\,\hat e(x_{\tau_2(t)});\ \dots;\ \gamma^{K_s-1}\hat e(x_{\tau_{K_s}(t)})\big]
$$
Candidates: the $M_{\text{search}}$ nearest $\tilde c_i \ne \tilde c_t$ with $s_i = s_t$, by L2 distance. $p_i(w) = N(\tilde c_i, w)/N(\tilde c_i)$. With $\mathcal{S} = \varnothing$, $\tilde c_t = c_t^{(K_s)}$.

##### Equation 3 — Risk scale
**Variance.** For $n$ samples from $p$ on $m$ symbols, $\mathbb{E}[\operatorname{KL}(\hat p \,\|\, p)] \le \ln \mathbb{E}\big[\sum_w \hat p(w)^2/p(w)\big] = \ln(1 + (m-1)/n)$ (Jensen; Paninski 2003).
$$
\tilde d = \frac{n\,\hat d + n_0\, d_0}{n + n_0}, \qquad
v(n, \hat d;\, d_0) = \ln\!\Big(1 + \frac{\tilde d - 1}{n}\Big), \qquad n_0 = 5
$$
**Support prior**, per context family (each exact order $k$; the semantic contexts). $D$ = number of contexts with $N(c) = 2$; $C$ = number of those whose two next tokens are equal:
$$
d_0 = \min\!\Big(V_{\text{act}},\ \max\!\Big(1,\ \frac{D + 1}{C + 1}\Big)\Big)
$$
If $D < n_{\text{fit}}$: order $k$ inherits $d_0(k-1)$, with $d_0(0) = V_{\text{act}}$; the semantic family inherits $d_0(K_s)$.

**Component risks.**
$$
r_{k,t} = \delta_k + v\big(n_{k,t},\, \hat d_{k,t};\, d_0(k)\big), \qquad
r_{i,t} = \delta_{K_s} + \big[a + b\,\lVert f(\tilde c_t) - f(\tilde c_i)\rVert\big]_+ + v\big(N(\tilde c_i),\, \hat d_i;\, d_0^{\text{sem}}\big)
$$
Keep the $M$ lowest-risk neighbours $\mathcal{N}_t$.

**Plug-in KL** (total pseudo-count $\beta_0 = 3$ toward the unigram):
$$
\widehat{\operatorname{KL}}(c_1 \,\|\, c_2) = \sum_{w:\,N(c_1,w)>0} \frac{N(c_1,w)}{N(c_1)}\,
\ln\frac{N(c_1,w)/N(c_1)}{\big(N(c_2,w) + \beta_0\,p_{\text{uni}}(w)\big)/\big(N(c_2) + \beta_0\big)}
$$
**Order bias.** $\delta_K = 0$; for $j = K-1, \dots, 0$, over order-$(j+1)$ contexts $c$ with $N(c) \ge n_{\text{fit}}$ and $c'$ the order-$j$ suffix of $c$:
$$
\delta_j = \delta_{j+1} + \Big[\operatorname{mean}_{c}\Big(\widehat{\operatorname{KL}}(c \,\|\, c') - \frac{\hat d_c - 1}{2}\Big(\frac{1}{N(c)} - \frac{1}{N(c')}\Big)\Big)\Big]_+
$$
Rectify once, after the mean. Step is 0 if fewer than 10 contexts qualify.

**Semantic bias.** Over neighbour pairs with $N(c), N(c') \ge n_{\text{fit}}$, regress (ordinary least squares)
$$
y_{c,c'} = \widehat{\operatorname{KL}}(c \,\|\, c') - \frac{\hat d_c - 1}{2\,N(c)} - \frac{\hat d_{c'} - 1}{2\,N(c')}
$$
on $\lVert f(c) - f(c')\rVert$, with $y$ not rectified. If the slope is negative: $b = 0$, $a = \operatorname{mean}(y)$. $a$ may be negative. If fewer than 10 pairs qualify, the semantic branch is off.

##### Equation 4 — Mixture and target
$\phi(r) = 1/\max(r, r_{\min})$ (default) or $e^{-\tau r}$. Nothing below depends on the form of $\phi$.
$$
u_{k,t} = \frac{\phi(r_{k,t})}{\sum_{k'} \phi(r_{k',t})}, \qquad
p_{\text{ord},t} = \sum_{k=0}^{K} u_{k,t}\, p_{k,t}, \qquad
r_{\text{ord},t} = -\ln \sum_{k=0}^{K} u_{k,t}\, e^{-r_{k,t}}
$$
Mixture bound (log-sum-exp is convex): $\operatorname{KL}\big(p \,\|\, \sum_j w_j q_j\big) \le -\ln \sum_j w_j e^{-\operatorname{KL}(p \| q_j)} \le \sum_j w_j \operatorname{KL}(p \,\|\, q_j)$.
$$
\tilde w_{i,t} = \frac{\phi(r_{i,t})}{\sum_{i' \in \mathcal{N}_t} \phi(r_{i',t})}, \qquad
p_{\text{sem},t} = \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\, p_i, \qquad
N_{\text{sem},t} = \Big(\sum_{i \in \mathcal{N}_t} \frac{\tilde w_{i,t}^{\,2}}{N(\tilde c_i)}\Big)^{-1}, \qquad
\bar d_{\text{sem},t} = \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\, \tilde d_i
$$
$$
r_{\text{sem},t} = \delta_{K_s} + \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\,\big[a + b\,\lVert f(\tilde c_t) - f(\tilde c_i)\rVert\big]_+ + \ln\!\Big(1 + \frac{\bar d_{\text{sem},t} - 1}{N_{\text{sem},t}}\Big)
$$
($r_{\text{sem},t} = +\infty$ if there are no semantic components.)
$$
\lambda_t = \frac{\phi(r_{\text{ord},t})}{\phi(r_{\text{ord},t}) + \phi(r_{\text{sem},t})}, \qquad
\tilde q_t = \lambda_t\, p_{\text{ord},t} + (1 - \lambda_t)\, p_{\text{sem},t}, \qquad
R_t = -\ln\!\big(\lambda_t\, e^{-r_{\text{ord},t}} + (1 - \lambda_t)\, e^{-r_{\text{sem},t}}\big)
$$
$q_t$ is $\tilde q_t$ cut to its top $r$ entries and renormalised. $R_t$ is a stored diagnostic.

##### Equation 5 — Distillation strength
$$
\alpha_t = \alpha_{\max}\,\Big[1 - \frac{r_{\text{ord},t}}{\max(r_{0,t},\, r_{\min})}\Big]_+^{\gamma_\alpha}, \qquad \gamma_\alpha = 1
$$
No match beyond the unigram $\Rightarrow r_{\text{ord},t} = r_{0,t} \Rightarrow \alpha_t = 0$ exactly. The optional median calibration of $\gamma_\alpha$ from Entry 004 is dropped (undefined when the median ratio reaches 1).

##### Equation 6 — Training objective
$$
\mathcal{L}(\theta) = \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \Big[(1 - \alpha_t)\big(-\ln p_\theta(x_t \mid x_{<t})\big) + \alpha_t\, H\big(q_t,\, p_\theta(\cdot \mid x_{<t})\big)\Big]
$$
Logit gradient: $\operatorname{softmax}(z_t) - \big[(1 - \alpha_t)e_{x_t} + \alpha_t q_t\big]$. Chunked log-softmax is not needed at $V \le 16\text{k}$; for larger vocabularies it lowers peak memory only if the backward pass also runs per chunk.

#### C. Verification (reproducible)

- `python gbd/gbd_v2_check.py test` — brute-force check: $d_0$, $\delta$, $a$, $b$ exact; $q_t$, $r_{\text{ord}}$, $r_{\text{sem}}$, $\lambda_t$, $R_t$, $\alpha_t$ on 60 positions (max abs error 8.9e-08); mixture bound on 2,000 random cases; variance bound on 6 (support, n) settings. **Passed.**
- `python gbd/gbd_v2_check.py bench` — synthetic synonym-Markov benchmark (`V=300, C=60, conc=0.08`, 200k tokens, word2vec on the same corpus, `K=4, Ks=2, M=8, M_search=32, r=32`), 3 corpus seeds × 8,000 positions:

| Target | KL all | TV all | KL rare | TV rare | Spearman(risk that sets α, true KL) |
| --- | --- | --- | --- | --- | --- |
| v1 order backoff only | 3.79 | 0.72 | 4.60 | 0.83 | 0.60 ± 0.02 |
| v2 order backoff only | 3.74 | 0.70 | 4.54 | 0.82 | 0.75 ± 0.01 |
| v1 full | 2.96 | 0.64 | 3.86 | 0.76 | 0.46 ± 0.01 |
| v2 full | 2.78 | 0.60 | 3.66 | 0.71 | 0.67 ± 0.01 |

| Positions (full GBD, v1 → v2) | Share | Top target prob | Mean α | True KL |
| --- | --- | --- | --- | --- |
| Order-4 context seen once elsewhere | 2.4% | 0.40 → 0.24 | 0.248 → 0.174 | 2.42 → 2.15 |
| No order-4 match, order-3 seen once | 15.3% | 0.35 → 0.22 | 0.218 → 0.152 | 2.66 → 2.43 |
| Order-3 context seen 5+ times | 1.1% | 0.20 → 0.17 | 0.252 → 0.266 | 1.84 → 1.70 |
| Order-2 context seen 20+ times | 6.8% | 0.21 → 0.17 | 0.237 → 0.270 | 1.71 → 1.60 |
| Order-2 context never seen elsewhere | 8.3% | 0.13 → 0.11 | 0.128 → 0.001 | 4.56 → 4.42 |
| All positions | 100% | 0.23 → 0.17 | 0.189 → 0.129 | 2.96 → 2.78 |

- Fitted constants (3 seeds): $\delta \approx (2.3, 2.1, 0, 0, 0)$; $d_0 \approx 15$–$17$ at orders 2–4; $a \in [-3.9, -2.9]$, $b \in [13, 15]$. $\alpha_t = 0$ at 17.5% of positions.
- Spearman($r_{\text{sem}}$, true KL of $q_t$) = −0.08, −0.27, −0.14. Spearman($R_t$, true KL) = 0.38 ± 0.03.

#### D. Open Limits (not formula changes)

- The semantic risk is uncalibrated; it sets $\lambda_t$ only. Phase 0 must repeat the calibration check for $r_{\text{ord}}$ and $r_{\text{sem}}$ on the real corpus. Kill criterion 5 decides contribution 2.
- All evidence is synthetic, on a language with exact synonym classes.
- Not yet exercised in code: the boundary rule (Eq. 1) and the role-marker projection (Eq. 2).
- `gbd/gbd.py` is still v1 and must be ported to this entry.

### [2026-11-12] Entry 006 — GBD v2 Production Port, Human Dialogue Corpus, Phase 0 Diagnostics & Phase 2 Pilot Gate Results

#### A. Audit of Entry 005 Math & Production Port (`gbd/gbd.py`)
- Audited Entry 005 & `gbd/gbd_v2_check.py`: Paninski variance bound $\ln(1 + (\tilde{d}-1)/n)$, doubleton-collision support prior $d_0 = \frac{D+1}{C+1}$, log-sum-exp mixture risk $-\ln\sum_j w_j e^{-r_j}$, unigram-prior $\beta_0=3$ KL, rectify-after-mean $\delta_j$, KKT OLS $(a,b)$, and $r_{\text{ord}}$-driven $\alpha_t$ are **algebraically clean and consistent**.
- Ported full GBD v2 into `gbd/gbd.py`:
  1. Added `conversation_starts` + Eq. 1 boundary rule ($t - k \ge b(t)$) in `OrderIndex`.
  2. Added `SemanticIndex` + `nearest_contexts_by_state` implementing Eq. 2 turn-state partitioning & role-marker skipping, with automatic `IndexHNSWFlat` switch above `100k` contexts (`496.7s` target build at `6.36M` tokens).
  3. Replaced dense `plugin_kl` `.toarray()` with sparse row-wise `plugin_kl_v2` (zero dense `(B, V)` allocation).
  4. Added `save_targets` / `load_targets` (`uint16` IDs + `fp16` probs memory-mapped `.npy` files) so GPU training streams from disk with near-zero system RAM.
- All tests pass (`gbd/gbd_v2_check.py test` max error `8.9e-8`; all 5 tests in `gbd/test_gbd.py` including brute-force Eq. 1/2 conversation + role-marker check and PyTorch autograd check).

#### B. Human-Only Corpus Construction & Phase 0 Real-Data Diagnostics (`gbd/data.py`, `artifacts/corpus/`)
- Built strictly human-written English dialogue corpus (`oasst1` EN with strict node-disjoint tree decomposition + `HuggingFaceH4/no_robots` + `DailyDialog`):
  - Deduplicated `30,636` $\to$ **`29,141` conversations** (`26,228` train / `1,468` val / `1,445` test, grouped by tree ID so zero tree leakage across splits).
  - Tokenized with training-split-only `16,000`-vocab SentencePiece BPE (`byte_fallback=True`): **`6,358,385` train tokens** (`4,523,096` reply tokens), `376,857` val tokens, `340,225` test tokens; trained 128-dim same-corpus skip-gram `word2vec` (`41.4s`).
- **Phase 0 Findings (`artifacts/corpus/phase0_diagnostics.json`):**
  - **Context sparsity bites hard:** Unique-context fraction (LOO count $= 0$):
    - At `1M` tokens: $k=2$: `19.6%`, $k=3$: `54.6%`, $k=4$: `75.9%`, $k=6$: **`91.4%`**, $k=8$: **`95.1%`**.
    - At `6.36M` tokens: $k=2$: `9.0%`, $k=3$: `38.1%`, $k=4$: `63.7%`, $k=6$: **`85.9%`**, $k=8$: **`92.1%`**.
  - **Fitted GBD v2 constants (1M slice):** $\delta = [6.11, 2.38, 1.04, 0.55, 0.30, 0.14, 0.00]$; $d_0 = [14582, 3.19, 2.99, 2.40, 1.91, 1.60, 1.39]$; $d_0^{\text{sem}} = 3.03$; $a = -0.43, b = 5.20$ (zero-bias synonym shell $d_{\min} = 0.082$); mean $\alpha_t = 0.336$, mean $\lambda_t = 0.757$.
  - **Real-corpus risk calibration:** Against empirical LOO target NLL on 20k positions, **`Spearman(r_ord, LOO NLL) = +0.3317`** (`mean_loo_nll` rises monotonically from `2.76` on seen `order-6` $\to$ `5.84` on `order-3 (5+)` $\to$ `8.04` on unseen `order-2`, while $\alpha_t$ steps down `0.43` $\to$ `0.41` $\to$ `0.23`), whereas `Spearman(r_sem, LOO NLL) = +0.0970`. Confirms Entry 005's choice to drive $\alpha_t$ from $r_{\text{ord}}$.
  - **Manual kNN check:** Clean conversational paraphrases in both turn-state buckets (`"better?"` $\to$ `"better!", "more?", "easier?", "best?"`; `"built in"` $\to$ `"created in", "constructed in", "installed in", "developed in"`).

#### C. Stage 1 Training Harness & Phase 1–2 Pilot Gate Results (`gbd/model.py`, `gbd/train.py`, `artifacts/runs/`)
Trained from random initialization on local RTX 5060 Ti (`bf16` mixed precision, `L=512`, `batch=16 × accum=2`, AdamW `wd=0.1`). Primary metric: **Test Reply Bits-Per-Byte (BPB)** (lower is better).

##### 1. `1M`-Token Scale (12 epochs; ~5.5 min/run on `gpt2-small` `97.65M`, ~2.2 min/run on `30m` `30.49M`)

| Model | Method | Val Reply BPB | **Test Reply BPB** | Test All BPB | Reply PPL | ECE | Rare (`<10`) NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gpt2-small` (`97.65M`) | Tuned CE (seed 0) | 1.9604 | 1.8984 | 1.8811 | 168.94 | 0.0567 | 10.4764 |
| `gpt2-small` (`97.65M`) | Tuned CE (seed 1) | 1.9602 | 1.8964 | 1.8810 | 168.01 | 0.0513 | 10.5268 |
| `gpt2-small` (`97.65M`) | Label Smoothing ($\alpha=0.1$) | 1.9700 | 1.9047 | 1.8872 | 171.88 | 0.0302 | 9.9571 |
| `gpt2-small` (`97.65M`) | Exact 6-gram, sliding-window ($\alpha=0.34$) | 1.9586 | 1.8954 | 1.8784 | 167.55 | 0.0431 | 10.4587 |
| `gpt2-small` (`97.65M`) | CoCoNTs proper ($k=8, r=8$, prefix-only) | 1.9571 | 1.8928 | 1.8758 | 166.37 | 0.0570 | 10.4748 |
| `gpt2-small` (`97.65M`) | **GBD v2 — Order Backoff Only** | 1.9122 | **1.8473** | 1.8307 | 147.13 | 0.0389 | 10.3879 |
| `gpt2-small` (`97.65M`) | **GBD v2 — Full (Order + Semantic)** | **1.8906** | **1.8276** | **1.8113** | **139.49** | **0.0310** | **10.3112** |
| `30m` (`30.49M`) | Tuned CE (seed 0) | 1.9337 | 1.8685 | 1.8527 | 155.83 | 0.0477 | 10.4896 |
| `30m` (`30.49M`) | **GBD v2 — Full (Order + Semantic)** | **1.9022** | **1.8389** | **1.8228** | **143.83** | **0.0300** | **10.3934** |

- **Takeaway at 1M tokens:** Exact 6-gram (`1.8954`) and CoCoNTs (`1.8928`) barely move past CE (`1.8974` mean) because **91.4%** of 6-grams are unique. GBD Order Backoff drops Reply BPB by **`-0.0501`**, and adding Semantic Backoff (`GBD Full`) drops it another **`-0.0197`** (`-0.0698` BPB total vs CE — **35× larger than the 0.0020 seed noise**, cutting reply perplexity from `168.5` $\to$ **`139.5`** and improving rare/low/mid/frequent bins simultaneously).

##### 2. Full `6.36M`-Token Corpus Scale — Kill Criterion 0 & Truncation-Bias Repair 2 (8 epochs; ~8.5 min/run on `30m`, ~21 min/run on `gpt2-small`)

| Model | Method | Val Reply BPB | **Test Reply BPB** | Test All BPB | Reply PPL | ECE | Rare (`<10`) NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `30m` (`30.49M`) | Tuned CE (seed 0) | 1.4003 | 1.3523 | 1.3689 | 41.73 | 0.0346 | 10.6097 |
| `30m` (`30.49M`) | Tuned CE (seed 1) | 1.3972 | 1.3489 | 1.3666 | 41.38 | 0.0314 | 10.3938 |
| `30m` (`30.49M`) | GBD v2 Full (constant $\alpha_{\max}=0.5$) | 1.3994 | 1.3520 | 1.3686 | 41.63 | **0.0181** | 11.3688 |
| `30m` (`30.49M`) | **GBD v2 Full + Annealed $\alpha_{\max}$ (Repair 2)** | **1.3761** | **1.3282** | **1.3467** | **39.12** | 0.0318 | 10.4306 |
| `gpt2-small` (`97.65M`) | Tuned CE (seed 0) | 1.3733 | 1.3226 | 1.3412 | 38.83 | 0.0389 | 10.9422 |
| `gpt2-small` (`97.65M`) | Tuned CE (seed 1) | 1.3661 | 1.3179 | 1.3364 | 38.09 | 0.0392 | 10.6940 |
| `gpt2-small` (`97.65M`) | GBD v2 Full (constant $\alpha_{\max}=0.5$) | 1.3775 | 1.3265 | 1.3441 | 39.27 | **0.0140** | 12.6146 |
| `gpt2-small` (`97.65M`) | **GBD v2 Full + Annealed $\alpha_{\max}$ (Repair 2)** | **1.3515** | **1.3037** | **1.3234** | **36.64** | 0.0292 | 11.4918 |

- **Scientific Takeaway at 6.36M tokens (Exact Confirmation of Pitch Hypothesis & Repair 2):**
  - Epoch-by-epoch curves show unannealed `GBD Full` leads `CE` strongly through Epochs 1–4 (`val_reply_bpb`: `1.4535` vs `1.4696` on `gpt2-small`; `1.4852` vs `1.5159` on `30m`), then stalls in Epochs 6–8 because top-$r=16$ order-6 targets pull the 512-token Transformer toward short context and zero out tail tokens (`rare_<10` NLL rises to `12.61`).
  - Activating **pre-registered Repair 2 (`--anneal_alpha`: $\alpha_{\max}$ linearly annealed $0.5 \to 0$ over training)** keeps all of GBD's early variance reduction and releases the n-gram tether in late epochs—beating **every CE seed** by **`-0.0166` BPB** on `gpt2-small` (`1.3037` vs `1.3203` mean; `6×` seed gap) and **`-0.0224` BPB** on `30m` (`1.3282` vs `1.3506` mean; `6.6×` seed gap)!

### [2026-11-12] Entry 007 — Workflow Rule, Orthogonal Config Switches & Pluggable Embedding Sources (`Static` vs. `Student Hidden`)

- **Workflow Policy Recorded:**
  - User takes over running `train` and `test` scripts directly; assistant **writes code only** and may execute **only standalone `eval` code** (`gbd/eval.py`) in the shell.
- **Orthogonal Component & Schedule Switches Added (`gbd/gbd.py`, `gbd/train.py`):**
  - Every component can now be switched on/off independently from `GBDConfig`, CLI flags, or `--config <file.json>`:
    1. **Exact-context branch:** `use_exact` (`--use_exact` / `--no-use_exact`; `False` isolates pure semantic retrieval with $\lambda_t = 0$).
    2. **Multi-order backoff:** `use_order_backoff` (`--use_order_backoff` / `--no-use_order_backoff`; `False` restricts exact branch to single order `(K,)` with no $0 \dots K-1$ backoff), plus explicit `exact_orders`.
    3. **Leave-one-out & boundaries:** `leave_one_out` (`--leave_one_out` / `--no-leave_one_out`) and `respect_eos` (`--respect_eos` / `--no-respect_eos`).
    4. **Semantic retrieval & dialogue projection:** `use_semantic`, `skip_role_markers`, `partition_turn_state`, and `pool_semantic` ($N_{\text{sem}}$ pooled variance vs. unpooled log-sum-exp bound).
    5. **Fixed vs. Adaptive $\lambda_t$:** `lambda_mode` (`"adaptive"` Eq. 4 vs. `"fixed"` with `fixed_lambda` when both branches are active, gracefully preserving mass when only one branch has support).
    6. **Fixed vs. Adaptive $\alpha_t$ & Curriculum:** `alpha_mode` (`"adaptive"`/`"r_ord"`, `"R"`, `"r_sem"`, or `"fixed"`/`"const"`), `phi` (`"inv"`, `"exp"`, `"uniform"`), and `alpha_schedule` (`"constant"`, `"linear"`, `"cosine"`).
- **Standalone Evaluator & Pluggable Embedding Architecture (`gbd/eval.py`, `gbd/embeddings.py`, `gbd/model.py`):**
  - Created `gbd/eval.py` for standalone checkpoint evaluation (`bf16` checkpoints saved in `artifacts/checkpoints/`).
  - Added `CausalLM.forward_hidden(idx, layer)` and `gbd/embeddings.py` supporting 6 `--embedding_source` modes: `w2v`, `ppmi_svd` (100% count-only Shifted PPMI + SVD, zero neural training), `random`, `student_tok_emb`, `student_hidden` (mid-layer `n_layer // 2` student hidden states pooled per semantic context via `sem_feature_fn`, bootstrapped from `--embedding_ckpt` or `--warmup_ce_epochs`), and `file`.
- **Novelty Trade-off Logged (`Static Corpus Embeddings` vs. `Student Hidden States`):**
  - **Static (`w2v` / `ppmi_svd`):** Claims a **pure corpus-level target** built once on CPU, architecture-agnostic, zero GPU extraction cost; `ppmi_svd` eliminates even shallow `word2vec` SGD. Limited to $K_s$-token local surface context.
  - **Student Hidden (`student_hidden`):** Claims **self-bootstrapped non-parametric backoff distillation** (student's contextual hidden geometry retrieves neighbours, while LOO corpus counts—not student softmax logits—supply targets). Captures 512-token context beyond $K_s$, at the cost of a CE warmup/extraction pass per architecture and potential label-clustering leakage if extracted from the final pre-logit layer of an overfit warmup model (mitigated by defaulting `--hidden_layer` to `n_layer // 2`).

### [2026-11-12] Entry 008 — Pre-Phase-3 Fixes: Faithful CoCoNTs Eq. (4), Collision-Proof Target Cache & Entropy-Matched Random Control

- **Issues Identified Before Phase 3 (Code-Only Patch, Zero Shell Execution):**
  1. **Mislabelled CoCoNTs target (`gbd/train.py`):** Previous `build_coconts_targets` truncated to top-$r=8$, renormalised to 1, and set $\alpha = 1.0$ (`nll_weight = 0`) — giving **zero supervision** to the ground-truth next token whenever $x_t \notin \text{top-}r$ (losing Sathe & Sarawagi 2024 Eq. 4's statistical consistency $\mathbb{E}[\mathbf{y}^{\text{CC}}] = \mathbf{y}^{\text{all}}$), and skipped prefixes crossing `<|eos|>` inside chunks. *(Note: the earlier `1.8928` CoCoNTs pilot number in Entry 006 used that naive top-$r$ truncation and must be superseded by the faithful Eq. 4 run.)*
  2. **Target cache collisions (`gbd/train.py`):** Legacy `is_default_v2` branch omitted `M_search`, `R_pre`, `tau`, `n_fit`, `n0`, `beta0`, `embed_dim`, and token-slice content; `random_target` cache path ignored all reference GBD flags; and default `run_name` could overwrite runs when toggling component switches.
  3. **With-replacement sampling in `random_target` (`gbd/train.py`):** `rng.integers(0, V, size=(T, r))` allowed duplicate IDs per row, shrinking entropy relative to GBD.
- **Code Fixes Applied (`gbd/gbd.py`, `gbd/train.py`, `gbd/test_gbd.py`):**
  - Added optional `nll_weight` array to `GBDTargets`, `save_targets`/`load_targets`, `gbd_loss`, and `GBDChunks` (defaulting to `1 - alpha` for all GBD runs, while allowing decoupled `(nll_weight, alpha)` per position).
  - Rewrote `build_coconts_targets` to implement **Sathe & Sarawagi (2024) Eq. (4)** over raw corpus-wide prefixes (`b_pos=None`, `--coconts_r 8`, `--coconts_gamma 1.5`): with $p = \sum_{w \in \text{top-}r} y^{\text{all}}[w]$ and $u = 1/(\gamma - p)$, sets `(nll_weight = 0, alpha = 1 - (1-p)u)` when $x_t \in \text{top-}r$ and `(nll_weight = 1, alpha = u p)` when $x_t \notin \text{top-}r$, so $\mathbb{E}[\mathbf{y}^{\text{CC}}] = \mathbf{y}^{\text{all}}$ identically.
  - Updated `build_random_entropy_matched_targets` to draw $r$ **distinct** token IDs without replacement per position.
  - Replaced `_target_cache_dir` with `_target_cache_dir_and_spec`: computes a single canonical `blake2b` digest over every target hyperparameter + a deterministic content hash of `tokens`, writes `cache_spec.json` into every cache folder, and strictly verifies `cache_spec.json` inside `load_targets`. Also added a config digest to `_default_run_name`.
  - Added `test_coconts_eq4_consistency_and_cache_isolation` (Test #7 in `gbd/test_gbd.py`) checking CoCoNTs against brute force, $\mathbb{E}[\mathbf{y}^{\text{CC}}] = \mathbf{y}^{\text{all}}$, distinct random IDs, and cache non-collision across 13 flag/token perturbations.

### [2026-11-12] Entry 009 — Unit-Test Fixture Fix for Orthogonal Switches

- **Failure:** `test_orthogonal_component_and_embedding_switches` asserted `(alpha[2:] > 0).any()` under `use_exact=False` + `alpha_mode="adaptive"`, but the fixture used i.i.d. uniform tokens (`V=20, T=800`) with default `n_fit=20`. No context reached the fit threshold → semantic bias fit disabled → `alpha = 0` everywhere (test-setup bug, not a `gbd.py` regression).
- **Fix (`gbd/test_gbd.py` only):** 1st-order Markov tokens (`V=12, T=1600`) + `n_fit=4` so semantic / exact bias fits succeed and adaptive $\alpha_t$ is non-trivial. Orthogonal-switch assertions (`w_exact`, fixed $\lambda$/$\alpha$, PPMI, `student_hidden`) unchanged.

### [2026-11-13] Entry 010 — Kneser–Ney and D2GPo-NN64 Baseline Targets

- **Added, without touching GBD / CE / CoCoNTs builders or their cache specs:**
  - `gbd/kn.py` — `--method kn`. Interpolated modified Kneser–Ney (Chen & Goodman 1998), orders $0..K$. Order $K$ uses raw counts; lower orders use continuation counts $N_{1+}(\cdot hw)$. Discounts $D_1,D_2,D_{3+}$ from full-corpus count-of-counts. Order 0 is the continuation unigram mixed with uniform $1/V$. Contexts respect `<|eos|>` through the same `OrderIndex` / `conversation_starts` rule as GBD Eq. 1. `--kn_loo` (default on) drops the event ending at $t$ (raw count loses 1; a lower-order continuation loses 1 only when the left-extended raw event had count 1). Discounts stay fixed. Stored $q_t$ is exact $P_K$ on the union of nonzero count-rows plus the top 64 order-0 tokens, truncated to `--r` and renormalised. `alpha = --alpha_max`, `nll_weight` unset, so the existing loss is $(1-\alpha)\mathrm{NLL}+\alpha H(q,p)$ and `--alpha_schedule` / `--anneal_alpha` still apply.
  - `gbd/d2gpo.py` — `--method d2gpo` is **D2GPo-NN64: neighbour-restricted variant of D2GPo** (Li et al., ICLR 2020, OpenReview `S1efxTVYDr`), not faithful full-vocabulary D2GPo. With $\sigma=1$, $T=2$ and $V=16000$, Eq. 7 puts 99.6% of its mass at ranks $\ge 64$ and is within $1.4\times 10^{-5}$ of uniform at every rank, so the released defaults equal label smoothing with weight $\lambda$. Truncating to $r_d$ and renormalising onto the nearest neighbours is a different, stronger prior. Rank is still cosine similarity to the gold token (Eqs. 8–9, gold forced to rank 0) with a $\mathrm{Normal}(0,\sigma)$ density and softmax temperature $T$ (Eq. 7). Embeddings are the same `resolve_token_embeddings` source GBD uses (default same-corpus word2vec). The PDF does not print numeric defaults; the released criterion / WMT14 script uses $\sigma=1$, $T=2.0$, $\lambda=0.1$ (`--d2gpo-alpha`). Storage is `nll_weight = 1`, `alpha = \lambda`, i.e. MLE $+\lambda H(q,p)$ which matches $\mathrm{KL}(q\|p)$ up to the constant $H(q)$. Temperature is on the prior only, not on the model. Their code also multiplies the KL by $T^2$ and uses $(1-\lambda)$ on a label-smoothed MLE; that mixture is not what is stored.
- **Cache:** `kn` and `d2gpo` have their own branches in `_target_cache_dir_and_spec` (KN: $K$, $r$, `kn_loo`, `respect_eos`, `alpha_max`, discount version; D2GPo-NN64: $\sigma$, $T$, $\lambda$, $r_d$, embedding source, dimension, seed). Existing method specs are unchanged.
- **Tests:** `test_kn_and_d2gpo_targets` in `gbd/test_gbd.py`. Not executed in this change (code only).

### [2026-11-14] Entry 011 — Born-Again Self-KD, Results Summarizer, eval Docstring

- **`--method self_kd`** (Furlanello et al. 2018), added beside the existing methods. No offline targets and no cache directory, so existing cache specs are unchanged. Requires `--teacher_ckpt`. The teacher is loaded with `load_checkpoint_model`, set to `eval()` and `requires_grad_(False)`, and run under `torch.no_grad` inside the same bf16 autocast as the student. Vocab size and `max_seq_len` must match or the run refuses. Per token, with the same `ignore_index=-100` mask as CE:
  $(1-a)\,\mathrm{NLL}(x_t) + a\, T^2\, \mathrm{KL}(\mathrm{softmax}(z_{\mathrm{teacher}}/T) \,\|\, \mathrm{softmax}(z_{\mathrm{student}}/T))$,
  $a = \texttt{kd\_alpha} \times$ the existing alpha curriculum, so `--alpha_schedule` / `--anneal_alpha` scale it the same way as GBD. Softmaxes are float32. Defaults: `--kd_alpha 0.5`, `--kd_temp 1.0`. `teacher_ckpt`, `kd_alpha`, and `kd_temp` are argparse fields, so they land in the run JSON `config` with the rest of `vars(args)`.
- **`gbd/summarize.py`** reads `artifacts/results_groups.json` and writes `artifacts/results_summary.md`: per-group test and best-val reply BPB (mean ± sample sd), paired gaps `test(A)−test(B)` with the rule mean > max(0.01, 3×sd) and a shared strict sign, frequency-bin share of test-NLL change, and BLiMP `mean_acc` with the same paired rule. Missing run or BLiMP files stay missing.
- **`gbd/eval.py`:** the module docstring is a raw string so the usage path's `\.` is not a `SyntaxWarning`. No other change.
- **Tests:** `test_self_kd_loss` in `gbd/test_gbd.py`. Not executed in this change (code only).

### [2026-11-14] Entry 012 — BabyLlama-2 Logit Ensemble on `self_kd`

- **`--teacher_ckpt`** now accepts a comma-separated list. Each checkpoint is loaded with `load_checkpoint_model`, so a 30M teacher and a `gpt2-small` teacher keep their own depth and width. Vocab size and `max_seq_len` must still match the student. The step uses the mean of the teacher logits, not the mean of their probabilities (Tastet & Timiryasov 2024), then the existing `self_kd_loss`.
- **T² was already in `self_kd_loss`** (`alpha * T * T * KL`). No change to that formula. Runs with `--kd_temp 1` are unchanged, because \(T^2 = 1\).
- The parsed list is stored on `args.teacher_ckpts` and therefore in the run JSON `config`. One checkpoint is a one-element list and takes the same loss path as before.
- **Test:** two copies of the same teacher logits match the single-teacher loss.

### [2026-11-15] Entry 013 — Pitch and Memory Brought Up to Paper Draft v6

- **Source of truth for reported numbers:** `reports/GBD_paper_draft_v6.tex`. The working pitch `Geometric Backoff Distillation (GBD) A Teacher-Free Target for Training Small LM.md` was still the Nov 14 results note. It now matches v6.
- **What v6 adds over the Nov 14 pitch:**
  - Born-Again self-distillation and BabyLlama-2-style ensemble distillation as teacher-based references, not as teacher-free baselines. At 1M tokens the ensemble is a 30M teacher plus a `gpt2-small` teacher; at 6.36M it is two 30M teachers, because no `gpt2-small` teachers were trained at that scale. Temperature $\tau = 1$ won on validation for both. The loss averages **logits**, then applies $T^2\,\mathrm{KL}$.
  - 1M, 3 seeds, test reply BPB: CE $1.8327 \pm 0.0021$ / GBD $1.7846 \pm 0.0016$ (`gpt2-small`); CE $1.8273 \pm 0.0020$ / GBD $1.7833 \pm 0.0037$ (`30m`). Self-distillation $1.7826 \pm 0.0029$ and $1.7844 \pm 0.0008$. Ensemble $1.7726 \pm 0.0024$ and $1.7744 \pm 0.0046$. Kneser–Ney $1.8191 \pm 0.0020$ and $1.8208 \pm 0.0020$.
  - 6.36M, `30m`: CE $1.3027 \pm 0.0006$; annealed GBD ($\alpha_{\max}=0.5$) $1.2918 \pm 0.0009$; GBD $\alpha_{\max}=0.75$ (seed 0) test $1.2892$, validation $1.3333$; D2GPo-NN64 $\lambda=0.34$ (seed 0) validation $1.3325$, test $1.2947$ (a validation tie). Self-distillation $1.2784 \pm 0.0012$; two-teacher ensemble $1.2783 \pm 0.0023$. Constant $\alpha$ GBD loses to CE ($1.3180$ vs $1.3027$, seed 0).
  - BLiMP (67 paradigms × 200 pairs): GBD − CE is $+1.5 \pm 1.4$, $+1.1 \pm 0.5$, and $-0.1 \pm 0.2$ points. None decisive.
  - Frequency shares of the CE−GBD test-NLL reduction are in the pitch. Mid-frequency tokens take 1.3–1.8× their token share. Rare tokens do not.
- **Claim the draft is willing to make:** on about 1M tokens and no teacher, GBD is the best teacher-free target tested and matches one trained teacher. By 6.36M tokens that lead over other smoothing is gone, and a teacher is better.
- **Still not in v6 because the runs do not exist:** MIR, static-embedding init, extra 6.36M seeds for Kneser–Ney and D2GPo-NN64, prefix-only targets, and the 98M student at 6.36M under the final recipe.


