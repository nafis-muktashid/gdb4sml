# Geometric Backoff Distillation (GBD): A Teacher-Free Target for Training Small LMs

Nov 10, 2026 · revised Nov 15, 2026 · @Nafis

> **Status as of Nov 15 (paper draft v6).** The method is implemented (version 2, `gbd/gbd.py`) and has been trained on a human-written English dialogue corpus of 6.36M tokens, with 30M and 98M students trained from scratch. The paper draft is `reports/GBD_paper_draft_v6.tex`. In short:
>
> - **1M tokens:** GBD beats tuned cross-entropy by 0.044–0.048 reply bits per byte (3 seeds, both model sizes) and beats every teacher-free baseline by at least 0.032. It ties Born-Again self-distillation and recovers 80–83% of a two-teacher ensemble's gain, with no teacher of its own.
> - **6.36M tokens:** annealed GBD still beats CE by 0.011 (3 seeds), but the other annealed soft targets catch up. Trained teachers beat GBD by 0.013 on every seed.
> - **The gain is in mid-frequency tokens, not rare ones.**
> - **BLiMP:** GBD is 1–1.5 points above CE at 1M tokens (not decisive) and ties CE at 6.36M.
> - **Not run yet:** masked-input regularisation and static-embedding initialisation. PS-KD is not the self-distillation that was run; Born-Again and the BabyLlama-2-style ensemble were.
>
> The pre-registered claim and kill criteria are left as written, and each now carries its outcome.

## The pitch

GBD replaces the teacher network in knowledge distillation with a next-token distribution estimated from the training corpus itself. Where a context is too rare to have useful counts, the estimate backs off in two directions at once: to shorter contexts, and to contexts that are close in a static embedding space. Every target carries an estimated KL risk, and that risk decides both how the two backoffs are mixed and how hard each token is distilled. No pretrained model appears anywhere in the pipeline. The student is a small decoder LM trained from random initialisation, for conversation reply, entirely on local hardware. The course's reference scale is about 30M parameters, and both of the students trained so far (30.5M and 97.7M parameters) were trained on one consumer GPU.

**Contributions**

1. **Two-way backoff target for long contexts.** Prior n-gram targets (CoCoNTs, NDP) use exact matches only, and apply them only where the n-gram is the model's whole visible context: the first k positions of a training block in CoCoNTs, whole-prefix keys in NDP. Semantic Smoothing (2026) pools across similar contexts but only at bigram order, and only for count models. GBD combines order backoff and semantic backoff for arbitrary order and uses the result as a training target for a neural LM at every position. *Evidence (1M tokens, 98M student, locked recipe, 3 seeds):* order backoff beats leave-one-out Kneser–Ney by 0.013 reply BPB (1.806 vs 1.819), and the semantic branch adds 0.021 on top (1.785). Random vectors keep only 0.003 of that semantic gain (seed 0). Not yet ablated at 6.36M tokens.
2. **Risk-calibrated distillation.** A per-token bias–variance estimate of KL risk sets the mixing weight λ\_t between the two backoffs and the distillation strength α\_t. Reliable targets are distilled hard. Unreliable ones fall back to plain cross-entropy. Neither prior method weights tokens by target reliability: CoCoNTs uses a plain KL term with no weight, and NDP replaces the one-hot label outright. This contribution is conditional. Version 2 of the risk scale repairs the order-backoff risk, which now tracks target error in the synthetic check. The semantic risk does not, so α\_t is set from the order-backoff risk alone (see the version 2 check). The contribution is claimed only if that holds on the real corpus. *Evidence so far:* on the real corpus, Spearman(r\_ord, leave-one-out target NLL) = 0.33, against 0.10 for r\_sem, and α\_t falls monotonically across context groups as target NLL rises. Removing every use of risk (equal weights, fixed λ, constant α) costs 0.014 bits per byte at 1M tokens (3 seeds each). Replacing the risk weights φ(r) with equal weights costs 0.008 by itself (1 seed). Per-token α\_t beats constant α by 0.005 at the final recipe but not at the pilot recipe (1 seed each). A fixed λ costs nothing measurable. Under kill criterion 5, per-token α\_t is therefore off the contribution list until 3-seed runs show it. What stands is the narrower claim that risk-weighted mixing helps.
3. **Leave-one-out targets.** The target at position t is built only from other occurrences of its context. On a tiny corpus, most long contexts occur once, so the usual target is the training label in disguise. Leave-one-out removes that leak. *Evidence so far:* leaving the label out improves test reply bits per byte by 0.018 at the final recipe (1.789 vs 1.807, 1 seed each) and by 0.028 at the pilot recipe. This holds even though leave-one-out makes the target itself slightly less accurate on synthetic data.
4. **One family, built-in ablations.** Label smoothing, unigram targets and sliding-window exact n-gram targets are special cases of the GBD objective, so those baselines are GBD with one component switched off. CoCoNTs proper, Kneser–Ney and D2GPo are not special cases and run as separate baselines.

**Central claim to test.** On human-written English conversation data of at most about 20M tokens, with students trained from scratch, GBD beats tuned cross-entropy, label smoothing, sliding-window exact n-gram targets at matched α, and prefix-only CoCoNTs on held-out bits per byte of replies, and the gain concentrates in rare tokens. If it does not, the kill criteria in the experimental plan say what we drop. A small pilot (phase 2) gates the full plan, because applying an n-gram target at every position is the untested part of the design.

**Status of the central claim (Nov 15, paper draft v6).** The corpus that met the human-written-only rule has 6.36M training tokens, so 6.36M is the top data scale. Against the claim's four parts:

- **Beats tuned CE:** holds at 1M tokens (both model sizes, gap 0.044–0.048) and at 6.36M tokens with annealed α\_max (30M model, gap 0.011; 3 seeds each). Without annealing, GBD loses to CE at 6.36M. This is the truncation bias the plan predicted.
- **Beats label smoothing, exact n-gram targets and CoCoNTs:** holds at 1M tokens, where GBD leads every teacher-free baseline by at least 0.032. Exact n-gram and CoCoNTs were run only at the pilot recipe, where they matched CE and GBD led them by about 0.06. At 6.36M, every annealed soft target improves on CE by 0.008–0.011, and the best of each lands within 0.006. Selected on validation, GBD at α\_max = 0.75 and D2GPo-NN64 at λ = 0.34 are a tie (validation 1.3333 vs 1.3325).
- **Against teachers, which the original claim did not include:** at 1M tokens GBD ties Born-Again self-distillation and recovers 80–83% of a two-teacher ensemble's gain over CE. At 6.36M, self-distillation and a two-teacher ensemble of 30M models both beat GBD by 0.013 on every seed.
- **Gain concentrates in rare tokens:** not supported. The gain is in mid-frequency tokens. At 6.36M, rare tokens (seen fewer than 10 times) take a negative share of the NLL reduction.

What the paper claims: on about a million tokens, with no teacher, GBD is the strongest teacher-free target tested and matches a trained teacher. By 6.36M tokens the teacher-free advantage over simple smoothing is gone, and a trained teacher is the better target.

## The problem

The course asks for a source of soft next-token targets that needs no pretrained teacher and beats plain cross-entropy on a small dataset. Exact corpus n-gram targets come close to that bar on mid-size data: CoCoNTs improves fine-tuning perplexity on WikiText-103 (about 114M tokens) and, after from-scratch pretraining on about 100M words, improves CoLA. Neither CoCoNTs nor NDP reports from-scratch perplexity, and neither is tested on a corpus of 20M tokens or less. That small-data setting, simulated here with subsamples of an English conversation corpus, is where exact targets are expected to fail.

**Why a teacher-free target can work at all.** In data-limited LM training, distillation helps even when the teacher saw no extra data. Baby Llama distilled same-corpus teachers on 10M words into a 58M student that beat both teachers. Tf-KD showed that KD is a learned form of label smoothing. A large share of the KD gain therefore comes from the structure of the soft target, not from knowledge the teacher brought in. A target built from the corpus can supply that structure.

**Why exact n-gram targets break on tiny corpora.** An order-k target at position t is the empirical distribution of tokens that followed the same k-token context elsewhere in the corpus. As the corpus shrinks, most long contexts occur exactly once. The "distribution" for a unique context is then the one-hot label at t itself, so the soft target carries no information and KD collapses to plain cross-entropy. CoCoNTs reports the same symptom from the other side: small n-gram indices make the target over-confident and hurt perplexity.

**What is missing.** Backing off to shorter contexts restores counts but blurs meaning. Backing off to similar contexts keeps meaning but adds bias. Semantic Smoothing (2026) proves when the second kind is safe: under a Lipschitz-logit model, KL between next-token distributions is bounded by embedding distance. No one has combined both backoffs on one risk scale, or used the result to train a neural LM.

**The main design risk: the target sees at most K tokens, the student sees the whole context.** CoCoNTs and NDP apply their targets only where the n-gram is everything the model can see, so the target is an unbiased estimate of what the model should output. GBD applies its target at every position, where the student conditions on up to 512 tokens and the target on at most K. With weight α\_t the loss is minimised by a blend of the full-context distribution and an n-gram estimate, so GBD trades bias toward an n-gram model for lower variance. That trade should pay when the corpus is so small that the student cannot use long context anyway, and should stop paying as the corpus grows. Prior evidence is cautionary. Frydenlund et al. argue a pretrained n-gram model is too weak a teacher to match directly. Edman et al. found that trigram soft labels helped a low-resource Transformer less than shortening its context did (masked LM, not causal). Every risk is measured against the order-K conditional, so α\_t cannot see this bias. The pilot in phase 2 measured it directly at 1M and 6.36M tokens. The bias is real at 6.36M (see Results).

**First measurement, before any training (done in phase 0).** Share of positions whose k-token context occurs nowhere else in the training split, so that a leave-one-out exact target does not exist:

| Training tokens | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 | k = 8 |
| --- | --- | --- | --- | --- | --- | --- |
| 1M | 19.6% | 54.6% | 75.9% | 86.4% | 91.4% | 95.1% |
| 5M | 10.1% | 40.3% | 65.6% | 79.5% | 86.8% | 92.5% |
| 6.36M (full) | 9.0% | 38.1% | 63.7% | 78.3% | 85.9% | 92.1% |

At K = 6, exact targets are missing at 86–91% of positions at every scale we have, so sparsity bites throughout. K = 6 and Kₛ = 2 were kept.

## Background and positioning

GBD sits where two lines of work meet without touching. One line uses same-corpus n-gram counts as soft training targets. The other proves that embedding proximity bounds the KL distance between next-token distributions. The table lists what each closest paper does and what GBD takes from it or adds.

| Work | What it does | Gap GBD addresses |
| --- | --- | --- |
| [Hinton et al. 2015](https://arxiv.org/abs/1503.02531) | Student matches temperature-softened teacher distributions: CE + α·KL. | GBD keeps this loss and replaces the teacher. |
| [Yuan et al. 2020, Tf-KD](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.html) | Shows KD is learned label smoothing; uses a hand-designed, class-agnostic virtual teacher. | GBD's target is context-specific and token-specific. |
| [Timiryasov & Tastet 2023, Baby Llama](https://arxiv.org/abs/2308.02019) | Same-corpus teachers on 10M words improve a 58M student. | Motivation: gains need no outside knowledge. |
| [Sathe & Sarawagi 2024, CoCoNTs](https://aclanthology.org/2024.findings-acl.717/) | On the first k positions of each fixed-length block, replaces next-token loss with KL to the corpus n-gram distribution (default k = 8, top r = 8); plain next-token loss on the rest. No weight on the KL term. Targets are stored with the corpus. Perplexity results are fine-tuning of pretrained 125M to 1.3B checkpoints on WikiText-103 (gpt2-125m: 24.28 to 20.72). From scratch on BabyLM strict, about 100M words (opt-125m), it reports downstream scores only: CoLA MCC 0.339 vs 0.288, with MRPC and RTE closer. The gain shows at all 256 positions although the loss touches only the prefix, and it shrinks slightly with longer training. Small indices make targets over-confident. | Exact match only; applied only at block-initial positions; no per-token weight; target includes the label's own occurrence. GBD backs off, applies at every position, weights by risk and leaves the label out. |
| [Ruan et al. 2024, NDP](https://arxiv.org/html/2408.17377) | Replaces the one-hot label with the distribution of tokens that follow the same prefix, counted from the start of the sequence, with a preset share of mass on unseen tokens. No online cost. Tested by fine-tuning pretrained models (Gemma-2B, LLaMA3-8B, T5, Qwen2-7B); its only from-scratch run is a synthetic convergence study. | Not evidence for from-scratch training on natural text. Exact match only, label included, no backoff. With no repeated prefixes it reduces to plain next-token training, as the paper notes. |
| [Malagutti et al. 2024](https://aclanthology.org/2024.naacl-long.382) | Proves label smoothing equals add-λ smoothing; turns any n-gram smoother into a neural-LM regulariser. | GBD is a new smoother in this frame, with semantic backoff and per-token weights. |
| [Balasundaram et al. 2026, Semantic Smoothing](https://arxiv.org/abs/2605.07994) | KL(P(·\|c) ‖ P(·\|c̃)) ≤ 2L‖f\_c − f\_c̃‖ + 4ε. Interpolation estimator with risk O(min{Δ, d/n}); weights ∝ φ(risk proxy). Bigram count models only. | GBD extends it to long contexts, puts order backoff and semantic backoff on one risk scale, and trains a neural LM with it. |
| [Li et al. 2020, D2GPo](https://iclr.cc/virtual/2020/poster/1925) | Spreads mass to tokens whose embeddings are near the label (target-side similarity). | CoCoNTs beats it. GBD uses context-side similarity instead. |
| [Liu et al. 2024, Infini-gram](https://arxiv.org/abs/2401.17377) | Suffix-array counts with longest-suffix backoff. | Data structure option for the exact component. |
| [Frydenlund et al. 2022, LM via Learning to Rank](https://arxiv.org/abs/2110.06961) | Uses n-gram statistics as a teacher at every position. Argues that matching a pretrained n-gram model's distribution directly gives a weak teacher, so it distils ranks instead; on par with BERT and Born-Again teachers. | The closest prior art for every-position n-gram targets, and a warning about them. GBD must show its risk-weighted mixture avoids the weak-teacher problem. |
| [Edman et al. 2021](https://arxiv.org/abs/2205.04810) | With 10k sentences a trigram model beats a Transformer on masked-word accuracy. Trigram soft labels help the Transformer, but less than shortening its context. | Supports n-gram targets in the tiny-corpus regime, and suggests the gain may be a locality effect. Masked LM, not causal. |
| [Sarnthein et al. 2023, Random Teachers](https://arxiv.org/abs/2302.12091) | A random teacher still helps; some KD gain needs no information. | Requires a random-target control in the experiments. |
| [Xu et al. 2026, MIR](https://arxiv.org/abs/2606.06888); [Kim et al. 2026](https://arxiv.org/abs/2509.14786) | Strong weight decay and masked-input regularisation make data-constrained CE much stronger. | Our CE baseline must be tuned this hard. |

**What GBD does not claim.** The smoothing idea is Balasundaram et al.'s, and n-gram targets are CoCoNTs' and NDP's. The novelty is the joint two-way estimator, the risk-calibrated per-token weights, leave-one-out construction, and the test in the regime where exact targets collapse. Applying an n-gram target at every position is not new either (Frydenlund et al.). It is the riskiest design choice, not a contribution.

## Method

The GBD target at position t mixes two branches: exact-context estimates at every order, and a pooled estimate from semantically near contexts. Every component carries an estimated KL risk against the order-K conditional, as bias plus sampling variance. The equations below are version 2 of the risk scale (Nov 12). `gbd/gbd.py` implements them, conversation boundaries and role markers included, and is the code used for every result in this document. `gbd/gbd_v2_check.py` is an independent single-document reference, and the production code matches it (see Reference implementation). Version 1 is kept in `gbd.py` behind `risk_version=1` for comparison only.

### Notation

Corpus x₁ … x\_T over vocabulary V. Conversations are separated by an end-of-sequence token, and b(t) is the index of the first token of the conversation that contains position t. 𝒮 is the set of role-marker tokens (empty for plain text). The order-k context at t is c\_t^(k) = x\_{t−k:t−1}. N(c, w) is the number of positions where context c is followed by w, and N(c) = Σ\_w N(c, w). V\_act is the number of token types that occur in the corpus, and p\_uni(w) is the unigram frequency of w. For a count vector, d is the number of distinct next tokens, f₁ the number seen exactly once, and d̂ = d + f₁ + 1.

### 1. Exact components, leave-one-out (orders k = 0 … K)

An order-k context is valid only if it stays inside one conversation, that is t − k ≥ b(t). Position t is removed from its own counts:

```latex
n_{k,t} = N\big(c_t^{(k)}\big) - 1, \qquad
p_{k,t}(w) = \frac{N\big(c_t^{(k)}, w\big) - \mathbb{1}[w = x_t]}{n_{k,t}}
```

A component exists only if its context is valid and n\_{k,t} ≥ 1. Otherwise its risk is +∞ and its weight is zero. For a unique context it vanishes instead of turning into the label. d and f₁ are taken from the same leave-one-out counts. Each component is truncated to its top R\_pre tokens and renormalised before mixing.

### 2. Semantic components (order Kₛ contexts)

The semantic context skips role markers but remembers the turn state, so a role marker neither hides the user's turn nor merges a reply opening with other positions:

```latex
s_t = \begin{cases} x_{t-1} & \text{if } t > b(t) \text{ and } x_{t-1} \in \mathcal{S} \\ 0 & \text{otherwise} \end{cases}
\qquad
\tilde c_t = \big(s_t,\ x_{\tau_1(t)},\ \dots,\ x_{\tau_{K_s}(t)}\big)
```

τ₁(t) > τ₂(t) > … > τ\_{Kₛ}(t) ≥ b(t) are the Kₛ most recent positions before t whose tokens are not in 𝒮. If fewer than Kₛ such positions exist, position t has no semantic components. Context features stack unit-norm static token vectors ê, trained teacher-free (word2vec on the same corpus):

```latex
f(\tilde c_t) = \big[\hat e(x_{\tau_1(t)});\ \gamma\,\hat e(x_{\tau_2(t)});\ \dots;\ \gamma^{K_s-1}\hat e(x_{\tau_{K_s}(t)})\big]
```

With γ = 1, ‖f(c) − f(c′)‖ is the Frobenius distance in Corollary 3.2 of Balasundaram et al. Candidates are the M\_search nearest contexts c̃\_i ≠ c̃\_t with the same turn state, by L2 distance. Each gives p\_i(w) = N(c̃\_i, w) / N(c̃\_i). Position t is never in these counts, because c̃\_i ≠ c̃\_t. With 𝒮 empty, c̃\_t is the plain order-Kₛ context.

### 3. One risk scale for both backoffs

**Sampling variance.** For the empirical distribution p̂ of n samples from a distribution p on m symbols, Jensen's inequality gives E\[KL(p̂ ‖ p)\] ≤ ln E\[Σ\_w p̂(w)² / p(w)\] = ln(1 + (m − 1)/n) (Paninski 2003). The bound holds for every n ≥ 1, and it is exact at n = 1 for a uniform p. GBD uses it with an estimated support d̃:

```latex
\tilde d = \frac{n\,\hat d + n_0\, d_0}{n + n_0}, \qquad
v(n, \hat d;\, d_0) = \ln\!\Big(1 + \frac{\tilde d - 1}{n}\Big), \qquad
\hat d = d + f_1 + 1, \quad n_0 = 5
```

d₀ is a prior support, measured on the corpus for each context family (each exact order k, and the semantic contexts). Let D be the number of contexts in the family that occur exactly twice, and C the number of those whose two next tokens are equal:

```latex
d_0 = \min\!\Big(V_{\text{act}},\ \max\!\Big(1,\ \frac{D + 1}{C + 1}\Big)\Big)
```

A context that occurs twice is exactly the leave-one-out case n = 1, so C/D is the measured rate at which a single other occurrence predicts the label, and its inverse is the effective support. If D \< n\_fit, order k takes d₀ from order k − 1, with d₀ = V\_act at order 0. The semantic family takes d₀ of exact order Kₛ in that case.

**Component risks.**

```latex
r_{k,t} = \delta_k + v\big(n_{k,t},\, \hat d_{k,t};\, d_0(k)\big), \qquad
r_{i,t} = \delta_{K_s} + \big[a + b\,\lVert f(\tilde c_t) - f(\tilde c_i)\rVert\big]_+ + v\big(N(\tilde c_i),\, \hat d_i;\, d_0^{\text{sem}}\big)
```

From the M\_search candidates, the M lowest-risk neighbours are kept. The paper also selects neighbours by a risk proxy, not by distance alone.

**Fitting δ\_k, a and b.** All three are fitted on training contexts with N ≥ n\_fit, using a plug-in KL whose reference distribution is smoothed toward the unigram with total pseudo-count β₀ = 3, whatever the vocabulary size:

```latex
\widehat{\mathrm{KL}}(c_1 \,\Vert\, c_2) = \sum_{w:\,N(c_1,w)>0} \frac{N(c_1,w)}{N(c_1)}\,
\ln\frac{N(c_1,w)/N(c_1)}{\big(N(c_2,w) + \beta_0\,p_{\text{uni}}(w)\big)/\big(N(c_2) + \beta_0\big)}
```

**Order bias.** δ\_k is the bias of truncating to order k. δ\_K = 0, and for j = K − 1 down to 0, with c an order-(j+1) context and c′ its order-j suffix:

```latex
\delta_j = \delta_{j+1} + \Big[\operatorname{mean}_{c}\Big(\widehat{\mathrm{KL}}(c \,\Vert\, c') - \frac{\hat d_c - 1}{2}\Big(\frac{1}{N(c)} - \frac{1}{N(c')}\Big)\Big)\Big]_+
```

The subtracted term is the expected KL when the two true distributions are equal. The counts of c are a subset of the counts of c′, so the two sampling errors are correlated and the term carries a minus sign. The mean is taken first and rectified once; rectifying each context separately would bias δ upward. If fewer than 10 contexts qualify, the step is 0.

**Semantic bias.** a + b‖Δf‖ is the paper's bound 2L‖f\_c − f\_c̃‖ + 4ε. For neighbour pairs (c, c′) with both counts ≥ n\_fit, the regression target is

```latex
y_{c,c'} = \widehat{\mathrm{KL}}(c \,\Vert\, c') - \frac{\hat d_c - 1}{2\,N(c)} - \frac{\hat d_{c'} - 1}{2\,N(c')}
```

Here the counts are disjoint, so the two noise terms add. y is not rectified. (a, b) is the least-squares fit of y on ‖f(c) − f(c′)‖ under b ≥ 0: if the unconstrained slope is negative, b = 0 and a = mean(y). a may be negative, because noisy embeddings put true synonyms at a positive distance; the bias is rectified where it is used, \[a + b‖Δf‖\]₊. If fewer than 10 pairs qualify, the semantic branch is switched off.

### 4. Mixture and target

Weights come from φ(r) = 1/max(r, r\_min), the weighting the paper uses in its experiments. φ(r) = exp(−τr) is its softmin alternative. No formula below depends on which φ is used.

**Order branch.**

```latex
u_{k,t} = \frac{\phi(r_{k,t})}{\sum_{k'} \phi(r_{k',t})}, \qquad
p_{\text{ord},t} = \sum_{k=0}^{K} u_{k,t}\, p_{k,t}, \qquad
r_{\text{ord},t} = -\ln \sum_{k=0}^{K} u_{k,t}\, e^{-r_{k,t}}
```

r\_ord is the risk of the mixture that is actually used. For any distribution p, components q\_j and weights w\_j, convexity of log-sum-exp gives

```latex
\mathrm{KL}\Big(p \,\Big\Vert\, \sum_j w_j q_j\Big) \;\le\; -\ln \sum_j w_j\, e^{-\mathrm{KL}(p \,\Vert\, q_j)} \;\le\; \sum_j w_j\, \mathrm{KL}(p \,\Vert\, q_j)
```

The middle term never falls below the smallest component risk, and a component with infinite risk drops out of it.

**Semantic branch.** The kept neighbours 𝒩\_t are pooled. Their counts are disjoint, so the variance of the pooled estimate shrinks with an effective sample size, while the bias does not average out:

```latex
\tilde w_{i,t} = \frac{\phi(r_{i,t})}{\sum_{i' \in \mathcal{N}_t} \phi(r_{i',t})}, \qquad
p_{\text{sem},t} = \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\, p_i, \qquad
N_{\text{sem},t} = \Big(\sum_{i \in \mathcal{N}_t} \frac{\tilde w_{i,t}^{\,2}}{N(\tilde c_i)}\Big)^{-1}, \qquad
\bar d_{\text{sem},t} = \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\, \tilde d_i
```

```latex
r_{\text{sem},t} = \delta_{K_s} + \sum_{i \in \mathcal{N}_t} \tilde w_{i,t}\,\big[a + b\,\lVert f(\tilde c_t) - f(\tilde c_i)\rVert\big]_+ + \ln\!\Big(1 + \frac{\bar d_{\text{sem},t} - 1}{N_{\text{sem},t}}\Big)
```

If position t has no semantic components, r\_sem = +∞.

**Two-way mixture.**

```latex
\lambda_t = \frac{\phi(r_{\text{ord},t})}{\phi(r_{\text{ord},t}) + \phi(r_{\text{sem},t})}, \qquad
\tilde q_t = \lambda_t\, p_{\text{ord},t} + (1 - \lambda_t)\, p_{\text{sem},t}, \qquad
R_t = -\ln\!\big(\lambda_t\, e^{-r_{\text{ord},t}} + (1 - \lambda_t)\, e^{-r_{\text{sem},t}}\big)
```

q\_t is q̃\_t cut to its top r entries and renormalised. R\_t is the same mixture bound applied to the two branches. It is stored as a diagnostic.

### 5. Per-token distillation strength

```latex
\alpha_t = \alpha_{\max}\,\Big[1 - \frac{r_{\text{ord},t}}{\max(r_{0,t},\, r_{\min})}\Big]_+^{\gamma_\alpha}, \qquad \gamma_\alpha = 1
```

r\_{0,t} is the unigram component's risk, which always exists. The bracket is the share of the gap between the unigram and the order-K conditional that the order branch closes. If position t has no match beyond the unigram, r\_ord = r\_{0,t} and α\_t = 0 exactly, so that token is trained with plain cross-entropy. α\_t reads r\_ord and not R\_t on purpose: in the synthetic check the semantic risk does not track the error of the target, while r\_ord does (see the version 2 check below).

### 6. Training objective

```latex
\mathcal{L}(\theta) = \frac{1}{T}\sum_t \Big[(1-\alpha_t)\big(-\log p_\theta(x_t \mid x_{<t})\big) + \alpha_t\, H\big(q_t,\, p_\theta(\cdot \mid x_{<t})\big)\Big]
```

H(q, p) = KL(q ‖ p) + H(q), and H(q\_t) does not depend on θ, so this is the KD objective with q\_t as the teacher. The gradient with respect to the logits z\_t is softmax(z\_t) − ỹ\_t, where ỹ\_t = (1 − α\_t)·e\_{x\_t} + α\_t·q\_t. Training is plain cross-entropy against one precomputed soft label per token. There is no extra forward pass and no online cost.

**Annealed α\_max (repair 2, used at the full corpus scale).** At training step s of S, every α\_t is multiplied by 1 − s/S, so the soft target is applied fully at the start and not at all at the end:

```latex
\alpha_t^{(s)} = \Big(1 - \frac{s}{S}\Big)\,\alpha_t
```

The target then acts as an early curriculum, and the student is free to use context beyond K tokens late in training. This was pre-registered as repair 2 under kill criterion 0. At 1M tokens it is not used, because it costs 0.012 bits per byte there (see Results). The schedule is therefore chosen per data scale. At 6.36M tokens, every baseline with a mixing weight (label smoothing, Kneser–Ney, D2GPo-NN64) is annealed the same way.

### Cost

Targets are built once, offline, on CPU: counts by hashing and sparse matrices, neighbours by FAISS. Targets are stored as uint16 token ids and fp16 probabilities, T × r × (2 + 2) bytes, plus α\_t and a few float32 diagnostics (r\_ord, r\_sem, R\_t) per position. For the full corpus (T = 6.36M, r = 16) that is 0.41 GB for q\_t, and the whole target folder is about 0.57 GB. Building the full GBD target at 6.36M tokens took 497 s on CPU, and 84 s at 1M tokens. Training costs about the same as plain CE per step. Measured wall time was 29–32% higher in the two cases checked (807 s vs 612 s for the 98M model at 1M tokens, 24 epochs; 1,185 s vs 917 s for the 30M model at 6.36M tokens, 16 epochs). The likely cause is reading the targets off disk, but that has not been profiled. Target building is the RAM-heavy step, so all target variants are built during a temporary 32 GB RAM window and saved to disk as memory-mapped arrays. Training then streams them in chunks and runs within the machine's normal 16 GB.

### What is proven and what is not

Proven, and checked numerically in `gbd_v2_check.py`:

- The variance bound ln(1 + (m − 1)/n), for a known support m.
- The mixture bound behind r\_ord and R\_t, given the component risks.
- The sign of the nested noise term in δ\_k, to leading order in 1/N.
- Leave-one-out is unbiased over a context's occurrences: averaging p\_{k,t}(w) over the positions that share a context gives N(c, w)/N(c).

Heuristic:

- The component risks are proxies. The bias terms δ\_k and a + b‖Δf‖ are fitted, the support d̃ is estimated, and the variance bound is for KL(p̂ ‖ p) while the mixture bound needs KL(p ‖ q). The two directions agree to leading order only. For large n the variance term is about twice the leading-order value (d̂ − 1)/(2n), so it errs on the cautious side.
- The pooled semantic risk, the rule for λ\_t and the α\_t schedule. The single-component pieces rest on Theorems 3.1 and 4.2 of Balasundaram et al. Extending Theorem 4.2 to a mixture with fitted biases is an open item.

Two known limits of the risk scale:

- **It ignores truncation bias.** Every risk is measured against the order-K conditional, not against the full-context distribution the student models. α\_t is therefore blind to the bias described under the main design risk.
- **The semantic risk is not calibrated.** In the synthetic check r\_sem has no positive correlation with the true error of the target (Spearman −0.08 to −0.27 over three corpus seeds). Contexts that contain rare tokens have nearer neighbours in embedding space but worse ones, because rare-token vectors are poorly trained, and a and b are fitted on frequent contexts only. r\_sem still sets λ\_t, and the two-way target is closer to the truth than either branch alone (KL 2.74 against 3.81 for the order branch and 3.17 for the semantic branch, seed 0), but it is kept out of α\_t. Phase 0 repeated the check on the real corpus with the true distribution replaced by the NLL of the actual next token under the leave-one-out target. Spearman correlation with that NLL is 0.33 for r\_ord and 0.10 for r\_sem. The same pattern holds.

## Special cases and ablations

Several baselines are GBD with something switched off, so they double as ablations. The config column gives the `GBDConfig` field or `train.py` flag that produces each one. Every switch is independent of the others. "Constant α" means α\_t = α\_max at every position, in place of the rule in section 5. CoCoNTs and NDP proper are not in this table: they apply targets only where the n-gram is the model's whole context, which is a different placement, not a switch. "Final recipe" and "pilot recipe" are defined under Results.

| Setting | Recovers or isolates | Config | Run so far |
| --- | --- | --- | --- |
| One exact order, label included, constant α | Sliding-window exact n-gram target at every position (not CoCoNTs or NDP) | `--method exact_ngram` | Pilot recipe, 1M, 1 seed |
| Order 0 only, constant α | Unigram-frequency target | `--method unigram` | Pilot recipe, 1M, 1 seed |
| Uniform q, constant α | Label smoothing (= add-λ smoothing, Malagutti et al.) | `--method label_smoothing` | Final recipe, both scales, 1 seed per λ |
| No semantic components | Order backoff alone: isolates the semantic backoff | `--method gbd_order` (`use_semantic=False`) | Final recipe, 1M, 3 seeds |
| Label included | Isolates leave-one-out (contribution 3) | `leave_one_out=False` | Final and pilot recipe, 1M, 1 seed each |
| Constant α | Isolates per-token α\_t (contribution 2) | `alpha_mode="fixed"` | Final and pilot recipe, 1M, 1 seed each |
| Fixed λ | Isolates the risk-based branch weight λ\_t | `lambda_mode="fixed"`, `fixed_lambda` | Pilot recipe, 1M, 1 seed |
| Equal weights over orders and neighbours | Isolates risk-weighted mixing inside each branch | `phi="uniform"` | Final recipe, 1M, 1 seed |
| No risk anywhere | Equal weights, fixed λ = 0.73, constant α = 0.34: the whole risk scale off | the three rows above together | Final recipe, 1M, 3 seeds |
| α\_t from R\_t or r\_sem instead of r\_ord | Whether the semantic risk should drive α\_t | `alpha_mode="R"` or `"r_sem"` | Not run |
| Softmin weights | Weighting rule: paper's softmin vs. its experimental 1/x | `phi="exp"` | Not run |
| Hand-set biases | Fitted vs. fixed constants (paper: L = 5) | `delta=..., a=..., b=...` | Not run |
| Random token vectors | Does the geometry matter, or only the extra pooling? | `--embedding_source random` | Pilot recipe, 1M, 1 seed |
| Count-only geometry | Word2vec replaced by shifted-PPMI + SVD vectors (no gradient training at all) | `--embedding_source ppmi_svd` | Not run |
| Target on the first k positions of each chunk only | Placement: every position vs. CoCoNTs-style prefix-only (repair 1 for truncation bias) | `--prefix_only_k` | Not run |
| α\_max annealed to zero over training | Target as an early curriculum (repair 2 for truncation bias) | `--anneal_alpha` | Final recipe, both scales |

One caution for the write-up: D2GPo-style target-side smoothing (mass spread to tokens near the label) is **not** a special case. It stays a separate baseline.

## Reference implementation

Everything below is in `gbd/` and runs on one machine. Target construction is CPU-only (NumPy, SciPy, FAISS, gensim). Only student training and evaluation use the GPU (PyTorch, bf16).

| File | Role |
| --- | --- |
| `gbd.py` | GBD version 2 target construction (Method sections 1–5), with every component on its own switch; memory-mapped target storage; `gbd_loss`; `GBDChunks` loader |
| `gbd_v2_check.py` | Independent single-document reference for version 2, plus the synthetic three-seed benchmark |
| `data.py` | Human-written dialogue loader and deduplication, SentencePiece tokeniser, same-corpus word2vec, phase 0 diagnostics |
| `embeddings.py` | Teacher-free geometry sources: same-corpus word2vec (default), shifted-PPMI + SVD, random (control) |
| `model.py` | From-scratch causal decoders: `30m` (30.49M parameters) and `gpt2-small` shape (97.65M at V = 16k) |
| `train.py` | One trainer for every method and baseline, with a target cache keyed by a hash of every target setting |
| `kn.py`, `d2gpo.py` | Kneser–Ney and D2GPo-NN64 baseline targets |
| `eval.py` | Bits per byte (replies and all tokens), NLL by training-frequency bin, ECE, BLiMP |
| `test_gbd.py`, `sanity_synthetic.py` | Correctness tests; version 1 synthetic sanity check |

### Usage

```python
import numpy as np
from gbd import GBDConfig, build_targets, save_targets, load_targets, gbd_loss, GBDChunks

tokens = np.load("train_tokens.npy")        # token ids; conversations separated by EOS_ID
E = np.load("w2v_embeddings.npy")           # (V, D) static vectors, trained teacher-free on the same corpus
cfg = GBDConfig(K=6, Ks=2, M=8, r=16, alpha_max=0.5, eos_id=EOS_ID, role_ids=ROLE_IDS)
tg = build_targets(tokens, V, E, cfg)       # offline, once, on CPU
save_targets(tg, out_dir)                   # uint16 ids + fp16 probs, memory-mapped
tg = load_targets(out_dir, mmap_mode="r")

loader = torch.utils.data.DataLoader(GBDChunks(tokens, tg, L=512), batch_size=16, shuffle=True)
for x, y, ids, p, a in loader:
    logits = model(x)                        # any causal LM returning (B, L, V)
    loss, stats = gbd_loss(logits, y, ids, p, a * anneal_scale(step))   # anneal_scale = 1 at 1M tokens
    loss.backward(); opt.step(); opt.zero_grad()
```

### The training loss

```python
def gbd_loss(logits, labels, tgt_ids, tgt_probs, alpha, ignore_index=-100, nll_weight=None):
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = labels != ignore_index
    nll  = -logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    soft = -(tgt_probs.float() * logp.gather(-1, tgt_ids.long())).sum(-1)   # H(q_t, p_theta)
    w_nll = (1 - alpha) if nll_weight is None else nll_weight                # CoCoNTs and D2GPo set their own
    per_tok = w_nll * nll + alpha * soft
    return (per_tok * mask).sum() / mask.sum().clamp_min(1)
```

### How `build_targets` maps to the math

1. **Counts:** a rolling 64-bit hash gives each order-k context an id. A context that would cross an EOS token is marked invalid and never counted (t − k ≥ b(t), Method section 1). Counts are a sparse (contexts × V) matrix per order. Leave-one-out is exact and costs nothing: subtract 1 from N(c) and from N(c, x\_t).
2. **Semantic contexts:** role markers are skipped and the turn state s\_t is kept (Method section 2). Neighbours are searched only among contexts with the same turn state, with exact FAISS search below 100k contexts and HNSW above.
3. **Constants:** d₀ from contexts seen exactly twice; δ\_k, a and b from contexts with N ≥ n\_fit, using the sparse plug-in KL with β₀ = 3 (Method section 3).
4. **Mixture:** per chunk of positions, the weighted components are summed as sparse matrix products, r\_ord, r\_sem, λ\_t, R\_t and α\_t are computed as in sections 4–5, and q\_t is cut to top-r.

### Correctness tests

`test_gbd.py` has eight tests:

| Test | What it checks |
| --- | --- |
| Version 2 vs. reference | `gbd.py` reproduces `gbd_v2_check.py` on a single document. |
| Boundaries and role markers vs. brute force | Method sections 1–2 on conversations with EOS tokens and role markers, against a direct per-position implementation. |
| Leave-one-out and α modes | On a corpus where every context is unique, leave-one-out gives α\_t = 0; constant α behaves as specified. |
| Memory-mapped round trip | Saved and reloaded targets are identical. |
| Loss gradient | Autograd gradient equals softmax(z) − \[(1−α)e\_y + α q\]; α = 0 reproduces cross-entropy exactly. |
| Orthogonal switches | Every component switch and embedding source changes only what it should. |
| CoCoNTs Eq. 4 and cache isolation | The CoCoNTs baseline matches brute force and its expected target equals the all-n-gram distribution; no two settings share a target cache. |
| Kneser–Ney and D2GPo-NN64 | The two baseline targets (added Nov 13). |

The first seven passed on Nov 12 (project log, entries 006–009). The eighth was added on Nov 13, and its result is not yet recorded. `python gbd_v2_check.py test` checks d₀, δ\_k, a, b, q\_t, r\_ord, r\_sem, λ\_t, R\_t and α\_t against brute force on 60 positions (max error 8.9 × 10⁻⁸), the mixture bound on 2,000 random cases and the variance bound on six (support, n) settings.

### Synthetic sanity check (version 1)

Version 1 numbers, kept for the one-hot and exact-target rows. Version 2 is in the next section. This checks that the estimator does what the math says. It is not evidence about real language. The synthetic language has 300 tokens in 60 synonym classes. Classes follow a sparse second-order Markov chain, so the true next-token distribution is known. The corpus has 200k tokens, and 40% of evaluation positions have an order-2 context seen ≤ 3 other times ("rare"). Embeddings are word2vec trained on the same corpus. Scores are averaged over 4,000 positions. Lower is better.

| Target | KL to truth, all | TV, all | KL, rare | TV, rare |
| --- | --- | --- | --- | --- |
| One-hot label (plain CE) | 6.27 | 0.94 | 6.27 | 0.94 |
| Exact order-2, label included (sliding-window) | 4.50 | 0.73 | 5.64 | 0.87 |
| Exact order-2, leave-one-out | 4.73 | 0.76 | 5.95 | 0.90 |
| GBD, order backoff only | 3.88 | 0.73 | 4.67 | 0.84 |
| **GBD full** | **2.98** | **0.64** | **3.83** | **0.75** |

KL uses q floored with 1% uniform mass so it stays finite. TV is total variation distance.

What this shows:

- Semantic backoff gives the largest single gain, and it is largest on rare contexts (KL 5.64 → 3.83 against the exact target).
- The fitted biases recover the generator's structure: δ = 0 for orders 2–4 (the true order is 2), and δ ≈ 2.0–2.2 nats for orders 0–1.
- The risk proxy tracks real error: Spearman(R\_t, true KL) = 0.45. True KL is 2.26 in the lowest-risk quartile and 3.61 in the highest.
- **Leave-one-out costs target accuracy** (4.50 → 4.73), because it discards one real sample. Its case rests on training dynamics, not on target accuracy, and the ablation must show that benefit or LOO is dropped.

### Version 2 check (Nov 12)

Same generator as above, three corpus seeds, 8,000 positions each, K = 4, Kₛ = 2, α\_max = 0.5. `python gbd_v2_check.py bench` reproduces every number. "Risk vs. error" is the Spearman correlation between the risk that sets α\_t (R\_t in version 1, r\_ord in version 2) and the true KL of the target, as mean ± standard deviation over seeds.

| Target | KL to truth, all | TV, all | KL, rare | TV, rare | Risk vs. error |
| --- | --- | --- | --- | --- | --- |
| v1, order backoff only | 3.79 | 0.72 | 4.60 | 0.83 | 0.60 ± 0.02 |
| v2, order backoff only | 3.74 | 0.70 | 4.54 | 0.82 | 0.75 ± 0.01 |
| v1, full | 2.96 | 0.64 | 3.86 | 0.76 | 0.46 ± 0.01 |
| **v2, full** | **2.78** | **0.60** | **3.66** | **0.71** | **0.67 ± 0.01** |

Calibration by group for full GBD, version 1 → version 2:

| Positions | Share | Top target prob | Mean α\_t | True KL |
| --- | --- | --- | --- | --- |
| Order-4 context seen once elsewhere | 2.4% | 0.40 → 0.24 | 0.248 → 0.174 | 2.42 → 2.15 |
| No order-4 match, order-3 context seen once | 15.3% | 0.35 → 0.22 | 0.218 → 0.152 | 2.66 → 2.43 |
| Order-3 context seen 5+ times | 1.1% | 0.20 → 0.17 | 0.252 → 0.266 | 1.84 → 1.70 |
| Order-2 context seen 20+ times | 6.8% | 0.21 → 0.17 | 0.237 → 0.270 | 1.71 → 1.60 |
| Order-2 context never seen elsewhere | 8.3% | 0.13 → 0.11 | 0.128 → 0.001 | 4.56 → 4.42 |
| All positions | 100% | 0.23 → 0.17 | 0.189 → 0.129 | 2.96 → 2.78 |

What this shows:

- The target itself improves only a little (KL 2.96 → 2.78). The larger change is calibration. α\_t now orders the five groups by their true error: it is lowest (0.001) where the error is highest and highest (0.27) where it is lowest. It is exactly 0 at 17.5% of positions.
- A single other occurrence of a long context no longer takes half the target mass. With order backoff only, the top target probability in the first group falls from 0.49 to 0.36.
- **The semantic risk is not calibrated.** Spearman(r\_sem, true KL) is −0.08, −0.27 and −0.14 on the three seeds. R\_t, which includes r\_sem, reaches only 0.38 ± 0.03, below version 1. This is why α\_t reads r\_ord. The semantic branch still improves the target: on seed 0 the two-way target has KL 2.74, against 3.81 for the order branch alone and 3.17 for the semantic branch alone.
- Fitted constants are stable across seeds: δ ≈ (2.3, 2.1, 0, 0, 0), which matches the generator's true order of 2; d₀ between 15 and 17 for orders 2 to 4; a between −3.9 and −2.9 and b between 13 and 15.
- This is still synthetic data. Phase 0 repeated the calibration check on the real corpus (see Results, phase 0).

## Experimental plan

All runs are small decoder LMs trained from random initialisation, for conversation reply, on human-written English conversation data. Everything runs in-house on one machine: no cloud, no API, no pretrained weights. The small-data setting is simulated by subsampling, and nothing in the project is multilingual. Every method shares one tuned training recipe per scale and differs only in its target. The plan below was written before any run. The **Status** columns and the outcomes under the kill criteria were added on Nov 14.

### Setup

- **Initialisation:** from scratch. The course does not require random initialisation, so this is a choice. A pretrained checkpoint has already seen far more text than the corpus, which removes the small-data regime that GBD is built for.
- **Task:** conversation reply. Conversations are flattened into plain text with role markers, and an end-of-sequence token separates conversations.
- **Corpus (built):** `OpenAssistant/oasst1` (English, split into node-disjoint conversations so branches of one tree do not share text), `HuggingFaceH4/no_robots` and DailyDialog. After deduplication there are 29,141 conversations, split by tree into 26,228 train, 1,468 validation and 1,445 test. That gives 6,358,385 training tokens (4,523,096 of them in replies), 376,857 validation tokens and 340,225 test tokens. Synthetic chat data is excluded, because text written by an LLM would bring a teacher back in as sequence-level distillation.
- **Data scales:** the first 1M training tokens, and the full 6.36M. The planned 5M and 20M scales are replaced by these two. 20M tokens of human-written dialogue is not available from these sources, and 5M is too close to 6.36M to add information. Adding `oasst2` or `prosocial` could extend the corpus. Validation and test splits are fixed.
- **Tokeniser:** SentencePiece BPE with 16,000 types, trained on the training split only, with `byte_fallback=True` so BLiMP words never map to `<unk>`.
- **Students (Stage 1):** `30m`, a 30.49M-parameter decoder at the course's reference scale, and `gpt2-small`, a GPT-2-small-shaped decoder with 12 layers and d = 768. The latter has 97.65M parameters at V = 16k, not 124M, because the embedding table is smaller. Stage 2 is unchanged: larger from-scratch configurations of Granite, Llama, Gemma, Phi and Hymba, capped at roughly 0.5–0.9B parameters by full-parameter AdamW in 16 GB (estimate), and run only if Stage 1 passes.
- **Hardware:** one RTX 5060 Ti (16 GB VRAM), 16 GB system RAM (32 GB during the target-building window), 12 logical CPU cores.
- **Recipe:** AdamW, sequence length 512, batch 16 × 2 accumulation steps, 5% warmup, bf16, best checkpoint by validation reply bits per byte. Dropout and epochs were swept for CE on validation, and every method reuses the winner (see Results). Learning rate and weight decay were barely swept: one run at weight decay 0.3, which was worse. That is weaker tuning than the data-constrained literature recommends, and it is listed as an open item.

### Baselines (matched α where the method has one, same recipe)

| # | Baseline | Rules out | Status |
| --- | --- | --- | --- |
| 1 | Tuned cross-entropy | Gains from a weak baseline | Done, 3 seeds per cell |
| 2 | Uniform label smoothing | Gains needing no information | Done, final recipe, λ ∈ {0.05, 0.1}, 1 seed |
| 3 | Random target, entropy matched to GBD | Gains from soft-target dynamics alone | Pilot recipe only, 1M, 1 seed |
| 4 | Unigram target | Gains from frequency alone | Pilot recipe only, 1M, 1 seed |
| 5 | PS-KD self-distillation | Gains a model can give itself | Not implemented (awaits the professor's ruling) |
| 6 | D2GPo-style embedding-similarity target | Target-side geometry only | Done as D2GPo-NN64 (see note), λ ∈ {0.1, 0.34}, 1 seed |
| 7 | Sliding-window exact n-gram target, every position | Backoff adds nothing | Pilot recipe only, 1M, 1 seed |
| 7a | CoCoNTs proper: KL in place of next-token loss on the first k positions of each fixed block, corpus-wide n-gram counts, no weight | Every-position placement adds nothing beyond prefix-only exact targets | Pilot recipe only, 1M, 1 seed |
| 7b | Kneser–Ney-smoothed n-gram target | Semantic backoff adds nothing beyond the best order-only smoother | Done; 3 seeds at 1M, 1 seed per α at 6.36M |
| 8 | Static embeddings as initialisation | Geometry is cheaper as init than as loss | Implemented, not run |
| 9 | Masked-input regularisation (MIR) | GBD is just another regulariser | Implemented, not run |
| 10 | Goldfish English model at the matching data size (if one exists; general text, so a reference only) | Our models are below public practice | Not run |

NDP is not run separately. On chunked pretraining text its prefix keys repeat only in the first few positions of a block, where it coincides with baseline 7a apart from how unseen tokens are smoothed (our reading of the paper).

**Note on baseline 6.** With the released settings (σ = 1, T = 2) and V = 16k, D2GPo's prior is within 1.4 × 10⁻⁵ of uniform at every rank, so faithful D2GPo here would just be label smoothing. We therefore run D2GPo-NN64: the same Gaussian-of-rank prior, restricted to the 64 nearest tokens to the label in the same word2vec space GBD uses, and renormalised. This is a stronger and different prior than the paper's, and it is named as such.

### Metrics

- Held-out **bits per byte** on reply tokens of the test conversations (primary; comparable across tokenisers). Bits per byte over all tokens is reported alongside.
- **Rare-token NLL** by training-frequency bin (< 10, 10–99, 100–999, 1000+ occurrences). The plan expected GBD to gain most in the rarest bin. It does not (see Results).
- **Grammaticality** on BLiMP minimal pairs. Implemented in `eval.py`, not yet run.
- **Calibration** (ECE).
- Optional: bits per byte of a larger pretrained model on the same test set, run locally for evaluation only, as a reference point. It is never used to train. Not run.
- Target-building time and storage, reported next to training cost.

Every configuration was meant to run 3 seeds with 95% confidence intervals. So far only the CE-vs-GBD cells, GBD order-only, Kneser–Ney at 1M and the no-risk ablation have 3 seeds. With 3 seeds a 95% interval is wide, so tables report mean ± standard deviation instead.

### Run order

| Phase | Runs | Decides | Status |
| --- | --- | --- | --- |
| 0. Diagnostics | Corpus choice and token count; deduplication; unique-context rate by order and scale; risk calibration table for r\_ord and r\_sem, and a manual look at nearest neighbours on real data; δ\_k, a, b on the real corpus | K, Kₛ, M; whether sparsity actually bites | Done |
| 1. Recipe | CE sweep (LR × weight decay × epochs), 124M, at 1M and 20M tokens | Shared recipe per scale | Partly done: dropout × epochs swept; LR and weight decay barely |
| 2. Pilot (gate) | CE with 2 seeds (noise estimate), GBD order backoff only, GBD full; 124M model; 1M and 20M tokens; α\_max = 0.5; 8 runs in all | Kill criterion 0: does an every-position target survive at 20M? | Done at 1M and 6.36M; criterion 0 triggered at 6.36M; repair 2 run, repair 1 not |
| 3. Main | Baselines 1–9 (including 7a and 7b) and GBD, α\_max ∈ {0.25, 0.5, 0.75}, 3 seeds | Kill criteria 1–3 | Partly done: baselines 1, 2, 6, 7b; mostly 1 seed |
| 4. Ablations | The special-cases table | Which contributions survive | Partly done at 1M (see the special-cases table) |
| 5. Scale | CE, sliding-window exact n-gram and GBD at 1M / 5M / 20M, ~30M and 124M | Where the gain appears | 1M: both models. 6.36M: final recipe on the 30M model only |
| 6. Stage 2 (conditional) | CE and GBD on larger from-scratch configurations that fit 16 GB | Whether the gain holds at larger model size | Not started |

### Kill criteria (fixed before any run)

The criteria are as written on Nov 11–12. The outcome under each was added on Nov 14. "20M tokens" reads as the full 6.36M corpus.

0. **Pilot gate (phase 2).** GBD full is not worse than tuned CE on held-out bits per byte at 20M tokens by more than the gap between the two CE seeds. If it is worse at 20M but better at 1M, phase 3 does not run as planned: rerun the pilot with the two placement repairs (prefix-only targets, annealed α\_max) and narrow the claim to the corpus sizes where GBD wins. If it is worse at both scales, stop.
   - **Outcome: triggered.** At 6.36M tokens with constant α\_max = 0.5, GBD was worse than CE: 1.3265 vs 1.3179 / 1.3226 on the 98M model (pilot recipe), and 1.3180 vs 1.3027 on the 30M model (final recipe). At 1M it was better. Repair 2 (annealing) was run and fixes the loss at 6.36M. Repair 1 (prefix-only) has not been run. Some phase 3 baselines were run before repair 1, which departs from the gate as written.
1. GBD beats the exact and Kneser–Ney n-gram targets (baselines 7 and 7b) and CoCoNTs proper (7a) on held-out bits per byte, at matched α, across 3 seeds. If not, stop: the backoff adds nothing.
   - **Outcome: holds at 1M, open at 6.36M.** At 1M, GBD (1.7846 ± 0.0016) beats Kneser–Ney (1.8191 ± 0.0020), both with 3 seeds. Exact n-gram and CoCoNTs were beaten by 0.06 at the pilot recipe with 1 seed. At 6.36M, GBD (1.2918 ± 0.0009) vs annealed Kneser–Ney (1.2937, 1 seed) is a gap of 0.002. Exact n-gram and CoCoNTs have not been run there.
2. GBD beats static-embedding initialisation (baseline 8). If not, the geometry belongs in the init, not the loss.
   - **Outcome: not tested.** Baseline 8 is implemented but not run.
3. The gain concentrates in rare-token bins. If it is uniform, it is probably generic regularisation, so compare against MIR.
   - **Outcome: fails.** At 1M the per-token gain is smallest in the rarest bin. At 6.36M GBD is worse than CE in that bin. Under this criterion the MIR comparison is now required. MIR is implemented but not run.
4. Removed Nov 11. The corpus is English only, so there is no cross-lingual variant and no knowledge-transfer claim. GBD is claimed as a regulariser. The number is kept so the other criteria keep theirs.
5. Per-token α\_t beats constant α, and leave-one-out beats label-included. Any that fails is dropped from the contribution list. The α\_t test uses the version 2 risk scale. If the phase 0 calibration check on the real corpus shows that r\_ord does not track target error, contribution 2 is dropped without running it.
   - **Outcome: leave-one-out holds; per-token α\_t is not shown.** Leave-one-out beats label-included at both recipes (1 seed each). Per-token α\_t beats constant α by 0.005 at the final recipe but not at the pilot recipe (1.8320 vs 1.8314). Under the rule, per-token α\_t comes off the contribution list until 3-seed runs show it. The r\_ord calibration check passed (Spearman 0.33 on real data). The risk scale as a whole does matter: removing all of it costs 0.014 (3 seeds), and replacing φ(r) with equal weights alone costs 0.008 (1 seed).

## Results (as of Nov 14)

Every number below is from `artifacts/runs/*.json` and `artifacts/corpus/phase0_diagnostics.json`. The metric is test bits per byte on reply tokens, lower is better, with validation in brackets where it matters. "± x" is the standard deviation over 3 seeds. Rows without it are 1 seed. Seed-to-seed standard deviation is about 0.002 at 1M tokens and about 0.001 at 6.36M, so single-seed gaps smaller than about 0.005 at 1M (0.003 at 6.36M) are not evidence.

### Recipes

- **Pilot recipe (Nov 12):** 12 epochs, dropout 0.1 at 1M tokens. 8 epochs at 6.36M, with dropout 0.1 on the 98M model and 0.05 on the 30M model. Learning rate 6 × 10⁻⁴ (98M) or 8 × 10⁻⁴ (30M), weight decay 0.1.
- **Final recipe (Nov 13):** chosen for CE on validation. At 1M: 24 epochs, dropout 0.3 (swept 0.2–0.4) on both models. At 6.36M on the 30M model: 16 epochs, dropout 0.25 (swept 0.05–0.25 at 16 epochs, and 0.35 at 24). In each of the three sweeps, GBD's best dropout on validation was the same as CE's, so the shared recipe does not favour either. Dropout 0.4 was not tried for GBD on the 98M model.
- Tuning CE harder shrank GBD's lead at 1M from 0.070 (pilot recipe) to 0.048 (final recipe). This is why the baseline has to be tuned before any comparison counts.

### Main result: GBD vs tuned cross-entropy (final recipe, 3 seeds each)

| Data | Student | CE | GBD | Difference | Reply perplexity, CE → GBD | ECE, CE → GBD |
| --- | --- | --- | --- | --- | --- | --- |
| 1M | 97.7M | 1.8327 ± 0.0021 \[1.8982\] | **1.7846 ± 0.0016** \[1.8512\] | −0.048 | 141.5 → 124.2 | 0.067 → 0.050 |
| 1M | 30.5M | 1.8273 ± 0.0020 \[1.8910\] | **1.7833 ± 0.0037** \[1.8471\] | −0.044 | 139.4 → 123.8 | 0.071 → 0.050 |
| 6.36M | 30.5M | 1.3027 ± 0.0006 \[1.3495\] | **1.2918 ± 0.0009** \[1.3377\] (annealed) | −0.011 | 33.8 → 32.8 | 0.058 → 0.052 |
| 6.36M | 30.5M | | 1.3180 \[1.3646\] (constant α\_max, 1 seed) | +0.015 | | |

GBD uses α\_max = 0.5. At 1M the schedule is constant. At 6.36M it is annealed (Method section 6), and without annealing GBD loses to CE. The 98M model has not yet been trained at 6.36M under the final recipe. At the pilot recipe it showed the same pattern: constant α\_max lost to CE (1.3265 vs 1.3203 mean of 2 seeds), and annealing won (1.3037).

### Baselines (final recipe)

**1M tokens, 97.7M student, constant schedule:**

| Method | Setting | Test \[validation\] |
| --- | --- | --- |
| Cross-entropy | | 1.8327 ± 0.0021 \[1.8982\] |
| Label smoothing | λ = 0.05 / 0.1 | 1.8324 \[1.8996\] / 1.8360 \[1.9015\] |
| D2GPo-NN64 | λ = 0.1 / 0.34 | 1.8305 \[1.8981\] / 1.8760 \[1.9389\] |
| Kneser–Ney target (K = 6, leave-one-out) | α = 0.34 / 0.5 | 1.8191 ± 0.0020 \[1.8870\] / 1.8486 \[1.9225\] |
| GBD, order backoff only | α\_max = 0.5 | 1.8060 ± 0.0039 \[1.8743\] |
| **GBD full** | α\_max = 0.5 | **1.7846 ± 0.0016** \[1.8512\] |
| Born-Again self-distillation | α = 0.5, τ = 1, one CE teacher | 1.7826 ± 0.0029 |
| Ensemble distillation (BabyLlama-2 style) | α = 0.5, τ = 1, mean of a 30M and a 98M teacher's logits | 1.7726 ± 0.0024 |

**6.36M tokens, 30.5M student, every mixing weight annealed:**

| Method | Setting | Test \[validation\] |
| --- | --- | --- |
| Cross-entropy | | 1.3027 ± 0.0006 \[1.3495\] |
| Label smoothing | λ = 0.05 / 0.1 | 1.2970 \[1.3457\] / 1.2951 \[1.3435\] |
| D2GPo-NN64 | λ = 0.1 / 0.34 | 1.2980 \[1.3401\] / 1.2947 \[1.3325\] |
| Kneser–Ney target | α = 0.34 / 0.5 | 1.2937 \[1.3404\] / 1.2942 \[1.3386\] |
| **GBD full** | α\_max = 0.5 / 0.75 | **1.2918 ± 0.0009** \[1.3377\] / **1.2892** \[1.3333\] |
| Born-Again self-distillation | α = 0.5, τ = 1, annealed, one 30M CE teacher | 1.2784 ± 0.0012 \[1.3217\] |
| Ensemble distillation | α = 0.5, τ = 1, annealed, two 30M teachers | 1.2783 ± 0.0023 \[1.3233\] |

On the 30M model at 1M tokens the same teacher-based references score 1.7844 ± 0.0008 (one teacher) and 1.7744 ± 0.0046 (30M plus GPT-2-small teachers). GBD there is 1.7833 ± 0.0037. The ensemble beats GBD on every seed of both models (by 0.0089 ± 0.0032 on 30M, just under the decisive bar, and by 0.0120 ± 0.0036 on the 98M model). As a fraction of the gain over CE, GBD recovers 83% (30M) and 80% (98M). Wall-clock to produce one 30M student, including the target build or the teachers: CE 239 s, GBD 341 s, one teacher 646 s, the ensemble 1450 s.

At 6.36M the teachers pull ahead. Self-distillation beats annealed GBD (α\_max = 0.5) by 0.0134 ± 0.0003 on test, on every seed. Two 30M teachers give the same number (1.2783). Same-size teachers trained on the same data add nothing over one, whereas the mixed-size pair at 1M added about 0.010.

At the pilot recipe (1M, 98M student, 1 seed each), the other matched-α controls were all worse than GBD (1.8320) and than CE (1.8958): exact 6-gram, α = 0.34: 1.8936; CoCoNTs, Eq. 4: 1.8937; label smoothing, α = 0.34: 1.9865; unigram target, α = 0.34: 2.0189; random target, entropy matched: 2.0014. So label smoothing, unigram and random targets at GBD's strength hurt, and GBD's gain is not explained by the strength of the soft term.

### Ablations (1M tokens, 97.7M student)

| Variant | Final recipe | Pilot recipe |
| --- | --- | --- |
| GBD full | 1.7846 ± 0.0016 | 1.8320 |
| No semantic branch (order backoff only) | 1.8060 ± 0.0039 | 1.8477 |
| Random token vectors in place of word2vec | 1.7990 (seed 0; full GBD at that seed 1.7833, order backoff 1.8020) | 1.8448 |
| No risk anywhere (equal weights, λ = 0.73, α = 0.34) | 1.7985 ± 0.0016 | – |
| Equal weights φ, everything else as GBD | 1.7917 | – |
| Fixed λ = 0.73 | – | 1.8334 |
| Constant α = 0.34 (the mean of α\_t) | 1.7894 | 1.8314 |
| Constant α = 0.34, label included (no leave-one-out) | 1.8071 | 1.8600 (adaptive α) |
| Annealed α\_max | 1.7965 | – |

### Where the gain is: NLL by training frequency (test, nats per token, mean of 3 seeds)

| Training count of the target token | < 10 | 10–99 | 100–999 | 1000+ |
| --- | --- | --- | --- | --- |
| 1M, 97.7M: CE | 10.38 | 7.95 | 5.67 | 2.69 |
| 1M, 97.7M: GBD | 10.32 | 7.83 | 5.50 | 2.57 |
| Difference | −0.06 | −0.12 | −0.17 | −0.12 |
| 6.36M, 30.5M: CE | 10.75 | 6.55 | 5.28 | 2.54 |
| 6.36M, 30.5M: GBD (annealed) | 11.20 | 6.51 | 5.23 | 2.52 |
| Difference | **+0.45** | −0.04 | −0.05 | −0.02 |

The rarest bin is about 4% of scored test tokens at 1M and 0.1% at 6.36M. Share of the total test-NLL reduction (CE − GBD), pooled over 3 seeds, against that bin's share of reply tokens:

| Frequency in training | 1M token share | 1M gain share, 98M | 1M gain share, 30M | 6.36M token share | 6.36M gain share, 30M |
| --- | --- | --- | --- | --- | --- |
| Rare (< 10) | 4.1% | 1.9% | 0.7% | 0.1% | −1.0% |
| Low (10–99) | 22.4% | 20.2% | 19.1% | 7.5% | 11.3% |
| Mid (100–999) | 25.8% | 34.3% | 34.5% | 24.7% | 43.8% |
| Frequent (≥ 1000) | 47.7% | 43.6% | 45.7% | 67.7% | 45.8% |

Mid-frequency tokens receive 1.3–1.8 times their share of the gain. On the 98M model at 1M tokens, 67% of order backoff's gain over Kneser–Ney falls on low-frequency tokens, while the semantic branch's gain over order backoff falls on mid (31%) and frequent (42%) tokens.

For comparison, at 1M label smoothing (λ = 0.05) and D2GPo-NN64 (λ = 0.1) reach 9.92 and 9.84 in the rarest bin, about 0.5 better than CE, without improving total bits per byte. GBD with constant α\_max at 6.36M reaches 12.45 there (CE 10.85) and the best ECE of any run (0.013 against 0.058). The truncation bias falls hardest on rare tokens, and annealing removes only part of it.

### Grammaticality (BLiMP)

All 67 paradigms, 200 minimal pairs each. A pair is correct when the grammatical sentence gets the higher log-probability. Gap = GBD − CE, in percentage points. None is decisive under the paired-seed rule.

| Model | Tokens | CE | GBD | Gap |
| --- | --- | --- | --- | --- |
| 97.7M | 1M | 55.6 ± 0.5 | 57.1 ± 1.3 | +1.5 ± 1.4 |
| 30.5M | 1M | 56.2 ± 0.5 | 57.3 ± 0.2 | +1.1 ± 0.5 |
| 30.5M | 6.36M | 64.2 ± 0.6 | 64.1 ± 0.4 | −0.1 ± 0.2 |

Chance is 50%. At this scale BLiMP separates methods weakly. GBD does not hurt grammatical judgement.

### Phase 0 on the real corpus (1M-token slice)

Fitted constants: δ = (6.11, 2.38, 1.04, 0.55, 0.30, 0.14, 0) for orders 0–6; d₀ = 3.19 to 1.39 for orders 1–6; a = −0.43, b = 5.20. Mean α\_t is 0.336 (5th–95th percentile 0.22–0.43), and α\_t = 0 at 0.5% of positions. Mean λ\_t is 0.76. Calibration against the NLL of the true next token under the leave-one-out target, on 20k sampled positions:

| Positions | Share | Top target prob | Mean r\_ord | Mean α\_t | Target NLL |
| --- | --- | --- | --- | --- | --- |
| Order-6 context seen once elsewhere | 5.4% | 0.76 | 0.86 | 0.43 | 2.76 |
| Order-3 context seen 5+ times | 20.0% | 0.42 | 1.16 | 0.41 | 5.84 |
| Order-2 context seen 20+ times | 35.8% | 0.39 | 1.49 | 0.38 | 6.84 |
| Order-3 seen once, no longer match | 7.4% | 0.53 | 1.83 | 0.35 | 7.06 |
| Order-2 context never seen elsewhere | 20.1% | 0.24 | 3.27 | 0.23 | 8.04 |

α\_t falls in the same order as target NLL rises. Spearman correlation with target NLL is 0.33 for r\_ord, 0.10 for r\_sem and 0.33 for R\_t. Nearest semantic neighbours look like paraphrases ("built in" → "created in", "constructed in", "installed in"; "eyes ." → "nose .", "ears .", "mouth .").

### What this shows

- **At 1M tokens GBD is the strongest teacher-free target tested,** by at least 0.032 reply BPB, and it ties a trained teacher. Order backoff beats Kneser–Ney by 0.013, the semantic branch adds 0.021, and random vectors keep only a small slice of that semantic gain.
- **At 6.36M tokens the every-position target needs a curriculum, and then it only matches other smoothing.** Annealed GBD still beats CE by 0.011, but label smoothing, D2GPo-NN64 and Kneser–Ney gain 0.008–0.009 as well. A trained teacher beats GBD by 0.013. Two same-size teachers add nothing over one at this scale.
- **The gain is not a rare-token effect.** Mid-frequency tokens take 1.3–1.8 times their share. Kill criterion 3 therefore points to the generic-regularisation explanation, and MIR is the test that decides it.
- **Leave-one-out matters more than its effect on target accuracy suggests,** at both recipes (0.018 at the locked recipe, seed 0).
- **The risk scale matters as a whole (0.014, 3 seeds).** Per-token α\_t alone is 0.006 (1 seed) and is not a separate contribution.

### Runs still needed, by what they decide

Necessary, because each one settles a kill criterion the paper still leaves open:

1. **MIR** (baseline 9) at 1M and 6.36M, 3 seeds. Kill criterion 3.
2. **Static-embedding initialisation** (baseline 8) at 1M, 3 seeds. Kill criterion 2.
3. **Kneser–Ney and D2GPo-NN64 at 6.36M** with 2 more seeds each. The paper's tie between GBD and those baselines is one seed on their side.

Useful but not deciding: repair 1 (prefix-only) at 6.36M; the 98M student at 6.36M under the final recipe; exact n-gram and CoCoNTs at the final recipe; the semantic ablation at 6.36M; whether GBD and a teacher add when both are used; a learning-rate and weight-decay sweep for CE. BLiMP is done.

## Anticipated questions

These are the questions a careful reviewer will ask first, with the answer we would give.

**Isn't this CoCoNTs plus Semantic Smoothing glued together?** The two pieces don't fit without new parts. Semantic Smoothing works for one bigram context and a count model. CoCoNTs matches exact n-grams on the first k positions of a block with an unweighted KL term. Joining them needs a common risk scale across orders and neighbours (δ\_k), per-token α\_t, and leave-one-out targets. Each of those is a separate ablation that can fail.

**Word2vec is a trained model. Isn't that a teacher?** It is a static table of token vectors from a shallow log-linear model trained on the same corpus. It never predicts a next-token distribution and only decides which contexts count as similar. The supervision itself is counts. Whether this meets the course's "no teacher" rule is an open item to confirm with the professor. The random-vector ablation shows how much the geometry contributes. At the pilot recipe, swapping word2vec for random vectors removes most of the semantic branch's gain (1.8448 vs 1.8320, with order backoff alone at 1.8477; 1 seed each). The count-only alternative, shifted-PPMI + SVD, is implemented, so the semantic branch can be run with no gradient-trained component at all if the professor rules word2vec out.

**Why would a smoothed count target help a neural LM at all?** With finite data, CE fits each training label and overfits. A soft target shares evidence across contexts that should behave alike, which lowers variance. Malagutti et al. show n-gram smoothers act as neural-LM regularisers. CoCoNTs shows the gain empirically: lower fine-tuning perplexity on WikiText-103, and better CoLA after from-scratch pretraining on BabyLM. NDP shows it when fine-tuning pretrained models. Our kill criteria test whether GBD's version adds to that.

**The target conditions on at most K tokens, the student on 512. Doesn't this pull the student toward an n-gram model?** Yes, by design, and it is the main risk. CoCoNTs and NDP avoid it by applying targets only where the n-gram is the whole context. GBD accepts the bias in exchange for lower variance at every position, which should pay on tiny corpora and fade as data grows. The phase 2 pilot tested this. At 1M tokens the bias does not matter and GBD wins clearly. At 6.36M it does matter: with constant α\_max, GBD is ahead of CE for the first few epochs and then falls behind it. Annealing α\_max to zero (repair 2) fixes this. Prefix-only targets (repair 1) are implemented but not yet run. CoCoNTs found that a prefix-only loss still lowers loss at every position, which is the evidence that the first repair can work.

**α\_t barely varied in your first check. Is risk calibration a real contribution?** It was not in version 1: α\_t ranged from 0.21 to 0.25 across groups whose true error differed by 1.5 nats. Version 2 changes the variance term, the support prior, the mixture risk and the α\_t rule. On the same synthetic language α\_t now runs from 0 to about 0.27, and the risk that sets it has Spearman 0.67 with true target error, against 0.46 before. On the real corpus the ordering holds (see Results, phase 0), but in training, per-token α\_t beats constant α only at the final recipe, by 0.005 with 1 seed, and not at the pilot recipe. What training does show is that the risk scale as a whole helps (0.014, 3 seeds). So per-token α\_t is not claimed until kill criterion 5 is rerun with 3 seeds.

**Why ln(1 + (d̃ − 1)/n) as the variance term?** It is an upper bound on the expected KL of an empirical estimate on d̃ symbols, and it holds for every n, including n = 1. Version 1 used the leading-order term (d̂ − 1)/(2n), the term Semantic Smoothing uses for its base weight. That term can never exceed 1 nat, so a context seen once elsewhere took about half the target mass. The paper estimates support with a Chebyshev-polynomial estimator. We use observed support plus singletons plus one, shrunk toward a prior support measured on contexts seen exactly twice. The support estimator is an ablation.

**Why not a Kneser–Ney-smoothed n-gram target instead?** KN is the strongest order-only backoff, so it is a required baseline (7b in the plan). At 1M, GBD's order branch alone beats leave-one-out KN by 0.013 (3 seeds, decisive), and the full target beats it by 0.034 on the 98M model and 0.037 on the 30M model. At 6.36M the test gap is about 0.003–0.005 with one KN seed, and validation does not separate GBD from the other annealed soft targets.

**Isn't fitting δ, a, b on the corpus circular?** They are fitted on frequent training contexts only, net of sampling noise. Validation and test text is never touched. Hand-set constants (the paper's L = 5) are an ablation.

**Your own sanity check shows leave-one-out hurts target accuracy. Why keep it?** The training ablation decided in its favour. The argument for it is about the training signal: the soft term should carry information beyond the label the CE term already has. Removing leave-one-out costs 0.018 bits per byte at the final recipe and 0.028 at the pilot recipe (1 seed each).

**How is α\_max set?** The plan sweeps α\_max over {0.25, 0.5, 0.75}. So far 0.5 is used everywhere, and 0.75 was run once at 6.36M, where it was better on both validation and test (1.3333 and 1.2892, against 1.3377 and 1.2918). The sweep is not finished. Version 2 has no ρ: α\_t is anchored on the unigram risk at the same position, which makes the schedule scale-free without a corpus-level constant. γ\_α stays at 1. Constant α is the ablation.

**How expensive is target building?** Measured on the real corpus: 84 s at 1M tokens and 497 s at 6.36M, on CPU, with HNSW neighbour search above 100k distinct contexts. Storage is about 0.57 GB at 6.36M tokens. One training run takes 4–20 minutes on the RTX 5060 Ti.

**Hash collisions in context ids?** With 64-bit hashes and at most 2×10⁷ distinct contexts, the expected number of colliding pairs is about 10⁻⁵.

**What scale does the course ask for?** About 30M parameters, trained in-house with no cloud or API, on small data, with no teacher. An earlier draft read this as "30B-level", which was a million/billion mix-up. The claim is about the training objective: at the same data and model size, GBD beats the other teacher-free targets and tuned cross-entropy. A larger model's score, if reported, is a reference point, not a target.

**Why only human-written dialogue?** Most open chat datasets are generated or rewritten by large LLMs. Training on them is sequence-level distillation from that LLM, which would put a teacher back into a teacher-free method. The corpus candidates were chosen to rule that out.

**What if the corpus is too small for good word2vec vectors?** Then same-corpus geometry is noisy. CBOW analogy accuracy is 23% at 24M words in Mikolov et al. (Table 2). With an English-only, teacher-free setup there is no outside geometry to fall back on. Phase 0 inspected nearest neighbours by hand, and at the 2-token context level they look like paraphrases (see Results). The random-vector ablation shows the geometry does add something at 1M tokens (pilot recipe, 1 seed). It has not been repeated at 6.36M.

**Isn't annealing a fix chosen after seeing the result?** It was written into kill criterion 0 as repair 2 before any run, along with prefix-only targets. It was used only after the gate triggered. It is applied to every baseline that has a mixing weight at 6.36M. It is not used at 1M, where it hurts (1.7965 vs 1.7846). The schedule therefore depends on data scale, and the paper has to say so.

**Is GBD just another regulariser?** At 1M tokens, no. Matched-strength label smoothing, unigram and random targets all hurt while GBD helped, and GBD leads Kneser–Ney, label smoothing and D2GPo-NN64 by at least 0.032. At 6.36M the answer is much closer to yes: every annealed teacher-free target gains about the same, while a trained teacher still gains 0.024 over CE. MIR (baseline 9) is the comparison the plan names for the regularisation question, and it has not been run.

**How does GBD compare with training a teacher?** At 1M tokens it ties Born-Again self-distillation from one CE teacher (gaps of 0.001–0.002, seeds split, not decisive) and recovers 80–83% of the gain of distilling from the mean logits of a 30M teacher and a 98M teacher. It costs 341 s per 30M student against 646 s for one teacher and 1450 s for the ensemble. At 6.36M a trained teacher beats GBD by 0.013 on every seed, and a second 30M teacher adds nothing. The paper's line is: use GBD when a teacher cannot be trained; use the teacher once the corpus is a few million tokens. Whether the two signals add is untested.

**Was anything chosen on the test set?** Dropout, epochs and the best checkpoint were chosen on validation. The kill criterion 0 decision was read off test bits per byte, as the criterion was written, and validation gives the same decision (on the 98M model, constant-α GBD scores 1.3775 against CE's 1.3733 and 1.3661). Results that went against GBD are reported alongside the rest.

## Risks and open items

The two biggest risks, as of Nov 15 (paper draft v6):

1. **At a few million tokens the gain looks like generic regularisation.** The gain is not concentrated in rare tokens. At 6.36M every annealed teacher-free target gives a similar improvement, while a trained teacher still pulls ahead by 0.013. MIR, the comparison the plan names for this, has not been run.
2. **The teacher-free advantage does not survive the data we have.** It is gone as a lead over other soft targets by 6.36M tokens. 6.36M is the largest human-written corpus available from the current sources, so the hoped-for 20M point was never tested.

Truncation bias, the risk the original plan put first, has been confirmed and repaired with annealing at the one scale where it showed.

**To confirm with the professor**

- [ ] Do static word vectors trained on the same corpus count as "no teacher"? If not, the shifted-PPMI + SVD source is the count-only fallback (implemented, not yet run).
- [x] Self-distillation is a teacher-based reference, not a teacher-free baseline. Born-Again (one CE teacher) and BabyLlama-2-style ensemble distillation (mean of the teachers' logits) are in draft v6. PS-KD was not run.
- [ ] Final deliverables: paper confirmed; video length to confirm.
- [x] Reference scale: about 30M parameters, in-house, small data, no teacher (the earlier "30B" was a mix-up).
- [x] Random initialisation is not required. We train from scratch by choice.

**Technical risks**

- **Seed counts.** Headline comparisons, the ablation ladder, self-distillation and the ensemble use 3 seeds. Several 6.36M baselines and the leave-one-out ablation use one seed. The runs still listed under Results are what kill criteria 1 (at full scale), 2 and 3 still need.
- **CE tuning depth.** Dropout and epochs were swept; learning rate and weight decay barely were. The data-constrained literature (Xu et al., Kim et al.) finds much stronger weight decay helps CE, and a stronger CE would shrink every margin in this document.
- **Scale-dependent schedule.** Annealing helps at 6.36M and hurts at 1M. A method whose schedule must be picked per scale needs a rule for picking it, or the paper must report both.
- **Rare tokens get worse at 6.36M.** The rarest-bin NLL rises by 0.45 over CE with annealing, and by 1.7 without it. If the course or reviewers weight rare-word quality, this is a weakness, not a neutral finding.
- **Risk proxies are heuristics.** On real data r\_ord has Spearman 0.33 with target NLL and r\_sem has 0.10. The mixture bound is a theorem given the component risks, but the component risks themselves are estimates.
- **Neighbours see only Kₛ tokens.** Context beyond Kₛ = 2 is ignored by the semantic part; its bias is absorbed into δ\_{Kₛ}. A self-bootstrapped alternative (neighbours from the student's own mid-layer hidden states) is implemented as `--embedding_source student_hidden` but not run. It would change the "corpus-only target" framing, so it is out of scope unless the static version fails.
- **Subword contexts.** With a 16k SentencePiece vocabulary, a Kₛ = 2 context is often less than one word. The phase 0 neighbours look sensible, but that is a manual check of a handful of queries.
- **Stage 2 vocabularies.** `gbd_loss` computes log-softmax over the full (batch, length, V) tensor. At V = 16k that fits in 16 GB. Larger vocabularies would need the loss and its backward pass run per sequence chunk.
- **Resolved since Nov 12:** corpus chosen and deduplicated; conversation boundaries and role-marker handling implemented and tested; `gbd.py` ported to version 2 (the five fixes listed in the Nov 12 draft); compute measured (4–20 minutes per run); D2GPo's prior checked (see the note under Baselines).

**To verify before the proposal goes out**

- [x] CoCoNTs re-read in full Nov 11 from the PDF: on the first k positions KL replaces next-token loss (Eq. 3), with no α in the objective or in the hyperparameter appendix (A.1).
- [x] NDP re-read Nov 11 from the PDF: prefix keys from the start of the sequence, fine-tuning experiments only.
- [x] CoCoNTs blocks are disjoint fixed-length blocks read in order (Sec. 4.1, A.3), and n-gram counts are corpus-wide (Eq. 2). Baseline 7a follows that.
- [x] D2GPo's exact prior checked against the released code (Nov 13). Faithful D2GPo equals label smoothing at V = 16k, so baseline 6 runs as the named variant D2GPo-NN64.
- [ ] Exact arXiv IDs of the papers cited from the literature review that were not re-opened (listed there).

## References

Newest first. Three papers were re-checked for this pitch: Semantic Smoothing (estimator, Theorems 3.1 and 4.2, experimental settings), CoCoNTs (full paper) and NDP (full paper). The rest are carried over from the Nov 6 literature review.

- Balasundaram, Narashiman, Mathur & Thangaraj 2026. [Semantic Smoothing for Language Models via Distribution Estimation and Embeddings](https://arxiv.org/abs/2605.07994). arXiv.
- Xu et al. 2026. [Data-Constrained LM Pretraining: Improved Regularization and Scaling Laws](https://arxiv.org/abs/2606.06888). arXiv.
- Kim, Kotha, Liang & Hashimoto 2026. [Pre-training under Infinite Compute](https://arxiv.org/abs/2509.14786). ICLR.
- Chang, Arnett, Tu & Bergen 2026. [Goldfish: Monolingual Language Models for 350 Languages](https://aclanthology.org/2026.lrec-1.300/). LREC.
- Sathe & Sarawagi 2024. [Efficient Training of Language Models with Compact and Consistent Next Token Distributions (CoCoNTs)](https://aclanthology.org/2024.findings-acl.717/). ACL Findings.
- Ruan et al. 2024. [NDP: Next Distribution Prediction as a More Broad Target](https://arxiv.org/html/2408.17377). arXiv.
- Malagutti et al. 2024. [The Role of n-gram Smoothing in the Age of Neural Networks](https://aclanthology.org/2024.naacl-long.382). NAACL.
- Liu et al. 2024. [Infini-gram](https://arxiv.org/abs/2401.17377). COLM.
- Timiryasov & Tastet 2023. [Baby Llama](https://arxiv.org/abs/2308.02019). BabyLM.
- Tastet & Timiryasov 2024. [BabyLlama-2: Ensemble-Distilled Models Consistently Outperform Teachers With Limited Data](https://aclanthology.org/2024.conll-babylm.26/). CoNLL BabyLM. The ensemble baseline averages teacher logits, then applies the tempered KL.
- Furlanello, Lipton, Tschannen, Itti & Anandkumar 2018. [Born Again Neural Networks](https://doi.org/10.48550/arxiv.1805.04770). ICML. The single-teacher self-distillation reference.
- Sarnthein et al. 2023. [Random Teachers are Good Teachers](https://arxiv.org/abs/2302.12091). ICML.
- Frydenlund et al. 2022. [Language Modelling via Learning to Rank](https://arxiv.org/abs/2110.06961). AAAI.
- Edman, Toral & van Noord 2021. [The Importance of Context in Very Low Resource Language Modeling](https://arxiv.org/abs/2205.04810). ICON.
- Kim et al. 2021. [PS-KD: Self-Knowledge Distillation with Progressive Refinement of Targets](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.html). ICCV.
- Yuan et al. 2020. [Revisiting Knowledge Distillation via Label Smoothing Regularization (Tf-KD)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.html). CVPR.
- Li et al. 2020. [Data-dependent Gaussian Prior Objective (D2GPo)](https://iclr.cc/virtual/2020/poster/1925). ICLR.
- Welch et al. 2020. [In-Domain Embedding Initialisation](https://arxiv.org/abs/2009.14109). arXiv.
- Szegedy et al. 2016. [Rethinking the Inception Architecture (label smoothing)](https://arxiv.org/abs/1512.00567). CVPR.
- Hinton, Vinyals & Dean 2015. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). NIPS workshop.
- Mikolov, Chen, Corrado & Dean 2013. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). ICLR workshop.
- Paninski 2003. Estimation of Entropy and Mutual Information. Neural Computation 15(6). Source of the variance bound in Method section 3.
- Chen & Goodman 1998. An Empirical Study of Smoothing Techniques for Language Modeling. Technical Report TR-10-98, Harvard University. Interpolated modified Kneser–Ney, used for baseline 7b.

## Revision log

**Nov 11, 2026.** CoCoNTs and NDP were re-read and the pitch corrected.

- CoCoNTs and NDP apply targets only where the n-gram is the model's whole context. The pitch had described them as sliding-window targets with a fixed α. Baseline 7 is renamed, and CoCoNTs proper is added as baseline 7a.
- Truncation bias from every-position targets is now the stated main risk, with a pilot (phase 2) and kill criterion 0 in front of the main runs.
- A risk calibration check on the synthetic corpus shows α\_t is nearly constant. Contribution 2 is conditional on repairing the variance term for n ≤ 3.
- Added: deduplication step, subword-context and compute risks, Frydenlund et al. and Edman et al. as prior art.
- `gbd.py` is unchanged. The Method section still matches it one to one.

**Nov 11, 2026, second pass.** CoCoNTs re-read in full from the PDF, and the scope set to English only.

- CoCoNTs: on the first k positions KL replaces next-token loss; there is no α anywhere; blocks are disjoint and fixed-length. Its perplexity gains are from fine-tuning pretrained checkpoints, and its from-scratch BabyLM result is downstream scores only. Neither CoCoNTs nor NDP reports from-scratch perplexity.
- English only: the cross-lingual variant, phase 6 and kill criterion 4 are removed, and GBD is claimed as a regulariser. Corpus candidates are BabyLM and WikiText-103 subsamples, grammaticality is BLiMP, and the large-model comparison is a ceiling.

**Nov 12, 2026.** Plan aligned with the project log. The method, `gbd.py` and every number in the Method and Reference implementation sections are unchanged.

- Course scale clarified: about 30M parameters, in-house, small data, no teacher. The "30B-level" question is closed and its anticipated-question answer rewritten.
- Initialisation fixed as from scratch (random init is not required by the course; chosen deliberately). Task set to conversation reply on human-written English dialogue only; synthetic chat data is excluded because it hides an LLM teacher.
- Students staged: Stage 1 is a 124M GPT-2-small-shaped model plus a ~30M model; Stage 2 scales up from scratch only if Stage 1 passes, capped by what full-parameter training fits in 16 GB. Recipe sweep and pilot move from 39M to 124M. The new phase 6 is this Stage 2 scale-up, not the removed cross-lingual phase.
- Primary metric is bits per byte on reply tokens. The large-model ceiling becomes an optional local reference.
- Added: hardware and the 32 GB target-building window, dialogue-formatting risk, and the five known `gbd.py` fixes needed before real-data runs.

**Nov 12, 2026, second pass.** The risk scale is replaced by version 2 after an audit of the math. The loss, the storage format and the experimental plan are unchanged.

- Variance term: ln(1 + (d̃ − 1)/n), a bound that holds for every n, replaces (d̂ − 1)/(2n), which could not exceed 1 nat. The support d̃ is shrunk toward a prior measured on contexts seen exactly twice.
- Bias fits: the noise term for nested contexts has a minus sign, rectification happens after the mean, the regression target is not rectified, and the plug-in KL uses a fixed total pseudo-count toward the unigram.
- Mixture: the semantic neighbours are pooled into one branch, λ\_t is explicit, and the risk of a mixture is the bound −ln Σ w e^(−r), which holds for any weights.
- α\_t is anchored on the unigram risk and reads the order-branch risk only. ρ is removed.
- Conversation boundaries and role markers are specified in Method sections 1 and 2.
- New: `gbd_v2_check.py` (brute-force test and a three-seed benchmark) and the version 2 check section. The semantic risk is reported as not calibrated.
- `gbd.py` is unchanged and still implements version 1.

**Nov 14, 2026.** Results added, and the document brought in line with the code and the runs from Nov 12–13. The Method equations are unchanged. They were already version 2, and `gbd.py` now implements them.

- New: a status note at the top, a Results section (main comparison, baselines, ablations, frequency bins, phase 0 on real data, runs still needed), evidence notes under each contribution, and a status for the central claim.
- The pre-registered central claim and kill criteria are unchanged. Each kill criterion now carries its outcome. Criterion 0 triggered at 6.36M tokens and was repaired with annealing. Criterion 1 holds at 1M and is open at 6.36M. Criteria 2 and 3 are not yet tested and failed, respectively; criterion 3's failure makes the MIR comparison required. Under criterion 5, per-token α\_t is off the contribution list until it is shown with 3 seeds, and leave-one-out stays.
- Setup corrected to what was run: the corpus is built (6.36M training tokens), so the data scales are 1M and 6.36M rather than 1M / 5M / 20M; the GPT-2-small-shaped student has 97.65M parameters at V = 16k, not 124M; the recipe is the one tuned on validation.
- Method section 6: the annealed α\_max schedule (repair 2) is written out.
- Reference implementation rewritten for the version 2 code and its tests. The version 1 risk calibration table is removed; the version 1 → version 2 calibration table in the version 2 check covers the same ground.
- Special-cases table: config names updated to the current switches, with a column for what has been run. Prefix-only targets and annealing are now implemented.
- Baseline 6 is D2GPo-NN64, with the reason stated. Kneser–Ney reference added.
- Anticipated questions and risks updated with measured numbers. Three questions added: whether annealing was post hoc, whether GBD is just regularisation, and whether anything was chosen on test.

**Nov 15, 2026.** Brought in line with paper draft v6 (`reports/GBD_paper_draft_v6.tex`). The Method equations are unchanged.

- The status note, the central claim, and the Results section now include Born-Again self-distillation and BabyLlama-2-style ensemble distillation, the BLiMP table, the frequency-share table, and the compute comparison.
- At 1M tokens the paper's claim is: GBD is the best teacher-free method by at least 0.032 reply BPB, ties one trained teacher, and recovers 80–83% of a two-teacher ensemble's gain.
- At 6.36M tokens the paper's claim is: annealed GBD still beats CE by 0.011, ties the other annealed soft targets, and loses to a trained teacher by 0.013.
- "Not run yet" no longer lists BLiMP or PS-KD. What remains open is MIR and static-embedding initialisation, plus extra seeds for the 6.36M smoothing baselines.

