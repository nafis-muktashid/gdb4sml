"""
D2GPo-NN64: neighbour-restricted variant of D2GPo
(Li et al., ICLR 2020, OpenReview S1efxTVYDr).

With sigma=1, T=2 on a V=16000 vocabulary the full Eq. 7 prior puts 99.6% of
its mass at ranks >= 64 and is within 1.4e-5 of uniform at every rank, so
faithful D2GPo at the released defaults equals label smoothing with weight
lambda. Truncating to r_d renormalises onto the r_d nearest neighbours, which
is a different (stronger) prior. This module is that neighbour-restricted
variant (default r_d = 64), not the full-vocabulary prior.

Prior, as the paper defines it:
  Eq. (8)  dist(i, j) = cosine similarity of the token embeddings.
  Eq. (9)  ORDER(y_i) ranks every token by that similarity. y_i itself is rank 0
           (the authors' generator argsorts -cosine; the appendix's "small to large"
           wording calls cosine a distance, but the gold token is top priority).
  Sec. 4   f(rank) = Normal(mean=0, std=sigma).pdf(rank), the Gaussian density
           on that order. sigma = 1 and offset = 0 in the released generator.
  Eq. (7)  q_i = exp(f_i / T) / sum_j exp(f_j / T). T = 2.0 in the released
           criterion (`--d2gpo-temperature`, default 2.0) and in the WMT14 script.

The PDF states the formulas above but does not print numeric defaults. sigma=1,
T=2.0 and lambda=0.1 are the defaults of the authors' training criterion
(`--d2gpo-alpha 0.1`) used for the reported Transformer runs.

Loss storage (task, matching the released nn.KLDivLoss direction rather than the
KL(p || q) wording of Eq. (4)):
  nll_weight = 1, alpha = lambda,
  so gbd_loss is MLE + lambda * H(q, p) = MLE + lambda * KL(q || p) + const,
  and H(q) does not depend on the model. Eq. (6) is L_MLE + lambda * L_O.
Temperature is applied to the prior only (Eq. 7). It is not applied to the
model distribution. The released criterion also multiplies its KL term by T^2
and down-weights the smoothed MLE by (1 - lambda); that mixture is not used
here, because alpha is lambda itself so --alpha_schedule / --anneal_alpha
scale only the KL term.

The stored row keeps ranks 0..r_d-1 and renormalises them. Those ranks are the
r_d nearest tokens, because the Gaussian density on ranks 0,1,2,... is strictly
decreasing. Every row has the same probability vector. That renormalised head
is D2GPo-NN64, not Eq. 7 itself.
"""
from __future__ import annotations

import numpy as np

from gbd import GBDTargets


def gaussian_rank_pdf(ranks: np.ndarray, sigma: float) -> np.ndarray:
    """Normal(0, sigma) density. Matches scipy.stats.norm(0, sigma).pdf."""
    if sigma <= 0.0:
        raise ValueError(f"d2gpo sigma must be positive, got {sigma}")
    z = np.asarray(ranks, dtype=np.float64) / float(sigma)
    return np.exp(-0.5 * z * z) / (float(sigma) * np.sqrt(2.0 * np.pi))


def build_d2gpo_prior(
    E: np.ndarray,
    sigma: float = 1.0,
    temp: float = 2.0,
    r_d: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids (V, r_d), probs (r_d,)).

    probs[j] is the renormalised Eq. (7) mass of rank j. ids[i, j] is the token
    at rank j for gold token i, with ids[i, 0] == i.
    """
    if temp <= 0.0:
        raise ValueError(f"d2gpo temperature must be positive, got {temp}")
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"embeddings must have shape (V, D), got {E.shape}")
    V = int(E.shape[0])
    r_d = int(min(max(r_d, 1), V))
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    # Eq. (7) over the full vocabulary, then keep the head and renormalise.
    # The  Normal density depends only on rank, so every gold token shares `probs`.
    ranks = np.arange(V, dtype=np.float64)
    logits = gaussian_rank_pdf(ranks, sigma) / float(temp)
    logits = logits - logits.max()
    q = np.exp(logits)
    q /= q.sum()
    probs = q[:r_d]
    probs = probs / probs.sum()

    ids = np.empty((V, r_d), dtype=np.int32)
    block = 256
    eye = np.arange(V)
    for s in range(0, V, block):
        e = min(s + block, V)
        sims = En[s:e] @ En.T  # (block, V); never a dense (V, V) matrix
        local = np.arange(e - s)
        sims[local, eye[s:e]] = np.inf  # gold token is rank 0 even if another row is identical
        if r_d < V:
            part = np.argpartition(-sims, r_d - 1, axis=1)[:, :r_d]
        else:
            part = np.broadcast_to(eye, (e - s, V)).copy()
        part_sims = np.take_along_axis(sims, part, axis=1)
        # Stable tie-break: smaller token id first, then higher similarity.
        by_id = np.argsort(part, axis=1, kind="mergesort")
        part = np.take_along_axis(part, by_id, axis=1)
        part_sims = np.take_along_axis(part_sims, by_id, axis=1)
        by_sim = np.argsort(-part_sims, axis=1, kind="mergesort")
        ids[s:e] = np.take_along_axis(part, by_sim, axis=1).astype(np.int32, copy=False)
    return ids, probs.astype(np.float64, copy=False)


def build_d2gpo_targets(
    tokens: np.ndarray,
    E: np.ndarray,
    sigma: float = 1.0,
    temp: float = 2.0,
    lam: float = 0.1,
    r_d: int = 64,
) -> GBDTargets:
    """Per-position target is the prior row of the gold token x_t.

    nll_weight = 1 and alpha = lambda, so the training loss is
    MLE + lambda * H(q, p). Temperature stays inside q (Eq. 7); the model
    distribution is not divided by T.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    T = int(len(tokens))
    ids_table, probs = build_d2gpo_prior(E, sigma=sigma, temp=temp, r_d=r_d)
    r = int(probs.shape[0])
    if T == 0:
        z = np.zeros(0, dtype=np.float32)
        return GBDTargets(
            ids=np.zeros((0, r), np.int32),
            probs=np.zeros((0, r), np.float32),
            alpha=z,
            R=z,
            w_exact=z,
            k_star=np.zeros(0, np.int16),
            nll_weight=z,
            fitted={"sigma": float(sigma), "temp": float(temp), "lambda": float(lam), "r_d": r},
        )
    out_ids = ids_table[tokens]
    out_p = np.broadcast_to(probs.astype(np.float32, copy=False), (T, r)).copy()
    z = np.zeros(T, dtype=np.float32)
    return GBDTargets(
        ids=out_ids,
        probs=out_p,
        alpha=np.full(T, float(lam), dtype=np.float32),
        R=z,
        w_exact=z,
        k_star=np.zeros(T, dtype=np.int16),
        nll_weight=np.ones(T, dtype=np.float32),
        fitted={
            "sigma": float(sigma),
            "temp": float(temp),
            "lambda": float(lam),
            "r_d": r,
            "temperature_on_model": False,
        },
    )
