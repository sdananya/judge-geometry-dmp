"""Both papers' metric families, in one small numpy-only module.

Pre-registered operationalizations (2026-07-09), fixed BEFORE any experimental comparison:

- Score matrices are items x columns (columns = rubrics for Paper-B-style data, or judge
  instances for single-rubric AITA-style data). We z-score each column across items unless
  the caller says otherwise, then take the subspace spanned by the columns.
- Principal angles: orthonormalize both column spans (thin SVD), angles = arccos of the
  singular values of Qa.T @ Qb (Bjorck & Golub 1973). We report the FULL set; summaries use
  the largest angle (Paper B appears to report a single angle; theirs is unspecified, ours
  is fixed here).
- Effective rank r95: min k s.t. top-k singular values carry >=95% of squared-singular-value
  mass of the column-centered matrix (Paper B's stated definition).
- Stretch/rotate/residual decomposition of a score change (Paper B Sec 4.8): energy of the
  centered delta along the base-judge direction, along the human direction orthogonalized
  against base, and everything else.
- Noise floor: exact E|P_h - P_j| for independent binomial proportion estimates — the value
  the |dP| alignment metric takes when model and humans have IDENTICAL true distributions.
"""

from __future__ import annotations

import numpy as np


# ---------- basic hygiene ----------

def zscore_columns(m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center and scale each column across rows (items). Constant columns become zero."""
    m = np.asarray(m, dtype=float)
    mu = m.mean(axis=0, keepdims=True)
    sd = m.std(axis=0, keepdims=True)
    return (m - mu) / np.maximum(sd, eps)


# ---------- Paper B family ----------

def sigma_ratio(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
    """Spread ratio sigma_J / sigma_H. Inputs are flat arrays of scores on the same scale."""
    hs = np.std(np.asarray(human_scores, dtype=float))
    if hs == 0:
        return np.nan
    return float(np.std(np.asarray(judge_scores, dtype=float)) / hs)


def effective_rank_r95(m: np.ndarray, threshold: float = 0.95, center: bool = True) -> int:
    """Paper B's r95 on an items x rubrics matrix: how many dimensions carry 95% of variance."""
    m = np.asarray(m, dtype=float)
    if center:
        m = m - m.mean(axis=0, keepdims=True)
    s = np.linalg.svd(m, compute_uv=False)
    energy = s**2
    total = energy.sum()
    if total == 0:
        return 0
    frac = np.cumsum(energy) / total
    return int(np.searchsorted(frac, threshold) + 1)


def _orthonormal_basis(m: np.ndarray, rank_tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for the column span of m (items x cols), numerically rank-aware."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    r = int((s > rank_tol * max(s[0], 1e-300)).sum()) if s.size else 0
    return u[:, :r]


def principal_angles_deg(a: np.ndarray, b: np.ndarray, zscore: bool = True) -> np.ndarray:
    """All principal angles (degrees, ascending) between column spans of a and b.

    a, b: items x cols score matrices sharing the same item rows. A 1-D array is a single
    score vector (the AITA case: human side = one P(acceptable) column).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if zscore:
        a = zscore_columns(a if a.ndim == 2 else a[:, None])
        b = zscore_columns(b if b.ndim == 2 else b[:, None])
    qa, qb = _orthonormal_basis(a), _orthonormal_basis(b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return np.array([])
    cos = np.linalg.svd(qa.T @ qb, compute_uv=False)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))  # svd gives descending cos, i.e. ascending angles


def largest_principal_angle_deg(a: np.ndarray, b: np.ndarray, zscore: bool = True) -> float:
    ang = principal_angles_deg(a, b, zscore=zscore)
    return float(ang[-1]) if ang.size else np.nan


def stacked_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two flat score vectors (stacked over items x rubrics)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def correlation_families(judge_matrix: np.ndarray, human_matrix: np.ndarray) -> dict:
    """Mean pairwise r within judges (r_ll), judge-to-human (r_lh), within humans (r_hh).

    judge_matrix: items x n_judges. human_matrix: items x n_humans (>=1 columns).
    """
    j = np.asarray(judge_matrix, dtype=float)
    h = np.asarray(human_matrix, dtype=float)
    if h.ndim == 1:
        h = h[:, None]

    def mean_pairwise(m1, m2, same):
        vals = []
        for i in range(m1.shape[1]):
            for k in range(m2.shape[1]):
                if same and k <= i:
                    continue
                vals.append(stacked_pearson(m1[:, i], m2[:, k]))
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "r_ll": mean_pairwise(j, j, same=True),
        "r_lh": mean_pairwise(j, h, same=False),
        "r_hh": mean_pairwise(h, h, same=True) if h.shape[1] > 1 else np.nan,
    }


def stretch_rotate_residual(delta: np.ndarray, base: np.ndarray, human: np.ndarray) -> dict:
    """Paper B Sec-4.8 energy decomposition of a score change (all vectors over the same items).

    delta = conditioned_scores - base_scores. Fractions sum to 1:
      stretch  — along the base judge's own (centered) direction: "more of the same"
      rotate   — along the human direction after removing the base direction: the only
                 component that can move the judge TOWARD humans
      residual — everything else (noise / off-axis change)
    """
    d = np.asarray(delta, dtype=float).ravel()
    d = d - d.mean()
    b = np.asarray(base, dtype=float).ravel()
    b = b - b.mean()
    h = np.asarray(human, dtype=float).ravel()
    h = h - h.mean()

    total = float(d @ d)
    if total == 0 or b @ b == 0 or h @ h == 0:
        return {"stretch": np.nan, "rotate": np.nan, "residual": np.nan, "total_energy": total}

    u_b = b / np.linalg.norm(b)
    h_perp = h - (h @ u_b) * u_b
    stretch = float((d @ u_b) ** 2) / total
    if np.linalg.norm(h_perp) < 1e-12:
        rotate = 0.0
    else:
        u_h = h_perp / np.linalg.norm(h_perp)
        rotate = float((d @ u_h) ** 2) / total
    return {
        "stretch": stretch,
        "rotate": rotate,
        "residual": 1.0 - stretch - rotate,
        "total_energy": total,
    }


# ---------- Paper A family ----------

def abs_diff_alignment(p_human: np.ndarray, p_model: np.ndarray) -> float:
    """Paper A's headline metric: mean per-item |P_human(acceptable) - P_model(acceptable)|."""
    ph = np.asarray(p_human, dtype=float)
    pm = np.asarray(p_model, dtype=float)
    return float(np.mean(np.abs(ph - pm)))


def noise_floor_absdiff(p: float, n_h: int, n_j: int) -> float:
    """Exact E|P_h - P_j| when BOTH sides sample from the SAME true Bernoulli(p).

    This is the score a PERFECTLY aligned model gets on this metric — any reported
    alignment gap must be compared against it, not against zero.
    """
    kh = np.arange(n_h + 1)
    kj = np.arange(n_j + 1)
    from math import lgamma

    def log_binom_pmf(k, n, pp):
        if pp in (0.0, 1.0):
            out = np.full_like(k, -np.inf, dtype=float)
            out[k == (n if pp == 1.0 else 0)] = 0.0
            return out
        lg = np.vectorize(lambda kk: lgamma(n + 1) - lgamma(kk + 1) - lgamma(n - kk + 1))
        return lg(k) + k * np.log(pp) + (n - k) * np.log1p(-pp)

    ph = np.exp(log_binom_pmf(kh, n_h, float(p)))
    pj = np.exp(log_binom_pmf(kj, n_j, float(p)))
    diff = np.abs(kh[:, None] / n_h - kj[None, :] / n_j)
    return float(ph @ diff @ pj)


def noise_floor_for_items(p_human: np.ndarray, n_h: int, n_j: int) -> float:
    """Mean exact floor across items, using each item's observed human rate as its true p."""
    return float(np.mean([noise_floor_absdiff(p, n_h, n_j) for p in np.asarray(p_human, float)]))


def top_k_mass(counts: np.ndarray, k: int = 10) -> float:
    """Fraction of all value mentions carried by the k most-used values (Paper A: 81.6% vs 35.2%)."""
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    tot = c.sum()
    return float(c[:k].sum() / tot) if tot > 0 else np.nan


def normalized_entropy(dist: np.ndarray) -> float:
    """Shannon entropy of a distribution, normalized to [0,1] by log(support size)."""
    d = np.asarray(dist, dtype=float)
    d = d[d > 0]
    if d.size <= 1:
        return 0.0
    d = d / d.sum()
    return float(-(d * np.log(d)).sum() / np.log(d.size))
