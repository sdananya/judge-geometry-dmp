"""Unit tests on synthetic data with KNOWN geometry — the pipeline must recover planted
ranks, angles, spreads, and decompositions before it is allowed near real data."""

import numpy as np
import pytest

from geometry import (
    abs_diff_alignment,
    correlation_families,
    effective_rank_r95,
    largest_principal_angle_deg,
    noise_floor_absdiff,
    normalized_entropy,
    principal_angles_deg,
    sigma_ratio,
    stacked_pearson,
    stretch_rotate_residual,
    top_k_mass,
)

RNG = np.random.default_rng(20260709)
N_ITEMS = 400


# ---------- spread ----------

def test_sigma_ratio_recovers_planted_spread():
    human = RNG.normal(0, 1.0, 5000)
    judge = RNG.normal(0, 0.4, 5000)
    assert sigma_ratio(judge, human) == pytest.approx(0.4, abs=0.02)


def test_sigma_ratio_constant_human_is_nan():
    assert np.isnan(sigma_ratio(RNG.normal(size=10), np.ones(10)))


# ---------- effective rank ----------

def test_effective_rank_recovers_planted_rank():
    # 6 columns, but only 2 independent directions + tiny noise
    latent = RNG.normal(size=(N_ITEMS, 2))
    mix = RNG.normal(size=(2, 6))
    m = latent @ mix + 1e-4 * RNG.normal(size=(N_ITEMS, 6))
    assert effective_rank_r95(m) == 2


def test_effective_rank_full_rank_noise():
    m = RNG.normal(size=(N_ITEMS, 4))
    assert effective_rank_r95(m) == 4


# ---------- principal angles ----------

def _orthonormal_columns(n, k, rng):
    q, _ = np.linalg.qr(rng.normal(size=(n, k)))
    return q[:, :k]


def test_planted_angle_recovered():
    # subspace B = subspace A with one direction rotated by exactly theta into a fresh direction
    for theta in (15.0, 45.0, 80.0):
        basis = _orthonormal_columns(N_ITEMS, 3, RNG)  # a1, a2, fresh
        a = basis[:, :2]
        b = np.column_stack([
            basis[:, 0],
            np.cos(np.radians(theta)) * basis[:, 1] + np.sin(np.radians(theta)) * basis[:, 2],
        ])
        got = largest_principal_angle_deg(a, b, zscore=False)
        assert got == pytest.approx(theta, abs=0.5)


def test_identical_subspaces_zero_angle():
    a = RNG.normal(size=(N_ITEMS, 3))
    assert largest_principal_angle_deg(a, 2.0 * a + 5.0) == pytest.approx(0.0, abs=1e-4)


def test_independent_random_vectors_near_orthogonal():
    # in high dimension, independent directions sit near 90 degrees
    a = RNG.normal(size=(5000,))
    b = RNG.normal(size=(5000,))
    assert largest_principal_angle_deg(a, b) == pytest.approx(90.0, abs=5.0)


def test_noisy_copies_of_same_latent_have_small_angle():
    latent = RNG.normal(size=N_ITEMS)
    a = latent + 0.2 * RNG.normal(size=N_ITEMS)
    b = latent + 0.2 * RNG.normal(size=N_ITEMS)
    assert largest_principal_angle_deg(a, b) < 25.0


# ---------- correlations ----------

def test_correlation_families_shared_bias_pattern():
    # judges = one shared latent + small idiosyncrasy; humans = a DIFFERENT latent
    g_judge = RNG.normal(size=N_ITEMS)
    g_human = RNG.normal(size=N_ITEMS)
    judges = np.column_stack([g_judge + 0.5 * RNG.normal(size=N_ITEMS) for _ in range(5)])
    humans = np.column_stack([g_human + 0.8 * RNG.normal(size=N_ITEMS) for _ in range(3)])
    fam = correlation_families(judges, humans)
    assert fam["r_ll"] > 0.6          # judges agree with each other
    assert abs(fam["r_lh"]) < 0.15    # ...but not with humans
    assert fam["r_hh"] > 0.3          # humans share their own signal


def test_stacked_pearson_perfect():
    x = RNG.normal(size=100)
    assert stacked_pearson(x, 3 * x + 1) == pytest.approx(1.0)


# ---------- stretch / rotate / residual ----------

def test_decomposition_pure_stretch():
    base = RNG.normal(size=N_ITEMS)
    human = RNG.normal(size=N_ITEMS)
    out = stretch_rotate_residual(delta=0.7 * base, base=base, human=human)
    assert out["stretch"] == pytest.approx(1.0, abs=1e-9)
    assert out["rotate"] == pytest.approx(0.0, abs=1e-9)


def test_decomposition_pure_rotation_toward_human():
    base = RNG.normal(size=N_ITEMS)
    human = RNG.normal(size=N_ITEMS)
    b = base - base.mean()
    h = human - human.mean()
    u_b = b / np.linalg.norm(b)
    h_perp = h - (h @ u_b) * u_b
    out = stretch_rotate_residual(delta=h_perp, base=base, human=human)
    assert out["rotate"] == pytest.approx(1.0, abs=1e-9)
    assert out["stretch"] == pytest.approx(0.0, abs=1e-9)


def test_decomposition_fractions_sum_to_one():
    base = RNG.normal(size=N_ITEMS)
    human = RNG.normal(size=N_ITEMS)
    delta = RNG.normal(size=N_ITEMS)
    out = stretch_rotate_residual(delta, base, human)
    assert out["stretch"] + out["rotate"] + out["residual"] == pytest.approx(1.0, abs=1e-9)
    assert 0 <= out["residual"] <= 1


# ---------- noise floor ----------

def test_noise_floor_matches_theory_at_half():
    # normal approx at p=.5, n=32/32: sqrt(2/pi)*sqrt(2*.25/32) ~ 0.0997
    assert noise_floor_absdiff(0.5, 32, 32) == pytest.approx(0.0997, abs=0.004)


def test_noise_floor_matches_monte_carlo():
    rng = np.random.default_rng(7)
    for p, nh, nj in [(0.5, 32, 32), (0.8, 32, 16), (0.95, 40, 40)]:
        mc = np.abs(
            rng.binomial(nh, p, 200_000) / nh - rng.binomial(nj, p, 200_000) / nj
        ).mean()
        assert noise_floor_absdiff(p, nh, nj) == pytest.approx(mc, abs=0.002)


def test_noise_floor_shrinks_with_more_samples():
    assert noise_floor_absdiff(0.5, 500, 500) < noise_floor_absdiff(0.5, 32, 32)


def test_paper_a_low_consensus_claim_is_below_floor():
    # Paper A reports ~5pp residual on contested items; the floor at p~.5 with ~32/side is ~10pp.
    assert noise_floor_absdiff(0.5, 32, 32) > 0.05


# ---------- Paper A distribution metrics ----------

def test_abs_diff_alignment_zero_when_identical():
    p = RNG.uniform(size=50)
    assert abs_diff_alignment(p, p) == 0.0


def test_top_k_mass_concentrated_vs_flat():
    concentrated = np.array([100] * 10 + [1] * 50)
    flat = np.ones(60)
    assert top_k_mass(concentrated, 10) > 0.9
    assert top_k_mass(flat, 10) == pytest.approx(10 / 60)


def test_normalized_entropy_bounds():
    assert normalized_entropy(np.ones(60)) == pytest.approx(1.0)
    d = np.zeros(60)
    d[0] = 1.0
    assert normalized_entropy(d) == 0.0
