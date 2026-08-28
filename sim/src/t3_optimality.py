"""
T3: which perturbations can be covert at all?

T1 and T2 analysed ONE perturbation: boost pilots by (1+eta), deboost
data to hold energy fixed.  The obvious objection is that the negative
result might be an artifact of that particular choice.  This module
answers the general question in closed form.

Setup.  A perturbation is a direction v in R^N acting on RE amplitudes,
    A_k(eta) = 1 + eta v_k + O(eta^2).
The nuisance is a vector of unknown gains g_1..g_L acting on a
partition of the REs into "nuisance blocks" B_1..B_L (one unknown
complex gain per coherence block; L = 1 is the flat-channel case of
T1/T2).

Two functionals of v matter:

  sensing gain rate   s(v) = sum_{k in P} v_k
      because a passive receiver's cross-ambiguity reference is the
      deterministic pilot component alone (T1), so its matched-filter
      energy is  N_p + 2 eta s(v) + O(eta^2).

  detectability       I_eff(v) = v' M v,   M = A - sum_l b_l b_l' / S_l
      the adversary's efficient Fisher information about eta after
      profiling out every g_l, with A = diag(J_k),
      b_l = restriction of (J_k) to block l, S_l = sum_{k in B_l} J_k.

The covertness-limited sensing gain scales as s/sqrt(I_eff), because
the budget gives eta <= c/sqrt(I_eff) and the gain is proportional to
eta*s.  So the figure of merit is the Rayleigh quotient

    Psi(v) = s(v)^2 / I_eff(v)                  "covert sensing efficiency"

maximised subject to power neutrality 1'v = 0.  Note Psi is invariant
to the scale of v, as it must be.

Key structural fact: M is singular, and
    null(M) = { block-constant vectors } = range(G).
A block-constant v is exactly a per-block gain change, which the
nuisance absorbs -- undetectable, and the reason the power constraint
is needed for the problem to be well posed.
"""

from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------
# Cell reduction
# ---------------------------------------------------------------
# The objective and both constraints are invariant to permuting REs
# within each (block x {pilot,data}) cell, and v'Mv is convex, so
# replacing v by its cell-average leaves s(v) and 1'v unchanged and
# cannot increase v'Mv (Jensen).  The optimum may therefore be taken
# constant on cells, reducing an N-dimensional problem to 2L.
# verify_t3.py checks this reduction against brute force.

def build_cells(n_p_per_block, n_d_per_block, j_p, j_d):
    """Return (counts, J, block_index) over the 2L cells, ordered
    [block0-pilot, block0-data, block1-pilot, ...]."""
    counts, jvec, blk = [], [], []
    for l, (npb, ndb) in enumerate(zip(n_p_per_block, n_d_per_block)):
        counts += [npb, ndb]
        jvec += [j_p, j_d]
        blk += [l, l]
    return np.array(counts, float), np.array(jvec, float), np.array(blk)


def fisher_M(counts, jvec, blk):
    """Efficient-information quadratic form M in cell coordinates.

    In cell coordinates a direction v assigns one value per cell, and
        I_etaeta = sum_c n_c J_c v_c^2                        -> A
        I_eta,g_l = sum_{c in block l} n_c J_c v_c            -> b_l
        I_g_l,g_l = sum_{c in block l} n_c J_c                -> S_l
    with I_gg diagonal because blocks are disjoint.
    """
    w = counts * jvec                       # n_c J_c
    A = np.diag(w)
    M = A.copy()
    for l in np.unique(blk):
        m = (blk == l)
        b = np.where(m, w, 0.0)
        S = w[m].sum()
        M -= np.outer(b, b) / S
    return M


def psi_max(counts, jvec, blk, tol=1e-9):
    """Maximum covert sensing efficiency subject to power neutrality.

    Returns (Psi_max, v_opt).  Psi_max = inf means covert gain is
    available at zero detectability -- the pilot-density loophole.
    """
    M = fisher_M(counts, jvec, blk)
    p = np.where(np.arange(len(counts)) % 2 == 0, counts, 0.0)  # pilots
    w = counts.copy()                                            # power

    # basis for {v : w'v = 0}
    B = np.linalg.svd(w[None, :])[2][1:].T        # (2L, 2L-1)
    Mr = B.T @ M @ B
    pr = B.T @ p

    ev = np.linalg.eigvalsh(Mr)
    if ev.min() <= tol * max(1.0, ev.max()):
        # M singular on the constraint subspace: is p orthogonal to the
        # null direction?  If not, Psi is unbounded.
        U = np.linalg.eigh(Mr)[1][:, ev <= tol * max(1.0, ev.max())]
        if np.linalg.norm(U.T @ pr) > 1e-8:
            return math.inf, B @ U[:, 0]
        Mr_pinv = np.linalg.pinv(Mr)
        return float(pr @ Mr_pinv @ pr), B @ (Mr_pinv @ pr)
    v = B @ np.linalg.solve(Mr, pr)
    return float(pr @ np.linalg.solve(Mr, pr)), v


# ---------------------------------------------------------------
# Closed forms for the homogeneous (single-block) case
# ---------------------------------------------------------------
def psi_max_homogeneous(n_p, n_d, j_p, j_d) -> float:
    """Single nuisance block.  Minimising J_p sum_P v^2 + J_d sum_D v^2
    at fixed s with sum_P v = s, sum_D v = -s makes v constant within
    each class, giving

        Psi_max = 1 / [ J_p/N_p + J_d/N_d - (J_p - J_d)^2 / S ],
        S = N_p J_p + N_d J_d.

    For J_p = J_d = J this collapses to Psi_max = N_p N_d / (N J).
    """
    S = n_p * j_p + n_d * j_d
    return 1.0 / (j_p / n_p + j_d / n_d - (j_p - j_d) ** 2 / S)


def optimal_direction_homogeneous(n_p, n_d):
    """The maximiser is v_k = 1/N_p on pilots, -1/N_d on data, i.e.
    proportional to (1, -a) with a = N_p/N_d -- EXACTLY the
    power-neutral perturbation analysed in T1 and T2."""
    return 1.0 / n_p, -1.0 / n_d


def psi_of_scheme(n_p, n_d, j_p, j_d, v_p, v_d) -> float:
    """Psi for an arbitrary single-block scheme, for comparison."""
    s = n_p * v_p
    i_ee = n_p * j_p * v_p ** 2 + n_d * j_d * v_d ** 2
    i_eg = n_p * j_p * v_p + n_d * j_d * v_d
    i_gg = n_p * j_p + n_d * j_d
    i_eff = i_ee - i_eg ** 2 / i_gg
    return s * s / i_eff if i_eff > 0 else math.inf


# ---------------------------------------------------------------
# The dichotomy
# ---------------------------------------------------------------
def covert_gain_available(n_p_per_block, n_d_per_block) -> bool:
    """Unbounded covert gain exists iff some block-constant, power-
    neutral direction has nonzero pilot sum:

        exists c with  sum_l c_l N^(l) = 0  and  sum_l c_l N_p^(l) != 0

    which holds iff the vectors (N_p^(l)) and (N^(l)) are NOT parallel,
    i.e. iff the PILOT DENSITY VARIES across nuisance blocks.
    """
    npb = np.asarray(n_p_per_block, float)
    ndb = np.asarray(n_d_per_block, float)
    ntot = npb + ndb
    alpha = npb / ntot
    return bool(np.ptp(alpha) > 1e-12)


def optimal_loophole(n_p_per_block, n_d_per_block):
    """When densities differ, the zero-detectability direction with the
    largest pilot sum: block-constant c, power-neutral, maximising
    sum_l c_l N_p^(l).  Shift power toward pilot-dense blocks."""
    npb = np.asarray(n_p_per_block, float)
    ndb = np.asarray(n_d_per_block, float)
    ntot = npb + ndb
    # maximise c.npb s.t. c.ntot = 0, ||c||=1  ->  c ∝ npb - (npb.ntot/|ntot|^2) ntot
    c = npb - (npb @ ntot) / (ntot @ ntot) * ntot
    nrm = np.linalg.norm(c)
    return (c / nrm, float(npb @ (c / nrm))) if nrm > 1e-12 else (c, 0.0)
