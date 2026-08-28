"""
T1: derived perturbation law for power-neutral pilot perturbation.

Replaces the phenomenological pair that is often assumed,
    sensing gain  rho_r(eta)/rho_r(0) = (1 + eta)          [assumed]
    signature     Delta_S_RF(eta)     = eta^2 / 2          [assumed]
with quantities derived from the OFDM resource grid.

Notation (one OFDM symbol, active subcarrier set of size N):
    N_p    number of pilot (reference-signal) REs
    N_d    number of data REs,  N = N_p + N_d
    alpha  pilot fraction N_p / N
    a      pilot-to-data RE ratio N_p / N_d = alpha / (1 - alpha)
    eta    pilot amplitude perturbation, pilots scaled by (1 + eta)
    beta   data amplitude scaling enforcing power neutrality
    gamma  observer SNR |h|^2 / sigma^2 (linear)
    kappa  gamma / (1 + gamma), the observer's signal-power fraction

All symbols are constant-modulus (QPSK data, unit-modulus DM-RS), so
per-RE received power is a scaled noncentral chi-square with two
degrees of freedom in every case; this is used throughout instead of
the Gaussian-codebook approximation.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------
# Power neutrality
# ---------------------------------------------------------------
def beta_sq(eta: float, a: float) -> float:
    """Data power scaling beta^2 that holds total slot energy fixed.

    Constraint  N_p (1+eta)^2 + N_d beta^2 = N_p + N_d  gives
        beta^2 = 1 - a (2 eta + eta^2),      a = N_p / N_d.
    """
    return 1.0 - a * (2.0 * eta + eta * eta)


def eta_max(alpha: float) -> float:
    """Largest feasible eta before beta^2 <= 0.

    beta^2 > 0  <=>  (1+eta)^2 < 1/alpha  <=>  eta < 1/sqrt(alpha) - 1.
    """
    return 1.0 / math.sqrt(alpha) - 1.0


# ---------------------------------------------------------------
# (a) Sensing gain
# ---------------------------------------------------------------
def sensing_gain(eta: float, n_p: float = 1.0, n_p0: float = 1.0) -> float:
    """Post-integration SNR gain of the passive sensing channel.

    An uncooperative passive receiver cannot reconstruct the data
    payload without decoding, so its cross-ambiguity reference is the
    deterministic reference-signal component only.  Coherent
    matched-filter gain is therefore proportional to the *pilot*
    energy N_p (1+eta)^2, giving

        G(eta, N_p) = (N_p / N_p0) (1 + eta)^2.

    Note the exponent: the derived gain is (1+eta)^2, not the (1+eta)
    that a phenomenological model would assume.
    """
    return (n_p / n_p0) * (1.0 + eta) ** 2


# ---------------------------------------------------------------
# (b) Observer divergence -- three adversary classes
# ---------------------------------------------------------------
def kl_a0(eta: float, a: float) -> float:
    """A0, energy detector: total received power only.

    Power neutrality makes the total slot energy identical under pi
    and pi_0, so the energy detector's observation law is unchanged
    and the divergence is exactly zero for every eta.  This is what
    licenses the term "emission-neutral".
    """
    return 0.0


def fisher_a1(gamma: float, n_p: float, n_d: float) -> float:
    """A1, structural adversary: per-RE powers, known pilot locations,
    UNKNOWN channel gain h.

    Any statistic must be invariant to scaling by |h|, so the
    adversary is confined to ratios.  The natural (asymptotically
    sufficient) invariant is the pilot-to-data power ratio

        T = (1/N_p sum_{k in P} |Z_k|^2) / (1/N_d sum_{k in D} |Z_k|^2)

    whose mean and variance give the Fisher information about eta at
    eta = 0:

        I_A1(0) = [2 (1+a) kappa]^2 / [ (1 - kappa^2)(1/N_p + 1/N_d) ]

    with kappa = gamma / (1 + gamma) and a = N_p / N_d.
    """
    a = n_p / n_d
    kappa = gamma / (1.0 + gamma)
    num = (2.0 * (1.0 + a) * kappa) ** 2
    den = (1.0 - kappa ** 2) * (1.0 / n_p + 1.0 / n_d)
    return num / den


def kl_a1(eta: float, gamma: float, n_p: float, n_d: float) -> float:
    """Per-OFDM-symbol divergence seen by A1, to second order in eta."""
    return 0.5 * fisher_a1(gamma, n_p, n_d) * eta * eta


def kl_a2(eta: float, gamma: float, n_p: float) -> float:
    """A2, coherent genie: complex observation, known channel h and
    known pilot sequence.

    Pilot RE k has law CN(h(1+eta)S_k, sigma^2) under pi and
    CN(h S_k, sigma^2) under pi_0, so the per-RE divergence is
    |h|^2 eta^2 / sigma^2 = gamma eta^2 exactly, and

        D_A2(eta) = N_p gamma eta^2      (per OFDM symbol).

    Restricting A2 to pilot REs only makes this a strict LOWER bound
    on the genie's divergence, which is the direction needed to prove
    that covertness fails.
    """
    return n_p * gamma * eta * eta


def fisher_a2(gamma: float, n_p: float) -> float:
    """I_A2(0) = 2 N_p gamma, so that kl_a2 = 0.5 * I * eta^2."""
    return 2.0 * n_p * gamma


# ---------------------------------------------------------------
# Per-RE divergence (single-RE observation scope)
# ---------------------------------------------------------------
def re_power_moments(amp: float, gamma: float) -> tuple[float, float]:
    """Mean and variance of |Z|^2 for one constant-modulus RE of
    amplitude `amp` observed at SNR gamma (|h| = 1, sigma^2 = 1/gamma).

    Z = amp * S + W, |S| = 1, W ~ CN(0, sigma^2), so |Z|^2 is a scaled
    noncentral chi-square with two degrees of freedom:
        E  = amp^2 + sigma^2
        Var = sigma^4 + 2 amp^2 sigma^2
    """
    s2 = 1.0 / gamma
    return amp * amp + s2, s2 * s2 + 2.0 * amp * amp * s2


def kl_per_re(eta: float, gamma: float) -> float:
    """Small-eta divergence of a SINGLE pilot-RE power sample.

    This is the per-RE quantity sometimes written Delta_S_RF and
    compared against eta^2 / 2.  Deflection of |Z|^2:

        d = (2 kappa eta)^2 / (1 - kappa^2),     KL ~ d / 2.

    It is smaller than the slot-level divergence by roughly the
    number of observed REs, so it must not be budgeted as if it were
    a per-slot quantity.
    """
    kappa = gamma / (1.0 + gamma)
    d = (2.0 * kappa * eta) ** 2 / (1.0 - kappa ** 2)
    return 0.5 * d


def kl_per_re_exact(eta: float, gamma: float) -> float:
    """Moment-matched Gaussian KL for one pilot-RE power sample, using
    the exact noncentral-chi-square moments rather than the small-eta
    expansion.  This is what a moment-matched plug-in estimator
    actually measures, so it is the right verification target at
    larger eta where kl_per_re() has visible curvature error.
    """
    m0, v0 = re_power_moments(1.0, gamma)
    m1, v1 = re_power_moments(1.0 + eta, gamma)
    return 0.5 * (math.log(v0 / v1) + v1 / v0 + (m1 - m0) ** 2 / v0 - 1.0)


def kl_moment_matched(s0, s1) -> float:
    """Moment-matched Gaussian KL from two sample arrays.

    The plug-in estimator is upward biased: both the (dmu)^2 and the
    variance-mismatch term are positive-definite quadratics in
    quantities estimated with O(1/n) error, so E[KLhat] = KL + b/n.
    At small eta this bias can exceed the signal, which is why a
    plug-in covertness estimate reads high there.  Use
    kl_moment_matched_debiased() to remove the leading term.
    """
    import numpy as np
    m0, v0 = float(np.mean(s0)), float(np.var(s0))
    m1, v1 = float(np.mean(s1)), float(np.var(s1))
    return 0.5 * (math.log(v0 / v1) + v1 / v0 + (m1 - m0) ** 2 / v0 - 1.0)


def kl_moment_matched_debiased(s0, s1) -> float:
    """Richardson extrapolation over sample size.

    With bias b/n, the half-sample estimate carries bias 2b/n, so
    2*KLhat(n) - KLhat(n/2) cancels the leading term.
    """
    import numpy as np
    full = kl_moment_matched(s0, s1)
    h = min(len(s0), len(s1)) // 2
    halves = [kl_moment_matched(s0[i * h:(i + 1) * h], s1[i * h:(i + 1) * h])
              for i in (0, 1)]
    return 2.0 * full - float(np.mean(halves))


# ---------------------------------------------------------------
# Mission-level covertness budget (feeds T2)
# ---------------------------------------------------------------
def eta_budget(delta: float, fisher: float, n_sym: int, n_slots: int) -> float:
    """Largest eta meeting TV <= delta over n_slots slots.

    KL accumulates additively over independent REs, symbols and slots,
    so total KL = 0.5 * I(0) * eta^2 * n_sym * n_slots.  Pinsker gives
    TV <= sqrt(KL/2) <= delta, hence

        eta <= 2 delta / sqrt(I(0) n_sym n_slots).
    """
    return 2.0 * delta / math.sqrt(fisher * n_sym * n_slots)


# ---------------------------------------------------------------
# (c) Rate cost
# ---------------------------------------------------------------
def rate_cost(eta: float, a: float, rho_c: float, n_avg: float) -> dict:
    """Communication rate under the power-neutral perturbation.

    Data REs carry power beta^2, so their SNR is beta^2 rho_c; pilots
    carry (1+eta)^2, which sharpens the MMSE channel estimate.  With
    n_avg pilots averaged inside a coherence patch,

        sigma_e^2 = 1 / (1 + (1+eta)^2 rho_c n_avg)
        rho_eff   = beta^2 rho_c / (1 + beta^2 rho_c sigma_e^2)

    The two effects oppose, so rho_eff can be non-monotone in eta.
    """
    b2 = beta_sq(eta, a)
    sig_e2 = 1.0 / (1.0 + (1.0 + eta) ** 2 * rho_c * n_avg)
    rho_eff = b2 * rho_c / (1.0 + b2 * rho_c * sig_e2)
    return dict(beta_sq=b2, sigma_e_sq=sig_e2, rho_eff=rho_eff,
                rate=math.log2(1.0 + rho_eff))
