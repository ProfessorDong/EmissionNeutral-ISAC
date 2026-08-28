"""
T2: minimax sequential covertness for power-neutral pilot perturbation.

T1 derived the per-symbol divergence against three fixed adversary
classes.  T2 asks the harder question: what can the BEST adversary do,
when it does not know the propagation channel, may apply any statistic
it likes to the raw IQ, sits at an unknown standoff, and observes M
slots?

The analysis is organised around Fisher information rather than KL
directly, because the nuisance parameter (unknown channel gain) is
what makes the problem composite, and Fisher information is what makes
"the nuisance costs the adversary X" a precise statement.

Parametrisation.  Every RE carries amplitude

    A_k(eta, g) = g (1 + eta)      k in P  (pilots)
                = g beta(eta)      k in D  (data)

with g > 0 an unknown common gain (|h|) and eta the parameter of
interest.  At (eta, g) = (0, 1) all amplitudes are unity, so every RE
has the SAME per-RE Fisher information about its own amplitude within
a given observation class -- which is what drives Proposition 1.

Derivatives at eta = 0, g = 1:
    dA_p/deta = 1        dA_p/dg = 1
    dA_d/deta = -a       dA_d/dg = 1        (a = N_p/N_d)
the -a following from beta^2 = 1 - a(2 eta + eta^2), so
dbeta/deta|_0 = -a.

Observation classes are distinguished ONLY by the per-RE information
about amplitude they admit:
    J_p  pilots, symbol known (DM-RS is standardised)
    J_d  data,   symbol unknown (QPSK mixture)
    J_pw power-only, symbol irrelevant
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------
# Per-RE Fisher information about amplitude A, at A = 1
# ---------------------------------------------------------------
def j_coherent(sigma2: float) -> float:
    """Known symbol, known channel phase: Z ~ CN(A S, sigma^2).

    For a complex Gaussian with parameter-dependent mean,
    I(theta) = 2 |dm/dtheta|^2 / sigma^2, so I(A) = 2 / sigma^2.
    Exact.
    """
    return 2.0 / sigma2


def j_power_deflection(sigma2: float, amp: float = 1.0) -> float:
    """Power-only observation, second-moment (deflection) value.

    |Z|^2 has mean A^2 + sigma^2 and variance sigma^4 + 2 A^2 sigma^2,
    so the deflection information about A is
        (d mean/dA)^2 / var = 4 A^2 / (sigma^4 + 2 A^2 sigma^2).

    This is a strict LOWER bound on the exact noncentral chi-square
    Fisher information (it uses only two moments).  Using it makes
    every impossibility statement below conservative.
    """
    return 4.0 * amp * amp / (sigma2 * sigma2 + 2.0 * amp * amp * sigma2)


# ---------------------------------------------------------------
# Fisher information matrix for (eta, g) and the efficient information
# ---------------------------------------------------------------
def fisher_block(n_p: float, n_d: float, j_p: float, j_d: float) -> dict:
    """Assemble I_eta,eta / I_eta,g / I_g,g and the efficient
    information about eta after profiling out the unknown gain g.

        I_ee = N_p J_p (dA_p/deta)^2 + N_d J_d (dA_d/deta)^2
             = N_p J_p + a^2 N_d J_d
        I_eg = N_p J_p (1)(1) + N_d J_d (-a)(1) = N_p (J_p - J_d)
        I_gg = N_p J_p + N_d J_d
        I_eff = I_ee - I_eg^2 / I_gg

    The middle line is the crux: a N_d = N_p exactly, by the
    power-neutrality constraint.  So I_eg vanishes iff J_p = J_d.
    """
    a = n_p / n_d
    i_ee = n_p * j_p + a * a * n_d * j_d
    i_eg = n_p * (j_p - j_d)
    i_gg = n_p * j_p + n_d * j_d
    i_eff = i_ee - i_eg * i_eg / i_gg
    return dict(a=a, I_ee=i_ee, I_eg=i_eg, I_gg=i_gg, I_eff=i_eff,
                nuisance_loss=1.0 - i_eff / i_ee)


def info_power_only(n_p: float, n_d: float, gamma: float) -> dict:
    """Adversary restricted to per-RE powers.  J_p = J_d = J_pw, so the
    information matrix is EXACTLY block-diagonal (Proposition 1) and
    ignorance of g costs nothing at all."""
    j = j_power_deflection(1.0 / gamma)
    return fisher_block(n_p, n_d, j, j)


def info_genie(n_p: float, n_d: float, gamma: float, j_d=None) -> float:
    """Adversary that knows the channel exactly.  No nuisance, so the
    information about eta is simply I_ee.  Upper bound on every
    adversary by the data-processing inequality."""
    s2 = 1.0 / gamma
    j_d = j_power_deflection(s2) if j_d is None else j_d
    return fisher_block(n_p, n_d, j_coherent(s2), j_d)['I_ee']


def info_blind(n_p: float, n_d: float, gamma: float, j_d=None) -> dict:
    """Adversary that knows the pilot SEQUENCE (standardised) but not
    the channel: coherent on pilots, power-domain on data, gain
    profiled out."""
    s2 = 1.0 / gamma
    j_d = j_power_deflection(s2) if j_d is None else j_d
    return fisher_block(n_p, n_d, j_coherent(s2), j_d)


def info_ratio_detector(n_p: float, n_d: float, gamma: float) -> float:
    """Realised information of the explicit channel-blind statistic

        T = (|u|^2 - sigma^2/N_p) / (Qbar - sigma^2),
        u = (1/N_p) sum_P Z_k S_k^*,   Qbar = (1/N_d) sum_D |Z_k|^2,

    which estimates (1+eta)^2 / beta^2.  Both numerator and denominator
    are debiased, so at eta = 0 both have unit mean and

        dlog E[T]/deta = 2 + 2a,
        relvar(num) = s^4 + 2 s^2      (s^2 = sigma^2/N_p),
        relvar(den) = (sigma^4 + 2 sigma^2)/N_d.

    This is ACHIEVABILITY: an adversary can actually run it.
    """
    s2 = 1.0 / gamma
    a = n_p / n_d
    sn2 = s2 / n_p
    slope = 2.0 * (1.0 + a)
    rv_num = sn2 * sn2 + 2.0 * sn2
    rv_den = (s2 * s2 + 2.0 * s2) / n_d
    return slope * slope / (rv_num + rv_den)


# ---------------------------------------------------------------
# From information to error probability: LAN / Gaussian shift
# ---------------------------------------------------------------
def optimal_accuracy(info: float, eta: float, n_sym: int, n_slots: int):
    """Asymptotic optimal accuracy of an equal-prior binary test.

    Slots are independent, so information adds: I_tot = I n_sym M.
    Under local alternatives eta = t / sqrt(M) the experiment is LAN
    and the limit is a Gaussian shift with separation
    sqrt(I_tot) * eta, whose Bayes accuracy is

        acc = Phi( sqrt(I_tot) |eta| / 2 ).

    Note the accuracy depends on (eta, M) only through eta sqrt(M) --
    this IS the square-root law, and it is checked numerically.
    """
    from scipy.stats import norm
    d = info * n_sym * n_slots * eta * eta      # deflection
    return float(norm.cdf(math.sqrt(d) / 2.0)), d


def kl_from_info(info: float, eta: float, n_sym: int, n_slots: int) -> float:
    """Total divergence over M slots, KL = I eta^2 n_sym M / 2.
    Exact additivity across independent slots (chain rule for product
    measures); the quadratic form is the leading term in eta."""
    return 0.5 * info * eta * eta * n_sym * n_slots


# ---------------------------------------------------------------
# Divergence -> total variation
# ---------------------------------------------------------------
def tv_pinsker(kl: float) -> float:
    """TV <= sqrt(KL/2).  VACUOUS for KL > 2, where a
    reported bound above unity is no bound at all."""
    return math.sqrt(kl / 2.0)


def tv_bretagnolle_huber(kl: float) -> float:
    """TV <= sqrt(1 - exp(-KL)).  Never vacuous, and tighter than
    Pinsker for KL > ~1.6.

    Bretagnolle and Huber (1979), Z. Wahrscheinlichkeitstheorie verw.
    Gebiete 47, 119-137, Eq. (2.2): exp(-KL) <= pi (2 - pi) <= 2 pi
    with pi = int min(dP, dQ) = 1 - TV.  Stopping the chain at
    pi(2-pi) gives this form; stopping at 2 pi gives the weaker
    pi >= exp(-KL)/2 that is often quoted instead.
    """
    return math.sqrt(1.0 - math.exp(-kl))


def tv_bound(kl: float) -> float:
    """Best of the two."""
    return min(tv_pinsker(kl), tv_bretagnolle_huber(kl))


# ---------------------------------------------------------------
# Covertness budget and geometry
# ---------------------------------------------------------------
def eta_budget(delta: float, info: float, n_sym: int, n_slots: int) -> float:
    """Largest eta with TV <= delta over M slots.

    Inverting Pinsker on KL = I eta^2 n_sym M / 2 gives
        eta <= 2 delta / sqrt(I n_sym M),
    the square-root law: sustainable perturbation decays as M^{-1/2}.
    """
    return 2.0 * delta / math.sqrt(info * n_sym * n_slots)


def gamma_at_standoff(d, d_ref: float, gamma_ref: float,
                      path_exp: float = 3.5) -> float:
    """Observer SNR at standoff d, referenced to gamma_ref at d_ref.
    gamma is decreasing in d, and every information functional above is
    increasing in gamma, so the worst case over an admissible region
    {d >= d_min} is attained at d = d_min -- by monotonicity, with no
    optimisation required."""
    return gamma_ref * (d_ref / d) ** path_exp


def eta_budget_at_standoff(delta, d_min, d_ref, gamma_ref, n_p, n_d,
                           n_sym, n_slots, path_exp=3.5) -> float:
    """Worst-case-geometry budget.  Uses the genie information, so the
    guarantee holds against every adversary at every admissible
    position."""
    g = gamma_at_standoff(d_min, d_ref, gamma_ref, path_exp)
    return eta_budget(delta, info_genie(n_p, n_d, g), n_sym, n_slots)
