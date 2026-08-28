"""
Does the constant-modulus assumption matter?

The analysis assumes QPSK data, so every data RE is unit-modulus and
the per-RE power law is an exact noncentral chi-square with fixed
noncentrality.  Real 5G NR adapts the modulation with the MCS, up to
256QAM, and a higher-order constellation carries amplitude randomness
of its own.  That randomness reduces the per-RE information J_d about
the amplitude scale, which by (9) makes I_eta,g = N_p (J_p - J_d)
nonzero even inside the power-only class.  Theorem 1's orthogonality
is therefore exact only for constant-modulus data, and the question
is how much cover a higher-order constellation actually buys.
"""
from __future__ import annotations
import math
import numpy as np

import t2_minimax as t2

N_P, N_D = 273, 3003
A_RATIO = N_P / N_D


def qam(order: int) -> np.ndarray:
    """Square M-QAM normalized to unit average power."""
    m = int(round(math.sqrt(order)))
    lv = np.arange(-(m - 1), m, 2)
    c = (lv[None, :] + 1j * lv[:, None]).ravel()
    return c / math.sqrt((np.abs(c) ** 2).mean())


def j_amplitude(const: np.ndarray, sigma2: float, n: int, rng,
                power_only: bool = False, chunk: int = 20000) -> float:
    """Fisher information about the amplitude scale A at A=1 for a RE
    carrying a symbol drawn uniformly from `const`.

    full complex:  f(z) = (1/M) sum_i CN(z; A c_i, s2)
    power only:    f(p) built from the same mixture, observed through
                   |z|^2 only; the score is obtained by conditioning on
                   the modulus, which for a mixture requires the
                   posterior over |c_i| as well as over the phase.
    """
    tot, seen = 0.0, 0
    while seen < n:
        b = min(chunk, n - seen)
        x = const[rng.integers(0, len(const), size=b)]
        w = math.sqrt(sigma2 / 2) * (rng.standard_normal(b)
                                     + 1j * rng.standard_normal(b))
        z = x + w
        if power_only:
            # observation is |z|^2; the sufficient statistic for the
            # mixture is the vector of component likelihoods of |z|^2,
            # each a noncentral chi-square with noncentrality |c_i|^2.
            p = np.abs(z) ** 2
            amp = np.unique(np.abs(const).round(12))
            wts = np.array([(np.abs(np.abs(const) - u) < 1e-9).mean()
                            for u in amp])
            from scipy.special import ive
            xa = 2.0 * np.outer(np.sqrt(p), amp) / sigma2
            # component density of |z|^2 up to a common factor
            ll = -(p[:, None] + amp[None, :] ** 2) / sigma2 \
                + np.log(ive(0, xa) + 1e-300) + xa
            ll += np.log(wts)[None, :]
            ll -= ll.max(axis=1, keepdims=True)
            po = np.exp(ll); po /= po.sum(axis=1, keepdims=True)
            r = ive(1, xa) / (ive(0, xa) + 1e-300)
            per = -2.0 * amp[None, :] ** 2 / sigma2 \
                + (2.0 * np.sqrt(p)[:, None] * amp[None, :] / sigma2) * r
            sc = (po * per).sum(axis=1)
        else:
            d = z[:, None] - const[None, :]
            ll = -np.abs(d) ** 2 / sigma2
            ll -= ll.max(axis=1, keepdims=True)
            po = np.exp(ll); po /= po.sum(axis=1, keepdims=True)
            per = (2.0 / sigma2) * np.real(d * np.conj(const)[None, :])
            sc = (po * per).sum(axis=1)
        tot += float(np.sum(sc ** 2)); seen += b
    return tot / n


def main():
    rng = np.random.default_rng(0)
    for gdb in (0.0, 10.0):
        gamma = 10 ** (gdb / 10)
        s2 = 1.0 / gamma
        jp = t2.j_coherent(s2)                    # pilots stay unit-modulus
        print(f"\n=== observer SNR {gdb:.0f} dB   (J_p = {jp:.4f}) ===")
        print(f"{'data mod.':>10} {'J_d':>8} {'J_p-J_d':>9} "
              f"{'I_ee':>9} {'I_eff':>9} {'nuis.loss':>10} {'vs QPSK':>9}")
        base = None
        for order, name in ((4, "QPSK"), (16, "16QAM"),
                            (64, "64QAM"), (256, "256QAM")):
            jd = j_amplitude(qam(order), s2, 200000, rng)
            fb = t2.fisher_block(N_P, N_D, jp, jd)
            if base is None:
                base = fb['I_eff']
            print(f"{name:>10} {jd:8.4f} {jp-jd:9.4f} {fb['I_ee']:9.4f} "
                  f"{fb['I_eff']:9.4f} {fb['nuisance_loss']:9.3%} "
                  f"{fb['I_eff']/base:8.4f}")
        # power-only class: is orthogonality still exact?
        print("  power-only class (Theorem 1 requires J_p = J_d there):")
        for order, name in ((4, "QPSK"), (64, "64QAM"), (256, "256QAM")):
            jw = j_amplitude(qam(order), s2, 120000, rng, power_only=True)
            jwp = j_amplitude(qam(4), s2, 120000, rng, power_only=True)
            fb = t2.fisher_block(N_P, N_D, jwp, jw)
            print(f"{name:>10}  J_pw(pilot)={jwp:.4f}  J_pw(data)={jw:.4f}"
                  f"  I_eg={fb['I_eg']:9.3f}  loss={fb['nuisance_loss']:.3%}")


if __name__ == "__main__":
    main()
