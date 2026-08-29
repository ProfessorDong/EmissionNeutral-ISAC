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
        print(f"{'data mod.':>10} {'J_d':>18} {'loss':>8} "
              f"{'I_eff':>10} {'red. vs QPSK':>13} {'extra eta':>10}")
        base = None
        # average several independent replicates: a single run is noisy
        # enough that J_d for QPSK can come out above the exact J_p,
        # which would contradict the ordering J_pw <= J_d <= J_p.
        for order, name in ((4, "QPSK"), (16, "16QAM"),
                            (64, "64QAM"), (256, "256QAM")):
            v = [j_amplitude(qam(order), s2, 400000, rng) for _ in range(5)]
            raw = float(np.mean(v)); se = float(np.std(v, ddof=1) / math.sqrt(5))
            # J_d <= J_p holds exactly, by convexity of Fisher information
            # in the mixture: not knowing the symbol cannot help.  At high
            # SNR a QPSK symbol is almost always decodable, so the true gap
            # is exponentially small and the estimator straddles J_p no
            # matter how many samples are drawn.  Verify the excess is
            # within sampling error, then clip so that every downstream
            # quantity respects the bound.
            assert raw <= jp + 4 * se, f"J_d {raw} exceeds J_p {jp} by > 4 SE"
            jd = min(raw, jp)
            fb = t2.fisher_block(N_P, N_D, jp, jd)
            if base is None:
                base = fb['I_eff']
            flag = "*" if raw > jp else " "
            print(f"{name:>10} {jd:10.4f}{flag}+-{se:<6.4f} "
                  f"{fb['nuisance_loss']:>7.2%} {fb['I_eff']:>10.1f} "
                  f"{1-fb['I_eff']/base:>12.2%} "
                  f"{(base/fb['I_eff'])**0.5-1:>9.2%}")
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
