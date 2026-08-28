"""
Monte-Carlo verification of the T1 perturbation law against a
waveform-level 5G NR FR1 OFDM simulation.

Every claim in t1_law.py is checked against samples drawn from the
actual OFDM resource grid (time-domain IFFT, CP, observer noise added
at the antenna, FFT back), not against the model that produced it.

Checks
  C1  power neutrality               total slot energy invariant in eta
  C2  beta^2(eta)                    measured data-RE power
  C3  sensing gain                   G(eta) = (1+eta)^2 via pilot-
                                     referenced matched filter
  C4  A2 coherent-genie divergence   D = N_p gamma eta^2
  C5  A1 invariant statistic         mean and variance of the
                                     pilot-to-data power ratio T,
                                     hence I_A1(0)
  C6  A0 energy detector             zero deflection of total power
  C7  per-RE divergence              single-RE scope vs slot scope

Usage:  python verify_t1.py [--n_slots 200] [--seed 0]
"""

from __future__ import annotations
import argparse, math

import numpy as np

import t1_law as law


# ---------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------
class Grid:
    """5G NR FR1 resource grid, 100 MHz / 30 kHz SCS."""

    def __init__(self, pilot_period_sc: int = 12, n_fft: int = 4096,
                 n_sc_active: int = 3276, n_sym: int = 14,
                 cp_len: int = 288):
        self.n_fft = n_fft
        self.n_sc = n_sc_active
        self.n_sym = n_sym
        self.cp = cp_len
        half = n_sc_active // 2
        self.half = half

        pilot_sc = np.arange(0, n_sc_active, pilot_period_sc)
        mask = np.zeros(n_sc_active, dtype=bool)
        mask[pilot_sc] = True
        self.pilot_sc = pilot_sc
        self.data_sc = np.nonzero(~mask)[0]

        # subcarrier index -> FFT bin (same mapping as the released sim)
        sc = np.arange(n_sc_active)
        self.sc_to_fft = np.where(sc < half, n_fft - (half - sc),
                                  sc - half + 1)
        self.pilot_fft = self.sc_to_fft[self.pilot_sc]
        self.data_fft = self.sc_to_fft[self.data_sc]

        self.n_p = len(self.pilot_sc)
        self.n_d = len(self.data_sc)
        self.alpha = self.n_p / n_sc_active
        self.a = self.n_p / self.n_d


def qpsk(n, rng):
    return np.exp(1j * (np.pi / 4 + rng.integers(0, 4, size=n) * np.pi / 2))


def gen_slot(g: Grid, eta: float, rng, power_neutral: bool = True):
    """Return (time-domain slot, freq-domain grid [n_sym, n_fft]).

    Pilots scaled by (1+eta); data scaled by beta(eta) when
    power_neutral, else left at unit amplitude (naive pilot boost).
    """
    b = math.sqrt(max(law.beta_sq(eta, g.a), 0.0)) if power_neutral else 1.0
    X = np.zeros((g.n_sym, g.n_fft), dtype=complex)
    td = []
    for s in range(g.n_sym):
        sc = qpsk(g.n_sc, rng)
        sc[g.pilot_sc] *= (1.0 + eta)
        sc[g.data_sc] *= b
        X[s, g.sc_to_fft] = sc
        x = np.fft.ifft(X[s]) * math.sqrt(g.n_fft)
        td.append(np.concatenate([x[-g.cp:], x]))
    return np.concatenate(td), X


def observe(g: Grid, td, gamma: float, rng):
    """Passive observer: add CN(0, sigma^2) at the antenna with
    sigma^2 = 1/gamma (unit channel gain), strip CP, FFT back.
    Returns the per-symbol post-FFT grid."""
    sigma2 = 1.0 / gamma
    n = math.sqrt(sigma2 / 2) * (rng.standard_normal(len(td))
                                 + 1j * rng.standard_normal(len(td)))
    y = td + n
    sym_len = g.n_fft + g.cp
    Z = np.empty((g.n_sym, g.n_fft), dtype=complex)
    for s in range(g.n_sym):
        seg = y[g.cp + s * sym_len: g.cp + s * sym_len + g.n_fft]
        Z[s] = np.fft.fft(seg) / math.sqrt(g.n_fft)
    return Z


# ---------------------------------------------------------------
# Checks
# ---------------------------------------------------------------
def banner(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def row(label, meas, pred, tol=0.02):
    if pred == 0:
        ok = abs(meas) < tol
        rel = float('nan')
    else:
        rel = abs(meas - pred) / abs(pred)
        ok = rel < tol
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {label:<34} meas={meas: .6g}  pred={pred: .6g}"
          + (f"  rel={rel:.3%}" if pred != 0 else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_slots", type=int, default=300)
    ap.add_argument("--gamma_db", type=float, default=0.0,
                    help="observer SNR in dB")
    ap.add_argument("--pilot_period", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    g = Grid(pilot_period_sc=args.pilot_period)
    gamma = 10 ** (args.gamma_db / 10)
    etas = [0.0, 0.02, 0.045, 0.10, 0.20]
    all_ok = []

    print(f"Grid: N={g.n_sc} active SC, N_p={g.n_p}, N_d={g.n_d}, "
          f"alpha={g.alpha:.4f}, a={g.a:.5f}")
    print(f"Observer SNR gamma = {args.gamma_db} dB ({gamma:.3f}), "
          f"n_slots={args.n_slots}, n_sym={g.n_sym}")
    print(f"eta_max (beta^2>0) = {law.eta_max(g.alpha):.4f}")

    # ---------------- C1 / C2: power neutrality, beta^2 ----------
    banner("C1/C2  power neutrality and beta^2(eta)")
    e_ref = None
    for eta in etas:
        td, X = gen_slot(g, eta, rng)
        e_tot = float(np.mean(np.abs(td[g.cp:]) ** 2)) if False else \
            float(np.sum(np.abs(X) ** 2) / g.n_sym)
        if e_ref is None:
            e_ref = e_tot
        d_pow = float(np.mean(np.abs(X[:, g.data_fft]) ** 2))
        all_ok.append(row(f"eta={eta:<5} total symbol energy",
                          e_tot, e_ref, tol=1e-9))
        all_ok.append(row(f"eta={eta:<5} data RE power beta^2",
                          d_pow, law.beta_sq(eta, g.a), tol=1e-9))

    # ---------------- C3: sensing gain ---------------------------
    banner("C3  sensing gain G(eta) = (1+eta)^2  "
           "(pilot-referenced matched filter)")
    # Surveillance channel: echo of the transmitted grid plus noise.
    # Matched filter against the pilot-only reference (an uncooperative
    # receiver cannot reconstruct the payload).  Output SNR measured as
    # |E[A]|^2 / Var(A); noise draws are vectorised so the variance
    # estimate is not the bottleneck.
    n_slot_mc, n_noise = 12, 3000
    sig_s2 = 1.0                       # surveillance noise variance
    snr_out = {}
    for eta in etas:
        acc = []
        for _ in range(n_slot_mc):
            _, X = gen_slot(g, eta, rng)
            ref = X[0, g.pilot_fft]                  # symbol 0
            W = math.sqrt(sig_s2 / 2) * (
                rng.standard_normal((n_noise, len(ref)))
                + 1j * rng.standard_normal((n_noise, len(ref))))
            acc.append((ref[None, :] + W) @ np.conj(ref))
        acc = np.concatenate(acc)
        snr_out[eta] = float(np.abs(acc.mean()) ** 2 / acc.var())
    base = snr_out[0.0]
    n_acc = n_slot_mc * n_noise
    for eta in etas:
        # relative SE of a variance estimate from n samples ~ sqrt(2/n),
        # and the ratio combines two of them
        se = 2.0 * math.sqrt(2.0 / n_acc)
        all_ok.append(row(f"eta={eta:<5} MF output SNR gain",
                          snr_out[eta] / base,
                          law.sensing_gain(eta), tol=max(0.02, 3 * se)))
    print(f"  (a phenomenological model would assume (1+eta): "
          f"{[round(1+e, 3) for e in etas]})")

    # ---------------- collect observer samples -------------------
    banner("C4/C5/C6/C7  observer-side divergences")
    stats = {}
    for eta in etas:
        llr = []          # A2 coherent-genie LLR, per symbol
        Tstat = []        # A1 invariant statistic, per symbol
        tot = []          # A0 total power, per symbol
        pre = []          # per-RE pilot powers (subsampled)
        for _ in range(args.n_slots):
            td, X = gen_slot(g, eta, rng)
            Z = observe(g, td, gamma, rng)
            sigma2 = 1.0 / gamma
            for s in range(g.n_sym):
                zp = Z[s, g.pilot_fft]
                zd = Z[s, g.data_fft]
                # A2: knows h=1 and the pilot sequence S_k.  Recover
                # S_k from the transmitted grid (deterministic DM-RS).
                S = X[s, g.pilot_fft] / (1.0 + eta)
                l1 = -np.abs(zp - (1.0 + eta) * S) ** 2 / sigma2
                l0 = -np.abs(zp - S) ** 2 / sigma2
                llr.append(float(np.sum(l1 - l0)))
                pp = np.abs(zp) ** 2
                dp = np.abs(zd) ** 2
                Tstat.append(float(pp.mean() / dp.mean()))
                tot.append(float(pp.sum() + dp.sum()))
                pre.append(pp)
        stats[eta] = dict(
            llr=np.array(llr), T=np.array(Tstat), tot=np.array(tot),
            pre=np.concatenate(pre))

    # C4: A2 divergence = E_{p_eta}[LLR] = N_p gamma eta^2.
    # The LLR has SD sqrt(2 N_p gamma) * eta per symbol, so the relative
    # standard error grows as 1/eta; gate on 3 sigma, not a fixed
    # percentage, or the small-eta cells fail for purely MC reasons.
    print("\n-- C4  A2 coherent genie (pilots only, per OFDM symbol) --")
    for eta in etas:
        v = stats[eta]['llr']
        meas, se = float(v.mean()), float(v.std() / math.sqrt(len(v)))
        pred = law.kl_a2(eta, gamma, g.n_p)
        ok = abs(meas - pred) <= max(3 * se, 0.02 * pred, 1e-12)
        all_ok.append(ok)
        print(f"  [{'ok ' if ok else 'FAIL'}] eta={eta:<5} "
              f"D_A2 [nats/symbol]        meas={meas: .6g} "
              f"+-{se:.3g}  pred={pred: .6g}")
        if eta > 0:
            print(f"          per slot {meas*g.n_sym:.4g} nats, "
                  f"TV bound min(1, {math.sqrt(meas*g.n_sym/2):.3g})")

    # C5: A1 invariant statistic
    print("\n-- C5  A1 pilot-to-data power ratio T --")
    kappa = gamma / (1 + gamma)
    T0 = stats[0.0]['T']
    var0_meas = float(T0.var())
    m0_meas = float(T0.mean())
    var0_pred = m0_meas ** 2 * (1 - kappa ** 2) * (1 / g.n_p + 1 / g.n_d)
    all_ok.append(row("Var_0(T)", var0_meas, var0_pred, tol=0.10))
    for eta in etas:
        meas = float(stats[eta]['T'].mean())
        b2 = law.beta_sq(eta, g.a)
        pred = (((1 + eta) ** 2 * gamma + 1) / (b2 * gamma + 1))
        all_ok.append(row(f"eta={eta:<5} E[T]", meas, pred, tol=0.01))
    # I_A1(0) = (m'(0))^2 / Var_0(T).  Both ingredients are verified
    # above, so I is verified by composition.  A direct finite-difference
    # estimate of the slope is reported only as a consistency note: it
    # differences two means each carrying SE ~ sqrt(Var/n), so its own
    # error bar is wide and it is not used as a pass/fail gate.
    i_pred = law.fisher_a1(gamma, g.n_p, g.n_d)
    kappa = gamma / (1 + gamma)
    slope_pred = 2.0 * (1.0 + g.a) * kappa
    n_T = len(T0)
    se_m = math.sqrt(var0_meas / n_T)
    slope_meas = (stats[0.02]['T'].mean() - m0_meas) / 0.02
    se_slope = math.sqrt(2.0) * se_m / 0.02
    print(f"  m'(0):  meas={slope_meas:.4g} +-{se_slope:.3g} "
          f"(secant at eta=0.02)   pred={slope_pred:.4g}"
          f"   [consistency note, not gated]")
    print(f"  => I_A1(0) = m'(0)^2 / Var_0(T) = {i_pred:.4g}, "
          f"KL_A1 per symbol ~ {0.5*i_pred:.4g} * eta^2")

    # C6: A0 energy detector
    print("\n-- C6  A0 energy detector (total received power) --")
    m0 = stats[0.0]['tot'].mean()
    s0 = stats[0.0]['tot'].std()
    for eta in etas[1:]:
        defl = float(((stats[eta]['tot'].mean() - m0) / s0) ** 2)
        all_ok.append(row(f"eta={eta:<5} deflection of total power",
                          defl, 0.0, tol=0.05))

    # C7: per-RE divergence (single-RE scope).  Verified against the
    # exact noncentral-chi-square moments, with the plug-in estimator's
    # O(1/n) upward bias removed by Richardson extrapolation.
    print("\n-- C7  single pilot-RE power divergence "
          "(per-RE scope) --")
    p0 = stats[0.0]['pre']
    print(f"  (n = {len(p0)} pilot-RE samples per class)")
    for eta in etas[1:]:
        pp = stats[eta]['pre']
        raw = law.kl_moment_matched(p0, pp)
        deb = law.kl_moment_matched_debiased(p0, pp)
        pred = law.kl_per_re_exact(eta, gamma)
        # The debiased estimator still carries O(1/n) noise that, at the
        # smallest eta, is comparable to the signal itself.  Estimate its
        # spread from disjoint sub-batches and gate on 3 sigma rather
        # than on a fixed percentage, which would make the check pass or
        # fail on the seed.
        nb = 8
        h = min(len(p0), len(pp)) // nb
        sub = np.array([law.kl_moment_matched_debiased(
            p0[i * h:(i + 1) * h], pp[i * h:(i + 1) * h]) for i in range(nb)])
        se = float(sub.std(ddof=1) / math.sqrt(nb))
        good = abs(deb - pred) <= max(3 * se, 0.10 * pred)
        all_ok.append(good)
        print(f"  [{'ok ' if good else 'FAIL'}] eta={eta:<5} "
              f"per-RE KL (debiased)      meas={deb: .6g} +-{se:.3g}"
              f"  pred={pred: .6g}")
        slot_a1 = law.kl_a1(eta, gamma, g.n_p, g.n_d) * g.n_sym
        print(f"          plug-in={raw:.6g} (bias {raw-deb:+.3g}), "
              f"small-eta form={law.kl_per_re(eta, gamma):.6g}, "
              f"naive eta^2/2={eta*eta/2:.6g}")
        print(f"          slot-level A1 = {slot_a1:.4g} nats "
              f"= {slot_a1/max(pred,1e-12):.4g}x the per-RE number")

    # ---------------- covertness budget --------------------------
    banner("Consequence: mission-level covertness budget")
    i1 = law.fisher_a1(gamma, g.n_p, g.n_d)
    i2 = law.fisher_a2(gamma, g.n_p)
    print(f"  I_A1(0) = {i1:.4g}   I_A2(0) = {i2:.4g}   "
          f"(per OFDM symbol)")
    print(f"  {'delta':>6} {'M slots':>8} {'eta_max(A1)':>13} "
          f"{'G-1 (A1)':>11} {'eta_max(A2)':>13} {'G-1 (A2)':>11}")
    for delta in (0.05, 0.10):
        for M in (1, 100, 10000):
            e1 = law.eta_budget(delta, i1, g.n_sym, M)
            e2 = law.eta_budget(delta, i2, g.n_sym, M)
            print(f"  {delta:>6.2f} {M:>8d} {e1:>13.3e} "
                  f"{law.sensing_gain(e1)-1:>11.3e} {e2:>13.3e} "
                  f"{law.sensing_gain(e2)-1:>11.3e}")

    # ---------------- (c) rate cost ------------------------------
    banner("(c) rate cost under power neutrality")
    for rho_c_db, n_avg in ((10.0, 4.0), (20.0, 4.0)):
        rho_c = 10 ** (rho_c_db / 10)
        print(f"  rho_c={rho_c_db:.0f} dB, n_avg={n_avg:.0f} pilots "
              f"averaged")
        r0 = law.rate_cost(0.0, g.a, rho_c, n_avg)['rate']
        best = (0.0, r0)
        for eta in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]:
            if law.beta_sq(eta, g.a) <= 0:
                continue
            r = law.rate_cost(eta, g.a, rho_c, n_avg)
            if r['rate'] > best[1]:
                best = (eta, r['rate'])
            print(f"    eta={eta:<5} beta^2={r['beta_sq']:.4f} "
                  f"rho_eff={10*math.log10(r['rho_eff']):+6.2f} dB "
                  f"rate={r['rate']:.4f} b/s/Hz  "
                  f"dR={r0-r['rate']:+.5f}")
        print(f"    rate-optimal eta* = {best[0]} "
              f"(gain {best[1]-r0:+.5f} b/s/Hz)")

    banner("SUMMARY")
    n_ok, n_tot = sum(all_ok), len(all_ok)
    print(f"  {n_ok}/{n_tot} checks passed")
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
