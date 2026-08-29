"""
Monte-Carlo verification of the T2 minimax covertness analysis.

V1  per-RE Fisher information about amplitude, three observation
    classes (coherent / QPSK-mixture / power-only), MC scores vs theory
V2  nuisance structure: exact orthogonality in the power-only class,
    and the residual coupling in the full complex class
V3  achievability: the explicit channel-blind ratio detector, realised
    deflection vs predicted information, bracketed against the genie
V4  square-root law: accuracy depends on (eta, M) only through
    eta sqrt(M), and matches the LAN prediction
V5  TV bounds: measured (attainable) TV vs Pinsker vs Bretagnolle-Huber
V6  information vs observer SNR -- corrects the T1 note's claim that
    I_A1 saturates

Usage:  python verify_t2.py [--n_mc 200000] [--seed 0]
"""

from __future__ import annotations
import argparse, math

import numpy as np
from scipy.stats import norm
from scipy.special import ive

import t2_minimax as t2
import t1_law as law
from verify_t1 import Grid, gen_slot, observe


N_P, N_D, N_SYM = 273, 3003, 14


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def row(label, meas, pred, tol=0.02, se=None):
    if pred == 0:
        ok, rel = abs(meas) <= max(tol, 3 * (se or 0)), float('nan')
    else:
        rel = abs(meas - pred) / abs(pred)
        ok = rel < tol or (se is not None and abs(meas - pred) <= 3 * se)
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:<40} meas={meas: .6g}"
          f"  pred={pred: .6g}" + (f"  rel={rel:.2%}" if pred != 0 else ""))
    return ok


# ---------------------------------------------------------------
# V1  per-RE Fisher information about amplitude, by MC scores
# ---------------------------------------------------------------
def mc_j_coherent(sigma2, n, rng):
    """Known symbol, known phase: score = (2/s2) Re(W)."""
    w = math.sqrt(sigma2 / 2) * (rng.standard_normal(n)
                                 + 1j * rng.standard_normal(n))
    s = (2.0 / sigma2) * w.real
    return float(np.mean(s ** 2))


def mc_j_qpsk(sigma2, n, rng, amp=1.0):
    """Unknown QPSK symbol, full complex observation.

    f(z) = (1/4) sum_s CN(z; A s, s2);  score_A = sum_s w_s (2/s2)
    Re((z - A s) conj(s)) with posterior weights w_s.
    """
    const = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    s_true = const[rng.integers(0, 4, size=n)]
    w = math.sqrt(sigma2 / 2) * (rng.standard_normal(n)
                                 + 1j * rng.standard_normal(n))
    z = amp * s_true + w
    d = z[:, None] - amp * const[None, :]            # (n, 4)
    ll = -np.abs(d) ** 2 / sigma2
    ll -= ll.max(axis=1, keepdims=True)
    wt = np.exp(ll)
    wt /= wt.sum(axis=1, keepdims=True)
    per = (2.0 / sigma2) * np.real(d * np.conj(const)[None, :])
    return float(np.mean((wt * per).sum(axis=1) ** 2))


def mc_j_power(sigma2, n, rng, amp=1.0):
    """Power-only observation of a constant-modulus RE.

    f_P(p) = (1/s2) exp(-(p+A^2)/s2) I_0(2 A sqrt(p)/s2)
    score_A = -2A/s2 + (2 sqrt(p)/s2) I_1(x)/I_0(x),  x = 2A sqrt(p)/s2
    """
    w = math.sqrt(sigma2 / 2) * (rng.standard_normal(n)
                                 + 1j * rng.standard_normal(n))
    p = np.abs(amp + w) ** 2
    x = 2.0 * amp * np.sqrt(p) / sigma2
    ratio = ive(1, x) / ive(0, x)          # exp-scaled: ratio is stable
    s = -2.0 * amp / sigma2 + (2.0 * np.sqrt(p) / sigma2) * ratio
    return float(np.mean(s ** 2))


# ---------------------------------------------------------------
# V3/V4  channel-blind ratio detector on OFDM symbols
# ---------------------------------------------------------------
def symbol_stats(n_batch, eta, gamma, rng):
    """Debiased (L, D) per OFDM symbol.

    L estimates g^2 (1+eta)^2 from the coherent pilot average, D
    estimates g^2 beta^2 from the data-RE power average; both are
    unbiased, so E[L] = E[D] = 1 at eta = 0, g = 1.

    Sampled from the exact laws rather than by building the grid.
    Pilots: u = (1+eta) + CN(0, sigma^2/N_p) after derotation by the
    known sequence.  Data: every RE is constant-modulus, so
    sum |b s_i + w_i|^2 has a law independent of the QPSK symbols and
    equals (sigma^2/2)[chi2(N_d - 1) + (Z + sqrt(lam))^2 + chi2(N_d)]
    with lam = 2 N_d b^2 / sigma^2, using
    ncx2(k, lam) = chi2(k-1) + (Z + sqrt(lam))^2.
    This is exact, not a CLT approximation, and is O(n) in memory.
    V0 cross-checks it against explicit grid generation.
    """
    s2 = 1.0 / gamma
    b2 = max(law.beta_sq(eta, N_P / N_D), 0.0)

    sp = math.sqrt(s2 / N_P / 2.0)
    u = (1.0 + eta) + sp * (rng.standard_normal(n_batch)
                            + 1j * rng.standard_normal(n_batch))
    L = np.abs(u) ** 2 - s2 / N_P

    lam = 2.0 * N_D * b2 / s2
    z = rng.standard_normal(n_batch)
    tot = (s2 / 2.0) * (rng.chisquare(N_D - 1, n_batch)
                        + (z + math.sqrt(lam)) ** 2
                        + rng.chisquare(N_D, n_batch))
    D = tot / N_D - s2
    return L, D


def symbol_stats_explicit(n_batch, eta, gamma, rng):
    """Same statistic, built from an explicit resource grid.  Slow and
    memory-hungry; used only to cross-check symbol_stats."""
    s2 = 1.0 / gamma
    b = math.sqrt(max(law.beta_sq(eta, N_P / N_D), 0.0))
    sc = math.sqrt(s2 / 2)
    wp = sc * (rng.standard_normal((n_batch, N_P))
               + 1j * rng.standard_normal((n_batch, N_P)))
    u = (1.0 + eta) + wp.mean(axis=1)
    L = np.abs(u) ** 2 - s2 / N_P
    sd = np.exp(1j * (np.pi / 4
                      + rng.integers(0, 4, size=(n_batch, N_D)) * np.pi / 2))
    wd = sc * (rng.standard_normal((n_batch, N_D))
               + 1j * rng.standard_normal((n_batch, N_D)))
    D = (np.abs(b * sd + wd) ** 2).mean(axis=1) - s2
    return L, D


def detector_stat(n_slots, n_trials, eta, gamma, rng, chunk=4_000_000):
    """Accumulated log-ratio statistic over n_slots slots.

    Generated in chunks so that every mission length is backed by the
    same number of trials.  Capping n_trials for large n_slots instead
    would give the long-mission points a larger error bar than the
    short-mission ones, which would read as a breakdown of the
    collapse rather than as thinner sampling.
    """
    per = n_slots * N_SYM
    out = np.empty(n_trials)
    step = max(1, chunk // per)
    i = 0
    while i < n_trials:
        b = min(step, n_trials - i)
        L, D = symbol_stats(b * per, eta, gamma, rng)
        t = np.log(np.maximum(L, 1e-9)) - np.log(np.maximum(D, 1e-9))
        out[i:i + b] = t.reshape(b, per).sum(axis=1)
        i += b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_mc", type=int, default=400000)
    ap.add_argument("--n_trials", type=int, default=20000)
    ap.add_argument("--gamma_db", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    gamma = 10 ** (args.gamma_db / 10)
    s2 = 1.0 / gamma
    ok = []

    print(f"N_p={N_P}  N_d={N_D}  n_sym={N_SYM}  "
          f"observer SNR={args.gamma_db} dB (gamma={gamma:g})")

    # ---------------- V1 ----------------
    banner("V1  per-RE Fisher information about amplitude A (at A=1)")
    se = lambda v: v * math.sqrt(2.0 / args.n_mc)     # rough
    jp = mc_j_coherent(s2, args.n_mc, rng)
    ok.append(row("J_p  coherent, symbol known", jp,
                  t2.j_coherent(s2), tol=0.03))
    jd = mc_j_qpsk(s2, args.n_mc, rng)
    jw = mc_j_power(s2, args.n_mc, rng)
    jw_defl = t2.j_power_deflection(s2)
    print(f"  [--- ] J_d  QPSK mixture (MC only)          meas={jd: .6g}")
    print(f"  [--- ] J_pw power-only, exact (MC)          meas={jw: .6g}")
    ok.append(row("J_pw >= deflection lower bound",
                  1.0 if jw >= jw_defl * 0.99 else 0.0, 1.0, tol=1e-9))
    print(f"         deflection bound J_pw >= {jw_defl:.6g} "
          f"(used for conservative claims)")
    print(f"         ordering J_pw <= J_d <= J_p : "
          f"{jw:.4f} <= {jd:.4f} <= {jp:.4f}  "
          f"{'ok' if jw <= jd <= jp * 1.001 else 'VIOLATED'}")

    # ---------------- V2 ----------------
    banner("V2  nuisance structure (unknown common gain g)")
    pw = t2.fisher_block(N_P, N_D, jw, jw)
    print(f"  power-only class  (J_p = J_d = {jw:.4f}):")
    ok.append(row("   I_eg  (exact orthogonality)", pw['I_eg'], 0.0,
                  tol=1e-9))
    print(f"         I_ee={pw['I_ee']:.4g}  I_eff={pw['I_eff']:.4g}  "
          f"nuisance loss={pw['nuisance_loss']:.3%}")
    fx = t2.fisher_block(N_P, N_D, jp, jd)
    print(f"  full complex class (J_p={jp:.4f}, J_d={jd:.4f}):")
    print(f"         I_ee={fx['I_ee']:.4g}  I_eg={fx['I_eg']:.4g}  "
          f"I_gg={fx['I_gg']:.4g}")
    print(f"         I_eff={fx['I_eff']:.4g}  "
          f"nuisance loss={fx['nuisance_loss']:.3%}")
    print(f"  => unknown channel gain costs the adversary only "
          f"{fx['nuisance_loss']:.2%} of its information.")

    # ---------------- V0 ----------------
    banner("V0  exact-law sampler vs explicit resource-grid generation")
    for e in (0.0, 0.05):
        La, Da = symbol_stats(60000, e, gamma, rng)
        Lb, Db = symbol_stats_explicit(60000, e, gamma, rng)
        sL = math.sqrt(La.var() / len(La) + Lb.var() / len(Lb))
        sD = math.sqrt(Da.var() / len(Da) + Db.var() / len(Db))
        ok.append(row(f"eta={e:<5} E[L] exact vs grid", La.mean(),
                      Lb.mean(), tol=0.005, se=sL))
        ok.append(row(f"eta={e:<5} E[D] exact vs grid", Da.mean(),
                      Db.mean(), tol=0.005, se=sD))
        ok.append(row(f"eta={e:<5} Var[D] exact vs grid", Da.var(),
                      Db.var(), tol=0.06))

    # ---------------- V3 ----------------
    banner("V3  achievability: explicit channel-blind ratio detector")
    # J_d and J_pw are estimated once in V1 and reused everywhere below,
    # so the bracket is internally consistent rather than re-drawn.
    i_pred = t2.info_ratio_detector(N_P, N_D, gamma)
    i_genie = t2.info_genie(N_P, N_D, gamma, j_d=jd)
    i_blind = t2.info_blind(N_P, N_D, gamma, j_d=jd)['I_eff']
    i_pow_defl = t2.info_power_only(N_P, N_D, gamma)['I_eff']
    i_pow_exact = t2.fisher_block(N_P, N_D, jw, jw)['I_eff']

    # Independent replicates: the deflection estimate differences two
    # sample means, so its own error bar is wide and must be reported.
    eta_probe, n_b, n_rep = 0.01, 400000, 6
    reps = []
    for _ in range(n_rep):
        L0, D0 = symbol_stats(n_b, 0.0, gamma, rng)
        L1, D1 = symbol_stats(n_b, eta_probe, gamma, rng)
        a = np.log(L0) - np.log(D0)
        b = np.log(L1) - np.log(D1)
        reps.append((b.mean() - a.mean()) ** 2 / a.var() / eta_probe ** 2)
    reps = np.array(reps)
    i_meas, se_i = float(reps.mean()), float(reps.std(ddof=1) / math.sqrt(n_rep))
    ok.append(row("realised info of ratio detector", i_meas, i_pred,
                  tol=0.03, se=se_i))
    print(f"         {n_rep} replicates, +-{se_i:.4g} (1 sigma)")

    print(f"\n  Information bracket at {args.gamma_db:.0f} dB "
          f"(analytic; MC confirms the achievable end):")
    print(f"    power-only ratio, deflection J   I = {i_pow_defl:8.4g}"
          f"   (conservative)")
    print(f"    power-only ratio, exact J        I = {i_pow_exact:8.4g}")
    print(f"    blind ratio detector, ACHIEVABLE I = {i_pred:8.4g}"
          f"   (MC {i_meas:.4g} +- {se_i:.3g})")
    print(f"    best blind, efficient info       I = {i_blind:8.4g}")
    print(f"    genie knows channel, CONVERSE    I = {i_genie:8.4g}")
    dpi = i_pred <= i_blind <= i_genie * (1 + 1e-9)
    ok.append(dpi)
    print(f"    [{'ok ' if dpi else 'FAIL'}] DPI ordering "
          f"achievable <= efficient <= converse")
    print(f"    achievable / converse = {i_pred / i_genie:.4f}"
          f"   -> unknown channel gain costs the adversary "
          f"{1 - i_pred / i_genie:.2%}")

    # ---------------- V4 ----------------
    banner("V4  square-root law: accuracy depends on eta*sqrt(M) only")
    print(f"  {'eta':>8} {'M':>6} {'eta*sqrt(M)':>12} {'acc meas':>10}"
          f" {'acc LAN':>9} {'KL tot':>9}")
    for eta, M in ((0.004, 1), (0.002, 4), (0.001, 16), (0.0005, 64)):
        a0 = detector_stat(M, args.n_trials, 0.0, gamma, rng)
        a1 = detector_stat(M, args.n_trials, eta, gamma, rng)
        thr = 0.5 * (a0.mean() + a1.mean())
        acc = 0.5 * ((a0 < thr).mean() + (a1 >= thr).mean())
        acc_lan, _ = t2.optimal_accuracy(i_pred, eta, N_SYM, M)
        kl = t2.kl_from_info(i_genie, eta, N_SYM, M)
        se_a = math.sqrt(acc * (1 - acc) / (2 * args.n_trials))
        ok.append(abs(acc - acc_lan) < max(3 * se_a, 0.01))
        print(f"  {eta:>8.4f} {M:>6d} {eta*math.sqrt(M):>12.5f}"
              f" {acc:>10.4f} {acc_lan:>9.4f} {kl:>9.4f}"
              f"  {'ok' if abs(acc-acc_lan) < max(3*se_a, 0.01) else 'FAIL'}")

    # ---------------- V5 ----------------
    banner("V5  divergence -> total variation: Pinsker vs "
           "Bretagnolle-Huber")
    print(f"  {'eta':>8} {'M':>5} {'KL':>10} {'TV attain':>10}"
          f" {'Pinsker':>10} {'B-H':>8}")
    for eta, M in ((0.001, 1), (0.005, 1), (0.02, 1), (0.045, 1),
                   (0.045, 10)):
        kl = t2.kl_from_info(i_genie, eta, N_SYM, M)
        a0 = detector_stat(M, args.n_trials, 0.0, gamma, rng)
        a1 = detector_stat(M, args.n_trials, eta, gamma, rng)
        thr = 0.5 * (a0.mean() + a1.mean())
        acc = 0.5 * ((a0 < thr).mean() + (a1 >= thr).mean())
        tv_att = 2 * acc - 1
        pk, bh = t2.tv_pinsker(kl), t2.tv_bretagnolle_huber(kl)
        good = tv_att <= min(pk, bh) + 3e-3
        ok.append(good)
        print(f"  {eta:>8.4f} {M:>5d} {kl:>10.4g} {tv_att:>10.4f}"
              f" {pk:>10.4f} {bh:>8.4f}  {'ok' if good else 'FAIL'}")
    print("  (Pinsker exceeds 1 -- i.e. is vacuous -- wherever KL > 2;")
    print("   Bretagnolle-Huber stays a genuine bound throughout.)")

    # ---------------- V6 ----------------
    banner("V6  information vs observer SNR (T1 note claimed "
           "saturation -- it does not)")
    print(f"  {'gamma dB':>9} {'I power-only':>13} {'I blind':>10}"
          f" {'I genie':>10} {'I/gamma':>9}")
    for gdb in (-20, -10, 0, 10, 20, 30):
        gg = 10 ** (gdb / 10)
        ipo = t2.info_power_only(N_P, N_D, gg)['I_eff']
        jdg = mc_j_qpsk(1.0 / gg, 60000, rng)
        ibl = t2.info_blind(N_P, N_D, gg, j_d=jdg)['I_eff']
        ige = t2.info_genie(N_P, N_D, gg, j_d=jdg)
        print(f"  {gdb:>9d} {ipo:>13.4g} {ibl:>10.4g} {ige:>10.4g}"
              f" {ige/gg:>9.4g}")
    a_ = N_P / N_D
    print("  I grows without bound, asymptotically linear in gamma.")
    print(f"  I_genie/gamma tends to 2 N_p (1+a) = "
          f"{2*N_P*(1+a_):.1f} -- the pilot term 2 N_p = {2*N_P} plus\n          the data term; NOT 2 N_p alone.")

    # ---------------- consequence ----------------
    banner("Consequence: worst-case covertness budget")
    print(f"  {'delta':>6} {'M slots':>8} {'eta_max':>11} {'G-1':>11}"
          f" {'dB':>9}")
    for delta in (0.05, 0.1):
        for M in (1, 100, 10000):
            e = t2.eta_budget(delta, i_genie, N_SYM, M)
            g_ = law.sensing_gain(e)
            print(f"  {delta:>6.2f} {M:>8d} {e:>11.3e} {g_-1:>11.3e}"
                  f" {10*math.log10(g_):>9.2e}")
    print("\n  standoff scaling (path exponent 3.5, gamma=0 dB at d_ref),"
          " delta=0.1, M=100:")
    print(f"  {'d/d_ref':>9} {'gamma dB':>9} {'eta_max':>11} {'dB gain':>10}")
    for dr in (1, 2, 5, 10, 20, 50):
        gg = t2.gamma_at_standoff(dr, 1.0, 1.0, 3.5)
        jdg = mc_j_qpsk(1.0 / gg, 60000, rng)
        e = t2.eta_budget(0.1, t2.info_genie(N_P, N_D, gg, j_d=jdg),
                          N_SYM, 100)
        print(f"  {dr:>9d} {10*math.log10(gg):>9.1f} {e:>11.3e}"
              f" {10*math.log10(law.sensing_gain(e)):>10.2e}")

    banner("SUMMARY")
    print(f"  {sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
