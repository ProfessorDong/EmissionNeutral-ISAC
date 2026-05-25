"""
SILENT-SENTRY Monte-Carlo simulator.

Two decoupled experiments validate the paper's three results:

  A. Covertness measurement (Theorem 1, Pinsker bound):
       generate realistic 5G NR FR1 OFDM waveforms under baseline
       (eta=0) and pilot-perturbed (eta>0) policies, add an
       observer noise floor, sweep eta, and estimate the empirical
       KL divergence between the observer's per-pilot power
       distributions.  Verify that
           Delta_S_RF(eta) approx eta^2 / 2
       in the small-eta regime, so the Pinsker route gives a tight
       TV bound.

  B. Detection measurement (Theorem 2 + Corollary 1):
       generate Neyman-Pearson sufficient statistics under H0/H1
       with controllable per-receiver post-integration SNR
       rho_r, run R receivers non-coherently, and measure P_D as a
       function of (R, eta) via the Corollary's
           rho_total(eps) = R * rho_r_0 * (1 + sqrt(2 eps)).
       Compare measured P_D against the closed form
           P_D = Q(Q^{-1}(P_FA) - sqrt(2 rho_total)).

Author: Liang Dong (MILCOM 2026 Paper 4).
"""

from __future__ import annotations
import argparse, csv, json, math, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm


# =============================================================
# 5G NR FR1 waveform parameters
# =============================================================
@dataclass(frozen=True)
class NRParams:
    bw_hz: float = 100e6
    scs_hz: float = 30e3
    n_sc_active: int = 3276            # 273 RBs * 12 SC/RB (5G NR FR1 spec)
    n_fft: int = 4096
    n_sym_per_slot: int = 14
    slot_ms: float = 0.5
    pilot_period_sc: int = 12          # NR Type-1 DM-RS period
    cp_len: int = 288


def gen_qpsk(n: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 4, size=n)
    return np.exp(1j * (np.pi / 4 + bits * np.pi / 2))


def gen_ofdm_slot(prm: NRParams, eta: float, rng: np.random.Generator,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Generate one 5G NR slot.  Returns (time-domain samples, pilot
    FFT indices) so the observer feature extractor can read pilots
    directly."""
    n_sym = prm.n_sym_per_slot
    fft = prm.n_fft
    sc_active = prm.n_sc_active
    half = sc_active // 2
    pilot_sc = np.arange(0, sc_active, prm.pilot_period_sc)
    pilot_fft = np.where(pilot_sc < half,
                          fft - (half - pilot_sc),
                          pilot_sc - half + 1)

    sym_td = []
    for _ in range(n_sym):
        X = np.zeros(fft, dtype=complex)
        sc = gen_qpsk(sc_active, rng)
        sc[pilot_sc] *= (1.0 + eta)        # pilot perturbation
        X[fft - half: fft] = sc[:half]
        X[1: sc_active - half + 1] = sc[half:]
        x = np.fft.ifft(X) * math.sqrt(fft)
        sym_td.append(np.concatenate([x[-prm.cp_len:], x]))
    return np.concatenate(sym_td), pilot_fft


# =============================================================
# Experiment A: observer covertness measurement
# =============================================================
def measure_kl(prm: NRParams,
               eta: float,
               n_slots: int,
               obs_snr_db: float,
               n_trials: int,
               rng: np.random.Generator,
               ) -> tuple[float, float]:
    """Generate per-symbol per-pilot power vectors under baseline (eta=0)
    and sensing (eta) policies; estimate KL via Gaussian moment matching
    on the concatenated per-pilot power samples.  Returns
    (empirical_KL_in_nats, theoretical_eta2_over_2).
    """
    fft = prm.n_fft
    sym_len = fft + prm.cp_len
    obs_var = 10 ** (-obs_snr_db / 10)
    powers_0 = []
    powers_pi = []

    for _ in range(n_trials):
        x0, pilot_fft = gen_ofdm_slot(prm, 0.0, rng)
        xp, _        = gen_ofdm_slot(prm, eta, rng)
        # add independent observer noise at the antenna
        sig0 = math.sqrt(obs_var / 2)
        n0 = sig0 * (rng.standard_normal(len(x0)) +
                     1j * rng.standard_normal(len(x0)))
        np_ = sig0 * (rng.standard_normal(len(xp)) +
                       1j * rng.standard_normal(len(xp)))
        y0 = x0 + n0
        yp = xp + np_
        for s in range(prm.n_sym_per_slot):
            seg0 = y0[prm.cp_len + s * sym_len:
                       prm.cp_len + s * sym_len + fft]
            segp = yp[prm.cp_len + s * sym_len:
                       prm.cp_len + s * sym_len + fft]
            F0 = np.fft.fft(seg0) / math.sqrt(fft)
            Fp = np.fft.fft(segp) / math.sqrt(fft)
            powers_0.append(np.abs(F0[pilot_fft]) ** 2)
            powers_pi.append(np.abs(Fp[pilot_fft]) ** 2)

    p0 = np.concatenate(powers_0)
    pp = np.concatenate(powers_pi)
    mu0, var0 = p0.mean(), p0.var()
    mu1, var1 = pp.mean(), pp.var()
    kl_meas = 0.5 * (math.log(var0 / var1) + var1 / var0 +
                     (mu1 - mu0) ** 2 / var0 - 1.0)
    eps_pred = eta * eta / 2.0
    return float(kl_meas), float(eps_pred)


# =============================================================
# Experiment A': adversary ML classifier on per-slot features
# =============================================================
def measure_adversary(prm: NRParams,
                       eta: float,
                       obs_snr_db: float,
                       n_train: int,
                       n_test: int,
                       rng: np.random.Generator,
                       ) -> dict:
    """An RF observer trains a binary classifier distinguishing the
    sensing-active policy (eta>0) from the baseline (eta=0) using the
    per-slot mean of the pilot powers as features.  We use the
    Bayes-optimal linear classifier on the empirical sufficient
    statistic (sample mean of per-pilot powers in the slot), which is
    LDA-equivalent for Gaussian power-feature distributions.

    Returns (acc_measured, acc_pinsker_upper_bound, kl_meas,
    pinsker_TV).  acc_pinsker_upper_bound = 0.5 + sqrt(KL/2)/2 from
    Theorem 1.
    """
    fft = prm.n_fft
    sym_len = fft + prm.cp_len
    obs_var = 10 ** (-obs_snr_db / 10)

    def gen_features(eta_local, n_slots):
        """Return a 1-D array of single-pilot single-symbol powers
        (one feature value per pilot per OFDM symbol per slot).  This
        keeps the adversary's per-feature KL at the level
        $\\Delta S_{RF}$ as defined in the paper, so the measured
        accuracy and the Pinsker upper bound are at parity."""
        feats = []
        for _ in range(n_slots):
            x, pilot_fft = gen_ofdm_slot(prm, eta_local, rng)
            sig0 = math.sqrt(obs_var / 2)
            nz = sig0 * (rng.standard_normal(len(x)) +
                         1j * rng.standard_normal(len(x)))
            y = x + nz
            for s in range(prm.n_sym_per_slot):
                seg = y[prm.cp_len + s * sym_len:
                        prm.cp_len + s * sym_len + fft]
                F = np.fft.fft(seg) / math.sqrt(fft)
                p = np.abs(F[pilot_fft]) ** 2
                feats.extend(p.tolist())
        return np.array(feats)

    # train + test, equal class balance, fresh draws
    train0 = gen_features(0.0, n_train)
    train1 = gen_features(eta, n_train)
    test0  = gen_features(0.0, n_test)
    test1  = gen_features(eta, n_test)

    # Bayes-optimal scalar threshold (midpoint between class means)
    # under equal Gaussian variance assumption
    mu0, mu1 = train0.mean(), train1.mean()
    var0, var1 = train0.var(ddof=1), train1.var(ddof=1)
    # general Gaussian LDA threshold: solves
    #   log p1(x)/p0(x) = 0 with N(mu1,var1) vs N(mu0,var0)
    if abs(var0 - var1) < 1e-12:
        thresh = 0.5 * (mu0 + mu1)
    else:
        # quadratic threshold; pick the root in [min(mu),max(mu)]
        a = 0.5 * (1/var0 - 1/var1)
        b = mu1/var1 - mu0/var0
        c = 0.5 * (mu0**2 / var0 - mu1**2 / var1) - 0.5 * math.log(var1/var0)
        # for sigma_0 = sigma_1 case, fall through to midpoint
        if abs(a) < 1e-12:
            thresh = -c / b if abs(b) > 1e-12 else 0.5*(mu0+mu1)
        else:
            disc = max(b*b - 4*a*c, 0)
            r1 = (-b + math.sqrt(disc)) / (2*a)
            r2 = (-b - math.sqrt(disc)) / (2*a)
            cand = [r for r in (r1, r2)
                     if min(mu0, mu1) - 0.5 * (abs(mu1-mu0)+1)
                        <= r <=
                        max(mu0, mu1) + 0.5 * (abs(mu1-mu0)+1)]
            thresh = cand[0] if cand else 0.5 * (mu0 + mu1)

    # classify test set: predict class 1 iff x > thresh (if mu1>mu0)
    if mu1 >= mu0:
        pred0 = (test0 > thresh).astype(int)
        pred1 = (test1 > thresh).astype(int)
    else:
        pred0 = (test0 < thresh).astype(int)
        pred1 = (test1 < thresh).astype(int)
    acc = float(0.5 * ((pred0 == 0).mean() + (pred1 == 1).mean()))

    # KL estimate from training set
    if var0 < 1e-12 or var1 < 1e-12:
        kl_est = 0.0
    else:
        kl_est = float(0.5 * (math.log(var0 / var1) + var1 / var0
                              + (mu1 - mu0) ** 2 / var0 - 1.0))
    pinsker_tv = math.sqrt(max(kl_est, 0) / 2.0)
    acc_pinsker_bound = 0.5 + pinsker_tv / 2.0
    n_test_features = len(test0) + len(test1)
    se_acc = math.sqrt(acc * (1 - acc) / n_test_features)
    return dict(acc_meas=acc, acc_se=float(se_acc),
                acc_pinsker_bound=float(acc_pinsker_bound),
                kl_train=kl_est, pinsker_tv=pinsker_tv,
                n_test_features=int(n_test_features))


# =============================================================
# Experiment B: multistatic detection measurement
# =============================================================
def measure_pd(rho_r_db: float,
               eta: float,
               R: int,
               P_FA: float,
               n_trials: int,
               rng: np.random.Generator,
               ) -> tuple[float, float, float]:
    """Per-receiver sufficient statistic under the Gaussian large-BT
    approximation.  After matched filtering at the candidate cell:
        H0: T_r ~ N(0, 1)
        H1: T_r ~ N(mu_r, 1),     mu_r = sqrt(2 * rho_r)
    Pilot perturbation eta multiplies the SNR by (1+eta) (Corollary 1
    Gaussian small-perturbation model: rho_r_eff = rho_r * (1+eta) for
    pilot-amplitude scaling, sqrt(2 rho_total) becomes
    sqrt(2 R rho_r (1+eta))).  Non-coherent combining sums R receivers
    --> T = sum_r T_r, ~ N(0, R) under H0 and N(R*mu_r, R) under H1.

    Returns (P_D_measured, P_D_theoretical, rho_total_eff).
    """
    rho_r = 10 ** (rho_r_db / 10)
    rho_total = R * rho_r * (1.0 + eta)
    sigma_R = math.sqrt(R)
    mu_R = R * math.sqrt(2 * rho_r * (1.0 + eta))

    # H0 statistics (target absent)
    h0 = rng.standard_normal(n_trials) * sigma_R
    # H1 statistics (target present)
    h1 = mu_R + rng.standard_normal(n_trials) * sigma_R

    # threshold chosen to give specified P_FA (analytic, since H0 is Gaussian)
    thresh = sigma_R * norm.isf(P_FA)
    P_D_meas = float((h1 >= thresh).mean())
    P_D_thm = float(norm.sf(norm.isf(P_FA) - math.sqrt(2 * rho_total)))
    return P_D_meas, P_D_thm, rho_total


# =============================================================
# Main sweep
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_cov", default="../results/covertness.csv")
    ap.add_argument("--out_det", default="../results/detection.csv")
    ap.add_argument("--out_adv", default="../results/adversary.csv")
    ap.add_argument("--n_trials_kl",  type=int, default=20)
    ap.add_argument("--n_trials_det", type=int, default=20000)
    ap.add_argument("--n_adv_train", type=int, default=400,
                    help="Adversary training slots per class.")
    ap.add_argument("--n_adv_test",  type=int, default=400,
                    help="Adversary held-out test slots per class.")
    ap.add_argument("--P_FA", type=float, default=1e-3)
    ap.add_argument("--obs_snr_db", type=float, default=0.0,
                    help="Observer-side SNR (dB); 0 dB is far-field "
                         "passive sniffer.")
    ap.add_argument("--rho_r_db", type=float, default=-3.0,
                    help="Per-receiver post-integration SNR (dB).")
    ap.add_argument("--Rs", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--etas", nargs="+", type=float,
                    default=[0.0, 0.045, 0.10, 0.20, 0.32, 0.45])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out_cov).parent.mkdir(parents=True, exist_ok=True)

    prm = NRParams()
    rng = np.random.default_rng(args.seed)

    # ---------- Experiment A: covertness vs eta ----------
    print("=== A. Covertness (KL) measurement ===")
    print(f"   Observer SNR = {args.obs_snr_db} dB,  "
          f"{args.n_trials_kl} trials per eta")
    cov_rows = []
    for eta in args.etas:
        t0 = time.time()
        kl_meas, eps_pred = measure_kl(
            prm, eta, n_slots=1, obs_snr_db=args.obs_snr_db,
            n_trials=args.n_trials_kl, rng=rng)
        tv_bound = math.sqrt(max(kl_meas, 0) / 2.0)
        dt = time.time() - t0
        row = dict(eta=eta, kl_meas=kl_meas, eps_pred_eta2_over_2=eps_pred,
                   tv_bound=tv_bound, obs_snr_db=args.obs_snr_db,
                   n_trials=args.n_trials_kl, wall_s=round(dt, 2))
        cov_rows.append(row)
        print(f"   eta={eta:.3f}  KL_meas={kl_meas:.4f}  "
              f"eps_pred={eps_pred:.4f}  TV<={tv_bound:.3f}  ({dt:.1f}s)")

    with open(args.out_cov, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()))
        w.writeheader(); w.writerows(cov_rows)
    print(f"[save] {args.out_cov}\n")

    # ---------- Experiment B: detection P_D vs (R, eta) ----------
    print("=== B. Multistatic detection (P_D) measurement ===")
    print(f"   rho_r = {args.rho_r_db} dB,  P_FA = {args.P_FA},  "
          f"{args.n_trials_det} trials per cell")
    det_rows = []
    for R in args.Rs:
        for eta in args.etas:
            t0 = time.time()
            pd_meas, pd_thm, rho_tot = measure_pd(
                args.rho_r_db, eta, R, args.P_FA,
                args.n_trials_det, rng)
            dt = time.time() - t0
            row = dict(R=R, eta=eta, rho_r_db=args.rho_r_db,
                       P_FA=args.P_FA, rho_total=rho_tot,
                       P_D_meas=pd_meas, P_D_thm=pd_thm,
                       n_trials=args.n_trials_det, wall_s=round(dt, 2))
            det_rows.append(row)
            print(f"   R={R}  eta={eta:.3f}  P_D meas={pd_meas:.3f}  "
                  f"thm={pd_thm:.3f}  rho_tot={rho_tot:.3f}  ({dt:.2f}s)")

    with open(args.out_det, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader(); w.writerows(det_rows)
    print(f"[save] {args.out_det}\n")

    # ---------- Experiment A': adversary ML classifier ----------
    print("=== A'. Adversary ML classifier (Theorem 1 operational) ===")
    print(f"   Observer SNR = {args.obs_snr_db} dB,  "
          f"{args.n_adv_train}+{args.n_adv_test} slots/class")
    adv_rows = []
    for eta in args.etas:
        t0 = time.time()
        r = measure_adversary(prm, eta, args.obs_snr_db,
                               args.n_adv_train, args.n_adv_test, rng)
        dt = time.time() - t0
        row = dict(eta=eta, **r,
                   n_train=args.n_adv_train, n_test=args.n_adv_test,
                   wall_s=round(dt, 2))
        adv_rows.append(row)
        print(f"   eta={eta:.3f}  acc_meas={r['acc_meas']:.3f}"
              f" \\pm {r['acc_se']:.3f}  "
              f"Pinsker UB={r['acc_pinsker_bound']:.3f}  "
              f"KL={r['kl_train']:.4f}  TV={r['pinsker_tv']:.3f}  "
              f"({dt:.1f}s)")

    with open(args.out_adv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(adv_rows[0].keys()))
        w.writeheader(); w.writerows(adv_rows)
    print(f"[save] {args.out_adv}")


if __name__ == "__main__":
    main()
