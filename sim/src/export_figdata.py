"""Export verified numerical results as .dat tables for pgfplots.

Figures in the manuscript are drawn natively in TikZ/pgfplots from
these tables, so they carry the document's own fonts and vector
geometry rather than a rasterizable external plot.
"""
from __future__ import annotations
import math
import numpy as np

import t1_law as law
import t2_minimax as t2
import t3_optimality as t3
from verify_t2 import mc_j_qpsk, detector_stat, N_P, N_D, N_SYM

OUT = "../results/"
import os
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)


def w(name, header, rows):
    with open(OUT + name, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(f"{x:.8g}" for x in r) + "\n")
    print(name, len(rows))


# 1. information bracket vs observer SNR
rows = []
for gdb in np.linspace(-20, 30, 26):
    g = 10 ** (gdb / 10)
    jd = mc_j_qpsk(1.0 / g, 2000000, rng)   # high count: the three curves
                                            # share an asymptote, so MC noise
                                            # can otherwise flip their order
    rows.append([g,
                 t2.info_power_only(N_P, N_D, g)["I_eff"],
                 t2.info_ratio_detector(N_P, N_D, g),
                 t2.info_blind(N_P, N_D, g, j_d=jd)["I_eff"],
                 t2.info_genie(N_P, N_D, g, j_d=jd)])
w("bracket.dat", "gamma powonly ratio blind genie", rows)

# 2. square-root law: LAN curve and measured points
gamma = 1.0
i_pred = t2.info_ratio_detector(N_P, N_D, gamma)
w("sqrtlaw_lan.dat", "x acc",
  [[x, t2.optimal_accuracy(i_pred, x, N_SYM, 1)[0]]
   for x in np.linspace(0, 0.010, 80)])
for M in (1, 4, 16, 64):
    rows = []
    for xe in (0.002, 0.004, 0.006, 0.008):
        eta = xe / math.sqrt(M)
        a0 = detector_stat(M, 30000, 0.0, gamma, rng)
        a1 = detector_stat(M, 30000, eta, gamma, rng)
        thr = 0.5 * (a0.mean() + a1.mean())
        rows.append([xe, 0.5 * ((a0 < thr).mean() + (a1 >= thr).mean())])
    w(f"sqrtlaw_M{M}.dat", "x acc", rows)

# 3. covert sensing efficiency vs total-power change rate
jp, jd = 2.0, mc_j_qpsk(1.0, 400000, rng)
a = N_P / N_D
rows = []
for c in np.linspace(0.0, 1.6 * a, 160):
    rows.append([2.0 * (N_P - N_D * c) / (N_P + N_D),
                 t3.psi_of_scheme(N_P, N_D, jp, jd, 1.0, -c)])
w("efficiency.dat", "dP psi", rows)
with open(OUT + "efficiency_opt.dat", "w") as f:
    f.write("dP psi\n0 %.8g\n" % t3.psi_max_homogeneous(N_P, N_D, jp, jd))

# 4. covertness-gain frontier.
#
# Computed WITHOUT any expansion in eta.  The quadratic information form
# and G - 1 = 2 eta are first order, and the upper curves of this figure
# reach eta of order unity, where the neglected O(eta^2) term is as large
# as the retained one.  So we solve the exact Pinsker condition using the
# exact divergence upper bound of the sandwich,
#     M n_sym [ N_p g eta^2 + N_d g (beta(eta) - 1)^2 ] <= 2 delta^2,
# and report the exact gain G = (1 + eta)^2 from the derived law.
from scipy.optimize import brentq

delta = 0.1
def eta_exact(g, M):
    def f(e):
        b2 = law.beta_sq(e, a)
        b = math.sqrt(b2) if b2 > 0 else 0.0
        kl = M * N_SYM * g * (N_P * e * e + N_D * (b - 1.0) ** 2)
        return kl - 2.0 * delta ** 2
    hi = law.eta_max(N_P / (N_P + N_D)) * (1 - 1e-9)
    return brentq(f, 1e-14, hi) if f(hi) > 0 else hi

for dr in (1, 5, 20, 50):
    g = t2.gamma_at_standoff(dr, 1.0, 1.0, 3.5)
    rows = [[M, 20 * math.log10(1.0 + eta_exact(g, M))]
            for M in np.logspace(0, 4, 41)]
    w(f"frontier_d{dr}.dat", "M gaindb", rows)
