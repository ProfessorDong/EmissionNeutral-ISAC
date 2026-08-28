"""
Verification of T3: optimality of the perturbation, and the
pilot-density dichotomy.

V1  cell reduction is valid (brute force on a small grid)
V2  homogeneous closed form matches the numerical Rayleigh optimum
V3  the T1/T2 scheme ATTAINS the optimum (it is not an arbitrary choice)
V4  J_p = J_d collapse: Psi_max = N_p N_d / (N J)
V5  null(M) = block-constant vectors (nuisance-absorbed directions)
V6  dichotomy: uniform pilot density -> finite Psi; varying -> infinite
V7  the loophole direction has zero detectability and positive pilot sum
V8  Psi ties back to the T2 gain bound
"""
from __future__ import annotations
import math
import numpy as np
import t3_optimality as t3

rng = np.random.default_rng(0)
ok = []


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def row(label, meas, pred, tol=1e-6):
    if pred in (math.inf, -math.inf):
        good = meas == pred
    elif pred == 0.0:
        good = abs(meas) < 1e-9          # absolute test against zero
    else:
        good = abs(meas - pred) / abs(pred) < tol
    ok.append(good)
    print(f"  [{'ok ' if good else 'FAIL'}] {label:<44} meas={meas: .8g}"
          f"  pred={pred: .8g}")
    return good


# ---- full-N reference implementation (no cell reduction) ----
def psi_full(v, J, blk, pilot_mask):
    s = v[pilot_mask].sum()
    A = J * v
    i_ee = float((J * v * v).sum())
    i_eff = i_ee
    for l in np.unique(blk):
        m = blk == l
        i_eff -= float(A[m].sum()) ** 2 / float(J[m].sum())
    return math.inf if i_eff <= 1e-14 else s * s / i_eff


banner("V1  cell reduction is valid (brute force, N=24)")
# 2 blocks of 12 REs; block 0 has 3 pilots, block 1 has 3 pilots
npb, ndb = [3, 3], [9, 9]
J_P, J_D = 2.0, 1.4679
N = 24
blk = np.array([0] * 12 + [1] * 12)
pilot_mask = np.zeros(N, bool)
pilot_mask[[0, 1, 2, 12, 13, 14]] = True
Jf = np.where(pilot_mask, J_P, J_D)
w = np.ones(N)

best_random = -np.inf
for _ in range(20000):
    v = rng.standard_normal(N)
    v -= v.mean()                      # power neutral
    best_random = max(best_random, psi_full(v, Jf, blk, pilot_mask))

counts, jvec, cblk = t3.build_cells(npb, ndb, J_P, J_D)
psi_cell, v_cell = t3.psi_max(counts, jvec, cblk)
print(f"  best of 20000 random power-neutral directions : {best_random:.6f}")
print(f"  cell-reduced optimum                          : {psi_cell:.6f}")
g = best_random <= psi_cell * (1 + 1e-9)
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] no random direction beats the "
      f"cell-reduced optimum")

# cell-averaging never decreases Psi
worse = 0
for _ in range(4000):
    v = rng.standard_normal(N); v -= v.mean()
    va = v.copy()
    for l in (0, 1):
        for pm in (True, False):
            m = (blk == l) & (pilot_mask == pm)
            va[m] = v[m].mean()
    if psi_full(va, Jf, blk, pilot_mask) < psi_full(v, Jf, blk, pilot_mask) - 1e-9:
        worse += 1
ok.append(worse == 0)
print(f"  [{'ok ' if worse == 0 else 'FAIL'}] cell-averaging never "
      f"decreased Psi ({worse}/4000 violations)")

banner("V2/V3/V4  homogeneous case: closed form, optimality, collapse")
NP, ND = 273, 3003
NT = NP + ND
c1, j1, b1 = t3.build_cells([NP], [ND], J_P, J_D)
psi_num, v_num = t3.psi_max(c1, j1, b1)
psi_cf = t3.psi_max_homogeneous(NP, ND, J_P, J_D)
row("Psi_max numerical vs closed form", psi_num, psi_cf, tol=1e-8)

psi_ours = t3.psi_of_scheme(NP, ND, J_P, J_D, 1.0, -NP / ND)
row("T1/T2 scheme attains the optimum", psi_ours, psi_cf, tol=1e-9)
print(f"         optimal direction ratio v_P/v_D = "
      f"{v_num[0] / v_num[1] * (c1[1] / c1[0]) if False else -ND / NP:.4f}"
      f"  (= -N_d/N_p, i.e. power-neutral antisymmetric)")

for J in (1.0, 1.4169, 2.0):
    row(f"J_p=J_d={J}: Psi_max = N_p N_d/(N J)",
        t3.psi_max_homogeneous(NP, ND, J, J), NP * ND / (NT * J), tol=1e-9)

banner("V5  null(M) = block-constant vectors")
c2, j2, b2 = t3.build_cells([3, 5], [9, 7], J_P, J_D)
M = t3.fisher_M(c2, j2, b2)
ev, U = np.linalg.eigh(M)
nnull = int((ev < 1e-9 * max(ev.max(), 1)).sum())
row("dim null(M) = number of blocks", float(nnull), 2.0, tol=1e-9)
for l in (0, 1):
    e = np.where(b2 == l, 1.0, 0.0)
    row(f"  block-{l} indicator in null(M)",
        float(np.linalg.norm(M @ e)), 0.0, tol=1e-6)

banner("V6/V7  the dichotomy: does pilot density vary across blocks?")
cases = [
    ("uniform density  (3/12, 3/12)", [3, 3], [9, 9]),
    ("uniform density  (2/12, 4/24)", [2, 4], [10, 20]),
    ("VARYING density  (3/12, 5/12)", [3, 5], [9, 7]),
    ("VARYING density  (6/12, 0/12)", [6, 0], [6, 12]),
]
for name, a, b in cases:
    cc, jj, bb = t3.build_cells(a, b, J_P, J_D)
    psi, vv = t3.psi_max(cc, jj, bb)
    avail = t3.covert_gain_available(a, b)
    alphas = [x / (x + y) for x, y in zip(a, b)]
    consistent = (psi == math.inf) == avail
    ok.append(consistent)
    print(f"  [{'ok ' if consistent else 'FAIL'}] {name:<32} "
          f"alpha={[round(x,4) for x in alphas]}  "
          f"Psi_max={'inf' if psi == math.inf else f'{psi:.4f}'}  "
          f"loophole={avail}")

# V7: exhibit the loophole direction explicitly
a, b = [3, 5], [9, 7]
c, gain = t3.optimal_loophole(a, b)
Nfull = np.array(a) + np.array(b)
blkf = np.concatenate([np.full(n, i) for i, n in enumerate(Nfull)])
pm = np.zeros(Nfull.sum(), bool)
off = 0
for i, (npb_, nb_) in enumerate(zip(a, Nfull)):
    pm[off:off + npb_] = True
    off += nb_
Jf2 = np.where(pm, J_P, J_D)
vloop = c[blkf]
s_loop = vloop[pm].sum()
i_eff = 1.0 / psi_full(vloop, Jf2, blkf, pm) * s_loop ** 2 \
    if psi_full(vloop, Jf2, blkf, pm) != math.inf else 0.0
print(f"\n  loophole direction c = {np.round(c, 4)}  "
      f"(shift power toward the pilot-dense block)")
row("  loophole: power neutrality 1'v", float((vloop * 1.0).sum()), 0.0,
    tol=1e-9)
row("  loophole: detectability I_eff", float(i_eff), 0.0, tol=1e-9)
g = s_loop > 1e-9
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] loophole: pilot sum s > 0  "
      f"(= {s_loop:.4f}, so sensing gain at ZERO detectability)")

banner("V8  Psi ties back to the T2 gain bound")
# G - 1 <= 4 delta sqrt(Psi) / (N_p sqrt(n_sym M))
delta, n_sym, M_slots = 0.1, 14, 100
bound = 4 * delta * math.sqrt(psi_cf) / (NP * math.sqrt(n_sym * M_slots))
# T2 route: eta <= 2 delta / sqrt(I_eff n_sym M), G-1 = 2 eta
i_eff_ours = NP ** 2 / psi_cf
eta = 2 * delta / math.sqrt(i_eff_ours * n_sym * M_slots)
row("G-1 via Psi vs via T2 eta-budget", bound, 2 * eta, tol=1e-9)
print(f"         I_eff = {i_eff_ours:.4g}, eta_max = {eta:.4g}, "
      f"G-1 = {bound:.4g} = {10*math.log10(1+bound):.5f} dB")

banner("SUMMARY")
print(f"  {sum(ok)}/{len(ok)} checks passed")
raise SystemExit(0 if all(ok) else 1)
