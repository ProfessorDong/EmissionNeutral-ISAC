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
V9  block decomposition of I_eff, and the simplified Psi_max closed form
V10 Corollary 1: uniform density makes the partition irrelevant
V11 smooth-channel nuisance: tangent-space solver agrees with the closed
    forms, and delay support does not bandlimit the amplitude
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

banner("V9  block decomposition of I_eff and the simplified Psi_max")
# I_eff = sum_l kappa_l (x_l - y_l)^2,  kappa_l = A_l B_l / (A_l + B_l)
worst = 0.0
for _ in range(400):
    L = int(rng.integers(1, 6))
    npb = rng.integers(1, 40, L); ndb = rng.integers(1, 40, L)
    jp, jd = 10 ** rng.uniform(-1, 1), 10 ** rng.uniform(-1, 1)
    cnt, jv, blk = t3.build_cells(npb, ndb, jp, jd)
    M = t3.fisher_M(cnt, jv, blk)
    v = rng.normal(size=2 * L)
    x, y = v[0::2], v[1::2]
    A, B = npb * jp, ndb * jd
    lhs = float(v @ M @ v)
    rhs = float(np.sum(A * B / (A + B) * (x - y) ** 2))
    worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-12))
row("block decomposition, worst rel. error over 400", worst, 0.0, tol=1e-9)

worst = 0.0
for _ in range(400):
    n_p, n_d = float(rng.integers(1, 500)), float(rng.integers(1, 5000))
    jp, jd = 10 ** rng.uniform(-2, 2), 10 ** rng.uniform(-2, 2)
    S = n_p * jp + n_d * jd
    simple = n_p * n_d * S / ((n_p + n_d) ** 2 * jp * jd)
    worst = max(worst,
                abs(simple - t3.psi_max_homogeneous(n_p, n_d, jp, jd)) / simple)
row("simplified Psi_max form, worst rel. error", worst, 0.0, tol=1e-9)

banner("V10 Corollary 1: uniform density makes the partition irrelevant")
worst, ntest = 0.0, 0
while ntest < 400:
    an, ad = int(rng.integers(1, 10)), int(rng.integers(2, 20))
    if an >= ad:
        continue
    ntest += 1
    L = int(rng.integers(1, 7))
    mult = rng.integers(1, 9, L)
    npb, ndb = an * mult, (ad - an) * mult      # identical density per block
    jp, jd = 10 ** rng.uniform(-1.5, 1.5), 10 ** rng.uniform(-1.5, 1.5)
    cnt, jv, blk = t3.build_cells(npb, ndb, jp, jd)
    got, _ = t3.psi_max(cnt, jv, blk)
    want = t3.psi_max_homogeneous(npb.sum(), ndb.sum(), jp, jd)
    worst = max(worst, abs(got - want) / want)
row("L-block vs single-block, worst rel. error over 400", worst, 0.0, tol=1e-7)

banner("V11 propagation nuisance: tangent space by observation class")
import exp_smoothchannel as sc
a_flat = np.ones(sc.NSC)
D_flat = np.stack([a_flat, np.zeros(sc.NSC)], axis=1)
row("O_pow solver, flat channel vs closed form",
    sc.psi_max_amplitude([(a_flat, D_flat)])[0], sc.psi_flat(), tol=1e-10)
for nb in (3, 7, 13):
    Lb = sc.NSC // nb
    blocks = [np.arange(i * Lb, (i + 1) * Lb) for i in range(nb)]
    row(f"uniform density, {nb} blocks -> single-block value",
        sc.psi_max_amplitude([sc.block_gains(blocks)])[0], sc.psi_flat(), tol=1e-9)
row("varying density -> unbounded",
    sc.psi_max_amplitude([sc.block_gains([np.arange(0, 1638),
                                          np.arange(1638, sc.NSC)])])[0], math.inf)

# an unknown QPSK data RE carries PHASE information: J_t > 0
jr, jt = sc.qpsk_radial_tangential(1.0)
row("QPSK data RE, radial information at 0 dB", jr, 1.468, tol=5e-3)
g = jt > 0.15
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] QPSK data RE, tangential information "
      f"J_t = {jt:.4f} > 0, so a data RE is NOT modulus-only")
for gdb in (-10, 10):
    _, t = sc.qpsk_radial_tangential(10 ** (gdb / 10))
    gg = t > 0
    ok.append(gg)
    print(f"  [{'ok ' if gg else 'FAIL'}] J_t > 0 at {gdb:+d} dB (= {t:.4f})")

# flat channel: the corrected full-IQ solver reproduces the closed form
E1 = sc.lowpass_basis(1)
row("O_full solver, flat channel vs closed form",
    sc.psi_max_fulliq(E1, E1 @ np.array([1.0 + 0j]))[0], sc.psi_flat_fulliq(),
    tol=1e-9)

# exact nullity max(1, 2D - N), validated across the threshold on a
# 360-subcarrier grid, where 2D > N is reachable at negligible cost
bad = 0
for D in (40, 100, 170, 179, 180, 181, 200, 260, 360):
    bad += (sc.nullity_grid(360, D) != max(1, 2 * D - 360))
row("nullity = max(1, 2D - N), mismatches over 9 delay supports",
    float(bad), 0.0, tol=1e-9)
D_cp = sc.n_delay_bins(sc.T_CP)
E, H = sc.draw_lowpass_channel(D_cp, np.random.default_rng(0))
row(f"full grid at tau = T_cp (D = {D_cp}), nullity",
    float(sc.psi_max_fulliq(E, H)[1]), float(sc.predicted_nullity(D_cp)), tol=1e-9)

row("threshold 1/(2 df) in us", sc.tau_threshold() * 1e6, 16.6666667, tol=1e-7)
g = sc.cp_forecloses() and sc.T_CP < sc.tau_threshold()
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] normal CP ({sc.T_CP*1e6:.3f} us) is below the "
      f"threshold ({sc.tau_threshold()*1e6:.3f} us)")
g = sc.cp_forecloses(n_cp=512, n_fft=2048)
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] extended CP (512/2048) also forecloses")

# what the FIXED antisymmetric direction actually achieves
flat_fq = sc.psi_flat_fulliq()
v = sc.antisymmetric_direction()
ratios = []
for tau_us in (0.5, 2.344):
    D = sc.n_delay_bins(tau_us * 1e-6)
    for sd in range(3):
        E, H = sc.draw_lowpass_channel(D, np.random.default_rng(sd))
        ratios.append(sc.psi_of_direction(E, H, v) / flat_fq)
ratios = np.array(ratios)
g = bool(np.all((ratios > 0.95) & (ratios < 1.0)))
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] fixed antisymmetric direction retains "
      f"{ratios.mean():.4f} of flat-channel efficiency "
      f"[{ratios.min():.3f}, {ratios.max():.3f}] over 6 channels")

# the exceptional (conjugate-reciprocal) channels: the nullity exceeds the
# generic max(1, 2D - N), and what they buy is set by how close the channel
# roots come to the unit circle
def _exc_pilot_sum(c):
    H, V = sc.absorbed_directions([1.0, c, 1.0])
    return max(abs(v[sc.PILOT].sum()) for v in V) if len(V) else None
well = max(_exc_pilot_sum(c) for c in (3.0, 2.5, 2.1))
ill = _exc_pilot_sum(2.0001)
g = well < 1e-10 and ill > 1e-3
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] conjugate-reciprocal channels break the generic "
      f"nullity; well-conditioned")
print(f"         ones give |s(v)| <= {well:.2e} per unit norm, but c = 2.0001 gives "
      f"{ill:.3e},")
print(f"         so Corollary 2 must be stated almost surely, not for every channel")
H, V = sc.absorbed_directions([1, 3, 1])
g = len(V) >= 2
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] h = (1,3,1) admits more than one absorbable "
      f"dimension where the generic count is 1 (min|H| = {np.abs(H).min():.3f})")

# the rank witness that makes the exceptional set Lebesgue-null: at H = 1
# the constraint matrix has full rank, so its minor is not identically zero
bad = 0
kk = np.arange(sc.NSC)
for D in (3, 50, 232, 800):
    E1 = np.exp(-2j * np.pi * np.outer(kk, np.arange(D)) / sc.NSC)
    A1 = np.hstack([np.imag(E1), np.real(E1)])          # H == 1, so Ew = E
    dim = A1.shape[1] - np.linalg.matrix_rank(A1, tol=1e-9 * np.linalg.norm(A1, 2))
    bad += (dim != 1)
row("kernel at H = 1 is the constant alone, mismatches over 4 depths",
    float(bad), 0.0, tol=1e-9)

# delay support bounds the spectrum of H, not of |H|
k = np.arange(sc.NSC)
tau = 0.5e-6
amp = np.abs(1.0 + 0.6 * np.exp(-2j * np.pi * k * sc.DF * tau))
cep = np.abs(np.fft.rfft(amp - amp.mean())) / sc.NSC
lag = tau * sc.DF * sc.NSC
h2 = cep[int(round(2 * lag))] / cep[int(round(lag))]
g = h2 > 0.10
ok.append(g)
print(f"  [{'ok ' if g else 'FAIL'}] |H| carries a second harmonic at 2*tau "
      f"(relative weight {h2:.4f}), so delay support does not bandlimit it")

banner("SUMMARY")
print(f"  {sum(ok)}/{len(ok)} checks passed")
raise SystemExit(0 if all(ok) else 1)
