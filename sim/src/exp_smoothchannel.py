"""
What does a physically smooth channel buy the network?

Theorem 4 models the nuisance as L independent block gains.  A real
adversary's nuisance is a propagation channel with bounded delay
support, which couples the blocks.  This module replaces the block
model by the physical one and computes the consequence exactly.

The right object is a TANGENT SPACE, and which tangent space depends on
the observation class.

  O_pow (per-RE powers).  The adversary sees only the modulus a_k A_k,
  so the nuisance absorbs any relative AMPLITUDE change it can realize,
      null(M) = { Re(dH_k / H_k) : dH in H_tau }.
  This is psi_max_amplitude() below.

  O_full (complex IQ, known pilot sequence).  On a pilot RE the
  adversary sees the complex mean H_k A_k S_k, so absorbing the
  perturbation means reproducing the whole complex change and not only
  its modulus:
      v_k H_k = dH_k  for every pilot k,  with v real.
  On data REs the symbol is unknown and only the modulus is seen, so
  there v is free to follow Re(dH_k / H_k).  This is psi_max_fulliq(),
  and it is the one that governs the security claim, because O_full
  dominates every other class by the data-processing inequality.

The two differ, and the difference decides the result.  A data RE does
NOT reduce to its modulus: its law is the QPSK mixture
(1/4) sum_s CN(x s, sigma^2) with x = H_k A_k, which is invariant under
x -> i x but not under arbitrary rotation, since E[Z^4] = x^4 E[S^4]
= -x^4 depends on arg x.  In the frame aligned with x the data RE
therefore carries a radial information J_r AND a tangential J_t, both
strictly positive: J_r = 1.468 (the J_d of the flat-channel theory)
and J_t = 0.202 at 0 dB.  Section III of the paper uses J_r alone,
correctly, because there both eta and the scalar gain move x radially.
A multi-tap nuisance moves x tangentially too, and J_t must be kept.

Carrying it, zero efficient information requires

    dH_k / H_k = v_k  real   on EVERY RE, pilot and data alike,

which is N real conditions on the 2D real coordinates of dH, so

    dim null(M) = max(1, 2D - N)                          (generic)

and a zero-detectability direction beyond the global gain needs
2D > N + 1, that is tau_max > 1/(2 df): half the useful symbol.  The
cyclic prefix lasts (n_cp/n_fft)/df, and every OFDM numerology has
n_cp/n_fft < 1/2 (NR: 144/2048 normal, 512/2048 extended), so the
prefix forecloses the covert regime outright, at every numerology and
for every pilot comb.

That last point is what this module exists to settle.  It is tempting
to argue that a channel of delay support tau_max realizes only
amplitude profiles bandlimited to |m| <= tau_max B, so that the comb is
unreachable below the Nyquist delay.  The argument is wrong: delay
support bounds the spectrum of H, not of |H|.  For
H = 1 + r exp(-i w tau) the modulus carries harmonics at every multiple
of tau.  The correct route is the full-IQ tangent above.
"""

from __future__ import annotations
import math
import numpy as np

# 3GPP TR 38.901 V18.0.0 Tables 7.7.2-1..5: (normalized delay, power dB).
# TDL-D and TDL-E list the LOS specular tap and its Rayleigh companion
# separately, both at delay 0.
TDL = {
 'A': [(0,-13.4),(0.3819,0),(0.4025,-2.2),(0.5868,-4),(0.461,-6),(0.5375,-8.2),
       (0.6708,-9.9),(0.575,-10.5),(0.7618,-7.5),(1.5375,-15.9),(1.8978,-6.6),
       (2.2242,-16.7),(2.1718,-12.4),(2.4942,-15.2),(2.5119,-10.8),(3.0582,-11.3),
       (4.081,-12.7),(4.4579,-16.2),(4.5695,-18.3),(4.7966,-18.9),(5.0066,-16.6),
       (5.3043,-19.9),(9.6586,-29.7)],
 'B': [(0,0),(0.1072,-2.2),(0.2155,-4),(0.2095,-3.2),(0.287,-9.8),(0.2986,-1.2),
       (0.3752,-3.4),(0.5055,-5.2),(0.3681,-7.6),(0.3697,-3),(0.57,-8.9),
       (0.5283,-9),(1.1021,-4.8),(1.2756,-5.7),(1.5474,-7.5),(1.7842,-1.9),
       (2.0169,-7.6),(2.8294,-12.2),(3.0219,-9.8),(3.6187,-11.4),(4.1067,-14.9),
       (4.279,-9.2),(4.7834,-11.3)],
 'C': [(0,-4.4),(0.2099,-1.2),(0.2219,-3.5),(0.2329,-5.2),(0.2176,-2.5),(0.6366,0),
       (0.6448,-2.2),(0.656,-3.9),(0.6584,-7.4),(0.7935,-7.1),(0.8213,-10.7),
       (0.9336,-11.1),(1.2285,-5.1),(1.3083,-6.8),(2.1704,-8.7),(2.7105,-13.2),
       (4.2589,-13.9),(4.6003,-13.9),(5.4902,-15.8),(5.6077,-17.1),(6.3065,-16),
       (6.6374,-15.7),(7.0427,-21.6),(8.6523,-22.8)],
 'D': [(0,-0.2),(0,-13.5),(0.035,-18.8),(0.612,-21),(1.363,-22.8),(1.405,-17.9),
       (1.804,-20.1),(2.596,-21.9),(1.775,-22.9),(4.042,-27.8),(7.937,-23.6),
       (9.424,-24.8),(9.708,-30),(12.525,-27.7)],
 'E': [(0,-0.03),(0,-22.03),(0.5133,-15.8),(0.544,-18.1),(0.563,-19.8),
       (0.544,-22.9),(0.7112,-22.4),(1.9092,-18.6),(1.9293,-20.8),(1.9589,-22.6),
       (2.6426,-22.3),(3.7136,-25.6),(5.4524,-20.2),(12.0034,-29.8),(20.6519,-29.2)],
}


DF, NSC, COMB, SIG2 = 30e3, 3276, 12, 1.0
PILOT = (np.arange(NSC) % COMB) == 0
N_P, N_D = int(PILOT.sum()), int((~PILOT).sum())
BW = NSC * DF                          # occupied bandwidth
N_FFT, N_CP = 2048, 144                # normal cyclic prefix, TS 38.211 Cl. 5.3.1
T_CP = N_CP / (N_FFT * DF)
TAU_NYQ = 1.0 / (COMB * DF)            # Nyquist delay of the pilot comb


# ------------------------------------------------------------------
# Closed forms
# ------------------------------------------------------------------
def tau_threshold(df=DF):
    """Delay support above which a zero-detectability direction exists.

    2D > N + 1 with D = ceil(tau B) + 1 and B = N df gives tau > 1/(2 df),
    half the useful symbol.  Independent of the pilot comb.
    """
    return 1.0 / (2.0 * df)


def cp_forecloses(n_fft=N_FFT, n_cp=N_CP):
    """T_cp < 1/(2 df)  <=>  n_cp/n_fft < 1/2, true for every numerology."""
    return n_cp / n_fft < 0.5


def predicted_nullity(n_delay_bins, n_re=NSC):
    return max(1, 2 * n_delay_bins - n_re)


def psi_flat():
    """Single unknown gain, flat channel: the closed form of Theorem 3."""
    j = per_re_information(np.ones(NSC))
    jp, jd = j[PILOT][0], j[~PILOT][0]
    return N_P * N_D * (N_P * jp + N_D * jd) / (NSC ** 2 * jp * jd)


def per_re_information(a):
    """J_p coherent on pilots, J_d power-only on data, at amplitude a_k."""
    j = np.empty(NSC)
    j[PILOT] = 2.0 / SIG2
    j[~PILOT] = 4 * a[~PILOT] ** 2 / (SIG2 ** 2 + 2 * a[~PILOT] ** 2 * SIG2)
    return j


# ------------------------------------------------------------------
# Channels
# ------------------------------------------------------------------
def n_delay_bins(tau_max):
    """A channel of delay support tau_max occupies this many delay bins."""
    return int(math.ceil(tau_max * BW)) + 1


def lowpass_basis(D):
    """Exact basis for {H : delay support <= (D-1)/BW}: the first D delay bins.

    Orthogonal by construction, so no rank truncation is needed -- unlike a
    list of 3GPP taps whose spacings fall below the delay resolution 1/BW.
    """
    k = np.arange(NSC)
    return np.exp(-2j * np.pi * np.outer(k, np.arange(D)) / NSC)


def draw_lowpass_channel(D, rng):
    E = lowpass_basis(D)
    prof = np.exp(-np.arange(D) / max(D / 4.0, 1.0))
    c = (rng.normal(size=D) + 1j * rng.normal(size=D)) * np.sqrt(prof / 2)
    H = E @ c
    return E, H / np.sqrt(np.mean(np.abs(H) ** 2))


def draw_channel(ds_ns, rng, profile='C'):
    """Tap delays (s) and one Rayleigh realization, from a 3GPP TDL profile."""
    taps = TDL[profile]
    d = np.array([t[0] for t in taps]) * ds_ns * 1e-9
    pw = 10 ** (np.array([t[1] for t in taps]) / 10.0)
    pw = pw / pw.sum()
    c = (rng.normal(size=len(d)) + 1j * rng.normal(size=len(d))) * np.sqrt(pw / 2)
    return d, c


def amplitude_and_jacobian(d, c):
    """a_k = |H_k| and D_kj = d a_k / d theta_j for theta = (Re c, Im c)."""
    k = np.arange(NSC)
    E = np.exp(-2j * np.pi * np.outer(k, d) * DF)
    H = E @ c
    a = np.abs(H)
    ph = np.conj(H) / a
    return a, np.concatenate([np.real(ph[:, None] * E),
                              np.real(ph[:, None] * 1j * E)], axis=1)


def block_gains(blocks):
    """The L-free-block nuisance of Theorem 4, in the same interface."""
    D = np.zeros((NSC, len(blocks)))
    for i, idx in enumerate(blocks):
        D[idx, i] = 1.0
    return np.ones(NSC), D


# ------------------------------------------------------------------
# Generic Rayleigh solve, shared by both classes
# ------------------------------------------------------------------
def _solve(lam, Bm, K, tol=1e-9):
    """max_{1'v=0} (p'v)^2 / (v'Lam v - (Bv)'K^+(Bv)).

    With K >= B Lam^{-1} B' this is Lam^{1/2}(I - T T')Lam^{1/2} for
    T = Lam^{-1/2} B' K^{+/2}, whose singular values lie in [0, 1]; the
    unit ones span null(M).  The all-ones direction is always null (a
    global gain change), so the sensing functional is reduced to its
    power-neutral part, which is orthogonal to it by construction.
    """
    w, U = np.linalg.eigh(K)
    keep = w > tol * w.max()
    T = (Bm.T @ (U[:, keep] / np.sqrt(w[keep]))) / np.sqrt(lam)[:, None]
    Uu, sv, _ = np.linalg.svd(T, full_matrices=False)
    gap = 1.0 - sv ** 2
    isnull = gap <= tol
    ptil = PILOT.astype(float) - N_P / NSC
    y = ptil / np.sqrt(lam)
    uty = Uu.T @ y
    if np.linalg.norm(uty[isnull]) > np.sqrt(tol) * max(np.linalg.norm(y), 1.0):
        return np.inf, int(isnull.sum())
    g = np.where(isnull, 0.0, 1.0 / np.where(isnull, 1.0, gap))
    return float(y @ (y + Uu @ ((g - 1.0) * uty))), int(isnull.sum())


def psi_max_amplitude(channels, tol=1e-9):
    """O_pow tangent: the nuisance may realize any relative amplitude change.

    Correct for the power-only class, and an optimistic relaxation for
    O_full, since it lets the nuisance absorb a modulus change without
    paying for the phase change that accompanies it.
    """
    lam = np.zeros(NSC)
    bs = []
    for a, Dj in channels:
        j = per_re_information(a)
        lam += j * a * a
        bs.append((j[:, None] * a[:, None] * Dj).T)
    Bm = np.vstack(bs)
    K = np.zeros((Bm.shape[0], Bm.shape[0]))
    off = 0
    for (a, Dj) in channels:
        j = per_re_information(a)
        n = Dj.shape[1]
        K[off:off + n, off:off + n] = (j[:, None] * Dj).T @ Dj
        off += n
    return _solve(lam, Bm, K, tol)


_QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
_JTAB = None


def qpsk_radial_tangential(gamma, n=2_000_000, seed=1):
    """Fisher information about x for Z = xS + W, S uniform QPSK, |x| = 1.

    Returned in the frame aligned with x: (radial, tangential).  The
    tangential part is strictly positive, which is the fact the
    modulus-only reading of a data RE misses.
    """
    rng = np.random.default_rng(seed)
    sig2 = 1.0 / gamma
    s = _QPSK[rng.integers(0, 4, n)]
    Z = s + (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(sig2 / 2)
    d = Z[:, None] - _QPSK[None, :]
    q = np.abs(d) ** 2
    q -= q.min(axis=1, keepdims=True)
    w = np.exp(-q / sig2)
    w /= w.sum(axis=1, keepdims=True)
    gr = (2.0 / sig2) * np.sum(w * np.real(np.conj(d) * _QPSK[None, :]), axis=1)
    gi = (2.0 / sig2) * np.sum(w * np.real(np.conj(d) * (1j * _QPSK[None, :])), axis=1)
    return float(np.mean(gr * gr)), float(np.mean(gi * gi))


def _jtable():
    global _JTAB
    if _JTAB is None:
        g = np.logspace(-2, 2, 21)
        v = np.array([qpsk_radial_tangential(x, n=150_000) for x in g])
        _JTAB = (g, v[:, 0], v[:, 1])
    return _JTAB


def _fulliq_pieces(E, H):
    """Local-frame weights and Jacobians for the full-IQ observation."""
    g, jr_t, jt_t = _jtable()
    a = np.abs(H)
    loc = a * a / SIG2                                  # per-RE SNR
    Jr = np.where(PILOT, 2.0 / SIG2, np.interp(loc, g, jr_t))
    Jt = np.where(PILOT, 2.0 / SIG2, np.interp(loc, g, jt_t))
    dH = np.concatenate([E, 1j * E], axis=1)
    u = (np.conj(H) / a)[:, None] * dH
    return Jr, Jt, np.real(u), np.imag(u), a


def psi_max_fulliq(E, H, tol=1e-9):
    """O_full tangent, counting the tangential information of QPSK data REs.

    Returns (Psi_max, dim null(M)).  Zero efficient information needs
    dH_k/H_k real on every RE, so the nullity is max(1, 2D - N) and the
    global gain is generically the only absorbable direction.
    """
    Jr, Jt, r_, t_, a = _fulliq_pieces(E, H)
    lam = Jr * a * a
    Bm = (Jr[:, None] * a[:, None] * r_).T
    K = (Jr[:, None] * r_).T @ r_ + (Jt[:, None] * t_).T @ t_
    return _solve(lam, Bm, K, tol)


def psi_of_direction(E, H, v):
    """Psi at a FIXED, channel-independent direction v under channel H.

    This is the operative quantity: the maximiser of psi_max_fulliq
    depends on H, which the transmitter does not observe.
    """
    Jr, Jt, r_, t_, a = _fulliq_pieces(E, H)
    i_ee = float(np.sum(Jr * a * a * v * v))
    i_et = (Jr * a * v) @ r_
    K = (Jr[:, None] * r_).T @ r_ + (Jt[:, None] * t_).T @ t_
    i_eff = i_ee - i_et @ np.linalg.pinv(K, rcond=1e-12) @ i_et
    s = float(np.sum(v[PILOT]))
    return s * s / i_eff


def psi_flat_fulliq():
    """Flat-channel Psi_max with the exact radial data information J_r.

    psi_flat() uses the second-moment value; the full-IQ comparison must
    use the same J_r that psi_max_fulliq() uses, or the ratio mixes
    conventions.
    """
    g, jr_t, _ = _jtable()
    jr = float(np.interp(1.0 / SIG2, g, jr_t))
    return N_P * N_D * (N_P * 2.0 / SIG2 + N_D * jr) / (NSC ** 2 * (2.0 / SIG2) * jr)


def nullity_grid(n_re, n_bins, seed=0):
    """dim { v real : v*H lies in the span of the first n_bins delay bins }.

    This is exactly null(M) of Proposition 3, read straight off the reality
    conditions Im(dH_k/H_k) = 0.  Kept size-parametric because validating
    max(1, 2D - N) across the threshold needs 2D > N, which is only cheap
    at modest N.
    """
    k = np.arange(n_re)
    E = np.exp(-2j * np.pi * np.outer(k, np.arange(n_bins)) / n_re)
    rng = np.random.default_rng(seed)
    prof = np.exp(-np.arange(n_bins) / max(n_bins / 4.0, 1.0))
    c = (rng.normal(size=n_bins) + 1j * rng.normal(size=n_bins)) * np.sqrt(prof / 2)
    H = E @ c
    Ew = E / H[:, None]
    A = np.hstack([np.imag(Ew), np.real(Ew)])
    return A.shape[1] - np.linalg.matrix_rank(A, tol=1e-8 * np.linalg.norm(A, 2))


def absorbed_directions(taps):
    """All real v with v*H in the span of the taps' delay bins, for an
    explicit tap vector.  Used to exhibit the exceptional channels of
    Remark 4, which random draws never produce.

    Returns (H, V) with V the rows spanning the absorbable set; V is
    empty if H vanishes somewhere, where the ratio is undefined.
    """
    taps = np.asarray(taps, complex)
    D = len(taps)
    k = np.arange(NSC)
    E = np.exp(-2j * np.pi * np.outer(k, np.arange(D)) / NSC)
    H = E @ taps
    if np.abs(H).min() < 1e-8:
        return H, np.zeros((0, NSC))
    Ew = E / H[:, None]
    A = np.hstack([np.imag(Ew), np.real(Ew)])
    _, sv, Vt = np.linalg.svd(A)
    mask = np.concatenate([sv <= 1e-8 * sv[0],
                           np.ones(2 * D - len(sv), bool)])
    out = []
    for rowv in Vt[mask]:
        c = rowv[:D] + 1j * rowv[D:]
        v = np.real((E @ c) / H)
        v = v - v.mean()                      # project onto power neutrality
        if np.linalg.norm(v) > 1e-9:
            out.append(v / np.linalg.norm(v))
    return H, np.array(out)


def antisymmetric_direction():
    """The maximiser of Theorem 2: +1/N_p on pilots, -1/N_d on data."""
    return np.where(PILOT, 1.0 / N_P, -1.0 / N_D)


if __name__ == '__main__':
    jr, jt = qpsk_radial_tangential(1.0 / SIG2)
    print(f"N = {NSC}, comb P = {COMB}, N_p = {N_P}, N_d = {N_D}")
    print(f"normal cyclic prefix   = {T_CP*1e6:.4f} us "
          f"({N_CP}/{N_FFT} = {N_CP/N_FFT:.4f} of the useful symbol)")
    print(f"threshold 1/(2 df)     = {tau_threshold()*1e6:.4f} us "
          f"(= 1/2 of the useful symbol)")
    print(f"cyclic prefix forecloses the covert regime: {cp_forecloses()}\n")

    print("(a) An unknown QPSK data RE is NOT a modulus-only observation.")
    print(f"    {'gamma(dB)':>10} {'J_radial':>10} {'J_tangential':>13} "
          f"{'coherent 2/sig2':>16}")
    for gdb in (-10, 0, 10):
        g = 10 ** (gdb / 10)
        r, t = qpsk_radial_tangential(g)
        print(f"    {gdb:>10} {r:>10.4f} {t:>13.4f} {2*g:>16.4f}")
    print("    J_tangential > 0 at every SNR, so a multi-tap nuisance cannot")
    print("    move the mean off the radial axis without being seen.\n")

    print("(b) Exact nullity of the corrected full-IQ tangent: max(1, 2D - N).")
    print("    Validated on a 360-subcarrier grid, where 2D can exceed N cheaply")
    print("    and the rank computation stays well conditioned.")
    print(f"    {'N':>6} {'D':>6} {'2D-N':>7} {'predicted':>10} {'measured':>9}")
    for D in (40, 100, 170, 179, 180, 181, 200, 260, 360):
        print(f"    {360:>6} {D:>6} {2*D-360:>7} {max(1,2*D-360):>10}"
              f" {nullity_grid(360, D):>9}")
    print("")
    print("    At full size, over the delay supports the cyclic prefix admits:")
    print(f"    {'tau(us)':>9} {'D':>6} {'2D-N':>8} {'predicted':>10} {'measured':>9}")
    for tau_us in (0.25, 0.5, 1.0, 2.0, T_CP * 1e6):
        D = n_delay_bins(tau_us * 1e-6)
        E, H = draw_lowpass_channel(D, np.random.default_rng(0))
        print(f"    {tau_us:>9.4f} {D:>6} {2*D-NSC:>8} {predicted_nullity(D):>10}"
              f" {psi_max_fulliq(E, H)[1]:>9}")
    print("    Beyond the prefix the diagonal model no longer holds, so the")
    print("    full-size solve is not carried there; the 360-grid sweep covers")
    print("    the threshold instead.  Within the prefix the nullity is always 1:")
    print("    the global gain, which power neutrality excludes.")
    print("")

    print("(c) What a FIXED, channel-independent perturbation actually gets.")
    flat = psi_flat_fulliq()
    v = antisymmetric_direction()
    print(f"    flat-channel reference Psi = {flat:.4f}")
    print(f"    {'tau(us)':>9} {'D':>6} {'Psi(v)':>10} {'/flat':>8}   [min,max, 6 draws]")
    for tau_us in (0.25, 0.5, 1.0, 2.0, 2.344):
        D = n_delay_bins(tau_us * 1e-6)
        vals = np.array([psi_of_direction(*draw_lowpass_channel(
            D, np.random.default_rng(sd)), v) for sd in range(6)])
        print(f"    {tau_us:>9.3f} {D:>6} {np.median(vals):>10.4f}"
              f" {np.median(vals)/flat:>8.4f}   "
              f"[{vals.min()/flat:.3f}, {vals.max()/flat:.3f}]")
    print("    Frequency selectivity moves it by about two percent, the wrong")
    print("    way for the network.")
