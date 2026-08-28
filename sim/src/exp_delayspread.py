"""
How smooth must the channel be for Theorem 3's uniform-density
conclusion to hold?

Remark 2 of the manuscript says only that a real adversary knows the
channel is smooth across frequency, which re-couples the nuisance
blocks and shrinks null(M).  That is qualitative.  This experiment
makes it exact.

Model the nuisance not as L independent block gains but as what it
physically is: a channel whose delay support is bounded by tau_max.
A real amplitude perturbation that the channel could mimic must lie in

    V_tau = { real sequences whose DFT over subcarriers is
              supported on bins |m| <= M },     M = tau_max * B,

because a delay tau maps to DFT bin m = tau * B, with B = N * SCS the
occupied bandwidth.  null(M) becomes V_tau, and Theorem 3 becomes a
statement about whether the pilot comb is reachable inside V_tau.

The pilot indicator on a period-P comb has DFT support at m = 0 and at
the comb sidebands |m| = N/P, 2N/P, ...  The m = 0 line is the
constant, which power neutrality already excludes.  Covert gain
therefore becomes available exactly when the first sideband enters
V_tau, that is when

    tau_max >= N/(P*B) = 1/(P*SCS) = 1 / (pilot spacing in Hz),

which is the Nyquist condition for pilot-based channel estimation.
"""
from __future__ import annotations
import math
import numpy as np

N, SCS, P = 3276, 30e3, 12
B = N * SCS
NYQ = 1.0 / (P * SCS)          # pilot-comb Nyquist delay


def psi_eff(tau_max: float, J: float = 1.0):
    """Detectability of the optimal power-neutral perturbation once the
    nuisance can realize any amplitude profile in V_tau.
    Returns (Psi_eff, residual energy fraction)."""
    M = int(round(tau_max * B))
    k = np.arange(N)
    a = (N / P) / (N - N / P)                    # N_p / N_d
    v = np.where(k % P == 0, 1.0, -a)            # optimal direction
    V = np.fft.fft(v)
    keep = np.zeros(N, bool)
    keep[0] = True                               # DC is the constant
    if M >= 1:
        keep[1:M + 1] = True
        keep[N - M:] = True
    Vr = V.copy(); Vr[keep] = 0.0                # what the channel cannot mimic
    v_perp = np.real(np.fft.ifft(Vr))
    s = v[k % P == 0].sum()
    det = J * float(v_perp @ v_perp)
    frac = float(v_perp @ v_perp) / float(v @ v)
    return (math.inf if det < 1e-12 else s * s / det), frac


def main():
    print(f"N={N}, SCS={SCS/1e3:.0f} kHz, B={B/1e6:.2f} MHz, pilot period {P}")
    print(f"pilot-comb Nyquist delay 1/(P*SCS) = {NYQ*1e6:.3f} us\n")
    print(f"{'tau_max (us)':>13} {'tau/Nyq':>9} {'bins M':>8} "
          f"{'residual':>10} {'Psi_eff':>12}")
    for t_us in (0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 2.7, 2.77, 2.79, 3.0, 5.0):
        t = t_us * 1e-6
        psi, frac = psi_eff(t)
        print(f"{t_us:>13.2f} {t/NYQ:>9.3f} {int(round(t*B)):>8d} "
              f"{frac:>10.6f} {'inf' if psi == math.inf else f'{psi:12.4f}'}")

    # The comb has sidebands at DFT bins j*N/P, j = 1..P/2, i.e. at
    # delays j/(P*SCS).  Each crossing absorbs one more sideband, so the
    # curve is a staircase; Psi diverges only when the LAST sideband is
    # inside V_tau, at tau = (P/2)/(P*SCS) = 1/(2*SCS).
    print("\nStaircase: each comb sideband absorbed in turn")
    print(f"{'j':>3} {'tau_j (us)':>11} {'residual':>10} {'Psi_eff':>12}")
    for j in range(1, P // 2 + 1):
        t = j / (P * SCS) * 1.001
        psi, frac = psi_eff(t)
        print(f"{j:>3} {j/(P*SCS)*1e6:>11.3f} {frac:>10.6f} "
              f"{'inf' if psi == math.inf else f'{psi:12.4f}'}")
    print(f"  full absorption at 1/(2*SCS) = {1e6/(2*SCS):.3f} us")

    print("\nTypical 3GPP delay-spread scales:")
    for name, ds in (("very short delay", 30e-9),
                     ("short delay / UMi", 100e-9),
                     ("nominal delay / UMa", 363e-9),
                     ("long delay", 1000e-9),
                     ("extreme / hilly", 3000e-9),
                     ("full absorption needs", 1.0/(2*SCS))):
        print(f"  {name:<22} {ds*1e9:6.0f} ns = {ds/NYQ:6.3f} x Nyquist"
              f"  ->  {'COVERT GAIN AVAILABLE' if ds >= NYQ else 'none'}")

    print("\nPilot period vs threshold (30 kHz SCS):")
    for p in (2, 4, 6, 12, 24):
        print(f"  P={p:>3}  spacing {p*SCS/1e3:>5.0f} kHz  "
              f"threshold {1e6/(p*SCS):.3f} us")


if __name__ == "__main__":
    main()
