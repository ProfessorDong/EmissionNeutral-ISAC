"""Render measured-vs-theory figures from the SILENT-SENTRY simulator."""
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG_OUT = HERE.parent.parent / "fig_signature_tradeoff.pdf"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    det = read_csv(RES / "detection.csv")
    cov = read_csv(RES / "covertness.csv")
    adv = read_csv(RES / "adversary.csv")

    Rs = sorted({int(r["R"]) for r in det})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.1))

    AX_LABEL_FS = 12       # axis label
    TICK_FS     = 10       # tick label
    LEGEND_FS   = 10       # legend
    TITLE_FS    = 11       # panel title

    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis="both", labelsize=TICK_FS)

    # ---------- Panel (a): P_D vs eta for several R ----------
    colors = {1: "#9ca3af", 2: "#3b82f6", 4: "#10b981", 8: "#ef4444"}
    markers = {1: "o", 2: "s", 4: "D", 8: "^"}
    for R in Rs:
        rows = [r for r in det if int(r["R"]) == R]
        etas = np.array([float(r["eta"]) for r in rows])
        pd_meas = np.array([float(r["P_D_meas"]) for r in rows])
        pd_thm = np.array([float(r["P_D_thm"]) for r in rows])
        ax1.plot(etas, pd_thm, "-", color=colors[R], lw=1.3,
                 label=f"theory $R{{=}}{R}$")
        ax1.plot(etas, pd_meas, markers[R], color=colors[R], ms=5.5,
                 markerfacecolor="white", markeredgewidth=1.0,
                 label=f"measured $R{{=}}{R}$")
    ax1.set_xlabel(r"pilot perturbation $\eta$",   fontsize=AX_LABEL_FS)
    ax1.set_ylabel(r"detection probability $P_D$", fontsize=AX_LABEL_FS)
    ax1.set_xlim(-0.02, 0.7)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(ncol=2, fontsize=8,
                bbox_to_anchor=(0.985, 0.60), loc="center right",
                handlelength=1.2, columnspacing=0.6,
                framealpha=0.92)
    ax1.set_title(r"(a) $P_D$ vs $\eta$ ($P_{FA}{=}10^{-6}$, $BT{=}10^6$)",
                  fontsize=TITLE_FS)

    # ---------- Panel (b): measured KL vs theory ----------
    etas_c = np.array([float(r["eta"]) for r in cov])
    kl_meas = np.array([float(r["kl_meas"]) for r in cov])
    ax2.plot(etas_c, kl_meas, "ko-", ms=5.5, mfc="white",
             lw=1.1, label="measured KL")
    ax2.plot(etas_c, etas_c**2, "b--", lw=1.1,
             label=r"$\eta^2$ (obs.\ $\sigma^2{=}1$)")
    ax2.plot(etas_c, etas_c**2 / 2.0, "g:", lw=1.1,
             label=r"$\eta^2/2$ (small-pert.)")
    ax2.set_xlabel(r"pilot perturbation $\eta$",   fontsize=AX_LABEL_FS)
    ax2.set_ylabel(r"$\Delta S_{RF}$ (KL nats)",   fontsize=AX_LABEL_FS)
    ax2.set_xlim(-0.02, 0.7)
    ax2.set_ylim(-0.02, 0.75)
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(fontsize=LEGEND_FS, loc="upper left",
                handlelength=1.6, framealpha=0.92)
    ax2.set_title(r"(b) measured $\Delta S_{RF}$ vs $\eta$",
                  fontsize=TITLE_FS)

    # ---------- Panel (c): adversary classifier accuracy ----------
    etas_a = np.array([float(r["eta"]) for r in adv])
    acc = np.array([float(r["acc_meas"]) for r in adv])
    acc_se = np.array([float(r["acc_se"]) for r in adv])
    bound = np.array([float(r["acc_pinsker_bound"]) for r in adv])
    ax3.fill_between(etas_a, acc - 2*acc_se, acc + 2*acc_se,
                     color="red", alpha=0.18)
    ax3.plot(etas_a, acc, "ro-", ms=5.5, mfc="white",
             lw=1.2, label="measured LDA acc.")
    ax3.plot(etas_a, bound, "k--", lw=1.1,
             label=r"Pinsker UB $0.5{+}\sqrt{\epsilon/2}/2$")
    ax3.axhline(0.5, color="gray", lw=0.7, ls=":",
                 label="random guessing")
    ax3.set_xlabel(r"pilot perturbation $\eta$",   fontsize=AX_LABEL_FS)
    ax3.set_ylabel("adversary classifier accuracy", fontsize=AX_LABEL_FS)
    ax3.set_xlim(-0.02, 0.7)
    ax3.set_ylim(0.49, 0.85)
    ax3.grid(True, ls=":", alpha=0.4)
    ax3.legend(fontsize=LEGEND_FS, loc="upper left",
                handlelength=1.6, framealpha=0.92)
    ax3.set_title(r"(c) Theorem 1 operational: acc.\ vs Pinsker bound",
                  fontsize=TITLE_FS)

    fig.tight_layout()
    fig.savefig(FIG_OUT, format="pdf", bbox_inches="tight")
    print(f"[save] {FIG_OUT}")

    # ---------- Summary table for the paper ----------
    print("\n=== Summary for paper Table III ===")
    print("eta   | KL_meas | TV bound | acc_meas | acc_PinskerUB | P_D(R=4) | P_D(R=8)")
    for r in cov:
        eta = float(r["eta"])
        kl = float(r["kl_meas"])
        tv = math.sqrt(max(kl, 0)/2)
        # find adversary acc and P_Ds at this eta
        am = next((float(a["acc_meas"]) for a in adv
                    if abs(float(a["eta"]) - eta) < 1e-6), None)
        ab = next((float(a["acc_pinsker_bound"]) for a in adv
                    if abs(float(a["eta"]) - eta) < 1e-6), None)
        pd4 = next((float(d["P_D_meas"]) for d in det
                     if abs(float(d["eta"]) - eta) < 1e-6
                     and int(d["R"]) == 4), None)
        pd8 = next((float(d["P_D_meas"]) for d in det
                     if abs(float(d["eta"]) - eta) < 1e-6
                     and int(d["R"]) == 8), None)
        print(f" {eta:.3f} | {kl:7.4f} |  {tv:.3f}  |  {am:.3f}  |    {ab:.3f}    |  {pd4:.3f}  |  {pd8:.3f}")


if __name__ == "__main__":
    main()
