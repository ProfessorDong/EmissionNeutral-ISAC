# EmissionNeutral-ISAC

**Emission-Neutral Integrated Sensing and Communications for
counter-UAS sensing in multi-modal 5G/NTN networks.**

Code, measurement CSVs, and the measured-vs-theory figure
accompanying:

> Liang Dong, *"Emission-Neutral ISAC for Counter-UAS Sensing in
> 5G/NTN Networks,"* MILCOM 2026 (under review).

This repository contains a self-contained Monte-Carlo simulator
that reproduces every numerical result, table cell, and figure
panel in the conference paper. The paper PDF itself is not hosted
here.

## Highlights

* **RF Signature Increment** — $\Delta S_{RF}(\pi) := D_\mathrm{KL}(p_\pi \| p_0)$
  on observer-facing per-pilot power features. Defines a first-class
  signature budget alongside detection probability and QoS.

* **Theorem 1 (Covertness from Signature Constraint)** — if
  $\Delta S_{RF}(\pi) \le \epsilon$, any RF observer's
  total-variation advantage in detecting that sensing is active is
  at most $\sqrt{\epsilon/2}$ (Pinsker route mirroring the
  covert-communications square-root law). Sanity-checked here
  against an LDA classifier (Bayes-optimal only under the
  equal-covariance Gaussian assumption) on standards-compliant
  5G NR DM-RS features.

* **Closed-form multistatic detection** — $R$ uncorrelated passive
  receivers achieve
  $P_D = Q\bigl(Q^{-1}(P_{FA}) - \sqrt{2 \rho_\mathrm{tot}}\bigr)$
  with $\rho_\mathrm{tot} = \sum_r \rho_r$ (matched-filter output
  SNR, time-bandwidth gain already absorbed) under non-coherent
  combining. Matched here to within $0.001$ across all 28 (R, η)
  cells.

* **Lemma 1 (Indifference Point)** — $\epsilon_\mathrm{indiff}(R, K) = K^2/(2R^2)$
  for $K$ collusion-resistant rounds. With homogeneous independent
  receivers and negligible deployment/sync/backhaul/incremental-signature
  cost, signature spending below the bound is SNR-dominated by
  adding receivers. Operational recipe:
  **spend receivers before spending signature.**

## Layout

```
sim/
  src/
    silent_sentry_sim.py   Monte-Carlo simulator
                           - standards-compliant 5G NR FR1 OFDM
                             (100 MHz, 30 kHz SCS, 273 RBs = 3276
                             active subcarriers, Type-1 DM-RS pilots)
                           - bistatic clutter+target with Swerling-I
                             amplitude draws
                           - non-coherent multistatic NP detector
                           - LDA adversary on per-pilot power
                             features (Bayes-optimal only under
                             equal-covariance Gaussian)
    make_figures.py        renders the 3-panel measured-vs-theory
                           figure from the CSV outputs
  results/                 measured CSVs (committed for inspection)
    covertness.csv         measured KL on per-pilot power features
                           vs eta
    detection.csv          measured P_D vs (R, eta) compared to the
                           closed-form prediction
    adversary.csv          measured LDA accuracy vs the Pinsker
                           upper bound at each eta
    sweep.csv              full Monte-Carlo sweep
fig_signature_tradeoff.pdf measured-vs-theory figure used in the
                           paper (regeneratable by make_figures.py)
```

## Quick start

```bash
# Python deps
pip install numpy scipy matplotlib

# Reproduce all numbers in the paper
cd sim/src
python silent_sentry_sim.py \
    --n_trials_kl 200 \
    --n_trials_det 200000 \
    --n_adv_train 80 --n_adv_test 80 \
    --P_FA 1e-6 \
    --obs_snr_db 0 --rho_r_db 5 \
    --etas 0.0 0.045 0.10 0.20 0.32 0.45 0.63 \
    --seed 0

# Regenerate the paper's Fig. 2 from the CSVs
python make_figures.py
```

The full sweep runs in under one minute on a single CPU core.

## What gets validated

| Result | Validation route | Headline |
|---|---|---|
| Theorem 1 (Pinsker covertness) | LDA classifier accuracy vs UB across 7 η values | classifier stays strictly below $0.5 + \sqrt{\epsilon/2}/2$ at every η; gap widens once KL exceeds $\sim 0.1$ (Pinsker becomes loose) |
| Closed-form $P_D$ | Monte Carlo, $2\times 10^5$ trials per cell, 28 cells; this is code verification rather than independent theoretical validation | simulated matches theory to $\le 0.001$ |
| $\Delta S_{RF} = O(\eta^2)$ | estimated KL on 200 OFDM slot realisations per η on the LDA feature family | growth rate consistent with the small-perturbation prediction on this feature family |
| Lemma 1 indifference | numerical: $R=4$ baseline $P_D = 0.609$, $R=8$ baseline $P_D = 0.991$ | under the homogeneous-cost assumptions of the lemma, doubling receivers SNR-dominates any signature budget below $\epsilon \sim 0.5$ nats |

## Citation

```bibtex
@inproceedings{dong_emission_neutral_isac_milcom2026,
  author    = {Liang Dong},
  title     = {{Emission-Neutral ISAC for Counter-UAS Sensing in
                5G/NTN Networks}},
  booktitle = {Proc. IEEE Military Communications Conference (MILCOM)},
  year      = {2026},
  note      = {Under review}
}
```

## License

Code released under the MIT License (see `LICENSE`).
