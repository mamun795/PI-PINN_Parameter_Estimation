# PI-PINN: Incorporating Gaussian Priors into Physics-Informed Neural Networks for Epidemiological Parameter Recovery

**A Gaussian-regularized Physics-Informed Neural Network (PI-PINN) framework for estimating epidemiological transmission parameters (β, γ) from noisy, real-world surveillance data.**

This repository contains the full experimental code accompanying our study on **PI-PINNs** for compartmental epidemic models (SIR / SEIR), evaluated on both synthetic data and **real dengue surveillance data** (Sri Lanka 2017, Philippines 2019). The core contribution is a lightweight **Gaussian maximum-a-posteriori (MAP) prior** term added to the standard PINN loss, which stabilises parameter recovery when the inverse problem is only weakly identifiable — as is typical for real epidemic data where no ground-truth β and γ exist.

---

## Table of Contents

- [Motivation](#motivation)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results Summary](#results-summary)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Physics-Informed Neural Networks (PINNs) can solve the *inverse* problem of recovering unknown ODE parameters directly from observations. However, on real epidemic data the inverse problem is **weakly identifiable**: many (β, γ) pairs fit the data almost equally well, so parameter estimates are highly sensitive to initialization, noise, and random seed.

This work introduces a **Prior-Informed PINN (PI-PINN)** that regularizes the estimate with a soft, literature-informed Gaussian prior over the parameters. The prior acts as a Bayesian MAP regularizer — it nudges estimates toward epidemiologically plausible values **without ever being treated as ground truth** — and is shown to substantially improve stability at negligible computational cost.

## Method

We fit compartmental epidemic models with a neural network trained on a composite loss:

```
L_total = λ₁ · L_data  +  λ₂ · L_physics  +  λ₃ · L_prior
```

- **L_data** — mean-squared error between predicted and observed S / I / R fractions
- **L_physics** — residual of the governing ODEs (SIR or SEIR) enforced at collocation points
- **L_prior** — Gaussian MAP penalty on the learnable parameters: `−log N(β; μ_β, σ_β) − log N(γ; μ_γ, σ_γ)`

Setting **λ₃ = 0** recovers a standard PINN, which serves as the baseline throughout. Priors are drawn from published dengue calibration studies (e.g. Brady et al., 2012) and used **only** as a soft regularizer.

Both **SIR** and **SEIR** (with an added Exposed compartment) variants are implemented.

## Repository Structure

| Notebook | Description |
|----------|-------------|
| `0_Data_Preparation_Dengue_SriLanka.ipynb` | Builds the real dengue S/I/R dataset (normalization, chronological split) |
| `1_Initialization_Sensitivity_RealData.ipynb` | Robustness to parameter initialization on real data (5 init configs × 2 models × 3 seeds) |
| `2_Noise_Robustness_Synthetic_SIR.ipynb` | Parameter recovery under 0 / 5 / 10 / 20 % Gaussian observation noise |
| `3_Multirun_Statistical_Reliability_RealData.ipynb` | 20-seed statistical reliability (Mean ± SD) on real data |
| `4_Prior_Weight_Sensitivity_SIR.ipynb` | Sensitivity of PI-PINN to the prior weight λ₃ (SIR) |
| `5_Prior_Weight_Sensitivity_SEIR.ipynb` | Sensitivity of PI-PINN to the prior weight λ₃ (SEIR) |
| `6_Incorrect_Prior_Robustness_SIR_RealData.ipynb` | Behaviour under a deliberately incorrect prior — SIR, real data |
| `7_Incorrect_Prior_Robustness_SEIR_RealData.ipynb` | Behaviour under a deliberately incorrect prior — SEIR, real data |
| `8_Ablation_Study_Prior_Loss_RealData.ipynb` | Ablation over prior configurations (none / weak / strong / incorrect / correct) |
| `9_Computational_Cost_Analysis.ipynb` | Training time, per-epoch cost & GPU memory: PI-PINN vs. baseline |
| `requirements.txt` | Pinned Python environment (CPU build of PyTorch) |
| `dengue_srilanka_2017.csv` | Real weekly dengue S/I/R data, Sri Lanka 2017 (52 points) |

Each notebook is **self-contained** and can be run top-to-bottom.

## Data

- **`dengue_srilanka_2017.csv`** — 51 weekly Susceptible/Infected/Recovered fractions for the 2017 Sri Lanka dengue season, derived from raw compartment counts divided by total population *N*. Because this is real time-series data, all splits are **chronological (80/10/10)** — never randomly shuffled.
<!-- - **Source:** OpenDengue database (Clarke et al., *Scientific Data*, 2024). -->
- **Source:** [OpenDengue database](https://github.com/OpenDengue/master-repo/blob/main/data/releases/V1.3/Temporal_extract_V1_3.zip) (Clarke et al., *Scientific Data*, 2024)
- **Note:** No ground-truth β / γ exist for real dengue transmission; robustness is therefore assessed via cross-initialization consistency, held-out RMSE, and literature-plausibility bands rather than error against a "true" value.

> Synthetic SIR data used in `2_Noise_Robustness_Synthetic_SIR.ipynb` is generated within the notebook.

## Installation

Requires **Python 3.11**.

```bash
# clone the repository
git clone https://github.com/mamun795/PI-PINN_Parameter_Estimation.git
cd PI-PINN_Parameter_Estimation

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies (CPU build of PyTorch)
pip install -r requirements.txt
```

> The `--extra-index-url` line in `requirements.txt` is **required** for the CPU PyTorch build.
> **On GPU / Google Colab:** do not use the `torch==2.13.0+cpu` pin — Colab ships a CUDA build of PyTorch. Install the remaining packages with `pip install -r requirements.txt --no-deps`, or replace the torch pin with a CUDA build from [pytorch.org](https://pytorch.org).

## Usage

Launch Jupyter and run any notebook end-to-end:

```bash
jupyter notebook
```

Start with `0_Data_Preparation_Dengue_SriLanka.ipynb` to build the dataset, then run the numbered experiments in any order — each is independent.

## Experiments

The experiments were designed to address standard reviewer requirements for ML-based scientific inference:

1. **Initialization robustness** — do runs from very different (β₀, γ₀) converge consistently?
2. **Noise robustness** — how does recovery degrade from 0 % to 20 % observation noise?
3. **Statistical reliability** — 20 independent seeds, reported as Mean ± SD.
4. **Prior-weight sensitivity** — is performance stable across a range of λ₃?
5. **Incorrect-prior robustness** — does a wrong prior harm estimates, and by how much?
6. **Ablation** — isolating the contribution of the prior loss term.
7. **Computational cost** — training time, per-epoch cost, and peak GPU memory vs. baseline.

Statistical comparisons use Welch's *t*-test, Mann–Whitney *U*, bootstrap 95 % CIs, and Levene's test for equality of variances.

## Results Summary

- The Gaussian prior **substantially reduces the variance** of recovered β / γ across seeds and initializations, while keeping estimates inside literature-plausible bands.
- The prior term adds **negligible computational overhead** — per-epoch cost is essentially unchanged from the baseline PINN.
- Benefits are **robust to the prior weight λ₃** and degrade gracefully (rather than catastrophically) under a moderately incorrect prior.

See individual notebooks for full tables and figures.

## Citation

If you use this code, please cite:

```bibtex
@article{yourkey,
  title   = {PI-PINN: Incorporating Gaussian Priors into Physics-Informed Neural Networks for Epidemiological Parameter Recovery},
  author  = {<Your Name> and <Co-authors>},
  journal = {<Journal>},
  year    = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
