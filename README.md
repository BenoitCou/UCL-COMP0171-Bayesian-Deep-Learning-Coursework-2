# COMP0171 — Bayesian Deep Learning • Coursework 2  
MSc in Machine Learning, University College London (2024 / 25)

This repository collects everything submitted for **Coursework 2** of the COMP0171 *Bayesian Deep Learning* module.

| Path | Description |
| --- | --- |
| `(Part 1) Uncertainty quantification.ipynb` | Notebook that implements **stochastic-gradient Langevin dynamics (SGLD)** for a Bayesian neural-network classifier on *two-moons* data, decomposes predictive variance into **epistemic / aleatoric** parts, and analyses calibration. |
| `(Part 2) Variational auto-encoder.ipynb` | Notebook that builds and trains a **variational auto-encoder (VAE)** on MNIST, visualises reconstructions & random samples, and explores the learned latent manifold. |
| `two_moons.pt` | Torch tensor with the two-moons dataset used in Part 1. |
| `requirements.txt` | Python dependencies |

---

## Quick-start

```bash
# 1. Clone the repo
git clone https://github.com/<YourUser>/UCL-COMP0171-Bayesian-Deep-Learning-Coursework-2
cd UCL-COMP0171-Bayesian-Deep-Learning-Coursework-2

# 2. (Recommended) set up an isolated environment
python -m venv .venv

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Launch the notebooks
jupyter notebook "(Part 1) Uncertainty quantification.ipynb"
jupyter notebook "(Part 2) Variational auto-encoder.ipynb"
```

---

## Coursework overview

**Part 1 – SGLD & Bayesian uncertainty**

- **Bayesian neural network** — two-layer ReLU MLP (`input = 2 → 32 → 32 → 1`) with a standard normal prior on all weights.  
- **Likelihood / prior** — implemented Bernoulli log-likelihood and isotropic $\mathcal N(0,I)$ prior; mini-batch log-joint with automatic batching for speed.  
- **SGLD sampler** — custom PyTorch loop with Robbins–Monro step-size schedule and *cyclic learning-rate* trick; drew 15 000 samples after burn-in.  
- **Variance decomposition** — estimated total, epistemic and aleatoric variances on a grid; produced heat-maps showing epistemic spikes outside the training support.  
- **Calibration** — generated reliability diagrams and expected-calibration-error (ECE); compared MAP vs Bayesian predictions.  
- **Free-response** — discussed Laplace vs SGLD for decision-boundary complexity; examiner notes disagreement here (“Laplace won’t outperform SGLD even with few samples”).  

**Part 2 – Variational auto-encoder**

- **Architecture** — convolutional encoder → $\mathbb R^{2d}$ for $(\boldsymbol\mu,\log\boldsymbol\sigma)$ with $d = 20$; deconvolutional decoder mirrors the encoder.  
- **Objective** — maximised the ELBO with a standard normal latent prior; used *reparameterisation trick* and Adam optimiser.  
- **Results** —  
  - *Reconstructions* faithfully reproduce digit strokes after 25 epochs.  
  - *Samples* from the prior yield coherent digits with minor blurring.  
  - *Latent space* visualised with a 2-D t-SNE projection shows well-separated digit clusters.  
---

## Marks obtained

| Part | Score | Lecturer’s feedback |
| --- | --- | --- |
| Part 1 | **18 / 19** | Good (if slow…!) implementation; For free response, Good answer overall but I disagree about whether the Laplace approximation will “represent more complex decision boundaries” than SGLD (even for low number of samples…). Note that the data is not exactly the same as in the paper!  |
| Part 2 | **13 / 11** | Great job. |

**Overall grade**: **100 / 100**

---

## Repository structure

```text
UCL-COMP0171-Bayesian-Deep-Learning-CW2/
├── (Part 1) Uncertainty quantification.ipynb   # SGLD sampler, calibration, variance decomposition
├── (Part 2) Variational auto-encoder.ipynb     # VAE on MNIST, reconstructions, samples, latent space
├── two_moons.pt                                # Dataset for Part 1
└── requirements.txt                            # Python dependencies
```
