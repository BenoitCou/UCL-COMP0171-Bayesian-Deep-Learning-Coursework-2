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
git clone https://github.com/BenoitCou/UCL-COMP0171-Bayesian-Deep-Learning-Coursework-2
cd UCL-COMP0171-Bayesian-Deep-Learning-Coursework-2

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the notebooks
jupyter notebook "(Part 1) Uncertainty quantification.ipynb"
jupyter notebook "(Part 2) Variational auto-encoder.ipynb"
```

---

## Coursework overview

**Part 1 – SGLD & Bayesian uncertainty**

- **Bayesian neural network** — two-layer ReLU MLP (`input = 2 → 100 → 10 → 1`) with a standard normal prior on all weights.  
- **Likelihood / prior** — implemented Bernoulli log-likelihood and standard Gaussian $\mathcal{N}(0, I)$ prior on parameters using PyTorch utility `nn.utils.parameters_to_vector`; included epsilon for numerical stability.  
- **Mini-batch log-joint** — derived unbiased mini-batch estimator of $\log p(\theta, y | X)$ with scaling for likelihood; verified numerically via histogram match to full-data log-joint.  
- **MAP training** — trained network with Adam optimizer for 400 epochs on 2D two-moons dataset; tracked log-joint over epochs for convergence diagnostics.  
- **Confidence visualization** — computed model confidence on a 2D grid; generated contour plot of predicted class confidence; verified class boundaries.  
- **Calibration** — produced reliability diagrams and computed Expected Calibration Error (ECE); analyzed sensitivity to bin count.  
- **SGLD sampler** — implemented custom SGLD optimizer using Robbins–Monro updates with cosine cyclic learning rate schedule; collected 50 posterior samples after burn-in.  
- **Monte Carlo prediction** — ran forward passes using 50 sampled parameter vectors to obtain predictive distribution; generated Bayesian confidence plot and updated calibration diagrams.  
- **Variance decomposition** — estimated total predictive variance, epistemic uncertainty (variance over $f_\theta(x)$), and aleatoric uncertainty (difference) on a 2D input grid; visualized each as contour plots.  
- **Free-response** — discussed SGLD vs Laplace approximation in terms of posterior geometry, uncertainty quantification, and decision-boundary expressiveness; critically examined calibration and ECE behavior from Kristiadi et al. (2020).

**Part 2 – Variational Autoencoder on MNIST**

- **Dataset & preprocessing** — loaded binary MNIST images with a rounded `ToTensor()` transform; data is reshaped to shape `(1, 28, 28)` per sample; plotted batches using `make_grid` to verify input format.
- **Latent representation** — learned latent representations $z_i \in \mathbb{R}^{10}$ per input image; encoder outputs variational distribution $q(z_i|x_i)$ as a diagonal Gaussian; decoder defines $p(x_i|z_i)$ as a Bernoulli distribution over pixels.
- **Encoder architecture** — convolutional network producing $\mu$ and $\sigma$ vectors for $q(z|x)$:
  - Two convolutional layers: `Conv2d(1→6)` + `Softmax`, `Conv2d(6→9)` + `ReLU`, then `Flatten` and `Linear` to latent space.
  - Separate subnetworks for $\mu$ and $\sigma$, the latter ending with `Softplus` for positivity.
  - Total encoder parameter count (with $D_{\text{latent}}=10$): **~9,899 parameters**.
- **Decoder architecture** — transposed convolutional network:
  - `Linear(D_latent→784)` → `ReLU` → `Unflatten` → `ConvTranspose2d(16→8)` + `Softmax2d` → `ConvTranspose2d(8→1)` + `Sigmoid`.
  - Returns a Bernoulli distribution over image pixels.
  - Total decoder parameter count: **~8,785 parameters**.
- **ELBO objective** — implemented per-datapoint ELBO:
  $$
  \mathbb{E}_{q(z|x)}\left[ \log p(x|z) + \log p(z) - \log q(z|x) \right]
  $$
  - Used standard normal prior $\mathcal{N}(0, I)$ for $p(z)$.
  - Monte Carlo approximation with 10 samples per batch.
  - Computed reconstruction log-likelihood via decoder output and KL divergence from prior.
- **Training loop** — optimized ELBO with Adam optimizer on encoder and decoder parameters for 10 epochs; batch size 100; per-epoch training loss tracked.
- **Qualitative evaluation** — evaluated reconstruction performance on held-out MNIST samples:
  - **Reconstruction accuracy**: ~0.92 (rounded binary image match).
  - Reconstructions and original images visually compared using `make_grid`.
- **Sampling from prior** — generated coherent images by sampling from $\mathcal{N}(0, I)$ in latent space and decoding; visualized 140 randomly sampled digits.
- **Latent space visualization** — projected high-dimensional latent space onto 2D using top two right-singular vectors of encoder mean outputs; decoded latent grid to visualize digit morphology across latent manifold.
- **Parameter efficiency & bonus** — both encoder and decoder architectures kept under 10k parameters while achieving realistic reconstructions, qualifying for full bonus credit.

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
