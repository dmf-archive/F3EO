# F3EO-Bench: A Lightweight Framework for Advanced Optimizer Evaluation

[简体中文](./README_CN.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/F3EO)

> **F3EO-Bench** is a research framework for prototyping and evaluating advanced neural network optimizers, centered on the principle of **Energy-Geometry Decoupling**.

## 1. Core Principle: Energy-Geometry Decoupling

Modern deep learning optimization faces a trade-off between statistical adaptivity (how fast to learn) and structural stability (what to learn). F3EO-Bench facilitates research into optimizers that decouple these concerns. Our flagship optimizer, **AdaRMSuon**, operationalizes this principle:

1. **Statistical Operator (Energy)**: Uses the scalar norm of AdamW's second-moment-corrected momentum (‖m̂ / √v̂‖) to determine the update magnitude. This acts as a computationally cheap proxy for the rate of free-energy descent.
2. **Structural Operator (Geometry)**: Employs Muon's Newton-Schulz iteration to find an orthogonal update direction (O_t). **AdaRMSuon** further introduces **Pre-whitening**, whitening the gradient with v_t before projection to ensure the update follows the geodesic of the Riemannian manifold.

The final update is a composition of these two operators: `g_update = scale * O_t`. This decouples "how fast" from "where to go," providing a robust and efficient path towards the minimum.

## 2. Advanced Evolution: From Geometry to Topology (ARS)

While AdaRMSuon converges extremely fast, it tends to fall into sharp local minima (overfitting). To address this, we introduced **ARS (AdaRMSuon Regularized Search)**, which adds topological flatness constraints on top of the geometric gliding.

- **Manifold-Aware SAM**: ARS calculates the adversarial direction not in Euclidean space, but on the Riemannian manifold defined by v_t, searching for "flat" geodesic regions.
- **Lazy Mode & Shear Force Injection**: To avoid the double computational cost of SAM, we implemented Lazy Mode (k > 1). In non-perturbation steps, we inject a "Shear Force" (v_flat) orthogonal to the base gradient, continuously pushing the model away from sharp regions.
- **Intensity Compensation**: Experiments show that in Lazy Mode, the injection intensity (α) must be increased to compensate for the bias caused by low-frequency corrections. The configuration k=5, α=0.1 achieves the best balance between training speed and generalization performance.

## 3. Key Experimental Results

### 3.1 Wikitext-2 Language Modeling

We validated these optimizers on Wikitext-2 (`line mode`). This mode preserves sentence boundaries, maximizing the contextual integrity of the input.

| Optimizer | Core Mechanism | Epoch 1 PPL | Best PPL | Final PPL | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Muon | SGD + Newton-Schulz | 233.30 | 161.09 | 161.09 | Baseline |
| RMSuon | Adam + Newton-Schulz | 146.52 | 99.07 | 134.11 | Early Version |
| AdaRMSuon | Pre-white + Energy | **133.68** | 83.88 | 87.61 | Fastest Convergence, Overfits |
| ARS (Sync) | Manifold SAM (ρ=0.05) | 156.62 | 83.70 | 83.70 | Good Generalization |
| **ARS (Sync)** | **Manifold SAM (ρ=0.1)** | 159.13 | **80.94** | **80.94** | **Best Quality** |
| **ARS (Lazy)** | **Shear Force (k=5, α=0.1)** | 158.21 | 82.10 | 82.10 | **Best Balance (~1.5x Speedup)** |

Results show:

1. **ARS (Sync, ρ=0.1)** achieves the best generalization performance (PPL 80.94), proving the effectiveness of manifold-aware perturbation in finding flat minima.
2. **ARS (Lazy)** achieves ~1.5x training speedup with only a marginal PPL cost (1.1 higher than Sync) thanks to the intensity compensation mechanism, making it the optimal choice for practical applications.

### 3.2 Grokking Phenomenon Acceleration

We validated optimizer acceleration effects on the Grokking phenomenon using modular addition tasks, where models need to learn intrinsic patterns of modular arithmetic.

| Optimizer | Fitting Speed | Grokking Moment | Convergence | Final Performance | Status |
| :-------- | :------------ | :-------------- | :---------- | :---------------- | :----- |
| **AdamW** | ~Epoch 140 | **Epoch 228** | Epoch 556 | 100.0% | ✅ Standard Grokking |
| **AdaRMSuon** | **Epoch 28** | **Epoch 54** | **Epoch 300** | 99.9% | 🚀 **Ultra-fast Grokking** |
| **ARS** | Epoch 17 | **Epoch 100** | Epoch 290 | 99.1% | 🚀 **Robust Grokking** |

**Key Finding**: **AdaRMSuon** accelerates the Grokking phenomenon by **4x** compared to the AdamW baseline (Epoch 228 → Epoch 54), demonstrating the critical role of "Energy-Geometry Decoupling" and "Manifold Flatness Constraints" in accelerating model generalization phase transitions.

## 4. Quick Start

### 4.1. Installation

```bash
# Using uv is recommended
uv sync
```

### 4.2. Reproduce Key Experiment

```bash
# 1. Run ARS Sync Mode (Best Quality)
python -m scripts.train --config config/wikitext2_line_mode_ars_rho_0.1.toml

# 2. Run ARS Lazy Mode (Best Balance)
python -m scripts.train --config config/wikitext2_line_mode_ars_rho_0.1_k5_alpha0.1.toml
```

Compare the `Eval Perplexity` in the generated summary files within the `outputs/` directory.

## 5. Framework Structure

The framework is designed for rapid prototyping and clear evaluation.

- **Optimizers**: Reside in [`optimizer/`](optimizer/). See [`optimizer/rmsuon.py`](optimizer/rmsuon.py) and [`optimizer/ars.py`](optimizer/ars.py).
- **Models**: Standard architectures are in [`model/`](model/).
- **Tasks**: Training and evaluation logic is defined in [`task/`](task/). The `line mode` implementation is in [`task/wikitext2_line.py`](task/wikitext2_line.py).
- **Configs**: Experiment configurations are managed via TOML files in [`config/`](config/).
- **Outputs**: All results, logs, and checkpoints are saved to [`outputs/`](outputs/).

## Citation

If you use F3EO-Bench in your research, please cite the underlying theoretical work:

```bibtex
@software{f3eo_bench_2025,
  author = {Rui, L.},
  title = {F3EO-Bench: A Lightweight Framework for Advanced Optimizer Evaluation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/F3EO},
  note = {A research framework for optimizers based on the Energy-Geometry Decoupling principle.}
}
```
