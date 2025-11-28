# F3EO-Bench: A Lightweight Framework for Advanced Optimizer Evaluation

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/F3EO)

> **F3EO-Bench** is a research framework for prototyping and evaluating advanced neural network optimizers, centered on the principle of **Energy-Geometry Decoupling**.

## 1. Core Principle: Energy-Geometry Decoupling

Modern deep learning optimization faces a trade-off between statistical adaptivity (how fast to learn) and structural stability (what to learn). F3EO-Bench facilitates research into optimizers that decouple these concerns. Our flagship optimizer, **RMSuon**, operationalizes this principle:

1. **Statistical Operator (Energy)**: Uses the scalar norm of AdamW's second-moment-corrected momentum (`||m̂ / √v̂||`) to determine the update magnitude. This acts as a computationally cheap proxy for the rate of free-energy descent.
2. **Structural Operator (Geometry)**: Employs Muon's Newton-Schulz iteration to find an orthogonal update direction (`O_t`). This constrains the update to a manifold that minimizes interference with the existing network structure, aligning with the Minimum Description Length (MDL) principle.

The final update is a composition of these two operators: `g_update = scale * O_t`, where `scale` is derived from the energy. This decouples "how fast" from "where to go," providing a robust and efficient path towards the minimum.

For a deeper theoretical dive, see [`.roo/rules/RMSuon.md`](.roo/rules/RMSuon.md).

## 2. Key Experimental Result: RMSuon vs. Muon

To validate this approach, we developed a `line mode` data preprocessor for Wikitext-2 that preserves sentence boundaries, maximizing the contextual integrity of the input. When tested with this high-quality data stream, RMSuon demonstrates a significant performance advantage over its predecessor, Muon.

| Optimizer | Data Mode | Epoch 1 PPL | Best PPL |
| :--- | :--- | :--- | :--- |
| Muon | `line mode` | 233.30 | 161.09 |
| **RMSuon** | `line mode` | **146.52** | **99.07** |

RMSuon converges dramatically faster, reaching a sub-100 perplexity that Muon fails to achieve. This result highlights the synergistic effect of a high-integration optimizer combined with a high-integration data pipeline.

## 3. Quick Start

### 3.1. Installation

```bash
# Using uv is recommended
uv sync
```

### 3.2. Reproduce Key Experiment

```bash
# 1. Run the Muon baseline
python -m scripts.train --config config/wikitext2_line_rope_muon.toml

# 2. Run the RMSuon experiment
python -m scripts.train --config config/wikitext2_line_rope_rmsuon.toml
```

Compare the `Eval Perplexity` in the generated summary files within the `outputs/` directory.

## 4. Framework Structure

The framework is designed for rapid prototyping and clear evaluation.

- **Optimizers**: Reside in [`optimizer/`](optimizer/). See [`optimizer/rmsuon.py`](optimizer/rmsuon.py).
- **Models**: Standard architectures are in [`model/`](model/).
- **Tasks**: Training and evaluation logic is defined in [`task/`](task/). The `line mode` implementation is in [`task/wikitext2_line.py`](task/wikitext2_line.py).
- **Configs**: Experiment configurations are managed via TOML files in [`config/`](config/).
- **Outputs**: All results, logs, and checkpoints are saved to [`outputs/`](outputs/).

## 5. Research Chronicle

This repository also serves as an archive of our research journey, including theoretical dead-ends. The progression from flawed third-order methods (`F3E` family) to the more robust operator composition paradigm is documented in our internal rules.

- **Failures Archive**: [`.roo/rules/failure-archive.md`](.roo/rules/failure-archive.md)
- **Hadron (KFAC+Muon)**: [`.roo/rules/Hadron.md`](.roo/rules/Hadron.md)

---

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
