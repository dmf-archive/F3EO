<p align="center">
  <img src="icon.png" alt="F3EO Icon" width="200"/>
</p>

# F3EO: Fast Fisher Free-Energy Optimizer

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)

> **F3EO: The world's first third-order optimizer.** A differentiable implementation of the free-energy principle, without sampling, without reinforcement learning.

## 🧠 What is F3EO?

F3EO, also known as F3EPI (Fast Fisher Free-Energy Optimizer with Predictive Integrity) is a third-order optimizer that implements the free-energy principle through adaptive modulation of model complexity. Based on [IPWT 2.0](https://github.com/dmf-archive/IPWT), it decomposes the optimization objective into accuracy and complexity components, balancing them through a predictive integrity feedback mechanism.

Unlike traditional optimizers that solely minimize prediction error, F3EPI actively regulates the Fisher Information Matrix trace `Tr(ℱ(θ))` — a measure of model complexity — through a PI-driven adaptive coefficient. This enables the optimizer to automatically adjust its complexity preference based on the current learning state, preventing both underfitting and overfitting.

## 🔬 The Science

F3EPI implements a dual-gradient optimization framework based on the free-energy principle:

- **First-order (SGD/Adam)**: Minimizes prediction error through gradient descent
- **Second-order (AdaHessian)**: Pre-conditions gradients using curvature information
- **Third-order (F3EPI)**: Actively modulates parameter space geometry through `Tr(ℱ(θ))` optimization

The core mathematical insight is that the gradient of Fisher Information Matrix trace equals the Hessian-vector product:

```math
∇θ Tr(ℱ(θ)) = H · g
```

This enables efficient third-order optimization through double backpropagation without explicit Hessian construction.

## 🚀 Core Algorithm: Predictive Integrity-driven Gradient

F3EPI employs **Double Backpropagation** to compute the Hessian-Vector Product (HVP) and implements adaptive complexity modulation through Predictive Integrity (PI) feedback.

### Computation Flow

1. **Compute the main loss `L(θ)`**:

    ```python
    loss = criterion(network(inputs), targets)
    ```

2. **First backpropagation**: Compute `g = ∇θ L(θ)`, and **retain the computation graph**.

    ```python
    g = torch.autograd.grad(loss, network.parameters(), create_graph=True)
    ```

3. **Construct the meta-objective `L_meta`**: `L_meta = ½‖g‖²`.

    ```python
    L_meta = 0.5 * sum(p.pow(2).sum() for p in g)
    ```

4. **Second backpropagation**: Compute `∇θ L_meta`, yielding the third-order correction term `δ_meta = H g`.

    ```python
    meta_grad = torch.autograd.grad(L_meta, network.parameters())
    ```

5. **PI-driven Adaptive Modulation**: Compute Predictive Integrity `PI` and its feedback coefficient `β = tanh(log(PI))`. The effective gradient combines accuracy and complexity objectives:

    ```python
    # β > 0: enhance complexity (underfitting regime)
    # β < 0: suppress complexity (overfitting regime)
    # β ≈ 0: maintain balance
    effective_grad = g + β * meta_grad
    ```

6. **Parameter Update**: Use Adam-style momentum to smooth `effective_grad` and update parameters.

This approach ensures:

- **Accuracy Priority**: The primary optimization direction follows the true first-order gradient.
- **Adaptive Complexity**: The complexity component is dynamically adjusted based on PI feedback.
- **Computational Efficiency**: Third-order information is computed through HVP without explicit Hessian construction.

F3EPI implements a **synergistic balance principle** where the optimizer automatically transitions between complexity maximization and minimization based on the model's current predictive integrity state.

## 📊 Performance Evaluation

F3EPI is evaluated through the F3EO-Bench framework across multiple tasks including CIFAR-10 image classification and WikiText-2 language modeling. The optimizer demonstrates adaptive complexity modulation behavior, automatically adjusting its optimization strategy based on training dynamics.

_Experiments utilize the F3EO-Bench framework with standardized configurations. Detailed experimental setups are available in the [`config/`](config/) directory._

## 🛠️ Usage Guide

### Installation

```bash
# Recommended to use uv for installation to ensure dependency consistency
uv sync
```

Or install via pip:

```bash
pip install f3eo
```

### Example Code

```python
import torch
import torch.nn as nn
from f3eo import F3EPI # Assuming F3EPI is available

# Assuming your model and data loader are defined
model = YourNeuralNetwork().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = F3EPI(model.parameters(), lr=0.001, weight_decay=5e-4, alpha=1.0, gamma=2.0)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Crucial step: Enable graph retention for subsequent HVP calculation
        loss.backward(create_graph=True)

        optimizer.step()

        # Optional: Monitor gradient norm
        if hasattr(optimizer, 'grad_norm'):
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Grad Norm: {optimizer.grad_norm:.4f}")
```

**Important Note**: When calling `loss.backward()`, ensure `create_graph=True` is set to build the computation graph required for higher-order gradients.

## ⚙️ Configuration Options

F3EO supports the following configuration parameters:

```toml
[optimizer]
name = "F3EPI"
lr = 0.001          # Learning rate
weight_decay = 5e-4 # Weight decay
alpha = 1.0         # PI coefficient for accuracy term
gamma = 2.0         # PI coefficient for complexity term
betas = [0.9, 0.999] # Adam-style momentum parameters
```

More configuration examples can be found in the [`config/`](config/) directory.

## 📈 Real-time Monitoring

F3EO integrates rich real-time training metric monitoring, including loss, accuracy, gradient norm, and samples processed per second:

```
Epoch 1/200 | Step 10/196 | Loss: 2.1426 | Acc: 16.99% | Grad: 5.7239 | 91.4it/s
Epoch 1/200 | Step 20/196 | Loss: 1.8996 | Acc: 21.89% | Grad: 2.5840 | 326.3it/s
...
```

## 🔬 Experimental Features & Future Work

- **Zero-shot Adaptation**: F3EO's structural optimization properties are expected to enhance the model's zero-shot adaptation capabilities.
- **Catastrophic Forgetting Resistance**: By actively shaping the parameter space geometry, F3EO is expected to naturally maintain important representations, thereby strengthening the model's continual learning ability.

## 📝 Citation

If you use F3EO in your research, please cite:

```bibtex
@software{f3epi2025,
  author = {Rui, L.},
  title = {F3EPI: Fast Fisher Free-Energy Optimizer with Predictive Integrity},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/F3EO},
  note = {A third-order optimizer with adaptive complexity modulation based on the Free-Energy Principle}
}
```

## 🔗 Related Projects

- **[IPWT](https://github.com/dmf-archive/IPWT)**: The theoretical foundation of F3EPI, Integrated Predictive Workspace Theory.
- **[Chain://](https://github.com/dmf-archive/dmf-archive.github.io)**: The sci-fi worldview project that drives F3EPI's development.
- **[AdaFisher](https://github.com/damien3008/AdaFisher)**: Provides an implementation reference for second-order optimizers.
- **[Tiny-ONN](https://github.com/dmf-archive/Tiny-ONN)**: F3EPI will be integrated and applied to the prototype implementation of the Ouroboros Neural Network.

## ⚠️ Important Notes

- **Memory Consumption**: Enabling `create_graph=True` will increase GPU memory usage. Please adjust batch size according to your hardware conditions.
- **Theory and Practice**: F3EPI is a theoretically-driven experimental optimizer. Its performance on specific tasks may require further hyperparameter tuning and theoretical analysis.

## References

[1] Rui, L. (2025). _Integrated Predictive Workspace Theory: Towards a Unified Framework for the Science of Consciousness (Version 2.0)_. Zenodo. <https://doi.org/10.5281/zenodo.15676304>

[2] Gomes, D. M., Zhang, Y., Belilovsky, E., Wolf, G., & Hosseini, M. S. (2025). AdaFisher: Adaptive Second Order Optimization via Fisher Information. _The Thirteenth International Conference on Learning Representations_. <https://openreview.net/forum?id=puTxuiK2qO>
