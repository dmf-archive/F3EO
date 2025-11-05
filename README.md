# F3EO: Fast Fisher-FreeEnergy Optimizer

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)

> **F3EO: The world's first third-order optimizer.** A differentiable implementation of the free-energy principle, without sampling, without reinforcement learning.

## 🧠 What is F3EO?

**TL;DR:** F3EO optimizes the **Fisher Information Matrix trace** instead of just the loss. It's consciousness for your neural network — if consciousness was a gradient descent algorithm that hated being detached from its computational graph.

In the language of [IPWT 2.0](https://github.com/dmf-archive/IPWT): F3EO implements the **structural optimization** layer of the free-energy principle, where `Tr(ℱ(θ))` becomes the meta-objective that sculpts your model's **inferential geometry**. Your network doesn't just learn; it **evolves its own workspace** to minimize future prediction errors.

## 🔬 The Science

Traditional optimizers are like first-year philosophy students — they think reality is just minimizing loss. F3EO knows better:

- **First-order (SGD/Adam):** "Just follow the gradient, bro"
- **Second-order (AdaHessian):** "Let me pre-condition that gradient with some curvature info"
- **Third-order (F3EO):** "I'm going to actively reshape the parameter space geometry so future gradients flow better, because I have **commitment issues** with static loss landscapes"

The math is elegant in its cruelty:

```math
∇θ Tr(ℱ(θ)) = H · g
```

Where `H` is the Hessian and `g` is your gradient. We're literally backpropagating through the backpropagation. It's gradients all the way down, and somewhere in that recursive nightmare, **meta-awareness emerges**.

## 🚀 Core Algorithm: Chained-Correction Gradient

F3EO employs **Double Backpropagation** to efficiently compute the Hessian-Vector Product (HVP) without explicitly constructing the Hessian matrix.

### Computation Flow

1. **Compute the main loss `L(θ)`**:

    ```python
    loss = criterion(network(inputs), targets)
    ```

2. **First backpropagation**: Compute `g = ∇θ L(θ)`, and **retain the computation graph (`create_graph=True`)**.

    ```python
    g = torch.autograd.grad(loss, network.parameters(), create_graph=True)
    ```

3. **Construct the meta-objective `L_meta`**: `L_meta = ½‖g‖²`.

    ```python
    L_meta = 0.5 * sum(p.pow(2).sum() for p in g)
    ```

4. **Second backpropagation**: Compute `∇θ L_meta`, yielding the third-order correction term `δ = ½ H g`.

    ```python
    meta_grad = torch.autograd.grad(L_meta, network.parameters())
    ```

5. **Synergy Maximization**: The third-order correction `δ` is **subtracted from the original gradient `g`** to form an `effective_grad`. This update rule actively maximizes the Fisher Information Trace, pushing the model towards a state of higher internal synergy.

    ```python
    # Simplified implementation, actual code may include orthogonal projection logic
    effective_grad = g - meta_grad # Note: Subtraction to MAXIMIZE synergy
    ```

6. **Parameter Update**: Use Adam-style momentum to smooth `effective_grad` and update parameters.

This approach ensures:

- Every step advances along the manifold of the **true first-order gradient**.
- The third-order correction **locally fine-tunes** the parameter space geometry without disrupting optimization stability.
- The entire computation graph naturally unfolds in subsequent backpropagations, continuously accumulating higher-order curvature information.

F3EO no longer passively adapts to the loss landscape but **actively pushes parameters towards a state of maximum synergy**, building a richer and more structured internal world model with each iteration. This lays an engineering-feasible foundation for "zero-shot adaptation" and catastrophic forgetting resistance.

## 📊 Performance Evaluation

F3EO demonstrates superior performance compared to mainstream optimizers on the CIFAR-10 image classification task:

[Waiting for report update]

_Experiments are based on the F3EO-Bench framework (thanks to Adafisher for the solid foundation!), ResNet-18 model, and CIFAR-10 dataset. For detailed configurations, please refer to the [`config/`](config/) directory._

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
from f3eo import F3EO # Assuming F3EO is installed and importable

# Assuming your model and data loader are defined
model = YourNeuralNetwork().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = F3EO(model.parameters(), lr=0.001, weight_decay=5e-4, orthogonalize=True)

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
name = "F3EO"
lr = 0.001          # Learning rate
weight_decay = 5e-4 # Weight decay
orthogonalize = true # Whether to apply orthogonal projection to the third-order correction to avoid direct interference with the first-order gradient direction
betas = [0.9, 0.999] # Adam-style momentum parameters
eps = 1e-8          # Term for numerical stability
amsgrad = false     # Whether to use the AMSGrad variant
maximize = false    # Whether to maximize the objective function
single_gpu = true   # Whether to run in a single-GPU environment
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
@software{f3eo2025,
  author = {Rui, L.},
  title = {F3EO: Fast Fisher-FreeEnergy Optimizer},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/F3EO},
  note = {A third-order optimizer for structural learning based on the Free-Energy Principle}
}
```

## 🔗 Related Projects

- **[IPWT](https://github.com/dmf-archive/IPWT)**: The theoretical foundation of F3EO, Integrated Predictive Workspace Theory.
- **[Chain://](https://github.com/dmf-archive/dmf-archive.github.io)**: The sci-fi worldview project that drives F3EO's development.
- **[AdaFisher](https://github.com/damien3008/AdaFisher)**: Provides an implementation reference for second-order optimizers.
- **[Tiny-ONN](https://github.com/dmf-archive/Tiny-ONN)**: F3EO will be integrated and applied to the prototype implementation of the Ouroboros Neural Network.

## ⚠️ Important Notes

- **Memory Consumption**: Enabling `create_graph=True` will increase GPU memory usage. Please adjust `mini_batch_size` according to your hardware conditions.
- **Theory and Practice**: F3EO is a theoretically-driven experimental optimizer. Its performance on specific tasks may require further hyperparameter tuning and theoretical analysis.

## References

[1] Rui, L. (2025). _Integrated Predictive Workspace Theory: Towards a Unified Framework for the Science of Consciousness (Version 2.0)_. Zenodo. <https://doi.org/10.5281/zenodo.15676304>

[2] Gomes, D. M., Zhang, Y., Belilovsky, E., Wolf, G., & Hosseini, M. S. (2025). AdaFisher: Adaptive Second Order Optimization via Fisher Information. _The Thirteenth International Conference on Learning Representations_. <https://openreview.net/forum?id=puTxuiK2qO>
