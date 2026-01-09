# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 256,
    "num_workers": 0,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "ARS",
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.0005,
    "ns_steps": 5,
    "rho": 0.1,
    "k": 1,
    "alpha": 0.7
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 200
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 10
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.5578 | 0.001000 | 0.089 | N/A | 0.164 | 2.2577 | 104.53 | 1829.8 | 59.69 | 1.10 |
| 2 | cifar10 | 0.9788 | 0.001000 | 0.239 | N/A | 0.109 | 1.3218 | 110.15 | 1916.8 | 76.35 | 0.68 |
| 3 | cifar10 | 0.7292 | 0.001000 | 0.326 | N/A | 0.082 | 1.0395 | 112.50 | 1917.7 | 80.90 | 0.56 |
| 4 | cifar10 | 0.5979 | 0.000999 | 0.406 | N/A | 0.069 | 0.8334 | 114.71 | 2004.9 | 82.15 | 0.53 |
| 5 | cifar10 | 0.5131 | 0.000999 | 0.461 | N/A | 0.060 | 0.7149 | 112.69 | 2091.4 | 85.64 | 0.42 |
| 6 | cifar10 | 0.4637 | 0.000998 | 0.502 | N/A | 0.055 | 0.6344 | 110.49 | 2003.5 | 86.76 | 0.38 |
| 7 | cifar10 | 0.4172 | 0.000998 | 0.532 | N/A | 0.050 | 0.5821 | 110.39 | 2091.7 | 88.16 | 0.35 |
| 8 | cifar10 | 0.3797 | 0.000997 | 0.558 | N/A | 0.046 | 0.5375 | 110.22 | 2005.3 | 89.77 | 0.30 |
| 9 | cifar10 | 0.3448 | 0.000996 | 0.592 | N/A | 0.042 | 0.4814 | 108.83 | 2090.8 | 89.97 | 0.29 |
| 10 | cifar10 | 0.3237 | 0.000995 | 0.600 | N/A | 0.040 | 0.4704 | 109.51 | 2003.3 | 89.43 | 0.30 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 89.97, cifar10 Loss: 1.10
- **Final Validation Metrics**: cifar10: {"loss": 0.303416782990098, "accuracy": 89.43}
