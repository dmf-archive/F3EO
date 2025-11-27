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
    "name": "RMSuon",
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.0005,
    "ns_steps": 5
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
| 1 | cifar10 | 1.3103 | 0.001000 | 0.007 | N/A | 0.131 | 4.8229 | 93.76 | 1739.4 | 70.08 | 0.87 |
| 2 | cifar10 | 0.8215 | 0.001000 | 0.001 | N/A | 0.084 | 7.1410 | 89.15 | 1739.3 | 79.31 | 0.60 |
| 3 | cifar10 | 0.6612 | 0.001000 | 0.000 | N/A | 0.067 | 9.4042 | 86.15 | 1739.3 | 80.07 | 0.59 |
| 4 | cifar10 | 0.5670 | 0.000999 | 0.000 | N/A | 0.058 | 11.4582 | 85.25 | 1739.3 | 82.95 | 0.53 |
| 5 | cifar10 | 0.5001 | 0.000999 | 0.000 | N/A | 0.051 | 13.3960 | 87.83 | 1739.3 | 84.40 | 0.47 |
| 6 | cifar10 | 0.4566 | 0.000998 | 0.000 | N/A | 0.046 | 15.2742 | 84.26 | 1739.3 | 86.10 | 0.41 |
| 7 | cifar10 | 0.4193 | 0.000998 | 0.000 | N/A | 0.043 | 17.0511 | 82.69 | 1739.3 | 86.86 | 0.42 |
| 8 | cifar10 | 0.3883 | 0.000997 | 0.000 | N/A | 0.039 | 18.7761 | 84.11 | 1739.3 | 88.76 | 0.35 |
| 9 | cifar10 | 0.3567 | 0.000996 | 0.000 | N/A | 0.036 | 20.3845 | 86.76 | 1739.3 | 88.35 | 0.37 |
| 10 | cifar10 | 0.3398 | 0.000995 | 0.000 | N/A | 0.034 | 21.9165 | 87.30 | 1739.3 | 89.09 | 0.34 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 89.09, cifar10 Loss: 0.87
- **Final Validation Metrics**: cifar10: {"loss": 0.33578573614358903, "accuracy": 89.09}
