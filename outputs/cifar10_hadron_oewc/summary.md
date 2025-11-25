# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda:0"
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 128,
    "num_workers": 0,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "HadronOEWC",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "lifelong_decay": 0.999,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.0005,
    "ewc_lambda": 1.0,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95
  },
  "scheduler": {
    "name": "multistep",
    "milestones": [
      25,
      40
    ],
    "gamma": 0.1
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
| 1 | cifar10 | 1.5849 | 0.001000 | 0.000 | N/A | 0.161 | 60.2432 | 199.20 | 2380.2 | 60.54 | 1.11 |
| 2 | cifar10 | 1.0497 | 0.001000 | 0.000 | N/A | 0.109 | 60.0551 | 198.57 | 2380.3 | 73.12 | 0.76 |
| 3 | cifar10 | 0.8232 | 0.001000 | 0.000 | N/A | 0.086 | 59.7908 | 198.15 | 2380.7 | 77.47 | 0.63 |
| 4 | cifar10 | 0.6997 | 0.001000 | 0.000 | N/A | 0.073 | 59.6713 | 197.49 | 2382.3 | 80.82 | 0.56 |
| 5 | cifar10 | 0.6124 | 0.001000 | 0.000 | N/A | 0.064 | 59.6284 | 198.60 | 2381.0 | 83.78 | 0.48 |
| 6 | cifar10 | 0.5450 | 0.001000 | 0.000 | N/A | 0.057 | 59.6190 | 198.70 | 2380.2 | 85.52 | 0.43 |
| 7 | cifar10 | 0.4972 | 0.001000 | 0.000 | N/A | 0.052 | 59.6194 | 197.15 | 2381.3 | 86.17 | 0.40 |
| 8 | cifar10 | 0.4565 | 0.001000 | 0.000 | N/A | 0.048 | 59.6571 | 198.20 | 2381.2 | 87.11 | 0.37 |
| 9 | cifar10 | 0.4210 | 0.001000 | 0.000 | N/A | 0.045 | 59.6824 | 230.56 | 2381.9 | 87.86 | 0.36 |
| 10 | cifar10 | 0.3892 | 0.001000 | 0.000 | N/A | 0.042 | 59.6825 | 290.94 | 2381.6 | 89.01 | 0.32 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 89.01, cifar10 Loss: 1.11
- **Final Validation Metrics**: cifar10: {"loss": 0.32331755889367453, "accuracy": 89.01}
