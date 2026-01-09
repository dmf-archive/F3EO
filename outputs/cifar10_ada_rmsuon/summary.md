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
    "name": "AdaRMSuon",
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
| 1 | cifar10 | 1.3242 | 0.001000 | 0.110 | N/A | 0.129 | 2.0741 | 67.93 | 1742.1 | 69.08 | 0.97 |
| 2 | cifar10 | 0.8178 | 0.001000 | 0.161 | N/A | 0.081 | 1.7437 | 67.03 | 1743.1 | 79.79 | 0.60 |
| 3 | cifar10 | 0.6469 | 0.001000 | 0.219 | N/A | 0.064 | 1.4570 | 68.96 | 1743.1 | 80.61 | 0.58 |
| 4 | cifar10 | 0.5557 | 0.000999 | 0.261 | N/A | 0.055 | 1.2879 | 69.95 | 1743.1 | 83.58 | 0.50 |
| 5 | cifar10 | 0.4905 | 0.000999 | 0.285 | N/A | 0.048 | 1.2079 | 71.42 | 1743.1 | 85.77 | 0.43 |
| 6 | cifar10 | 0.4467 | 0.000998 | 0.312 | N/A | 0.044 | 1.1197 | 72.70 | 1743.1 | 88.09 | 0.36 |
| 7 | cifar10 | 0.4092 | 0.000998 | 0.342 | N/A | 0.040 | 1.0337 | 72.99 | 1743.1 | 86.01 | 0.46 |
| 8 | cifar10 | 0.3849 | 0.000997 | 0.355 | N/A | 0.038 | 0.9979 | 72.95 | 1743.1 | 88.66 | 0.34 |
| 9 | cifar10 | 0.3536 | 0.000996 | 0.380 | N/A | 0.034 | 0.9328 | 72.65 | 1743.1 | 89.09 | 0.35 |
| 10 | cifar10 | 0.3305 | 0.000995 | 0.406 | N/A | 0.032 | 0.8693 | 73.44 | 1743.1 | 88.63 | 0.35 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 89.09, cifar10 Loss: 0.97
- **Final Validation Metrics**: cifar10: {"loss": 0.35476427972316743, "accuracy": 88.63}
