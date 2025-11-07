# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | AdamW_PI |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 50 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Log(PI) | Entropy | Time |
|-----|--|-----|-----|-----|-----|--------|---|------|
| 1 | 1.6326 | 1.6186 | 39.45 | 46.47 | 0.001000 | N/A | N/A | 105.25s |
| 2 | 1.2146 | 1.0298 | 55.93 | 63.81 | 0.001000 | N/A | N/A | 108.41s |
| 3 | 1.0265 | 1.0880 | 63.07 | 63.15 | 0.001000 | N/A | N/A | 105.27s |
| 4 | 0.9118 | 0.7987 | 67.42 | 72.24 | 0.001000 | N/A | N/A | 110.63s |
| 5 | 0.8116 | 1.1366 | 71.24 | 67.08 | 0.001000 | N/A | N/A | 108.30s |
| 6 | 0.7369 | 0.7668 | 73.93 | 75.22 | 0.001000 | N/A | N/A | 112.17s |
| 7 | 0.6747 | 0.9011 | 76.39 | 71.53 | 0.001000 | N/A | N/A | 107.82s |

## Performance Summary
- **Best Validation Accuracy (%)**: 75.22
- **Final Validation Accuracy (%)**: 71.53
- **Total Training Time**: 761.12s
- **Average Epoch Time**: 108.26s

## Configuration Details
```toml
{
  "experiment": {
    "task": "cifar10",
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 256,
    "num_workers": 5,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "AdamW_PI",
    "lr": 0.001,
    "weight_decay": 0.0005,
    "betas": [
      0.9,
      0.999
    ],
    "gamma": 0.5
  },
  "train": {
    "epochs": 50,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```
