# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | F3EWD |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 50 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Log(PI) | Entropy | Time |
|-----|--|-----|-----|-----|-----|--------|---|------|
| 0 | 1.6081 | 1.5153 | 40.62 | 49.66 | 0.001000 | 0.513 | 1.350 | 275.02s |
| 1 | 1.2015 | 1.0802 | 56.56 | 62.92 | 0.001000 | 0.618 | 1.156 | 277.74s |
| 2 | 1.0179 | 1.0549 | 63.59 | 64.67 | 0.001000 | 0.767 | 0.957 | 273.37s |
| 4 | 0.9034 | 0.8317 | 67.74 | 71.28 | 0.001000 | 0.870 | 0.957 | 278.52s |

## Performance Summary
- **Best Validation Accuracy (%)**: 71.28
- **Final Validation Accuracy (%)**: 71.28
- **Total Training Time**: 278.96s
- **Average Epoch Time**: 276.16s

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
    "name": "F3EWD",
    "lr": 0.001,
    "weight_decay": 0.0005,
    "betas": [
      0.9,
      0.999
    ],
    "gamma": 0.5,
    "meta_grad_clip_norm": 1.0,
    "orthogonalize": true
  },
  "train": {
    "epochs": 50,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```
