# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 1 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Time |
|-----|--|-----|-----|-----|-----|------|
| 1 | 1.1979 | 1.0592 | 56.46 | 63.61 | 0.001000 | 147.17s |

## Performance Summary
- **Best Validation Accuracy (%)**: 63.61
- **Final Validation Accuracy (%)**: 63.61
- **Total Training Time**: 147.18s
- **Average Epoch Time**: 147.17s

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
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0005
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 200
  },
  "train": {
    "epochs": 200,
    "log_every": 10,
    "ckpt_every": 50
  },
  "early_stop": {
    "patience": 10,
    "threshold": 0.3
  }
}
```
