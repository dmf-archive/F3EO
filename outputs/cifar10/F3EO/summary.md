# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-----------|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | F3EO |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 2 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Time |
|-------|------------|------------|-------------------|-------------------|---------------|------|
| 1 | 0.7296 | 0.7875 | 74.31 | 74.72 | 0.000998 | 249.37s |
| 2 | 0.6671 | 0.6263 | 76.56 | 78.54 | 0.000998 | 250.59s |

## Performance Summary
- **Best Validation Accuracy (%)**: 78.54
- **Final Validation Accuracy (%)**: 78.54
- **Total Training Time**: 500.89s
- **Average Epoch Time**: 249.98s

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
    "name": "F3EO",
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
