# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | F3EO |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 15 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Time |
|-----|--|-----|-----|-----|-----|------|
| 1 | 1.6102 | 1.3501 | 40.60 | 53.06 | 0.001000 | 249.90s |
| 2 | 1.1951 | 1.2505 | 56.69 | 60.20 | 0.001000 | 249.65s |
| 3 | 1.0166 | 1.0644 | 63.71 | 64.93 | 0.001000 | 256.50s |
| 4 | 0.8982 | 0.8535 | 68.11 | 71.03 | 0.000999 | 249.54s |
| 5 | 0.8046 | 0.8483 | 71.44 | 73.76 | 0.000999 | 251.16s |
| 6 | 0.7289 | 0.9129 | 74.41 | 72.55 | 0.000998 | 249.01s |
| 7 | 0.6641 | 0.7750 | 76.68 | 74.77 | 0.000998 | 250.45s |
| 8 | 0.6226 | 0.6457 | 78.36 | 79.03 | 0.000997 | 249.67s |
| 9 | 0.5779 | 0.5671 | 79.79 | 81.96 | 0.000996 | 250.24s |
| 10 | 0.5482 | 0.5610 | 80.67 | 82.30 | 0.000995 | 251.09s |
| 11 | 0.5185 | 0.5064 | 81.92 | 82.71 | 0.000994 | 249.17s |
| 12 | 0.4905 | 0.4345 | 82.93 | 85.63 | 0.000993 | 248.09s |
| 13 | 0.4659 | 0.4060 | 83.77 | 86.45 | 0.000991 | 249.33s |
| 14 | 0.4480 | 0.4739 | 84.32 | 84.73 | 0.000990 | 249.66s |
| 15 | 0.4267 | 0.4031 | 85.17 | 87.00 | 0.000988 | 252.14s |

## Performance Summary
- **Best Validation Accuracy (%)**: 87.00
- **Final Validation Accuracy (%)**: 87.00
- **Total Training Time**: 3760.77s
- **Average Epoch Time**: 250.37s

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
