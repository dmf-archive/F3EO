# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | cifar10 |
| Model | resnet18_cifar |
| Optimizer | F3EO |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 20 |
| Batch Size | 256 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy (%) | Valid Accuracy (%) | Learning Rate | Time |
|-----|--|-----|-----|-----|-----|------|
| 1 | 1.7275 | 1.7575 | 35.47 | 40.65 | 0.001000 | 248.51s |
| 2 | 1.3614 | 1.1632 | 50.14 | 57.91 | 0.001000 | 248.32s |
| 3 | 1.1519 | 1.1283 | 58.55 | 60.41 | 0.001000 | 247.23s |
| 4 | 1.0255 | 0.9342 | 63.53 | 66.83 | 0.000999 | 247.28s |
| 5 | 0.9499 | 0.9589 | 66.55 | 67.77 | 0.000999 | 246.96s |
| 6 | 0.8718 | 0.8472 | 69.65 | 70.59 | 0.000998 | 247.01s |
| 7 | 0.8021 | 0.9820 | 72.32 | 66.95 | 0.000998 | 247.90s |
| 8 | 0.7535 | 0.7810 | 74.26 | 72.69 | 0.000997 | 248.62s |
| 9 | 0.7136 | 0.7916 | 75.95 | 74.41 | 0.000996 | 253.85s |
| 10 | 0.6881 | 0.8390 | 76.85 | 71.74 | 0.000995 | 247.92s |
| 11 | 0.6629 | 0.6282 | 77.83 | 79.36 | 0.000994 | 250.93s |
| 12 | 0.6365 | 0.6067 | 78.98 | 79.85 | 0.000993 | 249.28s |
| 13 | 0.6212 | 0.5853 | 79.67 | 80.35 | 0.000991 | 248.25s |
| 14 | 0.6041 | 0.5977 | 80.35 | 79.71 | 0.000990 | 248.60s |
| 15 | 0.5973 | 0.5701 | 80.84 | 81.01 | 0.000988 | 248.49s |
| 16 | 0.5869 | 0.5289 | 81.23 | 82.70 | 0.000986 | 249.40s |
| 17 | 0.5707 | 0.5312 | 81.98 | 82.23 | 0.000984 | 249.27s |
| 18 | 0.5598 | 0.4589 | 82.34 | 85.16 | 0.000982 | 250.02s |
| 19 | 0.5646 | 0.6045 | 82.48 | 80.64 | 0.000980 | 250.58s |
| 20 | 0.5504 | 0.6534 | 82.79 | 79.89 | 0.000978 | 254.71s |

## Performance Summary
- **Best Validation Accuracy (%)**: 85.16
- **Final Validation Accuracy (%)**: 79.89
- **Total Training Time**: 4983.18s
- **Average Epoch Time**: 249.15s

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
