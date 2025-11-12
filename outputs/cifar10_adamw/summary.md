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
    "name": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.0005
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 200
  },
  "train": {
    "epochs": 2,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|---------------|-----------|
| 1 | cifar10 | 1.6119 | 0.001000 | 0.017 | 0.017 | 1.570 | 2.4914 | 50.29 | 1.48 |
| 2 | cifar10 | 1.1974 | 0.001000 | 0.021 | 0.021 | 1.188 | 2.6785 | 63.72 | 1.07 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 63.72, cifar10 Loss: 1.48
- **Final Validation Metrics**: cifar10: {"loss": 1.0694968864321708, "accuracy": 63.72}
