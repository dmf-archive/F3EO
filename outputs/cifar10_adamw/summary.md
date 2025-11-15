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
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.6505 | 0.001000 | 0.072 | N/A | 0.161 | 2.4674 | 86.10 | 1741.0 | 44.36 | 1.66 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 44.36, cifar10 Loss: 1.66
- **Final Validation Metrics**: cifar10: {"loss": 1.6615782171487807, "accuracy": 44.36}
