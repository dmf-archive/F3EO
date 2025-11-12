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
    "name": "Muon",
    "lr": 0.001,
    "weight_decay": 0.0005,
    "momentum": 0.95
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
| 1 | cifar10 | 1.7482 | 0.001000 | 0.048 | N/A | 1.777 | 1.2692 | 54.60 | 1.26 |
| 2 | cifar10 | 1.1947 | 0.001000 | 0.113 | N/A | 1.244 | 0.9387 | 65.53 | 0.99 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 65.53, cifar10 Loss: 1.26
- **Final Validation Metrics**: cifar10: {"loss": 0.9861368730664253, "accuracy": 65.53}
