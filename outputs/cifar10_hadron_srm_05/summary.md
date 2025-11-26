# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda:0"
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 128,
    "num_workers": 0,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "Hadron",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.0005,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95,
    "srm_gamma": 0.5
  },
  "scheduler": {
    "name": "multistep",
    "milestones": [
      25,
      40
    ],
    "gamma": 0.1
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
| 1 | cifar10 | 1.7457 | 0.001000 | 0.000 | N/A | 0.182 | 60.9337 | 154.92 | 1660.1 | 51.28 | 1.34 |
| 2 | cifar10 | 1.3154 | 0.001000 | 0.000 | N/A | 0.139 | 61.2067 | 155.34 | 1659.2 | 60.68 | 1.12 |
| 3 | cifar10 | 1.1239 | 0.001000 | 0.000 | N/A | 0.120 | 60.9547 | 156.04 | 1660.4 | 67.04 | 0.94 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 67.04, cifar10 Loss: 1.34
- **Final Validation Metrics**: cifar10: {"loss": 0.9389808162858214, "accuracy": 67.04}
