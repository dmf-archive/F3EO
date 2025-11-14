# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "mnist_cl",
      "fashion_cl"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "resnet18_mnist",
    "num_classes": 10
  },
  "data": {
    "batch_size": 64,
    "num_workers": 2
  },
  "optimizer": {
    "name": "FOG",
    "lr": 0.02,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.01,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95
  },
  "train": {
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 1
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Fashion_accuracy | Eval Fashion_loss | Eval Learning_shock | Eval Loss | Eval Mnist_accuracy | Eval Mnist_forget_accuracy | Eval Mnist_loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------------------|-------------------|---------------------|-----------|---------------------|----------------------------|-----------------|
| 1 | fashion_cl | 0.4313 | 0.020000 | 0.000 | N/A | 0.035 | 23.8565 | 134.14 | 151.8 | N/A | 89.62 | 0.28 | 5.33 | N/A | N/A | 13.22 | N/A |
| 2 | fashion_cl | 0.3702 | 0.020000 | 0.000 | N/A | 0.029 | 27.2394 | 137.88 | 151.8 | N/A | 91.09 | 0.24 | N/A | N/A | N/A | 12.15 | N/A |
| 3 | fashion_cl | 0.2788 | 0.020000 | 0.000 | N/A | 0.025 | 28.6799 | 143.34 | 151.8 | N/A | 91.72 | 0.23 | N/A | N/A | N/A | 34.67 | N/A |
| 1 | mnist_cl | 0.1436 | 0.020000 | 0.000 | N/A | 0.017 | 23.3743 | 128.93 | 152.2 | 98.64 | N/A | N/A | N/A | 0.04 | N/A | N/A | N/A |
| 2 | mnist_cl | 0.1221 | 0.020000 | 0.000 | N/A | 0.011 | 28.4024 | 126.04 | 151.8 | 98.90 | N/A | N/A | N/A | 0.04 | N/A | N/A | N/A |
| 3 | mnist_cl | 0.0623 | 0.020000 | 0.000 | N/A | 0.007 | 29.5749 | 134.83 | 151.8 | 99.13 | N/A | N/A | N/A | 0.03 | N/A | N/A | N/A |
| 4 | mnist_cl | 0.0940 | 0.020000 | 0.000 | N/A | 0.012 | 29.8277 | 131.80 | 152.2 | N/A | 17.97 | 4.11 | N/A | N/A | 99.15 | N/A | 0.02 |

## Performance Summary
- **Best Validation Metrics**: fashion_cl Fashion_accuracy: 91.72, fashion_cl Fashion_loss: 0.28, fashion_cl Learning_shock: 5.33, fashion_cl Mnist_forget_accuracy: 34.67, mnist_cl Accuracy: 99.13, mnist_cl Fashion_accuracy: 17.97, mnist_cl Fashion_loss: 4.11, mnist_cl Loss: 0.04, mnist_cl Mnist_accuracy: 99.15, mnist_cl Mnist_loss: 0.02
- **Final Validation Metrics**: fashion_cl: {"fashion_accuracy": 91.72, "fashion_loss": 0.2290931135938046, "mnist_forget_accuracy": 34.67}, mnist_cl: {"mnist_accuracy": 99.15, "mnist_loss": 0.022565432663103804, "fashion_accuracy": 17.97, "fashion_loss": 4.105186264985686, "learning_shock": null}
