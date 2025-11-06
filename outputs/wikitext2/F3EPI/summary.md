# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EPI |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0005 |
| Epochs | 30 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Log(PI) | Time |
|-----|--|-----|-----|-----|-----|--------|------|
| 1 | 27.0376 | 14.5479 | 552434587337.54 | 2079955.81 | 0.000100 | 4.864 | 959.47s |
| 2 | 11.4237 | 10.0314 | 91461.77 | 22729.21 | 0.000100 | 16.938 | 953.06s |

## Performance Summary
- **Best Validation Perplexity**: 22729.21
- **Final Validation Perplexity**: 22729.21
- **Total Training Time**: 1917.43s
- **Average Epoch Time**: 956.26s

## Configuration Details
```toml
{
  "experiment": {
    "task": "wikitext2",
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "nano_gpt",
    "vocabulary_size": 40479,
    "embedding_size": 768,
    "sequence_length": 256,
    "num_heads": 12,
    "num_layers": 4
  },
  "data": {
    "batch_size": 8,
    "num_workers": 4,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "F3EPI",
    "lr": 0.0001,
    "weight_decay": 0.0005,
    "alpha": 0.5,
    "gamma": 1.0,
    "betas": [
      0.5,
      0.999
    ]
  },
  "train": {
    "epochs": 30,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
