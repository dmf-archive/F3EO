# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | AdamW_PI |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0 |
| Epochs | 60 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | PI | Eff. Gamma | Entropy | Time |
|-----|--|-----|-----|-----|-----|----|---|---|------|
| 0 | 24.8216 | 13.0185 | 60241200822.21 | 450677.26 | 0.000100 | 0.000 | N/A | 7.107 | 259.30s |
| 1 | 10.7779 | 9.2934 | 47951.46 | 10866.49 | 0.000100 | 0.000 | N/A | 6.459 | 277.11s |
| 2 | 8.4478 | 8.1287 | 4664.74 | 3390.30 | 0.000100 | 0.000 | N/A | 6.458 | 277.27s |
| 3 | 7.6162 | 7.5611 | 2030.80 | 1921.88 | 0.000100 | 0.000 | N/A | 6.308 | 276.00s |
| 4 | 7.1441 | 7.1997 | 1266.66 | 1339.04 | 0.000100 | 0.000 | N/A | 5.901 | 275.02s |
| 6 | 6.8155 | 6.9420 | 911.87 | 1034.89 | 0.000100 | 0.000 | 0.000 | 6.394 | 274.62s |
| 7 | 6.5763 | 6.8183 | 717.85 | 914.46 | 0.000100 | 0.000 | 0.000 | 6.193 | 276.82s |

## Performance Summary
- **Best Validation Perplexity**: 914.46
- **Final Validation Perplexity**: 914.46
- **Total Training Time**: 557.02s
- **Average Epoch Time**: 273.73s

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
    "name": "AdamW_PI",
    "lr": 0.0001,
    "weight_decay": 0.0,
    "betas": [
      0.9,
      0.999
    ],
    "gamma": 1.0
  },
  "train": {
    "epochs": 60,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
