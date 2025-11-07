# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | AdamW_PI |
| Learning Rate | 0.0001 |
| Weight Decay | 0.01 |
| Epochs | 60 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Log(PI) | Entropy | Time |
|-----|--|-----|-----|-----|-----|--------|---|------|
| 0 | 25.1405 | 13.1237 | 82866836291.49 | 500656.33 | 0.000100 | N/A | N/A | 303.05s |
| 1 | 10.5618 | 9.3234 | 38631.82 | 11197.13 | 0.000100 | N/A | N/A | 303.29s |
| 2 | 8.4802 | 10.5099 | 4818.54 | 36675.38 | 0.000100 | N/A | N/A | 302.05s |
| 3 | 7.6772 | 7.5349 | 2158.49 | 1872.18 | 0.000100 | N/A | N/A | 300.97s |
| 4 | 7.1371 | 7.1974 | 1257.75 | 1335.94 | 0.000100 | N/A | N/A | 300.35s |
| 5 | 6.8195 | 6.9657 | 915.56 | 1059.62 | 0.000100 | N/A | N/A | 300.14s |
| 6 | 6.5913 | 6.8062 | 728.72 | 903.45 | 0.000100 | N/A | N/A | 300.20s |
| 7 | 6.3958 | 6.7076 | 599.33 | 818.59 | 0.000100 | N/A | N/A | 300.32s |
| 8 | 6.2161 | 6.6237 | 500.74 | 752.71 | 0.000100 | N/A | N/A | 300.48s |
| 9 | 6.0351 | 6.6183 | 417.83 | 748.69 | 0.000100 | N/A | N/A | 300.34s |
| 10 | 5.8501 | 6.5911 | 347.25 | 728.60 | 0.000100 | N/A | N/A | 300.34s |
| 11 | 5.6541 | 6.6249 | 285.45 | 753.66 | 0.000100 | N/A | N/A | 300.26s |
| 12 | 5.4327 | 6.6987 | 228.77 | 811.32 | 0.000100 | N/A | N/A | 299.88s |
| 13 | 5.1911 | 6.7729 | 179.66 | 873.82 | 0.000100 | N/A | N/A | 300.74s |
| 14 | 4.9239 | 7.0105 | 137.54 | 1108.22 | 0.000100 | N/A | N/A | 299.90s |
| 15 | 4.6347 | 7.2838 | 103.00 | 1456.58 | 0.000100 | N/A | N/A | 300.20s |
| 16 | 4.3266 | 7.5839 | 75.69 | 1966.19 | 0.000100 | N/A | N/A | 300.28s |
| 17 | 4.0026 | 7.9053 | 54.74 | 2711.57 | 0.000100 | N/A | N/A | 299.95s |
| 18 | 3.6725 | 8.3930 | 39.35 | 4416.17 | 0.000100 | N/A | N/A | 299.96s |
| 19 | 3.3395 | 8.7525 | 28.21 | 6326.69 | 0.000100 | N/A | N/A | 300.15s |
| 20 | 3.0159 | 9.3234 | 20.41 | 11197.08 | 0.000100 | N/A | N/A | 300.46s |
| 22 | 2.7031 | 9.9873 | 14.93 | 21747.53 | 0.000100 | N/A | N/A | 214.31s |

## Performance Summary
- **Best Validation Perplexity**: 728.60
- **Final Validation Perplexity**: 21747.53
- **Total Training Time**: 217.07s
- **Average Epoch Time**: 296.71s

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
    "weight_decay": 0.01,
    "betas": [
      0.9,
      0.999
    ],
    "gamma": 0.5
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
