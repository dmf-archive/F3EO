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
| 1 | 26.9416 | 14.7929 | 501847505746.51 | 2657405.85 | 0.000100 | 51.134 | 945.69s |
| 2 | 11.3821 | 9.7779 | 87738.24 | 17640.05 | 0.000100 | 27.271 | 944.95s |
| 3 | 8.9361 | 8.4930 | 7601.32 | 4880.60 | 0.000100 | 17.126 | 945.56s |
| 4 | 7.9895 | 7.6362 | 2949.96 | 2071.80 | 0.000100 | 22.850 | 946.96s |
| 5 | 7.4832 | 8.0276 | 1777.92 | 3064.28 | 0.000100 | 49.616 | 946.61s |
| 6 | 7.1329 | 7.1221 | 1252.50 | 1239.04 | 0.000100 | 25.520 | 945.20s |
| 7 | 6.8550 | 7.1080 | 948.58 | 1221.64 | 0.000100 | 16.035 | 946.30s |
| 8 | 6.6280 | 6.8762 | 755.94 | 968.90 | 0.000100 | 20.416 | 946.54s |
| 9 | 6.4275 | 6.7199 | 618.60 | 828.76 | 0.000100 | 14.992 | 945.72s |
| 10 | 6.2151 | 6.7949 | 500.23 | 893.24 | 0.000100 | 20.296 | 946.46s |
| 11 | 6.0248 | 6.6987 | 413.56 | 811.35 | 0.000100 | 18.246 | 947.10s |
| 12 | 5.8111 | 7.0238 | 333.99 | 1123.06 | 0.000100 | 13.511 | 945.90s |
| 13 | 5.5884 | 6.7964 | 267.31 | 894.62 | 0.000100 | 10.590 | 945.96s |
| 14 | 5.3376 | 6.9430 | 208.01 | 1035.84 | 0.000100 | 10.054 | 945.89s |
| 15 | 5.0742 | 7.0278 | 159.84 | 1127.60 | 0.000100 | 10.053 | 946.49s |
| 16 | 4.7802 | 7.2369 | 119.12 | 1389.84 | 0.000100 | 13.713 | 945.85s |
| 17 | 4.4631 | 7.7190 | 86.76 | 2250.69 | 0.000100 | 10.053 | 945.34s |
| 18 | 4.1375 | 7.9003 | 62.65 | 2698.18 | 0.000100 | 14.565 | 945.98s |
| 19 | 3.8000 | 8.1322 | 44.70 | 3402.23 | 0.000100 | 12.236 | 16055.67s |

## Performance Summary
- **Best Validation Perplexity**: 811.35
- **Final Validation Perplexity**: 3402.23
- **Total Training Time**: 33134.17s
- **Average Epoch Time**: 1741.27s

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
    "alpha": 1.0,
    "gamma": 2.0,
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
