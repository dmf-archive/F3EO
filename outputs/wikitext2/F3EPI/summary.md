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
| 0 | 25.1444 | 13.2034 | 83189930115.77 | 542225.78 | 0.000100 | -0.327 | 783.93s |
| 1 | 10.6344 | 9.2888 | 41538.64 | 10816.40 | 0.000100 | 7.913 | 789.36s |
| 2 | 8.4909 | 8.1251 | 4870.20 | 3378.11 | 0.000100 | 4.714 | 833.72s |
| 3 | 7.6552 | 7.5683 | 2111.50 | 1935.81 | 0.000100 | 11.594 | 836.71s |
| 4 | 7.2000 | 7.2116 | 1339.48 | 1355.01 | 0.000100 | 5.890 | 831.38s |
| 5 | 6.9013 | 7.0230 | 993.61 | 1122.15 | 0.000100 | 1.922 | 830.43s |
| 6 | 6.6751 | 6.9137 | 792.46 | 1005.99 | 0.000100 | 4.301 | 832.08s |
| 7 | 6.4861 | 6.7791 | 655.94 | 879.25 | 0.000100 | 1.496 | 842.48s |
| 8 | 6.3064 | 6.7403 | 548.09 | 845.81 | 0.000100 | 0.865 | 840.84s |
| 9 | 6.1292 | 6.6790 | 459.08 | 795.55 | 0.000100 | 1.834 | 838.91s |
| 10 | 5.9435 | 6.7039 | 381.26 | 815.58 | 0.000100 | 0.231 | 847.04s |
| 11 | 5.7467 | 6.7083 | 313.14 | 819.15 | 0.000100 | -0.148 | 842.16s |
| 12 | 5.5298 | 6.7575 | 252.09 | 860.52 | 0.000100 | -0.391 | 828.77s |
| 13 | 5.2913 | 6.9256 | 198.60 | 1018.01 | 0.000100 | 0.523 | 832.26s |
| 15 | 5.0274 | 7.0574 | 152.54 | 1161.47 | 0.000100 | 2.415 | 923.29s |
| 16 | 4.7385 | 7.2415 | 114.26 | 1396.20 | 0.000100 | 2.849 | 954.57s |
| 17 | 4.4310 | 7.5829 | 84.02 | 1964.40 | 0.000100 | 3.273 | 954.68s |
| 18 | 4.1095 | 7.9568 | 60.91 | 2854.82 | 0.000100 | 4.214 | 954.54s |
| 19 | 3.7818 | 8.1586 | 43.90 | 3493.26 | 0.000100 | 7.829 | 955.04s |
| 20 | 3.4556 | 8.7373 | 31.68 | 6230.91 | 0.000100 | 5.147 | 953.72s |

## Performance Summary

- **Best Validation Perplexity**: 795.55
- **Final Validation Perplexity**: 6230.91
- **Total Training Time**: 5713.54s
- **Average Epoch Time**: 865.30s

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
    "gamma": 1.0
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
