# F3EO-Bench Experiment Report

## Configuration Summary

| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EO |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0005 |
| Epochs | 30 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results

| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Time |
|-----|--|-----|-----|-----|-----|------|
| 1 | 24.7740 | 13.0200 | 57439894232.54 | 451370.91 | 0.000100 | 893.93s |
| 2 | 10.7133 | 9.2026 | 44949.25 | 9922.64 | 0.000100 | 892.80s |
| 3 | 8.4129 | 8.0420 | 4504.68 | 3108.82 | 0.000100 | 892.78s |
| 4 | 7.6078 | 7.5067 | 2013.76 | 1820.25 | 0.000100 | 893.21s |
| 5 | 7.1630 | 7.2618 | 1290.84 | 1424.75 | 0.000100 | 893.68s |
| 6 | 6.8752 | 6.9924 | 967.99 | 1088.28 | 0.000100 | 893.24s |
| 7 | 6.6503 | 6.8725 | 772.98 | 965.32 | 0.000100 | 893.16s |
| 8 | 6.4566 | 6.7733 | 636.90 | 874.21 | 0.000100 | 893.43s |
| 9 | 6.2756 | 6.7097 | 531.47 | 820.29 | 0.000100 | 893.64s |
| 10 | 6.0961 | 6.6698 | 444.10 | 788.26 | 0.000100 | 893.02s |
| 11 | 5.9059 | 6.6829 | 367.19 | 798.60 | 0.000100 | 893.10s |
| 12 | 5.7020 | 6.7464 | 299.45 | 851.03 | 0.000100 | 893.33s |

## Performance Summary

- **Best Validation Perplexity**: 788.26
- **Final Validation Perplexity**: 851.03
- **Total Training Time**: 10741.67s
- **Average Epoch Time**: 893.28s

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
    "name": "F3EO",
    "lr": 0.0001,
    "weight_decay": 0.0005
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
