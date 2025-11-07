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
| 1 | 26.7512 | 13.7601 | 414873794400.83 | 946112.93 | 0.000100 | 55.974 | 938.34s |
| 2 | 11.2039 | 9.4673 | 73415.40 | 12929.83 | 0.000100 | 26.015 | 940.87s |
| 3 | 8.8552 | 8.5669 | 7010.56 | 5254.77 | 0.000100 | 76.714 | 944.04s |
| 4 | 7.9265 | 7.5664 | 2769.75 | 1932.22 | 0.000100 | 12.495 | 935.30s |

## Performance Summary
- **Best Validation Perplexity**: 1932.22
- **Final Validation Perplexity**: 1932.22
- **Total Training Time**: 3768.91s
- **Average Epoch Time**: 939.64s

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
