# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | AdamW |
| Learning Rate | 1e-05 |
| Weight Decay | 0.1 |
| Epochs | 60 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Log(PI) | Time |
|-----|--|-----|-----|-----|-----|--------|------|
| 1 | 51.4875 | 31.2962 | 22947109221026064498688.00 | 39062142070683.29 | 0.000010 | N/A | 310.98s |
| 2 | 27.6824 | 26.0345 | 1052712603224.26 | 202602676084.84 | 0.000010 | N/A | 314.58s |
| 3 | 23.3952 | 22.2885 | 14467359326.12 | 4783563015.56 | 0.000010 | N/A | 312.60s |
| 4 | 20.2028 | 19.3733 | 594230469.25 | 259256249.48 | 0.000010 | N/A | 313.33s |
| 5 | 17.7113 | 17.1553 | 49193804.01 | 28213548.40 | 0.000010 | N/A | 311.59s |
| 6 | 15.8020 | 15.3884 | 7289928.83 | 4820741.27 | 0.000010 | N/A | 313.15s |
| 7 | 14.3269 | 14.0276 | 1667590.70 | 1236312.75 | 0.000010 | N/A | 315.05s |
| 8 | 13.1610 | 12.9684 | 519720.87 | 428634.30 | 0.000010 | N/A | 310.18s |
| 9 | 12.2350 | 12.1042 | 205869.40 | 180620.26 | 0.000010 | N/A | 310.23s |
| 10 | 11.4837 | 11.4187 | 97117.12 | 91006.09 | 0.000010 | N/A | 309.28s |
| 11 | 10.8627 | 10.8455 | 52193.98 | 51300.36 | 0.000010 | N/A | 309.43s |

## Performance Summary
- **Best Validation Perplexity**: 51300.36
- **Final Validation Perplexity**: 51300.36
- **Total Training Time**: 3461.11s
- **Average Epoch Time**: 311.85s

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
    "name": "AdamW",
    "lr": 1e-05,
    "weight_decay": 0.1,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "amsgrad": false
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
