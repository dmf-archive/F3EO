# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EO |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Epochs | 10 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Time |
|-----|--|-----|-----|-----|-----|------|
| 0 | 14.6909 | 13.1170 | 2399751.87 | 497329.36 | 0.001000 | 899.86s |
| 1 | 8.1701 | 7.9934 | 3533.52 | 2961.29 | 0.001000 | 900.45s |
| 3 | 8.2040 | 7.9197 | 3655.65 | 2750.87 | 0.001000 | 898.72s |

## Performance Summary
- **Best Validation Perplexity**: 2750.87
- **Final Validation Perplexity**: 2750.87
- **Total Training Time**: 900.72s
- **Average Epoch Time**: 899.68s

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
    "lr": 0.001,
    "weight_decay": 0.0005
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
