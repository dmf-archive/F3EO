# F3EO-Bench Experiment Report

## Configuration Summary

| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.1 |
| Epochs | 60 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results

| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | PI | Eff. Gamma | Entropy | Time |
|-----|--|-----|-----|-----|-----|----|---|---|------|
| 1 | 24.9645 | 16.7819 | 69494808174.53 | 19421299.16 | 0.000100 | N/A | N/A | N/A | 251.04s |
| 2 | 10.5296 | 9.1951 | 37408.05 | 9848.72 | 0.000100 | N/A | N/A | N/A | 258.66s |
| 3 | 8.3592 | 8.0580 | 4269.32 | 3158.82 | 0.000100 | N/A | N/A | N/A | 257.55s |
| 4 | 7.5433 | 7.4820 | 1888.08 | 1775.73 | 0.000100 | N/A | N/A | N/A | 256.97s |
| 5 | 7.0820 | 7.1357 | 1190.34 | 1255.96 | 0.000100 | N/A | N/A | N/A | 258.31s |
| 6 | 6.7737 | 6.9071 | 874.56 | 999.34 | 0.000100 | N/A | N/A | N/A | 260.27s |
| 7 | 6.5597 | 6.7607 | 706.05 | 863.23 | 0.000100 | N/A | N/A | N/A | 261.78s |
| 8 | 6.3799 | 6.6670 | 589.86 | 786.05 | 0.000100 | N/A | N/A | N/A | 254.44s |
| 9 | 6.2177 | 6.5802 | 501.57 | 720.68 | 0.000100 | N/A | N/A | N/A | 254.43s |
| 10 | 6.0581 | 6.5681 | 427.56 | 712.05 | 0.000100 | N/A | N/A | N/A | 253.89s |
| 11 | 5.9001 | 6.5051 | 365.09 | 668.54 | 0.000100 | N/A | N/A | N/A | 254.13s |
| 12 | 5.7301 | 6.5161 | 308.00 | 675.95 | 0.000100 | N/A | N/A | N/A | 255.59s |

## Performance Summary

- **Best Validation Perplexity**: 668.54
- **Final Validation Perplexity**: 675.95
- **Total Training Time**: 3108.72s
- **Average Epoch Time**: 256.42s

## Configuration Details

```toml
{
  "experiment": {
    "task": "wikitext2",
    "seed": 42,
    "device": "cuda",
    "config_name": "wikitext2_adamw_classic"
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
    "lr": 0.0001,
    "weight_decay": 0.1
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
