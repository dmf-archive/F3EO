# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "nano_gpt",
    "vocabulary_size": 40479,
    "embedding_size": 768,
    "sequence_length": 255,
    "num_heads": 12,
    "num_layers": 4
  },
  "data": {
    "batch_size": 8,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "RMSuon",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2 | 14.9802 | 0.001000 | 0.000 | N/A | 5.653 | 34.9703 | 692.78 | 2984.5 | 8.29 | 3991.17 |
| 2 | wikitext2 | 7.7203 | 0.001000 | 0.000 | N/A | 5.917 | 13.0865 | 708.75 | 2984.5 | 7.60 | 1996.02 |
| 3 | wikitext2 | 7.4891 | 0.001000 | 0.000 | N/A | 6.135 | 9.8468 | 847.82 | 2984.5 | 7.43 | 1679.24 |
| 4 | wikitext2 | 7.2186 | 0.001000 | 0.000 | N/A | 6.374 | 8.1247 | 739.80 | 2984.5 | 7.48 | 1772.73 |
| 5 | wikitext2 | 7.0947 | 0.001000 | 0.000 | N/A | 6.560 | 8.5076 | 680.66 | 2984.5 | 7.35 | 1550.31 |
| 6 | wikitext2 | 7.0640 | 0.001000 | 0.000 | N/A | 6.745 | 13.9720 | 682.54 | 2984.5 | 7.16 | 1287.14 |
| 7 | wikitext2 | 7.0667 | 0.001000 | 0.000 | N/A | 6.873 | 17.6851 | 685.66 | 2984.5 | 7.27 | 1438.78 |
| 8 | wikitext2 | 7.1174 | 0.001000 | 0.000 | N/A | 6.980 | 22.7595 | 688.30 | 2984.5 | 7.24 | 1391.31 |
| 9 | wikitext2 | 7.1153 | 0.001000 | 0.000 | N/A | 7.008 | 26.6525 | 693.48 | 2984.5 | 7.34 | 1543.54 |
| 10 | wikitext2 | 7.0933 | 0.001000 | 0.000 | N/A | 7.006 | 23.9953 | 705.34 | 2984.5 | 7.18 | 1307.60 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.29, wikitext2 Perplexity: 1287.14
- **Final Validation Metrics**: wikitext2: {"loss": 7.175949844820746, "perplexity": 1307.6015298545697}
