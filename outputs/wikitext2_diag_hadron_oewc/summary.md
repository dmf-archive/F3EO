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
    "name": "DiagHadronOEWC",
    "lr": 0.0001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "lifelong_decay": 0.999,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.1,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95,
    "adam_lr": 0.0001,
    "adam_weight_decay": 0.1,
    "adam_betas": [
      0.9,
      0.95
    ]
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
| 1 | wikitext2 | 20.8512 | 0.000100 | 0.000 | N/A | 5.054 | 52.0453 | 379.59 | 2788.0 | 8.39 | 4396.22 |
| 2 | wikitext2 | 7.3774 | 0.000100 | 0.000 | N/A | 5.338 | 20.1590 | 379.38 | 2788.0 | 6.90 | 989.58 |
| 3 | wikitext2 | 6.6174 | 0.000100 | 0.000 | N/A | 5.199 | 14.2717 | 379.35 | 2788.0 | 6.78 | 877.42 |
| 4 | wikitext2 | 6.2606 | 0.000100 | 0.000 | N/A | 5.003 | 12.2184 | 386.34 | 2788.0 | 6.56 | 704.03 |
| 5 | wikitext2 | 5.9461 | 0.000100 | 0.000 | N/A | 4.778 | 12.9776 | 381.15 | 2788.0 | 6.54 | 695.69 |
| 6 | wikitext2 | 5.6361 | 0.000100 | 0.000 | N/A | 4.520 | 14.0037 | 381.30 | 2788.0 | 6.54 | 693.27 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.39, wikitext2 Perplexity: 693.27
- **Final Validation Metrics**: wikitext2: {"loss": 6.541414024985884, "perplexity": 693.2661809835255}
