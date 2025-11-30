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
    "type": "rope",
    "vocabulary_size": 40479,
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 6,
    "num_layers": 4,
    "rope_theta": 10000.0,
    "intermediate_size": 2048,
    "tie_word_embeddings": true
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
    "weight_decay": 0.5,
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
| 1 | wikitext2 | 6.2468 | 0.000100 | 0.156 | N/A | 0.000 | 1.8607 | 631.92 | 2943.3 | 5.68 | 294.08 |
| 2 | wikitext2 | 5.0747 | 0.000100 | 0.105 | N/A | 0.000 | 2.2539 | 636.74 | 2943.3 | 5.32 | 203.66 |
| 3 | wikitext2 | 4.4394 | 0.000100 | 0.054 | N/A | 0.000 | 2.9224 | 646.44 | 2943.3 | 5.23 | 186.10 |
| 4 | wikitext2 | 3.8849 | 0.000100 | 0.023 | N/A | 0.000 | 3.7831 | 638.32 | 2943.3 | 5.28 | 196.81 |
| 5 | wikitext2 | 3.3775 | 0.000100 | 0.009 | N/A | 0.000 | 4.7111 | 656.98 | 2943.3 | 5.46 | 235.20 |
| 6 | wikitext2 | 2.9143 | 0.000100 | 0.004 | N/A | 0.000 | 5.6108 | 644.85 | 2943.3 | 5.70 | 298.56 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.70, wikitext2 Perplexity: 186.10
- **Final Validation Metrics**: wikitext2: {"loss": 5.698986382320009, "perplexity": 298.5646171648985}
