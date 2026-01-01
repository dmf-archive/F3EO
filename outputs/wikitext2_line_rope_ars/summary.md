# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2_line"
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
    "name": "ARS",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "rho": 0.05
  },
  "train": {
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2_line | 5.5382 | 0.000100 | 0.443 | N/A | 0.000 | 0.8138 | 796.32 | 2949.4 | 5.05 | 156.62 |
| 2 | wikitext2_line | 4.6032 | 0.000100 | 0.497 | N/A | 0.000 | 0.6982 | 785.66 | 3261.5 | 4.70 | 109.59 |
| 3 | wikitext2_line | 4.1862 | 0.000100 | 0.481 | N/A | 0.000 | 0.7327 | 809.80 | 3254.6 | 4.53 | 92.37 |
| 4 | wikitext2_line | 3.8480 | 0.000100 | 0.458 | N/A | 0.000 | 0.7801 | 799.43 | 3264.6 | 4.45 | 85.25 |
| 5 | wikitext2_line | 3.5375 | 0.000100 | 0.430 | N/A | 0.000 | 0.8440 | 789.78 | 3263.3 | 4.43 | 83.70 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.05, wikitext2_line Perplexity: 83.70
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.427282689222649, "perplexity": 83.70365875153283}
